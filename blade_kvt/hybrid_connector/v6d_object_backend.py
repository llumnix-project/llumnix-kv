import asyncio
import contextlib
import struct
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Optional

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector import (
    V6dObjectConnectorMetadata,
    V6dObjectConnectorScheduler,
    V6dObjectConnectorWorker,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hybrid_connector import (
    _SAVE_DONE_REQ,
    _SAVE_DONE_RESP,
    HB_SAVE_IORET,
    BackendMeta,
    HCSchedOutput,
    HybridBackend,
    IoDoneReqs,
    IoRet,
    get_param,
    hybridworker,
    kill_me_if_exception,
)
from vllm.v1.hybrid_connector.engine_proxy import (
    MsgpackDecoder,
    _sched,
    get_hybrid_worker_loop,
    sched_rpc_server,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)


# Worker → scheduler one-shot signal: background cudaHostRegister done
# for this tprank. Payload: IoDoneReqs with worker_tprank set.
_V6D_READY_REQ = 0x20181228
_V6D_READY_RESP = 0x91218103


@dataclass
class V6dObjectBackendMeta(BackendMeta):
    inner: V6dObjectConnectorMetadata = None  # type: ignore[assignment]
    # req_id -> num_external_tokens, carried to worker for IoRet.n
    external_tokens: dict[str, int] = None  # type: ignore[assignment]
    # Aborted stores. Worker drains any in-flight swap DMA for these
    # reqs before signalling save_done(n=0) — otherwise the scheduler
    # would discard the v6d blob while the GPU is still writing into
    # its memory (use-after-free against v6d's dlmalloc).
    aborted_save_ids: list[str] = field(default_factory=list)

    def __bool__(self):
        if self.inner is None:
            return False
        if self.aborted_save_ids:
            return True
        return not self.inner.is_empty()


class V6dObjectBackend(HybridBackend):

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._scheduler: Optional[V6dObjectConnectorScheduler] = None
        self._worker: Optional[V6dObjectConnectorWorker] = None
        self._num_external_tokens: dict[str, int] = {}
        self._v6d_ready: bool = not envs.VLLM_V6D_ASYNC_REGISTER
        self._v6d_ready_ranks: set[int] = set()
        self._v6d_ready_managers: set[int] = set()
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size

        if role == KVConnectorRole.SCHEDULER:
            self._num_groups = V6dObjectConnectorScheduler.compute_num_groups(
                kv_cache_config)
            self._scheduler = V6dObjectConnectorScheduler(
                vllm_config,
                kv_cache_config,
                protect_mamba_blocks_in_block_pool=True,
                is_hybrid_backend=True,
                v6d_backend=self if envs.VLLM_V6D_ASYNC_REGISTER else None,
            )
            assert self._num_groups == self._scheduler.num_groups
        elif role == KVConnectorRole.WORKER:
            self._worker = V6dObjectConnectorWorker(
                vllm_config, kv_cache_config
            )

        if role == KVConnectorRole.SCHEDULER and envs.VLLM_V6D_ASYNC_REGISTER:
            self._ready_dec = MsgpackDecoder(IoDoneReqs)
            sched_rpc_server().register_method(
                _V6D_READY_REQ, self._on_v6d_ready)

        # Worker-side state, set by HybridWorker after construction.
        self._bound_meta: Optional[V6dObjectBackendMeta] = None
        self._save_event_pool: list[torch.cuda.Event] = []
        self._loop = get_hybrid_worker_loop() \
            if role == KVConnectorRole.WORKER else None

        # Accumulates failed-store req_ids across chunks; drained by the
        # last_save RPC.
        self._failed_store_reqs: set[str] = set()

        # Per-req in-flight save tasks; abort drain awaits these so the
        # GPU swap finishes before the scheduler discards the v6d blob.
        self._inflight_save_tasks: dict[str, list[asyncio.Task]] = {}

        # Substep aborts arrive on the bypass loop thread where bind/clear
        # short-circuit; buffer them here (mutated only on the asyncio loop
        # thread via call_soon_threadsafe) and drain at the tail of the
        # next _async_do_store.
        self._bypass_pending_aborts: list[str] = []

    def _ensure_block_pool(self):
        """Lazily set block_pool on the inner scheduler connector."""
        if (self._scheduler is not None
                and self._scheduler._block_pool is None):
            self._scheduler.set_block_pool(
                _sched().kv_cache_manager.block_pool)

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)


    async def async_load_kv(
        self, m: BackendMeta
    ) -> AsyncGenerator[IoRet, None]:
        assert isinstance(m, V6dObjectBackendMeta)
        assert self._worker is not None
        meta = m.inner
        if not meta.reqs_to_load:
            return

        logger.debug(
            f"Worker starting async_load_kv for reqs: "
            f"{list(meta.reqs_to_load.keys())}")
        req_tasks = self._worker.async_start_load_kv(meta)
        if not req_tasks:
            return
        task_to_req = {t: rid for rid, t in req_tasks.items()}
        pending: set[asyncio.Task] = set(req_tasks.values())
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                req_id = task_to_req[task]
                exc = task.exception()
                if exc is not None:
                    logger.warning(
                        f"async_load_kv failed for req={req_id}: {exc}")
                    yield IoRet(reqid=req_id, n=0)
                else:
                    n = (m.external_tokens or {}).get(req_id)
                    yield IoRet(reqid=req_id, n=n)

    def async_save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        m: BackendMeta,
    ) -> Optional[AsyncGenerator[str, None]]:
        # Store is deferred to clear_backend_metadata (after all forward
        # work including draft model is done).
        return None

    def bind_backend_metadata(self, meta: BackendMeta):
        # Main step: _bound_meta is exclusively owned by the main step path;
        # the bypass substep goes through bypass_bind and must not touch it.
        self._bound_meta = meta

    def bypass_bind(self, meta: BackendMeta):
        # Bypass substep: must not touch _bound_meta (owned by the main step),
        # otherwise it would stomp on the main step's in-flight metadata and
        # silently drop its reqs_to_store, leaking entries in
        # HybridScheduler._saving. Only forward substep aborts to the loop
        # thread.
        if (isinstance(meta, V6dObjectBackendMeta)
                and meta.aborted_save_ids):
            # Copy: the loop thread reads after we return.
            ids = list(meta.aborted_save_ids)
            self._loop.call_soon_threadsafe(
                self._bypass_pending_aborts.extend, ids)

    def bypass_clear(self):
        # Bypass substep has no deferred store to trigger.
        return

    def clear_backend_metadata(self):
        """Trigger deferred store if there is pending store work.

        Called on the worker thread from HybridWorker.clear_connector_metadata
        after all forward work (target + draft) and mamba postprocess are done.
        """
        meta = self._bound_meta
        self._bound_meta = None
        if meta is None:
            return
        if not isinstance(meta, V6dObjectBackendMeta):
            logger.warning(
                "clear_backend_metadata: unexpected meta type: %s", type(meta))
            return
        inner = meta.inner
        aborted_save_ids = meta.aborted_save_ids
        if not inner.reqs_to_store and not aborted_save_ids:
            return

        if inner.reqs_to_store:
            event = (self._save_event_pool.pop()
                     if self._save_event_pool else torch.cuda.Event())
            event.record(torch.cuda.current_stream())
        else:
            event = None
        asyncio.run_coroutine_threadsafe(
            self._async_do_store(event, inner, aborted_save_ids), self._loop)

    async def _send_save_done(self, req_id: str, tprank: int):
        n: Optional[int] = None
        if req_id in self._failed_store_reqs:
            self._failed_store_reqs.discard(req_id)
            n = 0
            logger.warning(
                f"_send_save_done: signalling store failure "
                f"for req={req_id}")
        savereq = IoDoneReqs(
            worker_tprank=tprank,
            reqids=[IoRet(reqid=req_id, n=n)])
        await hybridworker().io_done_rpc(savereq, _SAVE_DONE_REQ, _SAVE_DONE_RESP)

    def _register_inflight(self, req_id: str, task: asyncio.Task) -> None:
        self._inflight_save_tasks.setdefault(req_id, []).append(task)

        def _on_done(t: asyncio.Task, _req_id: str = req_id) -> None:
            tasks = self._inflight_save_tasks.get(_req_id)
            if not tasks:
                return
            with contextlib.suppress(ValueError):
                tasks.remove(t)
            if not tasks:
                self._inflight_save_tasks.pop(_req_id, None)

        task.add_done_callback(_on_done)

    async def _drain_aborted_save(
        self, aborted_save_ids: list[str], tprank: int,
    ) -> None:
        for req_id in aborted_save_ids:
            tasks = self._inflight_save_tasks.get(req_id)
            if tasks:
                logger.info(
                    "_drain_aborted_save: awaiting %d in-flight save "
                    "task(s) for aborted req=%s", len(tasks), req_id)
                # Must NOT cancel: the swap kernel is queued on the
                # store stream writing into the v6d blob; cancelling
                # the asyncio.Task wouldn't stop the GPU work and
                # would let the scheduler discard the blob before the
                # DMA finishes.
                results = await asyncio.gather(
                    *list(tasks), return_exceptions=True)
                for r in results:
                    if isinstance(r, BaseException):
                        logger.warning(
                            "_drain_aborted_save: store failed for "
                            "aborted req=%s: %s", req_id, r)
            self._failed_store_reqs.add(req_id)
            await self._send_save_done(req_id, tprank)

    @kill_me_if_exception
    async def _async_do_store(
        self, event: Optional[torch.cuda.Event],
        inner: V6dObjectConnectorMetadata,
        aborted_save_ids: list[str],
    ):
        """Run deferred store on disaggw thread, send RPC for last-save reqs."""
        assert self._worker is not None
        try:
            noop_last_save_reqs: set[str] = set()
            real_reqs_to_store: dict[str, tuple] = {}
            for req_id, (groups_data, is_last_save) in (
                    inner.reqs_to_store.items()):
                if not groups_data and is_last_save:
                    noop_last_save_reqs.add(req_id)
                else:
                    real_reqs_to_store[req_id] = (groups_data, is_last_save)

            tprank = get_tensor_model_parallel_rank()

            for req_id in noop_last_save_reqs:
                logger.debug(
                    f"_async_do_store: noop last_save req={req_id}")
                await self._send_save_done(req_id, tprank)

            last_save_reqs: set[str] = set()
            for req_id, (_groups_data, is_last_save) in (
                    real_reqs_to_store.items()):
                if is_last_save:
                    last_save_reqs.add(req_id)

            if real_reqs_to_store:
                store_handler = self._worker._store_handler
                if store_handler is not None:
                    store_handler._stream.wait_event(event)

                logger.debug(
                    f"_async_do_store: starting store for "
                    f"reqs={list(real_reqs_to_store.keys())}")
                req_tasks = self._worker.async_start_store_kv(inner)
                if req_tasks:
                    for _rid, _t in req_tasks.items():
                        self._register_inflight(_rid, _t)
                    task_to_req = {t: rid for rid, t in req_tasks.items()}
                    pending_tasks: set[asyncio.Task] = set(
                        req_tasks.values())
                    while pending_tasks:
                        done, pending_tasks = await asyncio.wait(
                            pending_tasks,
                            return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            req_id = task_to_req[task]
                            exc = task.exception()
                            if exc is not None:
                                logger.warning(
                                    f"_async_do_store: store failed "
                                    f"for req={req_id}: {exc}")
                                self._failed_store_reqs.add(req_id)
                            if req_id in last_save_reqs:
                                logger.debug(
                                    f"_async_do_store: last_save "
                                    f"req={req_id}")
                                await self._send_save_done(req_id, tprank)
                                last_save_reqs.discard(req_id)

            for req_id in last_save_reqs:
                logger.warning(
                    f"_async_do_store: unexpected remaining "
                    f"last_save req={req_id}, sending RPC anyway")
                await self._send_save_done(req_id, tprank)

            if self._bypass_pending_aborts:
                aborted_save_ids.extend(self._bypass_pending_aborts)
                self._bypass_pending_aborts = []

            if aborted_save_ids:
                await self._drain_aborted_save(aborted_save_ids, tprank)
        finally:
            if event is not None:
                self._save_event_pool.append(event)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_operations(self, req: Request) -> tuple[int, int]:
        if not self._v6d_ready:
            return 0, 0
        # Skip v6d entirely for short prompts (aligns with VLLM_KVS_ON_MIN_LENGTH
        # semantics used by kvsbackend / mooncake_kvsbackend).
        if req.num_prompt_tokens <= envs.VLLM_KVS_ON_MIN_LENGTH + 1:
            return 0, 0
        # All requests go through async lookup
        load_count = 1
        # Save if prefill not done
        save_count = 1 if req.num_computed_tokens < req.num_prompt_tokens else 0
        return load_count, save_count

    def _check_v6d_ready(self) -> None:
        if self._v6d_ready:
            return
        if (len(self._v6d_ready_ranks) >= self._tp_size
                and len(self._v6d_ready_managers) >= self._num_groups):
            self._v6d_ready = True
            logger.info("v6d backend ready")

    def mark_manager_connected(self, group_id: int) -> None:
        if self._v6d_ready:
            return
        self._v6d_ready_managers.add(group_id)
        logger.info("v6d manager connected: group=%d (%d/%d)",
                    group_id, len(self._v6d_ready_managers),
                    self._num_groups)
        self._check_v6d_ready()

    def mark_v6d_ready(self, tprank: int, tp_size: int) -> None:
        if self._v6d_ready:
            return
        self._v6d_ready_ranks.add(tprank)
        logger.info("v6d ready: tprank=%d (%d/%d)",
                    tprank, len(self._v6d_ready_ranks), tp_size)
        self._check_v6d_ready()

    # sched thread (rpc handler)
    async def _on_v6d_ready(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        reqs: IoDoneReqs = self._ready_dec.decode(reqbuf)

        writer.write(struct.pack("=I", _V6D_READY_RESP))
        await writer.drain()

        tp_size = self._tp_size
        self.mark_v6d_ready(reqs.worker_tprank, tp_size)

    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        assert self._scheduler is not None
        hit_length, _ = await self._scheduler.async_get_num_new_matched_tokens(
            req, num_computed_tokens
        )
        return hit_length

    async def async_update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> Optional[IoRet]:
        assert self._scheduler is not None
        req_id = request.request_id
        try:
            await self._scheduler.async_update_state_after_alloc(
                request, blocks, num_external_tokens
            )
        except Exception as e:
            logger.warning(
                f"async_update_state_after_alloc failed for req {req_id}, "
                f"treating as no cache hit and recomputing on GPU: "
                f"{type(e).__name__}: {e}")
            self._scheduler._v6d_mamba_hit_reqs.discard(req_id)
            return IoRet(reqid=req_id, n=0)
        if num_external_tokens > 0:
            self._num_external_tokens[req_id] = num_external_tokens
        # Return None so HybridScheduler puts the request into _prepared
        # and waits for the worker-side _LOAD_DONE_REQ RPC callback.
        return None

    async def async_cleanup(self, req: Request):
        if self._scheduler is None:
            return
        req_id = req.request_id
        logger.debug(f"Cleaning up backend state for req {req_id}")


        # v6d worker signals store failure via n=0 in its save_done IoRet.
        save_ioret: Optional[IoRet] = get_param(req, HB_SAVE_IORET, None)
        save_failed = (save_ioret is not None
                       and save_ioret.n is not None
                       and save_ioret.n == 0)

        # Clean up external_tokens tracking
        self._num_external_tokens.pop(req_id, None)

        # Finalize v6d objects for stores (originally done in
        # update_connector_output). Must happen before _try_teardown_save
        # frees the blocks on the core thread.
        storing = self._scheduler._storing_block_hashes.pop(req_id, None)
        if storing is not None:
            for group_id, hashes in storing.items():
                manager = self._scheduler.managers.get(group_id)
                if manager is None:
                    continue
                if save_failed:
                    logger.warning(
                        f"Store failed for req {req_id}: discarding "
                        f"{len(hashes)} pending v6d objs in group "
                        f"{group_id}: ex={save_ioret.ex}")
                    manager.discard_pending(hashes)
                else:
                    manager.seal(hashes, request_id=req_id)
                    logger.debug(
                        f"Sealed blocks for req {req_id}, "
                        f"group {group_id}: {hashes}")

        self._scheduler._cleanup_req_tracking(req_id)
        self._scheduler._cleanup_loading_state(req_id)
        for manager in self._scheduler.managers.values():
            manager.cleanup_request(req_id)

        self._scheduler._pending_store_reqs.discard(req_id)

        # Release ref_cnt protection on mamba store blocks to prevent leaks
        self._ensure_block_pool()
        self._scheduler._release_protected_blocks(req_id)

    def build_backend_meta(
        self, sout: SchedulerOutput
    ) -> BackendMeta:
        assert self._scheduler is not None
        assert isinstance(sout, HCSchedOutput)
        self._ensure_block_pool()

        # Aborted saves: defer v6d-side cleanup to async_cleanup so it
        # runs AFTER the worker drains any in-flight swap DMA. Calling
        # obj.discard() (→ del_blob) here would race with the GPU
        # writing into the blob and trip v6d's dlmalloc.
        aborted_save_ids: list[str] = [
            areq.request_id for areq in sout.hc_aborted_save
        ]

        inner = self._scheduler.build_connector_meta(sout)
        # Drain external_tokens for requests in this meta's load set
        external_tokens: dict[str, int] = {}
        if inner.reqs_to_load:
            for req_id in inner.reqs_to_load:
                n = self._num_external_tokens.pop(req_id, None)
                if n is not None:
                    external_tokens[req_id] = n

        return V6dObjectBackendMeta(
            inner=inner,
            external_tokens=external_tokens,
            aborted_save_ids=aborted_save_ids,
        )
