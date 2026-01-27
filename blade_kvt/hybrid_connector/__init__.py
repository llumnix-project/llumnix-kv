import asyncio
import contextlib
import pickle
import struct
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Optional

import msgspec
import torch
import zmq
import zmq.asyncio
from zmq import (  # type: ignore
    SUB,
    SUBSCRIBE,
    XPUB,
    XPUB_VERBOSE,
)

import vllm.envs as envs
from vllm.utils.network_utils import get_ip, get_open_port

# yapf: disable
from .engine_proxy import (
    EngineCoreOutput,
    FinishReason,
    KVCacheBlocks,
    KVCacheConfig,
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    MsgpackDecoder,
    MsgpackEncoder,
    Request,
    SchedulerOutput,
    SupportsHMA,
    VllmConfig,
    get_hybrid_sched_loop,
    get_hybrid_worker_loop,
    get_logger,
    get_p_node_pop_len,
    get_param,
    get_tensor_model_parallel_rank,
    group_layers_by_index,
    merge_hybrid_blocks,
    sched_acquire_blocks,
    sched_add_req,
    sched_allocate_slots,
    sched_finish_req,
    sched_free_blocks,
    sched_get_blocks,
    sched_get_req,
    sched_rpc_server,
    sched_rpc_server_port,
    set_param,
    wakeup_core,
)

# yapf: enable
from .utils import (
    ConnPool,
    IoRet,
    IoState,
    handle_done_req,
    kill_me_if_exception,
    try_advance,
)

logger = get_logger(__name__)


class BackendMeta:

    def __bool__(self):
        return False


HB_IORET = "_HybridBackend_IORET"

# value: int
PREALLOC_KEY = "_hbprealloc"

# value: int
CLEANUP_RC_KEY = "_HybridBackend_cleanup_rc"

# value: bool
ADD_REQ_LOGGED_KEY = "_hb_add_req_logged"


def _inc_cleanup_rc(req: Request) -> int:
    rc: int = get_param(req, CLEANUP_RC_KEY, 0)
    rc += 1
    set_param(req, CLEANUP_RC_KEY, rc)
    return rc


def _dec_cleanup_rc(req: Request) -> int:
    rc: int = get_param(req, CLEANUP_RC_KEY, 0)
    assert rc > 0
    rc -= 1
    set_param(req, CLEANUP_RC_KEY, rc)
    return rc


# split this to BackendScheduler? BackendWorker?
class HybridBackend:
    # worker thread
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config
        self.is_hybrid = vllm_config.model_config.is_hybrid
        self.hybrid_model_send_layer = []

    def get_request(self, request_id: str) -> Optional[Request]:
        return sched_get_req(request_id)

    # ==============================
    # Worker-side methods
    # ==============================

    # worker thread
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        return

    # worker thread/disagg thread
    # bind_backend_metadata([R]) happen before async_load_kv(R)
    def bind_backend_metadata(self, meta: BackendMeta):
        return

    # worker thread/disagg thread
    def clear_backend_metadata(self):
        return

    # disaggw thread
    async def async_load_kv(self, m: BackendMeta) -> AsyncGenerator[IoRet, None]:
        raise NotImplementedError()
        yield "x"  # make vscode happy

    # worker thread
    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    # disagg thread
    # return: num_external_tokens.
    # If num_external_tokens = 0, it means no load operation will occur.
    # Otherwise, it indicates that the backend is responsible for loading
    # the kvcache corresponding to the token range
    # [num_computed_tokens, num_computed_tokens + num_external_tokens).
    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        return 0

    # disagg thread
    async def async_update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> Optional[IoRet]:
        return None

    # disagg thread
    async def async_cleanup(self, req: Request):
        return

    # core thread
    # return: (load_count, save_count)
    # load_count > 0 means need async load.
    # load_count = 0 means async_get_num_new_matched_tokens(R) must return 0.
    # save_count > 0 means need async save.
    # count is useful when load from/save to multiple srcs/dsts.
    def get_operations(self, req: Request) -> tuple[int, int]:
        return 0, 0

    # async def on_add_request(self, req: Request, blocks: "KVCacheBlocks"):
    #     local = req.num_computed_tokens
    #     remote = await self.async_get_num_new_matched_tokens(req, local)
    #     if remote == 0:
    #         return
    #     await self.update_state_after_alloc(req, blocks, remote)
    #     return
    #
    # Regarding the synchronization between `build_backend_meta`
    # and `on_add_request`. it is recommended to implement the following
    # pattern:
    #
    # self._q: collections.deque
    # As mentioned in https://ata.atatech.org/articles/11020344053,
    # deque can be considered thread-safe.
    # def update_state_after_alloc(self, req):
    #   self._q.append(req)
    #
    # def build_backend_meta(self, sout):
    #   reqs = popall(self._q)
    #   # add sout, reqs to meta

    # core thread
    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        return BackendMeta()


@dataclass
class HybridMetadata(KVConnectorMetadata):
    reqs: BackendMeta
    # some metadata used by HybridConnector


def rpc_port(cfg: VllmConfig):
    return sched_rpc_server_port(cfg)


class _LoadingReq(IoState):
    def __init__(self, req: Request):
        super().__init__()
        self._req = req
        return


class _SavingReq(IoState):
    def __init__(self, req: Request, kvblks: KVCacheBlocks, save_count: int = 1):
        super().__init__(signals_per_worker=save_count)
        self._req = req
        self.kvblks = kvblks
        return


class IoDoneReqs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    worker_tprank: int
    reqids: list[IoRet]


# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded IoDoneReqs
_SAVE_DONE_REQ = 0x20181221
_LOAD_DONE_REQ = 0x20181222

# Another format of _SAVE_DONE_REQ
# format: see handle_done_req
_SAVE_DONE2_REQ = 0x20181223

_SAVE_DONE_RESP = 0x91218102
_LOAD_DONE_RESP = _SAVE_DONE_RESP

# value: bool
_SAVE_PREPARED = "hbsaveprepared"

# value: bool
_ABORTED = "hbreqisaborted"

_GET_BYPASS_HANDLE = 0x20181226
_GET_BYPASS_HANDLE_RESP = 0x20181227

def req_aborted(req: Request) -> bool:
    return get_param(req, _ABORTED, False)


# core thread
def has_setup_save(req: Request) -> bool:
    return bool(get_param(req, _SAVE_PREPARED))


# core thread
def try_setup_save(req: Request):
    global _g_scheduler
    if has_setup_save(req):
        return
    assert _g_scheduler is not None

    kvblks = sched_get_blocks(req.request_id)
    _g_scheduler._setup_save(req, kvblks)
    assert has_setup_save(req)
    return


# core thread
def try_teardown_save(req: Request):
    global _g_scheduler
    assert _g_scheduler is not None
    _g_scheduler._try_teardown_save(req.request_id)
    return


def _get_backend_cls(cfg: VllmConfig) -> type[HybridBackend]:
    """Get the backend class based on the kv_transfer_config."""
    assert cfg.kv_transfer_config is not None, (
        "kv_transfer_config must be set in VllmConfig for HybridConnector"
    )

    backend = cfg.kv_transfer_config.get_from_extra_config("backend", None)

    assert backend is not None, (
        "backend must be set in kv_transfer_config for HybridConnector, "
        "choice from 'local_file', 'kvt', 'vineyard'"
    )

    if backend == "migration":
        from .migration.backend import MigrationBackend

        return MigrationBackend

    if backend == "kvt+migration" or backend == "migration+kvt":
        from .migration.kvtmigration import KVTMigration

        return KVTMigration

    if backend == "kvt+kvs" or backend == "kvs+kvt":
        from .kvsp import KVSP

        return KVSP

    if backend == "local_file":
        from .filekvtbackend import FileBackend

        return FileBackend
    elif backend == "kvt":
        if (
            cfg.kv_transfer_config.is_kv_producer
            and cfg.kv_transfer_config.is_kv_consumer
        ):
            from .filekvtbackend import FilePBackend

            return FilePBackend
        elif cfg.kv_transfer_config.is_kv_consumer:
            from .kvtbackend import DBackend

            return DBackend
        elif cfg.kv_transfer_config.is_kv_producer:
            from .kvtbackend import PBackend

            return PBackend
        else:
            raise ValueError(
                "kv_transfer_config must specify either is_kv_producer or "
                "is_kv_consumer for 'kvt' backend"
            )
    elif backend == "vineyard":
        assert cfg.kv_transfer_config.kv_role == "kv_both", (
            "For vineyard backend, kv_role must be 'kv_both' in kv_transfer_config"
        )
        from .vineyard_backend import VineyardBackend

        return VineyardBackend
    elif backend == "kvs":
        assert cfg.kv_transfer_config.kv_role == "kv_both", \
            "For kvs backend, kv_role must be 'kv_both' in kv_transfer_config"
        from .kvsbackend import VineyardKVSBackend
        return VineyardKVSBackend
    elif backend == "mooncake":
        assert cfg.kv_transfer_config.kv_role == "kv_both", \
            "For mooncake backend, kv_role must be 'kv_both' in kv_transfer_config"
        from .mooncake_kvsbackend import MooncakeKVSBackend
        return MooncakeKVSBackend

    raise ValueError(
        f"Unknown backend: {backend}. "
        "Supported backends are 'local_file', 'kvt', 'vineyard', and 'kvs'.")


@dataclass
class AbortReq:
    reqid: str
    output: bool
    reason: str


def _put_abort_resp(
    load_output: defaultdict[int, list[EngineCoreOutput]], req: Request
):
    load_output[req.client_index].append(
        EngineCoreOutput(request_id=req.request_id,
                         new_token_ids=[req.eos_token_id or 0],
                         finish_reason=FinishReason.ABORT,
                         queue_server_address=req.queue_server_address))
    return


class HybridScheduler:
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        self._cfg = vllm_config
        self._kv_cache_config = kv_cache_config
        self._packenc = msgspec.msgpack.Encoder()
        self._strdec = MsgpackDecoder(str)
        self.loop: asyncio.AbstractEventLoop = get_hybrid_sched_loop()
        self._start_rpc_server()

        ### R/W: disagg thread, core thread
        # req, load?, save?
        self._waiting: deque[tuple[Request, bool, bool]] = deque()
        self._loaded: deque[Request] = deque()
        self._saved: deque[str] = deque()
        self._prepared: deque[str] = deque()
        self._aborting: deque[AbortReq] = deque()
        ### R/W: core thread
        self._abortmeta_load: list[Request] = []
        self._abortmeta_save: list[Request] = []
        self._stop0: list[Request] = []

        ### W: core thread. R: disagg thread
        self._saving: dict[str, _SavingReq] = dict()
        self._loading: dict[str, _LoadingReq] = dict()

        self._reqs_ts: dict[str, dict[str, float]] = dict()
        ### R/W: disagg thread
        self._iodonedec = MsgpackDecoder(IoDoneReqs)

        BackendCls = _get_backend_cls(vllm_config)
        self._backend = BackendCls(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

        asyncio.run_coroutine_threadsafe(self.log_status(), self.loop)

        return

    def _add_ts(self, req: Request, tag: str):
        corre_id = get_param(req, "correlation_id")
        if corre_id is None:
            return
        if tag == "on_add_req":
            self._reqs_ts[corre_id] = dict()
        self._reqs_ts[corre_id][tag] = time.monotonic() * 1000
        if tag == "done":
            keys = list(self._reqs_ts[corre_id].keys())
            intervals = {}
            for i in range(1, len(keys)):
                prev_k = keys[i - 1]
                curr_k = keys[i]
                dt = (self._reqs_ts[corre_id][curr_k]
                      - self._reqs_ts[corre_id][prev_k])
                intervals[f"{prev_k} -> {curr_k}"] = dt
            logger.info(
                f"HybridScheduler time stats: correlation_id={corre_id},"
                f" num_prompt_tokens={req.num_prompt_tokens},"
                f" num_computed_tokens={req.num_computed_tokens},"
                f" num_external_computed_tokens={req.num_external_computed_tokens},"
                f" {intervals=}")
            self._reqs_ts.pop(corre_id)

    def _start_rpc_server(self):
        if envs.VLLM_ENABLE_BYPASS_TASK:
            self._start_bypass_rpc_server()

        self._rpc_server = sched_rpc_server()
        self._rpc_server.register_method(_SAVE_DONE_REQ, self._on_save_done)
        self._rpc_server.register_method(_SAVE_DONE2_REQ, self._on_save_done2)
        self._rpc_server.register_method(_LOAD_DONE_REQ, self._on_load_done)
        self._rpc_server.register_method(_GET_BYPASS_HANDLE, self._get_bypass_handle)

        return

    def _start_bypass_rpc_server(self):
        # create a publish-subscribe socket to communicate
        connect_ip = get_ip()
        remote_subscribe_port = get_open_port()
        self.bypass_remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"

        context = zmq.asyncio.Context()
        self.bypass_remote_socket = context.socket(XPUB)
        self.bypass_remote_socket.setsockopt(XPUB_VERBOSE, True)
        self.bypass_remote_socket.bind(self.bypass_remote_subscribe_addr)

        logger.info("bypass server socket started at: %s",
                    self.bypass_remote_subscribe_addr)
        return

    @kill_me_if_exception
    async def _send_bypass_task(self, kv_connector_metadata):
        data = pickle.dumps(kv_connector_metadata)
        await self.bypass_remote_socket.send(data)

    def send_bypass_task(self, kv_connector_metadata: HybridMetadata):
        if kv_connector_metadata.reqs:
            asyncio.run_coroutine_threadsafe(
                self._send_bypass_task(kv_connector_metadata),
                self.loop,
            )

    # disagg thread, core thread
    def _tp_size(self):
        return self._cfg.parallel_config.tensor_parallel_size

    # core thread
    def step(self) -> Optional[dict[int, list[EngineCoreOutput]]]:
        self._step_saved()
        self._step_waiting()
        self._step_loaded()
        return self._step_aborting()

    def _step_saved(self):
        while self._saved:
            reqid = self._saved.popleft()
            self._try_teardown_save(reqid)
        return

    # core thread
    def _setup_save(self, req: Request, kvblks: KVCacheBlocks, save_count: int = 1):
        assert not has_setup_save(req)
        _inc_cleanup_rc(req)

        assert req.request_id not in self._saving
        self._saving[req.request_id] = _SavingReq(req, kvblks, save_count)

        sched_acquire_blocks(kvblks)
        set_param(req, _SAVE_PREPARED, True)
        return

    # core thread
    def _try_teardown_save(self, reqid: str):
        state = self._saving.pop(reqid, None)
        if state is None:
            logger.info("teardown save twice: reqid=%s", reqid)
            # assert not has_setup_save(state._req)
            return
        assert has_setup_save(state._req)
        sched_free_blocks(state.kvblks)
        set_param(state._req, _SAVE_PREPARED, False)
        return

    # core thread
    def _step_waiting(self):
        while self._waiting:
            # In pd disagg, load means decode, save means prefill
            req, load_count, save_count = self._waiting[0]
            gamma = get_p_node_pop_len(self._cfg) - 1
            prealloc = get_param(req, PREALLOC_KEY, 0)
            # need notify prefill node to allocate slot for poped gamma+1 tokens
            kvblks = sched_allocate_slots(
                req, load_count > 0, save_count > 0, prealloc, gamma)

            # Only log on first attempt or successful allocation to avoid noise
            if kvblks is not None or not get_param(req, ADD_REQ_LOGGED_KEY, False):
                logger.info(
                    "add req. reqid=%s promptlen=%s computed=%s maxcomputed=%s load=%s save=%s nokvblks=%s",  # noqa: E501
                    req.request_id,
                    req.num_prompt_tokens,
                    req.num_computed_tokens,
                    None,
                    load_count,
                    save_count,
                    bool(kvblks is None),
                )
                set_param(req, ADD_REQ_LOGGED_KEY, True)

            if kvblks is None:
                break
            self._waiting.popleft()

            if save_count > 0:
                self._setup_save(req, kvblks, save_count)

            if load_count > 0:
                _inc_cleanup_rc(req)
                assert req.request_id not in self._loading
                self._loading[req.request_id] = _LoadingReq(req)
                self._add_ts(req, "load_enqueue")
                coro = self._on_add_req(req, kvblks)
                asyncio.run_coroutine_threadsafe(coro, self.loop)
            else:
                # fastpath for num_external_tokens = 0.
                # In this case, async load operation is not required, the
                # request is directly visible to the scheduler.
                self._loaded.append(req)
        return

    # core thread
    def _step_loaded(self):
        while self._loaded:
            req = self._loaded.popleft()
            ioret: Optional[IoRet] = get_param(req, HB_IORET, None)
            if ioret is None:
                ioret = IoRet(ex=None, n=0)

            self._loading.pop(req.request_id, None)
            req.num_computed_tokens += ioret.n or 0
            req.num_external_computed_tokens += ioret.n or 0

            max_computed = None
            if max_computed is not None:
                assert max_computed < req.num_prompt_tokens
                if max_computed <= req.num_computed_tokens:
                    req.set_max_computed_tokens(req.num_computed_tokens)
                    self._stop0.append(req)

            sched_add_req(req)
            self._add_ts(req, "done")
            if ioret.ex is not None or req_aborted(req):
                areq = AbortReq(
                    reqid=req.request_id,
                    output=not req_aborted(req),
                    reason="load fail",
                )
                self.on_abort_req(areq, iscore=True)
        return

    # core thread
    def on_add_req(self, req: "Request") -> bool:
        load_count, save_count = self._backend.get_operations(req)
        if load_count > 0 or save_count > 0:
            self._add_ts(req, "on_add_req")
            self._waiting.append((req, load_count, save_count))
            return True
        return False

    def _step_aborting(self) -> Optional[dict[int, list[EngineCoreOutput]]]:
        load_output: defaultdict[int, list[EngineCoreOutput]] = defaultdict(list)
        while self._aborting:
            areq = self._aborting.popleft()
            loadreq = self._loading.get(areq.reqid)
            if loadreq is not None:
                logger.info("abort loading. areq=%s", areq)
                assert sched_get_req(areq.reqid) is None
                set_param(loadreq._req, _ABORTED, True)
                self._abortmeta_load.append(loadreq._req)
                if areq.output:
                    _put_abort_resp(load_output, loadreq._req)
                continue

            req = sched_get_req(areq.reqid)
            if req is not None:
                logger.info(
                    "abort req. areq=%s eos=%s status=%s totaltokens=%s",
                    areq,
                    req.eos_token_id,
                    req.status,
                    len(req.all_token_ids),
                )
                sched_finish_req(areq.reqid)
                with contextlib.suppress(ValueError):
                    self._stop0.remove(req)
                if areq.output:
                    _put_abort_resp(load_output, req)

            savereq = self._saving.get(areq.reqid)
            if savereq is not None:
                logger.info("abort saving. areq=%s", areq)
                assert has_setup_save(savereq._req)
                self._abortmeta_save.append(savereq._req)
        return load_output

    # thread safe
    def on_abort_req(self, areq: AbortReq, iscore: bool):
        if iscore:
            self._aborting.append(areq)
        else:
            _q_append(self._aborting, areq)
        return

    # disagg thread
    @kill_me_if_exception
    async def _on_add_req(self, req: Request, kvblks: KVCacheBlocks):
        local = req.num_computed_tokens
        rmt = await self._backend.async_get_num_new_matched_tokens(req, local)

        ioret = await self._backend.async_update_state_after_alloc(req, kvblks, rmt)
        self._add_ts(req, "after_alloc")
        if ioret is not None and ioret.n is None:
            ioret.n = rmt

        if rmt <= 0 or ioret is not None:
            await self.mark_loaded(req, ioret)
        else:
            _q_append(self._prepared, req.request_id)
        return

    # disagg thread
    async def mark_loaded(self, req: Request, ioret: Optional[IoRet] = None):
        set_param(req, HB_IORET, ioret)
        _q_append(self._loaded, req)
        self._add_ts(req, "mark_loaded")
        await self._cleanup(req)
        return

    async def _do_save_done(self, worker_tprank: int, ioret: IoRet):
        tpsize = self._tp_size()
        state: Optional[_SavingReq] = try_advance(
            self._saving, ioret, worker_tprank, tpsize
        )
        if state is None:
            return

        logger.info("mark saved. reqid=%s", ioret.reqid)
        _q_append(self._saved, ioret.reqid)
        await self._cleanup(state._req)
        return

    async def _on_save_done2(self, reader, writer):
        await handle_done_req(reader, writer, self._do_save_done, _SAVE_DONE_RESP)
        return

    async def _on_save_done(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        reqs: IoDoneReqs = self._iodonedec.decode(reqbuf)

        tasks = []
        for reqid in reqs.reqids:
            tasks.append(self._do_save_done(reqs.worker_tprank, reqid))

        resp = struct.pack("=I", _SAVE_DONE_RESP)
        writer.write(resp)

        await asyncio.gather(*tasks)
        await writer.drain()
        return

    async def _do_load_done(self, worker_tprank, reqid: IoRet):
        state: Optional[_LoadingReq] = try_advance(self._loading, reqid,
                                                   worker_tprank,
                                                   self._tp_size())
        if state is None:
            return

        ioret = state.merge()
        logger.info("mark loaded. reqid=%s ioret=%s", state._req.request_id, ioret)
        self._add_ts(state._req, "load_done")
        await self.mark_loaded(state._req, ioret)

        return

    async def _on_load_done(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        reqs: IoDoneReqs = self._iodonedec.decode(reqbuf)

        tasks = []
        for reqid in reqs.reqids:
            tasks.append(self._do_load_done(reqs.worker_tprank, reqid))

        resp = struct.pack("=I", _LOAD_DONE_RESP)
        writer.write(resp)

        await asyncio.gather(*tasks)
        await writer.drain()
        return

    async def _get_bypass_handle(self, reader, writer):
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, _GET_BYPASS_HANDLE_RESP, 0)
        self._packenc.encode_into(self.bypass_remote_subscribe_addr, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        writer.write(msgbuf)
        await writer.drain()
        return

    async def _cleanup(self, req: Request):
        rc = _dec_cleanup_rc(req)
        if rc <= 0:
            await self._backend.async_cleanup(req)
        return

    # core thread
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return 0, False

    # core thread
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert num_external_tokens == 0
        return

    # core thread
    def build_connector_meta(self, sout: SchedulerOutput) -> HybridMetadata:
        self._prepared.clear()
        hcsout = HCSchedOutput(
            **sout.__dict__,
            hc_aborted_load=self._abortmeta_load,
            hc_aborted_save=self._abortmeta_save,
            hc_stop0=self._stop0,
        )
        self._abortmeta_load = []
        self._abortmeta_save = []
        self._stop0 = []
        reqs = self._backend.build_backend_meta(hcsout)
        return HybridMetadata(reqs=reqs)

    # core thread
    def has_requests(self) -> bool:
        if self._waiting:
            return True
        if self._loaded:
            return True
        if self._saved:
            return True
        if self._aborting:
            return True
        if self._abortmeta_load:
            return True
        if self._abortmeta_save:
            return True
        if self._stop0:
            return True
        return bool(self._prepared)

    @kill_me_if_exception
    async def log_status(self):
        while True:
            logger.info(
                "HybridScheduler status: "
                f"waiting={len(self._waiting)} "
                f"loaded={len(self._loaded)} "
                f"saved={len(self._saved)} "
                f"aborting={len(self._aborting)} "
                f"abortmeta_load={len(self._abortmeta_load)} "
                f"abortmeta_save={len(self._abortmeta_save)} "
                f"stop0={len(self._stop0)} "
                f"prepared={len(self._prepared)} "
                f"saving={len(self._saving)} "
                f"loading={len(self._loading)}"
            )
            await asyncio.sleep(10)


@dataclass
class HCSchedOutput(SchedulerOutput):
    hc_aborted_load: list[Request] = field(default_factory=list)
    hc_aborted_save: list[Request] = field(default_factory=list)
    hc_stop0: list[Request] = field(default_factory=list)


def _try_wakeup_core(q: deque[Any]):
    qlen = len(q)
    if qlen == 1:
        wakeup_core()
    return


def _q_append(q: deque[Any], item: Any):
    q.append(item)
    _try_wakeup_core(q)
    return


class _WSavingReq:
    def __init__(self):
        self._ready_layers: list[str] = []


class HybridWorker:
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        self._cfg = vllm_config
        self._kv_cache_config = kv_cache_config
        self._strdec = MsgpackDecoder(str)
        self.loop = get_hybrid_worker_loop()

        ### R/W: worker thread
        self._meta: Optional[HybridMetadata] = None

        ### R/W: disaggw thread
        self._saving: dict[str, _WSavingReq] = dict()
        core_ip = "127.0.0.1"  # get from kv_transfer config
        self._connpool = ConnPool(core_ip, rpc_port(self._cfg), 3)
        self._ioenc = MsgpackEncoder()

        BackendCls = _get_backend_cls(vllm_config)
        self._backend = BackendCls(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
        self._num_layers: Optional[int] = None

        if envs.VLLM_ENABLE_BYPASS_TASK:
            asyncio.run_coroutine_threadsafe(self.start_bypass_task_loop(), self.loop)

        return

    async def _get_bypass_handle(self):
        core_ip = "127.0.0.1"
        scheduler_port = rpc_port(self._cfg)
        # Scheduler Bypass server might not be ready yet
        await asyncio.sleep(2)
        last_log_time = 0.0
        req = struct.pack("=I", _GET_BYPASS_HANDLE)
        while True:
            try:
                # Send RPC to scheduler and wait for response
                reader, writer = await asyncio.open_connection(core_ip, scheduler_port)
                writer.write(req)
                await writer.drain()

                # Wait for response to confirm successful registration with timeout
                try:
                    respbuf = await asyncio.wait_for(reader.readexactly(4 + 4),
                                                     timeout=3.0)
                except asyncio.TimeoutError:
                    logger.info("Timeout waiting for scheduler response")
                    raise
                head, bodylen = struct.unpack("=II", respbuf)
                if head != _GET_BYPASS_HANDLE_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                respbuf = await reader.readexactly(bodylen)

                bypass_handle = self._strdec.decode(respbuf)

                writer.close()
                await writer.wait_closed()

                logger.info("hybrid worker get bypass handle: %s", bypass_handle)

                return bypass_handle

            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise e
                if time.time() - last_log_time > envs.VLLM_LOG_STATS_INTERVAL:
                    logger.warning("failed to get bypass handle: %s", e)
                    last_log_time = time.time()
                await asyncio.sleep(0.1)
                continue
        raise RuntimeError("dead code")

    # disagg thread
    def _do_bypass_meta(self, meta: HybridMetadata):
        self._backend.bind_backend_metadata(meta.reqs)
        self._backend.clear_backend_metadata()

        if self._backend.async_load_kv is None:
            return

        coro = self._async_load_kv(meta)
        asyncio.create_task(coro)
        return

    @kill_me_if_exception
    async def start_bypass_task_loop(self):
        socket_addr = await self._get_bypass_handle()
        context = zmq.asyncio.Context()
        bypass_socket = context.socket(SUB)
        bypass_socket.setsockopt_string(SUBSCRIBE, "")
        logger.info("bypass worker is connecting to %s", socket_addr)
        bypass_socket.connect(socket_addr)

        while True:
            recv = await bypass_socket.recv()
            kv_connector_metadata: HybridMetadata = pickle.loads(recv)
            assert isinstance(kv_connector_metadata, HybridMetadata)
            self._do_bypass_meta(kv_connector_metadata)
        return

    def bind_connector_metadata(self, meta: HybridMetadata):
        self._meta = meta
        self._backend.bind_backend_metadata(meta.reqs)
        return

    def has_connector_metadata(self) -> bool:
        return self._meta is not None

    def clear_connector_metadata(self):
        self._backend.clear_backend_metadata()
        self._meta = None
        return

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # For Hybrid models, allocated kv cache layer num
        # is equal to the number of self.attns layers
        self._num_layers = len(self._kv_cache_config.kv_cache_groups[0].layer_names)
        logger.info("Registering kv_caches, num_layers=%s", self._num_layers)
        return self._backend.register_kv_caches(kv_caches)

    def start_load_kv(self, **kwargs) -> None:
        if self._backend.async_load_kv is None:
            return
        assert self._meta is not None
        coro = self._async_load_kv(self._meta)
        asyncio.run_coroutine_threadsafe(coro, self.loop)
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, **kwargs) -> None:
        if self._meta is None:
            return
        save_coro = self._backend.async_save_kv_layer(
            layer_name, kv_layer, self._meta.reqs
        )
        if save_coro is None:
            return
        coro = self._async_save_layer(layer_name, save_coro)
        asyncio.run_coroutine_threadsafe(coro, self.loop)
        return

    async def _io_done_rpc(self, req: IoDoneReqs, head: int, reshead: int):
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, head, 0)
        reqbufs = self._ioenc.encode_into(req, msgbuf, 8)
        assert len(reqbufs) == 1
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        reader, writer = await self._connpool._acquire()
        writer.write(msgbuf)
        await writer.drain()

        respbuf = await reader.readexactly(4)
        (resphead,) = struct.unpack("=I", respbuf)
        assert resphead == reshead

        self._connpool._release((reader, writer))
        return

    @kill_me_if_exception
    async def _async_load_kv(self, m: HybridMetadata):
        tprank = get_tensor_model_parallel_rank()
        async for reqid in self._backend.async_load_kv(m.reqs):
            loadreq = IoDoneReqs(worker_tprank=tprank, reqids=[reqid])
            await self._io_done_rpc(loadreq, _LOAD_DONE_REQ, _LOAD_DONE_RESP)
        return

    async def _on_req_saved(self, reqid: str, layer_name: str):
        state = self._saving.get(reqid, None)
        if state is None:
            state = _WSavingReq()
            self._saving[reqid] = state
        state._ready_layers.append(layer_name)
        assert self._num_layers is not None
        if len(state._ready_layers) != self._num_layers:
            return

        tprank = get_tensor_model_parallel_rank()
        savereq = IoDoneReqs(worker_tprank=tprank, reqids=[IoRet(reqid=reqid)])
        await self._io_done_rpc(savereq, _SAVE_DONE_REQ, _SAVE_DONE_RESP)
        self._saving.pop(reqid)
        return

    @kill_me_if_exception
    async def _async_save_layer(self, layer_name: str, arg: AsyncGenerator[str, None]):
        async for reqid in arg:
            await self._on_req_saved(reqid, layer_name)
        return


class HybridConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        assert kv_cache_config is not None
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._sched: Optional[HybridScheduler] = None
        self._worker: Optional[HybridWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self._sched = HybridScheduler(vllm_config, kv_cache_config)

            global _g_scheduler
            assert _g_scheduler is None
            _g_scheduler = self._sched
        else:
            self._worker = HybridWorker(vllm_config, kv_cache_config)
        return

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self._sched is not None
        return self._sched.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self._sched is not None
        return self._sched.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self._sched is not None
        return self._sched.build_connector_meta(scheduler_output)

    def step(self) -> Optional[dict[int, list[EngineCoreOutput]]]:
        assert self._sched is not None
        return self._sched.step()

    def has_requests(self) -> bool:
        assert self._sched is not None
        return self._sched.has_requests()

    def on_add_req(self, req: "Request") -> bool:
        assert self._sched is not None
        return self._sched.on_add_req(req)

    def on_abort_req(
        self, reqid: str, reason: str = "", output: bool = True, iscore: bool = True
    ):
        assert self._sched is not None
        self._sched.on_abort_req(
            AbortReq(reqid=reqid, output=output, reason=reason), iscore
        )
        return

    ############################################################
    # Worker Side Methods
    ############################################################
    def bind_connector_metadata(self, m: KVConnectorMetadata) -> None:
        assert isinstance(m, HybridMetadata)
        assert self._worker is not None
        self._worker.bind_connector_metadata(m)
        return

    def has_connector_metadata(self) -> bool:
        assert self._worker is not None
        return self._worker.has_connector_metadata()

    def clear_connector_metadata(self) -> None:
        assert self._worker is not None
        self._worker.clear_connector_metadata()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context, **kwargs) -> None:
        assert self._worker is not None
        self._worker.start_load_kv(**kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kwargs
    ) -> None:
        assert self._worker is not None
        self._worker.save_kv_layer(layer_name, kv_layer, **kwargs)
        return

    def wait_for_save(self):
        return

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return (False, None)

_g_scheduler: Optional[HybridScheduler] = None

# schedule build_connector_meta to execute
def wakeup_scheduler():
    assert _g_scheduler is not None
    _q_append(_g_scheduler._prepared, '__FAKE_REQID_FOR_HYBRID_CONNECTOR__')
    return

def send_bypass_task(kv_connector_metadata):
    assert _g_scheduler is not None
    _g_scheduler.send_bypass_task(kv_connector_metadata)

def hybridsched() -> HybridScheduler:
    global _g_scheduler
    assert _g_scheduler is not None
    return _g_scheduler
