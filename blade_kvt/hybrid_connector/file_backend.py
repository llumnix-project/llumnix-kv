import asyncio
import os.path
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import torch

from . import BackendMeta, HybridBackend, IoRet
from .engine_proxy import (
    KVCacheBlocks,
    KVCacheConfig,
    KVConnectorRole,
    Request,
    SchedulerOutput,
    VllmConfig,
    get_logger,
    get_p_node_pop_len,
    get_tensor_model_parallel_rank,
    sched_get_kvblk_ids,
    sched_get_req,
)

logger = get_logger(__name__)

# ruff: noqa: G004


def _get_blk_hash(tokens: list[int]) -> str:
    return "_".join(str(t) for t in tokens)


@dataclass
class _ReqMeta:
    reqid: str
    is_store: bool
    # (block id, block hash)
    blks: list[tuple[int, str]]


@dataclass
class FileMeta(BackendMeta):
    reqs: list[_ReqMeta]

    def __bool__(self):
        return bool(self.reqs)


def _blk_ready(path) -> bool:
    if not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "ready"))


def _mark_blk_ready(blkdir):
    with open(os.path.join(blkdir, "ready"), "w") as _f:
        pass
    return


class FileBackend(HybridBackend):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        self._cfg = vllm_config
        self._role = role
        self._dir = self._cfg.kv_transfer_config.get_from_extra_config(
            "cachedir", "/tmp"
        )
        self._blksize = self._cfg.cache_config.block_size
        self._num_layers = self._cfg.model_config.get_num_layers(
            self._cfg.parallel_config
        )
        self._gamma = get_p_node_pop_len(self._cfg) - 1
        self._enable_prefix_caching = self._cfg.cache_config.enable_prefix_caching

        ### R/W disagg thread, core thread
        self._load_req: deque[_ReqMeta] = deque()
        self._save_coros: deque[tuple[str, _ReqMeta, Any]] = deque()
        logger.warning(
            "THIS BACKEND IS FOR DEMO ONLY. DO NOT USE IN TEST/PRODUCTION ENV"
        )
        return

    def _get_req(self, reqid: str) -> Optional[Request]:
        return sched_get_req(reqid)

    def _get_kvblks(self, reqid: str) -> list[int]:
        return sched_get_kvblk_ids(reqid, self._gamma, self._enable_prefix_caching)

    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        assert num_computed_tokens % self._blksize == 0
        numtokens = len(req.prompt_token_ids)
        matched_blk = 0
        for idx in range(num_computed_tokens, numtokens, self._blksize):
            eidx = idx + self._blksize
            tokens = req.prompt_token_ids[idx:eidx]
            if len(tokens) < self._blksize:
                break
            blkhash = _get_blk_hash(tokens)
            # The blkhash can be stored in the req using set_param to avoid
            # recalculation later.
            blkpath = os.path.join(self._dir, blkhash)
            ready = _blk_ready(blkpath)
            logger.info(
                f"async_get_num_new_matched_tokens check block "
                f"{req.request_id=} {blkpath=} {ready=}"
            )
            if not ready:
                break
            matched_blk += 1
        external_tokens = matched_blk * self._blksize
        logger.info(
            f"async_get_num_new_matched_tokens "
            f"{req.request_id=} {num_computed_tokens=} {external_tokens=}"
        )
        return external_tokens

    async def async_update_state_after_alloc(
        self, req: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return
        blkidss = blocks.get_block_ids()
        assert len(blkidss) == 1
        blkids = blkidss[0]
        assert num_external_tokens % self._blksize == 0
        assert req.num_computed_tokens % self._blksize == 0
        local_blk_idx = req.num_computed_tokens // self._blksize
        remote_blk_idx = num_external_tokens // self._blksize
        assert local_blk_idx < remote_blk_idx
        blks: list[tuple[int, str]] = []
        for blkidx in range(local_blk_idx, remote_blk_idx):
            tokens = req.prompt_token_ids[
                blkidx * self._blksize : (blkidx + 1) * self._blksize
            ]
            assert len(tokens) == self._blksize
            blkhash = _get_blk_hash(tokens)
            blks.append((blkids[blkidx], blkhash))
        self._load_req.append(_ReqMeta(reqid=req.request_id, is_store=False, blks=blks))
        return

    def get_operations(self, req: Request) -> tuple[int, int]:
        return 1, 1

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        ret: list[_ReqMeta] = []
        while self._load_req:
            r = self._load_req.popleft()
            ret.append(r)

        laststep_reqs: list[Request] = []
        for rdata in sout.scheduled_new_reqs:
            req = self._get_req(rdata.req_id)
            assert req is not None
            new_tokens = sout.num_scheduled_tokens[rdata.req_id]
            new_tokens += rdata.num_computed_tokens  # prompt cache
            num_tokens = len(rdata.prompt_token_ids)
            if new_tokens >= num_tokens:
                laststep_reqs.append(req)
                logger.info(
                    f"laststep req {req.request_id=} {len(req.prompt_token_ids)=} "
                    f"{new_tokens=}"
                )

        req_data = sout.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req = self._get_req(req_id)
            assert req is not None
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_token_ids = req_data.new_token_ids[i]
            if num_computed_tokens + len(new_token_ids) != len(req.prompt_token_ids):
                continue
            laststep_reqs.append(req)
            logger.info(
                f"laststep req {req.request_id=} {len(req.prompt_token_ids)=} "
                f"{len(new_token_ids)=} {num_computed_tokens=}"
            )

        for req in laststep_reqs:
            kvblks = self._get_kvblks(req.request_id)
            blks: list[tuple[int, str]] = []
            for blkidx, blkid in enumerate(kvblks):
                tokens = req.prompt_token_ids[
                    blkidx * self._blksize : (blkidx + 1) * self._blksize
                ]
                logger.info(f"{blkidx=} {len(tokens)=} {self._blksize=}")
                if len(tokens) < self._blksize:
                    break
                blkhash = _get_blk_hash(tokens)
                blks.append((blkid, blkhash))
            ret.append(_ReqMeta(reqid=req.request_id, is_store=True, blks=blks))
        if ret:
            logger.info(f"build_backend_meta {ret=}")
        return FileMeta(reqs=ret)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        self._rank = get_tensor_model_parallel_rank()
        return

    async def async_load_kv(self, m: BackendMeta) -> AsyncGenerator[IoRet, None]:
        q: asyncio.Queue[IoRet] = asyncio.Queue()

        async def async_load_blk(blkid: int, blkhash: str):
            blkdir = os.path.join(self._dir, blkhash)
            assert _blk_ready(blkdir)

            for layer, kvcache in self._kv_caches.items():
                datapath = os.path.join(blkdir, f"{self._rank}.{layer}.data")
                logger.info(f"async_load_blk {blkid=} {datapath=}")
                cpublk = torch.load(datapath)
                assert cpublk.shape == (
                    2,
                    kvcache.size(2),
                    kvcache.size(3),
                    kvcache.size(4),
                )
                kvcache[:, blkid] = cpublk.to(kvcache.device)
            return

        async def async_load_req(req: _ReqMeta):
            logger.info(f"async_load_req {req=}")
            tasks = []
            for blkmeta in req.blks:
                tasks.append(async_load_blk(*blkmeta))
            await asyncio.gather(*tasks)
            q.put_nowait(IoRet(reqid=req.reqid))
            return

        tasks = []
        assert isinstance(m, FileMeta)
        for req in m.reqs:
            assert isinstance(req, _ReqMeta)
            if req.is_store:
                continue
            tasks.append(asyncio.create_task(async_load_req(req)))

        actual_num = 0
        while actual_num < len(tasks):
            yield await q.get()
            actual_num += 1
        return

    async def _save_layer(
        self, layer: str, m: BackendMeta
    ) -> AsyncGenerator[str, None]:
        q: asyncio.Queue[str] = asyncio.Queue()

        async def do_save(layer: str, r: _ReqMeta):
            from vllm.model_executor.models.utils import extract_layer_index

            layer_idx = extract_layer_index(layer)
            kvcache = self._kv_caches[layer]
            logger.info(f"do_save {layer_idx=} {layer=} {r=} {self._num_layers=}")
            for blkmeta in r.blks:
                blkid, blkhash = blkmeta
                blkdir = os.path.join(self._dir, blkhash)
                ready = _blk_ready(blkdir)
                logger.info(f"do_save {blkdir=} {ready=}")
                if ready:
                    continue
                with suppress(FileExistsError):
                    os.mkdir(blkdir)
                datapath = os.path.join(blkdir, f"{self._rank}.{layer}.data")
                blkcpu = kvcache[:, blkid].cpu()
                torch.save(blkcpu, datapath)
                if layer_idx == self._num_layers - 1:
                    _mark_blk_ready(blkdir)
            q.put_nowait(r.reqid)
            return

        tasks = []
        assert isinstance(m, FileMeta)
        for req in m.reqs:
            assert isinstance(req, _ReqMeta)
            if not req.is_store:
                continue
            tasks.append(asyncio.create_task(do_save(layer, req)))

        actual_num = 0
        while actual_num < len(tasks):
            yield await q.get()
            actual_num += 1
        return

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        return self._save_layer(layer_name, m)
