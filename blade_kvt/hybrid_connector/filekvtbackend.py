import os
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import torch

from . import BackendMeta, HybridBackend, IoRet
from .engine_proxy import (
    KVCacheBlocks,
    KVCacheConfig,
    KVConnectorRole,
    Request,
    SchedulerOutput,
    VllmConfig,
)
from .file_backend import FileBackend
from .kvtbackend import PBackend


@dataclass
class FilePMeta(BackendMeta):
    file: BackendMeta
    kvtp: BackendMeta

    def __bool__(self):
        return self.file or self.kvtp


class FilePBackend(HybridBackend):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        if role == KVConnectorRole.WORKER:
            # see kvt/envcfg.h
            SEND_DONE_HEAD_KIND = "1"
            os.environ.setdefault("BLLM_KVTRANS_SDH_KIND", SEND_DONE_HEAD_KIND)
        self._file = FileBackend(vllm_config, role, kv_cache_config)
        self._kvtp = PBackend(vllm_config, role, kv_cache_config)

    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        n = await self._file.async_get_num_new_matched_tokens(req, num_computed_tokens)
        return n

    async def async_update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        await self._file.async_update_state_after_alloc(
            request, blocks, num_external_tokens
        )
        return

    async def async_cleanup(self, req: Request):
        await self._file.async_cleanup(req)
        await self._kvtp.async_cleanup(req)
        return

    def get_operations(self, req: Request) -> tuple[int, int]:
        return 1, 1

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        file = self._file.build_backend_meta(sout)
        kvtp = self._kvtp.build_backend_meta(sout)
        ret = FilePMeta(file=file, kvtp=kvtp)
        return ret

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._file.register_kv_caches(kv_caches)
        self._kvtp.register_kv_caches(kv_caches)
        return

    def bind_backend_metadata(self, meta: BackendMeta):
        assert isinstance(meta, FilePMeta)
        self._file.bind_backend_metadata(meta.file)
        self._kvtp.bind_backend_metadata(meta.kvtp)
        return

    def clear_backend_metadata(self):
        self._file.clear_backend_metadata()
        self._kvtp.clear_backend_metadata()
        return

    async def async_load_kv(self, m: BackendMeta) -> AsyncGenerator[IoRet, None]:
        assert isinstance(m, FilePMeta)
        async for reqid in self._file.async_load_kv(m.file):
            yield reqid

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        assert isinstance(m, FilePMeta)
        self._kvtp.async_save_kv_layer(layer_name, kv_layer, m.kvtp)
        r = self._file.async_save_kv_layer(layer_name, kv_layer, m.file)
        return r
