from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import torch

from . import (
    BackendMeta,
    HybridBackend,
    OperationPlan,
    merge_operation_plans,
)
from .engine_proxy import (
    KVCacheBlocks,
    KVCacheConfig,
    KVConnectorRole,
    Request,
    SchedulerOutput,
    VllmConfig,
)
from .kvtbackend import PBackend
from .utils import IoRet


@dataclass
class KVSPMeta(BackendMeta):
    s: BackendMeta
    p: BackendMeta

    def __bool__(self):
        return bool(self.s) or bool(self.p)

class KVSP(HybridBackend):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        kvs_backend = (vllm_config.kv_transfer_config.
                       get_from_extra_config("kvs_backend", "vineyard"))
        if kvs_backend == "vineyard":
            from .kvsbackend import VineyardKVSBackend
            self._s = VineyardKVSBackend(vllm_config, role, kv_cache_config)
        elif kvs_backend == "mooncake":
            from .mooncake_kvsbackend import MooncakeKVSBackend
            self._s = MooncakeKVSBackend(vllm_config, role, kv_cache_config)
        else:
            raise ValueError(
                "kvs_backend must specify either vineyard or "
                "mooncake for 'kvs+kvt' backend"
            )
        self._p = PBackend(vllm_config, role, kv_cache_config)

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._s.register_kv_caches(kv_caches)
        self._p.register_kv_caches(kv_caches)

    def bind_backend_metadata(self, meta: BackendMeta):
        self._s.bind_backend_metadata(meta.s)
        self._p.bind_backend_metadata(meta.p)

    def clear_backend_metadata(self):
        self._s.clear_backend_metadata()
        self._p.clear_backend_metadata()

    def bypass_bind(self, meta: BackendMeta):
        self._s.bypass_bind(meta.s)
        self._p.bypass_bind(meta.p)

    def bypass_clear(self):
        self._s.bypass_clear()
        self._p.bypass_clear()

    async def async_load_kv(
            self, m: BackendMeta) -> AsyncGenerator[IoRet, None]:
        assert isinstance(m, KVSPMeta)
        async for ret in self._s.async_load_kv(m.s):
            yield ret

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        assert isinstance(m, KVSPMeta)
        self._p.async_save_kv_layer(layer_name, kv_layer, m.p)
        return self._s.async_save_kv_layer(layer_name, kv_layer, m.s)

    def save_done_source(self) -> str | None:
        return self._s.save_done_source()

    # ==============================
    # Scheduler-side methods
    # ==============================
    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        return await self._s.async_get_num_new_matched_tokens(req, num_computed_tokens)

    async def async_update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> Optional[IoRet]:
        return await self._s.async_update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def get_operations(self, req: Request) -> OperationPlan:
        # p has to go first bc kvt will modify the request
        p_ops = self._p.get_operations(req)
        s_ops = self._s.get_operations(req)
        return merge_operation_plans(s_ops, p_ops)

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        return KVSPMeta(
            s=self._s.build_backend_meta(sout),
            p=self._p.build_backend_meta(sout),
        )
