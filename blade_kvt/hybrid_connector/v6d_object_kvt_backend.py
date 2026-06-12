"""
V6dObjectKVTBackend: Owner-Sidecar design for P-node KV cache management.

This backend combines two independent backends with distinct responsibilities:

- V6D (Owner): Manages shared KV cache in Vineyard for prefix caching within P-node.
  V6D is the sole decision-maker for load/save counts, controlling block allocation
  and determining when KV cache needs to be stored locally.

- KVT (Sidecar): Handles P→D direct transfer via RDMA/TCP for low-latency decode.
  KVT operates independently of V6D's decisions - it generates its own save meta
  based on P_KVT_STATE (set by D-node via _on_prefill RPC), not save_count.

Key design principle: V6D's save_count does NOT gate KVT's save operation.
This allows KVT to transfer KV cache to D-node even when V6D doesn't need to store
(e.g., when prefill is complete but D-node requests transfer).

Hybrid model truncation:
For Qwen3-Next hybrid models with speculative decoding, KVT truncates the prompt
by gamma+1 tokens before scheduling. This ensures both V6D and KVT operate on the
same truncated prompt, saving:
- P-node prefill computation for the last gamma+1 tokens
- V6D storage for the last gamma+1 tokens
The last gamma+1 tokens are prefill'd locally on D-node to ensure correct GDN state.
"""
import inspect
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional

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
from .kvtbackend import PBackend
from .v6d_object_backend import V6dObjectBackend


@dataclass
class V6dObjectKVTMeta(BackendMeta):
    v6d: BackendMeta
    kvt: BackendMeta

    def __bool__(self):
        return bool(self.v6d) or bool(self.kvt)


class V6dObjectKVTBackend(HybridBackend):

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
        if (vllm_config.kv_transfer_config is None
                or not vllm_config.kv_transfer_config.is_kv_producer):
            raise ValueError(
                "v6d_object+kvt backend requires producer-capable kv_role "
                "(kv_producer or kv_both)."
            )
        self._v6d = V6dObjectBackend(vllm_config, role, kv_cache_config)
        self._kvt = PBackend(vllm_config, role, kv_cache_config)

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._v6d.register_kv_caches(kv_caches)
        self._kvt.register_kv_caches(kv_caches)

    def bind_backend_metadata(self, meta: BackendMeta):
        assert isinstance(meta, V6dObjectKVTMeta)
        self._v6d.bind_backend_metadata(meta.v6d)
        self._kvt.bind_backend_metadata(meta.kvt)

    def clear_backend_metadata(self):
        self._v6d.clear_backend_metadata()
        self._kvt.clear_backend_metadata()

    def bypass_bind(self, meta: BackendMeta):
        assert isinstance(meta, V6dObjectKVTMeta)
        self._v6d.bypass_bind(meta.v6d)
        self._kvt.bypass_bind(meta.kvt)

    def bypass_clear(self):
        self._v6d.bypass_clear()
        self._kvt.bypass_clear()

    async def async_load_kv(
        self, m: BackendMeta
    ) -> AsyncGenerator[IoRet, None]:
        assert isinstance(m, V6dObjectKVTMeta)
        async for ret in self._v6d.async_load_kv(m.v6d):
            yield ret

    def async_save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        m: BackendMeta,
    ) -> Optional[AsyncGenerator[str, None]]:
        assert isinstance(m, V6dObjectKVTMeta)
        self._kvt.async_save_kv_layer(layer_name, kv_layer, m.kvt)
        return self._v6d.async_save_kv_layer(layer_name, kv_layer, m.v6d)

    # ==============================
    # Scheduler-side methods
    # ==============================
    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        return await self._v6d.async_get_num_new_matched_tokens(
            req, num_computed_tokens
        )

    async def async_update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> Optional[IoRet]:
        return await self._v6d.async_update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    async def async_cleanup(self, req: Request):
        await self._v6d.async_cleanup(req)

        cleanup = getattr(self._kvt, "async_cleanup", None)
        if not callable(cleanup):
            return

        maybe_awaitable = cleanup(req)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    def get_operations(self, req: Request) -> tuple[int, int]:
        # =====================================================================
        # Owner-Sidecar Pattern with Shared Truncation
        # =====================================================================
        # For hybrid models (Qwen3-Next with speculative decoding), KVT truncates
        # the prompt by gamma+1 tokens first. This ensures:
        # 1. Scheduler only schedules n-gamma-1 tokens for prefill
        # 2. V6D only stores n-gamma-1 tokens (saving storage)
        # 3. KVT only transfers n-gamma-1 tokens to D-node
        # 4. D-node locally prefills the last gamma+1 tokens for correct GDN state
        #
        # V6D remains the sole decision-maker for save_count (owner pattern).
        # KVT's truncate affects the request state, so both backends see the
        # same truncated prompt.
        kvt_load_count, kvt_save_count = self._kvt.get_operations(req)
        v6d_load_count, v6d_save_count = self._v6d.get_operations(req)
        return kvt_load_count + v6d_load_count, kvt_save_count + v6d_save_count

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        # Both backends build meta independently:
        # - V6D: checks has_setup_save() to decide if save meta is needed
        # - KVT: checks P_KVT_STATE to decide if save meta is needed
        # This allows KVT to generate save meta even when V6D doesn't (sidecar mode).
        return V6dObjectKVTMeta(
            v6d=self._v6d.build_backend_meta(sout),
            kvt=self._kvt.build_backend_meta(sout),
        )
