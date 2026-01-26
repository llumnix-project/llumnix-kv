import asyncio
import os
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import torch
import xxhash
from vineyard.llm.kvcachestore import KVCacheStore

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.tracing import TraceWrapper
from vllm.utils import EventPool
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from . import BackendMeta, HybridBackend, IoRet
from ..engine_proxy import (
    get_p_node_pop_len,
    sched_get_kvblk_ids,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from . import HCSchedOutput, hybridsched

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    request_id: str
    is_store: bool
    token_ids: list[int]
    block_ids: list[int]
    block_hashes: list[int]
    num_cached_tokens_local: int  # only matters for load
    is_last_chunk: bool = False  # only matters for save
    skip_store: bool = False  # useful when multiple draft steps
    udc_ttl: int = 0 # only matters for save
    trace_headers: str = ""

    def __repr__(self) -> str:
        if len(self.token_ids) <= 10:
            token_ids_str = str(self.token_ids)
        else:
            token_ids_str = (f'[{", ".join(map(str, self.token_ids[:5]))}'
                             f', ..., '
                             f'{", ".join(map(str, self.token_ids[-5:]))}]')

        if len(self.block_ids) <= 6:
            block_ids_str = str(self.block_ids)
        else:
            block_ids_str = (f'[{", ".join(map(str, self.block_ids[:3]))}'
                             f', ..., '
                             f'{", ".join(map(str, self.block_ids[-3:]))}]')
        
        if len(self.block_hashes) <= 6:
            block_hashes_str = str(self.block_hashes)
        else:
            block_hashes_str = (
                f'[{", ".join(map(str, self.block_hashes[:3]))}'
                f', ..., '
                f'{", ".join(map(str, self.block_hashes[-3:]))}]')  

        return (f"ReqMeta(request_id={self.request_id}, "
                f"is_store={self.is_store}, "
                f"token_ids({len(self.token_ids)})={token_ids_str}, "
                f"block_ids({len(self.block_ids)})={block_ids_str}, "
                f"block_hashes({len(self.block_hashes)})={block_hashes_str}, "
                f"num_cached_tokens_local={self.num_cached_tokens_local}, "
                f"is_last_chunk={self.is_last_chunk}, "
                f"skip_store={self.skip_store}), "
                f"udc_ttl={self.udc_ttl})")


@dataclass
class VineyardKVSBackendMetadata(BackendMeta):
    metas: list[ReqMeta] = field(default_factory=list)

    def add(self, meta: ReqMeta) -> None:
        if meta is not None:
            self.metas.append(meta)

    def __add__(
            self, other: "VineyardKVSBackendMetadata"
    ) -> "VineyardKVSBackendMetadata":
        if not isinstance(other, VineyardKVSBackendMetadata):
            raise TypeError(
                f"Cannot add {type(other)} to VineyardKVSBackendMetadata")
        new_meta = VineyardKVSBackendMetadata()
        new_meta.metas = self.metas + other.metas
        return new_meta

    def __bool__(self) -> bool:
        return bool(self.metas)


class VineyardKVSBackend(HybridBackend):

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.kv_role == "kv_both", \
            "VineyardKVSBackend only supports 'kv_both' kv_role"

        # worker-side variables
        self.vllm_config = vllm_config
        self.layer_name_to_index: dict[str, int] = {}
        self.num_layers = self.vllm_config.model_config.get_num_layers(
            self.vllm_config.parallel_config)
        self._gamma = 0
        if self.vllm_config.speculative_config is not None:
            if self.vllm_config.speculative_config.use_eagle():
                self.num_layers += (
                    self.vllm_config.speculative_config.draft_model_config.
                    get_num_layers(
                        self.vllm_config.speculative_config.draft_parallel_config))
                self._gamma = get_p_node_pop_len(self.vllm_config) - 1
        self._enable_prefix_caching = (
            self.vllm_config.cache_config.enable_prefix_caching
        )

        self.device = torch.device(torch.cuda.current_device())
        self.event_pool = EventPool(256, self.device)
        self.io_timeout_seconds = envs.VLLM_KVS_IO_TIMEOUT_SECONDS

        # scheduler-side variables
        self.metas_to_recv: deque[ReqMeta] = deque()

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert len(kv_caches) == self.num_layers, \
            "The number of kv caches is not equal to the number of layers"
        
        v6d_ipc_socket = self.vllm_config.kv_transfer_config.get_from_extra_config(
                "sock", None)
        if v6d_ipc_socket is None:
            logger.warning("V6D not launched. Will not registering kvs backend.")
            return
        
        for layer_name in kv_caches:
            self.layer_name_to_index[layer_name] = extract_layer_index(
                layer_name)

        kv_cache_tensors = list(kv_caches.values())
        llm_kv_cache_layer_shape = list(kv_cache_tensors[0].shape)
        kernel_block_size = llm_kv_cache_layer_shape[2]
        kv_manager_block_size = self.vllm_config.cache_config.block_size
        self.blocks_per_kv_block = kv_manager_block_size // kernel_block_size

        if envs.VLLM_KVS_USE_REQUEST_HASH:
            assert self.blocks_per_kv_block == 1

        if (self.blocks_per_kv_block != 1):
            logger.warning("The kernel block size is not equal to the "
                           "kv manager block size. This may cause a little "
                           "overhead in kv cache offloading.")

        self.kv_store = KVCacheStore(
            ipc_socket=v6d_ipc_socket,
            model_name=self.vllm_config.model_config.model,
            num_layers=len(kv_caches),
            llm_kv_cache_layer_shape=llm_kv_cache_layer_shape,
            item_size=kv_cache_tensors[0].element_size(),
            cache_space_size=self.vllm_config.cache_config.swap_space,
            num_rank=self.vllm_config.parallel_config.tensor_parallel_size)
        self.kv_store.init_rank(get_tp_group().rank)
        self.kv_store.bind_kv_caches(kv_cache_tensors)

    def _enlarge_list(self, lst: list, n: int):
        # res[i] = lst[i // n] * n + (i % n)
        len_lst = len(lst)
        len_res = len_lst * n
        res = [0] * len_res
        for i in range(len_res):
            res[i] = lst[i // n] * n + (i % n)
        return res

    async def async_load_kv(
            self, metadata: BackendMeta) -> AsyncGenerator[IoRet, None]:
        assert isinstance(metadata, VineyardKVSBackendMetadata)
        metas = [meta for meta in metadata.metas if not meta.is_store]

        futures: list[list[asyncio.Future]] = []
        future_to_request_id = {}
        loaded: dict[str, list[int]] = {}
        requests_to_yield: set[str] = set()

        for meta in metas:
            logger.debug(f"issue load {repr(meta)}")
            request_id = meta.request_id
            block_hashes = meta.block_hashes
            requests_to_yield.add(request_id)
            loaded[request_id] = []
            request_context = request_id
            if meta.trace_headers:
                request_context += f";{meta.trace_headers}"
            block_ids = meta.block_ids
            if (self.blocks_per_kv_block > 1):
                block_ids = self._enlarge_list(block_ids,
                                              self.blocks_per_kv_block)

            request_futures = await self.kv_store.load(
                meta.token_ids,
                block_ids,
                block_hashes,
                meta.num_cached_tokens_local,
                request_context=request_context)
            futures.extend(request_futures)
            for future in request_futures:
                future_to_request_id[future] = request_id

        async for future in as_completed(futures, self.io_timeout_seconds):
            request_id = future_to_request_id[future]
            request_loaded = loaded[request_id]
            request_loaded.append(future.result())
            if len(request_loaded) == self.num_layers:
                loaded_min = min(request_loaded)
                loaded_max = max(request_loaded)
                logger.debug(f"{loaded_min=}, {loaded_max=}, "
                             f"{request_id=}")
                requests_to_yield.remove(request_id)
                yield IoRet(reqid=request_id, n=loaded_min)

        if requests_to_yield:
            raise asyncio.TimeoutError(
                f"load timeout after {self.io_timeout_seconds}s for "
                f"{requests_to_yield}, v6d might error"
            )

    def async_save_kv_layer(
            self, layer_name: str, kv_layer: torch.Tensor,
            metadata: BackendMeta) -> AsyncGenerator[str, None]:
        assert isinstance(metadata, VineyardKVSBackendMetadata)
        metas = [
            meta for meta in metadata.metas
            if (meta.is_store and not meta.skip_store)
        ]

        if not metas:
            return None

        forward_stream = torch.cuda.current_stream()
        event = self.event_pool.get_event()
        event.record(forward_stream)

        layer_idx = self.layer_name_to_index[layer_name]

        async def _save_kv_layer(event) -> AsyncGenerator[str, None]:
            futures = []
            future_to_request_id = {}
            requests_to_yield = set()
            requests_without_ttl = set()

            for meta in metas:
                request_id = meta.request_id
                if not meta.token_ids:
                    yield request_id
                else:
                    request_context = request_id
                    if meta.trace_headers:
                        request_context += f";{meta.trace_headers}"
                        if layer_idx == 0:
                            request_context += ";op=start"
                        elif layer_idx == self.num_layers - 1:
                            request_context += ";op=end"
                        
                    if layer_idx == 0:
                        logger.debug(f"issue save {repr(meta)}")

                    if meta.is_last_chunk:
                        requests_to_yield.add(request_id)
                        if not meta.udc_ttl:
                            requests_without_ttl.add(request_id)

                    block_ids = meta.block_ids
                    if (self.blocks_per_kv_block > 1):
                        block_ids = self._enlarge_list(block_ids,
                                                      self.blocks_per_kv_block)
                    future = await self.kv_store.store_layer(meta.token_ids,
                                                            block_ids,
                                                            meta.block_hashes,
                                                            layer_idx,
                                                            event=event,
                                                            udc_ttl=meta.udc_ttl,
                                                            request_context=request_context)

                    futures.append(future)
                    future_to_request_id[future] = request_id

            async for future in as_completed(futures, self.io_timeout_seconds):
                await future # to raise exception if any
                request_id = future_to_request_id[future]
                if request_id in requests_to_yield:
                    requests_to_yield.remove(request_id)
                    yield request_id

            if requests_to_yield:
                if any(
                    request in requests_without_ttl for request in requests_to_yield):
                    raise asyncio.TimeoutError(
                        f"save layer {layer_idx} timeout after "
                        f"{self.io_timeout_seconds}s for {requests_to_yield}, "
                        f"v6d might error")

                logger.info(
                    f"save layer {layer_idx} timeout after "
                    f"{self.io_timeout_seconds}s for {requests_to_yield}, "
                    f"v6d might error")

                for request_id in requests_to_yield:
                    yield request_id

            self.event_pool.put_event(event)

        if layer_idx == self.num_layers - 1:
            for meta in metas:
                meta.skip_store = True

        return _save_kv_layer(event)

    def get_operations(self, req: Request) -> tuple[int, int]:
        return int(should_load(req)), int(should_save(req))

    async def async_get_num_new_matched_tokens(
        self,
        req: "Request",
        num_computed_tokens: int,
    ) -> int:
        # no lookup support for now, try to load all but the last token
        new_matched_tokens = req.num_tokens - num_computed_tokens - 1
        assert new_matched_tokens >= 0
        return new_matched_tokens

    async def async_update_state_after_alloc(self, request: "Request",
                                             blocks: "KVCacheBlocks",
                                             num_external_tokens: int):
        if num_external_tokens > 0:            
            if envs.VLLM_KVS_USE_REQUEST_HASH:
                # VLLM does not store incomplete blocks or compute their hashes.  
                # Adjust block_num to equal the number of block hashes.
                block_num = len(request.block_hashes)
                # This function maps 128-bit hashes to 64-bit 
                # Integers to maintain consistency with v6d hash calculations.
                block_hashes = xxh64_hash_bytes_list(request.block_hashes)[:block_num]
                hashed_tokens_num = block_num * self.vllm_config.cache_config.block_size
                # Fix: Ensure we never include the very last token of the prompt, 
                # even if the last block is perfectly aligned/full.
                hashed_tokens_num = min(hashed_tokens_num, 
                                        request.num_prompt_tokens - 1)
                tokens = request.prompt_token_ids[:hashed_tokens_num]
            else:
                block_num = cdiv(request.num_prompt_tokens - 1,
                            self.vllm_config.cache_config.block_size)
                block_hashes = []
                tokens = request.prompt_token_ids[:-1]
            
            block_ids = blocks.get_block_ids()[0][:block_num]
            assert len(block_ids) == block_num
            trace_headers = request.trace_wrapper.extract_trace_headers("kv_load")
            self.metas_to_recv.append(
                ReqMeta(request.request_id, False, tokens,
                        block_ids, block_hashes,
                        request.num_prompt_tokens - num_external_tokens - 1, 
                        trace_headers=trace_headers))

    def build_backend_meta(self,
                           scheduler_output: SchedulerOutput) -> BackendMeta:
        assert isinstance(scheduler_output, HCSchedOutput)
        from .utils import kill_me_if_exception
        @kill_me_if_exception
        async def abort_save(reqs: list[Request]):
            await asyncio.sleep(envs.VLLM_KVS_IO_TIMEOUT_SECONDS + 1)

            tasks = []
            for req in reqs:
                for rank in range(
                    self.vllm_config.parallel_config.tensor_parallel_size):
                    tasks.append(
                        hybridsched()._do_save_done(
                            rank, ioret=IoRet(req.request_id)))

            await asyncio.gather(*tasks)

        from .engine_proxy import get_hybrid_sched_loop
        if scheduler_output.hc_aborted_save:
            logger.debug(
                f"register abort save cb for "
                f"{[req.request_id for req in scheduler_output.hc_aborted_save]}")
            asyncio.run_coroutine_threadsafe(
                abort_save(scheduler_output.hc_aborted_save),
                get_hybrid_sched_loop())

        meta_load = self._build_load()
        meta_save = self._build_save(scheduler_output)
        meta = meta_load + meta_save
        logger.debug(f"build meta: {meta}")
        return meta

    def _build_load(self) -> VineyardKVSBackendMetadata:
        meta = VineyardKVSBackendMetadata()
        while self.metas_to_recv:
            req_meta = self.metas_to_recv.popleft()
            meta.add(req_meta)
        return meta

    def _build_save_meta_for_request(self, request_id: str,
                                     num_scheduled_tokens: int) -> ReqMeta:
        request = self.get_request(request_id)
        assert request is not None

        if request.num_computed_tokens == request.num_prompt_tokens - 1:
            return ReqMeta(request_id, True, [], [], [], 0, True)

        num_computed_token_ids = (request.num_computed_tokens +
                                num_scheduled_tokens)
        num_tokens_to_save = min(num_computed_token_ids,
                                request.num_prompt_tokens - 1)
        num_populated_blocks = cdiv(num_tokens_to_save,
                                    self.vllm_config.cache_config.block_size)
        if envs.VLLM_KVS_USE_REQUEST_HASH:
            # VLLM does not store incomplete blocks or compute their hashes.  
            # Adjust block_num to equal the number of block hashes.
            num_populated_blocks = min(len(request.block_hashes), num_populated_blocks)
            hashed_tokens_num = num_populated_blocks * \
                                    self.vllm_config.cache_config.block_size
            # Fix: Ensure we never include the very last token of the prompt, 
            # even if the last block is perfectly aligned/full.
            hashed_tokens_num = min(hashed_tokens_num, request.num_prompt_tokens - 1)
            token_ids_to_save = request.prompt_token_ids[:hashed_tokens_num]
            # This function maps 128-bit hashes to 64-bit
            # Integers to maintain consistency with v6d hash calculations.
            block_hashes = xxh64_hash_bytes_list(
            request.block_hashes[:num_populated_blocks])
        else:
            token_ids_to_save = request.prompt_token_ids[:num_tokens_to_save]
            block_hashes = []
        populated_block_ids = sched_get_kvblk_ids(
            request_id, self._gamma, self._enable_prefix_caching,
        )[:num_populated_blocks]

        udc_ttl = 300 if (request.cache_control_params is not None and \
            "NOCPFS" not in request_id) else 0

        trace_headers = request.trace_wrapper.extract_trace_headers("kv_save")
        return ReqMeta(request_id, True, token_ids_to_save,
                    populated_block_ids, block_hashes, 0, num_computed_token_ids
                    >= request.num_prompt_tokens, udc_ttl=udc_ttl, 
                    trace_headers=trace_headers)

    def _build_save(
            self,
            scheduler_output: SchedulerOutput) -> VineyardKVSBackendMetadata:
        meta = VineyardKVSBackendMetadata()

        for req in scheduler_output.scheduled_new_reqs:
            request_id = req.req_id
            request = self.get_request(request_id)
            assert request is not None
            if should_save(request):
                meta.add(
                    self._build_save_meta_for_request(
                        request_id,
                        scheduler_output.num_scheduled_tokens[request_id]))

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for request_id in cached_reqs.req_ids:
            if "NOSAVE" in request_id:
                continue
            request = self.get_request(request_id)
            assert request is not None
            if should_save(request):
                meta.add(
                    self._build_save_meta_for_request(
                        request_id,
                        scheduler_output.num_scheduled_tokens[request_id]))

        return meta


async def as_completed(futures: list[asyncio.Future], timeout: float):

    if not futures:
        return

    end_time = time.monotonic() + timeout
    while futures:
        remaining_time = end_time - time.monotonic()
        if remaining_time <= 0:
            break

        done, pending = await asyncio.wait(futures,
                                           timeout=remaining_time,
                                           return_when=asyncio.FIRST_COMPLETED)

        for future in done:
            yield future

        futures = pending

def should_save(req: Request) -> bool:
    if req.num_prompt_tokens <= envs.VLLM_KVS_ON_MIN_LENGTH + 1:
        return False
    return not req.prefill_done

def should_load(req: Request) -> bool:
    return not (req.num_prompt_tokens <= envs.VLLM_KVS_ON_MIN_LENGTH + 1)

def xxh64_hash_bytes_list(bs_list: list[bytes]) -> list[int]:
    return [xxhash.xxh64(b).intdigest() for b in bs_list]
