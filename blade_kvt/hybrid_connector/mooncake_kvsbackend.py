import asyncio
import json
import os
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Optional

import torch
from mooncake.store import MooncakeDistributedStore

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
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

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min
DEFAULT_MASTER_METRICS_PORT = 9003
DEFAULT_CHECK_SERVER = False
DISABLE_SAVE = os.getenv("VLLM_MCKVS_DISABLE_SAVE") == "1"
DISABLE_LOAD = os.getenv("VLLM_MCKVS_DISABLE_LOAD") == "1"

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    request_id: str
    is_store: bool
    token_ids: list[int]
    block_ids: list[int]
    block_hashes: list[str] | None
    num_cached_tokens_local: int  # only matters for load
    is_last_chunk: bool = False  # only matters for save
    skip_store: bool = False  # useful when multiple draft steps
    udc_ttl: int = 0  # only matters for save

    def __repr__(self) -> str:
        if len(self.token_ids) <= 10:
            token_ids_str = str(self.token_ids)
        else:
            token_ids_str = (
                f"[{', '.join(map(str, self.token_ids[:5]))}"
                f", ..., "
                f"{', '.join(map(str, self.token_ids[-5:]))}]"
            )

        if len(self.block_ids) <= 6:
            block_ids_str = str(self.block_ids)
        else:
            block_ids_str = (
                f"[{', '.join(map(str, self.block_ids[:3]))}"
                f", ..., "
                f"{', '.join(map(str, self.block_ids[-3:]))}]"
            )

        return (
            f"ReqMeta(request_id={self.request_id}, "
            f"is_store={self.is_store}, "
            f"token_ids({len(self.token_ids)})={token_ids_str}, "
            f"block_ids({len(self.block_ids)})={block_ids_str}, "
            f"block_hashes={len(self.block_hashes)}={self.block_hashes}, "
            f"num_cached_tokens_local={self.num_cached_tokens_local}, "
            f"is_last_chunk={self.is_last_chunk}, "
            f"skip_store={self.skip_store}), "
            f"udc_ttl={self.udc_ttl})"
        )


def _parse_global_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int
    check_server: bool

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """Load config from extra_config dictionary."""
        if "local_hostname" not in extra_config:
            from .engine_proxy import get_ip
            local_hostname = get_ip()
        else:
            local_hostname = extra_config["local_hostname"]
        if "master_server_address" not in extra_config:
            raise ValueError("master_server_address is required in extra_config")

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=extra_config.get("metadata_server", "P2PHANDSHAKE"),
            global_segment_size=_parse_global_segment_size(
                extra_config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=extra_config.get(
                "local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE
            ),
            protocol=extra_config.get("protocol", "tcp"),
            device_name=extra_config.get("device_name", ""),
            master_server_address=extra_config["master_server_address"],
            master_metrics_port=extra_config.get(
                "master_metrics_port", DEFAULT_MASTER_METRICS_PORT
            ),
            check_server=extra_config.get("check_server", DEFAULT_CHECK_SERVER),
        )


@dataclass
class MooncakeKVSBackendMetadata(BackendMeta):
    metas: list[ReqMeta] = field(default_factory=list)

    def add(self, meta: ReqMeta) -> None:
        if meta is not None:
            self.metas.append(meta)

    def __add__(
        self, other: "MooncakeKVSBackendMetadata"
    ) -> "MooncakeKVSBackendMetadata":
        if not isinstance(other, MooncakeKVSBackendMetadata):
            raise TypeError(f"Cannot add {type(other)} to MooncakeKVSBackendMetadata")
        new_meta = MooncakeKVSBackendMetadata()
        new_meta.metas = self.metas + other.metas
        return new_meta

    def __bool__(self) -> bool:
        return bool(self.metas)

class MooncakeKVSBackend(HybridBackend):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.kv_role == "kv_both", (
            "MooncakeKVSBackend only supports 'kv_both' kv_role"
        )

        # worker-side variables
        self.vllm_config = vllm_config
        self.layer_name_to_index: dict[str, int] = {}
        self.num_layers = self.vllm_config.model_config.get_num_layers(
            self.vllm_config.parallel_config
        )
        if self.vllm_config.speculative_config is not None:
            assert self.vllm_config.speculative_config.use_eagle(), (
                "MooncakeKVSBackend only supports eagle method"
            )

            self.num_layers += (
                self.vllm_config.speculative_config.draft_model_config.get_num_layers(
                    self.vllm_config.speculative_config.draft_parallel_config
                )
            )
        self._gamma = get_p_node_pop_len(self.vllm_config) - 1
        self._enable_prefix_caching = (
            self.vllm_config.cache_config.enable_prefix_caching
        )

        self.device = torch.device(torch.cuda.current_device())
        self.event_pool = EventPool(256, self.device)
        self.io_timeout_seconds = envs.VLLM_KVS_IO_TIMEOUT_SECONDS

        self.metas_to_recv: deque[ReqMeta] = deque()

        self.mooncake_config: MooncakeStoreConfig = (
            MooncakeStoreConfig.load_from_extra_config(
                self.vllm_config.kv_transfer_config.kv_connector_extra_config
            )
        )
        self.store = MooncakeDistributedStore()
        self.block_size = self.vllm_config.cache_config.block_size
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.block_bytes = None
        self.load_bytes_on_success = None
        self.kv_caches = None
        self.tp_rank = None
        self.request_saved_blocks: dict[str, int] = {}
        self.init_mooncake(role)

    def init_mooncake(self, role):
        if role == KVConnectorRole.SCHEDULER:
            ret_code = self.store.setup(
                self.mooncake_config.local_hostname,
                self.mooncake_config.metadata_server,
                0,
                DEFAULT_LOCAL_BUFFER_SIZE,
                "tcp",
                "",
                self.mooncake_config.master_server_address,
            )
            if ret_code:
                logger.error(
                    "failed to setup mooncake store for scheduler, error code: %d",
                    ret_code)

        elif role == KVConnectorRole.WORKER:
            per_tp_global_segment_size = (
                    self.mooncake_config.global_segment_size // self.tp_size
            )
            per_tp_local_buffer_size = (
                    self.mooncake_config.local_buffer_size // self.tp_size
            )

            self.tp_rank = get_tp_group().rank
            # Handle JSON device_name configuration
            device_name = self.mooncake_config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    device_name = device_config.get(self.tp_rank, "")
                    if not device_name:
                        device_name = device_config.get(str(self.tp_rank), "")

                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"Failed to parse device_name as JSON: {device_name}")
                    device_name = ""

            ret_code = self.store.setup(
                self.mooncake_config.local_hostname,
                self.mooncake_config.metadata_server,
                per_tp_global_segment_size,
                per_tp_local_buffer_size,
                self.mooncake_config.protocol,
                device_name,
                self.mooncake_config.master_server_address,
            )

            if ret_code:
                logger.error(
                    "failed to setup mooncake store for scheduler, error code: %d",
                    ret_code)

            logger.info("Connect to Mooncake store successfully.")
            self.warmup()
            logger.info("Mooncake store warmup successfully.")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert len(kv_caches) == self.num_layers, (
            "The number of kv caches is not equal to the number of layers"
        )
        self.kv_caches = kv_caches
        ret = self.store.register_kv_caches(list(kv_caches.values()), self.tp_rank)
        assert(all(x == 0 for x in ret))
        for layer_name, cache_tensor in kv_caches.items():
            self.layer_name_to_index[layer_name] = extract_layer_index(layer_name)
        sample_tensor = list(kv_caches.values())[0]
        if len(sample_tensor.shape) == 5:
            sample_block = sample_tensor[0, 0]
            self.block_bytes = sample_block.numel() * sample_block.element_size()
            self.load_bytes_on_success = self.block_bytes * self.num_layers * 2
        elif len(sample_tensor.shape) == 3:
            sample_block = sample_tensor[0]
            self.block_bytes = sample_block.numel() * sample_block.element_size()
            self.load_bytes_on_success = self.block_bytes * self.num_layers
        else:
            raise ValueError("Unsupported kv cache tensor shape.")

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB
        assert self.store.put(warmup_key, warmup_value) == 0
        assert self.store.is_exist(warmup_key) == 1
        assert self.store.get(warmup_key) == warmup_value

    async def async_load_kv(self, metadata: BackendMeta) -> AsyncGenerator[IoRet, None]:
        assert isinstance(metadata, MooncakeKVSBackendMetadata)
        metas = [meta for meta in metadata.metas if not meta.is_store]
        loop = asyncio.get_running_loop()
        tasks = []

        for meta in metas:
            assert len(meta.block_hashes) == len(meta.block_ids)
            if len(meta.block_ids) == 0:
                yield IoRet(reqid=meta.request_id, n=0)
                continue

            meta_futures = [loop.create_future() for _ in range(len(meta.block_ids))]
            self.store.batch_load_kv_async(meta.block_ids, meta.block_hashes,
                                           loop, meta_futures)

            async def meta_worker(meta, futures_for_meta):
                results = await asyncio.gather(*futures_for_meta)
                num_loaded_block = 0
                for get_bytes in results:
                    if get_bytes == self.load_bytes_on_success:
                        num_loaded_block += 1
                    else:
                        break

                num_loaded_token = min(
                    num_loaded_block * self.block_size, len(meta.token_ids)
                )
                if (num_loaded_token + meta.num_cached_tokens_local
                        == len(meta.token_ids)):
                    num_loaded_token -= 1
                return IoRet(reqid=meta.request_id, n=num_loaded_token)

            task = asyncio.create_task(
                meta_worker(meta, meta_futures)
            )
            tasks.append(task)
        if not tasks:
            return

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                io_ret = t.result()
                yield io_ret

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, metadata: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        assert isinstance(metadata, MooncakeKVSBackendMetadata)

        metas = [
            meta for meta in metadata.metas if (meta.is_store and not meta.skip_store)
        ]

        if not metas:
            return None

        forward_stream = torch.cuda.current_stream()
        event = self.event_pool.get_event()
        event.record(forward_stream)

        async def _save_kv_layer(event) -> AsyncGenerator[str, None]:
            loop = asyncio.get_running_loop()
            layer_idx = self.layer_name_to_index[layer_name]
            if layer_idx < self.num_layers - 1:
                for meta in metas:
                    if meta.is_last_chunk:
                        yield meta.request_id
                return

            hash_values = []
            block_ids_to_save = []
            requests_to_yield = set()

            for meta in metas:
                if not meta.token_ids:
                    yield meta.request_id
                    continue

                # for chunked prefill
                # record the number of saved blocks of previous chunk
                maybe_saved_blocks = self.request_saved_blocks.get(meta.request_id, 0)
                blocks_to_save = (
                        len(meta.token_ids) // self.block_size - maybe_saved_blocks
                )

                if blocks_to_save == 0:
                    self.request_saved_blocks.pop(meta.request_id, None)
                    yield meta.request_id
                    continue
                if meta.is_last_chunk:
                    requests_to_yield.add(meta.request_id)

                self.request_saved_blocks[meta.request_id] = (
                        maybe_saved_blocks + blocks_to_save
                )
                meta_hash_values = meta.block_hashes[
                    maybe_saved_blocks : maybe_saved_blocks + blocks_to_save]
                meta_block_ids_to_save = meta.block_ids[
                    maybe_saved_blocks : maybe_saved_blocks + blocks_to_save]

                hash_values.extend(meta_hash_values)
                block_ids_to_save.extend(meta_block_ids_to_save)

            await asyncio.to_thread(event.synchronize)
            self.event_pool.put_event(event)

            meta_futures = [loop.create_future() for _ in range(len(block_ids_to_save))]
            self.store.batch_save_kv_async(block_ids_to_save, hash_values,
                                           loop, meta_futures)

            results = await asyncio.gather(*meta_futures)  # noqa: F841
            for request_id in requests_to_yield:
                self.request_saved_blocks.pop(request_id, None)
                yield request_id

        return _save_kv_layer(event)

    def get_operations(self, req: Request) -> tuple[int, int]:
        return int(should_load(req)), int(should_save(req))

    async def async_get_num_new_matched_tokens(
        self,
        req: "Request",
        num_computed_tokens: int,
    ) -> int:
        if (not should_load(req) or
                req.num_prompt_tokens - num_computed_tokens < self.block_size):
            return 0

        num_computed_blocks = num_computed_tokens // self.block_size
        assert num_computed_tokens % self.block_size == 0
        num_uncomputed_blocks = ((req.num_tokens - num_computed_tokens)
                                 // self.block_size)
        uncomputed_blocks_hashes = [h.hex() for h in req.block_hashes[
            num_computed_blocks : num_computed_blocks + num_uncomputed_blocks]]

        keys = []
        for hash in uncomputed_blocks_hashes:
            for i in range(self.tp_size):
                keys.append(hash + "_" +str(i))

        ret = self.store.batch_is_exist(keys)

        new_matched_tokens = 0
        for i in range(0, len(ret), self.tp_size):
            all_tp_ret = ret[i : i + self.tp_size]
            if not all(x == 1 for x in all_tp_ret):
                break
            new_matched_tokens += self.block_size

        return new_matched_tokens

    async def async_update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens <= 0:
            return None

        block_num = cdiv(
            request.num_prompt_tokens - 1, self.vllm_config.cache_config.block_size
        )
        block_ids = blocks.get_block_ids()[0][:block_num]
        assert len(block_ids) == block_num
        assert request.block_hashes is not None
        # Request only stores hashes for full blocks; may be fewer than
        # block_num when prompt doesn't fill the last block.
        block_hashes = [h.hex() for h in request.block_hashes[:block_num]]

        num_computed_blocks = request.num_computed_tokens // self.block_size
        num_matched_blocks = num_external_tokens // self.block_size
        block_ids_to_load = block_ids[
            num_computed_blocks : num_computed_blocks + num_matched_blocks]
        block_hashes_to_load = block_hashes[
            num_computed_blocks : num_computed_blocks + num_matched_blocks]
        self.metas_to_recv.append(
            ReqMeta(
                request.request_id,
                False,
                request.prompt_token_ids,
                block_ids_to_load,
                block_hashes_to_load,
                request.num_computed_tokens,
            )
        )

    def build_backend_meta(self, scheduler_output: SchedulerOutput) -> BackendMeta:
        assert isinstance(scheduler_output, HCSchedOutput)
        from .utils import kill_me_if_exception

        @kill_me_if_exception
        async def abort_save(reqs: list[Request]):
            await asyncio.sleep(envs.VLLM_KVS_IO_TIMEOUT_SECONDS + 1)

            tasks = []
            for req in reqs:
                for rank in range(
                    self.vllm_config.parallel_config.tensor_parallel_size
                ):
                    tasks.append(
                        hybridsched._do_save_done(
                            rank, ioret=IoRet(req.request_id)
                        )
                    )

            await asyncio.gather(*tasks)

        from .engine_proxy import get_hybrid_sched_loop

        if scheduler_output.hc_aborted_save:
            logger.debug(
                f"register abort save cb for "
                f"{[req.request_id for req in scheduler_output.hc_aborted_save]}"
            )
            asyncio.run_coroutine_threadsafe(
                abort_save(scheduler_output.hc_aborted_save), get_hybrid_sched_loop()
            )

        meta_load = self._build_load()
        meta_save = self._build_save(scheduler_output)
        meta = meta_load + meta_save
        logger.debug(f"build meta: {meta}")
        return meta

    def _build_load(self) -> MooncakeKVSBackendMetadata:
        meta = MooncakeKVSBackendMetadata()
        while self.metas_to_recv:
            req_meta = self.metas_to_recv.popleft()
            meta.add(req_meta)
        return meta

    def _build_save_meta_for_request(
        self, request_id: str, num_scheduled_tokens: int
    ) -> ReqMeta:
        request = self.get_request(request_id)
        assert request is not None

        # if request.num_computed_tokens == request.num_prompt_tokens - 1:
        #     return ReqMeta(request_id, True, [], [], [],
        #                    0, True)

        num_computed_token_ids = request.num_computed_tokens + num_scheduled_tokens
        num_tokens_to_save = min(num_computed_token_ids, request.num_prompt_tokens)
        token_ids_to_save = request.prompt_token_ids[:num_tokens_to_save]

        num_populated_blocks = cdiv(
            num_tokens_to_save, self.vllm_config.cache_config.block_size
        )
        populated_block_ids = sched_get_kvblk_ids(
            request_id,
            self._gamma,
            self._enable_prefix_caching,
        )[:num_populated_blocks]

        assert request.block_hashes is not None
        block_hashes = [
            h.hex() for h in request.block_hashes[:num_populated_blocks]
        ]

        udc_ttl = (
            300
            if (request.cache_control_params is not None and "NOCPFS" not in request_id)
            else 0
        )

        return ReqMeta(
            request_id,
            True,
            token_ids_to_save,
            populated_block_ids,
            block_hashes,
            0,
            num_computed_token_ids >= request.num_prompt_tokens,
            udc_ttl=udc_ttl,
        )

    def _build_save(
        self, scheduler_output: SchedulerOutput
    ) -> MooncakeKVSBackendMetadata:
        meta = MooncakeKVSBackendMetadata()

        for req in scheduler_output.scheduled_new_reqs:
            request_id = req.req_id
            request = self.get_request(request_id)
            assert request is not None
            if should_save(request):
                meta.add(
                    self._build_save_meta_for_request(
                        request_id, scheduler_output.num_scheduled_tokens[request_id]
                    )
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for request_id in cached_reqs.req_ids:
            if "NOSAVE" in request_id:
                continue
            request = self.get_request(request_id)
            assert request is not None
            if should_save(request):
                meta.add(
                    self._build_save_meta_for_request(
                        request_id, scheduler_output.num_scheduled_tokens[request_id]
                    )
                )

        return meta


def should_save(req: Request) -> bool:
    if DISABLE_SAVE:
        return False
    if req.num_prompt_tokens <= envs.VLLM_KVS_ON_MIN_LENGTH + 1:
        return False
    return not req.prefill_done


def should_load(req: Request) -> bool:
    if DISABLE_LOAD:
        return False
    return not (req.num_prompt_tokens <= envs.VLLM_KVS_ON_MIN_LENGTH + 1)
