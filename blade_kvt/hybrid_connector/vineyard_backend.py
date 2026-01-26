import asyncio
import atexit
import multiprocessing
import os
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Union

import aiohttp
import torch
import vineyard

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.splitwise.sidecar import SidecarClient, run_sidecar
from vllm.splitwise.splitwise import EXTRA_DATA_NBYTES
from vllm.utils import EventPool
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from . import BackendMeta, HybridBackend, IoRet
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    request_id: str
    # The block ids of the request in the GPU memory
    block_ids: list[int]
    # The block ids of the request in the vineyard shared memory
    v_block_ids: list[int]
    # Whether the request is a store request or load request
    is_store: bool

    @staticmethod
    def make_meta(
        request_id: str, block_ids: list[int], v_block_ids: list[int], is_store: bool
    ) -> "ReqMeta":
        assert len(block_ids) == len(v_block_ids), (
            "The length of block_ids and v_block_ids must be the same"
        )
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids,
            v_block_ids=v_block_ids,
            is_store=is_store,
        )


@dataclass
class VineyardBackendMetadata(BackendMeta):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        block_ids: list[int],
        v_block_ids: list[int],
        is_store: bool,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, block_ids, v_block_ids, is_store)
        )

    def add_request_raw(self, meta: ReqMeta) -> None:
        self.requests.append(meta)

    def __add__(self, other: "VineyardBackendMetadata") -> "VineyardBackendMetadata":
        if not isinstance(other, VineyardBackendMetadata):
            raise TypeError(f"Cannot add {type(other)} to VineyardBackendMetadata")
        new_meta = VineyardBackendMetadata()
        new_meta.requests = self.requests + other.requests
        return new_meta


class VineyardBackend(HybridBackend):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.kv_role == "kv_both", (
            "Vineyard connector only supports 'kv_both' kv_role"
        )

        self._num_layers = self._vllm_config.model_config.get_num_layers(
            self._vllm_config.parallel_config
        )
        if self._vllm_config.speculative_config is not None:
            assert self._vllm_config.speculative_config.method in (
                "deepseek_mtp",
                "eagle3",
                "eagle",
            ), "only support VineyardBackend for eagle/MTP with 1 draft layer"
            self._num_layers += 1

        engine_id = vllm_config.kv_transfer_config.engine_id
        self.splitwise = Splitwise(
            vllm_config=vllm_config,
            role=role,
            engine_id=engine_id,
            num_layers=self._num_layers,
        )

        # worker-side variables
        self._kv_caches: list[torch.Tensor] = []
        self._layer_name_to_index: dict[str, int] = {}

        # scheduler-side variables
        self._request_trackers: dict[str, RequestTracker] = {}
        self._metas_to_recv: deque[ReqMeta] = deque()

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for idx, (name, layer) in enumerate(kv_caches.items()):
            self._kv_caches.append(layer)
            self._layer_name_to_index[name] = extract_layer_index(name)

        assert len(self._kv_caches) == self._num_layers, (
            f"The number of kv caches:{len(self._kv_caches)} is not "
            f"equal to the number of layers:{self._num_layers}"
        )

    async def async_load_kv(self, m: BackendMeta) -> AsyncGenerator[IoRet, None]:
        assert self._role == KVConnectorRole.WORKER, "Only worker can load kv cache"
        metadata = m
        tasks = []
        assert isinstance(metadata, VineyardBackendMetadata)
        for request in metadata.requests:
            if request.is_store:
                continue
            task = asyncio.create_task(
                self.splitwise.async_load_kv(
                    request.request_id,
                    request.v_block_ids,
                    request.block_ids,
                    self._kv_caches,
                )
            )
            tasks.append(task)

        for completed_task in asyncio.as_completed(tasks):
            try:
                request_id = await completed_task
            except Exception as e:
                logger.error("Failed to load kv cache for KVRequest %s", e)
                raise e
            else:
                yield IoRet(reqid=request_id)

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> AsyncGenerator[str, None]:
        forward_stream = torch.cuda.current_stream()
        event = self.splitwise.event_pool.get_event()
        event.record(forward_stream)
        layer_idx = self._layer_name_to_index[layer_name]
        metadata = m

        async def _save_kv_layer() -> AsyncGenerator[str, None]:
            tasks = []
            assert isinstance(metadata, VineyardBackendMetadata)
            for request in metadata.requests:
                assert isinstance(request, ReqMeta)
                if not request.is_store:
                    continue
                task = asyncio.create_task(
                    self.splitwise.async_save_kv_layer(
                        event,
                        request.request_id,
                        request.v_block_ids,
                        request.block_ids,
                        kv_layer,
                        layer_idx,
                    )
                )
                tasks.append(task)

            for completed_task in asyncio.as_completed(tasks):
                try:
                    request_id = await completed_task
                except Exception as e:
                    logger.error("Failed to save kv layer for KVRequest %s", e)
                    raise e
                else:
                    yield request_id

        return _save_kv_layer()

    # ==============================
    # Scheduler-side methods
    #
    # kv_transfer_params:
    #   - "do_remote_prefill": bool
    #   - "remote_host": str
    #   - "remote_port": int
    # ==============================

    def get_operations(self, req: Request) -> tuple[int, int]:
        params = req.kv_transfer_params
        if params is None:
            return 0, 0
        remote_prefill = params.get("do_remote_prefill", False)
        return int(remote_prefill), int(not remote_prefill)

    async def async_get_num_new_matched_tokens(
        self,
        req: "Request",
        num_computed_tokens: int,
    ) -> int:
        request = req
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            new_matched_tokens = request.num_tokens - num_computed_tokens - 1
            logger.debug(
                "D: KVRequest %s, request.num_tokens %s, "
                "num_computed_tokens: %s, new_matched_tokens: %s",
                request.request_id,
                request.num_tokens,
                num_computed_tokens,
                new_matched_tokens,
            )
            return new_matched_tokens
        else:
            return 0

    async def async_update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        if params is None or params.get("processed"):
            return
        params["processed"] = True  # only process the request once
        if params.get("do_remote_prefill"):
            block_num = cdiv(
                request.num_prompt_tokens - 1, self._vllm_config.cache_config.block_size
            )
            block_ids = blocks.get_block_ids()[0][:block_num]
            assert len(block_ids) == block_num, (
                f"Expected {block_num} block IDs, but got {len(block_ids)}, "
                "maybe not enough blocks allocated."
            )
            try:
                self.splitwise.allocate_blocks(request.request_id, block_num)
            except ValueError as e:
                logger.exception(
                    "Failed to allocate blocks for KVRequest %s", request.request_id
                )
                raise e
            logger.debug(
                "D: KVRequest %s, num_external_tokens: %s, block_size: %s, "
                "allocated %s blocks",
                request.request_id,
                num_external_tokens,
                self._vllm_config.cache_config.block_size,
                block_num,
            )
            req_id = request.request_id
            assert request.kv_transfer_params is not None
            params = request.kv_transfer_params
            prefill_v6d_address = params["remote_host"] + ":" + params["remote_port"]
            v_block_ids = self.splitwise._req_to_block_ids[req_id]
            logger.debug(
                "DEBUG NEW: %s, %s, 0, %s, %s, [], %s, %s, False",
                request.request_id,
                v_block_ids,
                prefill_v6d_address,
                req_id,
                block_ids,
                v_block_ids,
            )
            self.splitwise._sidecar_client.start_recv_request(
                request.request_id, v_block_ids, 0, prefill_v6d_address
            )
            logger.debug(
                "D: KVRequest %s, start_recv_request from %s, with v_block_ids: %s",
                request.request_id,
                prefill_v6d_address,
                v_block_ids,
            )
            self._metas_to_recv.append(
                ReqMeta.make_meta(request.request_id, block_ids, v_block_ids, False)
            )

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        meta_recving = self._build_connector_meta_recving(sout)
        meta_sending = self._build_connector_meta_sending(sout)
        assert isinstance(meta_recving, VineyardBackendMetadata)
        assert isinstance(meta_sending, VineyardBackendMetadata)
        return meta_recving + meta_sending

    def _build_connector_meta_recving(
        self, scheduler_output: SchedulerOutput
    ) -> BackendMeta:
        meta = VineyardBackendMetadata()
        while self._metas_to_recv:
            req_meta = self._metas_to_recv.popleft()
            meta.add_request_raw(req_meta)
        return meta

    def _build_connector_meta_sending(
        self, scheduler_output: SchedulerOutput
    ) -> BackendMeta:
        meta = VineyardBackendMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            request = self.get_request(new_req.req_id)
            assert request is not None
            params = request.kv_transfer_params
            if params is None:
                continue

            num_tokens_to_compute = (
                new_req.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[new_req.req_id]
            )
            request_tracker = RequestTracker.from_new_request(
                new_req, num_tokens_to_compute
            )
            self._request_trackers[new_req.req_id] = request_tracker

            if not request_tracker.all_scheduled:
                logger.debug(
                    "P/D: KVRequest %s, not enough tokens scheduled, "
                    "expect %s, but got %s",
                    request_tracker.request_id,
                    request_tracker.num_tokens,
                    len(request_tracker.token_ids),
                )
                continue
            self._request_trackers.pop(new_req.req_id, None)
            if params.get("do_remote_decode"):
                prompt_token_ids = request_tracker.token_ids[:-1]
                num_blocks = cdiv(
                    len(prompt_token_ids), self._vllm_config.cache_config.block_size
                )
                block_ids = request_tracker.allocated_block_ids[:num_blocks]
                request_id = new_req.req_id
                v6d_blocks = self.splitwise.allocate_blocks(request_id, num_blocks)
                v_block_ids = [blk.block_id for blk in v6d_blocks]
                self.splitwise._sidecar_client.start_send_request(
                    request_id, v_block_ids, 0
                )
                self.splitwise._sidecar_client.send_logits(request_id)
                meta.add_request(request_id, block_ids, v_block_ids, True)
                logger.debug(
                    "P: KVRequest %s, Allocated %s blocks, with v_block_ids: %s",
                    request_id,
                    num_blocks,
                    v_block_ids,
                )

        req_data = scheduler_output.scheduled_cached_reqs
        for i, request_id in enumerate(req_data.req_ids):
            request = self.get_request(request_id)
            assert request is not None
            params = request.kv_transfer_params
            if params is None or request_id not in self._request_trackers:
                continue
            request_tracker = self._request_trackers[request_id]
            request_tracker.update(req_data.new_token_ids[i], req_data.new_block_ids[i])
            if not request_tracker.all_scheduled:
                logger.debug(
                    "P/D: KVRequest %s, not enough tokens scheduled, "
                    "expect %s, but got %s",
                    request_tracker.request_id,
                    request_tracker.num_tokens,
                    len(request_tracker.token_ids),
                )
                continue
            self._request_trackers.pop(request_id, None)
            if params.get("do_remote_decode"):
                prompt_token_ids = request_tracker.token_ids[:-1]
                num_blocks = cdiv(
                    len(prompt_token_ids), self._vllm_config.cache_config.block_size
                )
                block_ids = request_tracker.allocated_block_ids[:num_blocks]
                v6d_blocks = self.splitwise.allocate_blocks(request_id, num_blocks)
                v_block_ids = [blk.block_id for blk in v6d_blocks]
                self.splitwise._sidecar_client.start_send_request(
                    request_id, v_block_ids, 0
                )
                self.splitwise._sidecar_client.send_logits(request_id)
                meta.add_request(request_id, block_ids, v_block_ids, True)
                logger.debug(
                    "P: cached KVRequest %s, Allocated %s blocks, with v_block_ids: %s",
                    request_id,
                    num_blocks,
                    v_block_ids,
                )
        return meta

    async def async_cleanup(self, req: Request):
        """
        Cleanup the request after kv cache is sent or received.
        """
        params = req.kv_transfer_params
        if params is None:
            return
        request_id = req.request_id
        if params.get("do_remote_prefill"):
            self.splitwise._sidecar_client.release_recved_request(request_id)
            self.splitwise.free_blocks(request_id)
            logger.debug("D: KVRequest %s, finished recving kv cache", request_id)
        elif params.get("do_remote_decode"):
            logger.debug("P: KVRequest %s, finished sending kv cache", request_id)
            # self.splitwise._sidecar_client.release_sent_request(request_id)
            self.splitwise.free_blocks(request_id)

    def _get_kv_cache_spec(self) -> KVCacheSpec:
        model_config = self._vllm_config.model_config
        parallel_config = self._vllm_config.parallel_config
        cache_config = self._vllm_config.cache_config
        # num_layers = model_config.get_num_layers(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        return AttentionSpec(
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=cache_config.block_size,
            dtype=kv_dtype,
            use_mla=model_config.use_mla,
        )

    @classmethod
    async def issue_prefill(
        cls,
        request_id: str,
        prefill_endpoint: str,
        model: str,
        prompt: Union[list[int], list[list[int]], str, list[str]],
    ) -> Any:
        async with aiohttp.ClientSession() as session:
            url = prefill_endpoint
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 1,
                "request_id": request_id,
                "prefill_only": True,
            }
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error("Failed to send prefill request: %s", response.status)
                    raise RuntimeError(
                        f"HTTP request failed with status {response.status}"
                    )
                return await response.json()

    @classmethod
    async def maybe_split_request(cls, request: Any, fallback: bool = False):
        # This method is used to demonstrate how to split the request
        # In production, a proxy API should be used to implement similar logic.
        from vllm.entrypoints.openai.protocol import CompletionRequest

        if isinstance(request, CompletionRequest):
            if request.prefill_only:
                # prefiller
                kv_transfer_params = {"do_remote_decode": True}
                request.kv_transfer_params = kv_transfer_params
            elif (
                request.prefill_v6d_address is not None
                and request.prefill_endpoint is not None
            ):
                assert request.request_id is not None, "request_id is None"
                assert request.model is not None, "model is None"
                if ":" not in request.prefill_v6d_address:
                    raise ValueError(
                        f"Invalid prefill_v6d_address: {request.prefill_v6d_address}"
                    )
                remote_host, remote_port = request.prefill_v6d_address.split(":", 1)
                kv_transfer_params = {
                    "do_remote_prefill": True,
                    "remote_host": remote_host,
                    "remote_port": remote_port,
                }
                request.kv_transfer_params = kv_transfer_params
                try:
                    resp = await cls.issue_prefill(
                        request.request_id,
                        request.prefill_endpoint,
                        request.model,
                        request.prompt,
                    )
                    logger.debug(
                        "Prefill request sent successfully. response: %s", resp
                    )
                except Exception as e:
                    logger.error("Prefill request failed: %s", e)
                    if not fallback:
                        raise e
                    logger.error("Falling back to local generation.")
                    request.kv_transfer_params = None
                    request.prefill_v6d_address = None
                    request.prefill_endpoint = None
        else:
            logger.error(
                "Unsupported request type"
                "Only CompletionRequest is supported."
                "Falling back to local generation."
            )


@dataclass
class RequestTracker:
    request_id: str
    # The token ids that has been scheduled so far
    token_ids: list[int]
    # The block ids that has been allocated so far
    allocated_block_ids: list[int]
    # The nums of total tokens in the request
    num_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        allocated_block_ids = new_request.block_ids[0].copy()
        return RequestTracker(
            request_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=allocated_block_ids,
            num_tokens=len(new_request.prompt_token_ids),
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: tuple[list[int], ...],
    ) -> None:
        self.token_ids.extend(new_token_ids)
        new_block_ids = new_block_ids[0]  # type: ignore
        self.allocated_block_ids.extend(new_block_ids)  # type: ignore

    @property
    def all_scheduled(self) -> bool:
        return len(self.token_ids) == self.num_tokens


class BlockIdPool:
    def __init__(self, num_ids: int):
        self.num_ids = num_ids
        self.available_ranges = [(0, num_ids - 1)]

    def allocate_contiguous_block_ids(self, num: int) -> list[int]:
        if num <= 0:
            return []
        for i, (start, end) in enumerate(self.available_ranges):
            range_size = end - start + 1
            if range_size >= num:
                allocated_ids = list(range(start, start + num))
                if range_size == num:
                    self.available_ranges.pop(i)
                else:
                    self.available_ranges[i] = (start + num, end)
                return allocated_ids
        # No contiguous block of sufficient size found
        raise ValueError(f"Cannot get {num} free blocks from the pool")

    def free_block_ids(self, block_ids: list[int]):
        if not block_ids:
            return

        for i in range(1, len(block_ids)):
            assert block_ids[i] - block_ids[i - 1] == 1, (
                "The block_ids items need to be contiguous integers!"
            )

        self._merge_range(block_ids[0], block_ids[-1])

    def _merge_range(self, start: int, end: int):
        """Helper method to merge a freed range back into available_ranges

        Ensures that available_ranges is always kept sorted by start position.
        """
        if start < 0 or end >= self.num_ids or start > end:
            raise ValueError(f"Invalid range: [{start}, {end}]")

        if not self.available_ranges:
            self.available_ranges.append((start, end))
            return

        insert_idx = self._find_insertion_point(start)

        new_ranges = []

        new_ranges.extend(self.available_ranges[:insert_idx])

        new_range = (start, end)

        if new_ranges and new_ranges[-1][1] + 1 >= start:
            new_ranges[-1] = (new_ranges[-1][0], max(new_ranges[-1][1], end))
        else:
            new_ranges.append(new_range)

        for i in range(insert_idx, len(self.available_ranges)):
            curr_range = self.available_ranges[i]

            if new_ranges[-1][1] + 1 >= curr_range[0]:
                new_ranges[-1] = (
                    new_ranges[-1][0],
                    max(new_ranges[-1][1], curr_range[1]),
                )
            else:
                new_ranges.append(curr_range)

        self.available_ranges = new_ranges

    def _find_insertion_point(self, start: int) -> int:
        """Binary search to find insertion point for a range starting at 'start'"""
        left, right = 0, len(self.available_ranges) - 1

        # Edge cases for quick checking
        if not self.available_ranges or start < self.available_ranges[0][0]:
            return 0
        if start > self.available_ranges[-1][0]:
            return len(self.available_ranges)

        while left <= right:
            mid = (left + right) // 2
            if self.available_ranges[mid][0] < start:
                left = mid + 1
            else:
                right = mid - 1

        return left


class Splitwise:
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        engine_id: str,
        num_layers=0,
    ):
        self._vllm_config = vllm_config
        self._role = role
        self._kv_caches: list[torch.Tensor] = []

        # copy engine
        self.device = torch.device(torch.cuda.current_device())
        self.swap_out_stream = torch.cuda.Stream()
        self.swap_in_stream = torch.cuda.Stream()
        self.event_pool = EventPool(256, self.device)

        # init parameters
        self._kv_cache_spec = self._calculate_kv_cache_spec()
        self._vocab_size = self._vllm_config.model_config.get_vocab_size()
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size

        if num_layers > 0:
            self._num_layers = num_layers
        else:
            self._num_layers = self._vllm_config.model_config.get_num_layers(
                self._vllm_config.parallel_config
            )
        self._logits_size = self._vocab_size + -(
            -EXTRA_DATA_NBYTES // self._kv_cache_spec.dtype.itemsize
        )

        self._request_next_layer: dict[str, int] = {}

        # init vineyard
        v6d_sock = vllm_config.kv_transfer_config.get_from_extra_config("sock", None)
        try:
            v6d_client = vineyard.connect(v6d_sock)
        except ConnectionError as e:
            logger.exception(
                "Failed to connect to vineyard at %s, role %s", v6d_sock, role
            )
            raise e
        fd, total_mem_size, offset = v6d_client.get_vineyard_mmap_fd()
        self._num_blocks = self._calculate_num_blocks(total_mem_size, offset)
        if role == KVConnectorRole.SCHEDULER:
            # init for scheduler
            self._req_to_block_ids: defaultdict[str, list[int]] = defaultdict(list)
            self._block_id_pool = BlockIdPool(self._num_blocks)
        else:
            self._rank = get_tp_group().rank
            # init for worker
            self._kv_caches = self._init_kv_caches(fd, total_mem_size, offset)

        # init sidecar
        self._engine_id = engine_id
        self._sidecar_client = self._init_sidecar()

    def _calculate_kv_cache_spec(self) -> AttentionSpec:
        model_config = self._vllm_config.model_config
        parallel_config = self._vllm_config.parallel_config
        cache_config = self._vllm_config.cache_config
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        return AttentionSpec(
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=cache_config.block_size,
            dtype=kv_dtype,
            use_mla=model_config.use_mla,
        )

    def _calculate_num_blocks(self, total_mem_size: int, offset: int) -> int:
        mem_size = total_mem_size
        reseved_logits_size = 1 * self._logits_size * self._kv_cache_spec.dtype.itemsize
        # use the whole memory of vineyard to store kv cache
        num_blocks = int(
            (mem_size - offset - reseved_logits_size)
            // (self._kv_cache_spec.page_size_bytes * self._tp_size * self._num_layers)
        )

        return num_blocks

    def _init_kv_caches(
        self, fd: int, total_mem_size: int, offset: int
    ) -> list[torch.Tensor]:
        kv_caches: list[torch.Tensor] = []
        coef = 1 if self._kv_cache_spec.use_mla else 2
        kv_caches_shape = [
            coef,
            self._num_blocks,
            self._kv_cache_spec.block_size,
            self._kv_cache_spec.num_kv_heads,
            self._kv_cache_spec.head_size,
        ]
        dtype_tensor = torch.empty(0, dtype=self._kv_cache_spec.dtype)
        ops.init_v6d_shared_memory(
            fd,
            total_mem_size,
            offset,
            self._tp_size,
            self._num_layers,
            kv_caches_shape,
            # reserve one logits to make v6d happy
            [1, self._logits_size],
            dtype_tensor,
            dtype_tensor,
        )
        logger.debug("V6D kv cache tensor shape: %s", kv_caches_shape)
        kv_cache_tensor = ops.get_v6d_kv_cache_tensor(self._rank)
        for i in range(self._num_layers):
            kv_caches.append(kv_cache_tensor[i])
        return kv_caches

    def _init_sidecar(self) -> SidecarClient:
        sidecar_ipc_path = f"ipc://{envs.VLLM_RPC_BASE_PATH}/sidecar_{self._engine_id}"
        if self._role == KVConnectorRole.SCHEDULER:
            coef = 1 if self._kv_cache_spec.use_mla else 2
            kv_caches_shape = [
                coef,
                self._num_blocks,
                self._kv_cache_spec.block_size,
                self._kv_cache_spec.num_kv_heads,
                self._kv_cache_spec.head_size,
            ]
            context = multiprocessing.get_context("spawn")
            sidecar_process = context.Process(
                target=run_sidecar,
                args=(
                    sidecar_ipc_path,
                    self._vllm_config.kv_transfer_config.get_from_extra_config(
                        "sock", None
                    ),
                    self._tp_size,
                    self._num_layers,
                    [self._num_layers] + kv_caches_shape,
                    self._kv_cache_spec.dtype.itemsize,
                    self._logits_size,
                    self._kv_cache_spec.dtype.itemsize,
                ),
            )
            sidecar_process.start()

            def _cleanup_ipc_path():
                socket_path = sidecar_ipc_path.replace("ipc://", "")
                if os.path.exists(socket_path):
                    os.remove(socket_path)

            atexit.register(_cleanup_ipc_path)
        sidecar = SidecarClient(sidecar_ipc_path)
        # sidecar.wait_for_ready()
        return sidecar

    def allocate_blocks(self, request_id: str, num_blocks: int) -> list[KVCacheBlock]:
        assert self._role == KVConnectorRole.SCHEDULER
        req_block_ids = self._req_to_block_ids[request_id]
        assert len(req_block_ids) == 0, (
            f"KVRequest {request_id} already has blocks allocated."
        )
        new_block_ids = self._block_id_pool.allocate_contiguous_block_ids(num_blocks)
        req_block_ids.extend(new_block_ids)
        return [KVCacheBlock(block_id=block_id) for block_id in new_block_ids]

    def free_blocks(self, request_id: str):
        assert self._role == KVConnectorRole.SCHEDULER
        req_blocks = self._req_to_block_ids[request_id]
        assert len(req_blocks) > 0, f"KVRequest {request_id} has no blocks allocated."
        self._block_id_pool.free_block_ids(req_blocks)
        del self._req_to_block_ids[request_id]

    async def async_load_kv(
        self,
        request_id: str,
        v_block_ids: list[int],
        block_ids: list[int],
        kv_caches: list[torch.Tensor],
    ) -> str:
        if len(v_block_ids) == 0:
            return request_id
        while True:
            recved = self._sidecar_client.recved_layer(
                request_id, self._rank, self._num_layers - 1
            )
            if recved:
                break
            if request_id in self._sidecar_client.exception_requests:
                logger.error(
                    "Failed to load kv cache for KVRequest %s, Forward anyway",
                    request_id,
                )
                self._sidecar_client.exception_requests.remove(request_id)
                break
                # raise RuntimeError(f"Failed to load kv cache for KVRequest {request_id}")  # noqa: E501
            await asyncio.sleep(0)
        src_block_ids = v_block_ids
        dst_block_ids = block_ids
        src_kv_caches = self._kv_caches
        dst_kv_caches = kv_caches
        event = self._sync_swap_blocks(
            src_kv_caches, dst_kv_caches, src_block_ids, dst_block_ids
        )
        while not event.query():
            await asyncio.sleep(0)
        self.event_pool.put_event(event)
        return request_id

    async def async_save_kv_layer(
        self,
        event,
        request_id: str,
        v_block_ids: list[int],
        block_ids: list[int],
        kv_caches_layer: torch.Tensor,
        layer_idx: int,
    ):
        if len(v_block_ids) == 0:
            return request_id

        if request_id not in self._request_next_layer:
            self._request_next_layer[request_id] = 0

        while self._request_next_layer[request_id] != layer_idx:
            await asyncio.sleep(0.001)

        src_block_ids = block_ids
        dst_block_ids = v_block_ids

        dst_kv_caches = self._kv_caches

        block_mapping = list(zip(src_block_ids, dst_block_ids))
        event.wait(self.swap_out_stream)
        with torch.cuda.stream(self.swap_out_stream):
            block_mapping_tensor = torch.tensor(block_mapping, device=self.device).view(
                -1, 2
            )
            if self._kv_cache_spec.use_mla:
                src_key_cache = kv_caches_layer
                dst_key_cache = dst_kv_caches[layer_idx][0]

                ops.swap_blocks(src_key_cache, dst_key_cache, block_mapping_tensor)
            else:
                src_key_cache = kv_caches_layer[0]
                dst_key_cache = dst_kv_caches[layer_idx][0]

                src_value_cache = kv_caches_layer[1]
                dst_value_cache = dst_kv_caches[layer_idx][1]

                ops.swap_blocks(src_key_cache, dst_key_cache, block_mapping_tensor)
                ops.swap_blocks(src_value_cache, dst_value_cache, block_mapping_tensor)
            event.record()
        while not event.query():
            await asyncio.sleep(0)
        self.event_pool.put_event(event)
        self._sidecar_client.send_layer([request_id], self._rank, layer_idx)
        if (
            self._rank == 0 and layer_idx == self._num_layers - 1
        ):  # last layer of rank 0
            while True:
                sent = self._sidecar_client.sent_request_and_release(request_id)
                if sent:
                    break
                if request_id in self._sidecar_client.exception_requests:
                    logger.error(
                        "Failed to save kv cache for KVRequest %s, skip", request_id
                    )
                    self._sidecar_client.exception_requests.remove(request_id)
                    break
                await asyncio.sleep(0)

        self._request_next_layer[request_id] = layer_idx + 1

        if layer_idx == self._num_layers - 1:
            del self._request_next_layer[request_id]

        return request_id

    def _sync_swap_blocks(
        self, src_kv_caches, dst_kv_caches, src_block_ids, dst_block_ids
    ):
        block_mapping = list(zip(src_block_ids, dst_block_ids))
        with torch.cuda.stream(self.swap_in_stream):
            block_mapping_tensor = torch.tensor(block_mapping, device=self.device).view(
                -1, 2
            )
            for layer_idx in range(self._num_layers):
                if self._kv_cache_spec.use_mla:
                    src_key_cache = src_kv_caches[layer_idx][0]
                    dst_key_cache = dst_kv_caches[layer_idx][0]

                    ops.swap_blocks(src_key_cache, dst_key_cache, block_mapping_tensor)
                else:
                    src_key_cache = src_kv_caches[layer_idx][0]
                    dst_key_cache = dst_kv_caches[layer_idx][0]

                    src_value_cache = src_kv_caches[layer_idx][1]
                    dst_value_cache = dst_kv_caches[layer_idx][1]

                    ops.swap_blocks(src_key_cache, dst_key_cache, block_mapping_tensor)
                    ops.swap_blocks(
                        src_value_cache, dst_value_cache, block_mapping_tensor
                    )
            event = self.event_pool.get_event()
            event.record()
            return event
