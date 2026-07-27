# The interaction between hybridconnector and engine is conducted
# exclusively through the interfaces in engine_proxy.
# It is expected that BladeLLM/SGLang only needs to implement the interfaces
# in this file to interface with the hybridconnector.

import functools
import os
from collections import defaultdict, deque
from math import lcm
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import vllm.envs as envs

# ServingArgs in BladeLLM
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler

# EngineCoreRequest is the serializable form of the request
# ServerRequest in BladeLLM
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    Qwen3NextIndexerAttentionSpec,
    UniformTypeKVCacheSpecs,
)

# A request corresponds to a Request object, which persists throughout
# the entire lifecycle of the request.
#
# HybridConnector does not know the implementation details of Request;
# it interacts with Request only through these interfaces.
#
# PagedRequestState in BladeLLM
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from .utils import RpcServer, start_asyncio_thread

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCoreProc

_g_core: Optional["EngineCoreProc"] = None

def get_logger(name: str):
    return init_logger(name)


def _sched() -> V1Scheduler:
    assert _g_core is not None
    sched = _g_core.scheduler
    assert isinstance(sched, V1Scheduler)
    return sched


@functools.cache
def is_kvt_consumer_backend(self: V1Scheduler) -> bool:
    kv_cfg = self.vllm_config.kv_transfer_config
    if kv_cfg is None or kv_cfg.kv_role != "kv_consumer":
        return False
    return kv_cfg.get_from_extra_config("backend", None) == "kvt"


@functools.cache
def use_kvt_decode_prefix_cache(self: V1Scheduler) -> bool:
    return (
        self.kv_cache_manager.enable_caching
        and self.vllm_config.model_config.is_hybrid
        and self.kv_cache_manager.has_mamba_layers
        and isinstance(
            self.kv_cache_manager.coordinator, HybridKVCacheCoordinator
        )
    )


def _get_computed_blocks_for_kvt(
    self: V1Scheduler,
    req: Request,
    save: bool,
    is_kvt_consumer: bool,
) -> tuple[KVCacheBlocks, int]:
    coordinator = self.kv_cache_manager.coordinator
    if not is_kvt_consumer or save:
        return self.kv_cache_manager.get_computed_blocks(req)
    assert isinstance(coordinator, HybridKVCacheCoordinator)

    computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
        req.block_hashes, req.num_tokens - 1
    )
    group_specs = self.kv_cache_manager.kv_cache_config.kv_cache_groups

    def _token_cache_block_size(spec: FullAttentionSpec) -> int:
        block_size = spec.block_size
        if not isinstance(
            spec, (MLAAttentionSpec, Qwen3NextIndexerAttentionSpec)
        ):
            block_size *= coordinator.dcp_world_size
        return block_size

    # KVT's seen_tokens is used by attention and indexer parsers.
    # GDN/Mamba state is not part of this boundary; it will be
    # always be transferred at request completion
    token_group_hits: list[int] = []
    token_group_block_sizes: list[int] = []
    for i, group in enumerate(group_specs):
        spec = group.kv_cache_spec
        if isinstance(spec, FullAttentionSpec):
            token_group_hits.append(per_group_hits[i])
            token_group_block_sizes.append(_token_cache_block_size(spec))

    if token_group_hits:
        # KVT can only use FullAttentionSpec's hits to determine the seen_tokens
        seen_tokens = min(token_group_hits)
        token_group_alignment = lcm(*token_group_block_sizes)
        seen_tokens = seen_tokens // token_group_alignment * token_group_alignment
    else:
        seen_tokens = 0

    filtered: list[list[Any]] = []
    for i, group_blocks in enumerate(computed):
        spec = group_specs[i].kv_cache_spec
        if seen_tokens > 0 and isinstance(spec, FullAttentionSpec):
            block_size = _token_cache_block_size(spec)
            filtered.append(list(group_blocks[:seen_tokens // block_size]))
        else:
            filtered.append([])

    if self.kv_cache_manager.log_stats:
        assert self.kv_cache_manager.prefix_cache_stats is not None
        self.kv_cache_manager.prefix_cache_stats.record(
            num_tokens=req.num_tokens,
            num_hits=seen_tokens,
            preempted=req.num_preemptions > 0,
        )

    return self.kv_cache_manager.create_kv_cache_blocks(tuple(filtered)), seen_tokens


def sched_allocate_slots(req: Request, load: bool, save: bool,
                         prealloc: int = 0, gamma: int = 0) -> Optional[KVCacheBlocks]:
    self = _sched()
    is_kvt_consumer = load and is_kvt_consumer_backend(self)
    use_decode_prefix_cache = (
        is_kvt_consumer and use_kvt_decode_prefix_cache(self)
    )
    new_computed_blocks, num_new_local_computed_tokens = \
        _get_computed_blocks_for_kvt(self, req, save, use_decode_prefix_cache)

    num_computed_tokens = num_new_local_computed_tokens
    num_new_tokens = req.num_tokens - num_computed_tokens

    if save:
        num_new_tokens += gamma + 1

    # Blocks filled by hybrid-connector load must not also be zeroed. With
    # bypass enabled, a KVS/KVT load can start before the next SchedulerOutput
    # drains pending zero IDs, so zeroing can race with or wipe the loaded KV.
    # For KVT consumer, skip zeroing at allocation time (coarse-grained).
    # For KVS-like backends, allocation-time skip is not possible since block
    # IDs are determined later; they call sched_discard_zero_block_ids() instead
    # (fine-grained). P-side save/prefill blocks still need normal zeroing.
    skip_zero_for_kvt_recv = is_kvt_consumer
    skip_new_block_zeroing = (
        skip_zero_for_kvt_recv or not envs.VLLM_ZERO_HYBRID_KV_CACHE)
    new_blocks = self.kv_cache_manager.allocate_slots(
        req,
        num_new_tokens + prealloc,
        num_new_local_computed_tokens,
        new_computed_blocks,
        delay_cache_blocks=True,
        skip_new_block_zeroing=skip_new_block_zeroing)
    if new_blocks is None:
        return None

    req.num_computed_tokens = num_new_local_computed_tokens
    req.skip_reading_prefix_cache = True
    return new_computed_blocks + new_blocks


def sched_discard_zero_block_ids(block_ids: Iterable[int]) -> None:
    """Remove hybrid-connector load targets from pending KV zeroing."""
    self = _sched()
    self.kv_cache_manager.discard_new_block_ids(block_ids)


def sched_cache_blocks(req: Request, num_computed_tokens: int):
    """Cache blocks for the request after KV transfer completes."""
    self = _sched()
    self.kv_cache_manager.cache_blocks(req, num_computed_tokens)


def _blk_check(oblks, nblks):
    if len(oblks) <= 0:
        return True
    assert oblks[0] is not nblks[0]
    if len(oblks[0]) <= 0:
        return True
    assert oblks[0][0] is nblks[0][0]
    return True


def sched_get_blocks(reqid: str) -> KVCacheBlocks:
    self = _sched()
    oblks = self.kv_cache_manager.coordinator.get_blocks(reqid)
    nblks = tuple(blk[:] for blk in oblks)
    assert _blk_check(oblks, nblks)
    return KVCacheBlocks(nblks)


# sched_release_blocks
def sched_free_blocks(blks: KVCacheBlocks):
    self = _sched()
    # SingleTypeKVCacheManager.free
    for blk in blks.blocks:
        blk = reversed(blk)
        self.kv_cache_manager.block_pool.free_blocks(blk)
    return


def sched_acquire_blocks(blks: KVCacheBlocks):
    for blk in blks.blocks:
        for block in blk:
            block.ref_cnt += 1
    return

def sched_finish_req(request_ids: Union[str, Iterable[str]]):
    self = _sched()
    self.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)


def sched_get_finished_req_ids() -> set[str]:
    """Get finished request ids from scheduler.

    This is needed in bypass loops where SchedulerOutput.finished_req_ids
    may be empty but scheduler.finished_req_ids contains requests that
    need to be processed.
    """
    return _sched().finished_req_ids


# get_request() should receive a parameter similar to a scheduler,
# but considering that mainstream LLM engines typically have only
# one engine per process, a global variable is used here instead.
def sched_get_req(reqid: str) -> Optional[Request]:
    return _sched().requests.get(reqid, None)


def sched_get_kvblk_ids(
    reqid: str,
) -> list[list[int]]:
    """Get KV block IDs for a request.
    Args:
        reqid: Request ID
    Returns:
        List of block ID lists for each layer group
    """
    sched = _sched()
    blk_ids = sched.kv_cache_manager.get_block_ids(reqid)
    return blk_ids


def sched_add_req(req: Request):
    sched = _sched()
    sched.add_request(req)
    return


def req2corereq(req: Request) -> EngineCoreRequest:
    # just make mypy happy~
    if req.sampling_params and req.sampling_params.extra_args is not None:
        req.sampling_params.extra_args["kv_transfer_params"] = None
    return EngineCoreRequest(
        request_id=req.request_id,
        prompt_token_ids=req.prompt_token_ids,
        mm_features=req.mm_features,
        sampling_params=req.sampling_params,
        pooling_params=req.pooling_params,
        eos_token_id=req.eos_token_id,
        arrival_time=req.arrival_time,
        lora_request=req.lora_request,
        cache_salt=req.cache_salt,
        data_parallel_rank=None,
        prompt_embeds=req.prompt_embeds,
        priority=req.priority,
        trace_headers=req.trace_headers,
        queue_server_address=req.queue_server_address,
    )


def core_abort_req(reqid: str, reason: str, output: bool):
    self = _sched()
    kvconn = self.get_kv_connector()
    assert kvconn is not None
    kvconn.on_abort_req(reqid, reason, output, iscore=False)
    return


def get_p_node_pop_len(vllm_config: VllmConfig) -> int:
    pop_len = 1
    # We enable the logic only for hybrid models at current time, as it breaks
    # normal P/D for other models, both offline and in dual-request mode.
    if (vllm_config.model_config.is_hybrid
        and vllm_config.speculative_config
        and vllm_config.speculative_config.num_speculative_tokens
    ):
        pop_len += vllm_config.speculative_config.num_speculative_tokens
    return pop_len


def core_add_req(request: EngineCoreRequest):
    assert _g_core is not None
    request = _g_core.preprocess_add_request(request)
    _g_core.input_queue.put_nowait((EngineCoreRequestType.ADD, request))
    return


def get_param(req: Request, key: str, defval: Any = None):
    if req.kv_transfer_params is None:
        return defval
    return req.kv_transfer_params.get(key, defval)


def set_param(req: Request, key: str, val: Any):
    if req.kv_transfer_params is None:
        req.kv_transfer_params = dict()
    req.kv_transfer_params[key] = val
    return


def pop_param(req: Request, key: str, defval: Any = None):
    if req.kv_transfer_params is None:
        return defval
    return req.kv_transfer_params.pop(key, defval)


def update_params(req: Request, newval: dict):
    if req.kv_transfer_params is None:
        req.kv_transfer_params = newval
    req.kv_transfer_params.update(newval)
    return


def core_get_param(req: EngineCoreRequest, key: str, defval: Any = None):
    if req.sampling_params is None:
        return defval
    args = req.sampling_params.extra_args
    if args is None:
        return defval
    param = args.get("kv_transfer_params", None)
    if param is None:
        return defval
    return param.get(key, defval)


def core_update_params(req: EngineCoreRequest, newval: dict):
    if req.sampling_params is None:
        req.sampling_params = SamplingParams()
    if req.sampling_params.extra_args is None:
        req.sampling_params.extra_args = dict()
    args = req.sampling_params.extra_args
    param = args.get("kv_transfer_params", None)
    if param is None:
        args["kv_transfer_params"] = newval
    else:
        param.update(newval)
    return


_g_sched_loop: Optional[Any] = None
_g_worker_loop: Optional[Any] = None


# It returns the loop for use by the hybrid scheduler;
# it might return the core loop in bladellm.
def get_hybrid_sched_loop():
    assert _g_sched_loop is not None
    return _g_sched_loop


def get_hybrid_worker_loop():
    assert _g_worker_loop is not None
    return _g_worker_loop


_g_sched_rpc_serv: Optional[RpcServer] = None


def sched_rpc_server() -> RpcServer:
    # Let the engine proxy be responsible for creating the RPC server,
    # so that the connector/backend are users rather than creators.
    assert _g_sched_rpc_serv is not None
    return _g_sched_rpc_serv


def sched_rpc_server_port(cfg: VllmConfig) -> int:
    if (envs.VLLM_DP_MASTER_PORT > 0
            and (os.getenv("AQUILA_RPC_PROTOCOL") or os.getenv("MASTER_ADDRESS"))):
        dprank = cfg.parallel_config.data_parallel_rank
        return envs.VLLM_DP_MASTER_PORT + 10 + dprank + 256
    rpc_port = cfg.kv_transfer_config.kv_connector_extra_config.get("rpc_port", None)
    if rpc_port is not None:
        return rpc_port
    assert cfg.kv_transfer_config.engine_available_port > 0
    return cfg.kv_transfer_config.engine_available_port


def scheduler_rpc_host(cfg: VllmConfig) -> str:
    if cfg.parallel_config.tensor_parallel_size > cuda_device_count_stateless():
        master_addr = os.getenv("MASTER_ADDRESS")
        assert master_addr
        return master_addr
    return "127.0.0.1"


def use_mla() -> bool:
    import vllm.envs as envs

    attn_backend = envs.VLLM_ATTENTION_BACKEND
    return attn_backend and "mla" in attn_backend.lower()

def use_sparse_mla() -> bool:
    import vllm.envs as envs

    attn_backend = envs.VLLM_ATTENTION_BACKEND
    return use_mla() and "sparse" in attn_backend.lower()

def use_flashinfer() -> bool:
    import vllm.envs as envs

    attn_backend = envs.VLLM_ATTENTION_BACKEND
    use_flashinfer = attn_backend and "flashinfer" in attn_backend.lower()
    if use_flashinfer:
        from vllm.platforms import current_platform
        capability = current_platform.get_device_capability()
        if capability is None or capability.major not in [10, 12]:
            cap_str = (
                f"SM{capability.major}.{capability.minor}"
                if capability is not None else "unknown"
            )
            raise ValueError(
                "FlashInfer KVT path requires Blackwell (SM 10.x) or Luna (SM 12.x); "
                f"detected device capability = {cap_str}. "
                f"Set VLLM_ATTENTION_BACKEND to a non-FlashInfer backend "
                f"(e.g. FLASH_ATTN) to run on this device."
            )
    return use_flashinfer

def _flashinfer_kvt_cache_shape(is_hybrid: bool) -> str:
    """
    FlashInferAttentionBackend.get_required_kv_cache_layout() forces HND only
    on Blackwell (SM 10.x, SM 12.x) keeps the default NHD layout.

    Hybrid:
      SM10 HND → QWEN3_NEXT_FLASHINFER_CACHE_SHAPE (=7)
      SM12 NHD → QWEN3_NEXT_FLASH_CACHE_SHAPE (=3)  # same physical layout
    Non-hybrid:
      SM10 HND → FLASHINFER_CACHE_SHAPE (=5)
      SM12 NHD → unsupported (blade-kvt has no non-hybrid NHD FlashInfer shape)
    """
    from vllm.platforms import current_platform
    capability = current_platform.get_device_capability()
    assert capability is not None
    if capability.major == 10:
        return "7" if is_hybrid else "5"
    if capability.major == 12:
        if is_hybrid:
            return "3"
        raise ValueError(
            "Non-hybrid FlashInfer KVT only supports HND, "
            f"detected device capability = SM{capability.major}.{capability.minor}."
        )
    raise ValueError(
        f"FlashInfer KVT path requires Blackwell (SM 10.x or SM 12.x);"
        f"detected device capability = "
        f"SM{capability.major}.{capability.minor}. "
        f"Set VLLM_ATTENTION_BACKEND to a non-FlashInfer backend "
        f"(e.g. FLASH_ATTN) to run on this device."
    )

def use_turboquant(cfg: VllmConfig) -> bool:
    import vllm.envs as envs

    if envs.VLLM_FLASH_ATTN_USE_FAST_TURBOQUANT:
        return True
    if envs.VLLM_FLASH_ATTN_USE_FAKE_TURBOQUANT:
        return True
    cache_dtype = cfg.cache_config.cache_dtype
    return bool(cache_dtype and "tq" in cache_dtype.lower())

def kvt_protocol() -> str:
    import vllm.envs as envs

    return envs.VLLM_KV_TRANS_PROTOCOL

def group_layers_by_index(kv_caches: dict[str, Any]) -> dict[int, list[str]]:
    from vllm.model_executor.models.utils import extract_layer_index

    index2name = defaultdict[Any, list](list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    return index2name

def wakeup_core():
    # too hack...!
    # abort non-exist req is noop
    assert _g_core is not None
    t = EngineCoreRequestType.ABORT
    v = ["__FAKE_REQID_FOR_HYBRID_CONNECTOR__"]
    _g_core.input_queue.put_nowait((t, v))
    return


def core_init(core: "EngineCoreProc", cfg: VllmConfig):
    global _g_core
    global _g_sched_loop
    global _g_sched_rpc_serv

    assert _g_core is None
    _g_core = core

    assert _g_sched_loop is None
    _g_sched_loop = start_asyncio_thread("hybridsched")

    assert _g_sched_rpc_serv is None
    port = sched_rpc_server_port(cfg)
    _g_sched_rpc_serv = RpcServer(port, "schedrpcserver")
    _g_sched_rpc_serv.start(_g_sched_loop)
    return


def worker_init(vllm_config: VllmConfig, local_rank: int):
    global _g_worker_loop
    assert _g_worker_loop is None
    _g_worker_loop = start_asyncio_thread('hybridworker', vllm_config,
                                          local_rank)


def _is_statemodel_spec(spec: KVCacheSpec) -> bool:
    if isinstance(spec, MambaSpec):
        return True
    if isinstance(spec, UniformTypeKVCacheSpecs):
        for nest_spec in spec.kv_cache_specs.values():
            if _is_statemodel_spec(nest_spec):
                return True
    return False


# State space model
def is_statemodel() -> bool:
    kvconn: Optional[KVConnectorBase_V1] = None
    if has_kv_transfer_group():
        kvconn = get_kv_transfer_group()
    else:
        kvconn = _sched().get_kv_connector()
    assert isinstance(kvconn, KVConnectorBase_V1)

    kvcache_cfg = kvconn._kv_cache_config
    assert kvcache_cfg is not None
    for grp in kvcache_cfg.kv_cache_groups:
        if _is_statemodel_spec(grp.kv_cache_spec):
            return True
    return False
