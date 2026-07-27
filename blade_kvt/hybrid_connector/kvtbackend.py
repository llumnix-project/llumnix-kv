from __future__ import annotations

import asyncio
import copy
import inspect
import os
import struct
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from functools import cache
from typing import Any, AsyncGenerator, List, Optional, Union

import msgspec
import torch

import vllm.envs as envs
from vllm.model_executor.models.registry import ModelRegistry
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.ops.turboquant import TurboQuantCache
from vllm.v1.hybrid_connector import HybridMetadata
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

from . import (
    BackendMeta,
    HCSchedOutput,
    HybridBackend,
    OperationPlan,
    mark_backend_save_done,
    req_aborted,
    scheduler_rpc_host,
    try_setup_save,
    wakeup_scheduler,
)

# yapf: disable
from .engine_proxy import (
    EngineCoreRequest,
    KVCacheBlocks,
    KVCacheConfig,
    KVConnectorRole,
    MsgpackDecoder,
    MsgpackEncoder,
    PlaceholderModule,
    Request,
    SchedulerOutput,
    VllmConfig,
    _flashinfer_kvt_cache_shape,
    core_abort_req,
    core_add_req,
    core_get_param,
    core_update_params,
    get_hybrid_sched_loop,
    get_hybrid_worker_loop,
    get_ip,
    get_logger,
    get_p_node_pop_len,
    get_param,
    get_tensor_model_parallel_rank,
    get_tp_group,
    group_layers_by_index,
    kvt_protocol,
    req2corereq,
    sched_discard_zero_block_ids,
    sched_get_kvblk_ids,
    sched_rpc_server,
    sched_rpc_server_port,
    set_param,
    use_flashinfer,
    use_mla,
    use_sparse_mla,
    use_turboquant,
)
from .migration import (
    _g_migrate_out_req_ids,
    _g_migrate_out_req_info,
    _g_migrate_out_req_info_lock,
)

# yapf: enable
from .utils import (
    CodeError,
    ConnManager,
    IoRet,
    IoState,
    PeerInfo,
    PeerManager,
    handle_done_req,
    kill_me_if_exception,
    try_advance,
)

try:
    import blade_kvt
    from blade_kvt import kv_transfer as bladekv
    from blade_kvt.kv_transfer import connect_naming
    from blade_kvt.nic_affinity import generate as generate_nic_affinity
except ImportError:
    blade_kvt = PlaceholderModule("blade_kvt")
    connect_naming = blade_kvt.placeholder_attr("connect_naming")
    generate_nic_affinity = blade_kvt.placeholder_attr("generate")
    bladekv = PlaceholderModule("blade_kvt.kv_transfer")

logger = get_logger(__name__)


# Turboquant Related
def _get_turboquant_spec(
    kv_cache_config: KVCacheConfig,
) -> Optional[TurboQuantAttentionSpec]:  # noqa: F821
    """Return the TurboQuantAttentionSpec from kv_cache_config if present."""
    from vllm.v1.kv_cache_interface import TurboQuantAttentionSpec
    for g in kv_cache_config.kv_cache_groups:
        if isinstance(g.kv_cache_spec, TurboQuantAttentionSpec):
            return g.kv_cache_spec
    return None


def _compute_turboquant_last_dim_size(
    spec: TurboQuantAttentionSpec,  # noqa: F821
) -> int:
    """Compute per-head last_dim_size for TurboQuant KV cache,
    mirroring TurboQuantAttentionSpec.real_page_size_bytes logic."""
    if envs.VLLM_FLASH_ATTN_USE_FAST_TURBOQUANT:
        return (
            spec.packed_key_dim * get_dtype_size(torch.uint8)
            + spec.head_size // 128 * get_dtype_size(torch.float16)
        )
    else:
        return (
            spec.packed_key_dim * get_dtype_size(torch.uint8)
            + 16
        )


def _resolve_turboquant_raw_tensors(
    kv_caches: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Convert TurboQuantCache values back to their underlying raw tensors.

    In gpu_model_runner, the TurboQuant KV cache is allocated as a single
    flat int8 buffer, then reshaped into a 4-D ``raw_tensor_view``
    ``[num_blocks, 2, block_size, num_heads * last_dim_size]``.
    ``key_cache`` and ``value_cache`` inside TurboQuantCache are sliced
    views (dim-1 == 0 and 1) of that same buffer, so they share the same
    ``untyped_storage``.

    We recover that shared storage and wrap it as a 1-D int8 tensor so
    that bladekv can transfer the raw bytes directly.
    """
    from vllm.v1.attention.ops.turboquant import TurboQuantCache
    resolved: dict[str, torch.Tensor] = {}
    for name, kv_cache in kv_caches.items():
        if isinstance(kv_cache, TurboQuantCache):
            key_storage = kv_cache.key_cache.indices.untyped_storage()
            val_storage = kv_cache.value_cache.indices.untyped_storage()
            assert key_storage.data_ptr() == val_storage.data_ptr(), (
                f"TurboQuantCache layer {name}: key and value indices "
                f"do not share the same underlying storage"
            )
            raw_tensor = torch.tensor(
                [], dtype=torch.int8, device=kv_cache.device
            )
            raw_tensor.set_(key_storage, 0, (key_storage.nbytes(),))
            resolved[name] = raw_tensor
        else:
            resolved[name] = kv_cache
    return resolved


# Trim gdn block list for pd-disagg / migration kv transfer.
def handle_hybrid_blocks(
    grouped_blks,
    gamma,
    num_gdn_layers: int,
    num_kv_cache_groups,
    has_null_prefix_block: bool,
    ple_group_index: Optional[int] = None,
) -> list[list[int]]:
    assert num_kv_cache_groups > num_gdn_layers > 0, (
        f"num_gdn_layers must be positive, got {num_gdn_layers}"
    )
    assert len(grouped_blks) == num_kv_cache_groups, (
        f"Block group count mismatch: expected {num_kv_cache_groups}, "
        f"got {len(grouped_blks)}"
    )

    # Index of the runtime block inside a GDN group's block list:
    #   * has_null_prefix_block == True  (mamba_cache_mode == "light"):
    #       MambaManager.allocate_new_blocks always inserts blocks[0] =
    #       null_block in light mode, independent of whether prefix caching
    #       is enabled. So the runtime block sits at blocks[1].
    #   * has_null_prefix_block == False (non-light mamba modes):
    #       no null prefix is inserted, runtime block sits at blocks[0].
    runtime_idx = 1 if has_null_prefix_block else 0
    # Minimum expected layout: [optional null] + 1 runtime + gamma MTP
    # workspace blocks.
    min_blocks = runtime_idx + 1 + gamma

    handled_blks = []
    for group_idx in range(len(grouped_blks)):
        if group_idx < num_gdn_layers or group_idx == ple_group_index:
            # In PD-disagg (and migration), prefill on P only computes the
            # truncated prompt (last gamma+1 tokens are recomputed locally on
            # D), so the MTP workspace blocks are not populated and have
            # nothing meaningful to transfer. Drop them to reduce kvt
            # bandwidth (saves gamma * page_size bytes per GDN/PLE layer per
            # request).
            blocks = grouped_blks[group_idx]
            assert len(blocks) >= min_blocks, (
                f"len(blocks) = {len(blocks)}, gamma = {gamma}, "
                f"runtime_idx = {runtime_idx}"
            )
            handled_blks.append(blocks[runtime_idx:runtime_idx + 1])
        else:
            handled_blks.append(grouped_blks[group_idx])
    return handled_blks


def get_kvblk_ids(
    backend: HybridBackend,
    reqid: str,
    blks: Optional[list[list[int]]] = None,
) -> list[list[int]]:
    if blks is None:
        blks = sched_get_kvblk_ids(reqid)
    # Apply hybrid trimming for any hybrid model with GDN layers, regardless
    # of whether prefix caching is enabled. Whether a null prefix block sits
    # at blocks[0] depends only on mamba_cache_mode == "light"
    # (MambaManager.allocate_new_blocks always inserts blocks[0] = null_block
    # in light mode), NOT on _enable_prefix_caching. So we cannot reuse
    # backend.has_null_blk here, which additionally requires prefix caching
    # being enabled.
    if backend.is_hybrid and backend._num_gdn_layers > 0:
        has_null_prefix_block = (
            backend._vllm_config.cache_config.mamba_cache_mode == "light"
        )
        blks = handle_hybrid_blocks(
            blks,
            backend._gamma,
            num_gdn_layers=backend._num_gdn_layers,
            num_kv_cache_groups=backend._num_kv_cache_groups,
            has_null_prefix_block=has_null_prefix_block,
            ple_group_index=backend._ple_group_index,
        )
    return blks


D_DISAGG = "ali_llumnix_disagg"
P_REMOTE_DECODE = "do_remote_decode"
D_REMOTE_PREFILL = "do_remote_prefill"

# value: str, p instance id.
D_PID = "_hbkvtpid"

_REGISTER_WORKER = 0x20181224
_REGISTER_WORKER_RESP = 0x20181225

# ignore p outputs in pd_disagg reqs
P_IGNORE_OUTPUTS = "ignore_output"

def _check_req_aborted(req: Request):
    if not req_aborted(req):
        return
    raise RuntimeError("req aborted")


def _rtcheck(left, right):
    if left != right:
        logger.error("RTCHECK FAILED left=%r right=%r", left, right)
        os.abort()

@cache
def _check_kvt_version() -> int:
    kvt_server_sig = inspect.signature(bladekv.KVTransferServer)
    kvt_client_sig = inspect.signature(bladekv.KVTransferClient)
    if "num_kv_heads" not in kvt_server_sig.parameters or \
            "num_kv_heads" not in kvt_client_sig.parameters:
        logger.error(
            "KVTransferServer/KVTransferClient missing 'num_kv_heads' "
            "parameter. ")
        raise RuntimeError("Need update kvt version.")
    if kvt_server_sig.parameters["block_bytes"].annotation == Union[List[int], int]:
        _rtcheck(
            kvt_client_sig.parameters["block_bytes"].annotation,
            Union[List[int], int]
        )
        _rtcheck(
            kvt_server_sig.parameters["token_bytes"].annotation,
            Union[List[int], int]
        )
        _rtcheck(
            kvt_client_sig.parameters["token_bytes"].annotation,
            Union[List[int], int]
        )
        _rtcheck(
            kvt_server_sig.parameters["layers"].annotation,
            Union[List[List[torch.Tensor]], List[torch.Tensor]]
        )
        _rtcheck(
            kvt_client_sig.parameters["layers"].annotation,
            Union[List[List[torch.Tensor]], List[torch.Tensor]]
        )
        logger.info("Use KVT version 2.0")
        return 2
    else:
        _rtcheck(kvt_server_sig.parameters["block_bytes"].annotation, int)
        _rtcheck(kvt_client_sig.parameters["block_bytes"].annotation, int)
        _rtcheck(kvt_server_sig.parameters["token_bytes"].annotation, int)
        _rtcheck(kvt_client_sig.parameters["token_bytes"].annotation, int)
        _rtcheck(
            kvt_client_sig.parameters["layers"].annotation,
            List[torch.Tensor]
        )
        _rtcheck(
            kvt_client_sig.parameters["layers"].annotation,
            List[torch.Tensor]
        )
        logger.info("Use KVT version 1.0")
        return 1

try:
    KVT_VERSION = _check_kvt_version()
except Exception:
    KVT_VERSION = None

def alloc_kv_cache_ppu(cache):
    isxpu = not blade_kvt.is_nv_gpu()
    _rtcheck(cache.storage_offset(), 0)
    _rtcheck(cache.requires_grad, False)
    orig_sizes = copy.deepcopy(cache.size())
    orig_strides = copy.deepcopy(cache.stride())
    orig_nbytes = copy.deepcopy(cache.nbytes)
    orig_dev = copy.deepcopy(cache.device)
    orig_elem_size = copy.deepcopy(cache.element_size())
    orig_storage_elem_size = copy.deepcopy(
        cache.untyped_storage().element_size())
    orgi_storage_dev = copy.deepcopy(cache.untyped_storage().device)
    orig_storage_nbytes = copy.deepcopy(cache.untyped_storage().nbytes())
    _rtcheck(orig_dev, orgi_storage_dev)

    if not isxpu or kvt_protocol() == 'tcp':
        return

    logger.info("alloc_kv_cache_ppu: size=%s dev=%s isxpu=%s",
                orig_storage_nbytes, orig_dev, isxpu)
    mem_storage = bladekv.alloc_phy_cont_mem(cache.untyped_storage().nbytes(),
                                             cache.device)
    cache.set_(mem_storage, 0,
               cache.size())  # this will deallocate origin memory.
    _rtcheck(orig_sizes, cache.size())
    _rtcheck(orig_strides, cache.stride())
    _rtcheck(orig_nbytes, cache.nbytes)
    _rtcheck(orig_dev, cache.device)
    _rtcheck(orig_elem_size, cache.element_size())
    _rtcheck(orig_storage_elem_size, cache.untyped_storage().element_size())
    _rtcheck(orgi_storage_dev, cache.untyped_storage().device)
    _rtcheck(orig_storage_nbytes, cache.untyped_storage().nbytes())
    _rtcheck(cache.untyped_storage().data_ptr(), mem_storage.data_ptr())
    cache.zero_()
    return

# kvcache transfer dest info
class KVTDInfo(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    instid: str
    blkids: list[list[int]]
    cached_tokens: int
    max_tokens: int
    d_workers_info: list[str]


# value: KVTDInfo
P_KVTD_INFO = "hbpkvtdinfo"


class RKVTDInfo(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    reqid: str
    dinfo: KVTDInfo
    migration: bool = False


CODE_OK = 0
CODE_REQNOTFOUND = 404
CODE_INTERNALERROR = 500
CODE_MIGRATE_REJECTED_BUSY = 1001
# 410 Gone: the reqid was previously known on P but it has been removed
# due to abort or dual_req_done.timeout. Unlike CODE_REQNOTFOUND
# (404, which is ambiguous and tells D "retry later, maybe
# the prefill hasn't finished yet"), CODE_REQGONE is terminal — D should
# stop retrying and fail the load immediately so the loading slot can be
# released. Tracked via PBackend._gone_reqs (TTL controlled by
# VLLM_KVT_GONE_REQ_TTL_S, 0 disables the LRU and preserves legacy 404
# behavior).
CODE_REQGONE = 410


class KVTResp(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # see CODE_*
    code: int
    cached: int
    computed: int
    output_token_ids: Optional[list[int]] = None


@dataclass
class KVTState:
    dinfo: KVTDInfo
    maxtokens: int
    untouched: bool = True


# value: KVTState
P_KVT_STATE = "hbpkvtstate"

# PREFILL len(prompt_token_ids) - 1
# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded EngineCoreRequest
PREFILL_REQ = 0x20181218

# body: KVTResp
# B.T.W Here, we can stream the EngineCoreOutput back to D to support
# log probabilities and prompt log probabilities.
PREFILL_RESP = 0x81218102

# Body: see utils.
SEND_DONE_REQ = 0x20181219
SEND_DONE_RESP = 0x91218102

# transfer_kv
# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded RKVTDInfo
# resp: KVTResp
TRANSFER_KV_REQ = 0x20210912
TRANSFER_KV_RESP = PREFILL_RESP

# COPY FROM migration/backend.py, move this rpc to hybridscheduler?
# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded list[str]
ABORT_REQS_REQ = 20250820
ABORT_REQS_RESP = 20250820


def rpc_port(cfg: VllmConfig):
    return sched_rpc_server_port(cfg)


def _get_main_node_pod_name():
    return os.environ.get("POD_NAME")


def _get_inst_id(cfg: VllmConfig, fake_naming: bool = False) -> str:
    assert cfg.kv_transfer_config is not None
    r = cfg.kv_transfer_config.get_from_extra_config("kvt_inst_id", None)
    if r is None:
        if fake_naming:
            r = f"{_get_main_node_pod_name()}-{rpc_port(cfg)}-{get_ip()}"
        else:
            r = _get_main_node_pod_name()
    if r is None:
        r = str(uuid.uuid4())
    return r


# For dual-request mode.
def _try_wakeup_core(q: deque):
    if len(q) != 1:
        return

    wakeup_scheduler()
    return


def _reg_naming(naming_cli, cfg: VllmConfig):
    assert cfg.kv_transfer_config is not None
    if cfg.kv_transfer_config.is_kv_producer:
        role = "prefill"
    else:
        role = "decode"
    tpsize = cfg.parallel_config.tensor_parallel_size
    dprank = cfg.parallel_config.data_parallel_rank
    info = PeerInfo(
        role=role,
        tpsize=tpsize,
        ctime_us=int(time.time_ns() // 1000),
        addr=get_ip(),
        dprank=dprank,
        port=rpc_port(cfg),
    )
    logger.info(f"register naming: {info}")
    naming_cli.store(f"endpoint{dprank}", info.serialize())
    return


def _set_worker_envs(cfg: VllmConfig):
    blade_kvt.set_envs()
    # FLASH_CACHE_SHAPE = 2, see envcfg.h in blade_kvt
    # NOTE: mla attn backend's kv shape:
    # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    # num_blocks = kv.shape[0], as for flash attn num_blocks = kv.shape[1],
    # thus the different default env
    if use_mla():
        if use_sparse_mla():
            # DPSK_V32_SPARSE_MLA_SHAPE = 4 covers BOTH:
            #   - FlashAttention sparse MLA (Hopper, FLASH_MLA_SPARSE)
            #   - FlashInfer sparse MLA   (Blackwell, FLASHINFER_MLA_SPARSE)
            os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "4")
        else:
            os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "1")
    elif cfg.model_config.is_hybrid:
        # Hybrid (Qwen3-next style): GDN + optional indexer + attn.
        if use_flashinfer():
            os.environ.setdefault(
                "BLLM_KVTRANS_CACHE_SHAPE",
                _flashinfer_kvt_cache_shape(is_hybrid=True),
            )
        else:
            # FlashAttn-style hybrid: QWEN3_NEXT_FLASH_CACHE_SHAPE (=3)
            os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "3")
    elif use_flashinfer():
        os.environ.setdefault(
            "BLLM_KVTRANS_CACHE_SHAPE",
            _flashinfer_kvt_cache_shape(is_hybrid=False),
        )
    elif use_turboquant(cfg):
        # TurboQuant 4D layout:
        # [num_blocks, 2, block_size, num_heads * last_dim_size]
        os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "6")
    else:
        os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "2")
    core_ip = scheduler_rpc_host(cfg)
    sdaddr = f"{core_ip}:{rpc_port(cfg)}"
    os.environ.setdefault("BLLM_KVTRANS_SEND_DONE_ADDR", sdaddr)
    return


@dataclass
class PReqMeta:
    # seen_tokens = 0 means submit_req_send
    # seen_tokens > 0 means submit_delta_send, p_block_ids/d_block_ids is empty
    reqid: str
    d_inst_id: str
    p_block_ids: list[list[int]]
    d_block_ids: list[list[int]]
    new_tokens: int
    has_last_token: bool
    seen_tokens: int
    d_workers_info: list[str] = field(default_factory=list)
    has_freeze: bool = False


@dataclass
class KVTPMeta(BackendMeta):
    stepid: int
    substepid: int
    sched_tokens: int
    freeze_metas: list[PReqMeta]
    abort_metas: list[PReqMeta]
    nonfreeze_metas: list[PReqMeta]

    def __bool__(self):
        return bool(self.freeze_metas or self.abort_metas or self.nonfreeze_metas)


def _flatten_cache(
    # turboquant's cache class would be: TurboQuantCache
    kv_caches: dict[str, torch.Tensor | TurboQuantCache],
    ) -> Union[list[torch.Tensor], list[list[torch.Tensor]]]:
    """
    return: layer_name -> [cache_tensor1, cache_tensor2, ...]
    """
    register_layer_num = len(kv_caches)

    from vllm.model_executor.models.utils import extract_layer_index
    index2name = group_layers_by_index(kv_caches)

    # dpsk v32 is special case
    if not use_sparse_mla():
        assert len(index2name) == register_layer_num, (
            f"The number of KV cache layers to be registered:"
            f"{register_layer_num}, does not match the actual number"
            f" registered: {len(index2name)}. Please check if multiple"
            f" layers have the same layer index."
        )

    runner_kv_caches = []
    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]

        if KVT_VERSION == 1:
            assert len(layer_names) == 1, \
            "KVT v1 only support single cache per layer"
            runner_kv_caches.append(kv_caches[layer_names[0]])
        else:
            assert KVT_VERSION == 2, "Unknown KVT version"
            layer_caches = []
            for layer_name in layer_names:
                layer_caches.append(kv_caches[layer_name])
            runner_kv_caches.append(layer_caches)
    return runner_kv_caches


# d_inst_id: dst_inst_name|dst_dprank|dst_tpsize
# Returns the list of (dst_inst_name, dst_worker_id, worker_tp_rank) this P
# rank must send to. Normally one target, but for P_tp < D_tp a single P rank
# fans out to group_n = D_tp / P_tp D ranks. Empty list means the TP config is
# invalid (not evenly divisible).
def _get_dist(d_inst_id: str) -> list[tuple[str, int, int]]:
    src_tprank = get_tp_group().rank_in_group
    src_tpsize = get_tp_group().world_size
    dst_inst_name, dst_dprank, dst_tpsize = d_inst_id.split("|")
    idst_dprank = int(dst_dprank)
    idst_tpsize = int(dst_tpsize)
    base = idst_dprank * idst_tpsize

    if idst_tpsize == src_tpsize:
        return [(dst_inst_name, base + src_tprank, src_tprank)]

    if src_tpsize > idst_tpsize:
        # P_tp > D_tp: several P ranks feed one D rank.
        group_n = src_tpsize // idst_tpsize
        if idst_tpsize * group_n != src_tpsize:
            return []
        dst_tprank = src_tprank // group_n
        return [(dst_inst_name, base + dst_tprank, dst_tprank)]

    # P_tp < D_tp: one P rank fans out to group_n D ranks.
    group_n = idst_tpsize // src_tpsize
    if src_tpsize * group_n != idst_tpsize:
        return []
    dst_base = src_tprank * group_n
    return [
        (dst_inst_name, base + dst_base + j, dst_base + j)
        for j in range(group_n)
    ]


# Returns the list of (dst_inst, dst_worker_id, dst_worker_info) targets for
# this P rank. Empty list means the TP config is invalid.
def _get_distinfo(meta: PReqMeta) -> list[tuple[str, int, Optional[str]]]:
    result = _get_dist(meta.d_inst_id)
    out: list[tuple[str, int, Optional[str]]] = []
    for dst_inst_name, dst_wid, worker_tp_rank in result:
        dst_worker_info = (
            None
            if not meta.d_workers_info
            else meta.d_workers_info[worker_tp_rank]
        )
        out.append((dst_inst_name, dst_wid, dst_worker_info))
    return out


class _SendingReq(IoState):
    def __init__(self, signals_per_worker: int = 1):
        super().__init__(signals_per_worker=signals_per_worker)
        self._fut = asyncio.get_running_loop().create_future()
        return

    def try_mark_done(self, ioret: IoRet):
        if self._fut.done():
            try:
                res: IoRet = self._fut.result()
                logger.warning("mark done twice: ioret=%s res=%s", ioret, res)
            except Exception:
                logger.exception("mark done failed: ioret=%s", ioret)
            return
        self._fut.set_result(ioret)
        return


class DualReq:
    def __init__(self, req: Request, pblkids: list[int], finished=False):
        self.req = req
        self.finished = finished
        self.pblkids = pblkids
        self.insert_ts = time.time()
        assert len(self.pblkids) > 0


def _build_kvt_args(
    backend: HybridBackend,
    kv_caches: dict[str, Any],
    role_log: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    """
    Compute KVTransferClient/Server constructor kwargs from kv_caches.
    """
    cfg = backend._cfg
    kv_cache_config = backend._kv_cache_config

    # MLA models use a latent KV cache, which only carries a single head 
    # regardless of what `num_key_value_heads` says in the HF config.
    if use_mla():
        total_num_kv_heads = 1
    else:
        total_num_kv_heads = cfg.model_config.get_total_num_kv_heads()

    _hybrid_kwargs: dict[str, Any] = {}

    # Detect TurboQuant and resolve raw tensors if needed
    tq_spec = _get_turboquant_spec(kv_cache_config)
    tq_last_dim_size = 0
    if tq_spec is not None:
        tq_last_dim_size = _compute_turboquant_last_dim_size(tq_spec)
        kv_caches = _resolve_turboquant_raw_tensors(kv_caches)
        logger.info(
            "%s TurboQuant detected: last_dim_size=%s, "
            "num_kv_heads=%s, packed_key_dim=%s",
            role_log, tq_last_dim_size, tq_spec.num_kv_heads,
            tq_spec.packed_key_dim,
        )

    if backend.is_hybrid:
        # Pick one representative tensor cache from hybrid kv caches.
        layer_cache: torch.Tensor | None = None
        for layer_name, cache in kv_caches.items():
            # gdn layer's kv cache is a list contains conv state and ssm state
            # ignore indexer's cache and use self.attn's kv cache to register
            if isinstance(cache, torch.Tensor) and "indexer" not in layer_name:
                layer_cache = cache
                break

        assert layer_cache is not None

        from vllm.model_executor.models.utils import extract_layer_index

        # When attn_pack_size > 1, the trailing `attn_pack_size` non-indexer
        # attn layers in each tensor_group.shared_by share a single physical
        # KV-cache page. We register one tensor per pack:
        #   - use the FIRST pack member's view (storage_offset=0, data_ptr
        #     at physical page base) as the registered tensor;
        #   - key it under the LAST pack member's name so that
        #     _flatten_cache lines up with hybrid_model_send_layer's
        #     "wait for last attn in pack" semantics.
        attn_pack_size = cfg.cache_config.attn_pack_size
        assert attn_pack_size >= 1
        _hybrid_kwargs['attn_pack_size'] = attn_pack_size

        physical_tensors: dict[str, torch.Tensor] = {}
        # num_gdn_layers is determined by kv_cache_groups, which defines block order
        assert backend._num_gdn_layers > 0, (
            f"_num_gdn_layers = {backend._num_gdn_layers}"
        )

        _hybrid_kwargs['num_gdn_layers'] = backend._num_gdn_layers
        _hybrid_kwargs['num_ple_layers'] = backend._num_ple_layers
        _hybrid_kwargs['ple_block_group'] = (
            backend._ple_group_index
            if backend._ple_group_index is not None else 0
        )
        hf_config = cfg.model_config.hf_text_config
        tp_size = cfg.parallel_config.tensor_parallel_size
        key_dim = hf_config.linear_num_key_heads \
            * hf_config.linear_key_head_dim // tp_size
        value_dim = hf_config.linear_num_value_heads \
            * hf_config.linear_value_head_dim // tp_size
        _hybrid_kwargs['gdn_conv_channel_dims'] = [key_dim, key_dim, value_dim]

        indexer_spec = next(
            (g.kv_cache_spec
             for g in kv_cache_config.kv_cache_groups
             if any("indexer" in n for n in g.layer_names)),
            None,
        )
        if indexer_spec:
            indexer_block_size = indexer_spec.block_size
            indexer_token_size = (indexer_spec.num_kv_heads
                                 * indexer_spec.head_size
                                 * get_dtype_size(indexer_spec.dtype))
            _hybrid_kwargs['indexer_blk_ntpb'] = indexer_block_size
            _hybrid_kwargs['hybrid_indexer_token_size'] = indexer_token_size
        # Trigger KV cache transfer using the last layer of each shared_by group.
        for tensor_group in kv_cache_config.kv_cache_tensors:
            shared_by = tensor_group.shared_by
            # Partition shared_by into recurrent state (list), indexer
            # (skip), and non-indexer attn (Tensor) entries. Qwen4 can append
            # a PLE short-conv list after the attention tensor, so attention
            # is no longer required to occupy the physical tail of shared_by.
            non_indexer_attn: list[tuple[int, str]] = []
            for lidx, layer in enumerate(shared_by):
                cache = kv_caches[layer]
                if isinstance(cache, torch.Tensor) and \
                        not layer.endswith(".indexer.k_cache"):
                    non_indexer_attn.append((lidx, layer))

            assert 0 < len(non_indexer_attn) <= attn_pack_size, (
                f"Expected 1..{attn_pack_size} non-indexer attn layers "
                f"in shared_by, got {len(non_indexer_attn)}: "
                f"{[n for _, n in non_indexer_attn]}"
            )
            attn_positions = [lidx for lidx, _ in non_indexer_attn]
            if len(attn_positions) > 1:
                expected_positions = list(
                    range(attn_positions[0],
                          attn_positions[0] + len(attn_positions))
                )
                assert attn_positions == expected_positions, (
                    f"attn pack entries must be contiguous: "
                    f"positions={attn_positions}, shared_by={shared_by}"
                )
            # Within the pack, layer indices must be increasing so that
            # the last entry triggers the per-pack send.
            attn_layer_idxs = [
                extract_layer_index(n) for _, n in non_indexer_attn
            ]
            assert sorted(attn_layer_idxs) == attn_layer_idxs, (
                f"attn pack layer indices must be sorted: "
                f"{attn_layer_idxs}"
            )
            attn_names = {n for _, n in non_indexer_attn}
            last_attn = non_indexer_attn[-1][1]
            last_attn_idx = extract_layer_index(last_attn)
            assert last_attn_idx == max(attn_layer_idxs), (
                "Pack's last attn layer must have max attn layer_idx in "
                f"shared_by; got last={last_attn_idx}, "
                f"all_attn={attn_layer_idxs}"
            )
            backend.hybrid_model_send_layer.append(last_attn_idx)

            # Register one tensor per pack. Use the first pack member's
            # view (data_ptr already at the physical page base because
            # its storage_offset is 0); store under last member's name.
            first_attn = non_indexer_attn[0][1]
            first_view = kv_caches[first_attn]
            last_view = kv_caches[last_attn]
            # Sanity: pack members share the same underlying storage.
            assert (
                first_view.untyped_storage().data_ptr()
                == last_view.untyped_storage().data_ptr()
            ), (
                f"Pack members {first_attn!r} and {last_attn!r} expected "
                "to share the same underlying storage but do not."
            )
            # Rewrap the storage as a flat 1-D view
            # Register size should be attn_pack_size * tensor_size
            assert first_view.storage_offset() == 0, (
                f"Expected first_view.storage_offset()==0 for "
                f"{first_attn!r}, got {first_view.storage_offset()}"
            )
            storage = first_view.untyped_storage()
            elem_size = first_view.element_size()
            assert storage.nbytes() % elem_size == 0, (
                f"Storage size {storage.nbytes()} not divisible by "
                f"element size {elem_size} for {first_attn!r}"
            )
            total_elems = storage.nbytes() // elem_size
            flat_view = torch.empty(
                0, dtype=first_view.dtype, device=first_view.device
            )
            flat_view.set_(storage, 0, (total_elems,), (1,))
            physical_tensors[last_attn] = flat_view

            # Walk all non-attention entries for recurrent-state metadata.
            # GDN has [conv, ssm]; Qwen4 PLE short-conv has [conv] only and
            # can appear after the attention tensor in shared_by.
            for layer in shared_by:
                cache = kv_caches[layer]
                if isinstance(cache, torch.Tensor):
                    if layer in attn_names or layer.endswith(".indexer.k_cache"):
                        continue
                    raise AssertionError(
                        f"Unexpected Tensor cache in shared_by: {layer}"
                    )
                assert isinstance(cache, list), (
                    f"Expected recurrent state list for {layer}, got "
                    f"{type(cache).__name__}"
                )

                if len(cache) == 2:
                    conv_state = cache[0]
                    ssm_state = cache[1]
                    assert len(conv_state.shape) == 3, \
                        f"{conv_state.shape=}"  # conv
                    assert len(ssm_state.shape) == 4, \
                        f"{ssm_state.shape=}"  # ssm
                    if 'conv_state_shape' not in _hybrid_kwargs:
                        _hybrid_kwargs.update(
                            conv_state_shape=list(conv_state.shape),
                            ssm_state_shape=list(ssm_state.shape),
                            gdn_conv_elem_size=conv_state.element_size(),
                            gdn_ssm_elem_size=ssm_state.element_size(),
                        )
                        assert 2*key_dim + value_dim == conv_state.shape[2], \
                            f"2*key_dim + value_dim = {2*key_dim + value_dim}, " \
                            f"conv_state.shape[2] = {conv_state.shape[2]}"
                        logger.info(
                            "%s's GDN state: "
                            "conv_elem_size=%s ssm_elem_size=%s",
                            role_log,
                            conv_state.element_size(),
                            ssm_state.element_size(),
                        )
                    if "QWEN3_NEXT_CONV_SHAPE" not in os.environ:
                        os.environ["QWEN3_NEXT_CONV_SHAPE"] = (
                            ",".join(map(str, conv_state.shape))
                        )
                    if "QWEN3_NEXT_SSM_SHAPE" not in os.environ:
                        os.environ["QWEN3_NEXT_SSM_SHAPE"] = (
                            ",".join(map(str, ssm_state.shape))
                        )
                    if "GDN_CONV_ELEM_SIZE" not in os.environ:
                        os.environ["GDN_CONV_ELEM_SIZE"] = (
                            str(conv_state.element_size())
                        )
                        logger.info(
                            "%s's GDN Conv state element size=%s",
                            role_log,
                            conv_state.element_size(),
                        )
                    if "GDN_SSM_ELEM_SIZE" not in os.environ:
                        os.environ["GDN_SSM_ELEM_SIZE"] = (
                            str(ssm_state.element_size())
                        )
                        logger.info(
                            "%s's GDN SSM state element size=%s",
                            role_log,
                            ssm_state.element_size(),
                        )
                    if "BLLM_KVTRANS_GDN_CONV_CHANNEL_DIMS" not in os.environ:
                        os.environ["BLLM_KVTRANS_GDN_CONV_CHANNEL_DIMS"] = (
                            f"{key_dim},{key_dim},{value_dim}"
                        )
                    continue

                assert len(cache) == 1, (
                    f"Unexpected recurrent state list length for {layer}: "
                    f"{len(cache)}"
                )
                conv_state = cache[0]
                assert len(conv_state.shape) == 3, \
                    f"{conv_state.shape=}"  # short conv
                if 'ple_conv_state_shape' not in _hybrid_kwargs:
                    ple_conv_dim = conv_state.shape[2]
                    _hybrid_kwargs.update(
                        ple_conv_state_shape=list(conv_state.shape),
                        ple_conv_elem_size=conv_state.element_size(),
                        ple_conv_channel_dims=[ple_conv_dim],
                    )
                    logger.info(
                        "%s's PLE state: conv_shape=%s conv_elem_size=%s",
                        role_log,
                        list(conv_state.shape),
                        conv_state.element_size(),
                    )
        if backend._num_ple_layers > 0:
            assert 'ple_conv_state_shape' in _hybrid_kwargs, (
                "PLE group exists but no short-conv state was found in kv_caches"
            )
        assert backend.hybrid_model_send_layer \
            == sorted(backend.hybrid_model_send_layer)
        assert len(backend.hybrid_model_send_layer) \
            == len(set(backend.hybrid_model_send_layer))
        logger.info("%s hybrid model send layer: %s",
                    role_log, backend.hybrid_model_send_layer)
        assert len(backend.hybrid_model_send_layer) == len(physical_tensors)
        kv_caches = physical_tensors
        cache_shape = layer_cache.shape
        if tq_spec is not None:
            # hybrid + turboquant: raw tensor is 1-D int8,
            # compute token_bytes from spec directly
            token_bytes = [
                2 * tq_spec.num_kv_heads * tq_last_dim_size
            ]
        else:
            # use full attn block to calculate block_bytes
            # mamba block will padding to full attn block size
            # [2 (k and v), num_blocks, block_size, kv_heads, head_dim]
            token_bytes = [
                2 * cache_shape[3] * cache_shape[4]
                * layer_cache.element_size()
            ]
    else:
        assert tq_spec is None, "TurboQuant only support hybrid mode"
        # Since it's unclear how the model architecture will evolve in the future,
        # we currently only handle special cases based on the Dpsk-V32 structure.
        tensor_shape_dict: dict[
            tuple[int, ...], list[tuple[str, torch.Tensor]]
        ] = defaultdict(list)

        for layer, layer_kv in kv_caches.items():
            tensor_shape_dict[layer_kv.shape].append((layer, layer_kv))

        if len(tensor_shape_dict) == 1:  # Normal case
            cache_shape, pairs_list = next(iter(tensor_shape_dict.items()))
            _, layer_tensor = pairs_list[0]
            if len(cache_shape) == 5:
                # flash attn:
                # [2 (k and v), num_blocks, block_size, kv_heads, head_dim]
                # or flashinfer
                # shape:
                # [num_blocks, 2 (k and v), block_size, kv_heads, head_dim]
                # stride:
                # [num_blocks, 2 (k and v), kv_heads, block_size, head_dim]
                token_bytes = [
                    2 * cache_shape[3] * cache_shape[4]
                    * layer_tensor.element_size()
                ]
            else:
                # for mla, which's kv shape like:
                # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
                assert len(cache_shape) == 3
                token_bytes = [cache_shape[2] * layer_tensor.element_size()]
        else:
            token_bytes = []
            for cache_shape, pair_list in tensor_shape_dict.items():
                # DPSK V3.2 sparse MLA: per-layer has two 3D tensors
                # (main MLA + indexer). Same shape applies to FA sparse
                # (FLASH_MLA_SPARSE) and FlashInfer sparse
                # (FLASHINFER_MLA_SPARSE).
                assert len(cache_shape) == 3, (
                    "Multi-shape per layer only supported for DPSK V3.2 "
                    "sparse MLA (FA / FlashInfer); got "
                    f"shape={cache_shape}"
                )
                _, layer_tensor = pair_list[0]
                # k cache & kv cache share same num_blocks and block_size
                token_bytes.append(cache_shape[2] * layer_tensor.element_size())

    block_size = cfg.cache_config.block_size
    # When attn_pack_size > 1 in hybrid models, each registered attn tensor
    # actually represents a physical KV page that bundles attn_pack_size
    # consecutive attention layers. block_bytes must reflect the physical
    # page size (= attn_pack_size * per_layer_block_bytes).
    attn_pack_size_for_block = (
        cfg.cache_config.attn_pack_size if backend.is_hybrid else 1
    )
    block_bytes = [
        token_byte * block_size * attn_pack_size_for_block
        for token_byte in token_bytes
    ]
    rank = get_tensor_model_parallel_rank()
    worker_id = get_tp_group().rank
    if kvt_protocol() == "rdma":
        protocol = bladekv.KVTransferProtocolType.RDMA_DIRECT
    elif kvt_protocol() == "tcp":
        protocol = bladekv.KVTransferProtocolType.TCP
    else:
        raise AssertionError(f"Unknown KVT Protocol: {kvt_protocol()}")

    if KVT_VERSION == 1:
        assert len(block_bytes) == len(token_bytes) == 1, \
            "KVT 1.0 only support one tensor per layer"
        block_bytes = block_bytes[0]
        token_bytes = token_bytes[0]

    for layer_name, layer_cache in kv_caches.items():
        if isinstance(layer_cache, torch.Tensor):
            logger.info(
                "Flatten Cache: layer_names=%s, "
                "layer_shape=%s, layer_stride=%s, layer_dtype=%s",
                layer_name, layer_cache.shape,
                layer_cache.stride(), layer_cache.dtype
            )
        else:
            logger.info(
                "Flatten Cache: layer_names=%s, type=%s",
                layer_name, type(layer_cache).__name__
            )

    factory_kwargs: dict[str, Any] = {
        "inst_id": backend._inst_id,
        "tp_size": cfg.parallel_config.tensor_parallel_size,
        "worker_id": worker_id,
        "worker_tp_rank": rank,
        "block_bytes": block_bytes,
        "token_bytes": token_bytes,
        "naming_url": backend._naming_url,
        "protocols": [protocol],
        "num_kv_heads": total_num_kv_heads,
        **_hybrid_kwargs,
    }
    return kv_caches, factory_kwargs, cache_shape


class PBackend(HybridBackend):
    SOURCE_LABEL = "kvt"

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._naming_url = vllm_config.kv_transfer_config.get_from_extra_config(
            "naming_url", "fake://"
        )
        if self._naming_url == "fake://":
            self._inst_id = _get_inst_id(vllm_config, fake_naming=True)
            self._naming_cli = None
        else:
            self._inst_id = _get_inst_id(vllm_config)
            self._naming_cli = connect_naming(self._inst_id, self._naming_url)
        self._cfg = vllm_config
        self._kv_cache_config = kv_cache_config
        self._gamma = get_p_node_pop_len(self._cfg) - 1
        self._enable_prefix_caching = self._cfg.cache_config.enable_prefix_caching

        if role == KVConnectorRole.WORKER:
            generate_nic_affinity()
            os.environ.setdefault("ACCL_RX_DEPTH", "4")
            os.environ.setdefault("BLLM_KVTRANS_RESERVE", "4096,128;")
            _set_worker_envs(vllm_config)
            self._bladkv_cli = None
        else:
            if self._naming_cli is not None:
                _reg_naming(self._naming_cli, vllm_config)
            rpcsrv = sched_rpc_server()
            rpcsrv.register_method(TRANSFER_KV_REQ, self._on_transfer_kv)
            rpcsrv.register_method(PREFILL_REQ, self._on_prefill)
            rpcsrv.register_method(SEND_DONE_REQ, self._on_send_done)
            rpcsrv.register_method(ABORT_REQS_REQ, self._on_abort_reqs)
            self._sending: dict[str, _SendingReq] = dict()
            self._reqdec = MsgpackDecoder(EngineCoreRequest)
            self._dec = msgspec.msgpack.Decoder()
            self._packenc = msgspec.msgpack.Encoder()
            self._kvtreqdec = msgspec.msgpack.Decoder(RKVTDInfo)

            # R/W: core thread
            self._dual_req_done: dict[str, DualReq] = dict()
            self._infly_kvt: dict[str, PReqMeta] = dict()

            self._dinfoq: deque[RKVTDInfo] = deque()

            # R/W: core thread. Used by _step_dinfoq to
            # distinguish "really gone" (CODE_REQGONE, terminal) from
            # "never seen yet" (CODE_REQNOTFOUND, transient/retryable).
            self._gone_reqs: OrderedDict[str, float] = OrderedDict()

        return

    # Hard cap on the gone-reqs LRU so we can never grow unbounded even if
    # the TTL is huge and the abort/timeout rate is high.
    _GONE_REQS_MAX = 8192

    def naming_cli(self):
        return self._naming_cli

    # disagg thread, core thread
    def _tp_size(self):
        return self._cfg.parallel_config.tensor_parallel_size

    def _kvt_signals_per_worker(self, dinfo) -> int:
        # For P_tp < D_tp each P worker fans out to group_n = D_tp / P_tp D
        # ranks and therefore emits group_n send-done signals per req (one per
        # stub), all tagged with the same P rank. Require group_n signals per
        # worker so a req is only marked done once every fan-out target reports.
        p_tp = self._tp_size()
        try:
            d_tp = int(dinfo.instid.split("|")[2])
        except (IndexError, ValueError, AttributeError):
            return 1
        if d_tp > p_tp and d_tp % p_tp == 0:
            return d_tp // p_tp
        return 1

    def _do_send_done(self, worker_tprank: int, ioret: IoRet):
        if ioret.source is None:
            ioret.source = self.source_label()
        tpsize = self._tp_size()
        state: Optional[_SendingReq] = try_advance(
            self._sending, ioret, worker_tprank, tpsize
        )
        if state is None:
            return

        ioret = state.merge()
        state.try_mark_done(ioret)

    async def _on_send_done(self, reader, writer):
        await handle_done_req(reader, writer, self._do_send_done, SEND_DONE_RESP)
        return

    def _gone_ttl_s(self) -> int:
        # Cached on the env-var to avoid re-reading on every step; envs.* is
        # already cached internally so this is essentially a dict lookup but
        # keeping it isolated makes future overrides easy.
        return int(envs.VLLM_KVT_GONE_REQ_TTL_S)

    def _mark_gone(self, reqid: str, reason: str):
        """Insert ``reqid`` into the gone-LRU with a fresh timestamp.

        Called from the core thread by abort/timeout paths in
        :meth:`_step_aborting` (and any future location that pops
        ``_dual_req_done`` due to an abort). No-op when the LRU is disabled.
        """
        ttl = self._gone_ttl_s()
        if ttl <= 0:
            return
        now = time.time()
        # Move-to-end is OK even if reqid was already present; the freshest
        # insert time is what we want for the TTL check.
        self._gone_reqs.pop(reqid, None)
        self._gone_reqs[reqid] = now
        # Hard cap eviction (insertion order ≈ insertion time, so popitem
        # from the front drops the oldest entries first).
        while len(self._gone_reqs) > self._GONE_REQS_MAX:
            self._gone_reqs.popitem(last=False)
        logger.info(
            "mark gone reqid. reqid=%s reason=%s gone_size=%s ttl_s=%s",
            reqid, reason, len(self._gone_reqs), ttl,
        )

    def _is_gone(self, reqid: str) -> bool:
        """Return True iff ``reqid`` is in the LRU and not yet TTL-expired.

        Side effect: TTL-expired entries hit during the lookup are evicted.
        """
        ttl = self._gone_ttl_s()
        if ttl <= 0:
            return False
        ts = self._gone_reqs.get(reqid)
        if ts is None:
            return False
        if time.time() - ts > ttl:
            # Expired — drop it and treat as "never seen".
            self._gone_reqs.pop(reqid, None)
            return False
        return True

    def _check_kvtdinfo(self, info: KVTDInfo) -> bool:
        assert info.cached_tokens < info.max_tokens
        if self.naming_cli():
            return True
        assert len(info.d_workers_info) > 0
        assert all(len(winfo) > 0 for winfo in info.d_workers_info)
        return True

    async def _wait_kvt_state(self, reqid: str, state: _SendingReq) -> KVTResp:
        start_ts = time.monotonic()
        ioret: IoRet = await state._fut
        end_ts = time.monotonic()
        kvt_dur_ms = (end_ts - start_ts) * 1000
        assert ioret.reqid == reqid
        code = CODE_OK
        if ioret.ex is not None:
            if isinstance(ioret.ex, CodeError):
                code = ioret.ex.code
            else:
                code = CODE_INTERNALERROR
        computed = -1
        req = self.get_request(reqid)
        if req is not None:
            computed = req.num_computed_tokens

        log_fn = logger.info if ioret.ex is None else logger.exception
        log_fn(
            "disagg kvt done. kvt_dur_ms=%s, reqid=%s cached=%s computed=%s",
            kvt_dur_ms,
            reqid,
            ioret.n,
            computed,
            exc_info=ioret.ex)

        resp = KVTResp(code=code,
                       cached=ioret.n or 0,
                       computed=computed,
                       output_token_ids=[])
        self._sending.pop(reqid)
        return resp

    async def submit_transfer_kv(self, req: RKVTDInfo) -> KVTResp:
        logger.info(
            "disagg start kvt. reqid=%s peer=%s computed=%s max=%s, d_workers_info=%s",
            req.reqid,
            req.dinfo.instid,
            req.dinfo.cached_tokens,
            req.dinfo.max_tokens,
            req.dinfo.d_workers_info,
        )
        assert self._check_kvtdinfo(req.dinfo), f"{req=}"

        assert req.reqid not in self._sending
        state = _SendingReq(self._kvt_signals_per_worker(req.dinfo))
        self._sending[req.reqid] = state

        self._dinfoq.append(req)
        _try_wakeup_core(self._dinfoq)
        resp = await self._wait_kvt_state(req.reqid, state)
        return resp

    async def _on_transfer_kv(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        req: RKVTDInfo = self._kvtreqdec.decode(reqbuf)

        resp = await self.submit_transfer_kv(req)
        respbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", respbuf, 0, PREFILL_RESP, 0)
        self._packenc.encode_into(resp, respbuf, 8)
        struct.pack_into("=I", respbuf, 4, len(respbuf) - 8)
        writer.write(respbuf)
        await writer.drain()
        return

    async def _on_prefill(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        req: EngineCoreRequest = self._reqdec.decode(reqbuf)
        dinfol = core_get_param(req, P_KVTD_INFO)
        assert dinfol is not None
        dinfo = KVTDInfo(*dinfol)
        assert self._check_kvtdinfo(dinfo), f"{dinfo=}"

        assert dinfo.max_tokens == len(req.prompt_token_ids) - (self._gamma + 1), (
            f"reqid={req.request_id}, dinfo.max_tokens={dinfo.max_tokens}, "
            f"prompt_len={len(req.prompt_token_ids)}, gamma={self._gamma}"
        )
        assert len(req.prompt_token_ids) > 1
        logger.info(
            "disagg start prefill. reqid=%s peer=%s computed=%s max=%s",
            req.request_id,
            dinfo.instid,
            dinfo.cached_tokens,
            dinfo.max_tokens,
        )

        assert req.request_id not in self._sending
        state = _SendingReq(self._kvt_signals_per_worker(dinfo))
        self._sending[req.request_id] = state

        # VLLM LIKE min_tokens <= max_tokens
        assert req.sampling_params is not None
        req.sampling_params.min_tokens = 0
        req.sampling_params.max_tokens = 1
        kvtstat = KVTState(dinfo=dinfo, maxtokens=dinfo.max_tokens)
        core_update_params(req, {P_KVT_STATE: kvtstat})
        core_update_params(req, {P_IGNORE_OUTPUTS: True})
        core_add_req(req)

        resp = await self._wait_kvt_state(req.request_id, state)

        respbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", respbuf, 0, PREFILL_RESP, 0)
        self._packenc.encode_into(resp, respbuf, 8)
        struct.pack_into("=I", respbuf, 4, len(respbuf) - 8)
        writer.write(respbuf)
        await writer.drain()
        return

    # disagg thread
    def _mark_send_done(self, ioret: IoRet):
        assert ioret.reqid is not None
        if ioret.source is None:
            ioret.source = self.source_label()
        state = self._sending.get(ioret.reqid)
        if state is None:
            logger.info("send done: no state: ioret=%s", ioret)
            return
        state.try_mark_done(ioret)
        return

    async def _on_abort_reqs(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)

        respbuf = struct.pack("=I", ABORT_REQS_RESP)
        writer.write(respbuf)

        reqs: list[str] = self._dec.decode(reqbuf)
        for req in reqs:
            core_abort_req(req, "pbackend.abort_req", True)
        logger.info("abort reqs=%s", reqs)
        await writer.drain()
        return

    def _dual_req_transfer(
        self, dual_req: DualReq, kvtstate: KVTState, ret: list[PReqMeta]
    ):
        dinfo = kvtstate.dinfo
        new_tokens = kvtstate.maxtokens - dinfo.cached_tokens
        pmeta = PReqMeta(
            reqid=dual_req.req.request_id,
            d_inst_id=dinfo.instid,
            p_block_ids=dual_req.pblkids,
            d_block_ids=dinfo.blkids,
            seen_tokens=dinfo.cached_tokens,
            new_tokens=new_tokens,
            has_last_token=True,
            d_workers_info=dinfo.d_workers_info,
            has_freeze=True,
        )
        ret.append(pmeta)
        return

    def _dual_req_finish(self, reqid: str, ret: list[PReqMeta]):
        dual_req = self._dual_req_done.get(reqid, None)
        if dual_req is None:
            return
        if dual_req.finished:
            assert get_param(dual_req.req, P_KVT_STATE) is None
            return
        dual_req.finished = True
        kvtstate: Optional[KVTState] = get_param(dual_req.req, P_KVT_STATE)
        if kvtstate is None:
            return
        dual_req = self._dual_req_done.pop(reqid)
        if not kvtstate.untouched:
            return
        assert kvtstate.untouched
        kvtstate.untouched = False
        self._dual_req_transfer(dual_req, kvtstate, ret)
        return

    # isdone?
    def _dual_req_get(self, reqid: str) -> tuple[Optional[Request], bool]:
        req = self._dual_req_done.get(reqid, None)
        if req is None:
            sreq = self.get_request(reqid)
            return sreq, False
        if not req.finished:
            return req.req, False
        return req.req, True

    def _migration_bypass(self, reqid: str, kvtstate: KVTState, ret: list[PReqMeta]):
        assert not kvtstate.untouched
        dinfo = kvtstate.dinfo
        new_tokens = kvtstate.maxtokens - dinfo.cached_tokens

        p_block_ids = get_kvblk_ids(self, reqid)
        pmeta = PReqMeta(
            reqid=reqid,
            d_inst_id=dinfo.instid,
            p_block_ids=p_block_ids,
            d_block_ids=dinfo.blkids,
            seen_tokens=dinfo.cached_tokens,
            new_tokens=new_tokens,
            has_last_token=True,
            d_workers_info=dinfo.d_workers_info,
            has_freeze=True,
        )
        assert len(pmeta.p_block_ids) > 0
        ret.append(pmeta)
        return

    # ret: OUT
    def _step_dinfoq(self, ret: list[PReqMeta]) -> set[str]:
        substep_news: set[str] = set()
        while self._dinfoq:
            rdinfo = self._dinfoq.popleft()
            dinfo = rdinfo.dinfo

            req, isdone = self._dual_req_get(rdinfo.reqid)
            if req is None:
                loop = get_hybrid_sched_loop()
                # If we've previously aborted/timed-out this reqid, give D a
                # terminal "gone" answer so it can fail fast and release the
                # loading slot.
                if self._is_gone(rdinfo.reqid):
                    code = CODE_REQGONE
                    reason = "req gone"
                else:
                    code = CODE_REQNOTFOUND
                    reason = "req not found"
                ioret = IoRet(
                    reqid=rdinfo.reqid, ex=CodeError(code, reason)
                )
                loop.call_soon_threadsafe(self._mark_send_done, ioret)
                continue
            assert get_param(req, P_KVT_STATE) is None

            maxcomputed = req.max_computed_tokens()
            maxtokens = min(req.num_tokens, dinfo.max_tokens)
            if maxcomputed is not None:
                maxtokens = min(maxtokens, maxcomputed)
            kvtstat = KVTState(dinfo=dinfo, maxtokens=maxtokens, untouched=False)
            set_param(req, P_KVT_STATE, kvtstat)

            if maxcomputed is not None and maxcomputed <= dinfo.cached_tokens:
                loop = get_hybrid_sched_loop()
                ioret = IoRet(reqid=req.request_id, n=dinfo.cached_tokens)
                loop.call_soon_threadsafe(self._mark_send_done, ioret)
                mark_backend_save_done(req, source=self.source_label())
                self._dual_req_done.pop(req.request_id, None)
                continue

            if not isdone:
                if rdinfo.migration:
                    setup = try_setup_save(req, source=self.source_label())
                    assert setup
                    self._migration_bypass(req.request_id, kvtstat, ret)
                else:
                    kvtstat.untouched = True
                    substep_news.add(req.request_id)
                continue

            if dinfo.max_tokens > req.num_prompt_tokens:
                reason = (
                    f"reqid={req.request_id}, dinfo.max_tokens={dinfo.max_tokens}, "
                    f"prompt_len={req.num_prompt_tokens}, gamma={self._gamma}"
                )
                logger.error("Aborting request due to token mismatch: %s", reason)
                core_abort_req(req.request_id, reason, True)
                continue
            dual_req = self._dual_req_done.pop(req.request_id)
            self._dual_req_transfer(dual_req, kvtstat, ret)
        return substep_news

    def _update_dual_req_done(self,
                          req: Request,
                          islast: bool,
                          pblkids: Optional[list[int]],
                          finished=False):
        if not get_param(req, P_REMOTE_DECODE):
            return
        if not islast:
            return
        if req.request_id in self._dual_req_done:
            # async scheduling~
            return

        if pblkids is None:
            pblkids = get_kvblk_ids(self, req.request_id)
            assert len(pblkids) > 0 and len(pblkids[0]) > 0
        self._dual_req_done[req.request_id] = DualReq(req, pblkids, finished)
        return

    def _check_kvtmeta(self, prevmeta: PReqMeta, curmeta: PReqMeta):
        return True

    def _update_infly_kvt(self, meta: PReqMeta):
        if meta.d_block_ids:
            assert meta.reqid not in self._infly_kvt
            if not meta.has_last_token:
                self._infly_kvt[meta.reqid] = meta
            return

        assert meta.reqid in self._infly_kvt
        if meta.has_last_token:
            prevmeta = self._infly_kvt.pop(meta.reqid)
            assert self._check_kvtmeta(prevmeta, meta)
            return

        prevmeta = self._infly_kvt[meta.reqid]
        assert self._check_kvtmeta(prevmeta, meta)
        self._infly_kvt[meta.reqid] = meta
        return
    def _step_finished_reqs(
        self,
        sout: SchedulerOutput,
        kvtpmeta: KVTPMeta,
    ):
        """Process requests in finished_req_ids.

        Args:
            sout: SchedulerOutput
            kvtpmeta: KVTPMeta that stores the generated metadata.
        """
        if len(self._infly_kvt) <= 0 and len(self._dual_req_done) <= 0:
            # fast path
            return
        for reqid in sout.finished_req_ids:
            meta = self._infly_kvt.pop(reqid, None)
            if meta is None:
                # Metadata produced by _dual_req_finish has has_freeze=True;
                # store it in freeze_metas.
                self._dual_req_finish(reqid, kvtpmeta.freeze_metas)
                continue
            assert reqid not in self._dual_req_done
            # assert R.untouched == False
            assert not meta.has_last_token and meta.new_tokens > 0
            new_seen_tokens = meta.seen_tokens + meta.new_tokens
            new_meta = PReqMeta(
                reqid=meta.reqid,
                d_inst_id="",
                p_block_ids=[],
                d_block_ids=[],
                new_tokens=0,
                has_last_token=True,
                seen_tokens=new_seen_tokens,
                d_workers_info=meta.d_workers_info,
            )
            # Store metadata taken from _infly_kvt in abort_metas.
            kvtpmeta.abort_metas.append(new_meta)
        return

    def _step_aborting(self, sout: HCSchedOutput):
        for areq in sout.hc_aborted_save:
            kvstate: Optional[KVTState] = get_param(areq, P_KVT_STATE)
            if kvstate is not None and not kvstate.untouched:
                continue
            assert areq.request_id not in self._infly_kvt

            self._dual_req_done.pop(areq.request_id, None)
            # Record the abort for future TRANSFER_KV_REQ for this reqid
            self._mark_gone(areq.request_id, "abort_save")
            if kvstate is not None:
                assert kvstate.untouched
                loop = get_hybrid_sched_loop()
                ioret = IoRet(
                    reqid=areq.request_id, ex=CodeError(CODE_INTERNALERROR, "aborted")
                )
                loop.call_soon_threadsafe(self._mark_send_done, ioret)

            mark_backend_save_done(areq, source=self.source_label())

        return

    def _step_stop0(self, sout: HCSchedOutput):
        for req in sout.hc_stop0:
            kvtstate: Optional[KVTState] = get_param(req, P_KVT_STATE)
            assert kvtstate is None

            assert req.request_id not in self._dual_req_done
            self._update_dual_req_done(req, True, None, True)
        return

    def _step_dual_req_done(self):
        now = time.time()
        timeout_s = envs.VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS
        for reqid, dual_req in self._dual_req_done.items():
            if not dual_req.finished:
                continue
            if dual_req.insert_ts + timeout_s > now:
                break
            core_abort_req(reqid, "dual_req_done.timeout", True)
        return

    def _step_sched_req(
        self,
        sout: SchedulerOutput,
        ret: list[PReqMeta],
        substep_news: Optional[set[str]] = None,
    ) -> None:
        substep_mode = substep_news is not None
        kvtstate: Optional[KVTState] = None
        for reqdata in sout.scheduled_new_reqs:
            if substep_mode and reqdata.req_id not in substep_news:
                continue
            req = self.get_request(reqdata.req_id)
            assert req is not None

            new_tokens = sout.num_scheduled_tokens[reqdata.req_id]
            new_tokens += reqdata.num_computed_tokens  # prompt cache
            num_tokens = req.max_computed_tokens()
            if num_tokens is None:
                num_tokens = req.num_prompt_tokens
            has_last_token = new_tokens >= num_tokens
            # same as other blocks handling
            # but don't change the blocks layout in request
            # reqdata came from scheduler, would have null blocks
            # when enable hybrid model prefix cache
            transfer_blk_ids = get_kvblk_ids(
                self,
                reqdata.req_id,
                reqdata.block_ids,
            )
            # here we won't need assert len(p_block_ids)==1,
            # blocks with different attn groups can be easily fit in kvt,
            # when layer sending, just traverse the groups
            # to get which blockid to send and use block id to get offset
            p_block_ids = transfer_blk_ids
            kvtstate = get_param(req, P_KVT_STATE)
            if kvtstate is None:
                assert not substep_mode
                self._update_dual_req_done(req, has_last_token, p_block_ids)
                continue
            d_computed_tokens = kvtstate.dinfo.cached_tokens
            if new_tokens <= d_computed_tokens:
                # no need to send kvcache
                continue

            num_tokens = kvtstate.maxtokens
            orig_has_last_token = has_last_token
            has_last_token = new_tokens >= num_tokens
            if orig_has_last_token:
                assert has_last_token
            if substep_mode:
                assert kvtstate.untouched
                if not has_last_token:
                    continue
            kvtstate.untouched = False
            if has_last_token:
                new_tokens = num_tokens

            d_inst_id = kvtstate.dinfo.instid
            d_blocks_ids = kvtstate.dinfo.blkids
            seen_tokens = d_computed_tokens
            new_tokens -= d_computed_tokens
            kvtmeta = PReqMeta(
                reqid=reqdata.req_id,
                d_inst_id=d_inst_id,
                p_block_ids=p_block_ids,
                d_block_ids=d_blocks_ids,
                new_tokens=new_tokens,
                has_last_token=has_last_token,
                seen_tokens=seen_tokens,
                d_workers_info=kvtstate.dinfo.d_workers_info,
            )
            if substep_mode:
                assert kvtmeta.has_last_token and len(kvtmeta.d_block_ids) > 0
            ret.append(kvtmeta)
            self._update_infly_kvt(kvtmeta)

        req_data = sout.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            if substep_mode and req_id not in substep_news:
                continue
            req = self.get_request(req_id)
            assert req is not None
            num_computed_tokens = req_data.num_computed_tokens[i]
            seen_tokens = num_computed_tokens
            new_tokens = sout.num_scheduled_tokens[req_id]
            end_tokens = seen_tokens + new_tokens
            num_tokens = req.max_computed_tokens()
            if num_tokens is None:
                num_tokens = req.num_prompt_tokens
            has_last_token = end_tokens >= num_tokens

            kvtstate = get_param(req, P_KVT_STATE)
            if kvtstate is None:
                assert not substep_mode
                self._update_dual_req_done(req, has_last_token, None)
                continue
            d_computed_tokens = kvtstate.dinfo.cached_tokens
            if end_tokens <= d_computed_tokens:
                continue

            num_tokens = kvtstate.maxtokens
            orig_has_last_token = has_last_token
            has_last_token = end_tokens >= num_tokens
            if orig_has_last_token:
                assert has_last_token
            if substep_mode:
                assert kvtstate.untouched
                if not has_last_token:
                    continue
            if has_last_token:
                end_tokens = num_tokens
            if kvtstate.untouched:
                seen_tokens = 0

            if seen_tokens <= d_computed_tokens:
                kvtstate.untouched = False

                d_inst_id = kvtstate.dinfo.instid
                seen_tokens = d_computed_tokens
                new_tokens = end_tokens - d_computed_tokens
                d_blocks_ids = kvtstate.dinfo.blkids
                p_block_ids = get_kvblk_ids(self, req.request_id)
                kvtmeta = PReqMeta(
                    reqid=req_id,
                    d_inst_id=d_inst_id,
                    p_block_ids=p_block_ids,
                    d_block_ids=d_blocks_ids,
                    new_tokens=new_tokens,
                    has_last_token=has_last_token,
                    seen_tokens=seen_tokens,
                    d_workers_info=kvtstate.dinfo.d_workers_info,
                )
                if substep_mode:
                    assert kvtmeta.has_last_token and len(kvtmeta.d_block_ids) > 0
                ret.append(kvtmeta)
                self._update_infly_kvt(kvtmeta)
            elif seen_tokens < num_tokens:
                # d_computed_tokens < seen_tokens < num_tokens
                assert not substep_mode
                assert not kvtstate.untouched
                new_tokens = end_tokens - seen_tokens

                kvtmeta = PReqMeta(
                    reqid=req_id,
                    d_inst_id="",
                    p_block_ids=[],
                    d_block_ids=[],
                    new_tokens=new_tokens,
                    has_last_token=has_last_token,
                    seen_tokens=seen_tokens,
                    d_workers_info=[],
                )
                ret.append(kvtmeta)
                self._update_infly_kvt(kvtmeta)

    # ==============================
    # Scheduler-side methods
    # ==============================
    async_get_num_new_matched_tokens = None  # type: ignore

    def get_operations(self, req: Request) -> OperationPlan:
        d_inst_id = get_param(req, P_KVT_STATE)
        if self._cfg.model_config.is_hybrid:
            if len(req.prompt_token_ids) <= self._gamma + 1:
                return 0, 0, (), ()
            # NOTE(llx): Decode will only use n-gamma-1 tokens' kv cache,
            # add these to fit qwen3-next's gdn state
            req.prompt_token_ids = req.prompt_token_ids[:-(self._gamma + 1)]
            req._all_token_ids = req.prompt_token_ids.copy()
            req.num_prompt_tokens = len(req.prompt_token_ids)
            req.all_token_ids = ConstantList(req._all_token_ids)
        if d_inst_id is not None:
            return 0, 1, (), (self.source_label(),)
        if get_param(req, P_REMOTE_DECODE):
            if req.num_prompt_tokens <= 1:
                return 0, 0, (), ()
            return 0, 1, (), (self.source_label(),)
        return 0, 0, (), ()

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        assert isinstance(sout, HCSchedOutput)

        kvtpmeta = KVTPMeta(
            stepid=sout.hc_stepid,
            substepid=sout.hc_substepid,
            sched_tokens=sout.total_num_scheduled_tokens,
            freeze_metas=[],
            abort_metas=[],
            nonfreeze_metas=[],
        )

        self._step_stop0(sout)
        self._step_aborting(sout)
        substep_news = self._step_dinfoq(kvtpmeta.freeze_metas)
        self._step_finished_reqs(sout, kvtpmeta)
        self._step_dual_req_done()

        if sout.hc_parent is None:
            self._step_sched_req(sout, kvtpmeta.nonfreeze_metas)
        elif substep_news and envs.VLLM_ENABLE_BYPASS_SUBSTEP:
            self._step_sched_req(
                sout.hc_parent,
                kvtpmeta.nonfreeze_metas,
                substep_news=substep_news,
            )

        return kvtpmeta

    # ==============================
    # Worker-side methods
    # ==============================

    def _send_error_done_in_loop(self, reqids: list[str]):
        """Schedule sending error send done request in worker loop.

        This method is called when _get_dist fails during bind_backend_metadata.
        It schedules send_error_done_req to run in the worker's asyncio loop,
        ensuring proper cleanup/wakeup of PBackend's waiting requests.
        """
        if not reqids:
            return

        loop = get_hybrid_worker_loop()

        async def do_send():
            try:
                await self._bladkv_cli.send_error_done_req(reqids)
            except Exception:
                logger.exception("_send_error_done_in_loop: failed")

        # Schedule the async send in the worker loop
        asyncio.run_coroutine_threadsafe(do_send(), loop)
        return

    def _build_req_send_batch(
        self,
        reqmetas: list[PReqMeta],
        expect_freeze: bool,
    ) -> tuple[list[bladekv.ReqMeta], list[str]]:
        failed_reqids: list[str] = []
        kvtmetas: list[bladekv.ReqMeta] = []
        for reqm in reqmetas:
            assert reqm.has_freeze == expect_freeze
            assert reqm.has_last_token and len(reqm.d_block_ids) > 0 and reqm.d_inst_id
            distinfos = _get_distinfo(reqm)
            if not distinfos:
                logger.warning(
                    "_build_req_send_batch: _get_distinfo failed req=%s",
                    reqm,
                )
                failed_reqids.append(reqm.reqid)
                continue

            try:
                # P_tp < D_tp fans out to multiple D ranks; one ReqMeta each.
                for dst_inst_name, dst_wid, dst_worker_info in distinfos:
                    kvtmeta = bladekv.ReqMeta()
                    kvtmeta.dst_inst = dst_inst_name
                    kvtmeta.dst_worker = dst_wid
                    kvtmeta.reqid = reqm.reqid
                    kvtmeta.seen_tokens = reqm.seen_tokens
                    kvtmeta.new_tokens = reqm.new_tokens
                    kvtmeta.src_block_ids = reqm.p_block_ids
                    kvtmeta.dst_block_ids = reqm.d_block_ids
                    kvtmeta.dst_worker_info = dst_worker_info
                    kvtmetas.append(kvtmeta)
            except Exception:
                logger.exception(
                    "_start_req_send: bladekv.ReqMeta failed req=%s", reqm
                )
                failed_reqids.append(reqm.reqid)
        return kvtmetas, failed_reqids

    # worker thread/bypass thread
    def _start_req_send(
        self, freeze_metas: list[PReqMeta], stepid: int = 0, substepid: int = 0
    ) -> list[str]:
        """Process freeze_metas and submit to KVT client.

        Args:
            freeze_metas: List of PReqMeta with has_freeze=True
            stepid: Step identifier
            substepid: Substep identifier

        Returns:
            List of request IDs that failed due to invalid TP config.
        """
        if len(freeze_metas) <= 0:
            return []

        kvtmetas, failed_reqids = self._build_req_send_batch(
            freeze_metas, expect_freeze=True
        )

        if kvtmetas:
            self._bladkv_cli.start_req_send(kvtmetas, stepid, substepid)
        return failed_reqids

    # bypass thread
    def _start_send_substep(
        self,
        nonfreeze_metas: list[PReqMeta],
        stepid: int,
        substepid: int,
    ) -> list[str]:
        if len(nonfreeze_metas) <= 0:
            return []

        kvtmetas, failed_reqids = self._build_req_send_batch(
            nonfreeze_metas, expect_freeze=False
        )
        if kvtmetas:
            self._bladkv_cli.start_send_substep(stepid, substepid, kvtmetas)
        return failed_reqids

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for layer_name, _ in kv_caches.items():
            logger.info("Prefill Registering KV Cache layer: %s", layer_name)
        logger.info("Get KV Cache config: %s", self._kv_cache_config)

        # Regenerate inst_id after application template
        if self._naming_cli is None:
            self._inst_id = _get_inst_id(self._cfg, fake_naming=True)

        kv_caches, factory_kwargs, cache_shape = _build_kvt_args(
            self, kv_caches, role_log="Prefill"
        )

        self._bladkv_cli = bladekv.KVTransferClient(
            layers=_flatten_cache(kv_caches),
            **factory_kwargs,
        )
        winfo = bladekv.current_worker_info('client')
        logger.info(
            "register_kv_caches. worker_id=%s winfo=%s kv_cache shape=%s",
            factory_kwargs["worker_id"], winfo, cache_shape
        )
        # self._naming_cli.store(f"worker_{worker_id}", winfo)
        return

    def _bind_prologue(
        self, meta: BackendMeta
    ) -> Optional[tuple[KVTPMeta, list[str]]]:
        """Shared prologue for bind_backend_metadata / bypass_bind.

        Validates the meta and sends freeze_metas. Returns
        (meta, failed_reqids), or None if there is no backend client and the
        caller should return early.
        """
        assert isinstance(meta, KVTPMeta)
        if self._bladkv_cli is None:
            assert not meta
            return None

        failed_reqids: list[str] = []
        # Process freeze_metas with has_freeze=True; _start_req_send asserts it.
        if meta.freeze_metas:
            failed_from_freeze = self._start_req_send(
                meta.freeze_metas, meta.stepid, meta.substepid
            )
            failed_reqids.extend(failed_from_freeze)
        return meta, failed_reqids

    def bind_backend_metadata(self, meta: BackendMeta):
        """Main step (substepid == 0).

        Called via HybridWorker.bind_connector_metadata on the model
        execution thread.
        """
        prologue = self._bind_prologue(meta)
        if prologue is None:
            return
        meta, failed_reqids = prologue

        # Read stepid, substepid, and sched_tokens from the metadata.
        stepid = meta.stepid
        substepid = meta.substepid
        sched_tokens = meta.sched_tokens
        assert substepid == 0, (
            f"bind_backend_metadata is the main step path,"
            f" got substepid={substepid}"
        )

        # Process abort_metas: follow-up sends for finished requests.
        for reqm in meta.abort_metas:
            assert reqm.new_tokens == 0 and reqm.has_last_token
            assert not reqm.has_freeze
            # abort_metas notify workers that requests have finished.
            # Their d_block_ids are empty, so use submit_delta_send.
            self._bladkv_cli.submit_delta_send(
                reqm.reqid,
                seen_tokens=reqm.seen_tokens,
                new_tokens=reqm.new_tokens,
                has_last_token=reqm.has_last_token,
                stepid=stepid,
            )

        # Process nonfreeze_metas with has_freeze=False.
        for reqm in meta.nonfreeze_metas:
            assert not reqm.has_freeze
            if reqm.d_block_ids:
                distinfos = _get_distinfo(reqm)
                if not distinfos:
                    logger.warning(
                        "bind_backend_metadata: _get_distinfo failed req=%s",
                        reqm
                    )
                    failed_reqids.append(reqm.reqid)
                    continue
                try:
                    # P_tp < D_tp fans out to multiple D ranks.
                    for dst_inst_name, dst_wid, dst_worker_info in distinfos:
                        self._bladkv_cli.submit_req_send2(
                            dst_inst_name,
                            dst_wid,
                            reqm.reqid,
                            seen_tokens=reqm.seen_tokens,
                            new_tokens=reqm.new_tokens,
                            has_last_token=reqm.has_last_token,
                            src_block_ids=reqm.p_block_ids,
                            dst_block_ids=reqm.d_block_ids,
                            dst_worker_info=dst_worker_info,
                            stepid=stepid,
                        )
                except Exception:
                    logger.exception(
                        "bind_backend_metadata: submit_req_send2 failed req=%s",
                        reqm
                    )
                    failed_reqids.append(reqm.reqid)
            else:
                self._bladkv_cli.submit_delta_send(
                    reqm.reqid,
                    seen_tokens=reqm.seen_tokens,
                    new_tokens=reqm.new_tokens,
                    has_last_token=reqm.has_last_token,
                    stepid=stepid,
                )

        # Send error done request for failed requests via worker loop
        if failed_reqids:
            self._send_error_done_in_loop(failed_reqids)

        # Main thread: call start_send_step.
        self._bladkv_cli.start_send_step(stepid=stepid, sched_tokens=sched_tokens)
        return

    def bypass_bind(self, meta: BackendMeta):
        """Bypass substep (substepid >= 1).

        Called via HybridWorker._do_bypass_meta on the bypass loop thread.
        Unlike the main step it does not handle abort_metas nor start a send
        step; it only kicks off the substep send.
        """
        prologue = self._bind_prologue(meta)
        if prologue is None:
            return
        meta, failed_reqids = prologue

        stepid = meta.stepid
        substepid = meta.substepid
        assert substepid != 0, (
            f"bypass_bind is the bypass substep path,"
            f" got substepid={substepid}"
        )

        assert not meta.abort_metas
        failed_from_substep = self._start_send_substep(
            meta.nonfreeze_metas,
            stepid,
            substepid,
        )
        failed_reqids.extend(failed_from_substep)
        if failed_reqids:
            self._send_error_done_in_loop(failed_reqids)
        return

    def clear_backend_metadata(self):
        # Main step: flush the send step queued in bind_backend_metadata.
        if self._bladkv_cli is None:
            return
        self._bladkv_cli.flush_send_step()
        return

    def bypass_clear(self):
        # Bypass substep does not own the send step, so it must not flush.
        return


    async_load_kv = None  # type: ignore

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        from vllm.model_executor.models.utils import extract_layer_index
        layer_idx = extract_layer_index(layer_name)
        if self.is_hybrid:
            idx = -1
            if layer_idx in self.hybrid_model_send_layer:
                idx = self.hybrid_model_send_layer.index(layer_idx)
                self._bladkv_cli.record_event(idx, torch.cuda.current_stream())
            # logger.info(f'async_save_kv_layer {layer_idx=} {layer_name=} {idx=}')
        else:
            self._bladkv_cli.record_event(layer_idx, torch.cuda.current_stream())
        return None


class DBackend(HybridBackend):
    SOURCE_LABEL = "kvt"

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._naming_url = vllm_config.kv_transfer_config.get_from_extra_config(
            "naming_url", "fake://"
        )
        if self._naming_url == "fake://":
            self._inst_id = _get_inst_id(vllm_config, fake_naming=True)
            self._naming_cli = None
        else:
            self._inst_id = _get_inst_id(vllm_config)
            self._naming_cli = connect_naming(self._inst_id, self._naming_url)

        if role == KVConnectorRole.WORKER:
            self._loop = get_hybrid_worker_loop()
        else:
            self._loop = get_hybrid_sched_loop()
        self._cfg = vllm_config
        self._kv_cache_config = kv_cache_config
        self._gamma = get_p_node_pop_len(self._cfg) - 1

        if role == KVConnectorRole.WORKER:
            generate_nic_affinity()
            # 32 is default value
            # os.environ.setdefault('ACCL_RX_DEPTH', '32')
            os.environ.setdefault("BLLM_KVTRANS_RESERVE", "4096,128;")
            _set_worker_envs(vllm_config)
        else:
            # _reg_naming(self._naming_cli, cfg)
            ### R/W: disagg thread
            self.my_addr_port = (get_ip(), rpc_port(vllm_config))

            interested_role: Optional[str] = "prefill"
            if (
                vllm_config.kv_transfer_config.get_from_extra_config("backend", None)
                == "kvt+migration"
            ):
                # Role-free backend, interested in all instances
                interested_role = None

            # Connection pool management
            self._conn_mgr = ConnManager(envs.VLLM_PD_CONNMANAGER_CAP)
            self._workers_info: list[str] = []
            # Thread synchronization for workers_info
            self._workers_info_event: Optional[threading.Event] = None
            self._workers_info_lock = threading.Lock()
            # No peer_manager needed if naming_url is fake
            if self._naming_cli is not None:
                self._pmgr: Optional[PeerManager] = PeerManager(
                    self._naming_cli, interested_role, self._conn_mgr
                )
                self._pmgr.start(self._loop)
            else:
                self._pmgr = None
                self._workers_info = [
                    ""
                ] * self._cfg.parallel_config.tensor_parallel_size
                # Initialize event for fake naming case
                self._workers_info_event = threading.Event()

            self._enc = MsgpackEncoder()
            self._enc_inline = MsgpackEncoder(size_threshold=2**62)
            self._packenc = msgspec.msgpack.Encoder()
            self._kvtrespdec = msgspec.msgpack.Decoder(KVTResp)
            # Add necessary data structures for worker registration

            self._strdec = MsgpackDecoder(str)

            # Setup RPC server for worker registration when naming is not available
            if self._naming_cli is None:
                rpcsrv = sched_rpc_server()
                rpcsrv.register_method(_REGISTER_WORKER, self._on_register_worker)
            if self._naming_cli is None and self._workers_info_event is not None:
                # Wait for workers_info to be filled using Event
                logger.info("DBackend waiting for workers_info to be filled")
                self._workers_info_event.wait()  # Block until event is set
                logger.info("DBackend workers_info filled: %s", self._workers_info)

        # Initialize delay list for KVT retry mechanism
        self._delay_s_list = self._generate_delay_list()
        return

    def _generate_delay_list(self) -> list[float]:
        """Generate delay list for KVT retry mechanism based on environment variable.
        """
        max_delay_ms = envs.VLLM_KVT_MAX_DELAY_MS
        head_delays_ms = [1, 3, 7, 11, 17]

        delays_ms: list[float] = []
        total_ms = 0.0
        for d in head_delays_ms:
            if total_ms + d >= max_delay_ms:
                break
            delays_ms.append(d)
            total_ms += d

        # Exponential backoff
        # cap the last delay so the cumulative sum is exactly max_delay_ms.
        interval_ms = 100.0
        while total_ms < max_delay_ms:
            d = min(interval_ms, max_delay_ms - total_ms)
            delays_ms.append(d)
            total_ms += d
            interval_ms *= 2

        return [d / 1000.0 for d in delays_ms]

    # disagg thread, core thread
    def _tp_size(self):
        return self._cfg.parallel_config.tensor_parallel_size

    async def _on_register_worker(self, reader, writer):
        """Handle worker registration request when naming is not available

        Request format:
        +-----+-----------------+
        | len | info            |
        +-----+-----------------+
        len: 4bytes, sizeof(info)
        info: encoded worker info (str) in format "{worker_id}|{winfo}"
        """
        # Read message length
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        infobuf = await reader.readexactly(bodylen)

        # Send response immediately to acknowledge receipt
        respbuf = struct.pack("=I", _REGISTER_WORKER_RESP)
        writer.write(respbuf)

        # Decode worker info: format is "{worker_id}|{winfo}"
        worker_reg_info = self._strdec.decode(infobuf)
        # Parse worker registration info
        parts = worker_reg_info.split(
            "|", 1
        )  # Split only on first "|" since winfo may contain "|"
        if len(parts) != 2:
            logger.error("Invalid worker registration format: %s", worker_reg_info)
            raise RuntimeError("Invalid worker registration format")

        try:
            worker_id = int(parts[0])
            winfo = parts[1]
        except ValueError as e:
            logger.error("Failed to parse worker_id: %s", e)
            await writer.drain()
            return

        # Thread-safe update of workers_info
        with self._workers_info_lock:
            # Initialize workers_info list if needed
            if len(self._workers_info) == 0:
                self._workers_info = [""] * self._tp_size()

            # Store winfo directly at worker_id position
            if worker_id < len(self._workers_info):
                self._workers_info[worker_id] = winfo
                logger.info(
                    "DBackend registered worker %d with winfo: %s", worker_id, winfo
                )

                # Check if all workers are registered
                if (
                    not any([winfo == "" for winfo in self._workers_info])
                    and self._workers_info_event is not None
                    and not self._workers_info_event.is_set()
                ):
                    # All workers registered, set the event
                    self._workers_info_event.set()
            else:
                logger.error(
                    "Worker ID %d exceeds tp_size %d", worker_id, self._tp_size()
                )

        await writer.drain()
        return

    def get_peer(
        self,
        hint: Optional[tuple[str, int]] = None,
        exclude: Optional[tuple[str, int]] = None,
    ) -> Optional[tuple[str, int]]:
        """Get peer's addr and port"""
        if hint is not None:
            return hint

        if self._pmgr is None:
            # No peer_manager in fake naming case, return None
            return None

        return self._pmgr.get_peer(exclude)

    async def _kvt_rpc(
        self, req: Request, msgbuf, peer_hint: Optional[tuple[str, int]]
    ) -> KVTResp:
        reqid = req.request_id
        peer_addr_port: Optional[tuple[str, int]] = None
        for retry in range(2):
            try:
                peer_addr_port = self.get_peer(
                    hint=peer_hint, exclude=self.my_addr_port
                )
                logger.info(
                    "disagg start kvt. reqid=%s retry=%s hint=%s exclude=%s peer=%s msglen=%s",  # noqa: E501
                    reqid,
                    retry,
                    peer_hint,
                    self.my_addr_port,
                    peer_addr_port,
                    len(msgbuf),
                )
                if peer_addr_port is None:
                    raise RuntimeError("no p")
                # Save peer address info for subsequent abort operations
                set_param(req, D_PID, f"{peer_addr_port[0]}:{peer_addr_port[1]}")
                _check_req_aborted(req)
                pconn = await self._conn_mgr.acquire_conn(peer_addr_port, retry > 0)

                pconn[1].write(msgbuf)
                await pconn[1].drain()
                respbuf = await pconn[0].readexactly(4 + 4)
                head, bodylen = struct.unpack("=II", respbuf)
                if head != PREFILL_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                respbuf = await pconn[0].readexactly(bodylen)
                kvtresp: KVTResp = self._kvtrespdec.decode(respbuf)

                self._conn_mgr.release_conn(peer_addr_port, pconn)
                return kvtresp
            except asyncio.IncompleteReadError:
                logger.exception(
                    "do kvt failed. peer=%s retry=%s hint=%s",
                    peer_addr_port,
                    retry,
                    peer_hint,
                )
            await asyncio.sleep(0)
        raise RuntimeError("kvt rpc failed")

    def naming_cli(self):
        return self._naming_cli

    async def _prefill_rpc(self, req: Request, blocks: KVCacheBlocks) -> int:
        corereq = req2corereq(req)
        blockids = get_kvblk_ids(self, req.request_id)
        sched_discard_zero_block_ids(
            block_id for group in blockids for block_id in group)
        # logger.info("req_id: %s, D original blks: %s, concated_blk_ids: %s",
        #             req.request_id, blockids, concated_blk_ids)

        dprank = self._cfg.parallel_config.data_parallel_rank
        tpsize = self._cfg.parallel_config.tensor_parallel_size
        inst_id = f"{self._inst_id}|{dprank}|{tpsize}"
        kvtdinfo = KVTDInfo(
            instid=inst_id,
            blkids=blockids,
            cached_tokens=req.num_computed_tokens,
            max_tokens=req.num_prompt_tokens -(self._gamma + 1),
            d_workers_info=self._workers_info,
        )
        core_update_params(
            corereq,
            {
                P_KVTD_INFO: kvtdinfo,
            },
        )
        reqid = req.request_id
        promptlen = req.num_prompt_tokens

        peer_hint: Optional[tuple[str, int]] = None
        remote_host = get_param(req, "remote_host")
        remote_port = get_param(req, "remote_port")
        if remote_host is not None and remote_port is not None:
            # eas:  "remote_port": INT
            # dual_req: "remote_port": "INT"
            remote_port = int(remote_port)
            peer_hint = (remote_host, remote_port)

        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, PREFILL_REQ, 0)
        self._enc_inline.encode_into(corereq, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        start_ts = time.monotonic()
        kvtresp = await self._kvt_rpc(req, msgbuf, peer_hint)
        end_ts = time.monotonic()
        dur_ms = (end_ts - start_ts) * 1000
        logger.info(
            "disagg end prefill. reqid=%s prompt=%s computed=%s dur_ms=%s kvtresp=%s",
            reqid,
            promptlen,
            req.num_computed_tokens,
            dur_ms,
            kvtresp,
        )
        if kvtresp.code != CODE_OK:
            raise RuntimeError("bad resp")
        return kvtresp.cached

    async def _dual_req_prefill_rpc(self, req: Request, blocks: KVCacheBlocks) -> int:
        blockids = get_kvblk_ids(self, req.request_id)
        sched_discard_zero_block_ids(
            block_id for group in blockids for block_id in group)
        dprank = self._cfg.parallel_config.data_parallel_rank
        tpsize = self._cfg.parallel_config.tensor_parallel_size
        d_workers_info = self._workers_info

        # fault inject for debug
        # fault_inject_types = get_param(req, "__dbg_fault_inject_type", [])
        # if fault_inject_types:
        #     if "bad_tpsize" in fault_inject_types:
        #         tpsize = 100
        #     if "bad_workerinfo" in fault_inject_types:
        #         # copy to avoid modifying self._workers_info
        #         d_workers_info = self._workers_info.copy()
        #         d_workers_info[0] = d_workers_info[0] + ",bad,workerinfo"

        inst_id = f"{self._inst_id}|{dprank}|{tpsize}"
        kvtdinfo = KVTDInfo(
            instid=inst_id,
            blkids=blockids,
            cached_tokens=req.num_computed_tokens,
            max_tokens=req.num_prompt_tokens - (self._gamma + 1),
            d_workers_info=d_workers_info,
        )
        reqid = req.request_id
        promptlen = req.num_prompt_tokens
        kvtreq = RKVTDInfo(reqid=reqid, dinfo=kvtdinfo)
        remote_host = get_param(req, "remote_host")
        assert remote_host is not None
        remote_port = get_param(req, "remote_port")
        remote_port = int(remote_port)
        peer_hint = (remote_host, remote_port)

        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, TRANSFER_KV_REQ, 0)
        self._packenc.encode_into(kvtreq, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        for retry in range(len(self._delay_s_list)):
            _check_req_aborted(req)
            start_ts = time.monotonic()
            kvtresp = await self._kvt_rpc(req, msgbuf, peer_hint)
            end_ts = time.monotonic()
            dur_ms = (end_ts - start_ts) * 1000
            logger.info(
                "disagg end dual_req prefill. reqid=%s prompt=%s computed=%s dur_ms=%s kvtresp=%s retry=%s",  # noqa: E501
                reqid,
                promptlen,
                req.num_computed_tokens,
                dur_ms,
                kvtresp,
                retry,
            )
            if kvtresp.code == CODE_OK:
                return kvtresp.cached
            if kvtresp.code == CODE_REQNOTFOUND:
                await asyncio.sleep(self._delay_s_list[retry])
                continue
            # CODE_REQGONE is terminal: P has explicitly told us this reqid
            # used to exist but its dual-request KV has been aborted or timed out.
            # Stop retrying and surface the failure so the loading slot can
            # be released immediately instead of looping for minutes/hours.
            if kvtresp.code == CODE_REQGONE:
                logger.warning(
                    "disagg kv gone on P. reqid=%s peer=%s retry=%s",
                    reqid, peer_hint, retry,
                )
                raise RuntimeError(
                    f"kv gone on P "
                    f"(reqid={reqid}, peer={peer_hint}, retry={retry})"
                )
            raise RuntimeError("bad resp")
        raise RuntimeError("bad 404 resp")

    async def _abort_rpc(self, addr_port_str: str, reqids: list[str]):
        # addr_port_str format is "addr:port"
        addr, port_str = addr_port_str.split(":")
        port = int(port_str)
        addr_port = (addr, port)

        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, ABORT_REQS_REQ, 0)
        self._packenc.encode_into(reqids, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        ok = False
        for retry in range(2):
            try:
                logger.info(
                    "abort rpc start. peer=%s reqids=%s retry=%s msglen=%s",
                    addr_port_str,
                    reqids,
                    retry,
                    len(msgbuf),
                )
                pconn = await self._conn_mgr.acquire_conn(addr_port, retry > 0)
                pconn[1].write(msgbuf)
                await pconn[1].drain()
                respbuf = await pconn[0].readexactly(4)
                (head,) = struct.unpack("=I", respbuf)
                if head != ABORT_REQS_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                self._conn_mgr.release_conn(addr_port, pconn)
                ok = True
                break
            except Exception:
                logger.exception(
                    "abort rpc failed. peer=%s reqids=%s", addr_port_str, reqids
                )
            await asyncio.sleep(0)
        if ok:
            logger.info("abort rpc end. peer=%s", addr_port_str)
        else:
            logger.error("abort rpc failed. peer=%s reqids=%s", addr_port_str, reqids)
        return

    @kill_me_if_exception
    async def _abort_prefill(self, reqs: list[Request]):
        peer_reqids: dict[str, list[str]] = defaultdict(list)
        for req in reqs:
            peer_addr_port: Optional[str] = get_param(req, D_PID, None)
            if peer_addr_port is None:
                logger.info("abort prefill:reqid=%s peer=None", req.request_id)
                continue
            peer_reqids[peer_addr_port].append(req.request_id)

        tasks = []
        for peer_addr_port, reqids in peer_reqids.items():
            tsk = asyncio.create_task(self._abort_rpc(peer_addr_port, reqids))
            tasks.append(tsk)
        await asyncio.gather(*tasks)
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    async def async_get_num_new_matched_tokens(
        self, req: Request, local_tokens: int
    ) -> int:
        block_size = self._cfg.cache_config.block_size
        assert local_tokens % block_size == 0
        assert local_tokens <= req.num_prompt_tokens
        left_tokens = req.num_prompt_tokens - local_tokens

        gamma = 0
        if (
            self._cfg.speculative_config
            and self._cfg.speculative_config.num_speculative_tokens
        ):
            gamma = self._cfg.speculative_config.num_speculative_tokens
        graph_query_len = gamma + 1

        if left_tokens <= graph_query_len:
            # prefill locally
            return 0

        return left_tokens - graph_query_len

    async def async_update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, rmt_tokens: int
    ) -> Optional[IoRet]:
        if rmt_tokens <= 0:
            return None

        ioret = IoRet(n=0)
        cached = 0
        try:
            if get_param(request, D_REMOTE_PREFILL):
                cached = await self._dual_req_prefill_rpc(request, blocks)
            else:
                cached = await self._prefill_rpc(request, blocks)
            ioret.n = cached - request.num_computed_tokens
            # TODO: Refactor this workaround.
            ioret.n = min(ioret.n, rmt_tokens)
        except Exception as e:
            logger.exception("load fail. req=%s", request.request_id)
            ioret.ex = e
        return ioret

    # return: (load_count, save_count, load_sources, save_sources)
    def get_operations(self, req: Request) -> OperationPlan:
        # Here, we first determine whether the request's Prefill
        # should be executed locally, based on information such as
        # the status of the P node in naming,
        # the current status of the D node,
        # and the characteristics of the request itself.
        if req.num_prompt_tokens <= 1:
            return 0, 0, (), ()

        if get_param(req, D_REMOTE_PREFILL):
            assert get_param(req, "remote_host") is not None
            assert get_param(req, "remote_port") is not None
            return 1, 0, (self.source_label(),), ()

        disagg = get_param(req, D_DISAGG, True)
        if not disagg:
            return 0, 0, (), ()

        return 1, 0, (self.source_label(),), ()

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        assert isinstance(sout, HCSchedOutput)
        if len(sout.hc_aborted_load) > 0:
            coro = self._abort_prefill(sout.hc_aborted_load)
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        return BackendMeta()

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for layer_name, _ in kv_caches.items():
            logger.info("Decode Registering KV Cache layer: %s", layer_name)
        logger.info("Get KV Cache config: %s", self._kv_cache_config)

        # Regenerate inst_id after application template
        if self._naming_cli is None:
            self._inst_id = _get_inst_id(self._cfg, fake_naming=True)

        kv_caches, factory_kwargs, cache_shape = _build_kvt_args(
            self, kv_caches, role_log="Decode"
        )

        worker_id = factory_kwargs["worker_id"]
        rank = factory_kwargs["worker_tp_rank"]
        logger.info(
            "[DBackend] Creating KVTransferServer: inst_id=%s, tp_size=%s, "
            "worker_id=%s, rank=%s, block_bytes=%s, token_bytes=%s, "
            "naming_url=%s, protocol=%s, num_layers=%d",
            self._inst_id, factory_kwargs["tp_size"],
            worker_id, rank,
            factory_kwargs["block_bytes"], factory_kwargs["token_bytes"],
            self._naming_url, factory_kwargs["protocols"][0], len(kv_caches),
        )
        self._bladkv_srv = bladekv.KVTransferServer(
            layers=_flatten_cache(kv_caches),
            **factory_kwargs,
        )

        logger.info("[DBackend] KVTransferServer created successfully")
        winfo = bladekv.current_worker_info('server')
        logger.info(
            "register_kv_caches. worker_id=%s winfo=%s, kv_cache shape=%s",
            worker_id, winfo, cache_shape
        )
        if self._naming_cli:
            self._naming_cli.store(f"worker_{worker_id}", winfo)
        else:
            # When naming is not available, register worker info via RPC to scheduler
            asyncio.run_coroutine_threadsafe(
                self._register_worker_rpc(rank, winfo), self._loop)
        return

    async def _register_worker_rpc(self, worker_tp_rank: int, winfo: str):
        """Send worker registration RPC to scheduler when naming is not available"""
        core_ip = scheduler_rpc_host(self._cfg)
        scheduler_port = rpc_port(self._cfg)

        # Send worker_id and winfo directly
        worker_reg_info = f"{worker_tp_rank}|{winfo}"

        # Encode worker info
        worker_info_encoded = msgspec.msgpack.encode(worker_reg_info)

        # Create RPC message
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, _REGISTER_WORKER, len(worker_info_encoded))
        msgbuf.extend(worker_info_encoded)

        # D Scheduler RPC server might not be ready yet
        await asyncio.sleep(2)
        last_log_time = 0.0
        while True:
            try:
                # Send RPC to scheduler and wait for response
                reader, writer = await asyncio.open_connection(core_ip, scheduler_port)
                writer.write(msgbuf)
                await writer.drain()

                # Wait for response to confirm successful registration with timeout
                try:
                    respbuf = await asyncio.wait_for(reader.readexactly(4), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.info("Timeout waiting for scheduler response")
                    raise

                (head,) = struct.unpack("=I", respbuf)
                if head != _REGISTER_WORKER_RESP:
                    raise RuntimeError(
                        f"Invalid response from scheduler: "
                        f"expected {_REGISTER_WORKER_RESP}, got {head}"
                    )

                writer.close()
                await writer.wait_closed()

                logger.info(
                    "DBackend worker %d registered via RPC successfully： %s",
                    worker_tp_rank,
                    winfo,
                )
                break

            except Exception as e:
                if isinstance(e, RuntimeError):
                    # error response code
                    raise e
                if time.time() - last_log_time > envs.VLLM_LOG_STATS_INTERVAL:
                    logger.warning(
                        "Failed to register worker %d via RPC: %s", worker_tp_rank, e
                    )
                    last_log_time = time.time()
                await asyncio.sleep(0.1)
                continue

    async_load_kv = None  # type: ignore


reg_naming = _reg_naming
get_inst_id = _get_inst_id
