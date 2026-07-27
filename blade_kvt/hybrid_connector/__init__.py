import asyncio
import contextlib
import json
import pickle
import struct
import time
from collections import Counter, defaultdict, deque
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
from vllm.v1.kv_cache_interface import MambaSpec

try:
    from vllm.v1.metrics.stats import HybridConnectorStats
except ImportError:
    @dataclass
    class HybridConnectorStats:
        """Compatibility stats type for vLLM versions without native support."""

        kv_cache_usage: float
        pre_scheduler_load_kv_cache_usage: float
        post_scheduler_save_kv_cache_usage: float
        saving_status: dict[str, int]
        loading_status: dict[str, int]
        saving_ready_to_teardown_count: int
        load_done_count: int
        load_failed_count: int
        load_latency_ms_sum: float
        load_latency_ms_max: float
        save_done_count: int
        save_failed_count: int
        save_latency_ms_sum: float
        save_latency_ms_max: float
        save_done_count_by_source: dict[str, int] = field(default_factory=dict)
        save_failed_count_by_source: dict[str, int] = field(default_factory=dict)
        save_latency_ms_sum_by_source: dict[str, float] = field(default_factory=dict)
        save_latency_ms_max_by_source: dict[str, float] = field(default_factory=dict)

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
    sched_acquire_blocks,
    sched_add_req,
    sched_allocate_slots,
    sched_cache_blocks,
    sched_finish_req,
    sched_free_blocks,
    sched_get_blocks,
    sched_get_req,
    sched_rpc_server,
    sched_rpc_server_port,
    scheduler_rpc_host,
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

# Save-side IoRet stash read by backend.async_cleanup to branch seal vs discard.
HB_SAVE_IORET = "_HybridBackend_SAVE_IORET"

# Operation source labels for status logs, e.g. ("kvt", "kvs") in KVSP.
HB_LOAD_SOURCES = "_HybridBackend_LOAD_SOURCES"
HB_SAVE_SOURCES = "_HybridBackend_SAVE_SOURCES"

# value: int
PREALLOC_KEY = "_hbprealloc"

# value: int
CLEANUP_RC_KEY = "_HybridBackend_cleanup_rc"

# value: bool
ADD_REQ_LOGGED_KEY = "_hb_add_req_logged"


OperationPlan = tuple[int, int, tuple[str, ...], tuple[str, ...]]


def set_operation_sources(
    req: Request,
    load_sources: tuple[str, ...] = (),
    save_sources: tuple[str, ...] = (),
) -> None:
    if load_sources:
        set_param(req, HB_LOAD_SOURCES, tuple(load_sources))
    if save_sources:
        set_param(req, HB_SAVE_SOURCES, tuple(save_sources))


def merge_operation_plans(*plans: OperationPlan) -> OperationPlan:
    load_count = sum(plan[0] for plan in plans)
    save_count = sum(plan[1] for plan in plans)
    load_sources = tuple(
        source for _, _, sources, _ in plans for source in sources
    )
    save_sources = tuple(
        source for _, _, _, sources in plans for source in sources
    )
    return load_count, save_count, load_sources, save_sources


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
    SOURCE_LABEL: str | None = None

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
        self._enable_prefix_caching = \
            vllm_config.cache_config.enable_prefix_caching
        self.has_null_blk = (
            self._enable_prefix_caching
            and self.is_hybrid
            and vllm_config.cache_config.mamba_cache_mode == "light"
        )
        if self.is_hybrid:
            assert (
                not self._enable_prefix_caching
                or vllm_config.cache_config.mamba_cache_mode == "light"
            ), "Only support light mamba cache mode in HybridConnector"
        self.hybrid_model_send_layer = []

        # Parse KV cache config to get layer group info for validation
        self._num_gdn_layers = 0
        self._num_ple_layers = 0
        self._ple_group_index: Optional[int] = None
        self._has_indexer_cache = False
        self._has_ple_cache = False
        self._num_kv_cache_groups = 0
        self._group_types: list[str] = []  # Track the type of each group
        if kv_cache_config is not None and self.is_hybrid:
            self._parse_kv_cache_config(kv_cache_config)

    def _parse_kv_cache_config(self, kv_cache_config: KVCacheConfig):
        """Parse kv_cache_config to extract layer group information.

        For hybrid models, the KV cache groups typically consist of:
        - num_gdn_layers GDN (linear attention) block groups
        - 1 indexer cache block group (if present)
        - 1 attention cache block group
        - 1 PLE short-conv cache block group (if present)

        Expected order: [GDN groups...] [indexer group (optional)] [attn group]
        [PLE group (optional)]

        This info is used to validate block IDs in handle_hybrid_blocks.
        """
        if not kv_cache_config.kv_cache_groups:
            return
        self._num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self._group_types = []
        for idx, group in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = group.layer_names
            spec = group.kv_cache_spec
            is_gdn = any("linear_attn" in n for n in layer_names)
            is_indexer = any("indexer" in n for n in layer_names)
            is_ple = (
                isinstance(spec, MambaSpec)
                and spec.mamba_type == "short_conv"
            ) or any(".ple" in n for n in layer_names)

            if is_gdn:
                group_type = "gdn"
            elif is_indexer:
                group_type = "indexer"
            elif is_ple:
                group_type = "ple"
            else:
                group_type = "attn"

            self._group_types.append(group_type)

        # num_gdn_layers is determined by kv_cache_groups, which defines block order
        self._num_gdn_layers = self._group_types.count("gdn")
        self._num_ple_layers = self._group_types.count("ple")
        self._ple_group_index = (
            self._group_types.index("ple") if self._num_ple_layers else None
        )

        # Validate group ordering: [gdn, ..., gdn, indexer?, attn, ple?]
        self._validate_group_ordering()

        self._has_indexer_cache = "indexer" in self._group_types
        self._has_ple_cache = "ple" in self._group_types
        logger.info(
            "HybridBackend parsed kv_cache_config: "
            "num_gdn_layers=%s, num_ple_layers=%s, "
            "has_indexer_cache=%s, has_ple_cache=%s, "
            "num_kv_cache_groups=%s, group_types=%s",
            self._num_gdn_layers,
            self._num_ple_layers,
            self._has_indexer_cache,
            self._has_ple_cache,
            self._num_kv_cache_groups,
            self._group_types,
        )

    def _validate_group_ordering(self):
        """Validate that KV cache groups follow the expected order.
        Expected order: [GDN groups...] [indexer?] [attn] [ple?]
        """
        if not self._group_types:
            return

        gdn_indices = [i for i, t in enumerate(self._group_types) if t == "gdn"]
        indexer_indices = [i for i, t in enumerate(self._group_types) if t == "indexer"]
        attn_indices = [i for i, t in enumerate(self._group_types) if t == "attn"]
        ple_indices = [i for i, t in enumerate(self._group_types) if t == "ple"]

        # Validate GDN groups: should be at the beginning and consecutive
        if gdn_indices:
            assert gdn_indices[0] == 0, (
                f"GDN groups should start at index 0, but first GDN is at "
                f"index {gdn_indices[0]}. group_types={self._group_types}"
            )
            # Check GDN groups are consecutive
            expected_gdn_sequence = list(range(len(gdn_indices)))
            assert gdn_indices == expected_gdn_sequence, (
                f"GDN groups should be consecutive at the beginning. "
                f"Found GDN indices: {gdn_indices}, expected: {expected_gdn_sequence}. "
                f"group_types={self._group_types}"
            )

        # Validate indexer group: should be at most one, and after GDN groups
        if indexer_indices:
            assert len(indexer_indices) == 1, (
                f"Expected at most one indexer group, "
                f"but found {len(indexer_indices)}. "
                f"group_types={self._group_types}"
            )
            indexer_idx = indexer_indices[0]
            # Indexer should come after all GDN groups
            if gdn_indices:
                assert indexer_idx == len(gdn_indices), (
                    f"Indexer group should come after all GDN groups. "
                    f"Found indexer at index {indexer_idx}, "
                    f"but have {len(gdn_indices)} "
                    f"GDN groups. group_types={self._group_types}"
                )
            else:
                assert indexer_idx == 0, (
                    f"Indexer group should be at index 0 when there are no GDN groups. "
                    f"Found indexer at index {indexer_idx}. "
                    f"group_types={self._group_types}"
                )

        # Validate attn group: should be after GDN/indexer and before PLE.
        if attn_indices:
            expected_attn_start = len(gdn_indices) + (1 if indexer_indices else 0)
            assert attn_indices[0] == expected_attn_start, (
                f"Attention group should start at index {expected_attn_start} "
                f"(after GDN and indexer groups). "
                f"Found attn at index {attn_indices[0]}. "
                f"group_types={self._group_types}"
            )
            # Should be exactly one attn group
            assert len(attn_indices) == 1, (
                f"Expected exactly one attention group, but found {len(attn_indices)}. "
                f"group_types={self._group_types}"
            )

        if ple_indices:
            assert len(ple_indices) == 1, (
                f"Expected at most one PLE group, but found "
                f"{len(ple_indices)}. group_types={self._group_types}"
            )
            assert attn_indices, (
                f"PLE group requires an attention group before it. "
                f"group_types={self._group_types}"
            )
            expected_ple_idx = attn_indices[0] + 1
            assert ple_indices[0] == expected_ple_idx, (
                f"PLE group should be immediately after attention group at "
                f"index {expected_ple_idx}, found {ple_indices[0]}. "
                f"group_types={self._group_types}"
            )

        # Validate total count
        expected_total = (len(gdn_indices) + len(indexer_indices)
                          + len(attn_indices) + len(ple_indices))
        assert expected_total == len(self._group_types), (
            f"Group count mismatch. Expected {expected_total} groups "
            f"({len(gdn_indices)} GDN + {len(indexer_indices)} indexer + "
            f"{len(attn_indices)} attn + {len(ple_indices)} ple), "
            f"but found {len(self._group_types)} groups. "
            f"group_types={self._group_types}"
        )

    def get_request(self, request_id: str) -> Optional[Request]:
        return sched_get_req(request_id)

    # ==============================
    # Worker-side methods
    # ==============================

    # worker thread
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        return

    # worker thread/disagg thread (main step)
    # bind_backend_metadata([R]) happen before async_load_kv(R)
    def bind_backend_metadata(self, meta: BackendMeta):
        return

    # worker thread/disagg thread (main step)
    def clear_backend_metadata(self):
        return

    # bypass loop thread (bypass substep)
    def bypass_bind(self, meta: BackendMeta):
        self.bind_backend_metadata(meta)

    # bypass loop thread (bypass substep)
    def bypass_clear(self):
        self.clear_backend_metadata()

    # disaggw thread
    async def async_load_kv(self, m: BackendMeta) -> AsyncGenerator[IoRet, None]:
        raise NotImplementedError()
        yield "x"  # make vscode happy

    # worker thread
    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        return None

    def source_label(self) -> str:
        assert self.SOURCE_LABEL is not None, (
            f"{type(self).__name__} must declare SOURCE_LABEL"
        )
        return self.SOURCE_LABEL

    # worker thread
    def save_done_source(self) -> str | None:
        return self.source_label()

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
    # return: (load_count, save_count, load_sources, save_sources)
    # load_count > 0 means need async load.
    # load_count = 0 means async_get_num_new_matched_tokens(R) must return 0.
    # save_count > 0 means need async save.
    # count is useful when load from/save to multiple srcs/dsts.
    def get_operations(self, req: Request) -> OperationPlan:
        return 0, 0, (), ()

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
    # deque append/pop operations are thread-safe in CPython.
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
    # stepid identifies the current step and starts at 1024.
    stepid: int = 0
    # substepid identifies the current substep. The main step uses 0 and
    # bypass substeps start at 1.
    substepid: int = 0


def rpc_port(cfg: VllmConfig):
    return sched_rpc_server_port(cfg)


class _LoadingReq(IoState):
    # Loading currently has one completion source per request. HB_LOAD_SOURCES
    # is used for status logging only; if a future backend loads from multiple
    # sources concurrently, track expected_sources here and require load_done
    # producers to set IoRet.source, mirroring _SavingReq.
    def __init__(self, req: Request):
        super().__init__()
        self._req = req
        self.start_time = time.monotonic()
        return


class _SavingReq(IoState):
    def __init__(
        self,
        req: Request,
        kvblks: KVCacheBlocks,
        save_count: int = 1,
        expected_sources: tuple[str, ...] = (),
    ):
        super().__init__(signals_per_worker=save_count)
        self.expected_sources = expected_sources
        self._req = req
        self.kvblks = kvblks
        self.start_time = time.monotonic()
        # source -> time when all TP workers have reported that source done.
        self.source_done_times: dict[str, float] = {}
        self.source_failed: dict[str, bool] = {}
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

# value: bool – save-specific abort forwarded flag, separate from _ABORTED
# to avoid cross-contamination when a request is in both _loading and _saving.
_SAVE_ABORTED = "hbreqsaveaborted"

# PD finish/save coordination state (stored on req via kv_transfer_params)
_PD_FINISH_REASON = "_pd_finish_reason"
_PD_STOP_REASON = "_pd_stop_reason"
_PD_CLIENT_INDEX = "_pd_client_index"
_PD_SAVED = "_pd_saved"

_GET_BYPASS_HANDLE = 0x20181226
_GET_BYPASS_HANDLE_RESP = 0x20181227

def req_aborted(req: Request) -> bool:
    return get_param(req, _ABORTED, False)


# core thread
def has_setup_save(req: Request) -> bool:
    return bool(get_param(req, _SAVE_PREPARED))


# core thread
def try_setup_save(req: Request, source: str) -> bool:
    global _g_scheduler
    if has_setup_save(req):
        return False
    assert _g_scheduler is not None

    set_operation_sources(req, save_sources=(source,))
    kvblks = sched_get_blocks(req.request_id)
    _g_scheduler._setup_save(req, kvblks)
    assert has_setup_save(req)
    return True


# core thread
def mark_backend_save_done(req: Request, source: str):
    global _g_scheduler
    assert _g_scheduler is not None
    assert source is not None, f"missing save done source for reqid={req.request_id}"
    tpsize = _g_scheduler._tp_size()
    loop = get_hybrid_sched_loop()
    for rank in range(tpsize):
        ioret = IoRet(reqid=req.request_id, source=source)
        asyncio.run_coroutine_threadsafe(
            _g_scheduler._do_save_done(rank, ioret), loop)


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
    elif backend == "v6d_object":
        from .v6d_object_backend import V6dObjectBackend
        return V6dObjectBackend
    elif backend == "v6d_object+kvt" or backend == "kvt+v6d_object":
        if not cfg.kv_transfer_config.is_kv_producer:
            raise ValueError(
                "For v6d_object+kvt and kvt+v6d_object backends, kv_role must "
                "be 'kv_producer' or 'kv_both' in kv_transfer_config because "
                "the KVT half is producer-only (PBackend)."
            )
        from .v6d_object_kvt_backend import V6dObjectKVTBackend
        return V6dObjectKVTBackend

    raise ValueError(
        f"Unknown backend: {backend}. "
        "Supported backends are 'local_file', 'kvt', 'vineyard', 'kvs', "
        "'mooncake', 'v6d_object', and 'v6d_object+kvt'.")


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
        self._waiting: deque[tuple[Request, int, int]] = deque()
        self._loaded: deque[Request] = deque()
        self._saved: deque[str] = deque()
        self._prepared: deque[str] = deque()
        self._aborting: deque[AbortReq] = deque()
        ### R/W: core thread
        self._abortmeta_load: list[Request] = []
        self._abortmeta_save: list[Request] = []
        self._stop0: list[Request] = []

        ### R/W: core thread - stepid/substepid management
        # stepid starts at 1024 and increments for every new step.
        self._stepid: int = 1024
        # substepid resets to 1 for each new step; bypass substeps start at 1.
        self._substepid: int = 1

        ### W: core thread. R: disagg thread
        self._saving: dict[str, _SavingReq] = dict()
        self._loading: dict[str, _LoadingReq] = dict()
        # Extra KV accounting is mutated only from the core scheduler thread.
        self._extra_kv_block_ids: dict[tuple[str, str], set[int]] = {}
        self._extra_kv_block_refcounts: Counter[int] = Counter()
        self._extra_kv_block_refcounts_by_kind: dict[str, Counter[int]] = (
            defaultdict(Counter)
        )
        self._reset_lifecycle_stats()

        self._reqs_ts: dict[str, dict[str, float]] = dict()
        ### R/W: disagg thread
        self._iodonedec = MsgpackDecoder(IoDoneReqs)

        BackendCls = _get_backend_cls(vllm_config)
        self._backend = BackendCls(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

        return

    @property
    def pending_requests(self) -> list[Request]:
        """Requests held by hybrid connector but not yet visible to scheduler."""
        reqs = [req for req, _, _ in self._waiting]
        reqs.extend(lr._req for lr in self._loading.values())
        return reqs

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

    @staticmethod
    def _kv_block_ids(kvblks: KVCacheBlocks) -> set[int]:
        return {
            block.block_id
            for group in kvblks.blocks
            for block in group
            if not block.is_null
        }

    def _track_extra_kv_blocks(
        self, kind: str, reqid: str, kvblks: KVCacheBlocks
    ) -> None:
        key = (kind, reqid)
        self._untrack_extra_kv_blocks(kind, reqid)
        block_ids = self._kv_block_ids(kvblks)
        if not block_ids:
            return
        self._extra_kv_block_ids[key] = block_ids
        self._extra_kv_block_refcounts.update(block_ids)
        self._extra_kv_block_refcounts_by_kind[kind].update(block_ids)

    def _untrack_extra_kv_blocks(self, kind: str, reqid: str) -> None:
        block_ids = self._extra_kv_block_ids.pop((kind, reqid), None)
        if not block_ids:
            return
        for block_id in block_ids:
            self._extra_kv_block_refcounts[block_id] -= 1
            if self._extra_kv_block_refcounts[block_id] <= 0:
                del self._extra_kv_block_refcounts[block_id]
            kind_refcounts = self._extra_kv_block_refcounts_by_kind[kind]
            kind_refcounts[block_id] -= 1
            if kind_refcounts[block_id] <= 0:
                del kind_refcounts[block_id]

    def _reset_lifecycle_stats(self) -> None:
        self._lifecycle_load_done_count = 0
        self._lifecycle_load_failed_count = 0
        self._lifecycle_load_latency_ms_sum = 0.0
        self._lifecycle_load_latency_ms_max = 0.0
        self._lifecycle_save_done_count = 0
        self._lifecycle_save_failed_count = 0
        self._lifecycle_save_latency_ms_sum = 0.0
        self._lifecycle_save_latency_ms_max = 0.0
        self._lifecycle_save_done_count_by_source: defaultdict[str, int] = (
            defaultdict(int))
        self._lifecycle_save_failed_count_by_source: defaultdict[str, int] = (
            defaultdict(int))
        self._lifecycle_save_latency_ms_sum_by_source: defaultdict[str, float] = (
            defaultdict(float))
        self._lifecycle_save_latency_ms_max_by_source: defaultdict[str, float] = (
            defaultdict(float))

    def _record_lifecycle_done(
        self,
        kind: str,
        start_time: float,
        failed: bool,
    ) -> None:
        latency_ms = max(0.0, (time.monotonic() - start_time) * 1000)
        if kind == "load":
            self._lifecycle_load_done_count += 1
            self._lifecycle_load_latency_ms_sum += latency_ms
            self._lifecycle_load_latency_ms_max = max(
                self._lifecycle_load_latency_ms_max, latency_ms)
            if failed:
                self._lifecycle_load_failed_count += 1
        elif kind == "save":
            self._lifecycle_save_done_count += 1
            self._lifecycle_save_latency_ms_sum += latency_ms
            self._lifecycle_save_latency_ms_max = max(
                self._lifecycle_save_latency_ms_max, latency_ms)
            if failed:
                self._lifecycle_save_failed_count += 1
        else:
            raise AssertionError(f"unknown hybrid connector lifecycle {kind=}")

    def _record_save_source_lifecycle_done(
        self,
        source: str,
        start_time: float,
        done_time: float,
        failed: bool,
    ) -> None:
        latency_ms = max(0.0, (done_time - start_time) * 1000)
        self._lifecycle_save_done_count_by_source[source] += 1
        self._lifecycle_save_latency_ms_sum_by_source[source] += latency_ms
        self._lifecycle_save_latency_ms_max_by_source[source] = max(
            self._lifecycle_save_latency_ms_max_by_source[source], latency_ms)
        if failed:
            self._lifecycle_save_failed_count_by_source[source] += 1

    def get_hybrid_connector_stats(
        self, num_gpu_blocks: int
    ) -> HybridConnectorStats:
        total_gpu_blocks = num_gpu_blocks - 1
        if total_gpu_blocks > 0:
            kv_cache_usage = (
                len(self._extra_kv_block_refcounts) / total_gpu_blocks
            )
            load_refcounts = self._extra_kv_block_refcounts_by_kind.get("load")
            pre_scheduler_load_kv_cache_usage = (
                len(load_refcounts) / total_gpu_blocks
                if load_refcounts is not None
                else 0.0
            )
            save_refcounts = self._extra_kv_block_refcounts_by_kind.get("save")
            post_scheduler_save_kv_cache_usage = (
                len(save_refcounts) / total_gpu_blocks
                if save_refcounts is not None
                else 0.0
            )
        else:
            kv_cache_usage = 0.0
            pre_scheduler_load_kv_cache_usage = 0.0
            post_scheduler_save_kv_cache_usage = 0.0

        saving_status, saving_ready_to_teardown_count = self._saving_req_status()

        load_done_count = self._lifecycle_load_done_count
        load_failed_count = self._lifecycle_load_failed_count
        load_latency_ms_sum = self._lifecycle_load_latency_ms_sum
        load_latency_ms_max = self._lifecycle_load_latency_ms_max
        save_done_count = self._lifecycle_save_done_count
        save_failed_count = self._lifecycle_save_failed_count
        save_latency_ms_sum = self._lifecycle_save_latency_ms_sum
        save_latency_ms_max = self._lifecycle_save_latency_ms_max
        save_done_count_by_source = dict(
            self._lifecycle_save_done_count_by_source)
        save_failed_count_by_source = dict(
            self._lifecycle_save_failed_count_by_source)
        save_latency_ms_sum_by_source = dict(
            self._lifecycle_save_latency_ms_sum_by_source)
        save_latency_ms_max_by_source = dict(
            self._lifecycle_save_latency_ms_max_by_source)
        self._reset_lifecycle_stats()
        return HybridConnectorStats(
            kv_cache_usage=kv_cache_usage,
            pre_scheduler_load_kv_cache_usage=pre_scheduler_load_kv_cache_usage,
            post_scheduler_save_kv_cache_usage=post_scheduler_save_kv_cache_usage,
            saving_status=saving_status,
            loading_status=self._loading_req_status(),
            saving_ready_to_teardown_count=saving_ready_to_teardown_count,
            load_done_count=load_done_count,
            load_failed_count=load_failed_count,
            load_latency_ms_sum=load_latency_ms_sum,
            load_latency_ms_max=load_latency_ms_max,
            save_done_count=save_done_count,
            save_failed_count=save_failed_count,
            save_latency_ms_sum=save_latency_ms_sum,
            save_latency_ms_max=save_latency_ms_max,
            save_done_count_by_source=save_done_count_by_source,
            save_failed_count_by_source=save_failed_count_by_source,
            save_latency_ms_sum_by_source=save_latency_ms_sum_by_source,
            save_latency_ms_max_by_source=save_latency_ms_max_by_source,
        )

    def mark_post_scheduler_save_extra(self, reqid: str) -> None:
        state = self._saving.get(reqid)
        if state is None:
            return
        self._track_extra_kv_blocks("save", reqid, state.kvblks)

    # core thread
    def step(self) -> Optional[dict[int, list[EngineCoreOutput]]]:
        from vllm.v1.engine.core import combine_outputs
        kvt_done = self._step_saved()
        self._step_waiting()
        self._step_loaded()
        return combine_outputs(kvt_done, self._step_aborting())

    def _step_saved(self) -> dict[int, list[EngineCoreOutput]]:
        kvt_done: dict[int, list[EngineCoreOutput]] = {}
        while self._saved:
            reqid = self._saved.popleft()
            state = self._saving.get(reqid)
            if state is not None:
                req = state._req
                save_ioret: Optional[IoRet] = get_param(
                    req, HB_SAVE_IORET, None)
                save_failed = (save_ioret is not None
                               and save_ioret.n is not None
                               and save_ioret.n == 0)
                self._record_lifecycle_done(
                    "save", state.start_time, save_failed or req_aborted(req))
                with state._signals_lock:
                    source_done_times = dict(state.source_done_times)
                    source_failed = dict(state.source_failed)
                for source, done_time in source_done_times.items():
                    self._record_save_source_lifecycle_done(
                        source,
                        state.start_time,
                        done_time,
                        source_failed.get(source, False) or req_aborted(req),
                    )

                finish_reason = get_param(req, _PD_FINISH_REASON)
                if finish_reason is not None:
                    # Finish arrived before save → emit kv_transfer_done now.
                    stop_reason = get_param(req, _PD_STOP_REASON)
                    client_index = get_param(req, _PD_CLIENT_INDEX)
                    kvt_done.setdefault(client_index, []).append(
                        EngineCoreOutput(
                            request_id=reqid,
                            new_token_ids=[],
                            finish_reason=finish_reason,
                            stop_reason=stop_reason,
                            # Carry prefix-cache hit count: this empty finish
                            # output's default 0 would otherwise overwrite
                            # req_state in OutputProcessor, making the final
                            # RequestOutput / vllm_req_stats report 0 cached
                            # for a cache-hitting prefill.
                            num_cached_tokens=max(req.num_cached_tokens, 0),
                            kv_transfer_params={"kv_transfer_done": True},
                        )
                    )
                else:
                    # Save arrived before finish → mark for
                    # request_finished_all_groups to emit directly.
                    set_param(req, _PD_SAVED, True)
            self._try_teardown_save(reqid)
        return kvt_done

    # core thread
    def _setup_save(self, req: Request, kvblks: KVCacheBlocks, save_count: int = 1):
        assert not has_setup_save(req)
        _inc_cleanup_rc(req)

        assert req.request_id not in self._saving
        sources = tuple(get_param(req, HB_SAVE_SOURCES, ()))
        assert sources, f"missing save sources for reqid={req.request_id}"
        assert len(sources) == save_count, (
            f"save sources mismatch for reqid={req.request_id}: "
            f"sources={sources} save_count={save_count}"
        )
        self._saving[req.request_id] = _SavingReq(
            req, kvblks, save_count, sources
        )

        sched_acquire_blocks(kvblks)
        set_param(req, _SAVE_PREPARED, True)
        return

    # core thread
    def _try_teardown_save(self, reqid: str):
        state = self._saving.pop(reqid, None)
        if state is None:
            logger.info("teardown save twice: reqid=%s", reqid)
            return
        self._untrack_extra_kv_blocks("save", reqid)
        assert has_setup_save(state._req)
        sched_free_blocks(state.kvblks)
        set_param(state._req, _SAVE_PREPARED, False)
        return

    async def _cleanup(self, req: Request):
        rc = _dec_cleanup_rc(req)
        if rc <= 0:
            await self._backend.async_cleanup(req)
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
                    req.max_computed_tokens(),
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
                self._track_extra_kv_blocks("load", req.request_id, kvblks)
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

            load_state = self._loading.pop(req.request_id, None)
            was_loading = load_state is not None
            loaded_tokens = ioret.n or 0
            req.num_computed_tokens += ioret.n or 0
            req.num_external_computed_tokens += ioret.n or 0

            max_computed = req.max_computed_tokens()
            # stop0 means the async load already reached the KV-transfer
            # computed-token boundary(D-side), no local prefill would be scheduled.
            will_stop0 = False
            if max_computed is not None:
                assert max_computed < req.num_prompt_tokens
                will_stop0 = max_computed <= req.num_computed_tokens

            if (was_loading
                    and ioret.ex is None
                    and not req_aborted(req)
                    and loaded_tokens > 0
                    and self._cfg.cache_config.enable_prefix_caching
                    and req.num_computed_tokens > 0
                    and will_stop0):
                sched_cache_blocks(req, req.num_computed_tokens)

            if will_stop0:
                req.set_max_computed_tokens(req.num_computed_tokens)
                self._stop0.append(req)

            load_failed = ioret.ex is not None or req_aborted(req)
            if load_state is not None:
                self._record_lifecycle_done(
                    "load", load_state.start_time, load_failed)
            self._untrack_extra_kv_blocks("load", req.request_id)
            sched_add_req(req)
            self._add_ts(req, "done")
            if load_failed:
                areq = AbortReq(
                    reqid=req.request_id,
                    output=not req_aborted(req),
                    reason="load fail",
                )
                self.on_abort_req(areq, iscore=True)
        return

    # core thread
    def on_add_req(self, req: "Request") -> bool:
        (load_count, save_count, load_sources,
         save_sources) = self._backend.get_operations(req)
        set_operation_sources(req, load_sources, save_sources)
        req.trace_wrapper.on_req_kv_start("load")
        req.trace_wrapper.on_req_kv_start("save")
        if load_count > 0 or save_count > 0:
            self._add_ts(req, "on_add_req")
            self._waiting.append((req, load_count, save_count))
            return True
        return False

    def _step_aborting(self) -> Optional[dict[int, list[EngineCoreOutput]]]:
        load_output: defaultdict[int, list[EngineCoreOutput]] = defaultdict(list)
        while self._aborting:
            areq = self._aborting.popleft()

            # Check _waiting queue first (request not yet processed by hybrid connector)
            waiting_req = None
            for i, (req, _, _) in enumerate(self._waiting):
                if req.request_id == areq.reqid:
                    waiting_req = req
                    del self._waiting[i]
                    break

            if waiting_req is not None:
                logger.info("abort waiting. areq=%s", areq)
                sched_add_req(waiting_req)
                sched_finish_req(areq.reqid)
                if areq.output:
                    _put_abort_resp(load_output, waiting_req)
                continue

            loadreq = self._loading.get(areq.reqid)
            if loadreq is not None:
                forwarded = get_param(loadreq._req, _ABORTED, False)
                logger.info("abort loading. areq=%s forwarded=%s", areq, forwarded)
                if forwarded:
                    continue
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
                save_fwd = get_param(savereq._req, _SAVE_ABORTED, False)
                logger.info("abort saving. areq=%s forwarded=%s", areq, save_fwd)
                if not save_fwd:
                    set_param(savereq._req, _SAVE_ABORTED, True)
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

    # disagg thread
    async def _do_save_done(self, worker_tprank: int, ioret: IoRet):
        state: Optional[_SavingReq] = try_advance(
            self._saving,
            ioret,
            worker_tprank,
            self._tp_size(),
            on_signal_advanced=self._mark_save_source_done_time,
        )
        if state is None:
            return

        # v6d backend signals store failure via n=0 in its save_done IoRet.
        with state._signals_lock:
            save_failed = any(state.source_failed.values())
        if save_failed:
            set_param(state._req, HB_SAVE_IORET, IoRet(n=0))

        logger.info("mark saved. reqid=%s", ioret.reqid)
        state._req.trace_wrapper.on_req_kv_end("save")
        _q_append(self._saved, ioret.reqid)
        await self._cleanup(state._req)
        return

    async def _do_save_done2(self, worker_tprank: int, ioret: IoRet):
        # KVT's legacy save-done protocol does not carry a source field.
        if ioret.source is None:
            ioret.source = "kvt"
        await self._do_save_done(worker_tprank, ioret)

    async def _on_save_done2(self, reader, writer):
        await handle_done_req(reader, writer, self._do_save_done2, _SAVE_DONE_RESP)
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
        state._req.trace_wrapper.on_req_kv_end("load")
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

    # core thread
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return 0, False

    # core thread
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
        **kwargs,
    ):
        assert num_external_tokens == 0
        return

    # core thread
    def build_connector_meta(self, sout: SchedulerOutput) -> HybridMetadata:
        self._prepared.clear()

        sout_data = dict(sout.__dict__)

        # Extract hc_parent from sout using Python's dynamic attributes.
        hc_parent: Optional[SchedulerOutput] = sout_data.get("hc_parent")

        # Allocate hc_stepid/hc_substepid.
        if hc_parent is None:
            # Main step: allocate a new hc_stepid and use hc_substepid 0.
            stepid = self._stepid
            substepid = 0
            # Prepare for the next step.
            self._stepid += 1
            self._substepid = 1  # Reset substepid.
        else:
            # Bypass substep: read IDs from hc_parent.kv_connector_metadata.
            # hc_stepid
            if hc_parent.kv_connector_metadata is None:
                hc_parent.kv_connector_metadata = HybridMetadata(
                    reqs=BackendMeta(),
                    stepid=self._stepid,
                    substepid=0)
                self._stepid += 1
                self._substepid = 1
            parent_meta = hc_parent.kv_connector_metadata
            assert isinstance(parent_meta, HybridMetadata)
            stepid = parent_meta.stepid
            substepid = self._substepid
            self._substepid += 1

        sout_data.update(
            hc_aborted_load=self._abortmeta_load,
            hc_aborted_save=self._abortmeta_save,
            hc_stop0=self._stop0,
            hc_parent=hc_parent,
            hc_stepid=stepid,
            hc_substepid=substepid,
        )
        hcsout = HCSchedOutput(**sout_data)
        self._abortmeta_load = []
        self._abortmeta_save = []
        self._stop0 = []
        reqs = self._backend.build_backend_meta(hcsout)

        return HybridMetadata(reqs=reqs, stepid=stepid, substepid=substepid)

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

    @staticmethod
    def _format_req_source_counts(counts: dict[str, int]) -> str:
        if not counts:
            return "{}"
        return json.dumps(dict(sorted(counts.items())))

    @staticmethod
    def _operation_source_label(sources: tuple[str, ...] | list[str] | None) -> str:
        if not sources:
            return "unknown"
        return "+".join(sources)

    def _source_done_for_all_workers(
        self, worker_signals: dict[int, list[IoRet]], source: str
    ) -> bool:
        tpsize = self._tp_size()
        for rank in range(tpsize):
            signals = worker_signals.get(rank, [])
            if not any(sig.source == source for sig in signals):
                return False
        return True

    @staticmethod
    def _source_failed(
        worker_signals: dict[int, list[IoRet]], source: str
    ) -> bool:
        return any(
            sig.source == source and sig.n is not None and sig.n == 0
            for signals in worker_signals.values()
            for sig in signals)

    def _mark_save_source_done_time(
        self,
        state: _SavingReq,
        ioret: IoRet,
    ) -> None:
        # Called by try_advance while state._signals_lock is held.
        source = ioret.source
        assert source is not None
        if source in state.source_done_times:
            return
        if not self._source_done_for_all_workers(state._worker_signals, source):
            return
        state.source_done_times[source] = time.monotonic()
        state.source_failed[source] = self._source_failed(
            state._worker_signals, source)

    def _saving_req_status(self) -> tuple[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        ready_to_teardown_count = 0
        for state in tuple(self._saving.values()):
            sources = tuple(get_param(state._req, HB_SAVE_SOURCES, ()))
            with state._signals_lock:
                if sources:
                    pending_sources = tuple(
                        source for source in sources
                        if not self._source_done_for_all_workers(
                            state._worker_signals, source)
                    )
                    if not pending_sources:
                        ready_to_teardown_count += 1
                        continue
                    label = self._operation_source_label(pending_sources)
                else:
                    label = "unknown"
            counts[label] += 1
        return self._format_req_source_counts(counts), ready_to_teardown_count

    def _loading_req_status(self) -> str:
        counts: defaultdict[str, int] = defaultdict(int)
        for state in tuple(self._loading.values()):
            sources = tuple(get_param(state._req, HB_LOAD_SOURCES, ()))
            label = self._operation_source_label(sources)
            counts[label] += 1
        return self._format_req_source_counts(counts)

@dataclass
class HCSchedOutput(SchedulerOutput):
    hc_aborted_load: list[Request] = field(default_factory=list)
    hc_aborted_save: list[Request] = field(default_factory=list)
    hc_stop0: list[Request] = field(default_factory=list)
    # A non-None hc_parent means this sout is a substep_sout; hc_parent
    # stores the owning step_sout.
    hc_parent: Optional[SchedulerOutput] = None
    # hc_stepid identifies the current step and starts at 1024.
    hc_stepid: int = 0
    # hc_substepid identifies the current substep. The main step uses 0 and
    # bypass substeps start at 1.
    hc_substepid: int = 0


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
        core_ip = scheduler_rpc_host(self._cfg)
        self._connpool = ConnPool(core_ip, rpc_port(self._cfg), 3)
        self._ioenc = MsgpackEncoder()

        BackendCls = _get_backend_cls(vllm_config)
        self._backend = BackendCls(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
        self._num_layers: Optional[int] = None

        if envs.VLLM_ENABLE_BYPASS_TASK:
            asyncio.run_coroutine_threadsafe(self.start_bypass_task_loop(), self.loop)

        return

    async def _get_bypass_handle(self):
        core_ip = scheduler_rpc_host(self._cfg)
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
        self._backend.bypass_bind(meta.reqs)
        self._backend.bypass_clear()

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
        self._num_layers = sum(
            len(grp.layer_names)
            for grp in self._kv_cache_config.kv_cache_groups
        )
        assert self._num_layers == len(kv_caches), (
            f"num_layers mismatch. expected {self._num_layers}, "
            f"got {len(kv_caches)}"
        )
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

    async def io_done_rpc(self, req: IoDoneReqs, head: int, reshead: int):
        return await self._io_done_rpc(req, head, reshead)

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

    async def _on_req_saved(self, reqid: str, layer_name: str, source: str):
        state = self._saving.get(reqid, None)
        if state is None:
            state = _WSavingReq()
            self._saving[reqid] = state
        state._ready_layers.append(layer_name)
        assert self._num_layers is not None
        if len(state._ready_layers) != self._num_layers:
            return

        tprank = get_tensor_model_parallel_rank()
        savereq = IoDoneReqs(
            worker_tprank=tprank,
            reqids=[IoRet(reqid=reqid, source=source)],
        )
        await self._io_done_rpc(savereq, _SAVE_DONE_REQ, _SAVE_DONE_RESP)
        self._saving.pop(reqid)
        return

    @kill_me_if_exception
    async def _async_save_layer(self, layer_name: str, arg: AsyncGenerator[str, None]):
        source = self._backend.save_done_source()
        assert source is not None, "async save producer must declare save_done_source"
        async for reqid in arg:
            await self._on_req_saved(reqid, layer_name, source)
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

            global _g_worker
            assert _g_worker is None
            _g_worker = self._worker
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
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int,
        **kwargs,
    ):
        assert self._sched is not None
        return self._sched.update_state_after_alloc(
            request, blocks, num_external_tokens, **kwargs
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

    @property
    def pending_requests(self) -> list[Request]:
        """Requests held by hybrid connector but not yet visible to scheduler."""
        assert self._sched is not None
        return self._sched.pending_requests

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
        from vllm.v1.hybrid_connector.kvtbackend import (
            P_IGNORE_OUTPUTS,
            P_REMOTE_DECODE,
        )

        assert self._sched is not None
        self._sched.mark_post_scheduler_save_extra(request.request_id)

        is_remote_decode = get_param(request, P_REMOTE_DECODE)
        is_ignored = get_param(request, P_IGNORE_OUTPUTS)
        if is_remote_decode and not is_ignored:
            finish_reason = request.get_finished_reason()
            if finish_reason == FinishReason.ABORT:
                # Aborted: emit output immediately so the dual-request
                # frontend does not hang.
                # delay_free_blocks=False: scheduler frees its refs, but
                # sched_acquire_blocks (from _setup_save) keeps blocks alive
                # until _try_teardown_save releases them.
                return (False, None)
            if get_param(request, _PD_SAVED):
                # mark_saved already fired → emit kv_transfer_done now.
                return (False, {"kv_transfer_done": True})
            # Finish arrived first; store info so _step_saved can emit
            # kv_transfer_done when mark_saved fires later.
            set_param(request, _PD_FINISH_REASON, finish_reason)
            set_param(request, _PD_STOP_REASON, request.stop_reason)
            set_param(request, _PD_CLIENT_INDEX, request.client_index)
            # delay_free_blocks=False: let scheduler free blocks normally.
            # sched_acquire_blocks (from _setup_save) keeps block ref_cnt > 0
            # so blocks won't actually be released until _try_teardown_save
            # calls sched_free_blocks when save completes.
            # NOT using delay here avoids block leak if mark_saved never fires.
            return (False, {"kv_transfer_pending": True})
        return (False, None)

    def get_hybrid_connector_stats(
        self, num_gpu_blocks: int
    ) -> HybridConnectorStats:
        assert self._sched is not None
        return self._sched.get_hybrid_connector_stats(num_gpu_blocks)


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

_g_worker: Optional[HybridWorker] = None

def hybridworker() -> HybridWorker:
    global _g_worker
    assert _g_worker is not None
    return _g_worker
