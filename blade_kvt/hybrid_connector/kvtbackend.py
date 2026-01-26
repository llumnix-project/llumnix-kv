import asyncio
import copy
import inspect
import os
import struct
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import cache
from typing import Any, AsyncGenerator, List, Optional, Union

import msgspec
import torch

import vllm.envs as envs
from vllm.model_executor.models.registry import ModelRegistry
from . import HybridMetadata
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

from . import (
    BackendMeta,
    HCSchedOutput,
    HybridBackend,
    req_aborted,
    try_setup_save,
    try_teardown_save,
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
    merge_hybrid_blocks,
    req2corereq,
    sched_get_kvblk_ids,
    sched_rpc_server,
    sched_rpc_server_port,
    set_param,
    use_flashinfer,
    use_mla,
    use_sparse_mla,
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
    blkids: list[int]
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
            r = f"{_get_main_node_pod_name()}-{rpc_port(cfg)}"
        else:
            r = _get_main_node_pod_name()
    if r is None:
        r = str(uuid.uuid4())
    return r


# for dash
def _isfakereq(req: EngineCoreRequest) -> bool:
    return len(req.prompt_token_ids) <= 0


# for dash
def _try_wakeup_core(q: deque):
    if len(q) != 1:
        return

    wakeup_scheduler()
    return


# for dash
def _fakecorereq(req: Request) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=req.request_id,
        prompt_token_ids=[],
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=None,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=req.arrival_time,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


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
            os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "4")
        else:
            os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "1")
    elif cfg.model_config.is_hybrid:
        os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "3")
    elif use_flashinfer():
        os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "5")
    else:
        os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "2")
    core_ip = "127.0.0.1"
    sdaddr = f"{core_ip}:{rpc_port(cfg)}"
    os.environ.setdefault("BLLM_KVTRANS_SEND_DONE_ADDR", sdaddr)
    return


@dataclass
class PReqMeta:
    # seen_tokens = 0 means submit_req_send
    # seen_tokens > 0 means submit_delta_send, p_block_ids/d_block_ids is empty
    reqid: str
    d_inst_id: str
    p_block_ids: list[int]
    d_block_ids: list[int]
    new_tokens: int
    has_last_token: bool
    seen_tokens: int
    d_workers_info: list[str] = field(default_factory=list)
    has_freeze: bool = False


@dataclass
class KVTPMeta(BackendMeta):
    reqs: list[PReqMeta]

    def __bool__(self):
        return bool(self.reqs)


def _flatten_cache(
    kv_caches: dict[str, torch.Tensor],
    ) -> Union[list[torch.Tensor], list[list[torch.Tensor]]]:
    """
    return: layer_name -> [cache_tensor1, cache_tensor2, ...]
    """
    from vllm.model_executor.models.utils import extract_layer_index
    index2name = group_layers_by_index(kv_caches)

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


# return dst_inst_name, dst_worker_id
# d_inst_id: dst_inst_name|dst_dprank|dst_tpsize
def _get_dist(d_inst_id: str) -> tuple[str, int, int]:
    src_tprank = get_tp_group().rank_in_group
    src_tpsize = get_tp_group().world_size
    dst_inst_name, dst_dprank, dst_tpsize = d_inst_id.split("|")
    idst_dprank = int(dst_dprank)
    idst_tpsize = int(dst_tpsize)

    if idst_tpsize == src_tpsize:
        return dst_inst_name, idst_dprank * idst_tpsize + src_tprank, src_tprank

    assert src_tpsize > idst_tpsize
    group_n = src_tpsize // idst_tpsize
    assert idst_tpsize * group_n == src_tpsize
    dst_tprank = src_tprank // group_n
    return dst_inst_name, idst_dprank * idst_tpsize + dst_tprank, dst_tprank


# dst_inst, dst_worker_id, dst_worker_info
def _get_distinfo(meta: PReqMeta) -> tuple[str, int, Optional[str]]:
    dst_inst_name, dst_wid, worker_tp_rank = _get_dist(meta.d_inst_id)
    dst_worker_info = (
        None
        if not meta.d_workers_info
        else meta.d_workers_info[worker_tp_rank]
    )
    return dst_inst_name, dst_wid, dst_worker_info


class _SendingReq(IoState):
    def __init__(self):
        super().__init__()
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


class DashReq:
    def __init__(self, req: Request, pblkids: list[int], finished=False):
        self.req = req
        self.finished = finished
        self.pblkids = pblkids
        self.insert_ts = time.time()
        assert len(self.pblkids) > 0


class PBackend(HybridBackend):
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
            self._main_tid = threading.get_native_id()
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
            self._dash_done: dict[str, DashReq] = dict()
            self._infly_kvt: dict[str, PReqMeta] = dict()

            self._dinfoq: deque[RKVTDInfo] = deque()
        return

    def naming_cli(self):
        return self._naming_cli

    # disagg thread, core thread
    def _tp_size(self):
        return self._cfg.parallel_config.tensor_parallel_size

    async def _do_send_done(self, worker_tprank: int, ioret: IoRet):
        tpsize = self._tp_size()
        state: Optional[_SendingReq] = try_advance(
            self._sending, ioret, worker_tprank, tpsize
        )
        if state is None:
            return

        ioret = state.merge()
        state.try_mark_done(ioret)
        return

    async def _on_send_done(self, reader, writer):
        await handle_done_req(reader, writer, self._do_send_done, SEND_DONE_RESP)
        return

    def _check_kvtdinfo(self, info: KVTDInfo) -> bool:
        assert info.cached_tokens < info.max_tokens
        blksize = self._cfg.cache_config.block_size
        assert info.max_tokens <= len(info.blkids) * blksize
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
        state = _SendingReq()
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
        state = _SendingReq()
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

    def _dash_transfer_req(self, reqid: str, kvtstate: KVTState, ret: list[PReqMeta]):
        assert not kvtstate.untouched
        dashreq = self._dash_done.pop(reqid)
        dinfo = kvtstate.dinfo
        new_tokens = kvtstate.maxtokens - dinfo.cached_tokens
        pmeta = PReqMeta(
            reqid=dashreq.req.request_id,
            d_inst_id=dinfo.instid,
            p_block_ids=dashreq.pblkids,
            d_block_ids=dinfo.blkids,
            seen_tokens=dinfo.cached_tokens,
            new_tokens=new_tokens,
            has_last_token=True,
            d_workers_info=dinfo.d_workers_info,
            has_freeze=True,
        )
        ret.append(pmeta)
        return

    def _dash_finish_req(self, reqid: str, ret: list[PReqMeta]):
        dashreq = self._dash_done.get(reqid, None)
        if dashreq is None:
            return
        if dashreq.finished:
            return
        dashreq.finished = True
        kvtstate: Optional[KVTState] = get_param(dashreq.req, P_KVT_STATE)
        if kvtstate is None:
            return
        assert kvtstate.untouched
        kvtstate.untouched = False
        self._dash_transfer_req(reqid, kvtstate, ret)
        return

    # isdone?
    def _dash_get_req(self, reqid: str) -> tuple[Optional[Request], bool]:
        req = self._dash_done.get(reqid, None)
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
        pmeta = PReqMeta(
            reqid=reqid,
            d_inst_id=dinfo.instid,
            p_block_ids=sched_get_kvblk_ids(
                reqid, self._gamma, self._enable_prefix_caching
            ),
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
    def _step_dinfoq(self, ret: list[PReqMeta]):
        while self._dinfoq:
            rdinfo = self._dinfoq.popleft()
            dinfo = rdinfo.dinfo

            req, isdone = self._dash_get_req(rdinfo.reqid)
            if req is None:
                loop = get_hybrid_sched_loop()
                ioret = IoRet(
                    reqid=rdinfo.reqid, ex=CodeError(CODE_REQNOTFOUND, "req not found")
                )
                loop.call_soon_threadsafe(self._mark_send_done, ioret)
                continue
            assert get_param(req, P_KVT_STATE) is None

            maxcomputed = None
            maxtokens = min(req.num_tokens, dinfo.max_tokens)
            if maxcomputed is not None:
                maxtokens = min(maxtokens, maxcomputed)
            kvtstat = KVTState(dinfo=dinfo, maxtokens=maxtokens, untouched=False)
            set_param(req, P_KVT_STATE, kvtstat)

            if maxcomputed is not None and maxcomputed <= dinfo.cached_tokens:
                loop = get_hybrid_sched_loop()
                ioret = IoRet(reqid=req.request_id, n=dinfo.cached_tokens)
                loop.call_soon_threadsafe(self._mark_send_done, ioret)
                try_teardown_save(req)
                self._dash_done.pop(req.request_id, None)
                continue

            if not isdone:
                try_setup_save(req)
                if rdinfo.migration:
                    self._migration_bypass(req.request_id, kvtstat, ret)
                else:
                    kvtstat.untouched = True
                continue

            if dinfo.max_tokens != req.num_prompt_tokens:
                reason = (
                    f"reqid={req.request_id}, dinfo.max_tokens={dinfo.max_tokens}, "
                    f"prompt_len={req.num_prompt_tokens}, gamma={self._gamma}"
                )
                logger.error("Aborting request due to token mismatch: %s", reason)
                core_abort_req(req.request_id, reason, True)
                continue
            self._dash_transfer_req(req.request_id, kvtstat, ret)
        return

    def _update_dash_done(self,
                          req: Request,
                          islast: bool,
                          pblkids: Optional[list[int]],
                          finished=False):
        if not get_param(req, P_REMOTE_DECODE):
            return
        if not islast:
            return
        if req.request_id in self._dash_done:
            # async scheduling~
            return

        if pblkids is None:
            pblkids = sched_get_kvblk_ids(
                req.request_id, self._gamma, self._enable_prefix_caching
            )
            assert len(pblkids) > 0
        self._dash_done[req.request_id] = DashReq(req, pblkids, finished)
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

    def _step_finished_reqs(self, sout: SchedulerOutput, ret: list[PReqMeta]):
        if len(self._infly_kvt) <= 0 and len(self._dash_done) <= 0:
            # fast path
            return
        for reqid in sout.finished_req_ids:
            meta = self._infly_kvt.pop(reqid, None)
            if meta is None:
                self._dash_finish_req(reqid, ret)
                continue
            assert reqid not in self._dash_done
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
            ret.append(new_meta)
        return

    def _step_aborting(self, sout: HCSchedOutput):
        for areq in sout.hc_aborted_save:
            kvstate: Optional[KVTState] = get_param(areq, P_KVT_STATE)
            if kvstate is not None and not kvstate.untouched:
                continue
            assert areq.request_id not in self._infly_kvt

            self._dash_done.pop(areq.request_id, None)
            if kvstate is not None:
                assert kvstate.untouched
                loop = get_hybrid_sched_loop()
                ioret = IoRet(
                    reqid=areq.request_id, ex=CodeError(CODE_INTERNALERROR, "aborted")
                )
                loop.call_soon_threadsafe(self._mark_send_done, ioret)

            try_teardown_save(areq)
        return

    def _step_stop0(self, sout: HCSchedOutput, ret: list[PReqMeta]):
        for req in sout.hc_stop0:
            kvtstate: Optional[KVTState] = get_param(req, P_KVT_STATE)
            assert kvtstate is None

            assert req.request_id not in self._dash_done
            self._update_dash_done(req, True, None, True)
        return

    def _step_dash_done(self):
        now = time.time()
        timeout_s = envs.VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS
        for reqid, dashreq in self._dash_done.items():
            if dashreq.insert_ts + timeout_s > now:
                break
            core_abort_req(reqid, "dashdone.timeout", True)
        return

    # worker thread/bypass thread
    def _start_req_send(self, freeze_metas: list[PReqMeta]):
        if len(freeze_metas) <= 0:
            return

        kvtmetas: list[bladekv.ReqMeta] = []
        for reqm in freeze_metas:
            dst_inst_name, dst_wid, dst_worker_info = _get_distinfo(reqm)
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

        self._bladkv_cli.start_req_send(kvtmetas)
        return

    # ==============================
    # Scheduler-side methods
    # ==============================
    async_get_num_new_matched_tokens = None  # type: ignore

    def get_operations(self, req: Request) -> tuple[int, int]:
        d_inst_id = get_param(req, P_KVT_STATE)
        # NOTE(llx): Decode will only use n-gamma-1 tokens' kv cache,
        # add these to fit qwen3-next's gdn state
        # NOTE(llx): re-init req, maybe has better imlementation?
        req.prompt_token_ids = req.prompt_token_ids[:-(self._gamma + 1)]
        req._all_token_ids = req.prompt_token_ids.copy()
        req.num_prompt_tokens = len(req.prompt_token_ids)
        req.all_token_ids = ConstantList(req._all_token_ids)
        if d_inst_id is not None:
            return 0, 1
        if get_param(req, P_REMOTE_DECODE):
            if req.num_prompt_tokens <= 1:
                return 0, 0
            return 0, 1
        return 0, 0

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        assert isinstance(sout, HCSchedOutput)
        ret: list[PReqMeta] = []

        self._step_stop0(sout, ret)
        self._step_aborting(sout)
        self._step_dinfoq(ret)
        self._step_finished_reqs(sout, ret)
        self._step_dash_done()

        kvtstate: Optional[KVTState] = None
        for reqdata in sout.scheduled_new_reqs:
            req = self.get_request(reqdata.req_id)
            assert req is not None

            new_tokens = sout.num_scheduled_tokens[reqdata.req_id]
            new_tokens += reqdata.num_computed_tokens  # prompt cache
            num_tokens = None
            if num_tokens is None:
                num_tokens = req.num_prompt_tokens
            has_last_token = new_tokens >= num_tokens
            # same as other blocks handling
            # but don't change the blocks layout in request
            # reqdata came from scheduler, would have null blocks
            # when enable hybrid model prefix cache
            has_null_blk = self._enable_prefix_caching and len(reqdata.block_ids) > 1
            concated_blk_ids = merge_hybrid_blocks(
                reqdata.block_ids,
                gamma=self._gamma,
                has_null_blk=has_null_blk
            )
            # here we won't need assert len(p_block_ids)==1,
            # blocks with different attn groups can be easily fit in kvt,
            # when layer sending, just traverse the groups
            # to get which blockid to send and use block id to get offset
            p_block_ids = concated_blk_ids
            kvtstate = get_param(req, P_KVT_STATE)
            if kvtstate is None:
                self._update_dash_done(req, has_last_token, p_block_ids)
                continue
            d_computed_tokens = kvtstate.dinfo.cached_tokens
            if new_tokens <= d_computed_tokens:
                # no need to send kvcache
                continue

            kvtstate.untouched = False
            num_tokens = kvtstate.maxtokens
            has_last_token = new_tokens >= num_tokens
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
            ret.append(kvtmeta)
            self._update_infly_kvt(kvtmeta)

        req_data = sout.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req = self.get_request(req_id)
            assert req is not None
            num_computed_tokens = req_data.num_computed_tokens[i]
            seen_tokens = num_computed_tokens
            new_tokens = sout.num_scheduled_tokens[req_id]
            end_tokens = seen_tokens + new_tokens
            num_tokens = None
            if num_tokens is None:
                num_tokens = req.num_prompt_tokens
            has_last_token = end_tokens >= num_tokens

            kvtstate = get_param(req, P_KVT_STATE)
            if kvtstate is None:
                self._update_dash_done(req, has_last_token, None)
                continue
            d_computed_tokens = kvtstate.dinfo.cached_tokens
            if end_tokens <= d_computed_tokens:
                continue

            num_tokens = kvtstate.maxtokens
            has_last_token = end_tokens >= num_tokens
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
                p_block_ids = sched_get_kvblk_ids(
                    req.request_id,
                    gamma=self._gamma,
                    enable_prefix_caching=self._enable_prefix_caching,
                )
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
                ret.append(kvtmeta)
                self._update_infly_kvt(kvtmeta)
            elif seen_tokens < num_tokens:
                # d_computed_tokens < seen_tokens < num_tokens
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
        return KVTPMeta(reqs=ret)

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.is_hybrid:
            # Pick one representative tensor cache from hybrid kv caches.
            layer_cache: torch.Tensor | None = None
            for cache in kv_caches.values():
                # gdn layer's kv cache is a list contains conv state and ssm state
                # we use self.attn's kv cache to register
                if isinstance(cache, torch.Tensor):
                    layer_cache = cache
                    break

            assert layer_cache is not None

            from vllm.model_executor.models.utils import extract_layer_index

            physical_tensors: dict[str, torch.Tensor] = {}
            # 注册时按照每个shared_by的最后一层触发kv cache传输
            for tensor_group in self._kv_cache_config.kv_cache_tensors:
                layer_idxs = [
                    extract_layer_index(layer_name)
                    for layer_name in tensor_group.shared_by
                ]
                last_group_layer = max(layer_idxs)
                self.hybrid_model_send_layer.append(last_group_layer)

                # TODO: Remove this after reconstruct task->blocks
                if "BLLM_KVTRANS_GDN_BLOCK_NUM" not in os.environ:
                    os.environ["BLLM_KVTRANS_GDN_BLOCK_NUM"] = \
                        str((len(layer_idxs)-1)*(self._gamma+1))

                for layer in tensor_group.shared_by:
                    # should only have one self.attn layer in each shared_by
                    if isinstance(kv_caches[layer], torch.Tensor):
                        physical_tensors[
                            tensor_group.shared_by[
                                layer_idxs.index(last_group_layer)
                            ]
                        ] = kv_caches[layer]
                    else:
                        assert isinstance(kv_caches[layer], list) \
                            and len(kv_caches[layer]) == 2, \
                                "Should contain conv and ssm state"
                        if "QWEN3_NEXT_CONV_SHAPE" not in os.environ:
                            os.environ["QWEN3_NEXT_CONV_SHAPE"] = \
                                ','.join(map(str, kv_caches[layer][0].shape))
                        if "QWEN3_NEXT_SSM_SHAPE" not in os.environ:
                            os.environ["QWEN3_NEXT_SSM_SHAPE"] = \
                                ','.join(map(str, kv_caches[layer][1].shape))
                        if "GDN_ELEMENT_SIZE" not in os.environ:
                            assert (kv_caches[layer][0].element_size() \
                            == kv_caches[layer][1].element_size()), \
                                "Conv and ssm state should have the same element size"
                            os.environ["GDN_ELEMENT_SIZE"] = \
                                str(kv_caches[layer][0].element_size())
            self.hybrid_model_send_layer = sorted(self.hybrid_model_send_layer)
            # use full attn block to calculate block_bytes
            # mamba block will padding to full attn block size
            # [2 (k and v), num_blocks, block_size, kv_heads, head_dim]
            kv_caches = physical_tensors
            cache_shape = layer_cache.shape
            token_bytes = [
                2 * cache_shape[3] * cache_shape[4]
                * layer_cache.element_size()
            ]
        else:
            # Since it's unclear how the model architecture will evolve in the future,
            # we currently only handle special cases based on the Dpsk-V32 structure.
            tensor_shape_dict: dict[
                tuple[int, ...], list[tuple[str, torch.Tensor]]
            ] = defaultdict(list)

            for layer, layer_kv in kv_caches.items():
                tensor_shape_dict[layer_kv.shape].append((layer, layer_kv))

            if len(tensor_shape_dict) == 1: # Normal case
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
                    if use_flashinfer():
                        # check hardware, flashinfer only support HND layout
                        from vllm.platforms import current_platform
                        capability = current_platform.get_device_capability()
                        # parsing HND layout kv cache needs to know head num
                        # only prefill node needs to know
                        if capability is not None and capability.major == 10:
                            os.environ.setdefault(
                                "BLLM_KVTRANS_ATTN_HEAD_NUM", str(cache_shape[3])
                            )
                        else:
                            raise ValueError(
                                "Currently KVT only support HND layout of flashinfer"
                            )
                else:
                    # for mla, which's kv shape like:
                    # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
                    assert len(cache_shape) == 3
                    token_bytes = [cache_shape[2] * layer_tensor.element_size()]
            else:
                token_bytes = []
                for cache_shape, pair_list in tensor_shape_dict.items():
                    assert len(cache_shape) == 3, "Currently only support Dpsk-V32"
                    _, layer_tensor = pair_list[0]
                    # k cache & kv cache share same num_blocks and block_size
                    token_bytes.append(cache_shape[2] * layer_tensor.element_size())
        block_size = self._cfg.cache_config.block_size
        block_bytes = [token_byte * block_size for token_byte in token_bytes]
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

        self._bladkv_cli = bladekv.KVTransferClient(
            self._inst_id,
            self._cfg.parallel_config.tensor_parallel_size,
            worker_id=worker_id,
            worker_tp_rank=rank,
            block_bytes=block_bytes,
            token_bytes=token_bytes,
            naming_url=self._naming_url,
            layers=_flatten_cache(kv_caches),
            protocols=[protocol],
        )
        winfo = bladekv.current_worker_info('client')
        logger.info(
            "register_kv_caches. worker_id=%s winfo=%s, kv_cache shape=%s",
            worker_id, winfo, cache_shape
        )
        # self._naming_cli.store(f"worker_{worker_id}", winfo)
        return

    def bind_backend_metadata(self, meta: BackendMeta):
        mytid = threading.get_native_id()
        assert isinstance(meta, KVTPMeta)
        if self._bladkv_cli is None:
            assert not meta
            return

        freeze_metas: list[PReqMeta] = []
        for reqm in meta.reqs:
            assert isinstance(reqm, PReqMeta)
            if reqm.has_freeze:
                assert (reqm.has_last_token and len(reqm.d_block_ids) > 0
                        and reqm.d_inst_id)
                freeze_metas.append(reqm)
                continue
            assert mytid == self._main_tid
            if reqm.d_block_ids:
                dst_inst_name, dst_wid, dst_worker_info = _get_distinfo(reqm)
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
                )
            else:
                self._bladkv_cli.submit_delta_send(
                    reqm.reqid,
                    seen_tokens=reqm.seen_tokens,
                    new_tokens=reqm.new_tokens,
                    has_last_token=reqm.has_last_token,
                )
        self._start_req_send(freeze_metas)
        if mytid != self._main_tid:
            # bypass thread
            # bypass start send
            return
        self._bladkv_cli.start_send_step()
        return

    def clear_backend_metadata(self):
        mytid = threading.get_native_id()
        if mytid != self._main_tid:
            # bypass thread
            # bypass flush send
            return
        if self._bladkv_cli is None:
            return
        self._bladkv_cli.flush_send_step()
        return

    async_load_kv = None  # type: ignore

    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        from vllm.model_executor.models.utils import extract_layer_index
        layer_idx = extract_layer_index(layer_name)
        if self.is_hybrid:
            if layer_idx in self.hybrid_model_send_layer:
                idx = self.hybrid_model_send_layer.index(layer_idx)
                self._bladkv_cli.record_event(idx, torch.cuda.current_stream())
        else:
            self._bladkv_cli.record_event(layer_idx, torch.cuda.current_stream())
        return None


class DBackend(HybridBackend):
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
        self._enable_prefix_caching = self._cfg.cache_config.enable_prefix_caching

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
        """Generate delay list for KVT retry mechanism based on environment variable."""
        max_delay_ms = envs.VLLM_KVT_MAX_DELAY_MS
        delay_s_list = [1 / 1000.0, 3 / 1000.0, 7 / 1000.0, 11 / 1000.0, 17 / 1000.0]

        # If max_delay_ms > 100, add additional delays at 100ms intervals
        if max_delay_ms > 100:
            current_delay_ms = 100
            while current_delay_ms < max_delay_ms:
                delay_s_list.append(current_delay_ms / 1000.0)
                current_delay_ms += 100

        return delay_s_list

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
        blockids = blocks.get_block_ids()
        # putting all kv cache in one group
        concated_blk_ids = merge_hybrid_blocks(blockids, gamma=self._gamma)

        dprank = self._cfg.parallel_config.data_parallel_rank
        tpsize = self._cfg.parallel_config.tensor_parallel_size
        inst_id = f"{self._inst_id}|{dprank}|{tpsize}"
        kvtdinfo = KVTDInfo(
            instid=inst_id,
            blkids=concated_blk_ids,
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
            # dash: "remote_port": "INT"
            remote_port = int(remote_port)
            peer_hint = (remote_host, remote_port)

        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, PREFILL_REQ, 0)
        reqbufs = self._enc.encode_into(corereq, msgbuf, 8)
        assert len(reqbufs) == 1  # Need Support MsgpackEncoder.aux_buffers
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

    async def _dash_prefill_rpc(self, req: Request, blocks: KVCacheBlocks) -> int:
        blockids = blocks.get_block_ids()
        # putting all kv cache in one group
        concated_blk_ids = merge_hybrid_blocks(blockids, gamma=self._gamma)

        dprank = self._cfg.parallel_config.data_parallel_rank
        tpsize = self._cfg.parallel_config.tensor_parallel_size
        inst_id = f"{self._inst_id}|{dprank}|{tpsize}"
        kvtdinfo = KVTDInfo(
            instid=inst_id,
            blkids=concated_blk_ids,
            cached_tokens=req.num_computed_tokens,
            max_tokens=req.num_prompt_tokens - (self._gamma + 1),
            d_workers_info=self._workers_info,
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
            start_ts = time.monotonic()
            kvtresp = await self._kvt_rpc(req, msgbuf, peer_hint)
            end_ts = time.monotonic()
            dur_ms = (end_ts - start_ts) * 1000
            logger.info(
                "disagg end dash prefill. reqid=%s prompt=%s computed=%s dur_ms=%s kvtresp=%s retry=%s",  # noqa: E501
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
        self, request: "Request", blocks: "KVCacheBlocks", rmt_tokens: int
    ) -> Optional[IoRet]:
        if rmt_tokens <= 0:
            return None

        ioret = IoRet(n=0)
        cached = 0
        try:
            if get_param(request, D_REMOTE_PREFILL):
                cached = await self._dash_prefill_rpc(request, blocks)
            else:
                cached = await self._prefill_rpc(request, blocks)
            ioret.n = cached - request.num_computed_tokens
            # Workaround, refactor at AONE 77521261
            ioret.n = min(ioret.n, rmt_tokens)
        except Exception as e:
            logger.exception("load fail. req=%s", request.request_id)
            ioret.ex = e
        return ioret

    # return: (load_count, save_count)
    def get_operations(self, req: Request) -> tuple[int, int]:
        # Here, we first determine whether the request's Prefill
        # should be executed locally, based on information such as
        # the status of the P node in naming,
        # the current status of the D node,
        # and the characteristics of the request itself.
        if req.num_prompt_tokens <= 1:
            return 0, 0

        if get_param(req, D_REMOTE_PREFILL):
            assert get_param(req, "remote_host") is not None
            assert get_param(req, "remote_port") is not None
            return 1, 0

        disagg = get_param(req, D_DISAGG, True)
        if not disagg:
            return 0, 0

        return 1, 0

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
        if self.is_hybrid:
            # Pick one representative tensor cache from hybrid kv caches.
            layer_cache: torch.Tensor | None = None
            for cache in kv_caches.values():
                # gdn layer's kv cache is a list contains conv state and ssm state
                # we use self.attn's kv cache to register
                if isinstance(cache, torch.Tensor):
                    layer_cache = cache
                    break

            assert layer_cache is not None

            from vllm.model_executor.models.utils import extract_layer_index

            physical_tensors: dict[str, torch.Tensor] = {}
            # 注册时按照每个shared_by的最后一层触发kv cache传输
            for tensor_group in self._kv_cache_config.kv_cache_tensors:
                layer_idxs = [
                    extract_layer_index(layer_name)
                    for layer_name in tensor_group.shared_by
                ]
                last_group_layer = max(layer_idxs)
                self.hybrid_model_send_layer.append(last_group_layer)
                for layer in tensor_group.shared_by:
                    if isinstance(kv_caches[layer], torch.Tensor):
                        physical_tensors[
                            tensor_group.shared_by[
                                layer_idxs.index(last_group_layer)
                            ]
                        ] = kv_caches[layer]
                        break
            self.hybrid_model_send_layer = sorted(self.hybrid_model_send_layer)
            # use full attn block to calculate block_bytes
            # mamba block will padding to full attn block size
            # [2 (k and v), num_blocks, block_size, kv_heads, head_dim]
            kv_caches = physical_tensors
            cache_shape = layer_cache.shape
            token_bytes = [
                2 * cache_shape[3] * cache_shape[4]
                * layer_cache.element_size()
            ]
        else:
            # Since it's unclear how the model architecture will evolve in the future,
            # we currently only handle special cases based on the Dpsk-V32 structure.
            tensor_shape_dict: dict[
                tuple[int, ...], list[tuple[str, torch.Tensor]]
            ] = defaultdict(list)

            for layer, layer_kv in kv_caches.items():
                tensor_shape_dict[layer_kv.shape].append((layer, layer_kv))

            if len(tensor_shape_dict) == 1: # Normal case
                cache_shape, pairs_list = next(iter(tensor_shape_dict.items()))
                _, layer_tensor = pairs_list[0]
                if len(cache_shape) == 5:
                    # [2 (k and v), num_blocks, block_size, kv_heads, head_dim]
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
                    assert len(cache_shape) == 3, "Currently only support Dpsk-V32"
                    _, layer_tensor = pair_list[0]
                    # k cache & kv cache share same num_blocks and block_size
                    token_bytes.append(cache_shape[2] * layer_tensor.element_size())
        block_size = self._cfg.cache_config.block_size
        block_bytes = [token_byte * block_size for token_byte in token_bytes]
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

        self._bladkv_srv = bladekv.KVTransferServer(
            self._inst_id,
            self._cfg.parallel_config.tensor_parallel_size,
            worker_id=worker_id,
            worker_tp_rank=rank,
            block_bytes=block_bytes,
            token_bytes=token_bytes,
            naming_url=self._naming_url,
            layers=_flatten_cache(kv_caches),
            protocols=[protocol],
        )

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
        # Get scheduler address from environment or config
        core_ip = "127.0.0.1"
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
