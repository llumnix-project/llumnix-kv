import asyncio
import struct
import time
from typing import Any, Optional, Union

import msgspec

import vllm.envs as envs
from vllm.v1.request import RequestStatus

from .. import PREALLOC_KEY, BackendMeta, HybridBackend
from ..engine_proxy import (
    EngineCoreOutputs,
    EngineCoreRequest,
    KVCacheBlocks,
    KVConnectorRole,
    MsgpackDecoder,
    PlaceholderModule,
    Request,
    SchedulerOutput,
    VllmConfig,
    core_abort_req,
    core_add_req,
    core_get_param,
    core_update_params,
    get_hybrid_sched_loop,
    get_ip,
    get_logger,
    get_param,
    sched_rpc_server,
    set_param,
)
from ..kvtbackend import (
    CODE_MIGRATE_REJECTED_BUSY,
    CODE_OK,
    CODE_REQNOTFOUND,
    TRANSFER_KV_REQ,
    TRANSFER_KV_RESP,
    KVTDInfo,
    KVTResp,
    RKVTDInfo,
    get_inst_id,
    reg_naming,
)
from ..utils import (
    CodeError,
    ConnManager,
    ConnPool,
    IoRet,
    PeerManager,
    _get_peer_id,
    kill_me_if_exception,
)
from . import (
    ABORT_REQS_REQ,
    ABORT_REQS_RESP,
    KVT_SUSPEND_REQ,
    MIGRATE_TO_REQ,
    MIGRATE_TO_RESP,
    NEW_OUTPUT_REQ,
    NEW_OUTPUT_RESP,
    NEW_REQ_REQ,
    NEW_REQ_RESP,
    OUTPUT_TOKENS_N,
    SRC_INFO,
    MigrateResp,
    _g_migrate_in_req_ids,
    _g_migrate_in_req_info,
    _g_migrate_in_req_info_lock,
    _g_migrate_out_req_ids,
    _g_migrate_out_req_info,
    _g_migrate_out_req_info_lock,
    int2ipport,
    ipport2int,
    is_migration,
)

try:
    import blade_kvt
    from blade_kvt.kv_transfer import connect_naming
except ImportError:
    blade_kvt = PlaceholderModule("blade_kvt")
    connect_naming = blade_kvt.placeholder_attr("connect_naming")

logger = get_logger(__name__)


# True: left += right, right is newer
def _try_merge(left: EngineCoreOutputs, right: EngineCoreOutputs) -> bool:
    assert left.engine_index == right.engine_index
    if left.utility_output and right.utility_output:
        return False

    left.outputs.extend(right.outputs)

    if right.scheduler_stats:
        left.scheduler_stats = right.scheduler_stats

    left.timestamp = max(left.timestamp, right.timestamp)

    if left.utility_output is None:
        left.utility_output = right.utility_output

    if right.finished_requests:
        if left.finished_requests:
            left.finished_requests.update(right.finished_requests)
        else:
            left.finished_requests = right.finished_requests
    return True


class SuspendReq(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    reqid: str
    # cached_tokens: int = 0
    # blkids: list[int] = []


class SuspendResp(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # reqid: str
    # cached_tokens: int = 0
    code: int
    output_token_ids: list[int] = []


# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded SuspendReq/SuspendResp
SUSPEND_REQ = 20080903
SUSPEND_RESP = 30908002

# evt: asyncio.Event
SUSPEND_EVT = "__HybridConnector_Migration_Suspend_Event__"


class MigrationBackend(HybridBackend):
    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, ncli: Optional[Any] = None
    ):
        super().__init__(vllm_config, role)
        if role == KVConnectorRole.WORKER:
            return

        self._cfg = vllm_config
        self._loop = get_hybrid_sched_loop()
        self._inst_id = get_inst_id(vllm_config)
        tpsize = vllm_config.parallel_config.tensor_parallel_size
        dprank = vllm_config.parallel_config.data_parallel_rank
        self.peer_id = _get_peer_id(self._inst_id, dprank, tpsize)

        self._naming_cli = ncli
        if self._naming_cli is None:
            self._naming_url = vllm_config.kv_transfer_config.get_from_extra_config(
                "naming_url", "badbad"
            )
            self._naming_cli = connect_naming(self._inst_id, self._naming_url)
            reg_naming(self._naming_cli, vllm_config)

        self._conn_mgr = ConnManager()
        self._pmgr = PeerManager(
            self._naming_cli, None, self._conn_mgr
        )  # None means ALL, empty connpool
        self._pmgr.start(self._loop)

        rpcsrv = sched_rpc_server()
        rpcsrv.register_method(ABORT_REQS_REQ, self._on_abort_reqs)
        rpcsrv.register_method(NEW_REQ_REQ, self._on_new_req)
        rpcsrv.register_method(MIGRATE_TO_REQ, self._on_migrate_to)
        rpcsrv.register_method(SUSPEND_REQ, self._on_suspend)

        self._engidx = ipport2int(get_ip(), rpcsrv.port())
        assert is_migration(self._engidx)

        # R/W: core thread. R: disagg thread
        self._reqstate: dict[str, Request] = {}

        # R/W: disagg thread
        self._front_outputs: dict[int, asyncio.Queue[EngineCoreOutputs]] = {}
        self._reqdec = MsgpackDecoder(EngineCoreRequest)
        self._susreqdec = msgspec.msgpack.Decoder(SuspendReq)
        self._susrespdec = msgspec.msgpack.Decoder(SuspendResp)
        self._dec = msgspec.msgpack.Decoder()
        self._packenc = msgspec.msgpack.Encoder()
        self._kvtrespdec = msgspec.msgpack.Decoder(KVTResp)

        # migration in limits
        self.max_migrate_in_reqs = envs.LLUMNIX_MAX_REQ_MIG_IN
        max_migrate_in_tokens = envs.LLUMNIX_MAX_TOKEN_MIG_IN
        max_migrate_in_kv_cache_usage_ratio = envs.LLUMNIX_MAX_KV_CACHE_USAGE_RATIO_MIG_IN
        total_tokens = self._cfg.cache_config.num_gpu_blocks * \
            self._cfg.cache_config.block_size
        # Only works when LLUMNIX_DETAILED_MIG_STATUS is True
        self.max_migrate_in_tokens = min(
            max_migrate_in_tokens,
            int(total_tokens * max_migrate_in_kv_cache_usage_ratio))

        global _g_backend
        _g_backend = self
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

    async def do_suspend(self, susreq: SuspendReq) -> SuspendResp:
        req = self._reqstate.get(susreq.reqid)
        if req is None:
            logger.info("suspend:finished: reqid=%s", susreq.reqid)
            return SuspendResp(code=CODE_REQNOTFOUND)
        core_abort_req(susreq.reqid, "migration.suspend", False)
        logger.info("suspend start: req=%s", susreq)

        evt: Optional[asyncio.Event] = get_param(req, SUSPEND_EVT)
        if evt is not None:
            logger.error("suspend:duplicated: reqid=%s", susreq.reqid)
            raise RuntimeError("suspend:duplicated")

        evt = asyncio.Event()
        set_param(req, SUSPEND_EVT, evt)

        if self._reqstate.get(susreq.reqid) is None:
            evt.set()

        await evt.wait()

        # In the case of client disconnection, once the migration-in endpoint
        # returns the first token, the API server can identify which engine
        # the token originates from, and theoretically, normal disconnection
        # handling can proceed accordingly. If the request is aborted before the
        # migration-in endpoint returns the first token, this may result in the
        # migration-in endpoint performing inference on an orphaned request.
        # But, it is acceptable not to distinguish, for two reasons:
        # 1) This is a corner case that occurs infrequently;
        # 2) Ultimately, all related states will still be properly cleaned up.
        stopped = req.status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ]
        if not stopped:
            resp = SuspendResp(code=CODE_OK, output_token_ids=req.output_token_ids[:])
        else:
            resp = SuspendResp(code=CODE_REQNOTFOUND)
        logger.info(
            "suspend end: reqid=%s, stopped=%s, outputs=%s",
            req.request_id,
            stopped,
            req.num_output_tokens,
        )
        return resp

    async def _on_suspend(self, reader, writer):
        susreq: Optional[SuspendReq] = None
        try:
            bodylenbuf = await reader.readexactly(4)
            (bodylen,) = struct.unpack("=I", bodylenbuf)
            reqbuf = await reader.readexactly(bodylen)
            susreq = self._susreqdec.decode(reqbuf)

            susresp = await self.do_suspend(susreq)

            msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
            struct.pack_into("=II", msgbuf, 0, SUSPEND_RESP, 0)
            self._packenc.encode_into(susresp, msgbuf, 8)
            struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

            writer.write(msgbuf)
            await writer.drain()
        finally:
            if susreq is not None:
                _g_migrate_out_req_ids.discard(susreq.reqid)
                if envs.LLUMNIX_DETAILED_MIG_STATUS:
                    with _g_migrate_out_req_info_lock:
                        _g_migrate_out_req_info.pop(susreq.reqid, None)
        return

    def _has_migrate_in_slots(self, req: EngineCoreRequest,
                              prealloc: int) -> bool:
        if len(_g_migrate_in_req_ids) >= envs.LLUMNIX_MAX_REQ_MIG_IN:
            return False
        if envs.LLUMNIX_DETAILED_MIG_STATUS:
            with _g_migrate_in_req_info_lock:
                total_tokens = sum(_g_migrate_in_req_info.values())
            total_tokens += len(req.prompt_token_ids) + prealloc
            if total_tokens > self.max_migrate_in_tokens:
                return False
        return True

    async def _on_migrate_to(self, reader, writer):
        req: Optional[EngineCoreRequest] = None
        try:
            bodylenbuf = await reader.readexactly(4)
            (bodylen,) = struct.unpack("=I", bodylenbuf)
            reqbuf = await reader.readexactly(bodylen)

            req = self._reqdec.decode(reqbuf)
            prealloc = core_get_param(req, OUTPUT_TOKENS_N, 0)
            if not self._has_migrate_in_slots(req, prealloc):
                migresp = MigrateResp(code=CODE_MIGRATE_REJECTED_BUSY)
                msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
                struct.pack_into("=II", msgbuf, 0, MIGRATE_TO_RESP, 0)
                self._packenc.encode_into(migresp, msgbuf, 8)
                struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)
                writer.write(msgbuf)
                await writer.drain()
                logger.warning("no migrate in slots available")
                return

            migresp = MigrateResp(code=CODE_OK)
            msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
            struct.pack_into("=II", msgbuf, 0, MIGRATE_TO_RESP, 0)
            self._packenc.encode_into(migresp, msgbuf, 8)
            struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)
            writer.write(msgbuf)

            # though SRC_INFO is set to tuple, but it is received as list
            srcinfo = core_get_param(req, SRC_INFO)
            assert srcinfo is not None
            srcinfo = tuple(srcinfo)
            core_update_params(req, {SRC_INFO: srcinfo})

            core_update_params(req, {PREALLOC_KEY: prealloc})
            core_update_params(req, {"migration_start_time": time.monotonic()})

            core_add_req(req)
            _g_migrate_in_req_ids.add(req.request_id)
            if envs.LLUMNIX_DETAILED_MIG_STATUS:
                with _g_migrate_in_req_info_lock:
                    _g_migrate_in_req_info[req.request_id] = len(
                        req.prompt_token_ids) + prealloc - 1
            logger.info(
                "migrate. req=%s prealloc=%s srcinfo=%s dstinfo=%s",
                req.request_id,
                prealloc,
                self._pmgr._running.get_peerid(srcinfo),
                self.peer_id,
            )
            await writer.drain()
        except Exception as e:
            if req is not None:
                _g_migrate_in_req_ids.discard(req.request_id)
                if envs.LLUMNIX_DETAILED_MIG_STATUS:
                    with _g_migrate_in_req_info_lock:
                        _g_migrate_in_req_info.pop(req.request_id, None)
            raise e
        return

    async def _on_abort_reqs(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)

        respbuf = struct.pack("=I", ABORT_REQS_RESP)
        writer.write(respbuf)

        reqs: list[str] = self._dec.decode(reqbuf)
        for req in reqs:
            core_abort_req(req, "migration.abort_req", True)
        logger.info("abort reqs=%s", reqs)
        await writer.drain()
        return

    async def _on_new_req(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)

        respbuf = struct.pack("=I", NEW_REQ_RESP)
        writer.write(respbuf)

        req: EngineCoreRequest = self._reqdec.decode(reqbuf)
        core_add_req(req)
        logger.info("new req=%s", req.request_id)
        await writer.drain()
        return

    async def _new_output_rpc(self, connp: ConnPool, outs: EngineCoreOutputs):
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, NEW_OUTPUT_REQ, 0)
        self._packenc.encode_into(outs, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        ok = False
        for idx in range(2):
            try:
                conn = await connp.acquire(idx > 0)
                conn[1].write(msgbuf)
                await conn[1].drain()

                respbuf = await conn[0].readexactly(4)
                (head,) = struct.unpack("=I", respbuf)
                if head != NEW_OUTPUT_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                connp.release(conn)

                ok = True
                break
            except Exception:
                front = f"{connp.host()}:{connp.port()}"
                logger.exception("new_out: outs=%s idx=%s front=%s", outs, idx, front)

            await asyncio.sleep(0)
        if not ok:
            raise RuntimeError("badbad")

        return

    @kill_me_if_exception
    async def _front_main(self, f: int, q: asyncio.Queue[EngineCoreOutputs]):
        ip, port = int2ipport(f)
        connpool = ConnPool(ip, port, 1)

        while True:
            # Exit if idle for too long
            outs = await q.get()
            new_outs: Optional[EngineCoreOutputs] = None

            while not q.empty():
                routs = q.get_nowait()
                merged = _try_merge(outs, routs)
                if not merged:
                    new_outs = routs
                    break

            await self._new_output_rpc(connpool, outs)
            if new_outs:
                await self._new_output_rpc(connpool, new_outs)

        return

    # disagg thread
    # must not async to keep Atomic!
    def _append_q(self, outputs: list[tuple[int, EngineCoreOutputs]]):
        for cli_idx, cli_outs in outputs:
            outq = self._front_outputs.get(cli_idx)
            if outq is None:
                outq = asyncio.Queue()
                self._front_outputs[cli_idx] = outq
                asyncio.create_task(self._front_main(cli_idx, outq))
            outq.put_nowait(cli_outs)
        return

    # core thread
    def _on_outputs(self, outputs: dict[int, EngineCoreOutputs]):
        migoutpus: list[tuple[int, EngineCoreOutputs]] = []
        for cli_idx, cli_outputs in outputs.items():
            if is_migration(cli_idx):
                cli_outputs.engine_index = self._engidx
                migoutpus.append((cli_idx, cli_outputs))

        for cli_idx, _ in migoutpus:
            outputs.pop(cli_idx)

        self._loop.call_soon_threadsafe(self._append_q, migoutpus)
        return

    # core thread
    def _finish_req(self, reqid: str):
        req = self._reqstate.pop(reqid, None)
        _g_migrate_out_req_ids.discard(reqid)
        if envs.LLUMNIX_DETAILED_MIG_STATUS:
            with _g_migrate_out_req_info_lock:
                _g_migrate_out_req_info.pop(reqid, None)
        if req is None:
            return
        evt: Optional[asyncio.Event] = get_param(req, SUSPEND_EVT)
        if evt is None:
            return
        self._loop.call_soon_threadsafe(evt.set)
        return

    async def _suspend_rpc(
        self, srcinfo: tuple[str, int], req: SuspendReq
    ) -> SuspendResp:
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, SUSPEND_REQ, 0)
        self._packenc.encode_into(req, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        resp: Optional[SuspendResp] = None
        for idx in range(2):
            try:
                # Convert srcid to addr:port
                peer_addr_port = self.get_peer(srcinfo)
                if peer_addr_port is None:
                    raise RuntimeError("no peer available")

                pconn = await self._conn_mgr.acquire_conn(peer_addr_port, idx > 0)
                pconn[1].write(msgbuf)
                await pconn[1].drain()

                respbuf = await pconn[0].readexactly(4 + 4)
                head, bodylen = struct.unpack("=II", respbuf)
                if head != SUSPEND_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                respbuf = await pconn[0].readexactly(bodylen)
                resp = self._susrespdec.decode(respbuf)
                self._conn_mgr.release_conn(peer_addr_port, pconn)
                break
            except Exception:
                logger.exception("suspend: srcinfo=%s req=%s", srcinfo, req)
            await asyncio.sleep(0)

        if resp is None:
            raise RuntimeError("sorry")
        return resp

    async def _kvt_rpc(
        self, srcinfo: tuple[str, int], kvtreq: RKVTDInfo
    ) -> Optional[KVTResp]:
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, TRANSFER_KV_REQ, 0)
        self._packenc.encode_into(kvtreq, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        kvtresp: Optional[KVTResp] = None
        for idx in range(2):
            try:
                if srcinfo is None:
                    raise RuntimeError("no peer available")

                pconn = await self._conn_mgr.acquire_conn(srcinfo)
                pconn[1].write(msgbuf)
                await pconn[1].drain()
                respbuf = await pconn[0].readexactly(4 + 4)
                head, bodylen = struct.unpack("=II", respbuf)
                if head != TRANSFER_KV_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                respbuf = await pconn[0].readexactly(bodylen)
                kvtresp = self._kvtrespdec.decode(respbuf)

                self._conn_mgr.release_conn(srcinfo, pconn)
                break
            except Exception:
                logger.exception(
                    "kvt rpc error: srcid=%s req=%s", srcinfo, kvtreq.reqid
                )
            await asyncio.sleep(0)
        return kvtresp

    async def _kvt_suspend_rpc(self, srcinfo: tuple[str, int],
                               kvtreq: RKVTDInfo) -> Optional[KVTResp]:
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, KVT_SUSPEND_REQ, 0)
        self._packenc.encode_into(kvtreq, msgbuf, 8)
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        kvtresp: Optional[KVTResp] = None
        for _ in range(2):
            try:
                if srcinfo is None:
                    raise RuntimeError("no peer available")

                pconn = await self._conn_mgr.acquire_conn(srcinfo)
                pconn[1].write(msgbuf)
                await pconn[1].drain()
                respbuf = await pconn[0].readexactly(4 + 4)
                head, bodylen = struct.unpack("=II", respbuf)
                if head != TRANSFER_KV_RESP:
                    raise RuntimeError(f"invalid resp {head=}")
                respbuf = await pconn[0].readexactly(bodylen)
                kvtresp = self._kvtrespdec.decode(respbuf)

                self._conn_mgr.release_conn(srcinfo, pconn)
                break
            except Exception:
                logger.exception("kvt rpc error: srcid=%s req=%s", srcinfo,
                                 kvtreq.reqid)
            await asyncio.sleep(0)
        return kvtresp

    async def _kvt_suspend(
            self, srcinfo: tuple[str, int], req: Request,
            blocks: KVCacheBlocks) -> KVTResp:
        blockids = blocks.get_block_ids()
        assert len(blockids) == 1
        blksize = self._cfg.cache_config.block_size
        numtokens = get_param(req, OUTPUT_TOKENS_N, 0)
        alltokens = numtokens + len(req.prompt_token_ids)
        assert len(blockids[0]) * blksize >= alltokens
        start_ts = time.monotonic()

        resp: Optional[KVTResp] = None
        if numtokens == 0:
            # req is still in waiting list
            suspend_req = SuspendReq(reqid=req.request_id)
            susresp = await self._suspend_rpc(srcinfo, suspend_req)
            resp = KVTResp(code=susresp.code,
                           cached=req.num_computed_tokens,
                           computed=-1,
                           output_token_ids=susresp.output_token_ids)
        else:
            dprank = self._cfg.parallel_config.data_parallel_rank
            tpsize = self._cfg.parallel_config.tensor_parallel_size
            inst_id = f"{self._inst_id}|{dprank}|{tpsize}"
            kvtdinfo = KVTDInfo(
                instid=inst_id,
                blkids=blockids[0],
                cached_tokens=req.num_computed_tokens,
                max_tokens=alltokens,
                d_workers_info=[],
            )
            kvtreq = RKVTDInfo(reqid=req.request_id,
                               migration=True,
                               dinfo=kvtdinfo)
            resp = await self._kvt_suspend_rpc(srcinfo, kvtreq)

        end_ts = time.monotonic()
        dur_ms = (end_ts - start_ts) * 1000
        logger.info(
            "migrate end kvt: dur_ms=%s, reqid=%s cached=%s max=%s kvtresp=%s",
            dur_ms, req.request_id, req.num_computed_tokens, alltokens, resp)

        if (
            resp is None or resp.code != CODE_OK or
            (not (req.num_computed_tokens <= resp.cached <= alltokens)) or
            resp.output_token_ids is None
        ):
            raise RuntimeError('migration backend kvt rpc failed')
        return resp

    async def _kvt(self, srcinfo: tuple[str, int], req: Request,
                   blocks: KVCacheBlocks):
        blockids = blocks.get_block_ids()
        assert len(blockids) == 1
        blksize = self._cfg.cache_config.block_size
        numtokens = get_param(req, OUTPUT_TOKENS_N, 0)
        alltokens = numtokens + len(req.prompt_token_ids)
        assert len(blockids[0]) * blksize >= alltokens
        dprank = self._cfg.parallel_config.data_parallel_rank
        tpsize = self._cfg.parallel_config.tensor_parallel_size
        inst_id = f"{self._inst_id}|{dprank}|{tpsize}"
        kvtdinfo = KVTDInfo(
            instid=inst_id,
            blkids=blockids[0],
            cached_tokens=req.num_computed_tokens,
            max_tokens=alltokens,
            d_workers_info=[],
        )
        kvtreq = RKVTDInfo(reqid=req.request_id, migration=True, dinfo=kvtdinfo)

        start_ts = time.monotonic()
        kvtresp = await self._kvt_rpc(srcinfo, kvtreq)

        if kvtresp is None:
            raise RuntimeError("migration backend kvt rpc failed")

        if kvtresp.code != CODE_OK:
            raise CodeError(kvtresp.code, "migration backend kvt rpc failed")

        end_ts = time.monotonic()
        dur_ms = (end_ts - start_ts) * 1000
        logger.info(
            "migrate end kvt: dur_ms=%s, reqid=%s cached=%s max=%s kvtresp=%s",
            dur_ms,
            req.request_id,
            req.num_computed_tokens,
            alltokens,
            kvtresp,
        )
        assert kvtresp.code == CODE_OK
        req.num_computed_tokens = kvtresp.cached
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_operations(self, req: Request) -> tuple[int, int]:
        # logger.info(f">>>DBG: {self._reqstate.get(req.request_id)=}")
        assert req.request_id not in self._reqstate
        self._reqstate[req.request_id] = req

        srcinfo = get_param(req, SRC_INFO)
        return int(srcinfo is not None), 0

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        for reqid in sout.finished_req_ids:
            self._finish_req(reqid)
        return BackendMeta()

    async def async_update_state_after_alloc(
        self, req: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> Optional[IoRet]:
        try:
            ioret = IoRet(n=None)
            srcinfo: tuple[str, int] = get_param(req, SRC_INFO)

            resp = await self._kvt_suspend(srcinfo, req, blocks)
            assert resp.output_token_ids is not None

            req.append_output_token_ids(resp.output_token_ids)
            req.num_computed_tokens = resp.cached
            assert req.num_computed_tokens <= req.num_tokens
            if req.num_computed_tokens >= req.num_tokens:
                assert req.num_computed_tokens >= 1
                gamma = 0
                if (
                        self._cfg.speculative_config
                        and self._cfg.speculative_config.num_speculative_tokens
                ):
                    gamma = self._cfg.speculative_config.num_speculative_tokens
                req.num_computed_tokens -= (1 + gamma)

            migration_end_time = time.monotonic()
            start_ts = get_param(req, "migration_start_time", 0)
            migration_e2e_time_ms = (migration_end_time - start_ts) * 1000
            logger.info(
                "migration end. reqid=%s, duration_ms=%s",
                req.request_id,
                migration_e2e_time_ms,
            )

        except Exception as e:
            logger.warning(
                "migration backend async_update failed. req_id=%s, srcinfo=%s, e=%s",
                req.request_id,
                srcinfo,
                e,
            )
            ioret.ex = e
        finally:
            _g_migrate_in_req_ids.discard(req.request_id)
            if envs.LLUMNIX_DETAILED_MIG_STATUS:
                with _g_migrate_in_req_info_lock:
                    _g_migrate_in_req_info.pop(req.request_id, None)
        return ioret

    # ==============================
    # Worker-side methods
    # ==============================

    async_load_kv = None  # type: ignore


_g_backend: Optional[MigrationBackend] = None


def on_outputs(outputs: dict[int, EngineCoreOutputs]):
    global _g_backend
    if _g_backend is None or not envs.VLLM_USE_MIGRATION_BACKEND_RETURN_TOKEN:
        return
    _g_backend._on_outputs(outputs)
    return
