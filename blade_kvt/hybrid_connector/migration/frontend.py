import asyncio
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

from vllm.v1.engine.core_client import EngineCoreClient

from ..engine_proxy import (
    EngineCoreOutputs,
    EngineCoreRequest,
    MsgpackDecoder,
    MsgpackEncoder,
    PlaceholderModule,
    VllmConfig,
    core_get_param,
    core_update_params,
    get_ip,
    get_logger,
    get_open_port,
)
from ..kvtbackend import get_inst_id
from ..utils import ConnManager, PeerManager, RpcServer, kill_me_if_exception
from . import (
    ABORT_REQS_REQ,
    ABORT_REQS_RESP,
    MIGRATE_TO_REQ,
    MIGRATE_TO_RESP,
    NEW_OUTPUT_REQ,
    NEW_OUTPUT_RESP,
    NEW_REQ_REQ,
    NEW_REQ_RESP,
    OUTPUT_TOKENS_N,
    SRC_INFO,
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


@dataclass
class Res:
    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False


@dataclass
class ReqState:
    req: EngineCoreRequest
    engines: list[tuple[str, int]]
    # engidx, tokens from engine
    engtokens: list[list[int]]
    num_tokens: int = 0


class MigrationCli(EngineCoreClient):
    def __init__(self, vllm_config: VllmConfig):
        self._cfg = vllm_config
        self._loop = asyncio.get_running_loop()
        self._output_q = asyncio.Queue[Union[EngineCoreOutputs, Exception]]()

        self._inst_id = get_inst_id(vllm_config)
        self._naming_url = vllm_config.kv_transfer_config.get_from_extra_config(
            "naming_url", "badbad"
        )
        self._naming_cli = connect_naming(self._inst_id, self._naming_url)
        self._conn_mgr = ConnManager()
        self._pmgr = PeerManager(
            self._naming_cli, None, self._conn_mgr
        )  # None means ALL, empty connpool
        self._pmgr.start(self._loop)

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, ReqState] = {}

        self._rpcsrv = RpcServer(get_open_port(), "MigrationCli")
        self._rpcsrv.register_method(NEW_OUTPUT_REQ, self._on_new_output)
        self._rpcsrv.start(self._loop)
        self._client_idx = ipport2int(get_ip(), self._rpcsrv.port())
        assert is_migration(self._client_idx)

        self._outputdec = MsgpackDecoder(EngineCoreOutputs)
        self._enc = MsgpackEncoder()

        self.resources = Res()
        logger.info("!!!! MigrationCli IS USED. THIS IS ONLY USED IN DEMO/CI")
        return

    async def _do_migrate(
        self, reqid: str, cureng: tuple[str, int], msgbuf, retry: int
    ) -> tuple[str, int]:
        neweng = self._pmgr.get_peer(exclude=cureng)
        if neweng is None:
            raise RuntimeError("no instance")
        logger.info(
            "migrate request. reqid=%s cureng=%s neweng=%s retry=%s",
            reqid,
            cureng,
            neweng,
            retry,
        )
        await self._rpc(neweng, msgbuf, MIGRATE_TO_RESP, retry)
        return neweng

    @kill_me_if_exception
    async def _migrate(self, req: ReqState):
        cureng = req.engines[-1]
        core_update_params(req.req, {SRC_INFO: cureng})
        core_update_params(req.req, {OUTPUT_TOKENS_N: req.num_tokens})
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, MIGRATE_TO_REQ, 0)
        reqbufs = self._enc.encode_into(req.req, msgbuf, 8)
        assert len(reqbufs) == 1  # Need Support MsgpackEncoder.aux_buffers
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)
        reqid = req.req.request_id

        peer_addr_port: Optional[tuple[str, int]] = None
        for idx in range(2):
            try:
                peer_addr_port = await self._do_migrate(reqid, cureng, msgbuf, idx)
                break
            except Exception:
                logger.exception("migrate failed:reqid=%s try=%s", reqid, idx)
            await asyncio.sleep(0)
        if not peer_addr_port:
            logger.warning("migrate:sorry")
            return

        req.engines.append(peer_addr_port)
        return

    ### DO NOT USE AWAIT TO KEEP ATOMIC
    def _do_new_outputs(self, outs: EngineCoreOutputs):
        for o in outs.outputs:
            reqid = o.request_id
            state = self.reqs_in_flight.get(reqid)
            if state is None:
                logger.info("bad out: out=%s", o)
                continue

            newtks = len(o.new_token_ids)
            state.num_tokens += newtks
            engidx = outs.engine_index
            neweng = not state.engtokens or state.engtokens[-1][0] != engidx
            if neweng:
                state.engtokens.append([engidx, newtks])
            else:
                state.engtokens[-1][1] += newtks

            if o.finished:
                self.reqs_in_flight.pop(reqid)
                logger.info("req finised:reqid=%s engtokens=%s", reqid, state.engtokens)
                continue
            else:
                forbidden_migration = core_get_param(
                    self.reqs_in_flight[reqid].req,
                    "__just_debug_forbidden_migration",
                    False,
                )
                if neweng and not forbidden_migration:
                    asyncio.create_task(self._migrate(state))

        return

    async def _on_new_output(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)

        respbuf = struct.pack("=I", NEW_OUTPUT_RESP)
        writer.write(respbuf)

        ### DO NOT USE AWAIT TO KEEP ATOMIC
        outputs: EngineCoreOutputs = self._outputdec.decode(reqbuf)
        logger.info("dbg: outputs=%s", outputs)
        self._do_new_outputs(outputs)
        if outputs.outputs or outputs.scheduler_stats:
            self._output_q.put_nowait(outputs)
        ### DO NOT USE AWAIT TO KEEP ATOMIC

        await writer.drain()
        return

    async def _rpc(
        self, peer_addr_port: tuple[str, int], msgbuf, resp: int, retry: int
    ):
        pconn = await self._conn_mgr.acquire_conn(peer_addr_port, retry > 0)
        pconn[1].write(msgbuf)
        await pconn[1].drain()

        respbuf = await pconn[0].readexactly(4)
        (head,) = struct.unpack("=I", respbuf)
        if head != resp:
            raise RuntimeError(f"invalid resp {head=}")

        self._conn_mgr.release_conn(peer_addr_port, pconn)
        return

    async def _add_req(
        self, reqid: str, msgbuf, retry: int, peer_addr_port: Optional[tuple[str, int]]
    ) -> tuple[str, int]:
        if peer_addr_port is None:
            peer_addr_port = self._pmgr.get_peer()
        if peer_addr_port is None:
            raise RuntimeError("no instance")
        logger.info(
            "migration add request. reqid=%s retry=%s peer=%s",
            reqid,
            retry,
            peer_addr_port,
        )
        await self._rpc(peer_addr_port, msgbuf, NEW_REQ_RESP, retry)
        return peer_addr_port

    async def add_request_async(self, req: EngineCoreRequest) -> None:
        logger.info("add_request_async: req=%s", req)
        req.client_index = self._client_idx
        reqid = req.request_id
        hint_peerid = core_get_param(req, "__just_debug_remote_instid")

        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, NEW_REQ_REQ, 0)
        reqbufs = self._enc.encode_into(req, msgbuf, 8)
        assert len(reqbufs) == 1  # Need Support MsgpackEncoder.aux_buffers
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)

        peer_addr_port: Optional[tuple[str, int]] = None
        for idx in range(2):
            try:
                peer_addr_port = await self._add_req(reqid, msgbuf, idx, hint_peerid)
                break
            except Exception:
                logger.exception("add req failed:reqid=%s try=%s", reqid, idx)
            await asyncio.sleep(0)
        if not peer_addr_port:
            raise RuntimeError("badbad")

        assert reqid not in self.reqs_in_flight
        self.reqs_in_flight[reqid] = ReqState(
            req=req, engines=[peer_addr_port], engtokens=[]
        )
        return

    @kill_me_if_exception
    async def _abort_reqs(self, msgbuf, peer_addr_port: tuple[str, int]):
        ok = False
        for idx in range(2):
            try:
                await self._rpc(peer_addr_port, msgbuf, ABORT_REQS_RESP, idx)
                ok = True
                break
            except Exception:
                logger.exception(
                    "abort req failed. peer_addr_port=%s retry=%s", peer_addr_port, idx
                )
            await asyncio.sleep(0)
        if not ok:
            raise RuntimeError("badbad")
        return

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # peerid: list[reqid]
        peer_reqs = defaultdict(set)
        for reqid in request_ids:
            reqstate = self.reqs_in_flight.pop(reqid, None)
            if reqstate is None:
                logger.warning("abort req: unknown peer: reqid=%s", reqid)
                continue
            for peer_addr_port in reqstate.engines:
                peer_reqs[peer_addr_port].add(reqid)

        for peer_addr_port, reqids_set in peer_reqs.items():
            reqids = list(reqids_set)
            logger.info(
                "abort req: peer_addr_port=%s reqids=%s", peer_addr_port, reqids
            )
            msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
            struct.pack_into("=II", msgbuf, 0, ABORT_REQS_REQ, 0)
            reqbufs = self._enc.encode_into(reqids, msgbuf, 8)
            assert len(reqbufs) == 1
            struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)
            asyncio.create_task(self._abort_reqs(msgbuf, peer_addr_port))
        return

    async def get_output_async(
        self, timeout: Optional[float] = None
    ) -> EngineCoreOutputs:
        outputs = await self._output_q.get()
        if isinstance(outputs, Exception):
            raise outputs from None
        return outputs

    def shutdown(self):
        logger.info("Game Over")
        return

    async def reset_mm_cache_async(self) -> None:
        logger.info("reset_mm_cache_async")
        return


_g_is_frontend: bool = False


def i_am_frontend():
    global _g_is_frontend
    _g_is_frontend = True


def is_frontend():
    global _g_is_frontend
    return _g_is_frontend
