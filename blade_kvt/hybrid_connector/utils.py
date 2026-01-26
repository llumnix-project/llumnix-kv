import asyncio
import os
import random
import struct
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional

import msgspec
import torch
import uvloop

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


# All coroutines used as the main of task must apply this decorator.
def kill_me_if_exception(async_func):
    async def wrapper(*args, **kwargs):
        try:
            # more detail see https://ata.atatech.org/articles/11020396051
            loop = asyncio.get_running_loop()
            if not hasattr(loop, "__kill_me_running_tasks"):
                loop.__kill_me_running_tasks = set()  # type: ignore
            running_task: set[asyncio.Task] = loop.__kill_me_running_tasks  # type: ignore
            task = asyncio.current_task()
            assert task is not None
            running_task.add(task)
            task.add_done_callback(running_task.discard)

            return await async_func(*args, **kwargs)
        except Exception:
            logger.exception("async_func: ex")
            os.abort()

    return wrapper


def _loop_on_ex(_loop, context):
    logger.exception("loop ex. context=%s", context)
    os.abort()


def _asyncio_loop_main(loop, vllm_config, local_rank):
    if vllm_config and local_rank:
        if envs.VLLM_USE_SWAP_ENGINE:
            device = vllm_config.device_config.device
        else:
            device = torch.device(f"cuda:{local_rank}")
        current_platform.set_device(device)

    asyncio.set_event_loop(loop)
    loop.run_forever()
    return


def start_asyncio_thread(name, vllm_config=None, local_rank=None):
    loop = uvloop.new_event_loop()
    loop.set_exception_handler(_loop_on_ex)
    threading.Thread(
        target=_asyncio_loop_main,
        args=(loop, vllm_config, local_rank),
        name=name,
        daemon=True,
    ).start()
    return loop


# A simple RPC implementation
#
# Message Format:
#
#  +------+------------+
#  | head |    body    |
#  +------+------------+
#
# head: 4bytes,
# body: based on the value of "head".

_RpcMethodType = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter], Coroutine[Any, Any, None]
]


class RpcServer:
    def __init__(self, port: int, name: str):
        self._name = name
        self._port = port
        self._methods: dict[int, _RpcMethodType] = {}
        return

    # R/W on self._methods is protected by GIL
    def register_method(self, head: int, callback: _RpcMethodType):
        assert head not in self._methods
        self._methods[head] = callback
        return

    def port(self) -> int:
        return self._port

    def start(self, loop):
        coro = self._main()
        asyncio.run_coroutine_threadsafe(coro, loop)
        return

    @kill_me_if_exception
    async def _client_main(self, reader, writer):
        # Create multiple connections for concurrency,
        try:
            while True:
                headbuf = await reader.readexactly(4)
                (head,) = struct.unpack("=I", headbuf)
                cb = self._methods.get(head, None)
                if cb is not None:
                    await cb(reader, writer)
                    continue
                logger.warning("%s: rpc unknown head. head=%s", self._name, head)
                break
        except asyncio.IncompleteReadError as ex:
            if ex.partial:  # get non-empty bytes object from readexactly
                logger.exception("%s: client main: ex", self._name)
        except Exception:
            logger.exception("%s: client main: ex", self._name)

        writer.close()

        try:
            await writer.wait_closed()
        except ConnectionResetError:
            logger.info("Connection was reset by peer during close. Ignoring.")
        except Exception:
            logger.exception("%s: client main: ex", self._name)

        return

    @kill_me_if_exception
    async def _main(self):
        p = self._port
        logger.info("%s RPC server starting on port %s", self._name, p)
        server = await asyncio.start_server(self._client_main, "0.0.0.0", p)
        async with server:
            await server.serve_forever()
        return


Conn = tuple[asyncio.StreamReader, asyncio.StreamWriter]


class ConnPool:
    def __init__(self, host: str, port: int, max_conn: int):
        self._host = host
        self._port = port
        self._max_conn = max_conn
        self._conns: list[Conn] = []

    async def _acquire(self, force_new: bool = False) -> Conn:
        if not force_new and self._conns:
            return self._conns.pop()
        return await asyncio.open_connection(self._host, self._port)

    def _release(self, conn: Conn):
        if len(self._conns) >= self._max_conn:
            # close the writer
            writer = conn[1]
            if not writer.is_closing():
                writer.close()
            return
        self._conns.append(conn)
        return

    acquire = _acquire
    release = _release

    def host(self):
        return self._host

    def port(self):
        return self._port


class ConnManager:
    """Connection manager that handles multiple connection pools"""

    def __init__(self, capacity = 512):
        self._cap = capacity
        self._connpool: OrderedDict[tuple[str, int], ConnPool] = OrderedDict()

    async def acquire_conn(
        self, addr_port: tuple[str, int], force_new: bool = False
    ) -> Conn:
        """Acquire connection using addr:port as key"""
        addr, port = addr_port

        pool = self._connpool.get(addr_port, None)
        if pool is None:
            pool = ConnPool(addr, port, 3)
            self._connpool[addr_port] = pool
            self._try_shrink()
        else:
            self._touch(addr_port)
        conn = await pool._acquire(force_new)
        return conn

    def release_conn(self, addr_port: tuple[str, int], conn: Conn):
        """Release connection"""
        addr, port = addr_port

        connpool = self._connpool.get(addr_port)
        if connpool is None:
            return
        connpool._release(conn)
        return

    def pop(self, addr_port: tuple[str, int], default=None) -> Optional[Conn]:
        connpool = self._connpool.pop(addr_port, default)
        return connpool

    def _touch(self, addr_port: tuple[str, int]):
        self._connpool.move_to_end(addr_port)
        return

    def _try_shrink(self):
        while len(self._connpool) > self._cap:
            self._connpool.popitem(last=False)
        return


@dataclass
class PeerInfo:
    # P,D common fields
    role: str = ""
    tpsize: int = -1
    # when this peer is registered
    ctime_us: int = -1

    # only used in P
    addr: str = ""
    port: int = 33000
    dprank: int = 0

    def serialize(self) -> str:
        s = f"{self.role}:{self.tpsize}:{self.ctime_us}:{self.addr}:{self.port}:{self.dprank}"  # noqa: E501
        return s

    @classmethod
    def deserialize(cls, data: str) -> "PeerInfo":
        parts = data.split(":")
        if len(parts) != 6:
            raise ValueError(f"PeerInfo.deserialize: BadVal={data}")

        role = parts[0]
        tpsize = int(parts[1])
        ctime_us = int(parts[2])
        addr = parts[3]
        port = int(parts[4])
        dprank = int(parts[5])

        return cls(
            role=role,
            tpsize=tpsize,
            dprank=dprank,
            ctime_us=ctime_us,
            addr=addr,
            port=port,
        )


class PeerMap:
    def __init__(self):
        self._addr2id: dict[tuple[str, int], str] = {}
        self._id2info: dict[str, PeerInfo] = {}

    def __bool__(self):
        return bool(self._id2info)

    def __setitem__(self, peerid: str, peer: PeerInfo):
        old_info = self._id2info.pop(peerid, None)
        if old_info is not None:
            old_addr = (old_info.addr, old_info.port)
            del self._addr2id[old_addr]

        new_addr = (peer.addr, peer.port)
        self._addr2id[new_addr] = peerid
        self._id2info[peerid] = peer

    def __getitem__(self, peerid: str) -> PeerInfo:
        return self._id2info[peerid]

    def get_peerid(self, addr: tuple[str, int]) -> Optional[str]:
        return self._addr2id.get(addr)

    def items(self):
        return self._id2info.items()

    def keys(self, exclude: Optional[tuple[str, int]] = None) -> list[str]:
        if exclude is None:
            return list(self._id2info.keys())
        res = []
        for peerid, peerinfo in self._id2info.items():
            if peerinfo.addr != exclude[0] or peerinfo.port != exclude[1]:
                res.append(peerid)
        return res

    def pop(self, peerid: str, default=None) -> Optional[PeerInfo]:
        peer_info = self._id2info.pop(peerid, default)
        del self._addr2id[(peer_info.addr, peer_info.port)]
        return peer_info

    def update(self, peerinfos: dict[str, PeerInfo]) -> None:
        for peerid, peerinfo in peerinfos.items():
            self.__setitem__(peerid, peerinfo)


def _get_peer_id(kvt_inst_id: str, dp_rank: int, tp_size: int) -> str:
    return f"{kvt_inst_id}|{dp_rank}|{tp_size}"


class PeerManager:
    def __init__(
        self, naming_cli, interested_role: Optional[str], conn_mgr: ConnManager
    ):
        self._naming_cli = naming_cli
        # None means ALL
        self._interested_role = interested_role
        # Modifications to 'running' and 'connpool' should be atomic,
        # meaning there shouldn't be any operations like 'await' in between.
        # It must be ensured that 'connpool[instid]' always corresponds to
        # the latest version of 'running[instid]', especially considering
        # scenarios like in-place restarts.
        self._running: PeerMap = PeerMap()
        self._conn_mgr = conn_mgr

    def _list_instance(self) -> list[tuple[str, PeerInfo]]:
        ret: list[tuple[str, PeerInfo]] = []
        all_peers = self._naming_cli.list()
        for peer_name in all_peers:
            peer_endpoints = self._naming_cli.search(peer_name, "endpoint")
            for peer_endpoint in peer_endpoints:
                peer_info = PeerInfo.deserialize(peer_endpoint)
                if (
                    self._interested_role is not None
                    and peer_info.role != self._interested_role
                ):
                    break
                peer_id = _get_peer_id(peer_name, peer_info.dprank, peer_info.tpsize)
                ret.append((peer_id, peer_info))
        return ret

    async def _do_main(self):
        peers = self._list_instance()
        new_running: dict[str, PeerInfo] = {}
        for peer_name, peer_info in peers:
            new_running[peer_name] = peer_info

        # Use updated_peers rather than inplace update to avoid the exception
        # "RuntimeError: dictionary keys changed during iteration".
        exited_peers, updated_peers = [], {}
        for peerid, peer in self._running.items():
            newpeer = new_running.pop(peerid, None)
            if newpeer is None:
                exited_peers.append(peerid)
                logger.info("PeerManager peer exited. peer=%s", peer)
                continue
            if peer.ctime_us != newpeer.ctime_us:
                logger.info(
                    "PeerManager peer updated. before=%s after=%s", peer, newpeer
                )
                updated_peers[peerid] = newpeer
                self._conn_mgr.pop(peerid, None)  # remove stale conn
                continue
        for peerid in exited_peers:
            self._running.pop(peerid)
            self._conn_mgr.pop(peerid, None)  # remove stale conn
        for peerid, peer in updated_peers.items():
            self._running[peerid] = peer

        if new_running:
            logger.info("PeerManager peer added. peers=%s", new_running)
            self._running.update(new_running)
        return

    @kill_me_if_exception
    async def _main(self):
        while True:
            try:
                await self._do_main()
            except Exception:
                logger.exception("PeerManager._main: exception")
            await asyncio.sleep(7)

    def start(self, loop):
        coro = self._main()
        asyncio.run_coroutine_threadsafe(coro, loop)
        return

    # None means no available peer
    def get_peer(
        self, exclude: Optional[tuple[str, int]] = None
    ) -> Optional[tuple[str, int]]:
        peers = self._running.keys(exclude)
        if not peers:
            return None

        peer_id = random.choice(peers)
        peer_info = self._running[peer_id]
        return (peer_info.addr, peer_info.port)


class IoRet(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    reqid: Optional[str] = None
    ex: Optional[Exception] = None
    n: Optional[int] = None


class IoState:
    def __init__(self, signals_per_worker: int = 1):
        # Number of signals each worker (tprank) needs to send before being ready
        self.signals_per_worker = signals_per_worker
        # worker_tprank -> list of iorets received from this worker
        self._worker_signals: dict[int, list[IoRet]] = {}

    @property
    def _ready_workers(self) -> list[tuple[int, IoRet]]:
        result = []
        for tprank, iorets in self._worker_signals.items():
            if len(iorets) >= self.signals_per_worker:
                result.append((tprank, iorets[-1]))
        return result

    def merge(self) -> IoRet:
        # can not merge signals from multiple backends
        assert self.signals_per_worker == 1

        ready = self._ready_workers
        assert ready
        _, ret = ready[0]
        for idx in range(1, len(ready)):
            _, wio = ready[idx]
            # assert ret.reqid == wio.reqid
            if wio.ex is not None and ret.ex is None:
                ret.ex = wio.ex
                break

            if ret.n is None:
                continue

            if wio.n is None:
                ret.n = None
                continue

            if ret.n > wio.n:
                ret.n = wio.n
        return ret


def try_advance(
    state_dict: dict[str, Any], ioret: IoRet, worker_tprank: int, tpsize: int
) -> Optional[Any]:
    reqid = ioret.reqid
    assert reqid is not None
    state: Optional[IoState] = state_dict.get(reqid)
    if state is None:
        logger.warning("try_advance: unknown ioret=%s tprank=%s", ioret, worker_tprank)
        return None

    if worker_tprank not in state._worker_signals:
        state._worker_signals[worker_tprank] = []
    
    worker_signals = state._worker_signals[worker_tprank]
    
    if len(worker_signals) >= state.signals_per_worker:
        logger.warning(
            "try_advance: dup worker=%s ioret=%s signals=%s",
            worker_tprank,
            ioret,
            worker_signals,
        )
        return None

    worker_signals.append(ioret)
    
    ready_workers_count = sum(
        1 for signals in state._worker_signals.values()
        if len(signals) >= state.signals_per_worker
    )
    
    if ready_workers_count < tpsize:
        return None
    return state


class CodeError(Exception):
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg


# Body:
# +-----------+-----+-----+--------+-----+--------+------+------+
# | worker tp | n   | len | reqid  | len | reqid  | plen | plen |
# +-----------+-----+-----+--------+-----+--------+------+------+
# worker tp: 4bytes, worker tp rank.
# n: 4bytes, number of (len, reqid)
# len: 4 bytes
# reqid: len, utf-8 encoded
# When a Python str is passed from Python to a C++ function that
# accepts std::string or char * as arguments, pybind11 will encode
# the Python string to UTF-8.
# plen: 4 bytes. If worker_tprank has the HAS_PLEN flag, there will be n
# `plen` fields, each storing the actual number of IO tokens for every req.
#
# ver2:
# +-----------+-----+-----+-----+-----+-------+-----+-----+-----+--------+
# | worker tp | n   | plen| code| len | reqid | plen| code| len | reqid  |
# +-----------+-----+-----+-----+-----+-------+-----+-----+-----+--------+
async def handle_done_req(
    reader, writer, cb: Callable[[int, IoRet], Coroutine[Any, Any, None]], respcode: int
):
    HAS_VER2 = 0x40000000
    HAS_PLEN = 0x80000000
    bodybuf = await reader.readexactly(4 + 4)
    worker_tprank, num_req = struct.unpack("=II", bodybuf)
    has_plen = worker_tprank & HAS_PLEN
    has_ver2 = worker_tprank & HAS_VER2
    worker_tprank = worker_tprank & 0xFFFF  # worker_tprank < 65536

    reqid: str = ""
    iorets: list[IoRet] = []
    if has_ver2:
        for _ in range(num_req):
            lenbuf = await reader.readexactly(4 + 4 + 4)
            plen, code, rlen = struct.unpack("=III", lenbuf)
            reqidbuf = await reader.readexactly(rlen)
            reqid = reqidbuf.decode("utf-8")
            ex = None
            if code != 0:
                ex = CodeError(code, "done req")
            iorets.append(IoRet(reqid=reqid, n=plen, ex=ex))
    else:
        for _ in range(num_req):
            lenbuf = await reader.readexactly(4)
            (len,) = struct.unpack("=I", lenbuf)
            reqidbuf = await reader.readexactly(len)
            reqid = reqidbuf.decode("utf-8")
            iorets.append(IoRet(reqid=reqid))
        if has_plen:
            for idx in range(num_req):
                lenbuf = await reader.readexactly(4)
                (len,) = struct.unpack("=I", lenbuf)
                iorets[idx].n = len

    resp = struct.pack("=I", respcode)
    writer.write(resp)

    tasks = []
    for ioret in iorets:
        tasks.append(cb(worker_tprank, ioret))
    await asyncio.gather(*tasks)

    await writer.drain()
    return
