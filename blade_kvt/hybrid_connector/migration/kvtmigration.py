import struct
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import msgspec
import torch

from vllm import envs

from .. import BackendMeta, HybridBackend
from ..engine_proxy import (
    KVCacheBlocks,
    KVCacheConfig,
    KVConnectorRole,
    Request,
    SchedulerOutput,
    VllmConfig,
    get_param,
    sched_rpc_server,
)

# yapf: disable
from ..kvtbackend import (
    CODE_OK,
    CODE_REQNOTFOUND,
    P_KVT_STATE,
    P_REMOTE_DECODE,
    PREFILL_RESP,
    DBackend,
    PBackend,
    RKVTDInfo,
    _g_migrate_out_req_ids,
    _g_migrate_out_req_info,
    _g_migrate_out_req_info_lock,
)

# yapf: enable
from ..utils import IoRet
from . import KVT_SUSPEND_REQ, SRC_INFO
from .backend import MigrationBackend, SuspendReq


class KVTMigration(HybridBackend):
    def __init__(
            self, cfg: VllmConfig, role: KVConnectorRole,
            kv_cache_config:KVCacheConfig = None):
        self._p = PBackend(cfg, role, kv_cache_config)
        self._d = DBackend(cfg, role, kv_cache_config)
        self._m = MigrationBackend(cfg, role, self._p.naming_cli())
        
        if role == KVConnectorRole.SCHEDULER:
            rpcsrv = sched_rpc_server()
            rpcsrv.register_method(KVT_SUSPEND_REQ, self._on_transfer_suspend_kv)
            self._packenc = msgspec.msgpack.Encoder()
            self._dec = msgspec.msgpack.Decoder()
            self._kvtreqdec = msgspec.msgpack.Decoder(RKVTDInfo)

        return

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._p.register_kv_caches(kv_caches)
        self._d.register_kv_caches(kv_caches)
        return

    # worker thread
    # bind_backend_metadata([R]) happen before async_load_kv(R)
    def bind_backend_metadata(self, meta: BackendMeta):
        self._p.bind_backend_metadata(meta)
        return

    # worker thread
    def clear_backend_metadata(self):
        self._p.clear_backend_metadata()
        return

    async_load_kv = None  # type: ignore

    # worker thread
    def async_save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, m: BackendMeta
    ) -> Optional[AsyncGenerator[str, None]]:
        r = self._p.async_save_kv_layer(layer_name, kv_layer, m)
        assert r is None
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================
    async def async_get_num_new_matched_tokens(
        self, req: Request, num_computed_tokens: int
    ) -> int:
        if get_param(req, SRC_INFO):
            return 0
        r = await self._d.async_get_num_new_matched_tokens(req, num_computed_tokens)
        return r

    async def async_update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> Optional[IoRet]:
        if get_param(request, SRC_INFO):
            r = await self._m.async_update_state_after_alloc(
                request, blocks, num_external_tokens
            )
        else:
            r = await self._d.async_update_state_after_alloc(
                request, blocks, num_external_tokens
            )
        return r

    def get_operations(self, req: Request) -> tuple[int, int]:
        m_load, m_save = self._m.get_operations(req)
        if get_param(req, P_KVT_STATE, None) is not None \
            or get_param(req, P_REMOTE_DECODE, None) is not None:
            kvt_load, kvt_save = self._p.get_operations(req)
        else:
            kvt_load, kvt_save = self._d.get_operations(req)
        return m_load + kvt_load, m_save + kvt_save

    def build_backend_meta(self, sout: SchedulerOutput) -> BackendMeta:
        ret = self._d.build_backend_meta(sout)
        assert isinstance(ret, BackendMeta)
        ret = self._m.build_backend_meta(sout)
        assert isinstance(ret, BackendMeta)
        kvt = self._p.build_backend_meta(sout)
        return kvt
    
    async def _on_transfer_suspend_kv(self, reader, writer):
        bodylenbuf = await reader.readexactly(4)
        (bodylen,) = struct.unpack("=I", bodylenbuf)
        reqbuf = await reader.readexactly(bodylen)
        req: RKVTDInfo = self._kvtreqdec.decode(reqbuf)
        resp = await self._p.submit_transfer_kv(req)
        if resp.computed == -1:
            resp.code = CODE_REQNOTFOUND
        if resp.code == CODE_OK:
            susreq = SuspendReq(reqid=req.reqid)
            susresp = await self._m.do_suspend(susreq)
            if susresp.code != CODE_OK:
                resp.code = susresp.code
            resp.output_token_ids = susresp.output_token_ids
        else:
            if self.get_request(req.reqid) is None:
                _g_migrate_out_req_ids.discard(req.reqid)
                if envs.LLUMNIX_DETAILED_MIG_STATUS:
                    with _g_migrate_out_req_info_lock:
                        _g_migrate_out_req_info.pop(req.reqid)

        respbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", respbuf, 0, PREFILL_RESP, 0)
        self._packenc.encode_into(resp, respbuf, 8)
        struct.pack_into("=I", respbuf, 4, len(respbuf) - 8)
        writer.write(respbuf)

        await writer.drain()
        return
    