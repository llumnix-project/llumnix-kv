import asyncio
import enum
import os
import socket
import struct
from typing import List, Optional, Union

import torch
import logging


# Default stepid and substepid for API compatibility with older vLLM
DEFAULT_STEPID = 1024
DEFAULT_SUBSTEPID = 0

from blade_kvt.kvtransfer_ops import (
    TransferProtocol,
    add_target,
    flush_send,
    init_kv_transfer_client,
    init_kv_transfer_server,
    lib_support_transfer_protocols,
    notify_event_record,
    start_send,
    submit_delta_send,
    submit_req_send2,
    start_req_send,
    start_send_substep,
    ReqMeta,
)


logger = logging.getLogger("blade_kvt")


# Constants for send done request
SEND_DONE_REQ = 0x20181219
SEND_DONE_RESP = 0x91218102
SAVE_DONE2_REQ = 0x20181223

SEND_DONE_HEAD_KIND = 1
SEND_SAVE_DONE_HEAD_KIND = 2

CODE_INTERNALERROR = 500


def _get_send_done_addr() -> Optional[tuple[str, int]]:
    """Parse BLLM_KVTRANS_SEND_DONE_ADDR env, return (ip, port) or None."""
    addr_str = os.environ.get("BLLM_KVTRANS_SEND_DONE_ADDR")
    if not addr_str:
        return None
    parts = addr_str.split(":")
    if len(parts) != 2:
        return None
    return parts[0], int(parts[1])


def _get_send_done_head_kind() -> int:
    """Get BLLM_KVTRANS_SDH_KIND env, default to SEND_SAVE_DONE_HEAD_KIND."""
    val = os.environ.get("BLLM_KVTRANS_SDH_KIND")
    if val is None:
        return SEND_SAVE_DONE_HEAD_KIND
    return int(val)


@enum.unique
class KVTransferProtocolType(enum.Enum):
    RDMA_DIRECT = enum.auto()
    TCP = enum.auto()
    UNKNOWN = enum.auto()

    def to_ops_protocol(self) -> TransferProtocol:
        match self:
            case KVTransferProtocolType.RDMA_DIRECT:
                return TransferProtocol(TransferProtocol.Kind.RDMA_DIRECT)
            case KVTransferProtocolType.TCP:
                return TransferProtocol(TransferProtocol.Kind.TCP)
            case _:
                raise RuntimeError("unknown protocol type: " + self.name)

    @classmethod
    def to_protocol_type(cls, protocol_kind: str) -> "KVTransferProtocolType":
        match protocol_kind:
            case "rdma":
                return KVTransferProtocolType.RDMA_DIRECT
            case "tcp":
                return KVTransferProtocolType.TCP
            case _:
                return KVTransferProtocolType.UNKNOWN


def support_transfers_protocols() -> List[KVTransferProtocolType]:
    supports = []
    for p in lib_support_transfer_protocols():
        match p.type:
            case TransferProtocol.Kind.RDMA_DIRECT:
                supports.append(KVTransferProtocolType.RDMA_DIRECT)
            case TransferProtocol.Kind.TCP:
                supports.append(KVTransferProtocolType.TCP)
            case _:
                supports.append(KVTransferProtocolType.UNKNOWN)
    return supports


def _get_layer_num_blocks(layers: List[torch.Tensor], block_bytes: int):
    # No need to check shape, we can just use layer_size/block_bytes to calculate num_blocks
    if block_bytes <= 0:
        raise ValueError(f"block_bytes must be positive, got {block_bytes}")

    first = layers[0]
    first_signature = (
        first.shape,
        first.stride(),
        first.dtype,
        first.device,
        first.nbytes,
    )
    for layer_idx, layer in enumerate(layers):
        signature = (
            layer.shape,
            layer.stride(),
            layer.dtype,
            layer.device,
            layer.nbytes,
        )
        if signature != first_signature:
            raise ValueError(
                "All registered KV layers must have identical layout; "
                f"layer_idx={layer_idx}, expected={first_signature}, "
                f"actual={signature}"
            )
        if layer.nbytes % block_bytes != 0:
            raise ValueError(
                "KV layer size is not divisible by block_bytes; "
                f"layer_idx={layer_idx}, nbytes={layer.nbytes}, "
                f"block_bytes={block_bytes}, shape={tuple(layer.shape)}, "
                f"stride={layer.stride()}"
            )
    return first.nbytes // block_bytes

class KVTransferClient:
    def __init__(
        self,
        inst_id: str,
        tp_size: int,
        worker_id: int,
        worker_tp_rank: int,
        block_bytes: Union[List[int], int],
        token_bytes: Union[List[int], int],
        naming_url: str,
        layers: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        protocols: List[KVTransferProtocolType],
        num_kv_heads: int = -1,
        num_gdn_layers: int = 3,
        indexer_blk_ntpb: int = 0,
        hybrid_indexer_token_size: int = 0,
        gdn_conv_elem_size: int = 1,
        gdn_ssm_elem_size: int = 1,
        conv_state_shape: Optional[List[int]] = None,
        ssm_state_shape: Optional[List[int]] = None,
        gdn_conv_channel_dims: Optional[List[int]] = None,
        kda_conv_dim_first: bool = False,
        hybrid_attn_token_size: int = 0,
        kda_page_stride: int = 0,
        attn_pack_size: int = 1,
        num_ple_layers: int = 0,
        ple_block_group: int = 0,
        ple_conv_elem_size: int = 1,
        ple_conv_state_shape: Optional[List[int]] = None,
        ple_conv_channel_dims: Optional[List[int]] = None,
    ):
        """
        Create and init a client used to send kv cache data to remote instances;

        Args:
                inst_id(int): the id of current instance in cluster;
                tp_size(int): the size of tensor parallel group on current instance;
                worker_id(int): the id of the current worker on current instance;
                worker_tp_rank(int): the rank of current worker among tensor parallel group;
                block_bytes(List[int]): the size in bytes of a block occupied in kv cache allocated, align with layers;
                token_bytes(List[int]): the size in bytes of one token occupied in a block, align with layers;
                naming_url(str): the address of the naming service used to register(find) workers endpoint in cluster;
                layers(List[List[torch.Tensor]]): the kv cache data by layers, each layer may contain multiple cache tensors;
                protocols(List[TransferProtocol]): the transfer protocols will be checked and initialized; They can be used to transfer kv data in later; An empty list is supported, which means all library supported transfer protocols
                will be checked and initialized;
        """
        # older version vllm, should only contain simple model architecture
        if isinstance(block_bytes, int) and isinstance(token_bytes, int):
            block_bytes = [block_bytes]
            token_bytes = [token_bytes]
            layers = [[layer] for layer in layers]
        else:
            if not isinstance(block_bytes, list) or not isinstance(token_bytes, list):
                raise TypeError("block_bytes and token_bytes must both be int or list")

        if any(len(layer) != len(layers[0]) for layer in layers):
            raise ValueError("All layers must have the same number of cache tensors")
        if not len(block_bytes) == len(token_bytes) == len(layers[0]):
            raise ValueError(
                "block_bytes and token_bytes must align with the number of "
                "cache tensors per layer"
            )

        layer_num_blocks_list = [
            _get_layer_num_blocks([
                layer[cache_index] for layer in layers
            ], block_bytes[cache_index]) for cache_index in range(len(block_bytes))
        ]
        if not all(layer_num_blocks == layer_num_blocks_list[0]
                   for layer_num_blocks in layer_num_blocks_list):
            raise ValueError(
                "All cache tensors in one layer must have the same number "
                f"of blocks, got {layer_num_blocks_list}"
            )
        layer_num_blocks = layer_num_blocks_list[0]

        device_id = layers[0][0].get_device()
        logger.info(
            "init kvt client:"
            " instid=%s tpsize=%s wid=%s wrank=%s"
            " blockbytes=%s tokenbytes=%s"
            " naming=%s protocols=%s"
            " layer_shape=%s layer_dtype=%s device_id=%s"
            " num_blocks=%s num_layers=%s",
            inst_id, tp_size, worker_id, worker_tp_rank,
            block_bytes, token_bytes,
            naming_url, protocols,
            [cache.shape for cache in layers[0]],
            [cache.dtype for cache in layers[0]],
            [cache.device for cache in layers[0]],
            layer_num_blocks, len(layers)
        )
        logger.info(
            "init kvt client memory layout: "
            "data_ptrs=%s nbytes=%s strides=%s storage_offsets=%s "
            "contiguous=%s registered_bytes_per_tensor=%s",
            [cache.data_ptr() for cache in layers[0]],
            [cache.nbytes for cache in layers[0]],
            [cache.stride() for cache in layers[0]],
            [cache.storage_offset() for cache in layers[0]],
            [cache.is_contiguous() for cache in layers[0]],
            [layer_num_blocks * size for size in block_bytes],
        )

        self._num_layers = len(layers)
        self._init_events()
        layer_addrs = [[l.data_ptr() for l in layer] for layer in layers]
        ops_protocols = [p.to_ops_protocol() for p in protocols]
        init_kv_transfer_client(
            inst_id,
            tp_size,
            worker_id,
            worker_tp_rank,
            device_id,
            block_bytes,
            token_bytes,
            layer_num_blocks,
            naming_url,
            self._event_addrs,
            layer_addrs,
            ops_protocols,
            num_kv_heads=num_kv_heads,
            num_gdn_layers=num_gdn_layers,
            indexer_blk_ntpb=indexer_blk_ntpb,
            hybrid_indexer_token_size=hybrid_indexer_token_size,
            gdn_conv_elem_size=gdn_conv_elem_size,
            gdn_ssm_elem_size=gdn_ssm_elem_size,
            conv_state_shape=conv_state_shape if conv_state_shape is not None else [],
            ssm_state_shape=ssm_state_shape if ssm_state_shape is not None else [],
            gdn_conv_channel_dims=gdn_conv_channel_dims if gdn_conv_channel_dims is not None else [],
            kda_conv_dim_first=kda_conv_dim_first,
            hybrid_attn_token_size=hybrid_attn_token_size,
            kda_page_stride=kda_page_stride,
            attn_pack_size=attn_pack_size,
            num_ple_layers=num_ple_layers,
            ple_block_group=ple_block_group,
            ple_conv_elem_size=ple_conv_elem_size,
            ple_conv_state_shape=ple_conv_state_shape if ple_conv_state_shape is not None else [],
            ple_conv_channel_dims=ple_conv_channel_dims if ple_conv_channel_dims is not None else [],
        )
        self._inited = True
        # None means that start_send is not invoked.
        self._cur_step_id: Optional[int] = None
        self._worker_tp_rank = worker_tp_rank
        self._async_sched_warned = False

    def _init_events(self):
        self._events = [torch.cuda.Event() for _ in range(self._num_layers)]
        for _event in self._events:
            _event.record()
        self._event_addrs = [_event.cuda_event for _event in self._events]

    def add_worker(
        self,
        inst_id: str,
        worker_id: int,
        start_layer: int,
        num_layers: int,
        protocol: Optional[KVTransferProtocolType],
    ):
        """
        Add a target remote worker which the client will send data to it later;
        Args:
            inst_id(int): the id of remote instance in cluster;
            worker_id(int): the id of the remote worker on remote instance;
            start_layer(int): from which layer of kv cache will be sent to this remote worker;
            num_layers(int): how many layers of kv cache will be sent to this remote worker;
            protocol(Optional[TransferProtocol]): the transfer protocols will be used to transfer kv data to the target worker, if not set, the library will choose one by itself;
        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")

        ops_protocol = None
        if protocol is not None:
            ops_protocol = protocol.to_ops_protocol()

        add_target(inst_id, worker_id, start_layer, num_layers, ops_protocol)

    def record_event(self, layer_id: int, stream=None):
        """
        Record event to after kv activations generated in each layer;
        """
        if self._cur_step_id is None:
            # warmup may be invoke record_event directly.
            return
        self._events[layer_id].record(stream)
        notify_event_record(self._cur_step_id)

    def submit_req_send(
        self,
        dst_inst_id: str,
        dst_worker_id: int,
        req_id: str,
        new_tokens: int,
        has_last_token: bool,
        src_block_ids: Union[List[List[int]], List[int]],
        dst_block_ids: Union[List[List[int]], List[int]],
        dst_worker_info: Optional[str] = None,
    ):
        if isinstance(src_block_ids, list) and src_block_ids and isinstance(src_block_ids[0], int):
            # older version vllm: list[int] -> list[list[int]]
            assert isinstance(dst_block_ids, list) and dst_block_ids and isinstance(dst_block_ids[0], int)
            src_block_ids = [src_block_ids]
            dst_block_ids = [dst_block_ids]
        return self.submit_req_send2(
            dst_inst_id, dst_worker_id, req_id, 0, new_tokens, has_last_token, src_block_ids, dst_block_ids, dst_worker_info
        )

    def start_req_send(self, metas: list[ReqMeta], stepid: int = DEFAULT_STEPID, substepid: int = DEFAULT_SUBSTEPID):
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        start_req_send(metas, stepid, substepid)

    def submit_req_send2(
        self,
        dst_inst_id: str,
        dst_worker_id: int,
        req_id: str,
        seen_tokens: int,
        new_tokens: int,
        has_last_token: bool,
        src_block_ids: Union[List[List[int]], List[int]],
        dst_block_ids: Union[List[List[int]], List[int]],
        dst_worker_info: Optional[str] = None,
        stepid: int = DEFAULT_STEPID,
    ):
        """
        Submit requests which need to send kv to remote;
        This function call just submit a task to the underlying transfer module,
        won't block current python thread, nor start any data transfer progress.

        Args:
            dst_inst_id(int): the id of remote instance;
            dst_worker_id(int): the id of remote worker;
            req_id(str): the id of request, the length of id should be less than 255;
            seen_tokens(int): number of tokens that have been seen previously;
            new_tokens(int): number of tokens that need to send,
                                tokens from [seen_tokens, seen_tokens + new_tokens) will be sent;
            has_last_token(bool): indicate whether the last token is included in this send;
            src_block_ids(List[int]): id of blocks of the request on current worker;
            dst_block_ids(List[int]): id of blocks on remote worker the request want to send to;
            dst_worker_info(Optional[str]): optional worker info for the destination worker;
            stepid(int): step identifier from upper layer, default 1024 for compatibility;
            substepid(int): substep identifier, default 0 for compatibility;
        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        if isinstance(src_block_ids, list) and src_block_ids and isinstance(src_block_ids[0], int):
            # older verson vllm
            assert isinstance(dst_block_ids, list) and dst_block_ids and isinstance(dst_block_ids[0], int)
            src_block_ids = [src_block_ids]
            dst_block_ids = [dst_block_ids]
        submit_req_send2(
            dst_inst_id, dst_worker_id, req_id, seen_tokens, new_tokens, has_last_token,
            src_block_ids, dst_block_ids, dst_worker_info, stepid, 0
        )

    def submit_delta_send(
        self,
        req_id: str,
        seen_tokens: int,
        new_tokens: int,
        has_last_token: bool,
        stepid: int = DEFAULT_STEPID,
    ):
        """
        Submit KV data send of new tokens of a previously submitted request;
        Args:
            req_id(str): the id of previously submitted request;
            seen_tokens: number of tokens has been submitted to send previously;
            new_tokens: number of tokens need to be sent this time;
            has_last_token: indicate whether the last token is included in this send;
            stepid(int): step identifier from upper layer, default 1024 for compatibility;
            substepid(int): substep identifier, default 0 for compatibility;
        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        submit_delta_send(req_id, seen_tokens, new_tokens, has_last_token, stepid, 0)

    def start_send_step(self, stepid: int = DEFAULT_STEPID, sched_tokens: int = 1):
        """
        Start to send kv data of requests submitted previously layer by layer;
        This function just tell the underlying transfer module to start the transfer progress,
        and won't wait the progress to finish, so the function call will return immediately
        without any blocking;

        Args:
            stepid(int): step identifier from upper layer, default 1024 for compatibility;
            sched_tokens(int): number of scheduled tokens for this step, default 1;
                - sched_tokens == 0: skip Step/StepGuard creation, return EMPTY_STEP_ID
                - sched_tokens > 0: create Step/StepGuard even if targets_tasks_buf_ is empty

        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")

        if self._cur_step_id is not None:
            raise RuntimeError("start_send_step has been called multiple times")

        # see EMPTY_STEP_ID in client.cpp
        EMPTY_STEP_ID = 9223372036854775807

        # assert self._cur_step_id is None
        step_id = start_send(stepid, sched_tokens)
        if step_id != EMPTY_STEP_ID:
            assert step_id == stepid
            self._cur_step_id = step_id
        # self._cur_step_id = start_send()

    def start_send_substep(
        self,
        stepid: int,
        substepid: int,
        metas: list[ReqMeta],
    ) -> None:
        """
        Submit bypass substep metas to the current step.
        This method is called from disaggw thread.

        Args:
            stepid(int): step identifier, must match current step;
            substepid(int): substep identifier, must be > 0 and monotonically increasing;
            metas(list[ReqMeta]): list of ReqMeta with has_freeze=false;
        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        start_send_substep(stepid, substepid, metas)

    def flush_send_step(self):
        """
        After calling 'start_send_step', the underlying transfer module start to transfer
        kv data of submitted requests. User can call this function to check whether the
        transfer progress is finished.

        Note:
            As a transfer progress always include two part: send and receive. If this
            function return True, it only promises that all data of submitted requests
            before this step had been sent out successfully.

        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")

        if self._cur_step_id is None:
            # _disagg_step -> post_step
            return

        # Without async scheduling, model forward should have completed
        # before flush_send is called. Check whether the last-layer event
        # is ready.
        if not self._events[-1].query() and not self._async_sched_warned:
            self._async_sched_warned = True
            logger.warning(
                "flush_send_step called before last layer event is ready; "
                "this indicates async sched may be enabled but not properly supported",
                stack_info=True,
            )

        flush_send(self._cur_step_id)
        self._cur_step_id = None

    async def send_error_done_req(
        self,
        reqids: list[str],
        plen: int = 0,
    ):
        """Send SEND_DONE_REQ or SAVE_DONE2_REQ with CODE_INTERNALERROR.

        This method is used to notify PBackend scheduler when _get_dist fails
        during worker-side processing. It mimics the rpc_send_done logic in
        tx_stub.cpp but always sends CODE_INTERNALERROR (500).

        Args:
            reqids: List of request IDs that failed.
            plen: The plen value (default 0, unused for error case).
        """
        if not reqids:
            return

        addr = _get_send_done_addr()
        assert addr is not None, "send_error_done_req: BLLM_KVTRANS_SEND_DONE_ADDR not set"

        ip, port = addr
        has_ver2 = 0x40000000
        worker_tp_rank = self._worker_tp_rank
        assert worker_tp_rank < 0xFFFF, f"worker_tp_rank too large: {worker_tp_rank}"
        worker_tp_rank_flags = worker_tp_rank | has_ver2
        num_req = len(reqids)

        # Build SEND_DONE_REQ body
        body = bytearray()
        body += struct.pack(
            "=III",
            SEND_DONE_REQ,
            worker_tp_rank_flags,
            num_req,
        )
        for reqid in reqids:
            reqid_bytes = reqid.encode("utf-8")
            rlen = len(reqid_bytes)
            body += struct.pack(
                "=III",
                plen,
                CODE_INTERNALERROR,
                rlen,
            )
            body += reqid_bytes

        # If SEND_SAVE_DONE_HEAD_KIND, append a copy with SAVE_DONE2_REQ header
        head_kind = _get_send_done_head_kind()
        respsize = 4
        if head_kind == SEND_SAVE_DONE_HEAD_KIND:
            reqsize = len(body)
            # Copy the entire body and change header to SAVE_DONE2_REQ
            body.extend(body)
            # Overwrite the second copy's header with SAVE_DONE2_REQ
            struct.pack_into("=I", body, reqsize, SAVE_DONE2_REQ)
            respsize = 8

        # Send via TCP socket using asyncio
        reader, writer = await asyncio.open_connection(ip, port)
        writer.write(body)
        await writer.drain()

        # Read response
        resp = await reader.readexactly(respsize)
        writer.close()
        await writer.wait_closed()

        logger.info(
            "send_error_done_req: sent error done reqids=%s addr=%s:%d head_kind=%d worker_tp_rank=%d",
            reqids, ip, port, head_kind, worker_tp_rank
        )



class KVTransferServer:
    def __init__(
        self,
        inst_id: str,
        tp_size: int,
        worker_id: int,
        worker_tp_rank: int,
        block_bytes: Union[List[int], int],
        token_bytes: Union[List[int], int],
        naming_url: str,
        layers: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        protocols: List[KVTransferProtocolType],
        num_kv_heads: int = -1,
        num_gdn_layers: int = 3,
        indexer_blk_ntpb: int = 0,
        hybrid_indexer_token_size: int = 0,
        gdn_conv_elem_size: int = 1,
        gdn_ssm_elem_size: int = 1,
        conv_state_shape: Optional[List[int]] = None,
        ssm_state_shape: Optional[List[int]] = None,
        gdn_conv_channel_dims: Optional[List[int]] = None,
        kda_conv_dim_first: bool = False,
        hybrid_attn_token_size: int = 0,
        kda_page_stride: int = 0,
        attn_pack_size: int = 1,
        num_ple_layers: int = 0,
        ple_block_group: int = 0,
        ple_conv_elem_size: int = 1,
        ple_conv_state_shape: Optional[List[int]] = None,
        ple_conv_channel_dims: Optional[List[int]] = None,
    ):
        # older version vllm, should only contain simple model architecture
        if isinstance(block_bytes, int) and isinstance(token_bytes, int):
            block_bytes = [block_bytes]
            token_bytes = [token_bytes]
            layers = [[layer] for layer in layers]
        else:
            if not isinstance(block_bytes, list) or not isinstance(token_bytes, list):
                raise TypeError("block_bytes and token_bytes must both be int or list")

        if any(len(layer) != len(layers[0]) for layer in layers):
            raise ValueError("All layers must have the same number of cache tensors")
        if not len(block_bytes) == len(token_bytes) == len(layers[0]):
            raise ValueError(
                "block_bytes and token_bytes must align with the number of "
                "cache tensors per layer"
            )

        layer_num_blocks_list = [
            _get_layer_num_blocks([
                layer[cache_index] for layer in layers
            ], block_bytes[cache_index]) for cache_index in range(len(block_bytes))
        ]
        if not all(layer_num_blocks == layer_num_blocks_list[0]
                   for layer_num_blocks in layer_num_blocks_list):
            raise ValueError(
                "All cache tensors in one layer must have the same number "
                f"of blocks, got {layer_num_blocks_list}"
            )
        layer_num_blocks = layer_num_blocks_list[0]

        device_id = layers[0][0].get_device()

        logger.info(
            "init kvt server:"
            " instid=%s tpsize=%s wid=%s wrank=%s"
            " blockbytes=%s tokenbytes=%s"
            " naming=%s protocols=%s"
            " layer_shape=%s layer_dtype=%s device_id=%s"
            " num_blocks=%s num_layers=%s",
            inst_id, tp_size, worker_id, worker_tp_rank,
            block_bytes, token_bytes,
            naming_url, protocols,
            [cache.shape for cache in layers[0]],
            [cache.dtype for cache in layers[0]],
            [cache.device for cache in layers[0]],
            layer_num_blocks, len(layers)
        )
        logger.info(
            "init kvt server memory layout: "
            "data_ptrs=%s nbytes=%s strides=%s storage_offsets=%s "
            "contiguous=%s registered_bytes_per_tensor=%s",
            [cache.data_ptr() for cache in layers[0]],
            [cache.nbytes for cache in layers[0]],
            [cache.stride() for cache in layers[0]],
            [cache.storage_offset() for cache in layers[0]],
            [cache.is_contiguous() for cache in layers[0]],
            [layer_num_blocks * size for size in block_bytes],
        )

        layer_addrs = [[l.data_ptr() for l in layer] for layer in layers]
        ops_protocols = [p.to_ops_protocol() for p in protocols]
        init_kv_transfer_server(
            inst_id,
            tp_size,
            worker_id,
            worker_tp_rank,
            device_id,
            block_bytes,
            token_bytes,
            layer_num_blocks,
            naming_url,
            layer_addrs,
            ops_protocols,
            num_kv_heads=num_kv_heads,
            num_gdn_layers=num_gdn_layers,
            indexer_blk_ntpb=indexer_blk_ntpb,
            hybrid_indexer_token_size=hybrid_indexer_token_size,
            gdn_conv_elem_size=gdn_conv_elem_size,
            gdn_ssm_elem_size=gdn_ssm_elem_size,
            conv_state_shape=conv_state_shape if conv_state_shape is not None else [],
            ssm_state_shape=ssm_state_shape if ssm_state_shape is not None else [],
            gdn_conv_channel_dims=gdn_conv_channel_dims if gdn_conv_channel_dims is not None else [],
            kda_conv_dim_first=kda_conv_dim_first,
            hybrid_attn_token_size=hybrid_attn_token_size,
            kda_page_stride=kda_page_stride,
            attn_pack_size=attn_pack_size,
            num_ple_layers=num_ple_layers,
            ple_block_group=ple_block_group,
            ple_conv_elem_size=ple_conv_elem_size,
            ple_conv_state_shape=ple_conv_state_shape if ple_conv_state_shape is not None else [],
            ple_conv_channel_dims=ple_conv_channel_dims if ple_conv_channel_dims is not None else [],
        )
        self._inited = True
