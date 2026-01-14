import enum
import os
from typing import List, Optional, Set, Union

import torch
import logging


from blade_kvt.kvtransfer_ops import (
    TransferProtocol,
    add_target,
    check_recv_done,
    check_transfer_done,
    flush_send,
    init_kv_transfer_client,
    init_kv_transfer_server,
    lib_support_transfer_protocols,
    notify_event_record,
    start_send,
    submit_delta_send,
    submit_req_recv,
    submit_req_send2,
    start_req_send,
    ReqMeta,
)


logger = logging.getLogger("blade_kvt")


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
    layer_shape = layers[0].shape
    for l in layers:
        assert l.shape == layer_shape, "All layers should have the same shape"
    return layers[0].nbytes // block_bytes

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
            assert isinstance(block_bytes, list) and isinstance(token_bytes, list)

        for layer in layers:
            assert len(layer) == len(layers[0]), "All layer should have the same number of cache tensors"
        assert (len(block_bytes) == len(token_bytes) == len(layers[0]), 
                "block_bytes and token_bytes should align with layers' cache number")
        
        layer_num_blocks_list = [
            _get_layer_num_blocks([
                layer[cache_index] for layer in layers
            ], block_bytes[cache_index]) for cache_index in range(len(block_bytes))
        ]
        assert (all(layer_num_blocks == layer_num_blocks_list[0] 
                   for layer_num_blocks in layer_num_blocks_list), 
                "Currently all cache in one layer should have the same number of blocks")
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
        )
        self._inited = True
        # None means that start_send is not invoked.
        self._cur_step_id: Optional[int] = None

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
        src_block_ids: List[int],
        dst_block_ids: List[int],
        dst_worker_info: Optional[str] = None,
    ):
        return self.submit_req_send2(
            dst_inst_id, dst_worker_id, req_id, 0, new_tokens, has_last_token, src_block_ids, dst_block_ids, dst_worker_info
        )

    def start_req_send(self, metas: list[ReqMeta]):
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        start_req_send(metas)

    def submit_req_send2(
        self,
        dst_inst_id: str,
        dst_worker_id: int,
        req_id: str,
        seen_tokens: int,
        new_tokens: int,
        has_last_token: bool,
        src_block_ids: List[int],
        dst_block_ids: List[int],
        dst_worker_info: Optional[str] = None,
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
        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        submit_req_send2(
            dst_inst_id, dst_worker_id, req_id, seen_tokens, new_tokens, has_last_token, src_block_ids, dst_block_ids, dst_worker_info
        )

    def submit_delta_send(self, req_id: str, seen_tokens: int, new_tokens: int, has_last_token: bool):
        """
        Submit KV data send of new tokens of a previously submitted request;
        Args:
            req_id(str): the id of previously submitted request;
            seen_tokens: number of tokens has been submitted to send previously;
            new_tokens: number of tokens need to be sent this time;
            has_last_token: indicate whether the last token is included in this send;
        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        submit_delta_send(req_id, seen_tokens, new_tokens, has_last_token)

    def start_send_step(self):
        """
        Start to send kv data of requests submitted previously layer by layer;
        This function just tell the underlying transfer module to start the transfer progress,
        and won't wait the progress to finish, so the function call will return immediately
        without any blocking;

        """
        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")

        if self._cur_step_id is not None:
            raise RuntimeError("start_send_step has been called multiple times")

        # see EMPTY_STEP_ID in client.cpp
        EMPTY_STEP_ID = 9223372036854775807

        # assert self._cur_step_id is None
        step_id = start_send()
        if step_id != EMPTY_STEP_ID:
            self._cur_step_id = step_id
        # self._cur_step_id = start_send()

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

        flush_send(self._cur_step_id)
        self._cur_step_id = None

    def check_req_transfer_done(self, req_id: str) -> bool:
        """
        Check if the request with the specify id has transferred all its kv data to the remote worker;
        A request may want to transfer its kv data to multiple workers, this function only check the
        transfer progress to one worker specified by the params.

        Args:
            req_id(str): id of request;

        Note:
            Different with the 'check_send_step_done' above, this function check transfer progress for
            specific request.
            Returns true if and only if the last token of the request had been submitted to send, and the
            underlying transfer module make sure all data had been written to remote worker's memory(e.g.
            received an acknowledgment from remote);
        """

        if not self._inited:
            raise RuntimeError("KVTransferClient not inited")
        is_done = check_transfer_done(req_id)
        return is_done


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

    ):
        # older version vllm, should only contain simple model architecture
        if isinstance(block_bytes, int) and isinstance(token_bytes, int):
            block_bytes = [block_bytes]
            token_bytes = [token_bytes]
            layers = [[layer] for layer in layers]
        else:
            assert isinstance(block_bytes, list) and isinstance(token_bytes, list)

        for layer in layers:
            assert len(layer) == len(layers[0]), "All layer should have the same number of cache tensors"
        assert (len(block_bytes) == len(token_bytes) == len(layers[0]), 
                "block_bytes and token_bytes should align with layers' cache number")

        layer_num_blocks_list = [
            _get_layer_num_blocks([
                layer[cache_index] for layer in layers
            ], block_bytes[cache_index]) for cache_index in range(len(block_bytes))
        ]
        assert (all(layer_num_blocks == layer_num_blocks_list[0]
                   for layer_num_blocks in layer_num_blocks_list), 
                "Currently all cache in one layer should have the same number of blocks")
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
        )
        self._inited = True
        self._recv_done_reqs: Set[str] = set()

    def submit_req_recv(self, src_inst_id: str, src_worker_id: int, req_id: str, dst_block_ids: List[int]):
        if not self._inited:
            raise RuntimeError("KVTransferServer not inited")

        submit_req_recv(src_inst_id, src_worker_id, req_id, dst_block_ids)

    def check_req_transfer_done(self, req_id: str) -> bool:
        if not self._inited:
            raise RuntimeError("KVTransferServer not inited")

        if req_id not in self._recv_done_reqs:
            is_done = check_recv_done(req_id)
            if is_done:
                self._recv_done_reqs.add(req_id)
            return is_done
        else:
            return True

    def clear_done_reqs(self, req_external_ids: List[str]):
        for _req_external_id in req_external_ids:
            self._recv_done_reqs.remove(_req_external_id)
