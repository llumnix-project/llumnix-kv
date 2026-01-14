# This file is a wrapper around kv_transfer_impl.py to allow bladellm
# to start in a non-DisaggPD form without building kvtransfer_ops
import enum

import torch

IS_AVAILABLE = True
try:
    import blade_kvt.kvtransfer_ops  # noqa: F401
except ImportError:
    IS_AVAILABLE = False

    @enum.unique
    class KVTransferProtocolType(enum.Enum):
        RDMA_DIRECT = enum.auto()  # make mypy happy~
        TCP = enum.auto()  # make mypy happy~
        DUMMY = enum.auto()

    class ReqMeta:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    class KVTransferClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def record_event(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def submit_req_send(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def submit_req_send2(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def start_req_send(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def submit_delta_send(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def start_send_step(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def flush_send_step(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    class KVTransferServer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    def support_transfers_protocols() -> list[KVTransferProtocolType]:
        raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    class NamingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def list(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def get(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

        def store(self, *args, **kwargs):
            raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    def connect_naming(*args, **kwargs):
        raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    def alloc_phy_cont_mem(size: int, device: torch.device) -> torch.UntypedStorage:
        raise RuntimeError("KVTransferOps is not available, please build BladeLLM with it.")

    def current_worker_info(kind: str = "any") -> str:
        return ""

else:
    from blade_kvt.kv_transfer_impl import (  # noqa: F401
        KVTransferClient,
        KVTransferProtocolType,
        KVTransferServer,
        support_transfers_protocols,
    )
    from blade_kvt.kvtransfer_ops import (  # noqa: F401
        NamingClient,
        ReqMeta,
        alloc_phy_cont_mem,
        connect_naming,
        current_worker_info,
    )


def get_kv_transfer_token_bytes(
    itemsize,
    head_dim=0,
    num_attention_heads=0,
    tp_size=0,
    kv_lora_rank=0,
    qk_rope_head_dim=0,
    num_query_group=None,
    is_MLA=False,
):
    if is_MLA:
        return itemsize * (kv_lora_rank + qk_rope_head_dim)
    _kv_tranfer_unit = num_attention_heads // tp_size if num_query_group is None else max(1, num_query_group // tp_size)
    return 2 * itemsize * head_dim * _kv_tranfer_unit
