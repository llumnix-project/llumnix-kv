import threading

import msgspec

# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded EngineCoreOutputs
NEW_OUTPUT_REQ = 20080808
NEW_OUTPUT_RESP = 80808002

# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded EngineCoreRequest
NEW_REQ_REQ = 20080818
NEW_REQ_RESP = 81808002

# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded list[str]
ABORT_REQS_REQ = 20080828
ABORT_REQS_RESP = 82808002

# value: int
OUTPUT_TOKENS_N = "_hbmigrateoutputtokensn"

# value: str, instance id
SRC_INFO = "__HybridConnector_Migration_Src_Info__"

# value: str, trigger policy
MIGRATION_TRIGGER_POLICY = "_hbmigratetriggerpolicy"

# Body:
# +-----+-----------------+
# | len | req             |
# +-----+-----------------+
# len: 4bytes, sizeof(req)
# req: encoded EngineCoreRequest
MIGRATE_TO_REQ = 20080901
MIGRATE_TO_RESP = 10908002

KVT_SUSPEND_REQ = 0x20240912

class MigrateResp(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]
    code: int


# migration related info
_g_migrate_in_req_ids: set[str] = set()
_g_migrate_out_req_ids: set[str] = set()
_g_migrate_in_req_info_lock = threading.Lock()
_g_migrate_out_req_info_lock = threading.Lock()
_g_migrate_in_req_info: dict[str, int] = {}
_g_migrate_out_req_info: dict[str, int] = {}


def migrate_in_pd_way(req=None):
    import vllm.envs as envs
    from ..engine_proxy import core_get_param, get_param
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request
    from ..kvtbackend import RKVTDInfo
    
    if req is None:
        return envs.LLUMNIX_MIGRATE_IN_PD_WAY

    req_condition = True
    if isinstance(req, EngineCoreRequest):
        migration_trigger_policy = core_get_param(req, MIGRATION_TRIGGER_POLICY, "")
        req_condition = "failover" not in migration_trigger_policy
    elif isinstance(req, Request):
        migration_trigger_policy = get_param(req, MIGRATION_TRIGGER_POLICY, "")
        req_condition = "failover" not in migration_trigger_policy
    elif isinstance(req, RKVTDInfo):
        migration_trigger_policy = req.migration_reason
        req_condition = "failover" not in migration_trigger_policy and req.migration

    return req_condition and envs.LLUMNIX_MIGRATE_IN_PD_WAY


def is_migration(idx: int) -> bool:
    return idx > 0xFFFF


def ipport2int(ip: str, port: int) -> int:
    ip_parts = ip.split(".")
    ip_int = (
        (int(ip_parts[0]) << 24)
        | (int(ip_parts[1]) << 16)
        | (int(ip_parts[2]) << 8)
        | int(ip_parts[3])
    )
    return (ip_int << 16) | port


def int2ipport(val: int) -> tuple[str, int]:
    port = val & 0xFFFF
    ip_int = val >> 16
    ip_parts = [
        str((ip_int >> 24) & 0xFF),
        str((ip_int >> 16) & 0xFF),
        str((ip_int >> 8) & 0xFF),
        str(ip_int & 0xFF),
    ]
    ip = ".".join(ip_parts)
    return ip, port
