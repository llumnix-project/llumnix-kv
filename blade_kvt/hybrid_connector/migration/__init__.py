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
