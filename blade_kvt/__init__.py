import functools
import logging
import os
import subprocess

logger = logging.getLogger("blade_kvt")


@functools.cache
def is_nv_gpu() -> bool:
    import torch

    return 'nvidia' in torch.cuda.get_device_name().lower()


@functools.cache
def is_erdma() -> bool:
    cmds = ["ibv_devinfo -l", "ibv_devices", "lsmod"]
    for cmd in cmds:
        logger.info("check is_erdma: try cmd=%s", cmd)

        output = b''
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        except Exception:
            logger.warning("check is_erdma: failed")
            continue

        ok = b'erdma' in output
        logger.info("check is_erdma: is_erdma=%s", ok)
        return ok

    logger.warning("check is_erdma: unknown, use False")
    return False


def set_envs():
    os.environ.setdefault('ACCL_TCP_TIMEOUT_MS', '1500')
    os.environ.setdefault('ACCL_MAX_USER_MR_GB', '10')
    os.environ.setdefault('ACCL_WRITEBATCH_OPT', '2')
    os.environ.setdefault('ACCL_TX_DEPTH', '1024')
    # see https://project.aone.alibaba-inc.com/v2/project/664220/req/61674124
    os.environ.setdefault('ACCL_SOFT_TX_DEPTH', '327680')
    if is_erdma():
        os.environ.setdefault('ACCL_SET_ERDMA', '1')
