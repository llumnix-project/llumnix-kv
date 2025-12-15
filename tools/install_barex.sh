#!/bin/bash
set -e

apt remove -y accl-barex-cuda
wget 'http://llm-cache-wulan.oss-cn-wulanchabu.aliyuncs.com/xiaoshi-test-memleak-v3-debug-accl-barex-cuda12.2-devel-1.5.1-3.deb?Expires=2081124723&OSSAccessKeyId=LTAI5tR9MQ4DdCf2p8qgUGMD&Signature=Ml3nswB1WjXgouVkkmz52J21h%2FA%3D' -O xiaoshi-test-memleak-v3-debug-accl-barex-cuda12.2-devel-1.5.1-3.deb
dpkg -i xiaoshi-test-memleak-v3-debug-accl-barex-cuda12.2-devel-1.5.1-3.deb
