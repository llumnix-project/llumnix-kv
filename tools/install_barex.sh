#!/bin/bash
set -e

apt remove -y accl-barex-cuda
wget https://eflop.oss-cn-beijing.aliyuncs.com/accl-barex/tcp/accl-barex-pkg-test-tcp-v1.1.3/accl-barex-cuda12.2-devel-1.5.2-2-tcp.deb
dpkg -i accl-barex-cuda12.2-devel-1.5.2-2-tcp.deb
