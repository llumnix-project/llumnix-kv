#!/bin/bash
set -e

apt remove -y accl-barex-cuda
wget https://eflop.oss-cn-beijing.aliyuncs.com/accl-barex/tcp/accl-barex-pkg-test-tcp-v1.1.7/mlx/accl-barex-cuda12.2-devel-1.5.3-1.deb
dpkg -i accl-barex-cuda12.2-devel-1.5.3-1.deb
