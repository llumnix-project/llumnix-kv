#!/bin/bash
set -e

apt remove -y accl-barex-cuda
wget https://eflop.oss-cn-beijing.aliyuncs.com/accl-barex/accl-barex-pkg-release_v1.5.2-1/mlx/cuda12/cxx11/accl-barex-cuda12-devel-1.5.2-1.deb
dpkg -i accl-barex-cuda12-devel-1.5.2-1.deb
