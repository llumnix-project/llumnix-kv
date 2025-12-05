#!/bin/bash
set -e

apt remove -y accl-barex-cuda
wget https://eflops.oss-cn-beijing.aliyuncs.com/xiaoshi/test/memleak/v3/debug/accl-barex-cuda12.2-devel-1.5.1-3.deb
dpkg -i accl-barex-cuda12.2-devel-1.5.1-3.deb
