#!/bin/bash
set -e

apt remove -y accl-barex-cuda 2>/dev/null || true
dpkg -i accl-barex-cuda12.2-devel-1.5.2-2-tcp.deb
