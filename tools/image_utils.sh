#!/bin/bash
set -e

ACR_ENDPOINT=pai-blade-registry.cn-beijing.cr.aliyuncs.com

function pull_blade_image() {
  echo "${ACR_PASSWORD}" | docker login --username "${ACR_USERNAME}" --password-stdin "${ACR_ENDPOINT}"
  docker pull $1
}