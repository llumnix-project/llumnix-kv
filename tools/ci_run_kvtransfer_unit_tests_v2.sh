#!/bin/bash

# Adapted from tools/ci_run_kvtransfer_unit_tests.sh for AoneCI.

set -ex

date_time=`date "+%Y%m%d_%H%M%S.%3N"`
tmp_local_dir="${PWD}/tmp_local_${date_time}"
mkdir -p ${tmp_local_dir}
container_name=${CONTAINER_NAME:-blade_kvt_ci}

function clean_up {
    docker rm -f ${container_name} || true
    rm -fr ${tmp_local_dir}
}
trap clean_up EXIT

runner_temp_flags=""
if [[ -n "${AONE_CI_SUMMARY_MD}" ]]; then
    runner_temp_flags="-v ${AONE_CI_SUMMARY_MD}:${AONE_CI_SUMMARY_MD} -e AONE_CI_SUMMARY_MD=${AONE_CI_SUMMARY_MD}"
fi

if [ -n "${CUDA_VISIBLE_DEVICES}" ]
then
  GPUS="\"device=${CUDA_VISIBLE_DEVICES}\""
else
  GPUS=all
fi
echo "Setting docker mount GPUS=${GPUS}"

docker run --rm -t --user $(id -u):$(id -g) \
  --name ${container_name} \
  --gpus $GPUS \
  --shm-size=4g \
  -v ${HOME}/.cache:${HOME}/.cache \
  -v ${tmp_local_dir}:${HOME}/.local \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v ${PWD}:/workspace \
  -e GITHUB_WORKFLOW="${GITHUB_WORKFLOW}" \
  -e HTTPS_PROXY="${HTTPS_PROXY}" \
  -e NO_PROXY="${NO_PROXY}" \
  ${runner_temp_flags} \
  -w /workspace \
  pai-blade-registry.cn-beijing.cr.aliyuncs.com/pai-blade/blade-llm-dev:latest-cu124 \
  bash -c "export PATH=\${HOME}/.local/bin:\${PATH} \
    && bash ./tools/install_barex.sh \
    && python3 -m pip install gcovr --user --progress-bar off \
    && cd ./kvtransfer/ && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DBUILD_RDMA=ON .. && cmake --build . \
    && BLLM_KVTRANS_TXSTUB_CAP=0 ctest -V \
    && BLLM_KVTRANS_TXSTUB_CAP=0 BLLM_KVTRANS_CACHE_SHAPE=2 ctest -R '.*SendStubTest.*' -V \
    && BLLM_KVTRANS_TXSTUB_CAP=0 BLLM_KVTRANS_CACHE_SHAPE=3 ctest -R '.*SendStubTest.*' -V \
    && gcovr -r /workspace/kvtransfer --filter ../src --filter ../include --exclude ../include/thrid_party --exclude ../src/third_party \
    && gcovr -r /workspace/kvtransfer --filter ../src --filter ../include --exclude ../include/thrid_party --exclude ../src/third_party --cobertura-pretty --cobertura /workspace/kvtransfer_tests.xml"
