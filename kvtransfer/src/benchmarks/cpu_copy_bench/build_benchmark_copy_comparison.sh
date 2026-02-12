#!/bin/bash

# Build script for benchmark_copy_comparison
# This benchmark compares:
# 1. copy_handle_data (CUDA runtime API - multiple cudaMemcpyAsync calls)
# 2. copy_handle_data_batch (CUDA runtime API - cudaMemcpyBatchAsync, if ENABLE_BATCH_COPY is defined)
# 3. copy_handle_data_with_kernel (CUDA kernel-based)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KVTRANSFER_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$KVTRANSFER_DIR/build/benchmark_copy_comparison"

echo "Building benchmark_copy_comparison..."
echo "  KVTRANSFER_DIR: $KVTRANSFER_DIR"
echo "  BUILD_DIR: $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Find CUDA
CUDA_ROOT=${CUDA_HOME:-/usr/local/cuda}
if [ ! -d "$CUDA_ROOT" ]; then
    echo "Error: CUDA not found at $CUDA_ROOT"
    echo "Please set CUDA_HOME environment variable"
    exit 1
fi

# Check for nvcc
NVCC=${NVCC:-$CUDA_ROOT/bin/nvcc}
if [ ! -f "$NVCC" ]; then
    echo "Error: nvcc not found at $NVCC"
    exit 1
fi

# Check if ENABLE_BATCH_COPY should be enabled
ENABLE_BATCH_COPY_FLAG=""
if [ "$ENABLE_BATCH_COPY" = "ON" ] || [ "$ENABLE_BATCH_COPY" = "1" ]; then
    ENABLE_BATCH_COPY_FLAG="-DENABLE_BATCH_COPY"
    echo "ENABLE_BATCH_COPY is enabled"
else
    echo "ENABLE_BATCH_COPY is disabled (use ENABLE_BATCH_COPY=ON to enable)"
fi

# Compile CUDA file first
echo "Compiling copy_kernels.cu..."
$NVCC -std=c++17 \
    -I"$KVTRANSFER_DIR/include" \
    -I"$KVTRANSFER_DIR/src" \
    -O2 -g \
    -arch=sm_80 \
    -c \
    -o copy_kernels.cu.o \
    "$KVTRANSFER_DIR/src/copy_kernels.cu"

# Compile C++ files
echo "Compiling copy_kernels.cpp..."
g++ -std=c++17 \
    -I"$KVTRANSFER_DIR/include" \
    -I"$KVTRANSFER_DIR/src" \
    -I"$CUDA_ROOT/include" \
    -O2 -g \
    -c \
    -o copy_kernels.cpp.o \
    "$KVTRANSFER_DIR/src/copy_kernels.cpp"

echo "Compiling cuda_runtime_api_functions.cpp..."
g++ -std=c++17 \
    -I"$KVTRANSFER_DIR/include" \
    -I"$KVTRANSFER_DIR/src" \
    -I"$CUDA_ROOT/include" \
    $ENABLE_BATCH_COPY_FLAG \
    -O2 -g \
    -c \
    -o cuda_runtime_api_functions.cpp.o \
    "$KVTRANSFER_DIR/src/benchmarks/cpu_copy_bench/cuda_runtime_api_functions.cpp"

echo "Compiling benchmark_copy_comparison.cpp..."
g++ -std=c++17 \
    -I"$KVTRANSFER_DIR/include" \
    -I"$KVTRANSFER_DIR/src" \
    -I"$CUDA_ROOT/include" \
    $ENABLE_BATCH_COPY_FLAG \
    -O2 -g \
    -c \
    -o benchmark_copy_comparison.cpp.o \
    "$KVTRANSFER_DIR/src/benchmarks/cpu_copy_bench/benchmark_copy_comparison.cpp"

# Compile logging library
echo "Compiling logging.cpp..."
g++ -std=c++17 \
    -I"$KVTRANSFER_DIR/include" \
    -I"$KVTRANSFER_DIR/src" \
    -O2 -g \
    -c \
    -o logging.cpp.o \
    "$KVTRANSFER_DIR/src/third_party/logging.cpp"

# Compile envcfg.cpp (needed for env_kernel_copy_max_block_num)
echo "Compiling envcfg.cpp..."
g++ -std=c++17 \
    -I"$KVTRANSFER_DIR/include" \
    -I"$KVTRANSFER_DIR/src" \
    -O2 -g \
    -c \
    -o envcfg.cpp.o \
    "$KVTRANSFER_DIR/src/envcfg.cpp"

# Link
echo "Linking..."
g++ -std=c++17 \
    -o benchmark_copy_comparison \
    benchmark_copy_comparison.cpp.o \
    cuda_runtime_api_functions.cpp.o \
    copy_kernels.cpp.o \
    copy_kernels.cu.o \
    logging.cpp.o \
    envcfg.cpp.o \
    -L"$CUDA_ROOT/lib64" \
    -lcudart \
    -lrt \
    -pthread

echo ""
echo "Build complete: $BUILD_DIR/benchmark_copy_comparison"
echo ""
echo "Usage:"
echo "  $BUILD_DIR/benchmark_copy_comparison [device_id] [num_blocks] [block_size] [total_gpu_size] [iterations] [warmup_iterations]"
echo ""
echo "Default parameters:"
echo "  device_id: 0"
echo "  num_blocks: 1000"
echo "  block_size: 256 bytes"
echo "  total_gpu_size: 10485760 bytes (10MB)"
echo "  iterations: 100"
echo "  warmup_iterations: 10"
echo ""
echo "Example:"
echo "  $BUILD_DIR/benchmark_copy_comparison 0 1000 256 10485760 100 10"
echo ""
echo "To enable batch copy comparison, rebuild with:"
echo "  ENABLE_BATCH_COPY=ON bash $SCRIPT_DIR/build_benchmark_copy_comparison.sh"

