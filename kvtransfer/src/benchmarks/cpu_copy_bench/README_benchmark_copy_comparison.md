# benchmark_copy_comparison - Copy 函数性能对比 Benchmark

这个 benchmark 程序用于对比三种不同的 copy 函数实现的性能：

1. **copy_handle_data** (CUDA runtime API) - 使用多个 `cudaMemcpyAsync` 调用
2. **copy_handle_data_batch** (CUDA runtime API) - 使用 `cudaMemcpyBatchAsync` (需要启用 `ENABLE_BATCH_COPY`)
3. **copy_handle_data_with_kernel** (CUDA kernel) - 使用 CUDA kernel 实现

## 编译

### 基本编译（不包含 batch copy 对比）

```bash
cd kvtransfer/src/benchmarks/cpu_copy_bench
bash build_benchmark_copy_comparison.sh
```

### 启用 batch copy 对比

如果要测试 `copy_handle_data_batch`，需要启用 `ENABLE_BATCH_COPY`：

```bash
ENABLE_BATCH_COPY=ON bash build_benchmark_copy_comparison.sh
```

编译完成后，可执行文件位于：
```
kvtransfer/build/benchmark_copy_comparison/benchmark_copy_comparison
```

## 运行

### 基本用法

```bash
./build/benchmark_copy_comparison/benchmark_copy_comparison
```

### 自定义参数

```bash
./build/benchmark_copy_comparison/benchmark_copy_comparison [device_id] [num_blocks] [block_size] [total_gpu_size] [iterations] [warmup_iterations]
```

参数说明：
- `device_id`: CUDA 设备 ID (默认: 0)
- `num_blocks`: block 数量 (默认: 1000)
- `block_size`: 每个 block 的大小（字节）(默认: 256)
- `total_gpu_size`: GPU 内存总大小（字节）(默认: 10485760, 即 10MB)
- `iterations`: 测试迭代次数 (默认: 100)
- `warmup_iterations`: 预热迭代次数 (默认: 10)

### 示例

```bash
# 使用默认参数
./build/benchmark_copy_comparison/benchmark_copy_comparison

# 自定义参数：设备 0，1000 个 blocks，每个 256 字节，GPU 内存 10MB，100 次迭代，10 次预热
./build/benchmark_copy_comparison/benchmark_copy_comparison 0 1000 256 10485760 100 10

# 测试更大的数据量
./build/benchmark_copy_comparison/benchmark_copy_comparison 0 5000 512 52428800 50 5
```

## 输出说明

Benchmark 会输出：

1. **测试参数**：显示所有配置参数
2. **每个函数的测试结果**：
   - 平均执行时间（微秒）
   - 验证结果（✓ PASSED 或 ✗ FAILED）
3. **性能对比摘要**：
   - 各函数的执行时间
   - 相对于 deprecated 版本的加速比
4. **带宽统计**：各函数的带宽（GB/s）

示例输出：
```
========================================
Copy Function Performance Comparison
========================================
Parameters:
  device_id: 0
  num_blocks: 1000
  block_size: 256 bytes
  total_gpu_size: 10485760 bytes
  iterations: 100
  warmup_iterations: 10
========================================

1. Testing copy_handle_data (CUDA runtime API - multiple cudaMemcpyAsync)...
  Average time: 1234.56 us
  Verified: ✓ PASSED

2. Testing copy_handle_data_batch (CUDA runtime API - cudaMemcpyBatchAsync)...
  Average time: 987.65 us
  Verified: ✓ PASSED

3. Testing copy_handle_data_with_kernel (CUDA kernel)...
  Average time: 456.78 us
  Verified: ✓ PASSED

========================================
Performance Comparison Summary
========================================
Function                              Time (us)        Speedup
----------------------------------------
copy_handle_data (CUDA runtime API)       1234.56         1.00x
copy_handle_data_batch (CUDA runtime API)  987.65          1.25x
copy_handle_data_with_kernel (CUDA kernel) 456.78          2.70x
========================================

Bandwidth (GB/s):
  copy_handle_data: 0.20 GB/s
  copy_handle_data_batch: 0.25 GB/s
  copy_handle_data_with_kernel: 0.54 GB/s
```

## 注意事项

1. **ENABLE_BATCH_COPY**: `copy_handle_data_batch` 函数只有在编译时定义了 `ENABLE_BATCH_COPY` 时才会被测试。如果未启用，benchmark 只会测试另外两个函数。

2. **CUDA 架构**: 构建脚本默认使用 `sm_80` 架构。如果你的 GPU 使用不同的架构，需要修改构建脚本中的 `-arch` 参数。

3. **内存要求**: 确保有足够的 GPU 内存来运行测试。`total_gpu_size` 参数不应超过可用 GPU 内存。

4. **验证**: Benchmark 会自动验证复制结果的正确性。如果验证失败，请检查 CUDA 环境配置。

## 相关文件

- `benchmark_copy_comparison.cpp`: Benchmark 主程序
- `cuda_runtime_api_functions.h` / `cuda_runtime_api_functions.cpp`: 基于 CUDA runtime API 的 copy 函数实现
- `copy_kernels.h` / `copy_kernels.cpp` / `copy_kernels.cu`: 基于 CUDA kernel 的 copy 实现
- `build_benchmark_copy_comparison.sh`: 构建脚本

