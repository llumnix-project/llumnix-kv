// Header file for CUDA runtime API-based copy functions
// These functions use cudaMemcpyAsync/cudaMemcpyBatchAsync and are kept for benchmarking purposes
// NOTE: These functions are NOT in blade_llm namespace - they are only for benchmark testing

#ifndef KVTRANSFER_BENCHMARKS_CUDA_RUNTIME_API_FUNCTIONS_H_
#define KVTRANSFER_BENCHMARKS_CUDA_RUNTIME_API_FUNCTIONS_H_

#include <cuda_runtime.h>
#include <vector>
#include "channel.h"

// Forward declare IpcBlock from blade_llm namespace
namespace blade_llm {
  struct IpcBlock;
}

// CUDA runtime API-based copy function using multiple cudaMemcpyAsync calls
// This function is kept only for benchmarking purposes
void copy_handle_data(
  char* tensor_buf_ptr,
  void* layer_gpu_ptr,
  const std::vector<blade_llm::IpcBlock>& blocks,
  size_t tensor_data_size,
  cudaStream_t stream
);

#ifdef ENABLE_BATCH_COPY
// CUDA runtime API-based copy function using cudaMemcpyBatchAsync
// This function is kept only for benchmarking purposes
void copy_handle_data_batch(
  char* tensor_buf_ptr,
  void* layer_gpu_ptr,
  const std::vector<blade_llm::IpcBlock>& blocks,
  size_t tensor_data_size,
  cudaStream_t stream
);
#endif  // ENABLE_BATCH_COPY

#endif  // KVTRANSFER_BENCHMARKS_CUDA_RUNTIME_API_FUNCTIONS_H_

