// CUDA runtime API-based copy functions for benchmarking
// These functions use cudaMemcpyAsync/cudaMemcpyBatchAsync and are kept for benchmarking purposes
// NOTE: These functions are NOT in blade_llm namespace - they are only for benchmark testing

#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstring>
#include "channel.h"
#include "thrid_party/logging.h"
#include "assert.h"
#ifdef ENABLE_BATCH_COPY
#include <accl/barex/barex_types.h>
#endif

using namespace blade_llm;

// CUDA runtime API-based copy function using multiple cudaMemcpyAsync calls
// This function is kept only for benchmarking purposes
void copy_handle_data(
  char* tensor_buf_ptr,
  void* layer_gpu_ptr,
  const std::vector<IpcBlock>& blocks,
  size_t tensor_data_size,
  cudaStream_t stream
){
  // Copy tensor data from CPU to GPU according to IpcBlock
  // The tensor data in buffer is continuous, arranged in the same order as blocks
  char* tensor_data_ptr = tensor_buf_ptr;
  size_t tensor_offset = 0;
  
  for (const auto& block : blocks) {
    assert(block.length > 0);
    assert(tensor_offset + block.length <= tensor_data_size);
    void* gpu_dst = reinterpret_cast<char*>(layer_gpu_ptr) + block.dst_offset;
    const void* cpu_src = tensor_data_ptr + tensor_offset;
    
    auto cuda_rt = cudaMemcpyAsync(gpu_dst, cpu_src, block.length, cudaMemcpyHostToDevice, stream);
    if (cuda_rt != cudaSuccess) {
      LOG(ERROR) << "copy_handle_data: cudaMemcpyAsync failed, error=" << cudaGetErrorString(cuda_rt);
      RTCHECK(cuda_rt == cudaSuccess);
    }
    tensor_offset += block.length;
  }
  assert(tensor_offset == tensor_data_size);
  auto cuda_rt_sync = cudaStreamSynchronize(stream);
  RTCHECK(cuda_rt_sync == cudaSuccess);
}

#ifdef ENABLE_BATCH_COPY
// CUDA runtime API-based copy function using cudaMemcpyBatchAsync
// This function is kept only for benchmarking purposes
void copy_handle_data_batch(
  char* tensor_buf_ptr,
  void* layer_gpu_ptr,
  const std::vector<IpcBlock>& blocks,
  size_t tensor_data_size,
  cudaStream_t stream
) {
  const size_t count = blocks.size();
  std::vector<void*> srcs(count);
  std::vector<void*> dsts(count);
  std::vector<size_t> sizes(count);
  char* tensor_data_ptr = tensor_buf_ptr;
  size_t tensor_offset = 0;
  size_t idx = 0;
  
  for (const auto& block : blocks) {
    assert(block.length > 0);
    assert(tensor_offset + block.length <= tensor_data_size);
    void* gpu_dst = reinterpret_cast<char*>(layer_gpu_ptr) + block.dst_offset;
    const void* cpu_src = tensor_data_ptr + tensor_offset;
    srcs[idx] = const_cast<void*>(cpu_src);  // Source: CPU buffer
    dsts[idx] = gpu_dst;                     // Destination: GPU memory
    sizes[idx] = block.length;
    
    tensor_offset += block.length;
    ++idx;
  }
  assert(tensor_offset == tensor_data_size);

  cudaMemcpyAttributes attrs = {};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  std::vector<size_t> attrsIdxs(count, 0);
  size_t failIdx = 0;
  
  auto cuda_rt = cudaMemcpyBatchAsync(
    dsts.data(),      // void** - destination pointers (GPU)
    srcs.data(),      // void** - source pointers (CPU)
    sizes.data(),     // size_t* - sizes array
    count,            // size_t - number of copies
    &attrs,           // cudaMemcpyAttributes* - pointer to attributes array
    attrsIdxs.data(), // size_t* - attributes indices array (all point to attrs[0])
    1,                // size_t - number of attributes
    &failIdx,         // size_t* - failure index output
    stream            // cudaStream_t - stream
  );
  
  if (cuda_rt != cudaSuccess) {
    LOG(ERROR) << "copy_handle_data_batch: cudaMemcpyBatchAsync failed with error: " 
                << cudaGetErrorString(cuda_rt) << " at index: " << failIdx;
    RTCHECK(cuda_rt == cudaSuccess);
  }
  auto cuda_rt_sync = cudaStreamSynchronize(stream);
  RTCHECK(cuda_rt_sync == cudaSuccess);
}
#endif  // ENABLE_BATCH_COPY

