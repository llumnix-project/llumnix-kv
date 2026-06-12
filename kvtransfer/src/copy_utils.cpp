
#include "copy_utils.h"
#include "copy_kernels.h"
#include "thrid_party/logging.h"
#include "assert.h"
#include <cuda_runtime.h>

namespace blade_llm {

struct CopyKernelCtx : public noncopyable {
  char* device_blk_buffer = nullptr;
  char* host_blk_buffer = nullptr;
  int device_id = -1;
  bool initialized = false;
  cudaStream_t h2d_stream = nullptr;
  cudaStream_t d2h_stream = nullptr;

  CopyKernelCtx() = default;
  CopyKernelCtx(CopyKernelCtx&&) = delete;
  CopyKernelCtx& operator=(CopyKernelCtx&&) = delete;

  cudaError_t init(int dev_id) {
    if (initialized && this->device_id == dev_id) {
      return cudaSuccess;
    }

    if (device_blk_buffer != nullptr) {
      cudaFree(device_blk_buffer);
      device_blk_buffer = nullptr;
    }
    if (host_blk_buffer != nullptr) {
      cudaFreeHost(host_blk_buffer);
      host_blk_buffer = nullptr;
    }

    auto buffer_num = 3;
    size_t buffer_size = buffer_num * sizeof(int64_t) * env_kernel_copy_max_block_num();

    cudaError_t err = cudaSetDevice(dev_id);
    RTASSERT(err == cudaSuccess);

    err = cudaMalloc(&device_blk_buffer, buffer_size);
    RTASSERT(err == cudaSuccess);
    err = cudaMallocHost(&host_blk_buffer, buffer_size);
    RTASSERT(err == cudaSuccess);

    LOG(INFO) << "CopyKernelCtx: initialized for device " << dev_id
              << ", block num=" << env_kernel_copy_max_block_num()
              << ", buffer_num=" << buffer_num
              << ", buffer_size=" << buffer_size;

    initialize_copy_method_profile(dev_id);

    this->device_id = dev_id;
    initialized = true;
    return cudaSuccess;
  }

  std::pair<char*, char*> get_buffers() const {
    return std::make_pair(device_blk_buffer, host_blk_buffer);
  }

  cudaStream_t get_h2d_stream() {
    if (h2d_stream == nullptr) {
      auto cuda_rt = cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking);
      RTCHECK(cuda_rt == cudaSuccess);
    }
    return h2d_stream;
  }

  void destroy_h2d_stream() {
    if (h2d_stream != nullptr) {
      cudaError_t err = cudaStreamDestroy(h2d_stream);
      if (err != cudaSuccess) {
        LOG(ERROR) << "CopyKernelCtx: Failed to destroy h2d_stream: " << cudaGetErrorString(err);
      }
      h2d_stream = nullptr;
    }
  }

  cudaStream_t get_d2h_stream() {
    if (d2h_stream == nullptr) {
      auto cuda_rt = cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking);
      RTCHECK(cuda_rt == cudaSuccess);
    }
    return d2h_stream;
  }

  void destroy_d2h_stream() {
    if (d2h_stream != nullptr) {
      cudaError_t err = cudaStreamDestroy(d2h_stream);
      if (err != cudaSuccess) {
        LOG(ERROR) << "CopyKernelCtx: Failed to destroy d2h_stream: " << cudaGetErrorString(err);
      }
      d2h_stream = nullptr;
    }
  }

  ~CopyKernelCtx() {
    if (device_blk_buffer != nullptr) {
      cudaFree(device_blk_buffer);
      device_blk_buffer = nullptr;
    }
    if (host_blk_buffer != nullptr) {
      cudaFreeHost(host_blk_buffer);
      host_blk_buffer = nullptr;
    }
    destroy_h2d_stream();
    destroy_d2h_stream();
  }
};

thread_local CopyKernelCtx g_copy_kernel_ctx;

std::pair<char*, char*> get_kernel_copy_buffer(int device_id) {
  cudaError_t err = g_copy_kernel_ctx.init(device_id);
  if (err != cudaSuccess) {
    LOG(ERROR) << "get_kernel_copy_buffer: failed to initialize buffer for device " << device_id
                << ", error=" << cudaGetErrorString(err);
    return std::make_pair(nullptr, nullptr);
  }
  return g_copy_kernel_ctx.get_buffers();
}

cudaStream_t get_thread_local_h2d_stream() {
  return g_copy_kernel_ctx.get_h2d_stream();
}

cudaStream_t get_thread_local_d2h_stream() {
  return g_copy_kernel_ctx.get_d2h_stream();
}

}  // namespace blade_llm
