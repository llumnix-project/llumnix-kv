#include <cuda_runtime.h>
#include <sys/un.h>
#include "thrid_party/logging.h"
#include "utils/cuda_helper.h"

namespace blade_llm {
#define checkCudaErrors(call) do {                  \
    cudaError_t err = call;                         \
    if (err != cudaSuccess) {                       \
      auto msg = cudaGetErrorString(err);           \
      LOG(ERROR) << "Get CUDA error: " << msg;      \
      throw std::runtime_error(msg);                \
    }                                               \
} while (0)

void cuda_set_device(int device_id) {
  checkCudaErrors(cudaSetDevice(device_id));
}

void cuda_wait_event(uintptr_t addr) {
  cudaEvent_t event = reinterpret_cast<cudaEvent_t>(addr);
  checkCudaErrors(cudaEventSynchronize(event));
}

bool cuda_check_peer_access(int device_id, int peer_device_id) {
  if (peer_device_id != device_id) {
    int can_access;
    checkCudaErrors(
        cudaDeviceCanAccessPeer(&can_access, peer_device_id, device_id));
    if (!can_access) {
      return false;
    }
    checkCudaErrors(cudaSetDevice(peer_device_id));
    checkCudaErrors(cudaDeviceEnablePeerAccess(device_id, 0));
    checkCudaErrors(cudaSetDevice(device_id));
  }
  return true;
}


bool cuda_check_ipc_support(int device_id) {
  LOG(INFO) << "check cuda ipc support ...";
  checkCudaErrors(cudaSetDevice(device_id));
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));

  // CUDA IPC is only supported on devices with unified addressing
  if (!prop.unifiedAddressing) {
    printf("Device %d does not support unified addressing, skipping...\n", device_id);
    return false;
  }
  // This sample requires two processes accessing each device, so we need
  // to ensure exclusive or prohibited mode is not set
  // if (prop.computeMode != cudaComputeModeDefault) {
  //   printf("Device %d is in an unsupported compute mode for this sample\n", device_id);
  //   return false;
  // }

  int dev_count;
  checkCudaErrors(cudaGetDeviceCount(&dev_count));
  for (int i = 0; i < dev_count; i++) {
    if (i != device_id) {
      if (!cuda_check_peer_access(device_id, i)) {
        return false;
      }
    }
  }
  return true;
}

void cuda_create_ipc_handles(const uint64_t *addrs, size_t num_addr, cudaIpcHandles *target) {
  if (num_addr > MAX_ARRAY_LEN) {
    LOG(ERROR) << "too many ipc handles create, max support =" << MAX_ARRAY_LEN;
    throw std::runtime_error("fail to create ipc handles, too many handles;");
  }
  memset(target, 0, sizeof(cudaIpcHandles));
  for (size_t i = 0; i < num_addr; i++) {
    void *ptr = reinterpret_cast<void *>((uintptr_t) addrs[i]);
    auto ret = cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &target->ipc_handles[i], ptr);
    if (ret != cudaSuccess) {
      LOG(ERROR) << "fail to create ipc handle for pointer("
                 << addrs[i]
                 << "), because of: "
                 << cudaGetErrorName(ret)
                 << ", "
                 << cudaGetErrorString(ret);
      throw std::runtime_error("fail to create ipc handle;");
    }
  }
}

void cuda_open_ipc_handle(const cudaIpcHandles *src, size_t num_handles, void **target) {
  if (num_handles > MAX_ARRAY_LEN) {
    LOG(ERROR) << "too many ipc handles to open, max support" << MAX_ARRAY_LEN;
    throw std::runtime_error("fail to open ipc handles, too many handles;");
  }
  for (size_t i = 0; i < num_handles; i++) {
    checkCudaErrors(
        cudaIpcOpenMemHandle(&target[i], src->ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
  }
}

void cuda_close_ipc_handle(void *ptr) {
  checkCudaErrors(cudaIpcCloseMemHandle(ptr));
}

void cuda_d2d_mem_copy(void *dst, const void *src, size_t length) {
  checkCudaErrors(cudaMemcpy(dst, src, length, cudaMemcpyDeviceToDevice));
}

void cuda_d2h_mem_copy(void *dst, const void *src, size_t length) {
  checkCudaErrors(cudaMemcpy(dst, src, length, cudaMemcpyDeviceToHost));
}

void cuda_malloc(void **ptr, size_t size) {
  checkCudaErrors(cudaMalloc(ptr, size));
}

void cuda_free(void *ptr) {
  checkCudaErrors(cudaFree(ptr));
}
}
