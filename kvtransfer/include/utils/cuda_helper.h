#ifndef KVTRANSFER_INCLUDE_UTILS_CUDA_HELPER_H_
#define KVTRANSFER_INCLUDE_UTILS_CUDA_HELPER_H_

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define MAX_ARRAY_LEN (128)
namespace blade_llm {
typedef union cudaIpcHandles_un {
  cudaIpcMemHandle_t ipc_handles[MAX_ARRAY_LEN];
  char buf[MAX_ARRAY_LEN * sizeof(cudaIpcMemHandle_t)];
} cudaIpcHandles;

void cuda_set_device(int device_id);
void cuda_wait_event(uintptr_t addr);
bool cuda_check_ipc_support(int device_id);
bool cuda_check_peer_access(int device_id, int peer_device_id);
void cuda_create_ipc_handles(const uint64_t *addrs, size_t num_addr, cudaIpcHandles *target);
void cuda_open_ipc_handle(const cudaIpcHandles *src, size_t num_handles, void **target);
void cuda_close_ipc_handle(void *ptr);
void cuda_d2d_mem_copy(void *dst, const void *src, size_t length);
void cuda_d2h_mem_copy(void *dst, const void *src, size_t length);
void cuda_malloc(void **ptr, size_t size);
void cuda_free(void *ptr);
}

#endif //KVTRANSFER_INCLUDE_UTILS_CUDA_HELPER_H_
