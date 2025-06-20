#ifndef KVTRANSFER_INCLUDE_UTILS_SHM_HELPER_H_
#define KVTRANSFER_INCLUDE_UTILS_SHM_HELPER_H_

#pragma once
#include <cstddef>

namespace blade_llm {

typedef struct sharedMemoryInfo_st {
  void *addr;
  size_t size;
  int shm_fd;
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);
int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);
void sharedMemoryClose(sharedMemoryInfo *info);

} // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_UTILS_SHM_HELPER_H_
