#include <sys/mman.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include "utils/shm_helper.h"

namespace blade_llm {
int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
  int status = 0;
  info->size = sz;
  info->shm_fd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shm_fd < 0) {
    return errno;
  }
  status = ftruncate(info->shm_fd, sz);
  if (status != 0) {
    return status;
  }
  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
  if (info->addr == NULL) {
    return errno;
  }
  return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
  info->size = sz;
  info->shm_fd = shm_open(name, O_RDWR, 0777);
  if (info->shm_fd < 0) {
    return errno;
  }
  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
  if (info->addr == NULL) {
    return errno;
  }
  return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info) {
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shm_fd) {
    close(info->shm_fd);
  }
}
} // namespace blade_llm
