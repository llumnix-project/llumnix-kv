#include <sys/un.h>
#include <stdexcept>
#include <cerrno>
#include <optional>
#include "common.h"
#include "naming/shm_naming.h"

#define MAX_WORKS (8)
#define MAX_INSTANCES (8)

namespace blade_llm {

typedef struct workerInfo_st {
  bool valid;
  char info[sizeof(WorkerInfo)];
} workerInfo;

typedef struct instInfo_st {
  workerInfo workers[MAX_WORKS];
} instInfo;

typedef struct instInfoList_st {
  instInfo instances[MAX_INSTANCES];
} instInfoList;

ShmNaming::ShmNaming(sharedMemoryInfo info) : is_manager_(true), info_(info) {}

bool ShmNaming::init(const std::string &url) {
  auto ret = sharedMemoryOpen(url.data(), sizeof(instInfoList), &info_);
  if (ret != 0 && ret == ENOENT) {
    throw std::runtime_error("share memory not mounted;");
  }
  return true;
};

bool ShmNaming::register_worker(const WorkerInfo &worker_info) {
  volatile auto *shm = (volatile instInfoList *) info_.addr;
  auto worker = &shm->instances[worker_info.inst_id]
      .workers[worker_info.worker_id];
  memcpy((void *) worker->info, &worker_info, sizeof(WorkerInfo));
  worker->valid = true;
  return true;
}

std::optional<WorkerInfo> ShmNaming::get_worker_info(uint32_t inst_id, uint32_t worker_id) {
  volatile auto *shm = (volatile instInfoList *) info_.addr;
  volatile workerInfo *info = &shm->instances[inst_id].workers[worker_id];
  if (!info->valid) {
    return std::nullopt;
  }

  WorkerInfo wi;
  memcpy(&wi, (void *) info->info, sizeof(WorkerInfo));
  return wi;
}

ShmNaming::~ShmNaming() {
  if (is_manager_) {
    sharedMemoryClose(&info_);
  }
}

std::shared_ptr<INamingService> create_shm_naming(const std::string &url) {
  volatile instInfoList *shm = nullptr;
  sharedMemoryInfo info;
  auto ret = sharedMemoryCreate(url.data(), sizeof(*shm), &info);
  if (ret != 0) {
    if (ret == EEXIST) {
      if (sharedMemoryOpen(url.data(), sizeof(*shm), &info) != 0) {
        throw std::runtime_error("failed to open shared memory naming service;");
      }
    }
  }
  shm = (volatile instInfoList *) info.addr;
  memset((void *) shm, 0, sizeof(*shm));
  return std::make_shared<ShmNaming>(info);
}
}