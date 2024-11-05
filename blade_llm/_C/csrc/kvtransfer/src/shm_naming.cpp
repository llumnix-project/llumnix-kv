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

void ShmNamingServer::start() {
  volatile instInfoList *shm = nullptr;
  auto ret = sharedMemoryCreate(path_.data(), sizeof(*shm), &info_);
  if (ret != 0) {
    if (ret == EEXIST) {
      if (sharedMemoryOpen(path_.data(), sizeof(*shm), &info_) != 0) {
        throw std::runtime_error("failed to open shared memory naming service;");
      }
    }
  }
  shm = (volatile instInfoList *) info_.addr;
  memset((void *) shm, 0, sizeof(*shm));
  for (auto i = 0; i < MAX_INSTANCES; i++) {
    for (auto j = 0; j < MAX_WORKS; j++) {
      volatile workerInfo *info = &shm->instances[i].workers[j];
      info->valid = false;
    }
  }
}

void ShmNamingServer::close() {
  sharedMemoryClose(&info_);
}

ShmNamingServer::~ShmNamingServer() {
  close();
}

void ShmNamingClient::connect(const Schema &schema, const std::string &path) {
  if (schema != SHARE_MEMORY_NAMING_SCHEMA) {
    throw std::runtime_error("invalid schema: " + schema);
  }
  auto ret = sharedMemoryOpen(path.data(), sizeof(instInfoList), &info_);
  if (ret != 0 && ret == ENOENT) {
    throw std::runtime_error("share memory not mounted;");
  }
};

bool ShmNamingClient::register_worker(const WorkerInfo &worker_info) {
  volatile auto *shm = (volatile instInfoList *) info_.addr;
  auto worker = &shm->instances[worker_info.inst_id]
      .workers[worker_info.worker_id];
  memcpy((void *) worker->info, &worker_info, sizeof(WorkerInfo));
  worker->valid = true;
  return true;
}

std::optional<WorkerInfo> ShmNamingClient::get_worker_info(uint32_t inst_id, uint32_t worker_id) {
  volatile auto *shm = (volatile instInfoList *) info_.addr;
  volatile workerInfo *info = &shm->instances[inst_id].workers[worker_id];
  if (!info->valid) {
    return std::nullopt;
  }

  WorkerInfo wi;
  memcpy(&wi, (void *) info->info, sizeof(WorkerInfo));
  return wi;
}
}
