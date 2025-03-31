#include <sys/un.h>
#include <stdexcept>
#include <cerrno>
#include <optional>
#include "common.h"
#include "naming/shm_naming.h"
#include "thrid_party/logging.h"

// THIS CODE IS OUT-OF-DATE!
#define MAX_WORKS (8)
#define MAX_INSTANCES (8)
#define MAX_WORKER_INFO_SIZE (16384)

namespace blade_llm {

typedef struct workerInfo_st {
  uint32_t size;
  char info[MAX_WORKER_INFO_SIZE];
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
      info->size = 0;
    }
  }
}

void ShmNamingServer::close() {
  sharedMemoryClose(&info_);
}

ShmNamingServer::~ShmNamingServer() {
  close();
}

void ShmNamingClient::connect(const std::string &path) {
  auto ret = sharedMemoryOpen(path.data(), sizeof(instInfoList), &info_);
  if (ret != 0 && ret == ENOENT) {
    throw std::runtime_error("share memory not mounted;");
  }
};

void ShmNamingClient::register_worker(const WorkerInfo &worker_info) {
  long inst_id;
  try {
    inst_id = std::stol(worker_info.inst_id);
  } catch (const std::exception &e) {
    LOG(ERROR) << "ShmNaming only support number instance name;";
    throw e;
  }

  if (inst_id >= MAX_INSTANCES) {
    throw std::runtime_error("unsupported instance id = " + worker_info.inst_id);
  }
  volatile auto *shm = (volatile instInfoList *) info_.addr;
  auto worker = &shm->instances[inst_id]
      .workers[worker_info.worker_id];
  auto bytes = worker_info.to_bytes();
  if (bytes.size() >= MAX_WORKER_INFO_SIZE) {
    throw std::runtime_error("worker info too large;");
  }
  worker->size = bytes.size();
  memcpy((void *) worker->info, bytes.data(), bytes.size());
}

std::optional<WorkerInfo> ShmNamingClient::get_worker_info(const InstanceId& name, uint32_t worker_id) {
  auto id = std::stoi(name);
  if (id < 0 || id >= MAX_INSTANCES) {
    throw std::runtime_error("invalid instance name: " + name);
  }
  volatile auto *shm = (volatile instInfoList *) info_.addr;
  volatile workerInfo *info = &shm->instances[id].workers[worker_id];
  if (info->size == 0) {
    return std::nullopt;
  }
  return WorkerInfo::from_bytes((unsigned char*)info->info, info->size);
}
}
