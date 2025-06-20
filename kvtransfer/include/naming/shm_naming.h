#ifndef KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
#pragma once

#include "naming.h"
#include "utils/shm_helper.h"

namespace blade_llm {

const static std::string SHARE_MEMORY_NAMING_SCHEMA = "shm";

class ShmNamingServer {
 public:
  explicit ShmNamingServer(const std::string &p) :
      path_(p) {};
  void start();
  void close();
  ~ShmNamingServer();
 private:
  const std::string path_;
  sharedMemoryInfo info_{};
};

class ShmNamingClient : public INamingWorkerClient {
 public:
  ShmNamingClient() = default;
  void connect(const std::string& path);
  void register_worker(const WorkerInfo &worker_info) override;
  std::optional<WorkerInfo> get_worker_info(const InstanceId& inst_id, uint32_t worker_id) override;
 private:
  sharedMemoryInfo info_{};
};
}
#endif //KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
