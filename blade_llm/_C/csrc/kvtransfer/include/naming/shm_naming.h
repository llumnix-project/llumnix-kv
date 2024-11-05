#ifndef KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
#pragma once

#include "naming.h"
#include "utils/shm_helper.h"

namespace blade_llm {

const static std::string SHARE_MEMORY_NAMING_SCHEMA = "shm";

class ShmNamingServer {
 public:
  const std::string url;
  explicit ShmNamingServer(const std::string &p) :
      path_(p),
      url(SHARE_MEMORY_NAMING_SCHEMA + ":" + p) {};
  void start();
  void close();
  ~ShmNamingServer();
 private:
  const std::string path_;
  sharedMemoryInfo info_{};
};

class ShmNamingClient : public INamingClient {
 public:
  ShmNamingClient() = default;
  void connect(const Schema& schema, const std::string& path) override;
  bool register_worker(const WorkerInfo &worker_info) override;
  std::optional<WorkerInfo> get_worker_info(uint32_t inst_id, uint32_t worker_id) override;
 private:
  sharedMemoryInfo info_{};
};

class ShmNamingClientFactory : public INamingClientFactory {
 public:
  const Schema & get_schema() override {
    return SHARE_MEMORY_NAMING_SCHEMA;
  }
  std::unique_ptr<INamingClient> create() override {
    return std::make_unique<ShmNamingClient>();
  };
};
}
#endif //KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
