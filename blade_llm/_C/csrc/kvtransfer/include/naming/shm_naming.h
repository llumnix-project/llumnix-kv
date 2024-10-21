#ifndef KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
#pragma once

#include "naming.h"
#include "utils/shm_helper.h"

namespace blade_llm {

const static std::string SHARE_MEMORY_NAMING_SCHEMA = "shm";

class ShmNaming : public INamingService {
 public:
  ShmNaming() = default;
  explicit ShmNaming(sharedMemoryInfo info);

  bool init(const std::string &url) override;
  bool register_worker(const WorkerInfo &worker_info) override;
  std::optional<WorkerInfo> get_worker_info(uint32_t inst_id, uint32_t worker_id) override;
  ~ShmNaming();
 private:
  bool is_manager_{false};
  sharedMemoryInfo info_;
};

std::shared_ptr<INamingService> create_shm_naming(const std::string &url);
}


#endif //KVTRANSFER_INCLUDE_NAMING_SHM_NAMING_H_
