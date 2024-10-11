#ifndef KVTRANSFER_INCLUDE_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_H_

#pragma once

#include "common.h"
#include <string>
#include <optional>
#include <memory>

namespace blade_llm {

const static char SCHEMA_DELIMITER = ':';

class INamingService {
 public:
  virtual bool init(const std::string &url) = 0;
  virtual bool register_worker(const WorkerInfo &worker_info) = 0;
  virtual std::optional<WorkerInfo> get_worker_info(InstanceId, WorkerId) = 0;
  virtual ~INamingService() = default;
};

INamingService* naming();
void connect_naming(const std::string &url);

} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_NAMING_H_
