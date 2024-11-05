#ifndef KVTRANSFER_INCLUDE_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_H_

#pragma once

#include "common.h"
#include <string>
#include <optional>
#include <memory>
#include <unordered_map>
#include <shared_mutex>

namespace blade_llm {

const static char SCHEMA_DELIMITER = ':';
typedef std::string Schema;

class INamingClient {
 public:
  virtual void connect(const Schema& schema, const std::string& path) = 0;
  virtual bool register_worker(const WorkerInfo &worker_info) = 0;
  virtual std::optional<WorkerInfo> get_worker_info(InstanceId, WorkerId) = 0;
  virtual ~INamingClient() = default;
};

class INamingClientFactory {
 public:
  virtual const Schema & get_schema() = 0;
  virtual std::unique_ptr<INamingClient> create() = 0;
  virtual ~INamingClientFactory() = default;
};

class NamingManager {
 public:
  NamingManager() = default;
  bool register_factory(std::unique_ptr<INamingClientFactory> &&);
  void remove_factory(const Schema &);
  std::unique_ptr<INamingClient> connect_naming(const std::string &url);
 private:
  std::shared_mutex shared_mutex_;
  std::unordered_map<Schema, std::unique_ptr<INamingClientFactory>> factories_;
};

} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_NAMING_H_
