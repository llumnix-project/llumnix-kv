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
  const InstanceId inst_id;
  explicit INamingClient(const InstanceId &n) : inst_id(n) {};
  virtual Schema get_schema() = 0;
  virtual void connect(const Schema &schema, const std::string &path) = 0;
  virtual void store(const std::string &k, const std::string &v) = 0;
  virtual void remove(const std::string &key) = 0;
  virtual std::optional<std::string> get(const InstanceId &, const std::string &k) = 0;
  virtual std::vector<std::string> search(const InstanceId &, const std::string &prefix) = 0;
  virtual const std::vector<std::string> &list() = 0;
  static constexpr bool is_binary_store() {
    return false;
  }
  virtual ~INamingClient() = default;
};

class INamingWorkerClient {
 public:
  virtual void register_worker(const WorkerInfo &worker_info) = 0;
  virtual std::optional<WorkerInfo> get_worker_info(const InstanceId &, WorkerId) = 0;
  virtual ~INamingWorkerClient() = default;
};

// used by python bind to expose the naming client to python;
// need to be copyable;
class GeneralNamingClient : public INamingClient {
 public:
  GeneralNamingClient() : INamingClient("unused") {};

  explicit GeneralNamingClient(std::unique_ptr<INamingClient> &&client) :
      INamingClient(client->inst_id),
      client_(std::move(client)) {};

  Schema get_schema() override {
    return client_->get_schema();
  }
  void connect(const Schema &schema, const std::string &path) override {
    throw std::runtime_error("the client is already connected and can't reconnect;");
  };
  void store(const std::string &k, const std::string &v) override {
    client_->store(k, v);
  };
  std::optional<std::string> get(const InstanceId &n, const std::string &k) override {
    return client_->get(n, k);
  };
  std::vector<std::string> search(const InstanceId &n, const std::string &prefix) override {
    return client_->search(n, prefix);
  };
  void remove(const std::string &key) override {
    client_->remove(key);
  };
  const std::vector<std::string> &list() override {
    return client_->list();
  };

  std::unique_ptr<INamingWorkerClient> create_naming_worker_client();

 private:
  std::shared_ptr<INamingClient> client_;
};

class WorkerNamingClient : public INamingWorkerClient {
 public:
  explicit WorkerNamingClient(INamingClient *client) : client_(client) {};

  void register_worker(const WorkerInfo &worker_info) override {
    auto worker_key = "worker_" + std::to_string(worker_info.worker_id);
    if (client_->is_binary_store()) {
      auto bytes = worker_info.to_bytes();
      std::string worker_value(bytes.begin(), bytes.end());
      client_->store(worker_key, worker_value);
    } else {
      client_->store(worker_key, worker_info.to_string());
    }
  };

  std::optional<WorkerInfo> get_worker_info(const InstanceId &name, WorkerId wid) override {
    auto worker_key = "worker_" + std::to_string(wid);
    auto worker_value = client_->get(name, worker_key);
    if (worker_value.has_value()) {
      auto worker_str = worker_value.value();
      if (client_->is_binary_store()) {
        return WorkerInfo::from_bytes((unsigned char *) worker_str.data(), worker_str.size());
      } else {
        return WorkerInfo::from_string(worker_str);
      }
    }
    return std::nullopt;
  };

 private:
  INamingClient *client_;
};

class INamingClientFactory {
 public:
  virtual const Schema &get_schema() = 0;
  virtual std::unique_ptr<INamingClient> create(const InstanceId &name) = 0;
  virtual ~INamingClientFactory() = default;
};

class NamingManager {
 public:
  NamingManager();
  GeneralNamingClient connect_naming(const InstanceId &myname, const std::string &url);
 private:
  std::shared_mutex shared_mutex_;
  std::unordered_map<Schema, std::unique_ptr<INamingClientFactory>> factories_;
  std::unordered_map<std::string, GeneralNamingClient> naming_clients_;
};
} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_NAMING_H_
