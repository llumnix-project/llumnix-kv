#ifndef KVTRANSFER_INCLUDE_NAMING_FILESYS_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_FILESYS_NAMING_H_

#pragma once

#include <mutex>
#include <filesystem>
#include <thread>
#include "naming.h"
#include "utils/timer.h"

namespace blade_llm {

const static std::string FILESYS_NAMING_SCHEMA = "file";

class FileSysNaming : public INamingClient, noncopyable {
 public:

  static void create_dir(const std::string &path);

  explicit FileSysNaming(const InstanceId &n) :
      INamingClient(n) {};
  Schema get_schema() override {
    return FILESYS_NAMING_SCHEMA;
  }

  void connect(const Schema &schema, const std::string &path) override;
  void store(const std::string &k, const std::string &v) override;
  std::optional<std::string> get(const InstanceId&, const std::string &k) override;
  std::vector<std::string> search(const InstanceId &id, const std::string &prefix) override;
  void remove(const std::string &key) override;
  const std::vector<std::string> &list() override;
  ~FileSysNaming() override {
    stop_.store(true, std::memory_order_release);
    if (periodic_task_.joinable()) {
      periodic_task_.join();
    }
  }
 private:
  void write_file(const std::string &path, const std::string &content);
  std::filesystem::path naming_path_;
  std::filesystem::path instance_path_;
  std::vector<std::string> list_cache_;
  std::mutex list_mutex_;
  TimeWatch timer_;
  std::thread periodic_task_;
  std::atomic_bool stop_{false};
};

class FileSysNamingClientFactory : public INamingClientFactory, noncopyable {
 public:
  const Schema &get_schema() override {
    return FILESYS_NAMING_SCHEMA;
  }

  std::unique_ptr<INamingClient> create(const InstanceId& inst_name) override {
    return std::make_unique<FileSysNaming>(inst_name);
  };
};

}

#endif //KVTRANSFER_INCLUDE_NAMING_FILESYS_NAMING_H_
