
#pragma once

#ifdef ENABLE_TORCH
#include "naming.h"
#include <vector>
#include <string>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace blade_llm {

// torch.TcpStore, url: tcp://$ip:$port
const static std::string TCP_NAMING_SCHEMA = "tcpstore";

class TCPStoreNaming : public INamingClient {
public:
  explicit TCPStoreNaming(const InstanceId& n): INamingClient(n) {};
  Schema get_schema() override {
    return TCP_NAMING_SCHEMA;
  }
  void connect(const Schema& schema, const std::string &url) override;
  void store(const std::string &k, const std::string &v) override;
  void remove(const std::string &key) override;
  std::optional<std::string> get(const InstanceId&, const std::string &k) override;
  std::vector<std::string> search(const InstanceId &, const std::string &prefix) override;
  const std::vector<std::string>& list() override;
private:
  std::optional<c10d::TCPStore> tcp_store_;
};

class TCPStoreNamingFactory : public INamingClientFactory {
 public:
  const Schema & get_schema() override {
    return TCP_NAMING_SCHEMA;
  }
 std::unique_ptr<INamingClient> create(const InstanceId& inst_name) override {
    return std::make_unique<TCPStoreNaming>(inst_name);
  }
};

}  // namespace blade_llm
#endif  // #ifdef ENABLE_TORCH

