
#pragma once

#ifdef TORCH_FOUND

#include "naming.h"
#include <vector>
#include <string>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace blade_llm {

// torch.TcpStore, url: tcp://$ip:$port
const static std::string TCP_NAMING_SCHEMA = "tcpstore";

class TCPStoreNaming : public INamingClient {
public:
  TCPStoreNaming() = default;
  void connect(const Schema& schema, const std::string &url) override;

  bool register_worker(const WorkerInfo& worker_info) override;

  std::optional<WorkerInfo> get_worker_info(uint32_t inst_id, uint32_t worker_id) override;
private:
  static std::string get_key(uint32_t inst_id, uint32_t worker_id);
  static std::vector<uint8_t> to_value(const WorkerInfo& info);
  static WorkerInfo from_value(const std::vector<uint8_t>& input);
private:
  std::optional<c10d::TCPStore> tcp_store_;
};

class TCPStoreNamingFactory : public INamingClientFactory {
 public:
  const Schema & get_schema() override {
    return TCP_NAMING_SCHEMA;
  }
  std::unique_ptr<INamingClient> create() override {
    return std::make_unique<TCPStoreNaming>();
  }
};

}  // namespace blade_llm {
#endif  // #ifdef TORCH_FOUND
