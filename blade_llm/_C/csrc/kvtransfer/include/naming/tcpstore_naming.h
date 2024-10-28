
#pragma once

#ifdef ENABLE_RDMA

#include "naming.h"
#include <vector>
#include <string>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace blade_llm {

// torch.TcpStore, url: tcp://$ip:$port
const static std::string TCP_NAMING_SCHEMA = "tcpstore";

class TCPStoreNaming : public INamingService {
public:
  bool init(const std::string &url) override;

  bool register_worker(const WorkerInfo& worker_info) override;

  std::optional<WorkerInfo> get_worker_info(uint32_t inst_id, uint32_t worker_id) override;
private:
  static std::string get_key(uint32_t inst_id, uint32_t worker_id);
  static std::vector<uint8_t> to_value(const WorkerInfo& info);
  static WorkerInfo from_value(const std::vector<uint8_t>& input);
private:
  std::optional<c10d::TCPStore> tcp_store_;
};

}  // namespace blade_llm {

#endif  // #ifdef ENABLE_RDMA