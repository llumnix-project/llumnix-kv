#ifndef KVTRANSFER_INCLUDE_SERVICE_H_
#define KVTRANSFER_INCLUDE_SERVICE_H_

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>
#include "common.h"
#include "context.h"
#include "rx_stub.h"
#include "server.h"
#include "error.h"

namespace blade_llm {

class KvTransferService : public ITransferService {
 public:
  explicit KvTransferService(std::unique_ptr<Context> &&ctx);
  void submit_recv(const InstanceId& , WorkerId, const RequestId&, const std::vector<uint32_t>& block_ids);
  bool check_recv_done(const RequestId&);
  void on_recv(const InstanceId&, WorkerId, const RequestId&, std::vector<uint32_t> &&) override;
  Context *get_context() { return ctx_.get(); };
 private:
  KvRecvStub &get_or_create_conn(const InstanceId&, WorkerId);
  // may return nullptr
  KvRecvStub* try_get_conn(const InstanceId&, WorkerId);

  std::unique_ptr<Context> ctx_;
  std::unordered_map<RequestId, std::vector<ReqRecvTask>> reqs_;
  std::shared_mutex conn_m_;
  std::unordered_map<InstanceId , std::vector<std::unique_ptr<KvRecvStub>>> recv_conns_;
};
}
#endif //KVTRANSFER_INCLUDE_SERVICE_H_
