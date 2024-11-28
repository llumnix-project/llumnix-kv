#ifndef KVTRANSFER_INCLUDE_CLIENT_H_
#define KVTRANSFER_INCLUDE_CLIENT_H_

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <queue>
#include "common.h"
#include "context.h"
#include "tx_stub.h"
#include "utils/semaphore.h"
#include "utils/thread_pool.h"
#include "error.h"

namespace blade_llm {

class KvTransferClient : public noncopyable {

 public:
  static std::unique_ptr<KvTransferClient> create(std::unique_ptr<Context> &&,
                                                  const std::vector<TransferProtocol> &,
                                                  std::unique_ptr<ISendStubFactory> &&f);

  KvTransferClient(std::unique_ptr<Context> &&, std::unique_ptr<ISendStubFactory> &&);

  void add_target(const InstanceId &,
                  const WorkerId &,
                  uint32_t start_layer,
                  uint32_t num_layers,
                  std::optional<TransferProtocol> = std::nullopt);
  void remove_target(const InstanceId &, const WorkerId &);

  void submit_req_send(const InstanceId &dst_inst,
                       const WorkerId &dst_worker,
                       const RequestId &,
                       uint32_t new_tokens,
                       bool has_last_token,
                       std::vector<uint32_t> src_block_ids,
                       std::vector<uint32_t> dst_block_ids);
  void submit_delta_send(const RequestId &,
                         uint32_t seen_tokens,
                         uint32_t new_tokens,
                         bool has_last_token);

  size_t start_send();
  void notify_event_record(size_t step_id);
  void flush_send(size_t step_id);

  bool check_transfer_done(const RequestId &);
  Context *context() { return ctx_.get(); };
  void enable_auto_connect() { auto_connect_ = true; }

 private:
  bool auto_connect_{false};
  size_t step_id_{0};
  std::unique_ptr<Context> ctx_;
  std::unordered_map<RequestId, std::vector<std::unique_ptr<RequestInfo>>> reqs_;
  std::unordered_map<InstanceId, std::vector<SendStub>> targets_;
  std::shared_ptr<StepGuard> last_step_guard_;  // may be null.
  std::unique_ptr<ISendStubFactory> stub_factory_;
  ThreadPool single_thd_;
};
}
#endif //KVTRANSFER_INCLUDE_CLIENT_H_
