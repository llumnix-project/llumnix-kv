#ifndef KVTRANSFER_INCLUDE_CLIENT_H_
#define KVTRANSFER_INCLUDE_CLIENT_H_

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <queue>
#include "protocol.h"
#include "common.h"
#include "context.h"
#include "tx_stub.h"
#include "utils/semaphore.h"
#include "error.h"

namespace blade_llm {

class KvTransferClient : public noncopyable {

 public:
  KvTransferClient(std::unique_ptr<Context> &&ctx,
                   std::unique_ptr<IChannelFactory> &&chf = nullptr);
  Result<bool> add_target(InstanceId, WorkerId, uint32_t start_layer, uint32_t num_layers);
  Result<bool> add_target(InstanceId, WorkerId, SendStub &&stub);
  Result<bool> submit_req_send(InstanceId dst_inst,
                               WorkerId dst_worker,
                               const RequestId &,
                               uint32_t new_tokens,
                               bool has_last_token,
                               const std::vector<uint32_t> &src_block_ids,
                               const std::vector<uint32_t> &dst_block_ids);
  Result<bool> submit_delta_send(const RequestId &,
                                 uint32_t seen_tokens,
                                 uint32_t new_tokens,
                                 bool has_last_token);
  Result<bool> start_send();
  Result<bool> notify_event_record();
  Result<bool> flush_send();
  Result<bool> check_transfer_done(const RequestId&);
  Context *context() { return ctx_.get(); };
  void enable_auto_connect() { auto_connect_ = true; }

 private:
  Result<bool> connect_stub(InstanceId, WorkerId, SendStub &stub);

  bool auto_connect_{false};
  size_t step_id_{0};
  std::unique_ptr<Context> ctx_;
  std::unordered_map<RequestId, std::vector<RequestInfo>> reqs_;
  std::unordered_map<InstanceId, std::vector<SendStub>> targets_;
  std::queue<std::shared_ptr<StepGuard>> step_guards_;
  std::unique_ptr<IChannelFactory> chf_;
};
}
#endif //KVTRANSFER_INCLUDE_CLIENT_H_
