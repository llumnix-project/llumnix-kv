#include "client.h"
#include "naming.h"
#include "logging.h"
#include "utils/timer.h"

#define MAX_WORKERS_PER_INST (8)

namespace blade_llm {
KvTransferClient::KvTransferClient(std::unique_ptr<Context> &&ctx,
                                   std::unique_ptr<ISendStubFactory> &&factory) :
    ctx_(std::move(ctx)),
    single_thd_(1),
    stub_factory_(std::move(factory)) {
  if (stub_factory_ == nullptr) {
    stub_factory_ = std::make_unique<KvSendStubFactory>(ctx_.get());
  }
}

Result<bool> KvTransferClient::connect_stub(InstanceId inst_id, WorkerId worker_id, SendStub &stub) {
  auto w_info = naming()->get_worker_info(inst_id, worker_id);
  if (!w_info.has_value()) {
    LOG(ERROR) << "KVT client: target worker(" << inst_id << ":" << worker_id << ") not found;";
    return Result<bool>::error(ErrorCode::TARGET_NOT_FOUND, "target worker not found in naming;");
  }
  try {
    stub->connect(ctx_.get(), w_info.value());
  } catch (const std::exception &e) {
    LOG(ERROR) << "KVT client: connect target worker(" << inst_id << ":" << worker_id << ") failed: " << e.what();
    return Result<bool>::error(ErrorCode::TARGET_DISCONNECTED, e.what());
  }
  return { true };
}

Result<bool> KvTransferClient::add_target(InstanceId inst_id,
                                          WorkerId worker_id,
                                          uint32_t start_layer,
                                          uint32_t num_layers) {
  if (worker_id >= MAX_WORKERS_PER_INST) {
    return Result<bool>::error(ErrorCode::INVALID_TARGET, "invalid worker id;");
  }
  auto &inst = targets_[inst_id];
  while (inst.size() <= worker_id) {
    inst.emplace_back(nullptr);
  }
  if (inst[worker_id] == nullptr) {
    inst[worker_id] = stub_factory_->create_stub(inst_id, worker_id, start_layer, num_layers);
  }
  if (!inst[worker_id]->is_running()) {
    return connect_stub(inst_id, worker_id, inst[worker_id]);
  }
  LOG(WARNING) << "KVT client: target worker(" << inst_id << ":" << worker_id
               << ") already connected, try stop first;";
  return { true };
}

Result<bool> KvTransferClient::submit_req_send(InstanceId dst_inst_id,
                                               WorkerId dst_worker_id,
                                               const RequestId &req_id,
                                               uint32_t new_tokens,
                                               bool has_last_token,
                                               const std::vector<uint32_t> &src_block_ids,
                                               const std::vector<uint32_t> &dst_block_ids) {

  if (dst_worker_id >= MAX_WORKERS_PER_INST) {
    LOG(ERROR) << "KVT client: invalid worker id: " << dst_worker_id << ";";
    return Result<bool>::error(ErrorCode::INVALID_REQUEST_PARAM, "invalid worker id;");
  }

  auto &inst = targets_[dst_inst_id];
  if (inst.size() <= dst_worker_id || inst[dst_worker_id] == nullptr || !inst[dst_worker_id]->is_running()) {
    if (auto_connect_) {
      LOG(WARNING) << "KVT client: target worker(" << dst_inst_id << ":" << dst_worker_id
                   << ") not connected, try to connect use default transfer config;";
      auto num_layers = ctx_->num_layers();
      auto ret = add_target(dst_inst_id, dst_worker_id, 0, num_layers);
      if (ret.is_err()) {
        return ret;
      }
    } else {
      return Result<bool>::error(ErrorCode::TARGET_DISCONNECTED, "target worker not connect;");
    }
  }
  reqs_[req_id].emplace_back(dst_inst_id, dst_worker_id, req_id, src_block_ids, dst_block_ids)
      .add_new_tokens(new_tokens, has_last_token);
  LOG(INFO) << "KVT client: accept send request(" << req_id
            << ") to worker(" << dst_inst_id << ":" << dst_worker_id << ") with "
            << new_tokens << " tokens;";
  return { true };
}
Result<bool> KvTransferClient::submit_delta_send(const RequestId &req_id,
                                                 uint32_t seen_tokens,
                                                 uint32_t new_tokens,
                                                 bool has_last_token) {
  auto r = reqs_.find(req_id);
  if (r == reqs_.end()) {
    LOG(ERROR) << "KVT client: request(" << req_id << ") not found for delta send;";
    return Result<bool>::error(ErrorCode::REQUEST_NOT_FOUND, "request not found;");
  }

  for (auto &req : r->second) {
    req.set_seen_tokens(seen_tokens)
        .add_new_tokens(new_tokens, has_last_token);
    LOG(INFO) << "KVT client: accept delta " << new_tokens << " send of request("
              << req_id << ") to worker (" << req.dst_inst_id << ":" << req.dst_worker_id << ");";
  }
  return { true };
}

Result<bool> KvTransferClient::start_send() {
  LOG(INFO) << "KVT client: start send step: " << step_id_ << " with " << reqs_.size() << " requests;";
  auto step = std::make_shared<Step>(step_id_++);
  std::vector<const RequestInfo *> batch_reqs;
  for (auto &[req_id, reqs] : reqs_) {
    for (auto &req : reqs) {
      if (req.new_tokens() > 0) {
        batch_reqs.push_back(&req);
      }
    }
  }
  BatchSendTask batch(step, std::move(batch_reqs));
  for (auto &[inst_id, conns] : targets_) {
    for (auto &tx : conns) {
      if (tx != nullptr && tx->is_running()) {
        tx->send_batch(batch);
      }
    }
  }
  auto ctx = context();
  auto step_guard = std::make_shared<StepGuard>(ctx, step);
  single_thd_.spawn([step_guard]() {
    LOG(INFO) << "KVT client: start to sync data ready by layer...";
    TimeWatch start;
    step_guard->wait_layers();
    LOG(INFO) << "KVT client: (step " << step_guard->step_id()
              << "): notify all layer ready, elapse: " << start.get_elapse_ms() << " ms;";
  });
  step_guards_.push(step_guard);
  return { true };
}

Result<bool> KvTransferClient::notify_event_record() {
  if (!step_guards_.empty()) {
    step_guards_.back()->after_record_one();
    return { true };
  }
  return Result<bool>::error(ErrorCode::INVALID_OPERATION, "event record before step start;");
}

Result<bool> KvTransferClient::flush_send() {
  if (!step_guards_.empty()) {
    step_guards_.back()->layer_ready_all();
    while (!step_guards_.empty()) {
      if (step_guards_.front()->step()->check_done()) {
        step_guards_.pop();
      } else {
        break;
      }
    }
  }
  return { true };
}

Result<bool> KvTransferClient::check_transfer_done(const RequestId &req_id) {
  auto r = reqs_.find(req_id);
  if (r == reqs_.end()) {
    return { true };
  }
  for (const auto &req : r->second) {
    if (!req.is_all_transferred()) {
      return { false };
    }
  }
  reqs_.erase(r);
  return { true };
}
}

