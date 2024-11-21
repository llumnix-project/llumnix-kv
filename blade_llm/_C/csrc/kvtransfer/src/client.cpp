#include <stdexcept>
#include "client.h"
#include "naming.h"
#include "thrid_party/logging.h"
#include "utils/timer.h"
#include "protocol/cuda_ipc.h"
#include "protocol/rdma_protocol.h"

#define MAX_WORKERS_PER_INST (8)

namespace blade_llm {

KvTransferClient::KvTransferClient(std::unique_ptr<Context> &&ctx,
                                   std::unique_ptr<ISendStubFactory> &&factory) :
    ctx_(std::move(ctx)),
    single_thd_(1),
    stub_factory_(std::move(factory)) {}

std::unique_ptr<KvTransferClient> KvTransferClient::create(std::unique_ptr<Context> &&ctx,
                                                           const std::vector<TransferProtocol> &protocols,
                                                           std::unique_ptr<ISendStubFactory> &&factory) {
  if (ctx == nullptr) {
    throw std::runtime_error("can't create client with context=nullptr;");
  }

  if (factory == nullptr) {
    throw std::runtime_error("can't create client with stub factory=nullptr;");
  }

  if (!protocols.empty()) {
    for (const auto &p : protocols) {
      std::unique_ptr<IProtocolContext> proto_ctx;
      switch (p.type) {
        case TransferProtocol::CUDA_IPC:proto_ctx = std::make_unique<CudaIpcContext>(ctx->device_id());
          break;
        case TransferProtocol::RDMA_DIRECT:
#ifdef ENABLE_RDMA
          proto_ctx = RDMAProtoContext::client_context("KVTClient", 4);
#else
          throw std::runtime_error("RDMA protocol is not enabled in this lib;");
#endif
          break;
        default:throw std::runtime_error("unknown protocol: " + p.to_string());
      }
      ctx->register_protocol(std::move(proto_ctx));
    }
  } else {
    auto cuda_ctx = std::make_unique<CudaIpcContext>(ctx->device_id());
    if (cuda_ctx->check_support()) {
      ctx->register_protocol(std::move(cuda_ctx));
    }
#ifdef ENABLE_RDMA
    auto rdma_ctx = RDMAProtoContext::client_context("KVTClient", 4);
    if (rdma_ctx->check_support()) {
      ctx->register_protocol(std::move(rdma_ctx));
    }
#endif
    auto protos = ctx->support_protocols().as_vector();
    if (protos.empty()) {
      throw std::runtime_error("no protocol supported on this node;");
    }
  }

  return std::make_unique<KvTransferClient>(std::move(ctx), std::move(factory));
}

void KvTransferClient::add_target(const InstanceId &inst_name,
                                  const WorkerId &worker_id,
                                  uint32_t start_layer,
                                  uint32_t num_layers,
                                  std::optional<TransferProtocol> proto_opt) {
  if (worker_id >= MAX_WORKERS_PER_INST) {
    throw KVTransferException(ErrorKind::INVALID_TARGET, "invalid worker id;");
  }
  auto &inst = targets_[inst_name];
  while (inst.size() <= worker_id) {
    inst.emplace_back(nullptr);
  }
  if (inst[worker_id] == nullptr) {
    try {
      inst[worker_id] = stub_factory_->create_stub(inst_name, worker_id, start_layer, num_layers, proto_opt);
    } catch (const std::exception &e) {
      LOG(ERROR) << "KVT client: connect target worker(" << inst_name << ":" << worker_id << ") failed: " << e.what();
      throw KVTransferException(ErrorKind::TARGET_CONNOT_CONNECT, e.what());
    }
    inst[worker_id]->start();
  }
  auto state = inst[worker_id]->check_state();
  if (state != StubState::WORKING) {
    throw KVTransferException(ErrorKind::TARGET_DISCONNECTED, "target worker disconnected");
  }
}

void KvTransferClient::remove_target(const InstanceId &inst_name, const WorkerId &worker_id) {
  if (worker_id >= MAX_WORKERS_PER_INST) {
    return;
  }
  auto &inst = targets_[inst_name];
  while (inst.size() <= worker_id) {
    return;
  }

  if (inst[worker_id] != nullptr) {
    inst[worker_id]->stop();
    auto state = inst[worker_id]->check_state();
    if (state == StubState::DISCARD) {
      inst[worker_id].reset(nullptr);
    }
  }
}

void KvTransferClient::submit_req_send(const InstanceId &dst_inst_name,
                                       const WorkerId &dst_worker_id,
                                       const RequestId &req_id,
                                       uint32_t new_tokens,
                                       bool has_last_token,
                                       std::vector<uint32_t> src_block_ids,
                                       std::vector<uint32_t> dst_block_ids) {

  if (dst_worker_id >= MAX_WORKERS_PER_INST) {
    LOG(ERROR) << "KVT client: invalid worker id: " << dst_worker_id << ";";
    throw KVTransferException(ErrorKind::INVALID_REQUEST_PARAM, "invalid worker id;");
  }

  auto &inst = targets_[dst_inst_name];
  if (inst.size() <= dst_worker_id || inst[dst_worker_id] == nullptr) {
    if (auto_connect_) {
      LOG(WARNING) << "KVT client: target worker(" << dst_inst_name << ":" << dst_worker_id
                   << ") not connected, try to connect use default transfer config;";
      auto num_layers = ctx_->num_layers();
      add_target(dst_inst_name, dst_worker_id, 0, num_layers);
    } else {
      LOG(ERROR) << "KVT client: submit request(" << req_id << ") to unknown target ("
                 << dst_inst_name << ":" << dst_worker_id << "), add it first;";
      throw KVTransferException(ErrorKind::TARGET_NOT_FOUND, "target worker not connect;");
    }
  }
  if (inst[dst_worker_id]->check_state() == StubState::POISONED) {
    LOG(ERROR) << "KVT client: detect stub to worker(" << dst_inst_name << ":" << dst_worker_id
               << ") disconnected because of error, try to reset; ";
    LOG(ERROR) << "KVT client: fail to submit request(" << req_id << ") because target worker disconnected.";
    throw KVTransferException(ErrorKind::TARGET_DISCONNECTED, "target worker disconnected;");
  }
  auto dst_inst_id = inst[dst_worker_id]->dst_info().inst_id;
  auto req_info = std::make_unique<RequestInfo>(dst_inst_id,
                                                dst_worker_id,
                                                req_id,
                                                std::move(src_block_ids),
                                                std::move(dst_block_ids));
  req_info->add_send_task(0, new_tokens, has_last_token);
  reqs_[req_id].emplace_back(std::move(req_info));
  LOG(INFO) << "KVT client step=" << step_id_ << ": accept send request(" << req_id
            << ") to worker(" << dst_inst_id << ":" << dst_worker_id << ") with "
            << new_tokens << " tokens. has_last_token=" << has_last_token;
}
void KvTransferClient::submit_delta_send(const RequestId &req_id,
                                         uint32_t seen_tokens,
                                         uint32_t new_tokens,
                                         bool has_last_token) {
  auto r = reqs_.find(req_id);
  if (r == reqs_.end()) {
    LOG(ERROR) << "KVT client: request(" << req_id << ") not found for delta send;";
    throw KVTransferException(ErrorKind::REQUEST_NOT_FOUND, "request not found;");
  }

  for (auto &req : r->second) {
    req->add_send_task(seen_tokens, new_tokens, has_last_token);
    LOG(INFO) << "KVT client step=" << step_id_ << ": accept delta " << seen_tokens
              << "," << new_tokens << "," << has_last_token
              << " send of request("
              << req_id << ") to worker (" << req->dst_inst_id << ":" << req->dst_worker_id << ");";
  }
}

void KvTransferClient::start_send() {
  LOG(INFO) << "KVT client: start send step: " << step_id_ << " with " << reqs_.size() << " requests;";
  auto step = std::make_shared<Step>(step_id_++);
  auto batch_reqs = std::make_shared<std::vector<ReqSendTask>>();
  for (auto &[req_id, reqs] : reqs_) {
    for (auto &req : reqs) {
      req->pop_tasks(*batch_reqs);
    }
  }
  BatchSendTask batch(step, std::move(batch_reqs));
  for (auto &[inst_id, conns] : targets_) {
    for (auto &tx : conns) {
      if (tx != nullptr) {
        if (tx->check_state() == StubState::WORKING) {
          tx->send_batch(batch);
        }
      }
    }
  }
  auto ctx = context();
  auto step_guard = std::make_shared<StepGuard>(ctx, step);
  single_thd_.spawn([step_guard]() {
    LOG(INFO) << "KVT client: (step " << step_guard->step_id() << "): start to sync data ready by layer...";
    TimeWatch start;
    step_guard->wait_layers();
    LOG(INFO) << "KVT client: (step " << step_guard->step_id()
              << "): notify all layer ready, elapse: " << start.get_elapse_ms() << " ms;";
  });
  step_guards_.push(step_guard);
}

void KvTransferClient::notify_event_record() {
  if (!step_guards_.empty()) {
    step_guards_.back()->after_record_one();
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "event record before step start;");
  }
}

void KvTransferClient::flush_send() {
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
}

bool KvTransferClient::check_transfer_done(const RequestId &req_id) {
  auto r = reqs_.find(req_id);
  if (r == reqs_.end()) {
    return true;
  }
  for (const auto &req : r->second) {
    if (!req->is_all_transferred()) {
      return false;
    }
  }
  reqs_.erase(r);
  return true;
}
}
