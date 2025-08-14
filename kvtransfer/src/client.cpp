#include <stdexcept>
#include "client.h"
#include "naming.h"
#include "thrid_party/logging.h"
#include "utils/timer.h"
#include "protocol/cuda_ipc.h"
#include "protocol/rdma_protocol.h"
#include "envcfg.h"


namespace blade_llm {

KvTransferClient::KvTransferClient(std::unique_ptr<Context> &&ctx,
                                   std::unique_ptr<ISendStubFactory> &&factory) :
    auto_remove_req_(env_send_done_addr() != nullptr),
    ctx_(std::move(ctx)),
    stub_factory_(std::move(factory)),
    single_thd_(4) {}

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
          proto_ctx = RDMAProtoContext::client_context("KVTClient");
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
    auto rdma_ctx = RDMAProtoContext::client_context("KVTClient");
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
  auto &inst = targets_[inst_name];
  if (inst.size() <= worker_id) {
    inst.resize(worker_id + 1);
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

void KvTransferClient::add_send_task(std::shared_ptr<RequestInfo> req, uint32_t seen, uint32_t new_tokens, bool has_last) {
  auto& self = *this;
  req->update_send(seen, new_tokens, has_last);

  BatchSendTask* worker_task_p = nullptr;
  auto& workers_task = self.targets_tasks_buf_[req->dst_inst_id];
  for (auto& worker_task : workers_task) {
    if (worker_task.first == req->dst_worker_id) {
      worker_task_p = &worker_task.second;
      break;
    }
  }
  if (worker_task_p == nullptr) {
    auto& ret = workers_task.emplace_back(req->dst_worker_id, BatchSendTask());
    worker_task_p = &ret.second;
  }
  assert(!worker_task_p->step);
  worker_task_p->tasks.emplace_back(std::move(req), seen, new_tokens, has_last);

  return ;
}

void KvTransferClient::submit_req_send(const InstanceId &dst_inst_name,
                                       const WorkerId &dst_worker_id,
                                       const RequestId &req_id,
                                       uint32_t seen_tokens,
                                       uint32_t new_tokens,
                                       bool has_last_token,
                                       std::vector<uint32_t> src_block_ids,
                                       std::vector<uint32_t> dst_block_ids) {

  if (new_tokens <= 0) {
    LOG(ERROR) << "KVT client: invalid new tokens=" << new_tokens << ";";
    throw KVTransferException(ErrorKind::INVALID_REQUEST_PARAM, "invalid new tokens;");
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
  auto req_info = std::make_shared<RequestInfo>(dst_inst_name,
                                                dst_worker_id,
                                                req_id,
                                                std::move(src_block_ids),
                                                std::move(dst_block_ids));
  add_send_task(req_info, seen_tokens, new_tokens, has_last_token);
  if (!auto_remove_req_ || !has_last_token) {
    reqs_[req_id].emplace_back(std::move(req_info));
  }
  LOG(INFO) << "submit_req_send:step=" << step_id_ << ";req_id=" << req_id
            << ";dst_worker=" << dst_inst_name << ',' << dst_worker_id
            << ";seen_tokens=" << seen_tokens << ";new_tokens=" << new_tokens
            << ";has_last_token=" << has_last_token;
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
  if (new_tokens <= 0) {
    assert(has_last_token);
  }

  for (auto &req : r->second) {
    add_send_task(req, seen_tokens, new_tokens, has_last_token);
    LOG(INFO) << "submit_delta_send:step=" << step_id_ << ";req_id=" << req_id
              << ";dst_worker=" << req->dst_inst_id << ',' << req->dst_worker_id
              << ";seen_tokens=" << seen_tokens << ";new_tokens=" << new_tokens
              << ";has_last_token=" << has_last_token;
  }

  if (has_last_token && auto_remove_req_) {
    reqs_.erase(r);
  }
}

static constexpr size_t EMPTY_STEP_ID = 9223372036854775807;

size_t KvTransferClient::start_send() {
  if (targets_tasks_buf_.empty()) {
    return EMPTY_STEP_ID;
  }

  auto step = std::make_shared<Step>(step_id_++);

  for (auto& [inst_id, workers_task] : targets_tasks_buf_) {
    auto& workers_stub = targets_[inst_id];
    assert(!workers_stub.empty());

    assert(!workers_task.empty());
    for (auto& [worker_id, worker_tasks] : workers_task) {
      assert(!worker_tasks.step);
      worker_tasks.step = step;
      assert(worker_tasks.step.get() == step.get());
      assert(!worker_tasks.tasks.empty());

      assert(workers_stub.size() > worker_id);
      auto& stub = *workers_stub[worker_id];
      RTCHECK(stub.check_state() == StubState::WORKING);
      // TODO(zhanyi.ww): 增加容错处理.
      stub.send_batch(std::move(worker_tasks));
    }
  }
  targets_tasks_buf_.clear();

  auto ctx = context();
  auto step_guard = std::make_shared<StepGuard>(ctx, step);
  single_thd_.spawn([step_guard]() {
#ifndef NDEBUG
    // see https://project.aone.alibaba-inc.com/v2/project/664220/req/60271832
    usleep(10 * 1000);  // sleep 10ms
#endif

    step_guard->step()->wait_layers_start_ts = SteadyClock::now();
    step_guard->wait_layers();
    step_guard->step()->wait_layers_end_ts = SteadyClock::now();

  });
  assert(!last_step_guard_);
  last_step_guard_ = std::move(step_guard);
  return last_step_guard_->step_id();
}

void KvTransferClient::notify_event_record(size_t step_id) {
  assert(targets_tasks_buf_.empty());
  if (!last_step_guard_ || last_step_guard_->step_id() != step_id) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "event record before step start;");
  }
  last_step_guard_->after_record_one();
}

void KvTransferClient::flush_send(size_t step_id) {
  assert(targets_tasks_buf_.empty());
  if (!last_step_guard_ || last_step_guard_->step_id() != step_id) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "event record before step start;");
  }
  last_step_guard_->layer_ready_all();
  last_step_guard_->step()->flush_send_ts = SteadyClock::now();

  last_step_guard_.reset();
}

ReqState KvTransferClient::check_transfer_done(const RequestId &req_id) {
  assert(!auto_remove_req_);
  auto r = reqs_.find(req_id);
  if (r == reqs_.end()) {
    return ReqState::OK;
  }

  ReqState ret = ReqState::OK;
  for (const auto &req : r->second) {
    auto rs = req->state();
    if (rs == ReqState::INPROCESS) {
      return ReqState::INPROCESS;
    }
    if (rs == ReqState::FAILED) {
      ret = ReqState::FAILED;
    }
  }
  reqs_.erase(r);
  return ret;
}
}
