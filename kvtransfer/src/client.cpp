#include <stdexcept>
#include "client.h"
#include "naming.h"
#include "thrid_party/logging.h"
#include "utils/timer.h"
#include "protocol/cuda_ipc.h"
#include "protocol/rdma_protocol.h"
#include "envcfg.h"
#include <accl/barex/xthreadpool.h>


namespace blade_llm {

void KvTransferClient::TargetMgr::try_create(
  const InstanceId& inst_id, WorkerId worker_id,
  uint32_t start_layer, uint32_t num_layers,
  const std::optional<std::string>& worker_info) {
  auto& self = *this;
  auto iter = self.peek(inst_id, worker_id);
  if (iter != self.targets_.end()) {
    return;
  }

  self.create(inst_id, worker_id, start_layer, num_layers, worker_info);
  return;
}

auto KvTransferClient::TargetMgr::get(const InstanceId& inst_id, WorkerId worker_id) -> Target* {
  auto& self = *this;
  auto iter = self.peek(inst_id, worker_id);
  if (iter == self.targets_.end()) {
    return nullptr;
  }
  self.targets_.splice(self.targets_.begin(), self.targets_, iter);
  return &*iter;
}

void KvTransferClient::TargetMgr::try_shrink() {
  auto& self = *this;
  if (self.targets_.size() <= (size_t)env_txstub_cap()) {
    return;
  }
  self.shrink();
  return;
}

auto KvTransferClient::TargetMgr::peek(const InstanceId& inst_id, WorkerId worker_id) -> std::list<Target>::iterator {
  auto& self = *this;
  auto iter = self.target_map_.find(inst_id);
  if (iter == self.target_map_.end()) {
    return self.targets_.end();
  }
  if (worker_id >= iter->second.size()) {
    return self.targets_.end();
  }
  const auto& targetiter = iter->second[worker_id];
  if (!targetiter.has_value()) {
    return self.targets_.end();
  }
  return *targetiter;
}


void KvTransferClient::TargetMgr::shrink() {
  auto& self = *this;
  auto ts_start = TimeWatch();
  auto const orig_size = self.targets_.size();
  const size_t cap = env_txstub_cap();
  std::vector<SendStub> gc_stubs;
  gc_stubs.reserve(16);
  while (self.targets_.size() > cap) {
    auto& target = self.targets_.back();
    auto inflycnt = target.inflycnt.load(std::memory_order_relaxed);
    assert(inflycnt >= 0);
    if (inflycnt > 0) {
      break;
    }
    const auto& dstid = target.stub->dstid();
    const auto dwid = target.stub->dstworkerid();
    const auto popret = self.try_pop_map(dstid, dwid);

    LOG(INFO) << "TargetMgr.shrink: instid=" << dstid << " wid=" << dwid
              << " target=" << &target << " popret=" << popret;
    gc_stubs.emplace_back(target.release_stub());
    self.targets_.pop_back();
  }
  self.shrink_thd_.spawn([gc_stubs=std::move(gc_stubs)] () {
    // 在 shrink 线程池析构 SendStub, 避免阻塞主线程.
  });
  LOG(INFO) << "TargetMgr.shrink. OrigSize=" << orig_size
            << "; CurSize=" << self.targets_.size()
            << "; MapSize=" << self.target_map_.size()
            << "; QueueSize=" << self.shrink_thd_.unsafe_size()
            << "; DurUs=" << ts_start.get_elapse_us();
}

int KvTransferClient::TargetMgr::try_pop_map(const InstanceId& dstid, WorkerId dwid) {
  auto& self = *this;
  auto targetiters_iter = self.target_map_.find(dstid);
  if (targetiters_iter == self.target_map_.end()) {
    return 0;
  }
  auto& targetiters = targetiters_iter->second;
  if (dwid >= targetiters.size()) {
    return 0;
  }
  assert(targetiters[dwid].has_value());
  targetiters[dwid] = std::nullopt;

  for (const auto& iteropt : targetiters) {
    if (iteropt.has_value()) {
      return 1;
    }
  }
  self.target_map_.erase(targetiters_iter);
  return 2;
}

void KvTransferClient::TargetMgr::create(
  const InstanceId& inst_id, WorkerId worker_id,
  uint32_t start_layer, uint32_t num_layers,
  const std::optional<std::string>& worker_info) {
  auto& self = *this;
  auto stub = self.stub_factory_->create_stub(inst_id, worker_id,
    start_layer, num_layers,
    std::nullopt, worker_info);
  self.targets_.emplace_front(std::move(stub));
  auto newiter = self.targets_.begin();
  auto& targets = self.target_map_[inst_id];
  if (worker_id >= targets.size()) {
    targets.resize(worker_id + 1);
  }
  assert(!targets.at(worker_id).has_value());
  targets[worker_id] = newiter;
  LOG(INFO) << "TargetMgr.create: instid=" << inst_id << " wid=" << worker_id
            << " worker_info=" << (worker_info ? worker_info->c_str() : "NULL")
            << " target=" << &*newiter;
  return;
}

KvTransferClient::KvTransferClient(std::unique_ptr<Context> &&ctx,
                                   std::unique_ptr<ISendStubFactory> &&factory) :
    auto_remove_req_(env_send_done_addr() != nullptr),
    ctx_(std::move(ctx)),
    single_thd_(env_waitlayer_tpsize()),
    mgr_(std::move(factory)) {
  auto result = accl::barex::XThreadpool::NewInstance(this->target_thdpool_, env_send_tpsize(), "clisend");
  RTASSERT_EQ(result, accl::barex::BAREX_SUCCESS);
  return;
}

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
                                  std::optional<TransferProtocol> proto_opt,
                                  const std::optional<std::string> &worker_info) {
  RTASSERT(false);
}

void KvTransferClient::remove_target(const InstanceId &inst_name, const WorkerId &worker_id) {
  RTASSERT(false);
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
                                       std::vector<uint32_t> dst_block_ids,
                                       const std::optional<std::string> &dst_worker_info) {

  if (new_tokens <= 0) {
    LOG(ERROR) << "KVT client: invalid new tokens=" << new_tokens << ";";
    throw KVTransferException(ErrorKind::INVALID_REQUEST_PARAM, "invalid new tokens;");
  }

  mgr_.try_create(dst_inst_name, dst_worker_id, 0, ctx_->num_layers(), dst_worker_info);
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
    assert(!workers_task.empty());
    for (auto& [worker_id, worker_tasks] : workers_task) {
      assert(!worker_tasks.step);
      worker_tasks.step = step;
      assert(worker_tasks.step.get() == step.get());
      assert(!worker_tasks.tasks.empty());

      auto* target = mgr_.get(inst_id, worker_id);
      assert(target != nullptr);
      auto cnt = target->inflycnt.fetch_add(1, std::memory_order_acquire);
      if (cnt == 0) {
        target->thread_hint = ++thread_hint_;
      }
      target_thdpool_->Submit([target, batch=std::move(worker_tasks)] () mutable {
        // std::function will COPY batch!
        target->stub->send_batch(std::move(batch));
        auto ncnt = target->inflycnt.fetch_sub(1, std::memory_order_release);
        assert(ncnt > 0);
      }, target->thread_hint);
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

  // 最合适的调用地方是某处可以与 gpu overlap 的地方, 后续 kvconnector 应该增加这种接口
  mgr_.try_shrink();
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
