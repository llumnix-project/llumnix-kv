#include <stdexcept>
#include "client.h"
#include "naming.h"
#include "thrid_party/logging.h"
#include "utils/timer.h"
#include "protocol/barex_protocol.h"
#include "envcfg.h"
#include "fault_inject.h"
#include <accl/barex/xthreadpool.h>


namespace blade_llm {

static std::optional<WorkerInfo> parse_worker_info(const std::optional<std::string>& worker_info_str) {
  if (!worker_info_str) {
    return std::nullopt;
  }
  return WorkerInfo::from_string(*worker_info_str);
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

void KvTransferClient::TargetMgr::try_shrink(size_t cap) {
  auto& self = *this;
  if (self.targets_.size() <= cap) {
    return;
  }
  self.shrink(cap);
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

KvTransferClient::TargetMgr::TargetMgr(Context* ctx, std::unique_ptr<ISendStubFactory> s):
  ctx_(ctx),
  stub_factory_(std::move(s)) {
  auto result = accl::barex::XThreadpool::NewInstance(this->target_thdpool_, env_send_tpsize(), "clisend");
  RTASSERT_EQ(result, accl::barex::BAREX_SUCCESS);
}

void KvTransferClient::TargetMgr::submit(std::shared_ptr<Step> step, StepTasks tasks) {
  auto& self = *this;
  self.mgr_thd_.spawn([step=std::move(step), tasks=std::move(tasks), this] () mutable {
    // this: OWNER GLOBAL
    this->do_submit(step, tasks);
  });
  return ;
}

void KvTransferClient::TargetMgr::do_submit(std::shared_ptr<Step>& step, StepTasks& steptasks) {
  auto& self= *this;
  const auto stepid = step->step_idx;
  const auto substepid = steptasks.substepid;

  // Compute and store sub_queue_time_us
  const auto submit_ts = SteadyClock::now();
  const auto queue_time_us = elapse_us(steptasks.send_ts, submit_ts);
  step->sub_queue_time_us.push_back(queue_time_us);

  for (auto& [inst_id, workers_task] : steptasks.tasks) {
    assert(!workers_task.empty());
    for (auto& [worker_id, worker_tasks] : workers_task) {
      assert(!worker_tasks.step);
      worker_tasks.step = step;
      worker_tasks.substepid = substepid;
      worker_tasks.send_ts = submit_ts;
      assert(worker_tasks.step.get() == step.get());
      assert(!worker_tasks.tasks.empty());

      auto* target = self.create_or_get(worker_tasks.tasks.front().req());
      assert(target != nullptr);
      auto cnt = target->inflycnt.fetch_add(1, std::memory_order_acquire);
      if (cnt == 0) {
        target->thread_hint = ++self.thread_hint_;
      }
      self.target_thdpool_->Submit([target, batch=std::move(worker_tasks)] () mutable {
        // std::function will COPY batch!
        fault_inject_sleep(100 * 1000);
        target->stub->send_batch(std::move(batch));
        auto ncnt = target->inflycnt.fetch_sub(1, std::memory_order_release);
        assert(ncnt > 0);
      }, target->thread_hint);
    }
  }

  self.try_shrink(env_txstub_cap());
  return;
}

auto KvTransferClient::TargetMgr::create_or_get(const RequestInfo& req) -> Target* {
  auto& self = *this;
  auto* target = self.get(req.dst_inst_id, req.dst_worker_id);
  if (target != nullptr) {
    return target;
  }
  return self.create(req.dst_inst_id, req.dst_worker_id,
                     0, self.ctx_->num_layers(),
                     req.dst_worker_info);
}

void KvTransferClient::TargetMgr::shrink(size_t cap) {
  auto& self = *this;
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
    self.targets_.pop_back();
  }
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

auto KvTransferClient::TargetMgr::create(
  const InstanceId& inst_id, WorkerId worker_id,
  uint32_t start_layer, uint32_t num_layers,
  const std::optional<WorkerInfo>& worker_info) -> Target* {
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
  auto* target = &*newiter;
  const auto worker_info_str = worker_info ? worker_info->to_string() : "NULL";
  LOG(INFO) << "TargetMgr.create: instid=" << inst_id << " wid=" << worker_id
            << " worker_info=" << worker_info_str
            << " target=" << target;
  return target;
}

KvTransferClient::KvTransferClient(std::unique_ptr<Context> &&ctx,
                                   std::unique_ptr<ISendStubFactory> &&factory) :
    ctx_(std::move(ctx)),
    single_thd_(env_waitlayer_tpsize()),
    mgr_(ctx_.get(), std::move(factory)) {
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
        case TransferProtocol::TCP:{
          LOG(INFO) << "KVT client: creating TCP protocol context...";
          proto_ctx = BarexProtoContext::client_context("KVTClient", TransferProtocol::Kind::TCP);
          break;
        }
        case TransferProtocol::RDMA_DIRECT:
#ifdef ENABLE_RDMA
          proto_ctx = BarexProtoContext::client_context("KVTClient", TransferProtocol::Kind::RDMA_DIRECT);
#else
          throw std::runtime_error("RDMA protocol is not enabled in this lib;");
#endif
          break;
        default:throw std::runtime_error("unknown protocol: " + p.to_string());
      }
      ctx->register_protocol(std::move(proto_ctx));
    }
  } else {
#ifdef ENABLE_RDMA
    auto rdma_ctx = BarexProtoContext::client_context("KVTClient", TransferProtocol::Kind::RDMA_DIRECT);
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

static void add_send_task(StepTasks& steptasks, std::shared_ptr<RequestInfo> req,
  uint32_t seen, uint32_t new_tokens, bool has_last) {
  req->update_send(seen, new_tokens, has_last);

  BatchSendTask* worker_task_p = nullptr;
  auto& workers_task = steptasks.tasks[req->dst_inst_id];
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

static StepTasks create_step_tasks(size_t stepid, uint32_t substepid, std::vector<ReqMeta>& metas, bool freeze) {
  StepTasks steptasks;
  steptasks.stepid = stepid;
  steptasks.substepid = substepid;

  for (auto& reqmeta : metas) {
    LOG(INFO) << "create_step_tasks: StepIdx=" << stepid << "." << substepid
              << ";req_id=" << reqmeta.reqid
              << ";dst_worker=" << reqmeta.dst_inst << ',' << reqmeta.dst_worker
              << ";freeze=" << freeze
              << ";seen_tokens=" << reqmeta.seen_tokens << ";new_tokens=" << reqmeta.new_tokens;
    RTASSERT(reqmeta.new_tokens > 0);

    auto reqinfo = std::make_shared<RequestInfo>(
      std::move(reqmeta.dst_inst),
      reqmeta.dst_worker,
      std::move(reqmeta.dst_worker_info),
      std::move(reqmeta.reqid),
      std::move(reqmeta.src_block_ids),
      std::move(reqmeta.dst_block_ids)
    );
    add_send_task(steptasks, std::move(reqinfo),
                  reqmeta.seen_tokens, reqmeta.new_tokens,
                  /* has_last_token */ true);
  }
  return steptasks;
}

// thread safe
void KvTransferClient::start_req_send(std::vector<ReqMeta>& metas,
                                      size_t stepid,
                                      uint32_t substepid) {
  auto& self = *this;

  auto steptasks = create_step_tasks(stepid, substepid, metas, true);
  if (steptasks.empty()) {
    return ;
  }

  auto step = std::make_shared<Step>(stepid);
  step->notify_layer_ready(self.ctx_->num_layers());
  step->flush_send_ts = step->start_send_ts;
  step->wait_layers_start_ts = step->start_send_ts;
  step->wait_layers_end_ts = step->start_send_ts;

  self.mgr_.submit(std::move(step), std::move(steptasks));
  return;
}

void KvTransferClient::submit_req_send(InstanceId dst_inst_name,
                                       WorkerId dst_worker_id,
                                       RequestId req_id,
                                       uint32_t seen_tokens,
                                       uint32_t new_tokens,
                                       bool has_last_token,
                                       BlockIds src_block_ids,
                                       BlockIds dst_block_ids,
                                       std::optional<std::string> dst_worker_info,
                                       size_t stepid,
                                       uint32_t substepid) {
  assert(substepid == 0);
  RTASSERT(new_tokens > 0);
  auto parsed_dst_worker_info = parse_worker_info(dst_worker_info);

  // Verify stepid/substepid matches targets_tasks_buf_
  if (!targets_tasks_buf_.empty()) {
    RTASSERT_EQ(targets_tasks_buf_.stepid, stepid);
    RTASSERT_EQ(targets_tasks_buf_.substepid, substepid);
  } else {
    targets_tasks_buf_.stepid = stepid;
    targets_tasks_buf_.substepid = substepid;
  }

  auto req_info = std::make_shared<RequestInfo>(std::move(dst_inst_name),
                                                dst_worker_id,
                                                std::move(parsed_dst_worker_info),
                                                std::move(req_id),
                                                std::move(src_block_ids),
                                                std::move(dst_block_ids));
  add_send_task(targets_tasks_buf_, req_info, seen_tokens, new_tokens, has_last_token);
  const auto* reqp = req_info.get();
  if (!has_last_token) {
    reqs_[req_info->req_id].emplace_back(std::move(req_info));
  }
  LOG(INFO) << "submit_req_send: StepIdx=" << stepid << "." << substepid
            << ";req_id=" << reqp->req_id
            << ";dst_worker=" << reqp->dst_inst_id << ',' << dst_worker_id
            << ";seen_tokens=" << seen_tokens << ";new_tokens=" << new_tokens
            << ";has_last_token=" << has_last_token;
}

void KvTransferClient::submit_delta_send(const RequestId &req_id,
                                         uint32_t seen_tokens,
                                         uint32_t new_tokens,
                                         bool has_last_token,
                                         size_t stepid,
                                         uint32_t substepid) {
  assert(substepid == 0);
  // Verify stepid/substepid matches targets_tasks_buf_
  if (!targets_tasks_buf_.empty()) {
    RTASSERT_EQ(targets_tasks_buf_.stepid, stepid);
    RTASSERT_EQ(targets_tasks_buf_.substepid, substepid);
  } else {
    targets_tasks_buf_.stepid = stepid;
    targets_tasks_buf_.substepid = substepid;
  }

  auto r = reqs_.find(req_id);
  if (r == reqs_.end()) {
    LOG(WARNING) << "KVT client: request not found for delta send;"
                 << ";req_id=" << req_id
                 << ";seen_tokens=" << seen_tokens
                 << ";new_tokens=" << new_tokens
                 << ";has_last_token=" << has_last_token;
    return;
  }
  if (new_tokens <= 0) {
    assert(has_last_token);
  }

  for (auto &req : r->second) {
    add_send_task(targets_tasks_buf_, req, seen_tokens, new_tokens, has_last_token);
    LOG(INFO) << "submit_delta_send: StepIdx=" << stepid << "." << substepid
              << ";req_id=" << req_id
              << ";dst_worker=" << req->dst_inst_id << ',' << req->dst_worker_id
              << ";seen_tokens=" << seen_tokens << ";new_tokens=" << new_tokens
              << ";has_last_token=" << has_last_token;
  }

  if (has_last_token) {
    reqs_.erase(r);
  }
}

// thread safe!
void KvTransferClient::start_send_substep(size_t stepid, uint32_t substepid, std::vector<ReqMeta>& metas) {
  auto& self = *this;

  if (metas.empty()) {
    return;
  }
  fault_inject_sleep(333 * 1000);

  enum class Action { kPending, kAttachCurrent, kNewFreeze };
  Action action = Action::kAttachCurrent;
  std::shared_ptr<Step> step;
  {
    std::lock_guard<std::mutex> guard(self.coord_lock_);
    if (stepid < self.coord_step_id_) {
      // Worker has already started the next step, meaning stepid's step is complete
      // Convert metas to has_last_token=true call, create a new Step
      action = Action::kNewFreeze;
    } else if (stepid > self.coord_step_id_) {
      if (!self.pending_step_metas_.empty()) {
        auto& meta = self.pending_step_metas_.back();
        RTASSERT_EQ(meta.stepid, stepid);
        RTASSERT(meta.substepid < substepid);
      }
      self.pending_step_metas_.emplace_back(StepMetas{stepid, substepid, std::move(metas)});
      return;
    } else if (self.last_step_guard_) {
      // stepid == coord_step_id_: directly append to the current step
      step = self.last_step_guard_->step();
      action = Action::kAttachCurrent;
    } else {
      // last_step_guard_ = None
      // stepid = self.coord_step_id_
      action = Action::kNewFreeze;
    }
  }

  if (action == Action::kNewFreeze) {
    // Lock released, call start_req_send to create a new Step and send
    start_req_send(metas, stepid, substepid);
    return;
  }

  auto steptasks = create_step_tasks(stepid, substepid, metas, false);
  assert(!steptasks.empty());
  self.mgr_.submit(std::move(step), std::move(steptasks));
  return ;
}

static constexpr size_t EMPTY_STEP_ID = 9223372036854775807;

size_t KvTransferClient::start_send(size_t stepid, size_t sched_tokens) {
  // sched_tokens == 0: optimization, skip Step/StepGuard creation
  if (sched_tokens == 0 && targets_tasks_buf_.empty()) {
    return EMPTY_STEP_ID;
  }
  fault_inject_sleep(777 * 1000);

  auto step = std::make_shared<Step>(stepid);
  auto ctx = context();
  auto step_guard = std::make_shared<StepGuard>(ctx, step);
  single_thd_.spawn([step_guard]() {
    fault_inject_sleep(10 * 1000);

    step_guard->step()->wait_layers_start_ts = SteadyClock::now();
    step_guard->wait_layers();
    step_guard->step()->wait_layers_end_ts = SteadyClock::now();

  });

  {
    StepTasks tmp_tasks;
    tmp_tasks.swap(targets_tasks_buf_);
    if (!tmp_tasks.empty()) {
      assert(tmp_tasks.substepid == 0);
      mgr_.submit(step, std::move(tmp_tasks));
    }
  }

  // Process coordination state under lock
  auto tmp_reqmetas = std::vector<StepMetas>();
  {
    auto& self = *this;
    std::lock_guard<std::mutex> guard(self.coord_lock_);
    assert(!self.last_step_guard_);
    self.coord_step_id_ = stepid;
    self.last_step_guard_ = std::move(step_guard);
    tmp_reqmetas.swap(self.pending_step_metas_);
  }

  uint32_t substepid = 0;
  for (auto& reqmetas: tmp_reqmetas) {
    assert(reqmetas.stepid == stepid);
    assert(substepid < reqmetas.substepid);
    substepid = reqmetas.substepid;
    auto steptasks = create_step_tasks(reqmetas.stepid, reqmetas.substepid, reqmetas.metas, false);
    assert(!steptasks.empty());
    mgr_.submit(step, std::move(steptasks));
  }

  return stepid;
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

  {
    auto& self = *this;
    std::lock_guard<std::mutex> guard(self.coord_lock_);
    // Assert freeze must be true
    // assert(self.last_step_guard_->freeze);
    // pending_step_metas_ must be empty
    assert(self.pending_step_metas_.empty());

    self.last_step_guard_.reset();
  }
}

}
