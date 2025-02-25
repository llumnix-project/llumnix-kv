#include "tx_stub.h"
#include "utils/timer.h"
#include "thrid_party/logging.h"
#include "channel.h"
#include "envcfg.h"
#include <unistd.h>
#include <random>

namespace blade_llm {

static void parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks) {
  // as tp size is same, token size should be same;
  assert(src_worker_info->tp_size == dst_worker_info->tp_size);
  assert(src_worker_info->token_size == dst_worker_info->token_size);
  assert(src_worker_info->block_size == dst_worker_info->block_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  auto token_size = src_worker_info->token_size;
  size_t block_size = src_worker_info->block_size;

  // ntpb: number tokens per block
  uint32_t src_ntpb = block_size / token_size;
  uint32_t dst_ntpb = block_size / token_size;
  assert(src_ntpb == dst_ntpb);

  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;
  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  assert(src_blocks.size() == dst_blocks.size());

  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / src_ntpb;
    auto src_token_idx = wrote_tokens % src_ntpb;
    size_t src_offset = src_blocks[src_block_idx] * block_size
        + src_token_idx * token_size;
    auto tokens = std::min(src_ntpb - src_token_idx, left_tokens);
    auto dst_block_idx = wrote_tokens / dst_ntpb;
    auto dst_token_idx = wrote_tokens % dst_ntpb;
    size_t dst_offset = dst_blocks[dst_block_idx] * block_size
        + dst_token_idx * token_size;
    size_t length = tokens * token_size;
    send_blocks.emplace_back(src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

static void do_parse_block_send(
  const WorkerInfo *p_info,  // src
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,  // dst
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &send_blocks) {
  assert(p_info->tp_size > d_info->tp_size);
  assert((p_info->tp_size % d_info->tp_size) == 0);
  const uint32_t group_n = p_info->tp_size / d_info->tp_size;
  assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
  const uint32_t group_off = p_info->worker_tp_rank % group_n;
  assert(d_info->token_size == p_info->token_size * group_n);
  assert(d_info->token_size % 2 == 0);
  assert(p_info->token_size % 2 == 0);
  // cache shape [num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]
  const size_t p_k_size = p_info->token_size / 2;
  const size_t d_k_size = d_info->token_size / 2;
  assert(d_k_size == p_k_size * group_n);
  // p_info->block_size: uint32_t 类型, 需要先转 size_t 不然仍有溢出风险.
  size_t p_block_size = p_info->block_size;
  size_t d_block_size = d_info->block_size;
  size_t p_token_size = p_info->token_size;
  size_t d_token_size = d_info->token_size;
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_info->token_size;
  assert(ntpb * p_info->token_size == p_block_size);
  assert(ntpb * d_info->token_size == d_block_size);
  assert(p_blocks.size() == d_blocks.size());

  while (left_tokens > 0) {
    const uint32_t block_idx = wrote_tokens / ntpb;
    const uint32_t token_idx_base = wrote_tokens % ntpb;
    const size_t p_blk_off = p_blocks[block_idx] * p_block_size;
    const size_t d_blk_off = d_blocks[block_idx] * d_block_size;
    const uint32_t tokens = std::min(ntpb - token_idx_base, left_tokens);

    for (uint32_t idx = 0; idx < tokens; ++idx) {
      const uint32_t token_idx = token_idx_base + idx;
      const size_t pk_token_off = p_blk_off + token_idx * p_token_size;
      const size_t pv_token_off = pk_token_off + p_k_size;
      const size_t d_token_off = d_blk_off + token_idx * d_token_size;
      const size_t dk_token_off = d_token_off + group_off * p_k_size;
      const size_t dv_token_off = dk_token_off + d_k_size;
      send_blocks.emplace_back(pk_token_off, dk_token_off, p_k_size);
      send_blocks.emplace_back(pv_token_off, dv_token_off, p_k_size);
    }

    wrote_tokens += tokens;
    left_tokens -= tokens;
  }

  return;
}

static void parse_block_send_p_gt_d(
  const WorkerInfo *p_info,  // src
  const WorkerInfo *d_info,  // dst
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks) {
  return do_parse_block_send(
    p_info, task->src_blocks(),
    d_info, task->dst_blocks(),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
}

static void parse_block_send_p_lt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks) {
  size_t sb_idx = send_blocks.size();
  do_parse_block_send(
    d_info, task->dst_blocks(),
    p_info, task->src_blocks(),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
  for (; sb_idx < send_blocks.size(); ++sb_idx) {
    std::swap(send_blocks[sb_idx].src_offset, send_blocks[sb_idx].dst_offset);
  }
  return ;
}

using ParseBlockFunc = decltype(parse_block_send_p_lt_d);

static bool same_dst(const WorkerInfo& l, const WorkerInfo& r) noexcept {
  return l.inst_id == r.inst_id &&
         l.worker_id == r.worker_id &&
         l.tp_size == r.tp_size &&
         l.worker_tp_rank == r.worker_tp_rank &&
         l.block_size == r.block_size &&
         l.token_size == r.token_size &&
         l.layer_num_blocks == r.layer_num_blocks &&
         l.num_layers == r.num_layers &&
         l.transfer_protocols == r.transfer_protocols;
}

void KvSendStub::update_dst_info(WorkerInfo&& new_dst) {
  auto& self = *this;
  assert(same_dst(self.dst_info_, new_dst));
  self.dst_info_.addr = std::move(new_dst.addr);
  self.dst_info_.other_info = std::move(new_dst.other_info);
  return ;
}

struct KvSendStub::TaskContext {
  KvSendStub* const stub = nullptr;
  TPKind const tpkind = TPKind::UNKNOWN;
  ParseBlockFunc* const parse_block = nullptr;

  // runtime state
  std::string flush_out_buf;
  std::vector<IpcBlock> send_blocks;
  std::vector<const ReqSendTask *> finished_req;
  Channel ch;
  uint64_t wait_time_us = 0;
  uint64_t wait_and_send_us = 0;
  uint64_t send_notify_us = 0;
  Timepoint iter_start_ts;
  Timepoint send_finish_ts;

private:
#ifdef NDEBUG
  void fault_inject() {}
#else
  static constexpr uint64_t SCALE = 100000;
  std::mt19937_64 rng_;
  std::uniform_int_distribution<uint64_t> dst_{0, 100 * SCALE - 1};

  void fault_inject() {
    auto& self = *this;
    uint64_t rate = (env_debug_tx_failrate() - 1) * SCALE;
    uint64_t rnd = self.dst_(self.rng_);
    if (rnd >= rate) {
      return ;
    }
    throw std::runtime_error("biubiu");
  }
#endif
public:
  TaskContext(KvSendStub* s, TPKind k, ParseBlockFunc* p) noexcept
    : stub(s), tpkind(k), parse_block(p) {
#ifndef NDEBUG
    this->rng_.seed(std::uint_fast64_t(s->dst_info_.worker_id));
#endif
  }

  void do_task(BatchSendTask& batch) {
    auto& self = *this;
    auto const step_idx = batch.step->step_idx;
    const auto& srcinfo = self.stub->src_info_;
    const auto& dstinfo = self.stub->dst_info_;
    const auto dst_id = dstinfo.inst_id;
    const auto dst_worker_id = dstinfo.worker_id;

#ifndef NDEBUG
    // 用于提高 https://aone.alibaba-inc.com/v2/project/664220/req/60815172 复现概率.
    usleep(300 * 1000);  // sleep 300ms
#endif

    self.iter_start_ts = SteadyClock::now();  // iterator begin;
    auto const queue_us = elapse_us(batch.step->start_send_ts, self.iter_start_ts);

    // prepare
    assert(self.flush_out_buf.empty());
    assert(self.finished_req.empty());
    assert(self.send_blocks.empty());
    assert(!batch.tasks.empty());
    for (const auto& task : batch.tasks) {
      assert(task.new_tokens > 0);
      assert(task.dst_inst_id() == dst_id);
      assert(task.dst_worker_id() == dst_worker_id);
      if (task.state() == ReqState::FAILED) {
        continue;
      }
      assert(task.state() == ReqState::INPROCESS);

      self.parse_block(&srcinfo, &dstinfo, &task, self.send_blocks);
      if (task.reach_last_token) {
        self.finished_req.emplace_back(&task);
      }
    }
    if (self.send_blocks.empty()) {
      return;
    }
    assert(!self.send_blocks.empty());

    try {
      self.try_create_channel();
      self.do_send(batch);
    } catch (std::exception& ex) {
      LOG(ERROR) << "KVT tx_stub fail to send data. DstId=" << dst_id
                 << ",DstWorkerId=" << dst_worker_id
                 << ",StepIdx=" << step_idx
                 << ',' << self.flush_out_buf
                 << ",ex=" << ex.what();
      self.ch.reset();
      for (const auto& task : batch.tasks) {
        assert(task.state() == ReqState::INPROCESS || task.state() == ReqState::FAILED);
        task.set_state(ReqState::FAILED);
      }
      return ;
    }
    assert(self.send_finish_ts != Timepoint());

    for (auto task : self.finished_req) {
      assert(task->reach_last_token);
      LOG(INFO) << "KVT tx_stub. DstId=" << dst_id << ",DstWorkerId=" << dst_worker_id
                << ",StepIdx=" << step_idx << ",FinishedReq=" << task->req_id();
      task->set_state(ReqState::OK);
      // DO NOT ACCESS task! IT MAY BE FREED!
      // Add clang tidy: USE-AFTER-MOVED to detect the bug.
    }

    batch.step->update_last_send_finish_ts(self.send_finish_ts);
    LOG(INFO) << "SendStubMetrics. DstId=" << dst_id << ",DstWorkerId=" << dst_worker_id
              << ",StepIdx=" << step_idx << ",FinishedReqSize=" << self.finished_req.size()
              << "," << self.flush_out_buf
              << ",QueueUs=" << queue_us
              << ",WaitUs=" << self.wait_time_us
              << ",WaitAndSendUs=" << self.wait_and_send_us
              << ",SendNotifyUs=" << self.send_notify_us
              << ",SendFinishTs=" << self.send_finish_ts.time_since_epoch().count();  //  send stub id
    return ;
  }

  void clear() noexcept {
    auto& self = *this;
    self.flush_out_buf.clear();
    self.send_blocks.clear();
    self.finished_req.clear();
  }
private:
  void do_send(BatchSendTask& batch) {
    auto& self = *this;
    const auto start_layer = self.stub->start_layer_;
    const auto num_layers = self.stub->num_layers_;

    self.ch->register_data(self.send_blocks, self.tpkind);
    self.wait_time_us = 0;
    for (auto i = start_layer; i < num_layers; ++i) {
      TimeWatch wait_start;
      batch.step->wait_layer_ready(i);
      self.wait_time_us += wait_start.get_elapse_us();

      // NOTE: write 可能是异步的! write 返回并不意味着数据发送了!
      self.ch->send_data(i);
    }
    self.fault_inject();
    self.ch->flush(self.flush_out_buf);
    self.send_finish_ts = SteadyClock::now();
    self.wait_and_send_us = elapse_us(self.iter_start_ts, self.send_finish_ts);

    if (self.finished_req.empty()) {
      self.send_notify_us = 0;
      return ;
    }

    self.ch->send_notification(self.finished_req);
    auto send_notify_end_ts = SteadyClock::now();
    self.send_notify_us = elapse_us(self.send_finish_ts, send_notify_end_ts);
    self.send_finish_ts = send_notify_end_ts;
    return ;
  }

  void refresh_dst_info() {
    auto& self = *this;
    const auto& dst_info = self.stub->dst_info_;
    auto dst_info_opt = self.stub->naming_->get_worker_info(dst_info.inst_id, dst_info.worker_id);
    if (!dst_info_opt) {
      LOG(WARNING) << "TaskContext.dst_info: none: use orig_dst_info=" << dst_info.to_string();
      return;
    }
    if (!same_dst(dst_info, *dst_info_opt)) {
      LOG(WARNING) << "TaskContext.dst_info: invalid dst: orig_dst_info="
                   << dst_info.to_string()
                   << "bad_dst_info=" << dst_info_opt->to_string();
      return ;
    }
    self.stub->update_dst_info(std::move(*dst_info_opt));
    return ;
  }

  void try_create_channel() {
    auto& self = *this;
    if (self.ch) {
      return ;
    }
    self.refresh_dst_info();
    self.ch = self.stub->channel_factory_->create(self.stub->dst_info_);
    LOG(INFO) << "create channel. dst=" << self.stub->dst_info_.to_string()
              << ";ch=" << self.ch.get();
    return ;
  }
};

void KvSendStub::start_async() {
  TPKind tpkind = TPKind::UNKNOWN;
  ParseBlockFunc* parse_block_send = nullptr;
  if (src_info_.tp_size == dst_info_.tp_size) {
    parse_block_send = parse_block_send_p_eq_d;
    tpkind = TPKind::PEQD;
  } else if (src_info_.tp_size > dst_info_.tp_size) {
    parse_block_send = parse_block_send_p_gt_d;
    tpkind = TPKind::PGTD;
  } else {
    parse_block_send = parse_block_send_p_lt_d;
    tpkind = TPKind::PLTD;
  }
  assert(tpkind != TPKind::UNKNOWN);
  auto taskctx = TaskContext(this, tpkind, parse_block_send);

  for (;;) {
    BatchSendTask batch;
    send_tasks_.pop(batch);
    if (batch.step == nullptr) {
      assert(send_tasks_.is_closed());
      break;
    }

    try {
      taskctx.do_task(batch);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "do task failed. DstId=" << dst_info_.inst_id
                  << ",DstWorkerId=" << dst_info_.worker_id
                  << ",StepIdx=" << batch.step->step_idx
                  << ",Ex=" << ex.what();
      state_.store(StubState::POISONED, std::memory_order_release);
      break;
    }

    taskctx.clear();
  }

  auto state = state_.load(std::memory_order_relaxed);
  if (state == StubState::STOPPING) {
    if (!state_.compare_exchange_strong(state, StubState::DISCARD, std::memory_order_seq_cst)) {
      state_.store(StubState::POISONED, std::memory_order_seq_cst);
    }
  }

  LOG(INFO) << "KVT tx_stub: stop to send kv of to worker("
            << dst_info_.inst_id << ":" << dst_info_.worker_id << ");";
}

void KvSendStub::start() {
  auto state = state_.load(std::memory_order_relaxed);
  if (state == StubState::INIT) {
    if (state_.compare_exchange_strong(state, StubState::WORKING, std::memory_order_seq_cst)) {
      send_backend_.emplace(&KvSendStub::start_async, this);
    }
  }
}

void KvSendStub::send_batch(BatchSendTask batch) {
  assert(!batch.tasks.empty());
  send_tasks_.push(std::move(batch));
}

void KvSendStub::stop() {
  auto state = state_.load(std::memory_order_relaxed);
  while (state == StubState::WORKING || state == StubState::INIT) {
    if (state_.compare_exchange_weak(state, StubState::STOPPING, std::memory_order_seq_cst)) {
      send_tasks_.close();
      break;
    }
  }
}

StubState KvSendStub::check_state() {
  return state_.load(std::memory_order_relaxed);
}

KvSendStub::~KvSendStub() {
  auto state = state_.load(std::memory_order_relaxed);
  if (state != StubState::DISCARD && state != StubState::POISONED) {
    LOG(WARNING) << "KVT: SendStub(" << dst_info_.inst_id << ":" << dst_info_.worker_id
                 << ") was not been dropped gracefully.";
    send_tasks_.close();
  }
  if (send_backend_.has_value()) {
    send_backend_->join();
  }
  send_backend_.reset();
}

SendStub KvSendStubFactory::create_stub(const InstanceId& dst_inst_name,
                                        WorkerId dst_worker_id,
                                        uint32_t start_layer,
                                        uint32_t num_layers,
                                        std::optional<TransferProtocol> proto_opt) {

  auto info_opt = naming_worker_->get_worker_info(dst_inst_name, dst_worker_id);
  if (!info_opt.has_value()) {
    LOG(ERROR) << "KVT: can't find worker(" << dst_inst_name << ":" << dst_worker_id << ") from naming service;";
    throw std::runtime_error("can not find worker(" +
        dst_inst_name + ":" + std::to_string(dst_worker_id) +
        ") from naming service;");
  }
  auto info = info_opt.value();
  auto channel_factory = std::make_unique<ChannelFactory>(ctx_, proto_opt);
  return std::make_unique<KvSendStub>(info, ctx_->worker_info(),
                                      start_layer, num_layers,
                                      std::move(channel_factory),
                                      naming_worker_.get());
}
}
