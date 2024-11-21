#include "tx_stub.h"
#include "utils/timer.h"
#include "thrid_party/logging.h"
#include "channel.h"
#include <unistd.h>

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

// src_offset, dst_offset, len
// return min_size, max_size, total_size, cnt
std::tuple<size_t, size_t, size_t, size_t> merge_interval(std::vector<IpcBlock> &input) {
  size_t min_size = std::numeric_limits<size_t>::max();
  size_t max_size = 0;
  size_t total_size = 0;
  size_t cnt = 0;
  size_t prev_idx = 0;
  std::sort(input.begin(), input.end(),
            [](IpcBlock x, IpcBlock y) { return x.src_offset < y.src_offset; });
  for (size_t idx = 1; idx < input.size(); ++idx) {
    auto &pre_block = input[prev_idx];
    assert(&pre_block.length == &(input[prev_idx].length));
    auto &cur_block = input[idx];
    assert(&cur_block.length == &(input[idx].length));

    if (cur_block.src_offset > pre_block.src_offset + pre_block.length) {
      // 不相邻.
      min_size = std::min(min_size, pre_block.length);
      max_size = std::max(max_size, pre_block.length);
      total_size += pre_block.length;
      cnt += 1;
      prev_idx = idx;
      continue;
    }

    if (cur_block.src_offset == pre_block.src_offset + pre_block.length) {
      // 相邻
      if (cur_block.dst_offset != pre_block.dst_offset + pre_block.length) {
        // 但 dst 不相邻.
        min_size = std::min(min_size, pre_block.length);
        max_size = std::max(max_size, pre_block.length);
        total_size += pre_block.length;
        cnt += 1;
        prev_idx = idx;
        continue;
      }

      pre_block.length += cur_block.length;
      cur_block.length = 0;
      continue;
    }
    // 交错, 在 prefix cache 存在时可能存在这种情况, 目前 prefix cache 尚未开启.
    abort();
  }
  if (prev_idx < input.size()) {
    auto prev_len = input[prev_idx].length;
    min_size = std::min(min_size, prev_len);
    max_size = std::max(max_size, prev_len);
    total_size += prev_len;
    cnt += 1;
  }
  // 暂没必要..
  // std::remove_if(input, [] len == 0)
  return {min_size, max_size, total_size, cnt};
}

void KvSendStub::start_async() {
  auto dst_id = dst_info_.inst_id;
  auto dst_worker_id = dst_info_.worker_id;
  std::vector<IpcBlock> send_blocks;
  std::vector<const ReqSendTask *> send_reqs;
  LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id << ":" << gettid()
            << "): start to send kv of layer[" << start_layer_ << ", " << num_layers_ << ");";

  size_t min_block_n = std::numeric_limits<size_t>::max();
  size_t max_block_n = 0;
  size_t total_block_n = 0;
  size_t finished_req_n = 0;

  decltype(parse_block_send_p_lt_d)* parse_block_send = nullptr;
  if (src_info_.tp_size == dst_info_.tp_size) {
    parse_block_send = parse_block_send_p_eq_d;
  } else if (src_info_.tp_size > dst_info_.tp_size) {
    parse_block_send = parse_block_send_p_gt_d;
  } else {
    parse_block_send = parse_block_send_p_lt_d;
  }

  for (;;) {
    BatchSendTask batch;
    send_tasks_.pop(batch);
    if (batch.step == nullptr) {
      assert(send_tasks_.is_closed());
      break;
    }

#ifndef NDEBUG
    usleep(300 * 1000);  // sleep 300ms
#endif

    uint64_t wait_time_us = 0;
    TimeWatch start;
    send_blocks.clear();
    send_reqs.clear();
    for (const auto& task : *batch.tasks) {
      if (task.new_tokens <= 0 ||
          task.dst_inst_id() != dst_id ||
          task.dst_worker_id() != dst_worker_id) {
        continue;
      }
      send_reqs.push_back(&task);
      parse_block_send(&src_info_, &dst_info_, &task, send_blocks);
    }
    if (send_reqs.empty()) {
      continue;
    }
    auto step_idx = batch.step->step_idx;
    LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
              << "): step(" << step_idx << ") start to send "
              << send_reqs.size() << " requests;";

    auto const sb_num = send_blocks.size();
    auto const [min, max, total, cnt] = merge_interval(send_blocks);

    try {
      for (auto i = start_layer_; i < num_layers_; ++i) {
        TimeWatch wait_start;
        batch.step->wait_layer_ready(i);
        wait_time_us += wait_start.get_elapse_us();

        // NOTE: write 可能是异步的! write 返回并不意味着数据发送了!
        ch_->send_data(i, send_blocks);
      }
      ch_->flush();
      auto elapse = start.get_elapse_us();
      batch.step->finish_one();
      LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
                << "): finish send " << send_reqs.size() << " requests("
                << sb_num << "->" << cnt << " blocks), block_size(min=" << min
                << ",max=" << max << ",total=" << total
                << "), elapse: total use "
                << elapse << "us, wait use " << wait_time_us << "us;";
      auto iter = Iterator<const ReqSendTask *>::copy_from(send_reqs)
          .filter([](auto t) { return t->reach_last_token; });
      if (iter.has_next()) {
        ch_->send_notification(&iter);
      }

      for (auto task : send_reqs) {
        if (!task->reach_last_token) {
          continue;
        }
        auto req_id = task->req_id();
        const auto block_n = task->dst_blocks().size();
        task->set_transfer_done();
        // DO NOT ACCESS task! IT MAY BE FREED!
        // Add clang tidy: USE-AFTER-MOVED to detect the bug.
        min_block_n = std::min(min_block_n, block_n);
        max_block_n = std::max(max_block_n, block_n);
        total_block_n += block_n;
        finished_req_n += 1;
        LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
                  << "): send finish notification of request(" << req_id << ");";
      }
    } catch (std::exception &e) {
      LOG(ERROR) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
                 << "): fail to send data, caused by: " << e.what() << ";";
      state_.store(StubState::POISONED, std::memory_order_release);
      break;
    }

    if (step_idx > 0 && (step_idx & 15) == 0) {
      LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
                << ") metric: req_blocks(min=" << min_block_n << ",max=" << max_block_n << ",total=" << total_block_n
                << "), finished_reqs=" << finished_req_n << ")";
    }
  }
  auto state = state_.load(std::memory_order_relaxed);
  if (state == StubState::STOPPING) {
    if (!state_.compare_exchange_weak(state, StubState::DISCARD, std::memory_order_seq_cst)) {
      state_.store(StubState::POISONED, std::memory_order_seq_cst);
    }
  }

  LOG(INFO) << "KVT tx_stub: stop to send kv of to worker("
            << dst_id << ":" << dst_worker_id << ");";
}

KvSendStub::KvSendStub(KvSendStub &&other) noexcept:
    dst_info_(other.dst_info_),
    src_info_(other.src_info_),
    start_layer_(other.start_layer_),
    num_layers_(other.num_layers_),
    ch_(std::move(other.ch_)) {
}

void KvSendStub::start() {
  auto state = state_.load(std::memory_order_relaxed);
  if (state == StubState::INIT) {
    if (state_.compare_exchange_weak(state, StubState::WORKING, std::memory_order_seq_cst)) {
      send_backend_.emplace(&KvSendStub::start_async, this);
    }
  }
}

void KvSendStub::send_batch(const BatchSendTask &batch) {
  auto num_tasks = batch.tasks->size();
  if (num_tasks > 0) {
    auto task = batch;
    batch.step->start_one();
    send_tasks_.push(std::move(task));
  }
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
  if (info_opt.has_value()) {
    auto info = info_opt.value();
    Channel ch;
    SupportTransferProtocols target_supports(info.transfer_protocols);
    if (proto_opt.has_value()) {
      auto &proto = proto_opt.value();
      if (target_supports.is_support(proto)) {
        ch = create_channel(ctx_, proto);
        ch->connect(info);
      } else {
        LOG(ERROR) << "KVT: target worker(" << dst_inst_name << ":" << dst_worker_id << ") does not support protocol("
                   << proto.to_string() << ");";
        throw std::runtime_error("target worker not support protocol: " + proto.to_string());
      }
    } else {
      auto protos = target_supports.as_vector();
      // TODO : set priority for each protocol;
      for (const auto &p : protos) {
        try {
          ch = create_channel(ctx_, p);
          ch->connect(info);
          LOG(INFO) << "KVT : connect target worker(" << dst_inst_name << ":" << dst_worker_id << ") use "
                    << p.to_string() << " protocol;";
          break;
        } catch (std::exception &e) {
          LOG(WARNING) << "KVT : can't connect target worker(" << dst_inst_name << ":" << dst_worker_id
                       << ") with " << p.to_string() << " protocol, try other protocol ...";
        }
      }
    }
    return std::make_unique<KvSendStub>(info, ctx_->worker_info(), start_layer, num_layers, std::move(ch));
  } else {
    LOG(ERROR) << "KVT: can't find worker(" << dst_inst_name << ":" << dst_worker_id << ") from naming service;";
    throw std::runtime_error("can not find worker(" +
        dst_inst_name + ":" + std::to_string(dst_worker_id) +
        ") from naming service;");
  }
}
}
