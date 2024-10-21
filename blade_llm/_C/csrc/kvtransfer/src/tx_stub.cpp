#include "tx_stub.h"
#include "utils/timer.h"
#include "logging.h"
#include "channel.h"

namespace blade_llm {
static inline void parse_block_send(const WorkerInfo *src_worker_info,
                                    const WorkerInfo *dst_worker_info,
                                    const RequestInfo *task,
                                    std::vector<IpcBlock> &send_blocks) {
  if (src_worker_info->tp_size == dst_worker_info->tp_size) {
    // as tp size is same, token size should be same;
    assert(src_worker_info->token_size == dst_worker_info->token_size);
    auto token_size = src_worker_info->token_size;

    // ntpb: number tokens per block
    uint32_t src_ntpb = src_worker_info->block_size / token_size;
    uint32_t dst_ntpb = dst_worker_info->block_size / token_size;

    auto wrote_tokens = task->seen_tokens();
    auto left_tokens = task->new_tokens();
    const auto &src_blocks = task->src_blocks();
    const auto &dst_blocks = task->dst_blocks();

    while (left_tokens > 0) {
      auto src_block_idx = wrote_tokens / src_ntpb;
      auto src_token_idx = wrote_tokens % src_ntpb;
      size_t src_offset = src_blocks[src_block_idx] * src_worker_info->block_size
          + src_token_idx * token_size;
      auto tokens = std::min(src_ntpb - src_token_idx, left_tokens);
      auto dst_block_idx = wrote_tokens / dst_ntpb;
      auto dst_token_idx = wrote_tokens % dst_ntpb;
      size_t dst_offset = dst_blocks[dst_block_idx] * dst_worker_info->block_size
          + dst_token_idx * token_size;
      size_t length = tokens * token_size;
      send_blocks.emplace_back(src_offset, dst_offset, length);
      wrote_tokens += tokens;
      left_tokens -= tokens;
    }
  } else {
    throw std::runtime_error("different tp size is not support now;");
  }
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

static void pull_send(size_t start_layer,
                      size_t num_layers,
                      const WorkerInfo *src_info,
                      const WorkerInfo *dst_info,
                      IChannel *channel,
                      std::atomic_bool *running,
                      BlockingQueue<BatchSendTask> *send_tasks) {
  auto dst_id = dst_info->inst_id;
  auto dst_worker_id = dst_info->worker_id;
  std::vector<IpcBlock> send_blocks;
  std::vector<const RequestInfo *> send_reqs;
  LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
            << "): start to send kv of layer[" << start_layer << ", " << num_layers << ");";

  size_t min_block_n = std::numeric_limits<size_t>::max();
  size_t max_block_n = 0;
  size_t total_block_n = 0;
  size_t finished_req_n = 0;
  size_t min_size = std::numeric_limits<size_t>::max();
  size_t max_size = 0;
  size_t total_size = 0;

  for (;;) {
    BatchSendTask batch;
    send_tasks->pop(batch);
    if (batch.step == nullptr) {
      assert(send_tasks->is_closed());
      break;
    }

    uint64_t wait_time_us = 0;
    TimeWatch start;
    send_blocks.clear();
    send_reqs.clear();
    for (auto task : *batch.tasks) {
      if (task->new_tokens() <= 0 ||
          task->dst_inst_id != dst_id ||
          task->dst_worker_id != dst_worker_id) {
        continue;
      }
      send_reqs.push_back(task);
      parse_block_send(src_info, dst_info, task, send_blocks);
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
    min_size = std::min(min, min_size);
    max_size = std::max(max, max_size);
    total_size += total;

    for (auto i = start_layer; i < num_layers; ++i) {
      TimeWatch wait_start;
      batch.step->wait_layer_ready(i);
      wait_time_us += wait_start.get_elapse_us();

      // NOTE: write 可能是异步的! write 返回并不意味着数据发送了!
      channel->send_data(i, send_blocks);
    }
    channel->flush();
    auto elapse = start.get_elapse_us();
    batch.step->finish_one();
    LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
              << "): finish send " << send_reqs.size() << " requests("
              << sb_num << "->" << cnt << " blocks), elapse: total use "
              << elapse << "us, wait use " << wait_time_us << "us;";
    auto iter = Iterator<const RequestInfo *>::copy_from(send_reqs)
        .filter([](auto t) { return t->reach_last_token(); });
    if (iter.has_next()) {
      channel->send_notification(&iter);
    }

    for (auto task : send_reqs) {
      task->clear_new_tokens();
      if (!task->reach_last_token()) {
        continue;
      }
      task->set_transfer_done();
      const auto block_n = task->dst_blocks().size();
      min_block_n = std::min(min_block_n, block_n);
      max_block_n = std::max(max_block_n, block_n);
      total_block_n += block_n;
      finished_req_n += 1;
      LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
                << "): send finish notification of request(" << task->req_id << ");";
    }

    if (step_idx > 0 && (step_idx & 15) == 0) {
      LOG(INFO) << "KVT tx_stub(" << dst_id << ":" << dst_worker_id
                << ") metric: block_size(min=" << min_size << ",max=" << max_size << "total=" << total_size
                << "), req_blocks(min=" << min_block_n << ",max=" << max_block_n << ", total=" << total_block_n
                << "), finished_reqs=" << finished_req_n << ")";
    }
  }
  running->store(false, std::memory_order_release);
  LOG(INFO) << "KVT tx_stub: stop to send kv of to worker("
            << dst_id << ":" << dst_worker_id << ");";
}

KvSendStub::KvSendStub(KvSendStub &&other) noexcept:
    dst_info_(other.dst_info_),
    start_layer_(other.start_layer_),
    num_layers_(other.num_layers_),
    ch_(std::move(other.ch_)),
    is_running_(false) {
  assert(!other.is_running());
}

void KvSendStub::connect(Context *ctx, const WorkerInfo &dst_info) {
  ch_->connect(dst_info);
  dst_info_ = dst_info;
  send_backend_.emplace(&pull_send, start_layer_, num_layers_, &ctx->worker_info(),
                        &dst_info_, ch_.get(), &is_running_, &send_tasks_);
  is_running_.store(true, std::memory_order_release);
}

void KvSendStub::send_batch(const BatchSendTask& batch) {
  auto num_tasks = batch.tasks->size();
  if (num_tasks > 0) {
    auto task = batch;
    batch.step->start_one();
    send_tasks_.push(std::move(task));
  }
}

bool KvSendStub::is_running() {
  return is_running_.load(std::memory_order_relaxed);
}

KvSendStub::~KvSendStub() {
  send_tasks_.close();
  if (send_backend_.has_value()) {
    send_backend_.value().join();
    send_backend_.reset();
  }
}
SendStub KvSendStubFactory::create_stub(InstanceId dst_inst_id,
                                        WorkerId dst_worker_id,
                                        uint32_t start_layer,
                                        uint32_t num_layers) {
  auto ch = create_channel(ctx_);
  return std::make_unique<KvSendStub>(dst_inst_id, dst_worker_id, start_layer, num_layers, create_channel(ctx_));
}
}
