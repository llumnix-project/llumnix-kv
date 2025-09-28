#include "tx_stub.h"
#include "utils/timer.h"
#include "thrid_party/logging.h"
#include "utils/socket_helper.h"
#include "channel.h"
#include "envcfg.h"
#include "naming/fake_naming.h"
#include <string.h>
#include <unistd.h>
#include "fault_inject.h"

namespace blade_llm {

static void do_parse_block_send_p_eq_d(
  size_t block_size, size_t token_size,
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks
) {
  // ntpb: number tokens per block
  uint32_t src_ntpb = block_size / token_size;
  uint32_t dst_ntpb = src_ntpb;
  assert(src_ntpb * token_size == block_size);

  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;
  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  // assert(src_blocks.size() == dst_blocks.size());

  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / src_ntpb;
    auto src_token_idx = wrote_tokens % src_ntpb;
    assert(src_block_idx < src_blocks.size());
    size_t src_offset = src_blocks[src_block_idx] * block_size
        + src_token_idx * token_size;
    auto tokens = std::min(src_ntpb - src_token_idx, left_tokens);
    auto dst_block_idx = wrote_tokens / dst_ntpb;
    auto dst_token_idx = wrote_tokens % dst_ntpb;
    assert(dst_block_idx < dst_blocks.size());
    size_t dst_offset = dst_blocks[dst_block_idx] * block_size
        + dst_token_idx * token_size;
    size_t length = tokens * token_size;
    send_blocks.emplace_back(src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return ;
}

// RAGGED_FLASH_CACHE_SHAPE
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
  return do_parse_block_send_p_eq_d(block_size, token_size, task, send_blocks);
}

// FLASH_CACHE_SHAPE
static void vllm_parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks) {
  // as tp size is same, token size should be same;
  assert(src_worker_info->tp_size == dst_worker_info->tp_size);
  assert(src_worker_info->token_size == dst_worker_info->token_size);
  assert(src_worker_info->block_size == dst_worker_info->block_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  size_t const kv_token_size = src_worker_info->token_size;
  size_t const kv_block_size = src_worker_info->block_size;
  size_t const k_token_size = kv_token_size / 2;
  size_t const k_block_size = kv_block_size / 2;
  assert(2 * k_token_size == kv_token_size);
  assert(2 * k_block_size == kv_block_size);
  size_t const src_layer_size = k_block_size * src_worker_info->layer_num_blocks;
  size_t const dst_layer_size = k_block_size * dst_worker_info->layer_num_blocks;
  // src_layer_size, dst_layer_size 并不一定要相等.

  size_t sb_idx = send_blocks.size();  // sb: send block~
  do_parse_block_send_p_eq_d(k_block_size, k_token_size, task, send_blocks);
  size_t const sb_end_idx = send_blocks.size();

  for (; sb_idx < sb_end_idx; ++sb_idx) {
    auto sb = send_blocks[sb_idx];
    sb.src_offset += src_layer_size;
    sb.dst_offset += dst_layer_size;
    send_blocks.emplace_back(std::move(sb));
  }

  return ;
}

static void parse_block_send_gt(
  size_t p_token_size, size_t p_k_size, size_t d_token_size,
  uint32_t ntpb, size_t group_off,
  const std::vector<uint32_t>& p_blocks,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &send_blocks
) {
  assert(p_k_size == p_token_size || p_k_size * 2 == p_token_size);
  const size_t p_block_size = p_token_size * ntpb;
  const size_t d_block_size = d_token_size * ntpb;

  while (left_tokens > 0) {
    const uint32_t block_idx = wrote_tokens / ntpb;
    assert(block_idx < p_blocks.size() && block_idx < d_blocks.size());
    const uint32_t token_idx_base = wrote_tokens % ntpb;
    const size_t p_blk_off = p_blocks[block_idx] * p_block_size;
    const size_t d_blk_off = d_blocks[block_idx] * d_block_size;
    const uint32_t tokens = std::min(ntpb - token_idx_base, left_tokens);

    for (uint32_t idx = 0; idx < tokens; ++idx) {
      const uint32_t token_idx = token_idx_base + idx;
      const size_t pk_token_off = p_blk_off + token_idx * p_token_size;
      const size_t d_token_off = d_blk_off + token_idx * d_token_size;
      const size_t dk_token_off = d_token_off + group_off * p_k_size;
      send_blocks.emplace_back(pk_token_off, dk_token_off, p_k_size);
    }

    wrote_tokens += tokens;
    left_tokens -= tokens;
  }

  return;
}

static void do_parse_block_send(
  const WorkerInfo *p_info,  // src
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,  // dst
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &send_blocks) {
  // for bladellm
  // cache shape [num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]
  // p_info->block_size: uint32_t 类型, 需要先转 size_t 不然仍有溢出风险.
  const size_t p_block_size = p_info->block_size;
  const size_t d_block_size = d_info->block_size;
  const size_t p_token_size = p_info->token_size;
  const size_t d_token_size = d_info->token_size;
  assert(p_info->tp_size > d_info->tp_size);
  assert((p_info->tp_size % d_info->tp_size) == 0);
  const uint32_t group_n = p_info->tp_size / d_info->tp_size;
  assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
  const uint32_t group_off = p_info->worker_tp_rank % group_n;
  assert(d_token_size == p_token_size * group_n);
  assert(d_token_size % 2 == 0);
  assert(p_token_size % 2 == 0);
  const size_t p_k_size = p_token_size / 2;
  const size_t d_k_size = d_token_size / 2;
  assert(d_k_size == p_k_size * group_n);
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_token_size;
  assert(ntpb * p_token_size == p_block_size);
  assert(ntpb * d_token_size == d_block_size);

  const size_t sbsize = send_blocks.size();
  parse_block_send_gt(
    p_token_size, p_k_size, d_token_size, ntpb, group_off,
    p_blocks, d_blocks, wrote_tokens, left_tokens, send_blocks);
  const size_t sbsize_after = send_blocks.size();

  for (size_t idx = sbsize; idx < sbsize_after; ++idx) {
    const auto& sb = send_blocks[idx];
    const size_t pv_token_off = sb.src_offset + p_k_size;
    const size_t dv_token_off = sb.dst_offset + d_k_size;
    assert(p_k_size == sb.length);
    send_blocks.emplace_back(pv_token_off, dv_token_off, p_k_size);
  }
  return;
}


// FLASH_CACHE_SHAPE
// (2, num_blocks, block_size, num_kv_heads, head_dim)
static void vllm_parse_block_send_p_gt_d(
  const WorkerInfo *p_info,  // prefill
  const WorkerInfo *d_info,  // decode
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks) {
  // p_info->block_size: uint32_t 类型, 需要先转 size_t 不然仍有溢出风险.
  const size_t p_block_size = p_info->block_size;
  const size_t d_block_size = d_info->block_size;
  const size_t p_token_size = p_info->token_size;
  const size_t d_token_size = d_info->token_size;
  assert(p_info->tp_size > d_info->tp_size);
  assert((p_info->tp_size % d_info->tp_size) == 0);
  const uint32_t group_n = p_info->tp_size / d_info->tp_size;
  assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
  const uint32_t group_off = p_info->worker_tp_rank % group_n;
  assert(d_token_size == p_token_size * group_n);
  assert(d_token_size % 2 == 0);
  assert(p_token_size % 2 == 0);
  const size_t p_k_size = p_token_size / 2;
  const size_t d_k_size = d_token_size / 2;
  assert(d_k_size == p_k_size * group_n);
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_token_size;
  assert(ntpb * p_token_size == p_block_size);
  assert(ntpb * d_token_size == d_block_size);

  size_t sb_idx = send_blocks.size();  // sb: send block~
  parse_block_send_gt(
    p_k_size, p_k_size, d_k_size, ntpb, group_off,
    task->src_blocks(), task->dst_blocks(),
    task->seen_tokens, task->new_tokens, send_blocks);
  size_t const sb_end_idx = send_blocks.size();

  size_t const pk_layer_size = ntpb * p_k_size * p_info->layer_num_blocks;
  size_t const dk_layer_size = ntpb * d_k_size * d_info->layer_num_blocks;
  for (; sb_idx < sb_end_idx; ++sb_idx) {
    auto sb = send_blocks[sb_idx];
    sb.src_offset += pk_layer_size;
    sb.dst_offset += dk_layer_size;
    send_blocks.emplace_back(std::move(sb));
  }

  return;
}

static void parse_block_send_p_gt_d(
  const WorkerInfo *p_info,  // src
  const WorkerInfo *d_info,  // dst
  const ReqSendTask *task,
  std::vector<IpcBlock> &send_blocks) {
  if (p_info->token_size == d_info->token_size) {
    // 此时表明 kvcache 在各个 P rank 之间是完全一样的.
    // 只需要 tp rank=0 的 worker 传输 kvcache 即可.
    // 后续这里可以让所有 worker 都参与进来, 每个 worker 传 1 部分.
    if (p_info->worker_tp_rank != 0) {
      return ;
    }

    assert(d_info->worker_tp_rank == 0);
    assert(p_info->block_size == d_info->block_size);
    auto token_size = p_info->token_size;
    size_t block_size = p_info->block_size;
    return do_parse_block_send_p_eq_d(block_size, token_size, task, send_blocks);
  }

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


static char* expand_vec(std::vector<char>& buf, size_t s) {
  auto old_s = buf.size();
  buf.resize(old_s + s);
  return buf.data() + old_s;
}

// see vllm vllm/v1/hybrid_connector/__init__.py
static constexpr uint32_t SEND_DONE_REQ = 0x20181219ul;
static constexpr uint32_t SAVE_DONE2_REQ = 0x20181223ul;
static constexpr uint32_t SEND_DONE_RESP = 0x91218102ul;

struct KvSendStub::TaskContext {
  KvSendStub* const stub = nullptr;

  // runtime state
  std::vector<char> send_done_buf;
  std::optional<FdGuard> send_done_sock;
  const sockaddr_in* const send_done_addr = env_send_done_addr();  // OWNER: GLOBAL
  std::string flush_out_buf;
  std::vector<IpcBlock> send_blocks;
  std::vector<const ReqSendTask *> send_req;
  std::vector<const ReqSendTask *> finished_req;
  TPKind tpkind = TPKind::UNKNOWN;
  ParseBlockFunc* parse_block = nullptr;
  std::optional<WorkerInfo> dstinfo;
  Channel ch;
  uint64_t wait_time_us = 0;
  uint64_t wait_and_send_us = 0;
  uint64_t send_notify_us = 0;
  Timepoint iter_start_ts;
  Timepoint send_finish_ts;

public:
  TaskContext(KvSendStub* s) noexcept
    : stub(s) {
  }

  void do_task(BatchSendTask& batch) {
    auto& self = *this;
    auto const step_idx = batch.step->step_idx;
    const auto& dst_id = self.stub->dstid_;
    const auto dst_worker_id = self.stub->dstworkerid_;

    fault_inject_sleep(300 * 1000);

    self.iter_start_ts = SteadyClock::now();  // iterator begin;
    auto const queue_us = elapse_us(batch.step->start_send_ts, self.iter_start_ts);

    // prepare
    assert(self.flush_out_buf.empty());
    assert(self.send_req.empty());
    assert(self.finished_req.empty());
    assert(self.send_blocks.empty());
    assert(!batch.tasks.empty());
    for (const auto& task : batch.tasks) {
      assert(task.dst_inst_id() == dst_id);
      assert(task.dst_worker_id() == dst_worker_id);
      if (task.reach_last_token) {
        self.finished_req.emplace_back(&task);
      }
      if (task.new_tokens <= 0) {
        continue;
      }

      if (task.state() == ReqState::FAILED) {
        continue;
      }
      assert(task.state() == ReqState::INPROCESS);
      assert(task.new_tokens > 0);
      self.send_req.emplace_back(&task);
    }

    if (!self.send_req.empty()) {
      try {
        self.try_create_channel();
        const auto& srcinfo = self.stub->src_info_;
        assert(self.parse_block != nullptr);
        assert(self.tpkind != TPKind::UNKNOWN);
        for (auto* task : self.send_req) {
          self.parse_block(&srcinfo, &self.dstinfo.value(), task, self.send_blocks);
        }
        if (!self.send_blocks.empty()) {
          self.do_send(batch);
        }
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
      }
    }

    for (auto task : self.finished_req) {
      assert(task->reach_last_token);
      if (task->state() == ReqState::INPROCESS) {
        task->set_state(ReqState::OK);
      }

      LOG(INFO) << "KVT tx_stub. DstId=" << dst_id << ",DstWorkerId=" << dst_worker_id
                << ",StepIdx=" << step_idx
                << ",State=" << static_cast<int>(task->state())
                << ",FinishedReq=" << task->req_id();
    }

    self.send_notify_us = 0;
    if (!self.finished_req.empty()) {
      self.send_done(self.finished_req);
      auto send_notify_end_ts = SteadyClock::now();
      self.send_notify_us = elapse_us(self.send_finish_ts, send_notify_end_ts);
      self.send_finish_ts = send_notify_end_ts;
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
    self.send_req.clear();
    self.finished_req.clear();
  }
private:
  // send self.send_done_buf to self.send_done_sock
  void do_rpc_send_done(int respsize) {
    auto& self = *this;

    if (!self.send_done_sock.has_value()) {
      int sock = socket(AF_INET, SOCK_STREAM, 0);
      if (sock == -1) {
        int errno_bak = errno;
        auto errmsg = std::string("rpc send done: create socket error. errno=");
        errmsg += std::to_string(errno_bak);
        throw std::runtime_error(std::move(errmsg));
      }
      self.send_done_sock.emplace(sock);

      auto* addr = reinterpret_cast<const sockaddr*>(self.send_done_addr);
      int sysok = connect(sock, addr, sizeof(*self.send_done_addr));
      if (sysok == -1) {
        int errno_bak = errno;
        auto errmsg = std::string("rpc send done: connect error. errno=");
        errmsg += std::to_string(errno_bak);
        throw std::runtime_error(std::move(errmsg));
      }
    }
    int sock = self.send_done_sock->fd();

    const auto& sbuf = self.send_done_buf;
    int sysok = write_sock(sock, sbuf.data(), sbuf.size());
    if (sysok == -1) {
      int errno_bak = errno;
      auto errmsg = std::string("rpc send done: write error. errno=");
      errmsg += std::to_string(errno_bak);
      throw std::runtime_error(std::move(errmsg));
    }

    char respbuf[16];
    assert(respsize < int(sizeof(respbuf)));
    sysok = read_sock(sock, respbuf, respsize);
    if (sysok == -1) {
      int errno_bak = errno;
      auto errmsg = std::string("rpc send done: read error. errno=");
      errmsg += std::to_string(errno_bak);
      throw std::runtime_error(std::move(errmsg));
    }
    if (sysok != respsize) {
      auto errmsg = std::string("rpc send done: invalid response");
      throw std::runtime_error(std::move(errmsg));
    }
    return ;
  }

  void rpc_send_done(const std::vector<const ReqSendTask*>& reqs) {
    auto& self = *this;

    // see vllm disagg.py
    int respsize = 4;
    uint32_t header = SEND_DONE_REQ;
    uint32_t worker_tp_rank = self.stub->src_info_.worker_tp_rank;
    static constexpr uint32_t HAS_VER2 = 0x40000000u;
    static constexpr uint32_t CODE_INTERNALERROR = 500;
    assert(worker_tp_rank < 0xffff);
    worker_tp_rank |= HAS_VER2;
    uint32_t num_req = reqs.size();

    // len('cfb0aa74-6752-9bcd-879e-19d1d1cf368b-ee5d8cdc-57d1-478f-a1d7-5b0c05bd8fb3')
    // == 73 一个典型的 dash reqid 长度.
    self.send_done_buf.reserve((4 + 4 + 4 + num_req * (4 + 73 + 4 + 4)) * 2ul);

    self.send_done_buf.resize(4 + 4 + 4);
    memcpy(self.send_done_buf.data() + 0, &header, 4);
    memcpy(self.send_done_buf.data() + 4, &worker_tp_rank, 4);
    memcpy(self.send_done_buf.data() + 8, &num_req, 4);
    for (const auto* req : reqs) {
      assert(req->reach_last_token);
      uint32_t code = (req->state() == ReqState::OK) ? 0 : CODE_INTERNALERROR;
      uint32_t plen = req->end_tokens();
      const std::string& reqid = req->req_id();
      uint32_t rlen = reqid.size();
      char* dst = expand_vec(self.send_done_buf, 4 + 4 + 4 + rlen);
      memcpy(dst, &plen, 4);
      memcpy(dst + 4, &code, 4);
      memcpy(dst + 8, &rlen, 4);
      memcpy(dst + 12, reqid.data(), rlen);
    }

    if (env_send_done_head_kind() == SEND_SAVE_DONE_HEAD_KIND) {
      size_t const reqsize = self.send_done_buf.size();
      char* dst = expand_vec(self.send_done_buf, reqsize);
      memcpy(dst, self.send_done_buf.data(), reqsize);
      uint32_t save_header = SAVE_DONE2_REQ;
      memcpy(dst, &save_header, sizeof(uint32_t));
      respsize = 8;
    }

    // 考虑到我们使用的是一个长链接, 其可能已经失效了, 这里会在需要的时候,
    // 进行重试, 重新创建一个连接. 当然, 一次就好~
    try {
      self.do_rpc_send_done(respsize);
      return ;
    } catch (const std::exception& ex) {
      LOG(ERROR) << "do_rpc_send_done failed. ex=" << ex.what();
      self.send_done_sock.reset();
    }

    try {
      self.do_rpc_send_done(respsize);
      return ;
    } catch (const std::exception& ex) {
      LOG(ERROR) << "do_rpc_send_done failed. ex=" << ex.what();
      self.send_done_sock.reset();
      throw;
    }
    return ;
}

  void send_done(const std::vector<const ReqSendTask*>& reqs) {
    auto& self = *this;
    if (self.send_done_addr == nullptr) {
      // channel send done, used in bladellm
      if (self.ch) {
        self.ch->send_notification(self.finished_req);
      }
    } else {
      // rpc send done, used in vllm
      self.rpc_send_done(self.finished_req);
    }
    return;
  }

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
    fault_inject_throw();
    self.ch->flush(self.flush_out_buf);
    self.send_finish_ts = SteadyClock::now();
    self.wait_and_send_us = elapse_us(self.iter_start_ts, self.send_finish_ts);
    return ;
  }

  void refresh_dst_info() {
    auto& self = *this;
    const auto& dstid = self.stub->dstid_;
    const auto dstwid = self.stub->dstworkerid_;
    auto dstinfoopt = self.stub->naming_->get_worker_info(dstid, dstwid);
    if (!dstinfoopt) {
      if (self.dstinfo.has_value()) {
        LOG(WARNING) << "refresh_dst_info:use origdstinfo=" << self.dstinfo->to_string();
        return;
      }
      LOG(ERROR) << "refresh_dst_info:origdstinfo=none dstid=" << dstid
                 << " dstwid=" << dstwid;
      throw std::runtime_error("can't find target worker");
    }
    if (self.dstinfo.has_value() && !same_dst(*self.dstinfo, *dstinfoopt)) {
      LOG(WARNING) << "TaskContext.dst_info: invalid dst: orig_dst_info="
                  << self.dstinfo->to_string()
                  << "bad_dst_info=" << dstinfoopt->to_string();
      return;
    }
    if (self.dstinfo.has_value()) {
      self.dstinfo->addr = std::move(dstinfoopt->addr);
      self.dstinfo->other_info = std::move(dstinfoopt->other_info);
      return;
    }
    self.dstinfo = dstinfoopt;

    const auto& srcinfo = self.stub->src_info_;
    assert(self.parse_block == nullptr);
    assert(self.tpkind == TPKind::UNKNOWN);
    const int cache_shape = env_cache_shape();
    if (cache_shape == RAGGED_FLASH_CACHE_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = parse_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        self.parse_block = parse_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
      self.parse_block = parse_block_send_p_lt_d;
      self.tpkind = TPKind::PLTD;
      return;
    }
    if (cache_shape == FLASH_CACHE_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    throw std::runtime_error("unsupported cache_shape");
  }

  void try_create_channel() {
    auto& self = *this;
    if (self.ch) {
      assert(self.dstinfo.has_value());
      return ;
    }
    self.refresh_dst_info();
    assert(self.dstinfo.has_value());
    self.ch = self.stub->channel_factory_->create(*self.dstinfo);
    LOG(INFO) << "create channel. dst=" << self.dstinfo->to_string()
              << ";ch=" << self.ch.get();
    return ;
  }
};

void KvSendStub::start_async() {
  auto taskctx = TaskContext(this);

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
      LOG(ERROR) << "do task failed. DstId=" << this->dstid_
                  << ",DstWorkerId=" << this->dstworkerid_
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
            << this->dstid_ << ":" << this->dstworkerid_ << ");";
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
    LOG(WARNING) << "KVT: SendStub(" << dstid_ << ":" << dstworkerid_
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
                                        std::optional<TransferProtocol> proto_opt,
                                        const std::optional<std::string> &dst_info) {
  auto channel_factory = std::make_unique<ChannelFactory>(ctx_, proto_opt);
  std::shared_ptr<INamingWorkerClient> naming_worker;
  if (dst_info) {
    naming_worker = std::make_shared<FakeNamingWorkerClient>(dst_info.value());
  } else {
    naming_worker = naming_worker_;
  }
  return std::make_unique<KvSendStub>(dst_inst_name, dst_worker_id, ctx_->worker_info(),
                                      start_layer, num_layers,
                                      std::move(channel_factory),
                                      std::move(naming_worker));
}
}
