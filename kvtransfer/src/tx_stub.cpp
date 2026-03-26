#include "tx_stub.h"
#include "parse_block_common.h"
#include "utils/timer.h"
#include "thrid_party/logging.h"
#include "utils/socket_helper.h"
#include "channel.h"
#include "envcfg.h"
#include "naming/fake_naming.h"
#include <string.h>
#include <unistd.h>
#include "fault_inject.h"
#include <cstdlib>
#include <cstring>

namespace blade_llm {

std::bitset<MAX_TP_SIZE> compute_valid_ranks_pd(uint32_t p_tp, uint32_t d_tp, int num_kv_heads) {
  std::bitset<MAX_TP_SIZE> result;

  if (num_kv_heads <= 0 || static_cast<uint32_t>(num_kv_heads) >= p_tp) {
    for (uint32_t i = 0; i < p_tp; ++i) {
      result.set(i);
    }
    return result;
  }

  if (d_tp <= static_cast<uint32_t>(num_kv_heads)) {
    assert(p_tp % num_kv_heads == 0);
    uint32_t stride = p_tp / static_cast<uint32_t>(num_kv_heads);
    for (int i = 0; i < num_kv_heads; ++i) {
      result.set(static_cast<uint32_t>(i) * stride);
    }
    return result;
  }

  assert(d_tp % num_kv_heads == 0);
  assert(p_tp % d_tp == 0);

  uint32_t ranks_per_head_d = d_tp / static_cast<uint32_t>(num_kv_heads);
  uint32_t p_ranks_per_d_rank = p_tp / d_tp;

  for (int head = 0; head < num_kv_heads; ++head) {
    uint32_t d_rank = head * ranks_per_head_d;
    for (uint32_t p_offset = 0; p_offset < p_ranks_per_d_rank; ++p_offset) {
      uint32_t p_rank = d_rank * p_ranks_per_d_rank + p_offset;
      result.set(p_rank);
    }
  }
  return result;
}

static bool same_dst(const WorkerInfo& l, const WorkerInfo& r) noexcept {
  return l.inst_id == r.inst_id &&
         l.worker_id == r.worker_id &&
         l.kvt_tp_size == r.kvt_tp_size &&
         l.worker_tp_rank == r.worker_tp_rank &&
         l.block_sizes == r.block_sizes &&
         l.token_sizes == r.token_sizes &&
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
  std::vector<std::vector<IpcBlock>> send_blocks;
  std::vector<const ReqSendTask *> send_req;
  std::vector<const ReqSendTask *> finished_req;
  TPKind tpkind = TPKind::UNKNOWN;
  ParseBlockFunc parse_block = nullptr;
  std::optional<WorkerInfo> dstinfo;
  Channel ch;

  // Dynamic computed values (depend on dst info)
  std::bitset<MAX_TP_SIZE> valid_ranks;
  uint32_t kvt_tp_size{0};
  uint32_t kvt_tp_rank{0};
  uint64_t wait_time_us = 0;
  uint64_t wait_and_send_us = 0;
  uint64_t send_notify_us = 0;
  Timepoint iter_start_ts;
  Timepoint send_finish_ts;
public:
  TaskContext(KvSendStub* s) noexcept
    : stub(s) {
  }

  void do_task(BatchSendTask& batch) noexcept {
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
          self.parse_block(&srcinfo, &self.dstinfo.value(), self.valid_ranks, self.kvt_tp_size, self.kvt_tp_rank, task, self.send_blocks);
        }
        if (!self.send_blocks.empty() && !self.send_blocks[0].empty()) {
#ifndef NDEBUG
          for (const auto& sb : self.send_blocks) {
            assert(!sb.empty());
          }
#endif
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
        auto final_state = ReqState::OK;
        if (task->new_tokens <= 0) {
          final_state = ReqState::FAILED;
        }
        task->set_state(final_state);
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
    self.send_blocks.shrink_to_fit();
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
    // == 73, a typical dual-request reqid length.
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

    // Since we use a persistent connection that may have become stale,
    // we retry and recreate the connection if needed. Just once though.
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
      // throw;
    }
    return ;
}

  void send_done(const std::vector<const ReqSendTask*>& reqs) {
    auto& self = *this;
    // env_send_done_addr() is always non-null, use rpc send done
    self.rpc_send_done(self.finished_req);
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
    const int cache_shape = env_cache_shape();

    self.valid_ranks = compute_valid_ranks_pd(
        srcinfo.kvt_tp_size,
        self.dstinfo->kvt_tp_size,
        srcinfo.num_kv_heads);
    self.kvt_tp_size = static_cast<uint32_t>(self.valid_ranks.count());
    self.kvt_tp_rank = static_cast<uint32_t>(
        (self.valid_ranks << (self.valid_ranks.size() - srcinfo.worker_tp_rank)).count());

    assert(self.parse_block == nullptr);
    assert(self.tpkind == TPKind::UNKNOWN);

    if (cache_shape == RAGGED_FLASH_CACHE_SHAPE) {
      if (srcinfo.kvt_tp_size == self.dstinfo->kvt_tp_size) {
        self.parse_block = parse_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.kvt_tp_size > self.dstinfo->kvt_tp_size) {
        if (srcinfo.token_sizes == self.dstinfo->token_sizes) {
          self.parse_block = parse_block_send_p_gt_d_dpsk;
          self.tpkind = TPKind::PEQD;
        } else {
          self.parse_block = parse_block_send_p_gt_d;
          self.tpkind = TPKind::PGTD;
        }
        return;
      }
      self.parse_block = parse_block_send_p_lt_d;
      self.tpkind = TPKind::PLTD;
      return;
    }
    if (cache_shape == FLASH_CACHE_SHAPE) {
      if (srcinfo.kvt_tp_size == self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.kvt_tp_size > self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    if (cache_shape == QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      auto p_tp_size = srcinfo.engine_tp_size;

      if (p_tp_size == self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_hybrid_block_send_p_eq_d_block_aligned;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (p_tp_size > self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_hybrid_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    if (cache_shape == DPSK_V32_SPARSE_MLA_SHAPE) {
      if (srcinfo.kvt_tp_size == self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_block_send_multi_tensor_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.kvt_tp_size > self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_block_send_multi_tensor_p_gt_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      self.parse_block = vllm_parse_block_send_multi_tensor_p_lt_d;
      self.tpkind = TPKind::PEQD;
      return;
    }
    if (cache_shape == FLASHINFER_CACHE_SHAPE) {
      if (srcinfo.kvt_tp_size == self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_flashinfer_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.kvt_tp_size > self.dstinfo->kvt_tp_size) {
        self.parse_block = vllm_parse_flashinfer_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    throw std::runtime_error("unsupported cache_shape");
  }

  void try_create_channel() {
    auto& self = *this;
    if (self.ch && self.ch->is_active()) {
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

void KvSendStub::send_batch(BatchSendTask batch) noexcept {
  assert(!batch.tasks.empty());
  auto& ctx = *this->taskctx_;
  ctx.do_task(batch);
  ctx.clear();
  return;
}

KvSendStub::~KvSendStub() {}

KvSendStub::KvSendStub(InstanceId dstid, WorkerId dstworkerid,
             const WorkerInfo &src_info,
             uint32_t start_layer,
             uint32_t num_layers,
             std::unique_ptr<IChannelFactory> channel_factory,
             std::shared_ptr<INamingWorkerClient> naming) :
      ISendStub(std::move(dstid), dstworkerid),
      src_info_(src_info),
      naming_(std::move(naming)),
      start_layer_(start_layer),
      num_layers_(num_layers),
      channel_factory_(std::move(channel_factory)),
      taskctx_(std::make_unique<TaskContext>(this)) {};

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
