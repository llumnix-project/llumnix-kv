
#include <future>
#include <sstream>
#include "protocol/rdma_channel.h"
#include "thrid_party/logging.h"
#include "assert.h"
#include "envcfg.h"
#include "utils/timer.h"
#include "utils/id_generator.h"
#include <iomanip>
#include <numeric>
#include <functional>
#include <zlib.h>
#include "fault_inject.h"

#ifdef ENABLE_RDMA
// just in this cpp
using namespace accl::barex;

namespace blade_llm {

static constexpr uint32_t MEM_HANDLES_REQ_MAGIC = 0x20181218;

// req:
// +--------+---------+--------+--------+--------+--------+
//    crc       cnt       off1     len1    off2     len2
// crc: uint32_t.
// cnt: uint32_t specifying the number of subsequent off/len pairs.
// off, len; uint64_t.
// resp:
// crc: uint32_t
static constexpr uint32_t REMOTE_CRC_REQ_MAGIC = 0x20250924;

struct GetMemHandles {
private:
  using Promise = std::promise<std::vector<RDMAMemHandle>>;
private:
  std::shared_ptr<XChannel> const ch_ = nullptr;
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;

public:
  GetMemHandles(std::shared_ptr<XChannel> ch, uint64_t r) noexcept:
    ch_(std::move(ch)),
    reqid_(r),
    pr_(std::make_shared<Promise>()) {}

  auto future() {
    return this->pr_->get_future();
  }

  void operator()(Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      std::stringstream ss;
      ss << "GetMemHandles error: ex=" << s.ErrMsg() << " reqid=" << self.reqid_ << " ch=" << self.ch_->ToString();
      auto ex = std::make_exception_ptr(std::runtime_error(std::move(ss).str()));
      self.pr_->set_exception(std::move(ex));
      return ;
    }
    RTASSERT(len >= RPC_HEADER);
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == MEM_HANDLES_REQ_MAGIC);
    RTASSERT(reqid == self.reqid_);

    buf += RPC_HEADER;
    len -= RPC_HEADER;
    RTASSERT(len % sizeof(RDMAMemHandle) == 0);
    size_t handle_n = len / sizeof(RDMAMemHandle);
    auto vec = std::vector<RDMAMemHandle>(handle_n);
    memcpy(vec.data(), buf, handle_n * sizeof(RDMAMemHandle));
    self.pr_->set_value(std::move(vec));
    return ;
  }
};

std::vector<RDMAMemHandle> get_mem_handles(const CliBarexCtx* ctx, std::shared_ptr<XChannel>& dst) {
  uint32_t magic = MEM_HANDLES_REQ_MAGIC;
  uint64_t reqid = new_id();
  memp_t bufmr = AllocCPUBuffer(dst, sizeof(magic) + sizeof(reqid));
  ser_rpc_header(bufmr.buf, magic, reqid);

  auto memhandles = GetMemHandles(dst, reqid);
  auto fut = memhandles.future();
  ctx->push(reqid, std::move(memhandles));

  Send(dst, std::move(bufmr), [ctx, reqid](Status s) {
    if (s.IsOk()) {
      return;
    }
    ctx->on_send_error(std::move(s), reqid);
  });

  auto futstate = fut.wait_for(std::chrono::seconds(env_rpc_timeout_s()));
  if (futstate != std::future_status::ready) {
    ctx->on_send_error(Status(BAREX_ERR_TIMEOUT), reqid);
  }
  return fut.get();
}

struct RemoteCrc {
private:
  using Promise = std::promise<uint32_t>;
private:
  std::shared_ptr<XChannel> const ch_ = nullptr;
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;

public:
  RemoteCrc(std::shared_ptr<XChannel> ch, uint64_t r) noexcept:
    ch_(std::move(ch)),
    reqid_(r),
    pr_(std::make_shared<Promise>()) {}

  auto future() {
    return this->pr_->get_future();
  }

  void operator()(Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      std::stringstream ss;
      ss << "RemoteCrc error: ex=" << s.ErrMsg() << " reqid=" << self.reqid_ << " ch=" << self.ch_->ToString();
      auto ex = std::make_exception_ptr(std::runtime_error(std::move(ss).str()));
      self.pr_->set_exception(std::move(ex));
      return ;
    }
    RTASSERT(len >= RPC_HEADER);
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == REMOTE_CRC_REQ_MAGIC);
    RTASSERT(reqid == self.reqid_);

    buf += RPC_HEADER;
    len -= RPC_HEADER;
    RTASSERT(len >= sizeof(uint32_t));
    uint32_t crc = 0;
    memcpy(&crc, buf, sizeof(uint32_t));
    self.pr_->set_value(crc);
    return;
  }
};

uint32_t get_remote_crc(CliBarexCtx* ctx, std::shared_ptr<XChannel>& dst, const std::vector<std::vector<IpcBlock>>* data, uint32_t lcrc) {
  assert(!data->empty());
  uint64_t const reqid = new_id();

  // calculate total size: tensor count + block count per tensor + all offset/length pairs
  uint32_t tensor_cnt = static_cast<uint32_t>(data->size());
  uint32_t total_blocks = 0;
  for (const auto& per_tensor_data : *data) {
    total_blocks += static_cast<uint32_t>(per_tensor_data.size());
  }

  // protocol format: 
  // (magic + reqid) + lcrc + tensor_cnt 
  // + [tensor0_block_cnt + off1+len1 + off2+len2 + ...] + [tensor1_block_cnt + ...] + ...
  const auto bodysize = sizeof(uint32_t) + // tensor_cnt
                        tensor_cnt * sizeof(uint32_t) + // number of blocks per tensor
                        total_blocks * (sizeof(uint64_t) + sizeof(uint64_t)); // all offset/length pairs
  memp_t bufmr = AllocCPUBuffer(dst, RPC_HEADER + sizeof(uint32_t) + bodysize);
  ser_rpc_header(bufmr.buf, REMOTE_CRC_REQ_MAGIC, reqid);
  char* bufstart = bufmr.buf + RPC_HEADER;
  const char* const bufend = bufmr.buf + bufmr.buf_len;
  memcpy(bufstart, &lcrc, sizeof(uint32_t));
  bufstart += sizeof(uint32_t);
  memcpy(bufstart, &tensor_cnt, sizeof(uint32_t));
  bufstart += sizeof(uint32_t);

  for (const auto& per_tensor_data : *data) {
    uint32_t block_cnt = static_cast<uint32_t>(per_tensor_data.size());
    memcpy(bufstart, &block_cnt, sizeof(uint32_t));
    bufstart += sizeof(uint32_t);

    for (const auto& sb : per_tensor_data) {
      RTASSERT((bufend - bufstart) >= 16 /* sizeof(uint64_t) * 2 */);
      memcpy(bufstart, &sb.dst_offset, sizeof(uint64_t));
      bufstart += sizeof(uint64_t);
      memcpy(bufstart, &sb.length, sizeof(uint64_t));
      bufstart += sizeof(uint64_t);
    }
  }

  auto req = RemoteCrc(dst, reqid);
  auto fut = req.future();
  ctx->push(reqid, std::move(req));

  Send(dst, std::move(bufmr), [ctx, reqid](Status s) {
    if (s.IsOk()) {
      return;
    }
    ctx->on_send_error(std::move(s), reqid);
  });

  auto futstate = fut.wait_for(std::chrono::seconds(env_rpc_timeout_s()));
  if (futstate != std::future_status::ready) {
    ctx->on_send_error(Status(BAREX_ERR_TIMEOUT), reqid);
  }
  return fut.get();
}

void RDMAServer::CtxCallback::resp_remote_crc(std::shared_ptr<XChannel>& channel, uint64_t reqid, char *inbuf, size_t inlen) {
  const auto tp1 = SteadyClock::now();
  const auto& layer_descs = this->server_->ctx_->layer_gdrcpy_mem();
  const auto& layer_mrs = this->server_->ctx_->layer_mrs();
  const char* const inbuf_bak = inbuf;
  const char* const inbuf_end = inbuf + inlen;
  RTASSERT(inlen > RPC_HEADER + sizeof(uint32_t) + sizeof(uint32_t));
  inbuf += RPC_HEADER;
  uint32_t local_crc, tensor_cnt;
  memcpy(&local_crc, inbuf, sizeof(uint32_t));
  inbuf += sizeof(uint32_t);
  memcpy(&tensor_cnt, inbuf, sizeof(uint32_t));
  inbuf += sizeof(uint32_t);

  size_t num_tensors_per_layer = layer_mrs.empty() ? 0 : layer_mrs[0].size();
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> tensor_offlens(tensor_cnt);

  bool crc_enabled = true;
  if (tensor_cnt != num_tensors_per_layer) {
    crc_enabled = false;
    LOG(WARNING) << "tensor_cnt != num_tensors_per_layer. tensor_cnt=" << tensor_cnt
    << " num_tensors_per_layer=" << num_tensors_per_layer << " reqid=" << reqid;
  } else {
    for (uint32_t tensor_idx = 0; tensor_idx < tensor_cnt; ++tensor_idx) {
      uint32_t block_cnt;
      RTASSERT(static_cast<size_t>(inbuf_end - inbuf) >= sizeof(uint32_t));
      memcpy(&block_cnt, inbuf, sizeof(uint32_t));
      inbuf += sizeof(uint32_t);

      tensor_offlens[tensor_idx].reserve(block_cnt);
      for (uint32_t i = 0; i < block_cnt; ++i) {
        RTASSERT((inbuf_end - inbuf) >= 16 /* sizeof(uint64_t) * 2 */);
        uint64_t off, len;
        memcpy(&off, inbuf, sizeof(uint64_t));
        inbuf += sizeof(uint64_t);
        memcpy(&len, inbuf, sizeof(uint64_t));
        inbuf += sizeof(uint64_t);
        tensor_offlens[tensor_idx].emplace_back(off, len);
      }
    }
  }
  const auto tp2 = SteadyClock::now();
  // rebuild crc based on layer_descs and tensor_offlens
  crc_enabled = !layer_descs.empty();
  uint32_t remote_crc = crc32_z(0L, Z_NULL, 0);
  for (size_t layer_idx = 0; layer_idx < layer_mrs.size(); ++layer_idx) {
    if (layer_idx >= layer_descs.size()) {
      crc_enabled = false;
      break;
    }
    const auto& layer_desc = layer_descs[layer_idx];
    if (tensor_cnt > layer_desc.size()) {
      crc_enabled = false;
      break;
    }
    for (uint32_t tensor_idx = 0; tensor_idx < tensor_cnt; ++tensor_idx) {
      const auto& tensor_desc = layer_desc[tensor_idx];
      if (!tensor_desc) {
        crc_enabled = false;
        break;
      }
      const Bytef *const tensor_cpu_ptr = (Bytef *)tensor_desc->cpu_ptr();

      for (const auto &[off, len] : tensor_offlens[tensor_idx]) {
        remote_crc = crc32_z(remote_crc, tensor_cpu_ptr + off, len);
      }
    }
  }
  const auto tp3 = SteadyClock::now();

  if (!crc_enabled) {
    LOG(WARNING) << "crc not enabled. use BLLM_KVTRANS_CRC=1";
    remote_crc = local_crc;
  }

  auto bufmr = AllocCPUBuffer(channel, RPC_HEADER + sizeof(uint32_t));
  memcpy(bufmr.buf, inbuf_bak, RPC_HEADER);
  memcpy(bufmr.buf + RPC_HEADER, &remote_crc, sizeof(uint32_t));
  Send(channel, std::move(bufmr),
       [channel, reqid, remote_crc, local_crc](Status s) {
         RTASSERT_EQ(local_crc, remote_crc);
         if (s.IsOk()) {
           return;
         }
         LOG(ERROR) << "resp_remote_crc err=" << s.ErrMsg()
                    << " reqid=" << reqid << " ch=" << channel->ToString();
       });
  const auto tp4 = SteadyClock::now();

  LOG(INFO) << "remote crc: reqid=" << reqid << " localcrc=" << local_crc
            << " remotecrc=" << remote_crc << " inlen=" << inlen
            << " parsedur_us=" << elapse_us(tp1, tp2)
            << " crcdur_us=" << elapse_us(tp2, tp3)
            << " senddur_us=" << elapse_us(tp3, tp4);
  return;
}

void RDMAServer::CtxCallback::resp_mem_handles(std::shared_ptr<XChannel>& channel, uint64_t reqid, char *inbuf, size_t inlen) {
  assert(inlen == RPC_HEADER);
  auto& self = *this;
  auto& handles = self.server_->info_.handles;
  const size_t msglen = handles.size() * sizeof(RDMAMemHandle);
  memp_t bufmr = AllocCPUBuffer(channel, RPC_HEADER + msglen);
  memcpy(bufmr.buf, inbuf, RPC_HEADER);
  memcpy(bufmr.buf + RPC_HEADER, handles.data(), msglen);

  fault_inject_throw();

  return Send(channel, std::move(bufmr), [channel, reqid](Status s) {
    if (s.IsOk()) {
      return;
    }
    LOG(ERROR) << "resp_mem_handles err=" << s.ErrMsg() << " reqid=" << reqid
               << " ch=" << channel->ToString();
  });
}

void RDMAServer::CtxCallback::OnRecvCall(std::shared_ptr<XChannel> ch, char *in_buf, size_t len, x_msg_header _header) noexcept {
  if (len >= RPC_HEADER) {
    const auto [magic, reqid] = deser_rpc_header(in_buf);
    if (magic == MEM_HANDLES_REQ_MAGIC) {
      try {
        this->resp_mem_handles(ch, reqid, in_buf, len);
      } catch (const std::exception &ex) {
        LOG(ERROR) << "resp mem handle: error=" << ex.what()
                   << ";ch=" << ch->ToString() << ";reqid=" << reqid;
      }
      return;
    }
    if (magic == REMOTE_CRC_REQ_MAGIC) {
      try {
        this->resp_remote_crc(ch, reqid, in_buf, len);
      } catch (const std::exception &ex) {
        LOG(ERROR) << "resp remote crc: error=" << ex.what()
                   << ";ch=" << ch->ToString() << ";reqid=" << reqid;
      }
      return;
    }
  }

  // No longer need to handle notification callbacks - using event-based mechanism now
  return;
}

void RDMAServer::start_server(Context *ctx) {
  auto &self = *this;
  RDMAInfo& info = self.info_;
  auto rdma_ctx = BarexProtoContext::server_context(
    "KVTServer",
    std::make_unique<CtxCallback>(this),
    TransferProtocol::Kind::RDMA_DIRECT
  );
  if (!rdma_ctx->check_support()) {
    throw std::runtime_error("can't start RDMA transfer server as RDMA protocol not support;");
  }
  ctx->register_protocol(std::move(rdma_ctx));

  WorkerInfo *winfo = ctx->worker_info_mutable();
  auto layer_num_blocks = ctx->layer_num_blocks();
  auto layer_ptr = ctx->layer_data_address();
  auto proto = TransferProtocol::rdma_direct();
  auto proto_ctx = ctx->get_protocol_ctx<BarexProtoContext>(proto);
  if (proto_ctx == nullptr) {
    throw std::runtime_error("KVT server: rdma context not register.");
  }
  auto barex_ctx = proto_ctx->barex_ctx();
  assert(self.ctx_ == nullptr);
  self.ctx_ = barex_ctx;

  info.handles.reserve(layer_ptr.size());
  for (size_t layer_idx = 0; layer_idx < layer_ptr.size(); ++layer_idx) {
    auto &layer_mrs = barex_ctx->layer_mrs()[layer_idx];
    auto &handle = info.handles.emplace_back();

    RTASSERT(layer_ptr[layer_idx].size() <= MAX_CACHE_NUM_PER_LAYER);
    RTASSERT_EQ(layer_ptr[layer_idx].size(), layer_mrs.size());
    // Now only support each layer have same number of cache
    for (size_t cache_idx = 0; cache_idx < layer_ptr[layer_idx].size();
         cache_idx++) {
      auto &out = layer_mrs[cache_idx].mr();
      auto layer_blk_p = reinterpret_cast<void *>(layer_ptr[layer_idx][cache_idx]);
      handle.ptrs[cache_idx] = layer_blk_p;
      handle.rkeys[cache_idx] = out.mr->rkey;
    }
  }
  RTASSERT_EQ(info.handles.size(), layer_ptr.size());

  get_ip(info.ip, INET_ADDRSTRLEN);
  info.port = env_port_base() + winfo->worker_id;
  RTASSERT(info.port > 0);

  {
    std::stringstream ss;
    ss << info.ip << ":" << info.port;
    winfo->addr = std::move(ss).str();
  }
  XListener *listener = nullptr;
  auto xctx = barex_ctx->xctx();
  if (xctx == nullptr) {
    LOG(ERROR) << "xctx is nullptr";
  }
  LOG(INFO) << "RDMAServer.start_server: ip=" << info.ip
            << " port=" << info.port
            << " layer_num_blocks=" << layer_num_blocks;
  auto result = XListener::NewInstance(listener, env_conn_tpsize(), info.port,
                                       TIMER_3S, {xctx});
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  self.listener_.reset(listener);
  result = self.listener_->Listen();
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
}

void RDMAChannel::connect(const WorkerInfo &dst_info) {
  auto &self = *this;
  std::string addr(dst_info.addr);
  auto pos = addr.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid rdma address: " + addr);
  }

  self.ip_ = addr.substr(0, pos);
  auto port_str = addr.substr(pos + 1, addr.size());
  self.port_ = std::stoi(port_str);
  for (auto &block_size : dst_info.block_sizes) {
    dst_layer_blk_sizes_.emplace_back(
      dst_info.layer_num_blocks * block_size // uint32_t * size_t = size_t
    );
  }
  dst_layer_num_ = dst_info.num_layers;
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());
  assert(dst_layer_blk_sizes_.size() == self.ctx_->layer_mrs().at(0).size());
}

RDMAChannel::~RDMAChannel() {
  delete_channels(this->ctx_, std::move(this->chs_));
  assert(this->chs_.empty());
  return;
}

void RDMAChannel::do_init() {
  auto &self = *this;
  if (valid_channels(self.chs_)) {
    assert(self.dst_handles_.size() == self.dst_layer_num_);
    return;
  }
  std::vector<BarexChannel> tmp_chs;
  tmp_chs.swap(self.chs_);
  delete_channels(self.ctx_, std::move(tmp_chs));

  auto init_time = TimeWatch();
  const int sp = env_send_parallel();
  assert(sp > 0);

  auto& chs = self.chs_;
  chs.reserve(sp);
  auto futs = std::vector<std::future<BarexChannel>>();
  futs.reserve(sp);
  auto conn = self.ctx_->connector();
  assert(conn != nullptr);
  for (int i = 0; i < sp; ++i) {
    auto fut = Connect(*conn, self.ip_, self.port_);
    fault_inject_throw();
    futs.emplace_back(std::move(fut));
  }

  for (auto &fut : futs) {
    chs.emplace_back(fut.get());
  }
  assert(!chs.empty());

#ifndef NDEBUG
  auto delay_ms = env_debug_tx_delay_ms();
  LOG(INFO) << "RDMAChannel connect: done: delayms=" << delay_ms;
  usleep(delay_ms * 1000);
#endif

  self.dst_handles_ = get_mem_handles(self.ctx_, chs[0].sch());
  LOG(INFO) << "RDMAChannel connect. dstip=" << self.ip_
            << ",init_us=" << init_time.get_elapse_us()
            << ",dstport=" << self.port_
            << ",dsthandles_n=" << self.dst_handles_.size();
  assert(valid_channels(self.chs_));
  assert(self.dst_handles_.size() == self.dst_layer_num_);
  return;
}

[[nodiscard]] static std::future<void>
WriteSingle(XChannel *ch, memp_t sdata, uint64_t raddr, uint32_t rkey) {
  // std::promise<void> pr;
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();

  // LOG(INFO) << "zydebug WriteSingle: raddr=" << raddr << " rkey=" << rkey;
  auto result = ch->WriteSingle(std::move(sdata), raddr, rkey, false, 0,
                                [pr = std::move(pr)](Status s) mutable {
                                  if (!s.IsOk()) {
                                    auto ex = std::make_exception_ptr(std::runtime_error("Write ERR: " + s.ErrMsg()));
                                    pr->set_exception(std::move(ex));
                                    return;
                                  }
                                  pr->set_value();
                                }
  );
  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<void>(std::move(ex));
  }

  return fut;
}

// return send_us
[[nodiscard]] static std::future<uint64_t> WriteBatch(XChannel *ch, std::shared_ptr<std::vector<rw_memp_t>> datasp) {
  // std::promise<void> pr;
  auto pr = std::make_shared<std::promise<uint64_t>>();
  auto fut = pr->get_future();
  auto datas = datasp;
  const auto write_start_ts = SteadyClock::now();
  auto result = ch->WriteBatch(std::move(datas),
                               [pr = std::move(pr), d = std::move(datasp), write_start_ts](Status s) mutable {
                                 // WriteBatch requires datasp to remain valid until the callback.
                                 if (!s.IsOk()) {
                                   auto ex = std::make_exception_ptr(std::runtime_error("Write ERR: " + s.ErrMsg()));
                                   pr->set_exception(std::move(ex));
                                   return;
                                 }
                                 auto const send_us = elapse_us(write_start_ts, SteadyClock::now());
                                 pr->set_value(send_us);
                               }
  );
  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<uint64_t>(std::move(ex));
  }

  return fut;
}

accl::barex::XChannel *RDMAChannel::ch() noexcept {
  return this->sch().get();
}

std::shared_ptr<accl::barex::XChannel>& RDMAChannel::sch() noexcept {
  auto &self = *this;
  int n = self.chs_.size();
  int idx = (++self.prev_ch_idx_) % n;
  return self.chs_[idx].sch();
}

// After grouping, input looks like:
// <src_off_1, dst_off_0, len1>
// <src_off_2, dst_off_0, len2>
// This means src_off_1,len1; src_off_2,len2 will be written to dst_off_0
// <src_off_3, dst_off_1, len3>
// <src_off_4, dst_off_1, len4>
// <src_off_5, dst_off_1, len5>
// return min_size, max_size, total_size, cnt
static std::tuple<size_t, size_t, size_t, size_t> group_by_dst(std::vector<IpcBlock>& input) {
  assert(!input.empty());

  std::sort(input.begin(), input.end(),
            [](const IpcBlock& x, const IpcBlock& y) { return x.dst_offset < y.dst_offset; });

  size_t min_size = UINT64_MAX;
  size_t max_size = 0;
  size_t total_size = 0;
  size_t cnt = 0;

  size_t prev_idx = 0;
  size_t prev_end = input[0].length + input[0].dst_offset;
  for (size_t idx = 1; idx < input.size(); ++idx) {
    auto& blk = input[idx];
    if (blk.dst_offset > prev_end) {
      cnt += 1;
      size_t blksize = prev_end - input[prev_idx].dst_offset;
      min_size = std::min(blksize, min_size);
      max_size = std::max(blksize, max_size);
      total_size += blksize;

      prev_idx = idx;
      prev_end = input[idx].length + input[idx].dst_offset;
      continue;
    }

    if (blk.dst_offset == prev_end) {
      input[idx].dst_offset = input[prev_idx].dst_offset;
      prev_end += input[idx].length;
      continue;
    }

    abort();
  }

  cnt += 1;
  size_t blksize = prev_end - input[prev_idx].dst_offset;
  min_size = std::min(blksize, min_size);
  max_size = std::max(blksize, max_size);
  total_size += blksize;
  return {min_size, max_size, total_size, cnt};
}

static size_t cdiv(size_t a, size_t b) { return (a + (b - 1)) / b; }

bool RDMAChannel::is_active() {
  auto& self = *this;
  if (self.chs_.empty()) {
    return true;
  }
  return valid_channels(self.chs_);
}

void RDMAChannel::register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) {
  auto& self = *this;
  assert(!data.empty());

  assert(self.data_ == nullptr);
  self.data_ = &data;
  self.kind_ = kind;
  self.do_init();

  self.enable_crc_ = env_crc();
  if (self.enable_crc_) {
    self.crc_ = crc32_z(0L, Z_NULL, 0);
  }

#ifndef NDEBUG
  size_t total_len_debug = 0;
  for (const auto &tensor_data : data) {
    for (const auto &item : tensor_data) {
      total_len_debug += item.length;
    }
  }
#endif

  self.origin_sb_num_ = 0;
  self.merged_sb_num_ = 0;
  self.sb_size_min_ = std::numeric_limits<size_t>::max();
  self.sb_size_max_ = 0;
  self.sb_size_total_ = 0;

  if (kind == TPKind::PEQD) {
    for (auto& tensor_data : data) {
      assert(!tensor_data.empty());
      self.origin_sb_num_ += tensor_data.size();
      auto const [min, max, total, cnt] = merge_interval(tensor_data);
      assert(cnt > 0);
      self.merged_sb_num_ += cnt;
      self.sb_size_min_ = std::min(self.sb_size_min_, min);
      self.sb_size_max_ = std::max(self.sb_size_max_, max);
      self.sb_size_total_ += total;

      auto new_end =
          std::remove_if(tensor_data.begin(), tensor_data.end(),
                         [](const IpcBlock &item) { return item.length == 0; });
      tensor_data.erase(new_end, tensor_data.end());
      assert(tensor_data.size() == cnt);
    }
  } else {
    assert(data.size() == 1);
    const auto& tensor_data = data[0];
    assert(!tensor_data.empty());
    self.origin_sb_num_ = tensor_data.size();
    self.merged_sb_num_ = tensor_data.size();
    self.sb_size_min_ = tensor_data[0].length;
    self.sb_size_max_ = tensor_data[0].length;
    self.sb_size_total_ = 0;
    for (const auto& blk : tensor_data) {
      self.sb_size_min_ = std::min(self.sb_size_min_, blk.length);
      self.sb_size_max_ = std::max(self.sb_size_max_, blk.length);
      self.sb_size_total_ += blk.length;
    }
  }
  assert(self.merged_sb_num_ > 0);
  assert(self.merged_sb_num_ <= self.origin_sb_num_);
  // assert(total_len_debug == self.sb_size_total_);

  self.dataperch_ = cdiv(self.merged_sb_num_, self.chs_.size());
  return;
}


// may be nullptr
static const Bytef* get_layer_tensor_cpu_ptr(CliBarexCtx* ctx, size_t layer_idx, size_t tensor_idx) {
  const auto& layer_descs = ctx->layer_gdrcpy_mem();
  const auto& layer_mrs = ctx->layer_mrs();
  if (layer_idx >= layer_mrs.size()) {
    return nullptr;
  }
  if (tensor_idx >= layer_descs[layer_idx].size()) {
    return nullptr;
  }
  const auto& tensor_desc = layer_descs[layer_idx][tensor_idx];
  return tensor_desc ? (Bytef*)tensor_desc->cpu_ptr() : (Bytef*)nullptr;
}

void RDMAChannel::send_data(size_t layer_idx) {
  auto &self = *this;
  assert(layer_idx < self.dst_layer_num_);
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());

  auto &dst_layer_handle = self.dst_handles_[layer_idx];
  const auto &data = *self.data_;
  assert(!data.empty());

  auto datasp = std::make_shared<std::vector<rw_memp_t>>();
  datasp->reserve(self.dataperch_);
  const auto &temp_src_mrs = self.ctx_->layer_mrs()[layer_idx];
  assert(data.size() == temp_src_mrs.size());

  uint32_t tensor_cnt = 0;
  for (const auto &per_tensor_data : data) {
    assert(tensor_cnt < self.ctx_->layer_blk_sizes.size());
    assert(tensor_cnt < self.dst_layer_blk_sizes_.size());
    assert(tensor_cnt < dst_layer_handle.ptrs.size());
    assert(tensor_cnt < temp_src_mrs.size());

    // Get the current tensor CPU pointer for CRC calculation
    const Bytef *const tensor_cpu_ptr = get_layer_tensor_cpu_ptr(self.ctx_, layer_idx, tensor_cnt);
    if (tensor_cpu_ptr == nullptr && self.enable_crc_) {
      LOG(WARNING) << "disable crc check. layer_idx=" << layer_idx << ", tensor_cnt=" << tensor_cnt;
      self.enable_crc_ = false;
    }

    for (const auto &[src_offset, dst_offset, len] : per_tensor_data) {
      const uint64_t layer_blk_size = self.ctx_->layer_blk_sizes[tensor_cnt];
      const uint64_t dst_layer_blk_size = self.dst_layer_blk_sizes_[tensor_cnt];
      assert(len > 0);
      assert(src_offset < layer_blk_size);
      assert(len < layer_blk_size);
      assert(src_offset + len <= layer_blk_size);
      assert(dst_offset < dst_layer_blk_size);
      assert(dst_offset + len <= dst_layer_blk_size);

      auto *rladdr = reinterpret_cast<char *>(dst_layer_handle.ptrs[tensor_cnt]);
      const auto &src_mr_guard = temp_src_mrs[tensor_cnt];
      const auto &src_mr = src_mr_guard.mr();

      auto rwmemp = rw_memp_t();
      rwmemp.r_addr = reinterpret_cast<uint64_t>(rladdr + dst_offset);
      rwmemp.r_key = dst_layer_handle.rkeys[tensor_cnt];
      rwmemp.sg.addr = uint64_t(src_mr.buf) + src_offset;
      rwmemp.sg.length = len;
      rwmemp.sg.lkey = src_mr.mr->lkey;

#if 0
      LOG(INFO) << "Send data: layer_idx=" << layer_idx << ", tensor_cnt=" << tensor_cnt
      << ", src_offset=" << src_offset << ", dst_offset=" << dst_offset << ", len=" << len
      << ", layer_blk_size=" << layer_blk_size << ", dst_layer_blk_size=" << dst_layer_blk_size
      << ", r_addr=" << reinterpret_cast<uint64_t>(rladdr) << ", sg.addr=" << uint64_t(src_mr.buf);
#endif
      datasp->emplace_back(std::move(rwmemp));
      if (datasp->size() >= self.dataperch_) {
        auto fut = WriteBatch(self.ch(), std::move(datasp));
        self.write_futs_.emplace_back(std::move(fut));
        datasp = std::make_shared<std::vector<rw_memp_t>>();
        datasp->reserve(self.dataperch_);
      }
      if (self.enable_crc_) {
        assert(tensor_cpu_ptr);
        auto *src_addr = tensor_cpu_ptr + src_offset;
        self.crc_ = crc32_z(self.crc_, src_addr, len);
      }
    }
    tensor_cnt++;
  }
  if (!datasp->empty()) {
    auto fut = WriteBatch(self.ch(), std::move(datasp));
    self.write_futs_.emplace_back(std::move(fut));
  }
  return;
}

// return send_us
[[nodiscard]] static std::future<uint64_t> WriteBySgList(
  XChannel *ch,
  uint64_t remote_addr,
  uint32_t rkey,
  std::shared_ptr<std::vector<memp_t>> prefills
) {
  auto pr = std::make_shared<std::promise<uint64_t>>();
  auto fut = pr->get_future();
  auto &datas = *prefills;

  const auto start_ts = SteadyClock::now();
  auto result = ch->WriteBySgList(
      datas, remote_addr, rkey,
      /* signal_peer */ false,
      /* imm_data */ 0,
      [prefills = std::move(prefills), pr = std::move(pr), start_ts](Status s) {
        // WriteBySgList requires prefills to remain valid until the callback.
        if (!s.IsOk()) {
          auto ex = std::make_exception_ptr(
              std::runtime_error("Write ERR: " + s.ErrMsg())
            );
          pr->set_exception(std::move(ex));
          return;
        }
        auto send_us = elapse_us(start_ts, SteadyClock::now());
        pr->set_value(send_us);
      });

  if (result != BAREX_SUCCESS) {
    auto ex =
        std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<uint64_t>(std::move(ex));
  }

  return fut;
}

static void when_all_succeed(std::vector<std::future<void>> &futs) {
  for (auto &fut : futs) {
    fut.get();
  }
  return;
}

void RDMAChannel::flush(std::string &outstr) {
  auto &self = *this;

  const auto inflyn = self.write_futs_.size();
  auto out = std::ostringstream();
  out << std::fixed << std::setprecision(3)
      << "OriginSbNum=" << self.origin_sb_num_
      << ",MergedSbNum=" << self.merged_sb_num_
      << ",SbSizeMin=" << self.sb_size_min_
      << ",SbSizeMax=" << self.sb_size_max_
      << ",SbSizeTotal=" << self.sb_size_total_ << ",CRC32=" << self.crc_
      << ",InflyWrite=" << inflyn;

  uint64_t send_us_min = UINT64_MAX;
  uint64_t send_us_max = 0;
  uint64_t send_us_total = 0;
  for (auto &fut : self.write_futs_) {
    uint64_t send_us = 0;
    try {
      send_us = fut.get();
    } catch (...) {
      outstr = std::move(out).str();
      throw;
    }

    send_us_min = std::min(send_us_min, send_us);
    send_us_max = std::max(send_us_max, send_us);
    send_us_total += send_us;
  }
  self.write_futs_.clear();

  out << ",SendUsMin=" << send_us_min << ",SendUsMax=" << send_us_max
      << ",SendUsAvg=" << send_us_total / float(inflyn);

  if (self.enable_crc_) {
    auto crcrt = TimeWatch();
    assert(!self.chs_.empty());
    auto rcrc = get_remote_crc(self.ctx_, self.chs_[0].sch(), self.data_, self.crc_);
    auto crc_dus_us = crcrt.get_elapse_us();
    out << ",CrcDurUs=" << crc_dus_us;
    RTASSERT_EQ(rcrc, self.crc_);
  }

  self.data_ = nullptr;
  outstr = std::move(out).str();
  return;
}

}  // namespace blade_llm

#endif // ENABLE_RDMA
