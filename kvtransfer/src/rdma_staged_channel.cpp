
#include <future>
#include <sstream>
#include "protocol/rdma_staged_channel.h"
#include "copy_utils.h"
#include "copy_kernels.h"
#include "thrid_party/logging.h"
#include "assert.h"
#include "envcfg.h"
#include "utils/timer.h"
#include "utils/id_generator.h"
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime.h>

#ifdef ENABLE_RDMA
using namespace accl::barex;

namespace blade_llm {

struct StagedPreallocResp {
  std::vector<StagedPreallocRespEntry> entries;
};

struct GetStagedPreallocResp {
 private:
  using Promise = std::promise<StagedPreallocResp>;
  std::shared_ptr<XChannel> const ch_ = nullptr;
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;

 public:
  GetStagedPreallocResp(std::shared_ptr<XChannel> ch, uint64_t reqid) noexcept:
      ch_(std::move(ch)),
      reqid_(reqid),
      pr_(std::make_shared<Promise>()) {}

  auto future() {
    return this->pr_->get_future();
  }

  void operator()(Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      auto ex = std::make_exception_ptr(
          std::runtime_error("GetStagedPreallocResp ERR: " + s.ErrMsg()));
      self.pr_->set_exception(std::move(ex));
      return;
    }
    RTASSERT(len >= RPC_HEADER + sizeof(uint32_t));
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == STAGED_PREALLOC_RESP_MAGIC);
    RTASSERT(reqid == self.reqid_);

    buf += RPC_HEADER;
    len -= RPC_HEADER;
    uint32_t layer_num = 0;
    memcpy(&layer_num, buf, sizeof(layer_num));
    buf += sizeof(layer_num);
    len -= sizeof(uint32_t);

    RTASSERT(len == size_t(layer_num) * sizeof(StagedPreallocRespEntry));
    auto resp = StagedPreallocResp();
    resp.entries.resize(layer_num);
    memcpy(resp.entries.data(), buf, len);
    // Corner case: Abort this request if buffer_id is invalid(Out of range).
    for (const auto& entry : resp.entries) {
      if (entry.buffer_id == 0 || entry.buffer_id > STAGED_IMM_BUFFER_MASK) {
        std::stringstream ss;
        ss << "request_staged_prealloc_async: invalid buffer_id=" << entry.buffer_id
           << ", max_buffer_id=" << STAGED_IMM_BUFFER_MASK;
        auto ex = std::make_exception_ptr(std::overflow_error(ss.str()));
        self.pr_->set_exception(std::move(ex));
        return;
      }
    }
    self.pr_->set_value(std::move(resp));
  }
};

[[nodiscard]] static std::pair<uint64_t, std::future<StagedPreallocResp>> request_staged_prealloc_async(
    CliBarexCtx* ctx, std::shared_ptr<XChannel>& ch, const std::vector<uint64_t>& layer_sizes) {
  RTASSERT(!layer_sizes.empty());
  const uint64_t reqid = new_id();
  const size_t body_size = sizeof(uint32_t) + layer_sizes.size() * sizeof(uint64_t);
  auto req_buf = AllocCPUBuffer(ch, RPC_HEADER + body_size);
  ser_rpc_header(req_buf.buf, STAGED_PREALLOC_REQ_MAGIC, reqid);
  auto* ptr = req_buf.buf + RPC_HEADER;
  const uint32_t layer_num = static_cast<uint32_t>(layer_sizes.size());
  memcpy(ptr, &layer_num, sizeof(layer_num));
  ptr += sizeof(layer_num);
  memcpy(ptr, layer_sizes.data(), layer_sizes.size() * sizeof(uint64_t));

  auto rpc = GetStagedPreallocResp(ch, reqid);
  auto fut = rpc.future();
  ctx->push(reqid, std::move(rpc));
  Send(ch, std::move(req_buf), [ctx, reqid](Status s) {
    if (!s.IsOk()) {
      ctx->on_send_error(std::move(s), reqid);
    }
  });
  return {reqid, std::move(fut)};
}

[[nodiscard]] static std::future<void> WriteSingleWithImm(
    std::shared_ptr<XChannel>& ch, memp_t sdata,
    uint64_t raddr, uint32_t rkey, uint32_t imm_data) {
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();
  auto buf_ptr = sdata.buf;
  auto buf_dtype = sdata.d_type;
  auto done = [pr, ch, buf_ptr, buf_dtype](Status s) {
    ch->ReleaseBuffer(buf_ptr, buf_dtype);
    if (!s.IsOk()) {
      auto ex = std::make_exception_ptr(std::runtime_error("WriteSingle ERR: " + s.ErrMsg()));
      pr->set_exception(std::move(ex));
      return;
    }
    pr->set_value();
  };
  auto result = ch->WriteSingle(
      std::move(sdata), raddr, rkey,
      /* signal_peer */ true, imm_data, std::move(done));
  if (result != BAREX_SUCCESS) {
    ch->ReleaseBuffer(buf_ptr, buf_dtype);
    auto ex = std::runtime_error("WriteSingle Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<void>(std::move(ex));
  }
  return fut;
}

RDMAStagedChannel::~RDMAStagedChannel() {
  auto* mp = this->ctx_->mp();
  for (auto& buf : this->host_buffers_) {
    if (buf.buf != nullptr) {
      mp->ReleaseBuffer(buf.buf, buf.d_type);
    }
  }
  this->host_buffers_.clear();
  this->remote_buffers_.clear();
  this->staged_payload_size_ = 0;
}

void RDMAStagedChannel::register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) {
  auto &self = *this;
  RDMAChannel::register_data(data, kind);

  if (self.host_buffers_.empty() && self.sb_size_total_ > 0) {
    assert(data.size() == 1 && "staged RDMA path currently supports single tensor per layer");
    const auto& tensor_data = data[0];
    size_t metadata_size = tensor_data.size();
    size_t header_bytes = RPC_HEADER + sizeof(uint64_t) + sizeof(size_t);
    size_t metadata_bytes = sizeof(size_t) + metadata_size * sizeof(IpcBlock);
    size_t tensor_bytes = self.sb_size_total_;
    size_t total_bytes = header_bytes + metadata_bytes + tensor_bytes;

    self.host_buffers_.reserve(self.dst_layer_num_);
    for (size_t i = 0; i < self.dst_layer_num_; ++i) {
      memp_t buffer_mr = AllocCudaHostBuffer(self.ch(), total_bytes);
      RTCHECK(buffer_mr.buf != nullptr);
      RTCHECK(buffer_mr.buf_len >= total_bytes);
      self.host_buffers_.emplace_back(std::move(buffer_mr));
    }
    LOG(INFO) << "RDMAStagedChannel::register_data: alloc cuda host buffer"
              << " total_bytes=" << total_bytes
              << " dst_layer_num=" << self.dst_layer_num_
              << " metadata_size=" << metadata_size
              << " header_bytes=" << header_bytes
              << " metadata_bytes=" << metadata_bytes
              << " tensor_bytes=" << tensor_bytes;
  }
  if (self.sb_size_total_ > 0) {
    RTASSERT(!self.host_buffers_.empty());
    RTASSERT(!self.chs_.empty());
    assert(data.size() == 1);
    const size_t md_size = data[0].size();
    const size_t payload = RPC_HEADER + sizeof(uint64_t) + sizeof(size_t)
                         + sizeof(size_t) + md_size * sizeof(IpcBlock)
                         + self.sb_size_total_;

    const bool can_reuse = (self.staged_payload_size_ >= payload
                            && self.remote_buffers_.size() == self.dst_layer_num_);
    if (can_reuse) {
      return;
    }

    self.staged_payload_size_ = payload;
    self.remote_buffers_.clear();
    std::vector<uint64_t> layer_sizes(self.dst_layer_num_, payload);
    auto& fixed_ch = self.chs_[0].sch();
    auto [prealloc_reqid, prealloc_fut] = request_staged_prealloc_async(self.ctx_, fixed_ch, layer_sizes);
    auto futstate = prealloc_fut.wait_for(std::chrono::seconds(env_rpc_timeout_s()));
    if (futstate != std::future_status::ready) {
      self.ctx_->on_send_error(Status(BAREX_ERR_TIMEOUT), prealloc_reqid);
    }
    auto prealloc_resp = prealloc_fut.get();
    RTASSERT(prealloc_resp.entries.size() == self.dst_layer_num_);
    self.remote_buffers_.reserve(self.dst_layer_num_);
    for (const auto& entry : prealloc_resp.entries) {
      RTASSERT(entry.remote_addr != 0);
      RTASSERT(entry.rkey != 0);
      self.remote_buffers_.push_back(RemoteStagedBuffer{
          entry.remote_addr, entry.size, entry.rkey, entry.buffer_id});
    }
    RTASSERT(self.remote_buffers_.size() == self.dst_layer_num_);
  }
}

void RDMAStagedChannel::send_data(size_t layer_idx) {
  auto &self = *this;
  assert(layer_idx < self.dst_layer_num_);
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());
  assert(self.data_ != nullptr);
  assert(self.data_->size() == 1 && "staged RDMA path currently supports single tensor per layer");
  const auto& data = (*self.data_)[0];
  assert(!data.empty());

  const auto send_data_start_ts = std::chrono::system_clock::now();
  RTASSERT(!self.chs_.empty());

  const uint32_t magic = KV_CACHE_DATA_MAGIC;
  const uint64_t reqid = new_id();
  const size_t metadata_size = data.size();
  const size_t header_bytes = RPC_HEADER + sizeof(uint64_t) + sizeof(size_t);
  const size_t metadata_bytes = sizeof(size_t) + metadata_size * sizeof(IpcBlock);
  const size_t tensor_send_size = self.sb_size_total_;
  const size_t total_send_size = header_bytes + metadata_bytes + tensor_send_size;

  assert(self.host_buffers_.size() == self.dst_layer_num_ - layer_idx);
  assert(self.host_buffers_.back().buf_len >= total_send_size);
  assert(self.remote_buffers_.size() == self.dst_layer_num_);
  const auto& remote_buf = self.remote_buffers_[layer_idx];
  RTASSERT(total_send_size <= remote_buf.size);
  RTASSERT(remote_buf.buffer_id <= STAGED_IMM_BUFFER_MASK);

  char* buf_ptr = self.host_buffers_.back().buf;
  char* meta_buf_ptr = buf_ptr + header_bytes;
  char* tensor_buf_ptr = buf_ptr + header_bytes + metadata_bytes;

  const auto& src_mrs = self.ctx_->layer_mrs()[layer_idx];
  const auto& src_mr_guard = src_mrs[0];
  const auto& src_mr_base = src_mr_guard.mr();

  const auto d2h_start_ts = std::chrono::system_clock::now();

  std::vector<IpcBlock> kernel_blocks;
  kernel_blocks.reserve(metadata_size);
  for (const auto& block : data) {
    kernel_blocks.emplace_back(0, block.src_offset, block.length);
  }

  int device_id = self.ctx_->device_id();
  void* gpu_src_ptr = reinterpret_cast<void*>(const_cast<char*>(src_mr_base.buf));

  auto [device_blk_buffer, host_blk_buffer] = get_kernel_copy_buffer(device_id);
  int64_t* device_blk_buffer_ptr = reinterpret_cast<int64_t*>(device_blk_buffer);
  int64_t* host_blk_buffer_ptr = reinterpret_cast<int64_t*>(host_blk_buffer);

  cudaStream_t d2h_stream = get_thread_local_d2h_stream();

  cudaError_t cuda_rt = copy_handle_data_with_kernel(
    tensor_buf_ptr, gpu_src_ptr, kernel_blocks, self.sb_size_total_,
    CopyDirection::D2H, device_id, d2h_stream,
    device_blk_buffer_ptr, host_blk_buffer_ptr);
  RTCHECK(cuda_rt == cudaSuccess);

  ser_rpc_header(buf_ptr, magic, reqid);
  buf_ptr += RPC_HEADER;
  const uint64_t actual_len = total_send_size;
  memcpy(buf_ptr, &actual_len, sizeof(actual_len));
  buf_ptr += sizeof(actual_len);
  memcpy(buf_ptr, &layer_idx, sizeof(size_t));

  memcpy(meta_buf_ptr, &metadata_size, sizeof(size_t));
  meta_buf_ptr += sizeof(size_t);
  memcpy(meta_buf_ptr, data.data(), metadata_size * sizeof(IpcBlock));

  auto cuda_rt_sync = cudaStreamSynchronize(d2h_stream);
  RTCHECK(cuda_rt_sync == cudaSuccess);
  const auto d2h_end_ts = std::chrono::system_clock::now();

  self.host_buffers_.back().buf_len = total_send_size;

  auto pr = std::make_shared<SendKVCacheData::Promise>();
  auto time_fut = pr->get_future();

  self.ctx_->push(reqid, SendKVCacheData(
    reqid, pr, send_data_start_ts, d2h_start_ts, d2h_end_ts
  ));

  memp_t send_buf = std::move(self.host_buffers_.back());
  self.host_buffers_.pop_back();
  uint32_t imm_data = encode_staged_imm_data(remote_buf.buffer_id);
  auto& fixed_ch = self.chs_[0].sch();
  auto submit_fut = WriteSingleWithImm(
      fixed_ch, std::move(send_buf), remote_buf.remote_addr, remote_buf.rkey, imm_data);
  self.staged_submit_futs_.emplace_back(std::move(submit_fut));

  self.staged_write_futs_.emplace_back(reqid, std::move(time_fut));
}

void RDMAStagedChannel::flush(std::string &outstr) {
  auto &self = *this;
  self.data_ = nullptr;
  const auto inflyn = self.staged_write_futs_.size();
  auto out = std::ostringstream();
  out << std::fixed << std::setprecision(3)
      << "OriginSbNum=" << self.origin_sb_num_
      << ",MergedSbNum=" << self.merged_sb_num_
      << ",SbSizeMin=" << self.sb_size_min_
      << ",SbSizeMax=" << self.sb_size_max_
      << ",SbSizeTotal=" << self.sb_size_total_
      << ",Staged=1,InflyWrite=" << inflyn;

  uint64_t send_us_min = UINT64_MAX, d2h_us_min = UINT64_MAX, h2d_us_min = UINT64_MAX, trans_us_min = UINT64_MAX;
  uint64_t send_us_max = 0, d2h_us_max = 0, h2d_us_max = 0, trans_us_max = 0;
  uint64_t send_us_total = 0, d2h_us_total = 0, h2d_us_total = 0, trans_us_total = 0;
  uint64_t link_tx_us_min = UINT64_MAX, link_tx_us_max = 0, link_tx_us_total = 0;
  uint64_t recv_us_min = UINT64_MAX, recv_us_max = 0, recv_us_total = 0;
  uint64_t onrecv_queue_us_min = UINT64_MAX, onrecv_queue_us_max = 0, onrecv_queue_us_total = 0;
  const auto timeout = std::chrono::seconds(env_rpc_timeout_s());

  try {
    for (auto& fut : self.staged_submit_futs_) {
      fut.get();
    }
  } catch (...) {
    self.staged_submit_futs_.clear();
    self.staged_write_futs_.clear();
    self.remote_buffers_.clear();
    self.staged_payload_size_ = 0;
    outstr = std::move(out).str();
    throw;
  }
  self.staged_submit_futs_.clear();

  for (auto& [reqid, fut] : self.staged_write_futs_) {
    StagedTimePoints time_points = {};
    try {
      auto futstate = fut.wait_for(timeout);
      if (futstate != std::future_status::ready) {
        LOG(ERROR) << "RDMAStagedChannel flush timeout! reqid=" << reqid;
        self.ctx_->on_send_error(Status(BAREX_ERR_TIMEOUT), reqid);
      }
      time_points = fut.get();
    } catch (...) {
      self.staged_write_futs_.clear();
      self.remote_buffers_.clear();
      self.staged_payload_size_ = 0;
      outstr = std::move(out).str();
      throw;
    }
    auto send_us = elapse_us_system(time_points.send_data_start_ts_, time_points.h2d_end_ts);
    auto d2h_us = elapse_us_system(time_points.d2h_start_ts_, time_points.d2h_end_ts_);
    auto h2d_us = elapse_us_system(time_points.h2d_start_ts, time_points.h2d_end_ts);
    auto trans_us = elapse_us_system(time_points.d2h_end_ts_, time_points.h2d_start_ts);
    auto link_tx_us = time_points.recv_start_ - time_point_to_microseconds(time_points.d2h_end_ts_);
    uint64_t recv_us = static_cast<uint64_t>(time_points.recv_time_);
    uint64_t onrecv_queue_us = static_cast<uint64_t>(time_points.onrecv_queue_us_);

    send_us_min = std::min(send_us_min, send_us);
    send_us_max = std::max(send_us_max, send_us);
    send_us_total += send_us;
    d2h_us_min = std::min(d2h_us_min, d2h_us);
    d2h_us_max = std::max(d2h_us_max, d2h_us);
    d2h_us_total += d2h_us;
    h2d_us_min = std::min(h2d_us_min, h2d_us);
    h2d_us_max = std::max(h2d_us_max, h2d_us);
    h2d_us_total += h2d_us;
    trans_us_min = std::min(trans_us_min, trans_us);
    trans_us_max = std::max(trans_us_max, trans_us);
    trans_us_total += trans_us;
    link_tx_us_min = std::min(link_tx_us_min, link_tx_us);
    link_tx_us_max = std::max(link_tx_us_max, link_tx_us);
    link_tx_us_total += link_tx_us;
    recv_us_min = std::min(recv_us_min, recv_us);
    recv_us_max = std::max(recv_us_max, recv_us);
    recv_us_total += recv_us;
    onrecv_queue_us_min = std::min(onrecv_queue_us_min, onrecv_queue_us);
    onrecv_queue_us_max = std::max(onrecv_queue_us_max, onrecv_queue_us);
    onrecv_queue_us_total += onrecv_queue_us;
  }
  self.staged_write_futs_.clear();
  assert(self.host_buffers_.empty());

  if (inflyn > 0) {
    out << ",SendUsMin=" << send_us_min
        << ",SendUsMax=" << send_us_max
        << ",SendUsAvg=" << send_us_total / float(inflyn)
        << ",D2HUsMin=" << d2h_us_min
        << ",D2HUsMax=" << d2h_us_max
        << ",D2HUsAvg=" << d2h_us_total / float(inflyn)
        << ",H2DUsMin=" << h2d_us_min
        << ",H2DUsMax=" << h2d_us_max
        << ",H2DUsAvg=" << h2d_us_total / float(inflyn)
        << ",TransUsMin=" << trans_us_min
        << ",TransUsMax=" << trans_us_max
        << ",TransUsAvg=" << trans_us_total / float(inflyn)
        << ",LinkTxUsMin=" << link_tx_us_min
        << ",LinkTxUsMax=" << link_tx_us_max
        << ",LinkTxUsAvg=" << link_tx_us_total / float(inflyn)
        << ",RecvUsMin=" << recv_us_min
        << ",RecvUsMax=" << recv_us_max
        << ",RecvUsAvg=" << recv_us_total / float(inflyn)
        << ",OnRecvQueueUsMin=" << onrecv_queue_us_min
        << ",OnRecvQueueUsMax=" << onrecv_queue_us_max
        << ",OnRecvQueueUsAvg=" << onrecv_queue_us_total / float(inflyn);
  }
  outstr = std::move(out).str();
}

}  // namespace blade_llm

#endif  // ENABLE_RDMA
