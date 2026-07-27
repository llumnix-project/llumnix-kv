
#include <future>
#include <sstream>
#include "protocol/tcp_channel.h"
#include "thrid_party/logging.h"
#include "assert.h"
#include "envcfg.h"
#include "utils/timer.h"
#include "utils/id_generator.h"
#include <iomanip>
#include <numeric>
#include <functional>
#include "fault_inject.h"
#include "copy_kernels.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef ENABLE_RDMA
// just in this cpp
using namespace accl::barex;
#endif

namespace blade_llm {

// -----------------------------------------------------------------------------
// TCP Server
// -----------------------------------------------------------------------------

cudaStream_t TCPServer::get_h2d_stream() {
  return get_thread_local_h2d_stream();
}

void TCPServer::start_server(Context *ctx) {
  auto &self = *this;
  TCPInfo& info = self.info_;
  auto tcp_ctx = BarexProtoContext::server_context(
    "KVTServer",
    std::make_unique<CtxCallback>(this),
    TransferProtocol::Kind::TCP
  );
  if (!tcp_ctx->check_support()) {
    throw std::runtime_error("can't start TCP transfer server as TCP protocol not support;");
  }
  ctx->register_protocol(std::move(tcp_ctx));

  WorkerInfo *winfo = ctx->worker_info_mutable();
  auto layer_num_blocks = ctx->layer_num_blocks();
  auto layer_ptr = ctx->layer_data_address();
  for (const size_t block_size : ctx->block_sizes()) {
    info.layer_blk_sizes.emplace_back(block_size * layer_num_blocks);
  }

  auto proto = TransferProtocol::tcp();
  auto proto_ctx = ctx->get_protocol_ctx<BarexProtoContext>(proto);
  if (proto_ctx == nullptr) {
    throw std::runtime_error("KVT server: tcp context not register.");
  }
  auto barex_ctx = proto_ctx->barex_ctx();
  assert(barex_ctx != nullptr);
  self.ctx_ = barex_ctx;
  self.num_layers_ = winfo->num_layers;


  // per-layer, per-tensor GPU cache base pointers (supports multi-tensor layers)
  info.ptrs.reserve(layer_ptr.size());
  for (size_t layer_idx = 0; layer_idx < layer_ptr.size(); ++layer_idx) {
    RTASSERT(layer_ptr[layer_idx].size() <= MAX_CACHE_NUM_PER_LAYER);
    std::vector<void *> tensor_ptrs;
    tensor_ptrs.reserve(layer_ptr[layer_idx].size());
    for (size_t cache_idx = 0; cache_idx < layer_ptr[layer_idx].size();
        cache_idx++) {
      auto layer_blk_p = reinterpret_cast<void *>(layer_ptr[layer_idx][cache_idx]);
      tensor_ptrs.emplace_back(layer_blk_p);
    }
    info.ptrs.emplace_back(std::move(tensor_ptrs));
  }
  RTASSERT_EQ(info.ptrs.size(), layer_ptr.size());

  get_ip(info.ip, INET_ADDRSTRLEN);
  info.port = env_port_base() + winfo->worker_id;
  RTCHECK(info.port > 0);

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
  LOG(INFO) << "TCPServer.start_server: ip=" << info.ip
            << " port=" << info.port
            << " layer_num_blocks=" << layer_num_blocks;
  auto result = XListener::NewInstance(listener, env_conn_tpsize(), info.port, 
                                       TIMER_3S, {xctx});
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  self.listener_.reset(listener);
  result = self.listener_->Listen();
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  // Initialize thread pool for async CUDA stream synchronization
  // size_t sync_thread_pool_size = static_cast<size_t>(env_h2d_sync_tpsize());
  // result = accl::barex::XThreadpool::NewInstance(self.sync_thread_pool_, sync_thread_pool_size, "tcpsync");
  // RTASSERT(result == accl::barex::BAREX_SUCCESS);
  // LOG(INFO) << "TCPServer: initialized sync thread pool with " << sync_thread_pool_size << " threads";

  // Pre-warm thread-local kernel copy buffers for all threads in the barex thread pool
  auto* threadpool = barex_ctx->tp();
  int device_id_val = barex_ctx->device_id();
  int thread_pool_size = env_ctx_tpsize();
  for (int i = 0; i < thread_pool_size; ++i) {
    threadpool->Submit([device_id_val]() {
      get_kernel_copy_buffer(device_id_val);
    }, i);
  }
  LOG(INFO) << "TCPServer: submitted kernel copy buffer initialization tasks for "
            << thread_pool_size << " thread pool threads";
}

void TCPServer::CtxCallback::OnRecvCall(std::shared_ptr<XChannel> ch, char *in_buf, size_t len, x_msg_header _header) noexcept {
  auto recv_start = _header.c_start;
  auto recv_time = _header.s_on_recv;
  auto onrecv_queue_us = _header.s_on_recv_call - recv_time;

  // Check if this is KV cache data
  if (len < sizeof(uint32_t)) {
    LOG(ERROR) << "TCP OnRecvCall: message too short, len=" << len;
    return;
  }
  uint32_t magic;
  memcpy(&magic, in_buf, sizeof(uint32_t));
  
  if (magic == KV_CACHE_DATA_MAGIC) {
    return this->handle_kv_cache_data(ch, in_buf, len, recv_start, recv_time, onrecv_queue_us);
  }
  return;
}

void TCPServer::CtxCallback::handle_kv_cache_data(
  std::shared_ptr<XChannel>& channel, char *in_buf, size_t len,
  uint64_t recv_start,
  uint32_t recv_time,
  uint32_t onrecv_queue_us) {
  const auto h2d_start_ts = std::chrono::system_clock::now(); // t4
  
  auto& self = *this;
  // Parse header: [magic (uint32_t)] [reqid (uint64_t)] [layer_idx (size_t)] [num_tensors (size_t)]
  //   then per tensor: [metadata_size (size_t)] [IpcBlock array] [tensor_data_size (size_t)] [tensor data]
  if (len < RPC_HEADER + sizeof(size_t) + sizeof(size_t)) {
    LOG(ERROR) << "TCP handle_kv_cache_data: message too short, len=" << len;
    return;
  }
  char* buf_ptr = in_buf;
  char* const buf_end = in_buf + len;
  // Parse RPC header: magic + reqid
  auto [magic, reqid] = deser_rpc_header(buf_ptr);
  buf_ptr += RPC_HEADER;
  assert(magic == KV_CACHE_DATA_MAGIC);

  size_t layer_idx;
  memcpy(&layer_idx, buf_ptr, sizeof(size_t));
  buf_ptr += sizeof(size_t);

  size_t num_tensors;
  memcpy(&num_tensors, buf_ptr, sizeof(size_t));
  buf_ptr += sizeof(size_t);

  auto& ptrs = self.server_->info_.ptrs;
  if (layer_idx >= ptrs.size()) {
    LOG(ERROR) << "TCP handle_kv_cache_data: invalid layer_idx=" << layer_idx << " max=" << ptrs.size();
    return;
  }
  if (num_tensors != ptrs[layer_idx].size()) {
    LOG(ERROR) << "TCP handle_kv_cache_data: num_tensors mismatch, got=" << num_tensors
               << " expected=" << ptrs[layer_idx].size() << " layer_idx=" << layer_idx;
    return;
  }

  // Get thread-local preallocated buffer for kernel metadata
  int device_id = self.server_->ctx_->device_id();
  auto [device_blk_buffer, host_blk_buffer] = get_kernel_copy_buffer(device_id);
  int64_t* device_blk_buffer_ptr = reinterpret_cast<int64_t*>(device_blk_buffer);
  int64_t* host_blk_buffer_ptr = reinterpret_cast<int64_t*>(host_blk_buffer);

  // Use thread-local stream to allow concurrent H2D copies from different threads
  cudaStream_t h2d_stream = TCPServer::get_h2d_stream();

  for (size_t t = 0; t < num_tensors; ++t) {
    if (buf_ptr + sizeof(size_t) > buf_end) {
      LOG(ERROR) << "TCP handle_kv_cache_data: truncated at tensor=" << t << " metadata_size";
      return;
    }
    size_t metadata_size;
    memcpy(&metadata_size, buf_ptr, sizeof(size_t));
    buf_ptr += sizeof(size_t);

    size_t metadata_bytes = metadata_size * sizeof(IpcBlock);
    if (buf_ptr + metadata_bytes + sizeof(size_t) > buf_end) {
      LOG(ERROR) << "TCP handle_kv_cache_data: incomplete metadata at tensor=" << t;
      return;
    }
    std::vector<IpcBlock> blocks;
    blocks.reserve(metadata_size);
    for (size_t i = 0; i < metadata_size; ++i) {
      size_t src_offset, dst_offset, length;
      memcpy(&src_offset, buf_ptr + i * sizeof(IpcBlock), sizeof(size_t));
      memcpy(&dst_offset, buf_ptr + i * sizeof(IpcBlock) + sizeof(size_t), sizeof(size_t));
      memcpy(&length, buf_ptr + i * sizeof(IpcBlock) + 2 * sizeof(size_t), sizeof(size_t));
      blocks.emplace_back(src_offset, dst_offset, length);
    }
    buf_ptr += metadata_bytes;

    size_t tensor_data_size;
    memcpy(&tensor_data_size, buf_ptr, sizeof(size_t));
    buf_ptr += sizeof(size_t);
    if (buf_ptr + tensor_data_size > buf_end) {
      LOG(ERROR) << "TCP handle_kv_cache_data: truncated tensor data, tensor=" << t
                 << " size=" << tensor_data_size;
      return;
    }
    char* tensor_buf = buf_ptr;
    buf_ptr += tensor_data_size;

    void* tensor_gpu_ptr = ptrs[layer_idx][t];
    assert(tensor_gpu_ptr != nullptr);

    const auto& layer_capacities = self.server_->info_.layer_blk_sizes;
    if (t >= layer_capacities.size()) {
      LOG(ERROR) << "TCP H2D missing registered capacity"
                 << ",layer_idx=" << layer_idx
                 << ",tensor_idx=" << t
                 << ",capacity_count=" << layer_capacities.size();
      return;
    }
    try {
      validate_ipc_block_bounds(
          blocks, layer_capacities[t], IpcBlockOffset::DESTINATION,
          "tcp_h2d", layer_idx, t);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "TCP H2D rejected invalid copy metadata"
                 << ",reqid=" << reqid << ",ex=" << ex.what();
      return;
    }

    cudaError_t cuda_rt = copy_handle_data_with_kernel(
      tensor_buf, tensor_gpu_ptr,
      blocks,
      tensor_data_size,
      CopyDirection::H2D,
      device_id,
      h2d_stream,
      device_blk_buffer_ptr,
      host_blk_buffer_ptr
    );
    RTASSERT(cuda_rt == cudaSuccess);
    // host_blk_buffer_ptr is thread-local scratch reused by the next tensor;
    // sync so this tensor's async metadata H2D finishes before it is overwritten.
    RTASSERT(cudaStreamSynchronize(h2d_stream) == cudaSuccess);
  }

  // TODO: add resp code to check success
  // Send response after copy operation completes
  auto buffer_size = RPC_HEADER 
                    + sizeof(h2d_start_ts) * 2 // h2d_start_ts and h2d_end_ts
                    + sizeof(recv_start) + sizeof(recv_time) 
                    + sizeof(onrecv_queue_us);

  memp_t resp_buf = AllocCPUBuffer(channel, buffer_size);
  auto resp_buf_ptr = resp_buf.buf;
  ser_rpc_header(resp_buf_ptr, magic, reqid);
  resp_buf_ptr += RPC_HEADER;
  memcpy(resp_buf_ptr, &recv_start, sizeof(recv_start));
  resp_buf_ptr += sizeof(recv_start);
  memcpy(resp_buf_ptr, &recv_time, sizeof(recv_time));
  resp_buf_ptr += sizeof(recv_time);
  memcpy(resp_buf_ptr, &onrecv_queue_us, sizeof(onrecv_queue_us));
  resp_buf_ptr += sizeof(onrecv_queue_us);
  memcpy(resp_buf_ptr, &h2d_start_ts, sizeof(h2d_start_ts));
  resp_buf_ptr += sizeof(h2d_start_ts);

  // Log before synchronization for debugging
  // Check stream status before synchronization
  cudaError_t stream_query_err = cudaStreamQuery(h2d_stream);
  if (stream_query_err != cudaSuccess && stream_query_err != cudaErrorNotReady) {
    LOG(ERROR) << "TCP handle_kv_cache_data: cudaStreamQuery failed before sync, "
               << "reqid=" << reqid
               << ", error=" << cudaGetErrorString(stream_query_err)
               << " (" << stream_query_err << ")";
  }
  
  // Check for any pending CUDA errors before synchronization
  cudaError_t pending_err = cudaGetLastError();
  if (pending_err != cudaSuccess) {
    LOG(ERROR) << "TCP handle_kv_cache_data: pending CUDA error before sync, "
               << "reqid=" << reqid
               << ", error=" << cudaGetErrorString(pending_err)
               << " (" << pending_err << ")";
  }

  auto cuda_rt_sync = cudaStreamSynchronize(h2d_stream);
  if (cuda_rt_sync != cudaSuccess) {
    LOG(ERROR) << "TCP handle_kv_cache_data: cudaStreamSynchronize failed, "
               << "reqid=" << reqid
               << ", layer_idx=" << layer_idx
               << ", device_id=" << device_id
               << ", h2d_stream=" << reinterpret_cast<void*>(h2d_stream)
               << ", num_tensors=" << num_tensors
               << ", error=" << cudaGetErrorString(cuda_rt_sync)
               << " (" << cuda_rt_sync << ")";
  }
  RTASSERT(cuda_rt_sync == cudaSuccess);
  const auto h2d_end_ts = std::chrono::system_clock::now(); // t5
  memcpy(resp_buf_ptr, &h2d_end_ts, sizeof(h2d_end_ts));

  Send(channel, std::move(resp_buf), [] (Status s) {
    if (s.IsOk()) {
      return;
    }
    LOG(ERROR) << "TCP handle_kv_cache_data: send response err=" << s.ErrMsg();
  });
  // });
  return;
}

// -----------------------------------------------------------------------------
// TCP Channel
// -----------------------------------------------------------------------------

TCPChannel::~TCPChannel() {
  auto* mp = this->ctx_->mp();
  for (auto& buf : this->host_buffers_) {
    if (buf.buf != nullptr) {
      mp->ReleaseBuffer(buf.buf, buf.d_type);
    }
  }
  this->host_buffers_.clear();
  delete_channels(this->ctx_, std::move(this->chs_));
  assert(this->chs_.empty());
  return;
}

cudaStream_t TCPChannel::get_d2h_stream() {
  return get_thread_local_d2h_stream();
}

void TCPChannel::register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) {
  auto& self = *this;

  assert(!data.empty());

  assert(self.data_ == nullptr);
  self.data_ = &data;
  self.kind_ = kind;

  self.do_init();

#ifndef NDEBUG
  size_t total_len_debug = 0;
  for (const auto &tensor_data : data) {
    for (const auto &item : tensor_data) {
      total_len_debug += item.length;
    }
  }
#endif
  // TODO: TCP CRC check
  self.origin_sb_num_ = 0;
  self.merged_sb_num_ = 0;
  self.sb_size_min_ = std::numeric_limits<size_t>::max();
  self.sb_size_max_ = 0;
  self.sb_size_total_ = 0;

  std::vector<size_t> tensor_sizes;

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
      tensor_sizes.emplace_back(total);
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
    tensor_sizes.emplace_back(self.sb_size_total_);
  }
  assert(self.merged_sb_num_ > 0);
  assert(self.merged_sb_num_ <= self.origin_sb_num_);
  // both fp8 length
  assert(total_len_debug == self.sb_size_total_);

  // per-tensor wire blob bytes (Σ block.length). For fp8 these are the
  // fp8-length sums produced by parse_block.
  self.tensor_send_bytes_ = tensor_sizes;

  // fp8 conversion is only supported for single-tensor layers; disable it for
  // multi-tensor cache shapes (e.g. DPSK_V32) and send the raw dtype instead.
  const bool multi_tensor = data.size() > 1;
  if (multi_tensor && self.cast2fp8_) {
    LOG(WARNING) << "TCPChannel: cast2fp8 not supported with multi-tensor layer (num_tensors="
                 << data.size() << "), sending raw dtype";
  }
  const bool use_fp8 = self.cast2fp8_ && !multi_tensor;

  if (use_fp8) {
    // block's length generated by parse_block is fp8 dtype length
    // multiply by 2 to get the actual (bf16 source) length
    self.sb_size_total_ *= 2;
  }

  // Allocate CPU host buffer per layer. Multi-tensor wire layout:
  //   [RPC_HEADER][layer_idx][num_tensors]
  //   per tensor: [metadata_size][IpcBlock array][tensor_data_size][blob]
  if (self.host_buffers_.size() == 0 && self.sb_size_total_ > 0) {
    self.host_buffers_.reserve(self.dst_layer_num_); // prepare for each layer previously
    const size_t header_bytes =
        RPC_HEADER + sizeof(size_t) /*layer_idx*/ + sizeof(size_t) /*num_tensors*/;
    size_t body_bytes = 0;
    for (size_t t = 0; t < data.size(); ++t) {
      body_bytes += sizeof(size_t)                     // metadata_size
                  + data[t].size() * sizeof(IpcBlock)  // IpcBlock array
                  + sizeof(size_t)                     // tensor_data_size
                  + self.tensor_send_bytes_[t];        // blob
    }
    size_t total_bytes = header_bytes + body_bytes;

    uint64_t alloc_us_min = UINT64_MAX, alloc_us_max = 0, alloc_us_total = 0;

    for (size_t i = 0; i < self.dst_layer_num_; ++i) {
      auto alloc_start = std::chrono::system_clock::now();
      memp_t buffer_mr = AllocCudaHostBuffer(self.ch(), total_bytes);
      auto alloc_end = std::chrono::system_clock::now();
      auto alloc_elapsed_us = elapse_us_system(alloc_start, alloc_end);
      RTCHECK(buffer_mr.buf != nullptr);
      RTCHECK(buffer_mr.buf_len >= total_bytes);
      self.host_buffers_.emplace_back(std::move(buffer_mr));
      alloc_us_min = std::min(alloc_us_min, alloc_elapsed_us);
      alloc_us_max = std::max(alloc_us_max, alloc_elapsed_us);
      alloc_us_total += alloc_elapsed_us;
    }
    LOG(INFO) << "TCPChannel::do_init: alloc cuda host buffer total_bytes=" << total_bytes
              << " time min=" << alloc_us_min
              << " max=" << alloc_us_max
              << " total=" << alloc_us_total
              << " avg=" << alloc_us_total / self.dst_layer_num_ << " us"
              << " dst_layer_num=" << self.dst_layer_num_
              << " num_tensors=" << data.size()
              << " header_bytes=" << header_bytes
              << " body_bytes=" << body_bytes;
  }
  assert(self.host_buffers_.size() == self.dst_layer_num_);
  return ;
}

void TCPChannel::send_data(size_t layer_idx) {
  auto &self = *this;
  assert(layer_idx < self.dst_layer_num_);
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());

  assert(self.data_ != nullptr);
  const auto& data = *self.data_;
  const size_t num_tensors = data.size();
  assert(num_tensors > 0);

  const auto send_data_start_ts = std::chrono::system_clock::now();  // t1

  // Wire format: [magic (uint32_t)] [reqid (uint64_t)] [layer_idx (size_t)] [num_tensors (size_t)]
  //   per tensor: [metadata_size (size_t)] [IpcBlock array] [tensor_data_size (size_t)] [tensor data]
  const uint32_t magic = KV_CACHE_DATA_MAGIC;
  const uint64_t reqid = new_id();
  const bool use_fp8 = self.cast2fp8_ && (num_tensors == 1);

  const auto& src_mrs = self.ctx_->layer_mrs()[layer_idx];
  assert(src_mrs.size() == num_tensors);

  const int device_id = self.ctx_->device_id();
  // Get thread-local preallocated buffer for kernel metadata (reused per tensor)
  auto [device_blk_buffer, host_blk_buffer] = get_kernel_copy_buffer(device_id);
  int64_t* device_blk_buffer_ptr = reinterpret_cast<int64_t*>(device_blk_buffer);
  int64_t* host_blk_buffer_ptr = reinterpret_cast<int64_t*>(host_blk_buffer);
  cudaStream_t d2h_stream = TCPChannel::get_d2h_stream();

  char* const base = self.host_buffers_.back().buf;
  char* p = base;
  // header: magic + reqid + layer_idx + num_tensors
  ser_rpc_header(p, magic, reqid);
  p += RPC_HEADER;
  memcpy(p, &layer_idx, sizeof(size_t));
  p += sizeof(size_t);
  memcpy(p, &num_tensors, sizeof(size_t));
  p += sizeof(size_t);

  const auto d2h_start_ts = std::chrono::system_clock::now();   // t2

  for (size_t t = 0; t < num_tensors; ++t) {
    const auto& tensor_data = data[t];
    assert(!tensor_data.empty());
    const size_t metadata_size = tensor_data.size();
    const size_t sent_bytes = self.tensor_send_bytes_[t];           // wire blob bytes (fp8 length when fp8)
    const size_t d2h_total = use_fp8 ? sent_bytes * 2 : sent_bytes; // copy-kernel total (bf16 source for fp8)

    // metadata_size
    memcpy(p, &metadata_size, sizeof(size_t));
    p += sizeof(size_t);
    // IpcBlock array (original blocks; server uses dst_offset for H2D scatter)
    memcpy(p, tensor_data.data(), metadata_size * sizeof(IpcBlock));
    p += metadata_size * sizeof(IpcBlock);
    // tensor_data_size
    memcpy(p, &sent_bytes, sizeof(size_t));
    p += sizeof(size_t);
    // tensor blob destination
    char* tensor_buf_ptr = p;
    p += sent_bytes;

    // kernel blocks: dst_offset = GPU source offset; length = GPU-side (bf16 for fp8) length
    std::vector<IpcBlock> kernel_blocks;
    kernel_blocks.reserve(metadata_size);
    for (const auto& block : tensor_data) {
      kernel_blocks.emplace_back(0, block.src_offset, block.length * (use_fp8 ? 2 : 1));
    }
    void* gpu_src_ptr = reinterpret_cast<void*>(const_cast<char*>(src_mrs[t].mr().buf));

    cudaError_t cuda_rt;
    if (use_fp8) {
      cuda_rt = copy_d2h_bf16_to_fp8(
        tensor_buf_ptr, gpu_src_ptr, kernel_blocks, d2h_total,
        device_id, d2h_stream, device_blk_buffer_ptr, host_blk_buffer_ptr);
    } else {
      cuda_rt = copy_handle_data_with_kernel(
        tensor_buf_ptr, gpu_src_ptr, kernel_blocks, d2h_total,
        CopyDirection::D2H, device_id, d2h_stream,
        device_blk_buffer_ptr, host_blk_buffer_ptr);
    }
    RTCHECK(cuda_rt == cudaSuccess);
    // host_blk_buffer_ptr is thread-local scratch reused by the next tensor;
    // sync so this tensor's async metadata H2D finishes before it is overwritten.
    RTCHECK(cudaStreamSynchronize(d2h_stream) == cudaSuccess);
  }

  const size_t total_send_size = static_cast<size_t>(p - base);
  assert(self.host_buffers_.size() == self.dst_layer_num_ - layer_idx);
  assert(self.host_buffers_.back().buf_len >= total_send_size);
  const auto d2h_end_ts = std::chrono::system_clock::now(); // t3

  self.host_buffers_.back().buf_len = total_send_size;

  auto pr = std::make_shared<SendKVCacheData::Promise>();
  auto time_fut = pr->get_future();

  self.ctx_->push(reqid, SendKVCacheData(
    reqid, pr, send_data_start_ts, d2h_start_ts, d2h_end_ts
  ));

  memp_t send_buf = std::move(self.host_buffers_.back());
  self.host_buffers_.pop_back();

  // After send, send_buf(host_buffer_) will be released.
  Send(self.sch(), std::move(send_buf), [&self, reqid](Status s) mutable {
    if (!s.IsOk()) {
      self.ctx_->on_send_error(std::move(s), reqid);
      return;
    }
  });
  self.write_futs_.emplace_back(reqid, std::move(time_fut));
  return;
}

accl::barex::XChannel *TCPChannel::ch() noexcept {
  return this->sch().get();
}

std::shared_ptr<accl::barex::XChannel>& TCPChannel::sch() noexcept {
  auto &self = *this;
  int n = self.chs_.size();
  int idx = (++self.prev_ch_idx_) % n;
  return self.chs_[idx].sch();
}

void TCPChannel::connect(const WorkerInfo &dst_info) {
  auto &self = *this;
  std::string addr(dst_info.addr);
  auto pos = addr.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid tcp address: " + addr);
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

bool TCPChannel::is_active() {
  auto& self = *this;
  if (self.chs_.empty()) {
    return true;
  }
  return valid_channels(self.chs_);
}

void TCPChannel::do_init() {
  auto &self = *this;
  if (valid_channels(self.chs_)) {
    assert(self.ctx_->layer_mrs().size() == self.dst_layer_num_);
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
  LOG(INFO) << "TCPChannel connect: done: delayms=" << delay_ms;
  usleep(delay_ms * 1000);
#endif

  LOG(INFO) << "TCPChannel connect. dstip=" << self.ip_
            << ",init_us=" << init_time.get_elapse_us()
            << ",dstport=" << self.port_;
  assert(valid_channels(self.chs_));
  return;
}

void TCPChannel::flush(std::string& outstr) {
  auto &self = *this;
  self.data_ = nullptr;
  const auto inflyn = self.write_futs_.size();
  auto out = std::ostringstream();
  out << std::fixed << std::setprecision(3)
      << "OriginSbNum=" << self.origin_sb_num_
      << ",MergedSbNum=" << self.merged_sb_num_
      << ",SbSizeMin=" << self.sb_size_min_
      << ",SbSizeMax=" << self.sb_size_max_
      << ",SbSizeTotal=" << self.sb_size_total_
      << ",InflyWrite=" << inflyn;
  uint64_t send_us_min = UINT64_MAX, d2h_us_min = UINT64_MAX, h2d_us_min = UINT64_MAX, trans_us_min = UINT64_MAX;
  uint64_t send_us_max = 0, d2h_us_max = 0, h2d_us_max = 0, trans_us_max = 0;
  uint64_t send_us_total = 0, d2h_us_total = 0, h2d_us_total = 0, trans_us_total = 0;
  uint64_t link_tx_us_min = UINT64_MAX, link_tx_us_max = 0, link_tx_us_total = 0;
  uint64_t recv_us_min = UINT64_MAX, recv_us_max = 0, recv_us_total = 0;
  uint64_t onrecv_queue_us_min = UINT64_MAX, onrecv_queue_us_max = 0, onrecv_queue_us_total = 0;
  const auto timeout = std::chrono::seconds(env_rpc_timeout_s());
  for (auto& [reqid, fut] : self.write_futs_) {
    TCPTimePoints time_points = {};
    try {
      auto futstate = fut.wait_for(timeout);
      if (futstate != std::future_status::ready) {
        LOG(ERROR) << "Flush timeout! reqid=" << reqid;
        self.ctx_->on_send_error(Status(BAREX_ERR_TIMEOUT), reqid);
      }
      time_points = fut.get();
    } catch (...) {
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
  self.write_futs_.clear();
  // After send done, host_buffer_ mempt's buffer should be released by Send callback.
  // And host_buffers_ should be empty.
  assert(self.host_buffers_.size() == 0);
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
  outstr = std::move(out).str();
  return;
}

} // namespace blade_llm
