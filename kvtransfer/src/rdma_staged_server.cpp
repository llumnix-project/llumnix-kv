

#include <future>
#include <sstream>
#include "protocol/rdma_staged_server.h"
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

namespace {

void release_staged_recv_buffers(
    BarexCtx* ctx,
    uintptr_t channel_key,
    std::unordered_map<uint32_t, RDMAStagedServer::StagedRecvBuffer> buffers,
    const char* reason) noexcept {
  if (ctx == nullptr || buffers.empty()) {
    return;
  }
  auto* mp = ctx->mp();
  size_t released = 0;
  for (auto& [_, buf] : buffers) {
    if (buf.mem.buf == nullptr) {
      continue;
    }
    ++released;
    auto ret = mp->ReleaseAndDeregBuffer(buf.mem.buf, buf.mem.d_type);
    if (ret != BAREX_SUCCESS) {
      LOG(WARNING) << reason << " release failed. channel_key=0x" << std::hex
                   << channel_key << std::dec
                   << " buf=" << buf.mem.buf << " dtype=" << int(buf.mem.d_type)
                   << " ret=" << ret;
    }
  }
  LOG(INFO) << reason << " released " << released
            << " buffers for channel_key=0x" << std::hex << channel_key << std::dec;
}

}  // namespace

// ---------------------------------------------------------------------------
// StagedBufferGuard
// ---------------------------------------------------------------------------

RDMAStagedServer::StagedBufferGuard::~StagedBufferGuard() noexcept {
  std::unordered_map<uint32_t, StagedRecvBuffer> old_bufs;
  {
    auto lock = std::lock_guard<std::mutex>(mtx_);
    old_bufs.swap(buffers_);
  }
  release_staged_recv_buffers(ctx_, channel_key_, std::move(old_bufs), "~StagedBufferGuard");
}

void RDMAStagedServer::StagedBufferGuard::release_all_buffers() noexcept {
  std::unordered_map<uint32_t, StagedRecvBuffer> old_bufs;
  {
    auto lock = std::lock_guard<std::mutex>(mtx_);
    old_bufs.swap(buffers_);
  }
  release_staged_recv_buffers(
      ctx_, channel_key_, std::move(old_bufs), "StagedBufferGuard::release_all_buffers");
}

void RDMAStagedServer::StagedBufferGuard::replace_buffers(
    std::unordered_map<uint32_t, StagedRecvBuffer> new_bufs) {
  std::unordered_map<uint32_t, StagedRecvBuffer> old_bufs;
  {
    auto lock = std::lock_guard<std::mutex>(mtx_);
    old_bufs.swap(buffers_);
    buffers_ = std::move(new_bufs);
  }
  release_staged_recv_buffers(
      ctx_, channel_key_, std::move(old_bufs), "StagedBufferGuard::replace_buffers");
}

std::vector<uint32_t> RDMAStagedServer::StagedBufferGuard::alloc_buffer_ids(uint32_t count) {
  std::vector<uint32_t> buffer_ids;
  if (count == 0) {
    return buffer_ids;
  }

  auto lock = std::lock_guard<std::mutex>(mtx_);
  if (next_buffer_id_ == 0 || next_buffer_id_ > STAGED_IMM_BUFFER_MASK ||
      count - 1 > STAGED_IMM_BUFFER_MASK - next_buffer_id_) {
    std::stringstream ss;
    ss << "staged buffer_id overflow: channel_key=0x" << std::hex << channel_key_ << std::dec
       << " next_buffer_id=" << next_buffer_id_
       << " alloc_count=" << count
       << " max_buffer_id=" << STAGED_IMM_BUFFER_MASK;
    throw std::overflow_error(ss.str());
  }

  buffer_ids.reserve(count);
  for (uint32_t idx = 0; idx < count; ++idx) {
    buffer_ids.push_back(next_buffer_id_++);
  }
  return buffer_ids;
}

bool RDMAStagedServer::StagedBufferGuard::find_buffer(uint32_t buffer_id, StagedRecvBuffer* out) {
  auto lock = std::lock_guard<std::mutex>(mtx_);
  auto it = buffers_.find(buffer_id);
  if (it == buffers_.end()) {
    return false;
  }
  if (out != nullptr) {
    *out = it->second;
  }
  return true;
}

// ---------------------------------------------------------------------------
// RDMAStagedServer::start_server
// ---------------------------------------------------------------------------

void RDMAStagedServer::start_server(Context *ctx) {
  auto &self = *this;
  RDMAInfo& info = self.info_;
  auto rdma_ctx = BarexProtoContext::server_context(
    "KVTServer",
    std::make_unique<StagedCtxCallback>(this),
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
    for (size_t cache_idx = 0; cache_idx < layer_ptr[layer_idx].size();
         cache_idx++) {
      auto &out = layer_mrs[cache_idx].mr();
      auto layer_blk_p = reinterpret_cast<void *>(layer_ptr[layer_idx][cache_idx]);
      handle.ptrs[cache_idx] = layer_blk_p;
      handle.rkeys[cache_idx] = out.mr->rkey;
    }
  }
  RTASSERT_EQ(info.handles.size(), layer_ptr.size());

  self.gpu_ptrs_.reserve(layer_ptr.size());
  for (size_t layer_idx = 0; layer_idx < layer_ptr.size(); ++layer_idx) {
    RTASSERT(!layer_ptr[layer_idx].empty());
    self.gpu_ptrs_.emplace_back(reinterpret_cast<void*>(layer_ptr[layer_idx][0]));
  }
  RTASSERT_EQ(self.gpu_ptrs_.size(), layer_ptr.size());

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
  LOG(INFO) << "RDMAStagedServer.start_server: ip=" << info.ip
            << " port=" << info.port
            << " layer_num_blocks=" << layer_num_blocks
            << " staged=true";

  auto closedHook = [this](XChannel* channel) {
    const uintptr_t channel_key = reinterpret_cast<uintptr_t>(channel);
    std::shared_ptr<StagedBufferGuard> guard;
    {
      auto lock = std::lock_guard<std::mutex>(staged_mtx_);
      auto guard_it = staged_guards_.find(channel_key);
      if (guard_it != staged_guards_.end()) {
        guard = std::move(guard_it->second);
        staged_guards_.erase(guard_it);
      }
    }
    if (guard) {
      guard->release_all_buffers();
      LOG(INFO) << "ClosedHook: released staged buffers for channel_key=0x"
                << std::hex << channel_key << std::dec;
    }
  };
  xctx->SetChannelClosedHook(closedHook);

  auto result = XListener::NewInstance(listener, env_conn_tpsize(), info.port,
                                       TIMER_3S, {xctx});
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  self.listener_.reset(listener);
  result = self.listener_->Listen();
  RTASSERT(result == accl::barex::BAREX_SUCCESS);

  auto* threadpool = barex_ctx->tp();
  int device_id_val = barex_ctx->device_id();
  int thread_pool_size = env_ctx_tpsize();
  for (int i = 0; i < thread_pool_size; ++i) {
    threadpool->Submit([device_id_val]() {
      get_kernel_copy_buffer(device_id_val);
    }, i);
  }
  LOG(INFO) << "RDMAStagedServer: submitted kernel copy buffer initialization for "
            << thread_pool_size << " thread pool threads";
}

// ---------------------------------------------------------------------------
// StagedCtxCallback::OnRecvCall
// ---------------------------------------------------------------------------

void RDMAStagedServer::StagedCtxCallback::OnRecvCall(
    std::shared_ptr<XChannel> ch, char *in_buf, size_t len, x_msg_header _header) noexcept {
  if (len >= RPC_HEADER) {
    const auto [magic, reqid] = deser_rpc_header(in_buf);
    if (magic == STAGED_PREALLOC_REQ_MAGIC) {
      try {
        this->resp_staged_prealloc(ch, reqid, in_buf, len);
      } catch (const std::exception& ex) {
        LOG(ERROR) << "resp staged prealloc: error=" << ex.what()
                   << ";ch=" << ch->ToString() << ";reqid=" << reqid;
      }
      return;
    }
    if (magic == KV_CACHE_DATA_MAGIC) {
      LOG(ERROR) << "unexpected KV_CACHE_DATA_MAGIC on OnRecvCall. "
                 << "staged RDMA data path should arrive via OnImmRecvCall"
                 << ";ch=" << ch->ToString() << ";reqid=" << reqid;
      return;
    }
  }
  CtxCallback::OnRecvCall(std::move(ch), in_buf, len, _header);
}

// ---------------------------------------------------------------------------
// StagedCtxCallback::resp_staged_prealloc
// ---------------------------------------------------------------------------

void RDMAStagedServer::StagedCtxCallback::resp_staged_prealloc(
    std::shared_ptr<XChannel>& channel, uint64_t reqid, char* inbuf, size_t inlen) {
  RTASSERT(inlen >= RPC_HEADER + sizeof(uint32_t));
  const auto& server = *this->staged_server_;
  auto* req_ptr = inbuf + RPC_HEADER;
  uint32_t layer_num = 0;
  memcpy(&layer_num, req_ptr, sizeof(layer_num));
  req_ptr += sizeof(layer_num);
  RTASSERT(layer_num > 0);
  RTASSERT(layer_num <= STAGED_PREALLOC_MAX_LAYERS);
  RTASSERT(inlen == RPC_HEADER + sizeof(layer_num) + size_t(layer_num) * sizeof(uint64_t));

  std::vector<uint64_t> layer_sizes(layer_num);
  memcpy(layer_sizes.data(), req_ptr, layer_num * sizeof(uint64_t));
  for (auto sz : layer_sizes) {
    RTASSERT(sz > 0);
  }

  const uintptr_t channel_key = reinterpret_cast<uintptr_t>(channel.get());
  std::shared_ptr<StagedBufferGuard> guard;
  {
    auto lock = std::lock_guard<std::mutex>(server.staged_mtx_);
    auto guard_it = server.staged_guards_.find(channel_key);
    if (guard_it == server.staged_guards_.end()) {
      guard = std::make_shared<StagedBufferGuard>(channel_key, server.ctx_);
      server.staged_guards_.emplace(channel_key, guard);
    } else {
      guard = guard_it->second;
    }
  }

  auto buffer_ids = guard->alloc_buffer_ids(layer_num);
  std::unordered_map<uint32_t, StagedRecvBuffer> new_bufs;
  std::vector<StagedPreallocRespEntry> resp_entries;
  resp_entries.reserve(layer_num);
  for (uint32_t idx = 0; idx < layer_num; ++idx) {
    auto mem = AllocCudaHostBuffer(channel.get(), layer_sizes[idx]);
    RTASSERT(mem.buf != nullptr);
    RTASSERT(mem.mr != nullptr);
    const uint32_t buffer_id = buffer_ids[idx];
    RTASSERT(buffer_id <= STAGED_IMM_BUFFER_MASK);
    resp_entries.push_back(StagedPreallocRespEntry{
        reinterpret_cast<uint64_t>(mem.buf), layer_sizes[idx],
        mem.mr->rkey, buffer_id});
    new_bufs.emplace(buffer_id, StagedRecvBuffer{
                         std::move(mem), layer_sizes[idx], buffer_id, idx});
  }
  guard->replace_buffers(std::move(new_bufs));

  const size_t resp_body = sizeof(uint32_t) + resp_entries.size() * sizeof(StagedPreallocRespEntry);
  auto resp_buf = AllocCPUBuffer(channel, RPC_HEADER + resp_body);
  auto* resp_ptr = resp_buf.buf;
  ser_rpc_header(resp_ptr, STAGED_PREALLOC_RESP_MAGIC, reqid);
  resp_ptr += RPC_HEADER;
  memcpy(resp_ptr, &layer_num, sizeof(layer_num));
  resp_ptr += sizeof(layer_num);
  memcpy(resp_ptr, resp_entries.data(), resp_entries.size() * sizeof(StagedPreallocRespEntry));
  Send(channel, std::move(resp_buf), [channel, guard, reqid](Status s) {
    if (!s.IsOk()) {
      LOG(ERROR) << "resp_staged_prealloc err=" << s.ErrMsg()
                 << " reqid=" << reqid << " ch=" << channel->ToString();
    }
  });
}

// ---------------------------------------------------------------------------
// StagedCtxCallback::OnImmRecvCall
// ---------------------------------------------------------------------------

void RDMAStagedServer::StagedCtxCallback::OnImmRecvCall(
    std::shared_ptr<XChannel> channel, uint32_t imm_data) noexcept {
  const auto& server = *this->staged_server_;
  const uint32_t buffer_id = decode_staged_imm_data(imm_data);
  if (buffer_id == 0) {
    LOG(ERROR) << "OnImmRecvCall invalid imm_data=" << imm_data;
    return;
  }

  const uintptr_t channel_key = reinterpret_cast<uintptr_t>(channel.get());
  std::shared_ptr<StagedBufferGuard> guard;
  {
    auto lock = std::lock_guard<std::mutex>(server.staged_mtx_);
    auto it = server.staged_guards_.find(channel_key);
    if (it == server.staged_guards_.end()) {
      LOG(ERROR) << "OnImmRecvCall unknown channel_key=0x" << std::hex
                 << channel_key << std::dec
                 << " imm_data=" << imm_data;
      return;
    }
    guard = it->second;
  }

  StagedRecvBuffer recv_buf{};
  if (!guard->find_buffer(buffer_id, &recv_buf)) {
    LOG(ERROR) << "OnImmRecvCall unknown buffer_id=" << buffer_id
               << " channel_key=0x" << std::hex << channel_key << std::dec;
    return;
  }

  const uint64_t recv_start = time_point_to_microseconds(std::chrono::system_clock::now());
  this->handle_kv_cache_data(channel, recv_buf.mem.buf, recv_buf.size, recv_start, 0, 0);
}

// ---------------------------------------------------------------------------
// get_h2d_stream / handle_kv_cache_data
// ---------------------------------------------------------------------------

cudaStream_t RDMAStagedServer::get_h2d_stream() {
  return get_thread_local_h2d_stream();
}

void RDMAStagedServer::StagedCtxCallback::handle_kv_cache_data(
  std::shared_ptr<XChannel>& channel, char *in_buf, size_t len,
  uint64_t recv_start, uint32_t recv_time, uint32_t onrecv_queue_us) {

  auto& self = *this;
  if (len < RPC_HEADER + sizeof(uint64_t) + sizeof(size_t) + sizeof(size_t)) {
    LOG(ERROR) << "RDMA handle_kv_cache_data: message too short, len=" << len;
    return;
  }

  char* buf_ptr = in_buf;
  auto [magic, reqid] = deser_rpc_header(buf_ptr);
  buf_ptr += RPC_HEADER;
  assert(magic == KV_CACHE_DATA_MAGIC);

  uint64_t actual_len;
  memcpy(&actual_len, buf_ptr, sizeof(actual_len));
  buf_ptr += sizeof(actual_len);
  RTASSERT(actual_len <= len);

  size_t layer_idx;
  memcpy(&layer_idx, buf_ptr, sizeof(size_t));
  buf_ptr += sizeof(size_t);

  size_t metadata_size;
  memcpy(&metadata_size, buf_ptr, sizeof(size_t));
  buf_ptr += sizeof(size_t);

  size_t metadata_bytes = metadata_size * sizeof(IpcBlock);
  size_t expected_len = RPC_HEADER + sizeof(uint64_t) + sizeof(size_t) + sizeof(size_t) + metadata_bytes;
  if (actual_len < expected_len) {
    LOG(ERROR) << "RDMA handle_kv_cache_data: incomplete metadata, expected=" << expected_len << ", actual=" << actual_len;
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
  size_t tensor_data_size = actual_len - expected_len;

  auto& ptrs = self.staged_server_->gpu_ptrs_;
  if (layer_idx >= ptrs.size()) {
    LOG(ERROR) << "RDMA handle_kv_cache_data: invalid layer_idx=" << layer_idx << " max=" << ptrs.size();
    return;
  }
  void* layer_gpu_ptr = ptrs[layer_idx];
  assert(layer_gpu_ptr != nullptr);

  int device_id = self.staged_server_->ctx_->device_id();
  auto [device_blk_buffer, host_blk_buffer] = get_kernel_copy_buffer(device_id);
  int64_t* device_blk_buffer_ptr = reinterpret_cast<int64_t*>(device_blk_buffer);
  int64_t* host_blk_buffer_ptr = reinterpret_cast<int64_t*>(host_blk_buffer);

  cudaStream_t h2d_stream = RDMAStagedServer::get_h2d_stream();

  const auto h2d_start_ts = std::chrono::system_clock::now();

  cudaError_t cuda_rt = copy_handle_data_with_kernel(
    buf_ptr, layer_gpu_ptr,
    blocks, tensor_data_size,
    CopyDirection::H2D, device_id, h2d_stream,
    device_blk_buffer_ptr, host_blk_buffer_ptr);
  RTASSERT(cuda_rt == cudaSuccess);

  auto buffer_size = RPC_HEADER
                    + sizeof(recv_start) + sizeof(recv_time)
                    + sizeof(onrecv_queue_us)
                    + sizeof(h2d_start_ts) * 2;

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

  auto cuda_rt_sync = cudaStreamSynchronize(h2d_stream);

  RTASSERT(cuda_rt_sync == cudaSuccess);
  const auto h2d_end_ts = std::chrono::system_clock::now();
  memcpy(resp_buf_ptr, &h2d_end_ts, sizeof(h2d_end_ts));

  Send(channel, std::move(resp_buf), [](Status s) {
    if (!s.IsOk()) {
      LOG(ERROR) << "RDMA handle_kv_cache_data: send response err=" << s.ErrMsg();
    }
  });
}

}  // namespace blade_llm

#endif  // ENABLE_RDMA
