
#ifndef KVTRANSFER_INCLUDE_COPY_UTILS_H_
#define KVTRANSFER_INCLUDE_COPY_UTILS_H_

#include <cuda_runtime.h>
#include <chrono>
#include <future>
#include <type_traits>
#include <utility>
#include "common.h"
#include "envcfg.h"

#ifdef ENABLE_RDMA
#include "protocol/barex_protocol.h"
#endif

namespace blade_llm {

struct StagedTimePoints {
  std::chrono::system_clock::time_point send_data_start_ts_;
  std::chrono::system_clock::time_point d2h_start_ts_;
  std::chrono::system_clock::time_point d2h_end_ts_;
  std::chrono::system_clock::time_point h2d_start_ts;
  std::chrono::system_clock::time_point h2d_end_ts;
  uint64_t recv_start_ = 0;
  uint32_t recv_time_ = 0;
  uint32_t onrecv_queue_us_ = 0;
};

#ifdef ENABLE_RDMA

inline constexpr uint32_t KV_CACHE_DATA_MAGIC = 0x4B564361;  /* KVCa (KV Cache data) */
inline constexpr uint32_t STAGED_PREALLOC_REQ_MAGIC = 0x53505251;  /* SPRQ */
inline constexpr uint32_t STAGED_PREALLOC_RESP_MAGIC = 0x53505253;  /* SPRS */
inline constexpr uint32_t STAGED_IMM_BUFFER_BITS = 24;
inline constexpr uint32_t STAGED_IMM_BUFFER_MASK = (1u << STAGED_IMM_BUFFER_BITS) - 1;
inline constexpr uint32_t STAGED_IMM_TOTAL_BITS = STAGED_IMM_BUFFER_BITS;
inline constexpr uint32_t STAGED_IMM_TOTAL_MASK = STAGED_IMM_BUFFER_MASK;
inline constexpr uint32_t STAGED_PREALLOC_MAX_LAYERS = STAGED_IMM_BUFFER_MASK;

inline uint32_t encode_staged_imm_data(uint32_t buffer_id) noexcept {
  return buffer_id & STAGED_IMM_TOTAL_MASK;
}

inline uint32_t decode_staged_imm_data(uint32_t imm_data) noexcept {
  return imm_data & STAGED_IMM_TOTAL_MASK;
}

struct StagedPreallocRespEntry {
  uint64_t remote_addr = 0;
  uint64_t size = 0;
  uint32_t rkey = 0;
  uint32_t buffer_id = 0;
};
static_assert(std::is_standard_layout_v<StagedPreallocRespEntry>);
static_assert(std::is_trivially_copyable_v<StagedPreallocRespEntry>, "Must be trivial!");

struct SendKVCacheData {
  using Promise = std::promise<StagedTimePoints>;
private:
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;
  std::chrono::system_clock::time_point send_data_start_ts_;
  std::chrono::system_clock::time_point d2h_start_ts_;
  std::chrono::system_clock::time_point d2h_end_ts_;
public:
  SendKVCacheData(
    uint64_t r, std::shared_ptr<Promise> pr,
    std::chrono::system_clock::time_point send_data_start_ts,
    std::chrono::system_clock::time_point d2h_start_ts,
    std::chrono::system_clock::time_point d2h_end_ts) noexcept:
    reqid_(r),
    pr_(std::move(pr)),
    send_data_start_ts_(send_data_start_ts),
    d2h_start_ts_(d2h_start_ts),
    d2h_end_ts_(d2h_end_ts) {}

  void operator()(accl::barex::Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      auto ex = std::make_exception_ptr(std::runtime_error("SendKVCacheData ERR: " + s.ErrMsg()));
      self.pr_->set_exception(std::move(ex));
      return;
    }
    RTASSERT(len >= RPC_HEADER);
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == KV_CACHE_DATA_MAGIC);
    RTASSERT(reqid == self.reqid_);

    std::chrono::system_clock::time_point h2d_start_ts;
    std::chrono::system_clock::time_point h2d_end_ts;
    uint64_t recv_start;
    uint32_t recv_time;
    uint32_t onrecv_queue_us;
    if (len >= RPC_HEADER + sizeof(h2d_start_ts) + sizeof(h2d_end_ts)) {
      auto buf_ptr = buf + RPC_HEADER;
      memcpy(&recv_start, buf_ptr, sizeof(recv_start));
      buf_ptr += sizeof(recv_start);
      memcpy(&recv_time, buf_ptr, sizeof(recv_time));
      buf_ptr += sizeof(recv_time);
      memcpy(&onrecv_queue_us, buf_ptr, sizeof(onrecv_queue_us));
      buf_ptr += sizeof(onrecv_queue_us);
      memcpy(&h2d_start_ts, buf_ptr, sizeof(h2d_start_ts));
      buf_ptr += sizeof(h2d_start_ts);
      memcpy(&h2d_end_ts, buf_ptr, sizeof(h2d_end_ts));
    } else {
      auto ex = std::make_exception_ptr(std::runtime_error("SendKVCacheData ERR: No h2d_start_ts or h2d_end_ts in response"));
      self.pr_->set_exception(std::move(ex));
      return;
    }
    StagedTimePoints time_points;
    time_points.send_data_start_ts_ = self.send_data_start_ts_;
    time_points.d2h_start_ts_ = self.d2h_start_ts_;
    time_points.d2h_end_ts_ = self.d2h_end_ts_;
    time_points.h2d_start_ts = h2d_start_ts;
    time_points.h2d_end_ts = h2d_end_ts;
    time_points.recv_start_ = recv_start;
    time_points.recv_time_ = recv_time;
    time_points.onrecv_queue_us_ = onrecv_queue_us;

    self.pr_->set_value(std::move(time_points));
  }
};

#endif  // ENABLE_RDMA

std::pair<char*, char*> get_kernel_copy_buffer(int device_id);
cudaStream_t get_thread_local_h2d_stream();
cudaStream_t get_thread_local_d2h_stream();

}  // namespace blade_llm
#endif  // KVTRANSFER_INCLUDE_COPY_UTILS_H_
