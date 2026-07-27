
#ifndef KVTRANSFER_RDMA_STAGED_SERVER_H
#define KVTRANSFER_RDMA_STAGED_SERVER_H

#include "protocol/rdma_channel.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_RDMA

namespace blade_llm {

class RDMAStagedServer : public RDMAServer {
 public:
  void start_server(Context *ctx) override;

 private:
  class StagedCtxCallback : public RDMAServer::CtxCallback {
    RDMAStagedServer* const staged_server_ = nullptr;
   public:
    explicit StagedCtxCallback(RDMAStagedServer* v) noexcept
        : CtxCallback(v), staged_server_(v) {}

    void OnRecvCall(std::shared_ptr<accl::barex::XChannel> channel,
                    char *in_buf,
                    size_t len,
                    accl::barex::x_msg_header header) noexcept override;

    void OnImmRecvCall(std::shared_ptr<accl::barex::XChannel> channel,
                       uint32_t imm_data) noexcept override;

   private:
    void resp_staged_prealloc(std::shared_ptr<accl::barex::XChannel>& channel, uint64_t reqid, char *in_buf, size_t len);
    void handle_kv_cache_data(std::shared_ptr<accl::barex::XChannel>& channel, char *in_buf, size_t len,
                               uint64_t recv_start, uint32_t recv_time, uint32_t onrecv_queue_us);
  };

 private:
  static cudaStream_t get_h2d_stream();

 public:
  struct StagedRecvBuffer {
    accl::barex::memp_t mem{};
    uint64_t size = 0;
    uint32_t buffer_id = 0;
    size_t layer_idx = 0;
  };

  class StagedBufferGuard {
   public:
    StagedBufferGuard(uintptr_t channel_key, BarexCtx* ctx) noexcept
        : channel_key_(channel_key), ctx_(ctx) {}

    ~StagedBufferGuard() noexcept;

    StagedBufferGuard(const StagedBufferGuard&) = delete;
    StagedBufferGuard& operator=(const StagedBufferGuard&) = delete;

    uintptr_t channel_key() const noexcept { return channel_key_; }

    void replace_buffers(std::unordered_map<uint32_t, StagedRecvBuffer> new_bufs);
    void release_all_buffers() noexcept;
    std::vector<uint32_t> alloc_buffer_ids(uint32_t count);

    bool find_buffer(uint32_t buffer_id, StagedRecvBuffer* out);

   private:
    uintptr_t channel_key_ = 0;
    BarexCtx* ctx_ = nullptr;
    std::mutex mtx_;
    std::unordered_map<uint32_t, StagedRecvBuffer> buffers_;
    uint32_t next_buffer_id_ = 1;
  };

 private:
  std::vector<void*> gpu_ptrs_;
  size_t layer_blk_size_{0};
  mutable std::mutex staged_mtx_;
  mutable std::unordered_map<uintptr_t, std::shared_ptr<StagedBufferGuard>> staged_guards_;
};

}  // namespace blade_llm

#endif  // ENABLE_RDMA
#endif  // KVTRANSFER_RDMA_STAGED_SERVER_H
