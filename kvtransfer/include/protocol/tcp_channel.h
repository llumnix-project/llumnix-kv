#include "common.h"
#include "context.h"
#include "channel.h"
#include "envcfg.h"
#include "server.h"
#include "utils/timer.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <thread>
#include <future>
#include <mutex>
#include <unordered_map>
#include "thrid_party/logging.h"
#include "protocol/rdma_protocol.h"
#include "utils/thread_pool.h"
#include <cuda_runtime.h>

#include <accl/barex/barex.h>
#include <accl/barex/barex_types.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>
#include <accl/barex/xtimer.h>

namespace blade_llm {

static constexpr uint32_t KERNEL_LAUNCH_ERROR = 505;
// KERNEL_COPY_MAX_BLOCK_NUM is now obtained from environment variable via env_kernel_copy_max_block_num()
// Default value is 8192 if BLLM_KVTRANS_KERNEL_COPY_MAX_BLOCK_NUM is not set

struct TCPTimePoints {
  std::chrono::system_clock::time_point send_data_start_ts_;
  std::chrono::system_clock::time_point d2h_start_ts_;
  std::chrono::system_clock::time_point d2h_end_ts_;
  std::chrono::system_clock::time_point h2d_start_ts;
  std::chrono::system_clock::time_point h2d_end_ts;
  uint64_t recv_start_ = 0;
  uint32_t recv_time_ = 0;
  uint32_t onrecv_queue_us_ = 0;
};

class TCPChannel : public IChannel, public noncopyable {
 public:
  TCPChannel(const InstanceId &inst_id, WorkerId worker_id, CliBarexCtx *ctx) noexcept:
      src_inst_id_(inst_id),
      src_worker_id_(worker_id),
      ctx_(ctx) {}

  ~TCPChannel();

  void connect(const WorkerInfo &dst_info) override;
  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) override;
  void send_data(size_t layer_index) override;
  void flush(std::string& out) override;
  void send_notification(const std::vector<const ReqSendTask*>& reqs) override;
  bool is_active() override;
  using IChannel::send_notification;

 private:
  // Get thread-local CUDA stream for D2H copy, lazily created on first use
  // Stream is created on the current device context when first accessed
  static cudaStream_t get_d2h_stream();

  // do real connect.
  void do_init();
  accl::barex::XChannel *ch() noexcept;
  std::shared_ptr<accl::barex::XChannel>& sch() noexcept;

 private:
  InstanceId const src_inst_id_;
  WorkerId const src_worker_id_ = 0;
  CliBarexCtx *const ctx_;  // owner: KvTransferClient

  std::string ip_;
  int port_{0};
  std::vector<size_t> dst_layer_blk_sizes_;
  uint32_t dst_layer_num_{0};

  int prev_ch_idx_ = 0;
  std::vector<std::vector<IpcBlock>> *data_ = nullptr;
  TPKind kind_ = TPKind::UNKNOWN;
  // sb is send block~
  size_t origin_sb_num_ = 0;
  size_t merged_sb_num_ = 0;
  size_t sb_size_min_ = 0;
  size_t sb_size_max_ = 0;
  size_t sb_size_total_ = 0;
  // write_us
  std::vector<std::future<TCPTimePoints>> write_futs_;
  std::vector<std::future<void>> send_futs_;

  // init by do_init
  std::vector<BarexChannel> chs_;   // owner: ctx_.connector_
  // CPU buffer for storing KV cache tensor data
  std::vector<accl::barex::memp_t> host_buffers_;
  const bool cast2fp8_ = env_bf162fp8_conversion(); // cast bf16 to fp8 before sending
};

struct TCPInfo {
  char ip[INET_ADDRSTRLEN]{'\0'};  // decode listen ip, 以 '\0' 结尾.
  int port = 0;  // decode listen port
  std::vector<void *> ptrs;
};

class TCPServer: public ITransferServer {
 public:
  void start_server(ITransferService *service, Context *ctx) override;

 private:
  class CtxCallback : public accl::barex::XChannelCallback {
    ITransferService* const ser_ = nullptr;   // owner: KV_SERVICE
    TCPServer* const server_ = nullptr;   // owner: KV_SERVER

   public:
    CtxCallback(ITransferService *s, TCPServer* v) noexcept:
        ser_(s), server_(v) {}

    void OnRecvCall(accl::barex::XChannel *channel,
                    char *in_buf,
                    size_t len,
                    accl::barex::x_msg_header header) {
      RTASSERT(false);
    }

    void OnRecvCall(std::shared_ptr<accl::barex::XChannel> channel,
                    char *in_buf,
                    size_t len,
                    accl::barex::x_msg_header header) noexcept override;

   private:
    void handle_kv_cache_data(std::shared_ptr<accl::barex::XChannel>& channel, char *in_buf, size_t len,
                               uint64_t recv_start, uint32_t recv_time, uint32_t onrecv_queue_us);
  };

 private:
  // Get thread-local CUDA stream for H2D copy, lazily created on first use
  static cudaStream_t get_h2d_stream();

 private:
  TCPInfo info_;
  BarexCtx* ctx_ = nullptr;  // OWNER: KvTransferService.ctx_
  uint32_t num_layers_ = 0;  // OWNER: KvTransferService.ctx (for num_layers access)
  std::unique_ptr<accl::barex::XListener, XListenerDeleter> listener_;
  // accl::barex::XThreadpool* sync_thread_pool_ = nullptr; // Thread pool for async CUDA stream synchronization
};

}