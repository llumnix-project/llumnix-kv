#ifndef KVTRANSFER_TCP_CHANNEL_H
#define KVTRANSFER_TCP_CHANNEL_H

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
#include "protocol/barex_protocol.h"
#include "utils/thread_pool.h"
#include <cuda_runtime.h>
#include "copy_utils.h"

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

using TCPTimePoints = StagedTimePoints;

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
  bool is_active() override;

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
  // per-tensor bytes actually put on the wire (Σ block.length for that tensor;
  // fp8 length when fp8 is enabled). Computed in register_data, used by send_data.
  std::vector<size_t> tensor_send_bytes_;
  // write_us: (reqid, future) pairs for timeout handling
  std::vector<std::pair<uint64_t, std::future<TCPTimePoints>>> write_futs_;
  std::vector<std::future<void>> send_futs_;

  // init by do_init
  std::vector<BarexChannel> chs_;   // owner: ctx_.connector_
  // CPU buffer for storing KV cache tensor data
  std::vector<accl::barex::memp_t> host_buffers_;
  const bool cast2fp8_ = env_bf162fp8_conversion(); // cast bf16 to fp8 before sending
};

struct TCPInfo {
  char ip[INET_ADDRSTRLEN]{'\0'};  // Decode listen IP, terminated by '\0'.
  int port = 0;  // decode listen port
  // per-layer, per-tensor GPU cache base pointers: ptrs[layer_idx][tensor_idx]
  std::vector<std::vector<void *>> ptrs;
  // Registered byte capacity for each cache tensor in a layer.
  std::vector<size_t> layer_blk_sizes;
};

class TCPServer: public ITransferServer {
 public:
  void start_server(Context *ctx) override;

 private:
  class CtxCallback : public accl::barex::XChannelCallback {
    TCPServer* const server_ = nullptr;   // owner: KV_SERVER

   public:
    explicit CtxCallback(TCPServer* v) noexcept: server_(v) {}

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
  BarexCtx* ctx_ = nullptr;
  uint32_t num_layers_ = 0;
  std::unique_ptr<accl::barex::XListener, XListenerDeleter> listener_;
};

}

#endif // KVTRANSFER_TCP_CHANNEL_H
