#include "common.h"
#include "context.h"
#include "channel.h"
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

struct TCPTimePoints {
  SteadyClock::time_point send_data_start_ts_;
  SteadyClock::time_point d2h_start_ts_;
  SteadyClock::time_point d2h_end_ts_;
  SteadyClock::time_point h2d_start_ts;
  SteadyClock::time_point h2d_end_ts;
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
  // do real connect.
  void do_init();
  accl::barex::XChannel *ch() noexcept;
  std::shared_ptr<accl::barex::XChannel>& sch() noexcept;

  // Copy GPU data to host buffer and prepare metadata for sending
  void copy_send_data(
      size_t layer_idx,
      uint32_t magic,
      uint64_t reqid,
      const std::vector<IpcBlock>& data,
      size_t metadata_size,
      char* buf_ptr,
      char* meta_buf_ptr,
      char* tensor_buf_ptr,
      const accl::barex::memp_t& src_mr_base);

#if ENABLE_BATCH_COPY
  void copy_send_data_batch(
      size_t layer_idx,
      uint32_t magic,
      uint64_t reqid,
      const std::vector<IpcBlock>& data,
      size_t metadata_size,
      char* buf_ptr,
      char* meta_buf_ptr,
      char* tensor_buf_ptr,
      const accl::barex::memp_t& src_mr_base);
#endif  // ENABLE_BATCH_COPY

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
  // CUDA stream for asynchronous memory copy
  cudaStream_t cpy_stream_{nullptr};
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
    const TCPServer* const server_ = nullptr;   // owner: KV_SERVER

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
    void handle_kv_cache_data(std::shared_ptr<accl::barex::XChannel>& channel, char *in_buf, size_t len);
  };

 private:
  TCPInfo info_;
  cudaStream_t h2d_cpy_stream_{nullptr};
  BarexCtx* ctx_ = nullptr;  // OWNER: KvTransferService.ctx_
  std::unique_ptr<accl::barex::XListener, XListenerDeleter> listener_;
};

}