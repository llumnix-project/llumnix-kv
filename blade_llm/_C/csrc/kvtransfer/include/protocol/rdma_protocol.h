

#ifndef KVTRANSFER_RDMA_PROTOCOL
#define KVTRANSFER_RDMA_PROTOCOL

#ifdef ENABLE_RDMA
#include "common.h"
#include "context.h"
#include "channel.h"
#include "server.h"
#include <accl/barex/barex.h>
#include <accl/barex/barex_types.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>
#include <accl/barex/xtimer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <thread>
#include <future>
#include "thrid_party/logging.h"

namespace blade_llm {

struct XMempoolDeleter {
  void operator()(accl::barex::XSimpleMempool *mp);
};

struct XContextDeleter {
  void operator()(accl::barex::XContext *mp);
};

struct XThreadpoolDeleter {
  void operator()(accl::barex::XThreadpool *mp);
};

struct XListenerDeleter {
  void operator()(accl::barex::XListener *mp);
};

struct XConnectorDeleter {
  void operator()(accl::barex::XConnector *mp);
};

class BarexMRGuard : public noncopyable {
  accl::barex::memp_t mr_;
  accl::barex::XSimpleMempool *mp_ = nullptr; // owner: BarexCtx
  bool const release_;  // 若为 true 则使用 ReleaseAndDeregBuffer 否则仅使用 DeregUserMr
 private:
  BarexMRGuard(accl::barex::memp_t &&mr, accl::barex::XSimpleMempool *mp, bool r) noexcept:
      mr_(std::move(mr)),
      mp_(mp),
      release_(r) {}
 public:
  static BarexMRGuard RelDeregGuard(accl::barex::memp_t &&mr, accl::barex::XSimpleMempool *mp) noexcept {
    return BarexMRGuard(std::move(mr), mp, true);
  }
  static BarexMRGuard DeregGuard(accl::barex::memp_t &&mr, accl::barex::XSimpleMempool *mp) noexcept {
    return BarexMRGuard(std::move(mr), mp, false);
  }

  BarexMRGuard(BarexMRGuard &&other) noexcept:
      mr_(std::move(other.mr_)),
      mp_(other.mp_),
      release_(other.release_) {
    other.mp_ = nullptr;
  }

  const auto &mr() const noexcept {
    return this->mr_;
  }

  ~BarexMRGuard();
};

struct BarexCtx : public noncopyable {
  BarexCtx(std::string mp_name,
           std::string tp_name,
           int tpcnt,
           Context *ctx,
           std::unique_ptr<accl::barex::XChannelCallback> ctxcb);

  ~BarexCtx();

  const auto &layer_mr() const noexcept {
    return this->layer_mr_;
  }

  auto *xctx() const noexcept {
    return this->xctx_.get();
  }
  auto *tp() const noexcept {
    return this->tp_.get();
  }
  auto *mp() const noexcept {
    return this->mp_.get();
  }
 private:
  using XDevice = accl::barex::XDevice;
  static XDevice *choose_nic(const std::vector<XDevice *> &nic_devs, int gpu_dev);
 private:
  // 记得这里的顺序决定了析构顺序, 要注意成员放置顺序.
  std::unique_ptr<accl::barex::XSimpleMempool, XMempoolDeleter> mp_;
  std::unique_ptr<accl::barex::XThreadpool, XThreadpoolDeleter> tp_;
  std::unique_ptr<accl::barex::XContext> xctx_;

  std::vector<BarexMRGuard> layer_mr_;
  std::atomic<bool> stopped_{false};
  std::optional<std::thread> ctx_loop_thd_;
};

struct CliBarexCtx : public BarexCtx {
  const uint64_t layer_blk_size;
  CliBarexCtx(std::string mp_name,
              std::string tp_name,
              int tpcnt,
              Context *ctx,
              std::unique_ptr<accl::barex::XChannelCallback> ctxcb);

  auto *connector() const noexcept {
    return this->connector_.get();
  }
 private:
  std::unique_ptr<accl::barex::XConnector, XConnectorDeleter> connector_;
};

// RDMAMemHandle 可以 memcpy.
static constexpr int LAYER_NUM_MAX = 100;
struct RDMAInfo {
  char ip[INET_ADDRSTRLEN]{'\0'};  // decode listen ip, 以 '\0' 结尾.
  int port = 0;  // decode listen port
};

static_assert(std::is_standard_layout_v<RDMAInfo>);
static_assert(sizeof(RDMAInfo) <= MAX_ADDRESS_LEN);

class RDMAChannel : public IChannel, public noncopyable {
 public:
  RDMAChannel(InstanceId inst_id, WorkerId worker_id, CliBarexCtx *ctx) noexcept:
      src_inst_id_(inst_id),
      src_worker_id_(worker_id),
      ctx_(ctx) {}

  // init 在主线程调用, 尽量不要阻塞.
  // write 在后台线程调用, 可以阻塞,
  // 上层会确保 init happen-before write.
  //
  // 后续可以把 INotifyChannel, IDataChannel 合并. 这样一方面可以共用 barex channel. 另外一方面 write 可以
  // 只需要发起并不需要等待实际写入结束, 由 notify_send_done 负责等待. 甚至这里 write 可以暂存请求在
  // notify_send_done 时或者择机攒批发送.
  void connect(const WorkerInfo &dst_info) override;
  void send_data(size_t layer_index, const std::vector<IpcBlock> &data) override;
  void flush() override;
  void send_notification(IIterator<const ReqSendTask *> *reqs) override;

 private:
  void do_write(uint32_t layer_idx,
                size_t src_offset,
                size_t dst_offset,
                size_t len);

  void do_notify_send_done(const std::string &reqid,
                           const std::vector<uint32_t> &block_ids);

  // do real connect.
  void do_init();

  accl::barex::XChannel *ch() noexcept;
 private:
  std::string ip_;
  int port_{0};
  CliBarexCtx *const ctx_;  // owner: KvTransferClient
  std::vector<void*> dst_ptrs_;
  std::vector<uint32_t> dst_rkeys_;
  size_t dst_layer_blk_size_{0};
  uint32_t dst_layer_num_{0};
  InstanceId src_inst_id_ = 0;
  WorkerId src_worker_id_ = 0;
  int prev_ch_idx_ = 0;

  std::vector<std::future<void>> write_futs_;
  std::vector<std::future<void>> send_futs_;

  // init by do_init
  std::vector<accl::barex::XChannel *> chs_;   // owner: ctx_.connector_
};

class RDMAServer : public ITransferServer {
 public:
  void start_server(ITransferService *service, Context *ctx) override;
  private:
  class CtxCallback : public accl::barex::XChannelCallback {
    ITransferService *ser_;
   public:
    CtxCallback(ITransferService *s) noexcept: ser_(s) {}

    void OnRecvCall(accl::barex::XChannel *channel,
                    char *in_buf,
                    size_t len,
                    accl::barex::x_msg_header header) override;
  };
 private:
  std::unique_ptr<accl::barex::XListener, XListenerDeleter> listener_;
};

class CliCtxCallback : public accl::barex::XChannelCallback {
 public:
  // prefill 端 callback, 并不需要做什么.
  void OnRecvCall(accl::barex::XChannel *channel,
                  char *in_buf,
                  size_t len,
                  accl::barex::x_msg_header header) override {}
};

class RDMAProtoContext : public IProtocolContext {
 public:
  const std::string name_prefix;
  const int num_threads;
  const bool is_server;

  static std::unique_ptr<RDMAProtoContext> server_context(std::string &&name,
                                                          int num_threads,
                                                          std::unique_ptr<accl::barex::XChannelCallback> &&);

  static std::unique_ptr<RDMAProtoContext> client_context(std::string &&name,
                                                          int num_threads);

  RDMAProtoContext(std::string &&name,
                   int num_threads,
                   bool is_server,
                   std::unique_ptr<accl::barex::XChannelCallback> &&cb) :
      name_prefix(std::move(name)),
      num_threads(num_threads),
      is_server(is_server),
      callback_(std::move(cb)) {}

  bool check_support() override;
  void init(Context *ctx) override;
  [[nodiscard]] const TransferProtocol &protocol() const override {
    return this->protocol_;
  };
  BarexCtx *barex_ctx() {
    if (is_server) {
      return barex_ctx_.get();
    } else {
      return nullptr;
    }
  }

  CliBarexCtx *cli_barex_ctx() {
    if (is_server) {
      return nullptr;
    } else {
      return cli_barex_ctx_.get();
    }
  }

 private:
  std::unique_ptr<BarexCtx> barex_ctx_{nullptr};
  std::unique_ptr<CliBarexCtx> cli_barex_ctx_{nullptr};
  std::unique_ptr<accl::barex::XChannelCallback> callback_{nullptr};
  TransferProtocol protocol_{TransferProtocol::Kind::RDMA_DIRECT};
};

}  // namespace blade_llm
#endif  // ENABLE_RDMA
#endif // KVTRANSFER_RDMA_PROTOCOL
