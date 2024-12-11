

#ifndef KVTRANSFER_RDMA_PROTOCOL
#define KVTRANSFER_RDMA_PROTOCOL

#include "common.h"
#include "context.h"
#include "channel.h"
#include "server.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <thread>
#include <future>
#include "thrid_party/logging.h"

#ifdef ENABLE_RDMA
#include <accl/barex/barex.h>
#include <accl/barex/barex_types.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>
#include <accl/barex/xtimer.h>
#endif

namespace blade_llm {

size_t get_encode_size(const InstanceId &inst_id,
                       uint32_t worker_id,
                       const std::string &reqid,
                       const std::vector<uint32_t> &block_id);

void encode_notification(char *ptr,
                         const InstanceId &inst_id,
                         uint32_t worker_id,
                         const std::string &reqid,
                         const std::vector<uint32_t> &block_ids);

bool decode_notification(const char *ptr,
                         size_t len,
                         InstanceId &inst_id,
                         uint32_t &worker_id,
                         std::string &req_id,
                         std::vector<uint32_t> &block_ids);

#ifdef ENABLE_RDMA
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
    return this->mp_;
  }
 private:
  // 记得这里的顺序决定了析构顺序, 要注意成员放置顺序.
  accl::barex::XSimpleMempool* mp_ = nullptr;
  std::unique_ptr<accl::barex::XThreadpool, XThreadpoolDeleter> tp_;
  std::unique_ptr<accl::barex::XContext> xctx_;

  std::vector<BarexMRGuard> layer_mr_;
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
  RDMAChannel(const InstanceId &inst_id, WorkerId worker_id, CliBarexCtx *ctx) noexcept:
      src_inst_id_(inst_id),
      src_worker_id_(worker_id),
      ctx_(ctx) {}

  // connect 在主线程调用, 尽量不要阻塞.
  // write 在后台线程调用, 可以阻塞,
  // 上层会确保 connect happen-before write.
  void connect(const WorkerInfo &dst_info) override;

  void register_data(std::vector<IpcBlock>& data, TPKind kind) override;
  void send_data(size_t layer_index) override;
  void flush(std::string& out) override;
  void send_notification(const std::vector<const ReqSendTask*>& reqs) override;
  using IChannel::send_notification;

 private:
  void send_data_pltd(size_t layer_index);

  void do_write(uint32_t layer_idx,
                size_t src_offset,
                size_t dst_offset,
                size_t len);

  // do real connect.
  void do_init();

  accl::barex::XChannel *ch() noexcept;
 private:
  std::string ip_;
  int port_{0};
  CliBarexCtx *const ctx_;  // owner: KvTransferClient
  std::vector<void *> dst_ptrs_;
  std::vector<uint32_t> dst_rkeys_;
  size_t dst_layer_blk_size_{0};
  uint32_t dst_layer_num_{0};
  InstanceId src_inst_id_;
  WorkerId src_worker_id_ = 0;
  int prev_ch_idx_ = 0;

  std::vector<IpcBlock>* data_;
  TPKind kind_ = TPKind::UNKNOWN;
  // sb is send block~
  size_t origin_sb_num_ = 0;
  size_t merged_sb_num_ = 0;
  size_t sb_size_min_ = 0;
  size_t sb_size_max_ = 0;
  size_t sb_size_total_ = 0;
  // write_us
  std::vector<std::future<uint64_t>> write_futs_;
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
  const bool is_server;

  static std::unique_ptr<RDMAProtoContext> server_context(std::string &&name,
                                                          std::unique_ptr<accl::barex::XChannelCallback> &&);

  static std::unique_ptr<RDMAProtoContext> client_context(std::string &&name);

  RDMAProtoContext(std::string &&name,
                   bool is_server,
                   std::unique_ptr<accl::barex::XChannelCallback> &&cb) :
      name_prefix(std::move(name)),
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
#endif  // ENABLE_RDMA
}  // namespace blade_llm
#endif // KVTRANSFER_RDMA_PROTOCOL
