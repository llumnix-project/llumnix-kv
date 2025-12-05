

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
#include <mutex>
#include <unordered_map>
#include "thrid_party/logging.h"

#ifdef ENABLE_RDMA
#include "utils/gdr.h"
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

class BarexChannel {
  // OWNER: CliBarexCtx.connector_
  accl::barex::XConnector* const connector_;
  // may be nullptr.
  std::shared_ptr<accl::barex::XChannel> channel_;
public:
  BarexChannel(accl::barex::XConnector* conn, std::shared_ptr<accl::barex::XChannel> ch) noexcept;

  BarexChannel(BarexChannel&& other) noexcept:
    connector_(other.connector_),
    channel_(std::move(other.channel_)) {
  }

  ~BarexChannel() noexcept;

  auto* ch() const noexcept {
    assert(this->channel_ != nullptr);
    return this->channel_.get();
  }

  auto& sch() noexcept {
    return this->channel_;
  }

private:
  void destroy();

  BarexChannel(const BarexChannel&) = delete;
  BarexChannel& operator=(BarexChannel&&) = delete;
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

  const auto& layer_gdrcpy_mem() const noexcept {
    return this->layer_gdrcpy_mem_;
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
  // may be null
  std::vector<std::unique_ptr<GdrMemDesc>> layer_gdrcpy_mem_;
};

// RDMAMemHandle 可以 memcpy.
struct RDMAMemHandle {
  void *ptr = nullptr;
  uint32_t rkey = 0;
  uint32_t lkey = 0; // 不必要, 正好这里 padding 4 字节, 不用白不用.
};
static_assert(std::is_standard_layout_v<RDMAMemHandle>);

struct CliBarexCtx : public BarexCtx {
  CliBarexCtx(std::string mp_name,
              std::string tp_name,
              int tpcnt,
              Context *ctx);

  auto *connector() const noexcept {
    return this->connector_.get();
  }

  // rpc
  std::vector<RDMAMemHandle> get_mem_handles(std::shared_ptr<accl::barex::XChannel>& dst) const;
  uint32_t get_remote_crc(std::shared_ptr<accl::barex::XChannel>& dst,
                          const std::vector<IpcBlock>* data,
                          uint32_t lcrc);

 private:
   struct RpcCtxCb : public accl::barex::XChannelCallback {
     RpcCtxCb(const CliBarexCtx* clictx) noexcept: cli_ctx_(clictx) {}

     void OnRecvCall(accl::barex::XChannel *channel,
                     char *in_buf,
                     size_t len,
                     accl::barex::x_msg_header header) override;
   private:
     const CliBarexCtx* const cli_ctx_ = nullptr;  // owner: KV_CLIENT.ctx_
   };

   std::unique_ptr<RpcCtxCb> get_ctx_cb() const noexcept {
    return std::make_unique<RpcCtxCb>(this);
   }

   using OnRespF = std::function<void(accl::barex::Status, char*, size_t)>;
   void push(uint64_t reqid, OnRespF on_resp) const;
   // may return empty function
   OnRespF pop(uint64_t reqid) const;
   void on_send_error(accl::barex::Status s, uint64_t reqid) const;

 public:
  const uint64_t layer_blk_size;
 private:
  std::mutex mtx_;
  // reqid, on resp callback
  std::unordered_map<uint64_t, OnRespF> rpc_;
  std::unique_ptr<accl::barex::XConnector, XConnectorDeleter> connector_;
};

static constexpr int LAYER_NUM_MAX = 100;
struct RDMAInfo {
  char ip[INET_ADDRSTRLEN]{'\0'};  // decode listen ip, 以 '\0' 结尾.
  int port = 0;  // decode listen port
  std::vector<RDMAMemHandle> handles;
};


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
  std::shared_ptr<accl::barex::XChannel>& sch() noexcept;
 private:
  InstanceId const src_inst_id_;
  WorkerId const src_worker_id_ = 0;
  CliBarexCtx *const ctx_;  // owner: KvTransferClient

  std::string ip_;
  int port_{0};
  size_t dst_layer_blk_size_{0};
  uint32_t dst_layer_num_{0};

  int prev_ch_idx_ = 0;
  std::vector<IpcBlock>* data_ = nullptr;
  TPKind kind_ = TPKind::UNKNOWN;
  // sb is send block~
  size_t origin_sb_num_ = 0;
  size_t merged_sb_num_ = 0;
  size_t sb_size_min_ = 0;
  size_t sb_size_max_ = 0;
  size_t sb_size_total_ = 0;
  uint32_t crc_ = 0;
  bool enable_crc_ = false;
  // write_us
  std::vector<std::future<uint64_t>> write_futs_;
  std::vector<std::future<void>> send_futs_;

  // init by do_init
  std::vector<BarexChannel> chs_;
  std::vector<RDMAMemHandle> dst_handles_;
};

class RDMAServer : public ITransferServer {
 public:
  void start_server(ITransferService *service, Context *ctx) override;
 private:
  class CtxCallback : public accl::barex::XChannelCallback {
    ITransferService* const ser_ = nullptr;   // owner: KV_SERVICE
    const RDMAServer* const server_ = nullptr;   // owner: KV_SERVER
   public:
    CtxCallback(ITransferService *s, RDMAServer* v) noexcept:
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
    void resp_mem_handles(std::shared_ptr<accl::barex::XChannel>& channel, uint64_t reqid, char *in_buf, size_t len);
    void resp_remote_crc(std::shared_ptr<accl::barex::XChannel>& channel, uint64_t reqid, char *in_buf, size_t len);
  };
 private:
  RDMAInfo info_;
  BarexCtx* ctx_ = nullptr;  // OWNER: KvTransferService.ctx_
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
                   bool is_server_,
                   std::unique_ptr<accl::barex::XChannelCallback> &&cb) :
      name_prefix(std::move(name)),
      is_server(is_server_),
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
