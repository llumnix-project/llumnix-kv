

#ifndef KVTRANSFER_RDMA_CHANNEL_H
#define KVTRANSFER_RDMA_CHANNEL_H

#include "protocol/barex_protocol.h"

#ifdef ENABLE_RDMA

namespace blade_llm {

struct RDMAInfo {
  char ip[INET_ADDRSTRLEN]{'\0'};  // Decode listen IP, terminated by '\0'.
  int port = 0;  // decode listen port
  std::vector<RDMAMemHandle> handles;
};


class RDMAChannel : public IChannel, public noncopyable {
 public:
  RDMAChannel(const InstanceId &inst_id, WorkerId worker_id, CliBarexCtx *ctx) noexcept:
      src_inst_id_(inst_id),
      src_worker_id_(worker_id),
      ctx_(ctx) {}

  ~RDMAChannel() override;

  // connect is called on the main thread and should avoid blocking.
  // write is called on a background thread and may block.
  // The caller guarantees that connect happens before write.
  void connect(const WorkerInfo &dst_info) override;

  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) override;

  void send_data(size_t layer_index) override;
  void flush(std::string& out) override;
  bool is_active() override;

 protected:
  // do real connect.
  void do_init();
  accl::barex::XChannel *ch() noexcept;
  std::shared_ptr<accl::barex::XChannel>& sch() noexcept;

 protected:
  InstanceId const src_inst_id_;
  WorkerId const src_worker_id_ = 0;
  CliBarexCtx *const ctx_;  // owner: KvTransferClient

  std::string ip_;
  int port_{0};
  std::vector<size_t> dst_layer_blk_sizes_;
  uint32_t dst_layer_num_{0};

  int prev_ch_idx_ = 0;
  // outside vec represent each tensor's block list
  // inside vec represent blocks need to send from one tensor
  std::vector<std::vector<IpcBlock>> *data_ = nullptr;
  size_t dataperch_ = 0;
  TPKind kind_ = TPKind::UNKNOWN;
  // sb is send block~
  size_t origin_sb_num_ = 0;
  size_t merged_sb_num_ = 0;
  size_t sb_size_min_ = 0;
  size_t sb_size_max_ = 0;
  size_t sb_size_total_ = 0;
  uint32_t crc_ = 0;
  bool enable_crc_ = false;

  // init by do_init
  std::vector<BarexChannel> chs_;
  std::vector<RDMAMemHandle> dst_handles_;

 private:
  std::vector<std::future<uint64_t>> write_futs_;
  std::vector<std::future<void>> send_futs_;
};

class RDMAServer : public ITransferServer {
 public:
  void start_server(Context *ctx) override;

 protected:
  class CtxCallback : public accl::barex::XChannelCallback {
   public:
    explicit CtxCallback(RDMAServer* v) noexcept: server_(v) {}

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

   protected:
    const RDMAServer* const server_ = nullptr;

    void resp_mem_handles(std::shared_ptr<accl::barex::XChannel>& channel, uint64_t reqid, char *in_buf, size_t len);
    void resp_remote_crc(std::shared_ptr<accl::barex::XChannel>& channel, uint64_t reqid, char *in_buf, size_t len);
  };

 protected:
  RDMAInfo info_;
  BarexCtx* ctx_ = nullptr;
  std::unique_ptr<accl::barex::XListener, XListenerDeleter> listener_;
};

}  // namespace blade_llm

#endif  // ENABLE_RDMA
#endif // KVTRANSFER_RDMA_CHANNEL_H
