
#ifndef KVTRANSFER_RDMA_STAGED_CHANNEL_H
#define KVTRANSFER_RDMA_STAGED_CHANNEL_H

#include "protocol/rdma_channel.h"
#include "copy_utils.h"

#ifdef ENABLE_RDMA

namespace blade_llm {

class RDMAStagedChannel : public RDMAChannel {
 public:
  RDMAStagedChannel(const InstanceId &inst_id, WorkerId worker_id, CliBarexCtx *ctx) noexcept:
      RDMAChannel(inst_id, worker_id, ctx) {}

  ~RDMAStagedChannel() override;

  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) override;
  void send_data(size_t layer_index) override;
  void flush(std::string& out) override;

 private:
  struct RemoteStagedBuffer {
    uint64_t remote_addr = 0;
    uint64_t size = 0;
    uint32_t rkey = 0;
    uint32_t buffer_id = 0;
  };
  std::vector<accl::barex::memp_t> host_buffers_;
  std::vector<RemoteStagedBuffer> remote_buffers_;
  std::vector<std::future<void>> staged_submit_futs_;
  std::vector<std::pair<uint64_t, std::future<StagedTimePoints>>> staged_write_futs_;
  uint64_t staged_payload_size_ = 0;
};

}  // namespace blade_llm

#endif  // ENABLE_RDMA
#endif  // KVTRANSFER_RDMA_STAGED_CHANNEL_H
