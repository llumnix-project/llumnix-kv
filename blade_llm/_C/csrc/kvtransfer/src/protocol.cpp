#include <memory>
#include <stdexcept>
#include "protocol.h"
#include "protocol/rdma_protocol.h"

namespace blade_llm {

TransferType type_from(uint32_t v) {
  switch (v) {
    case 1:return CUDA_IPC;
    case 2:return RDMA_DIRECT;
    default:return UNKNOWN;
  }
}

std::unique_ptr<IProtocolCtx> create_protocol_ctx(const WorkerInfo &info,
                                                  TransferType type,
                                                  uint32_t layer_num_blocks,
                                                  const std::vector<uint64_t> &layer_addrs) {
  switch (type) {
    case CUDA_IPC:
      // cuda ipc does not need the ctx.
      return nullptr;
    case RDMA_DIRECT: {
#ifdef ENABLE_RDMA
      auto gpu_dev_id = info.device_id;
      uint64_t layer_size = info.block_size * layer_num_blocks;
      return std::make_unique<CliBarexCtx>(gpu_dev_id, "prefill-mp", "prefill", 4,
                                           std::make_unique<CliCtxCallback>(),
                                           layer_addrs, layer_size);
#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    }
    default:throw std::runtime_error("Unknown transport type;");
  }
}

std::vector<TransferType> get_supported_transfer_types() {
  std::vector<TransferType> ret;
  ret.push_back(CUDA_IPC);
#ifdef ENABLE_RDMA
  ret.push_back(RDMA_DIRECT);
#endif
  return ret;
}
}
