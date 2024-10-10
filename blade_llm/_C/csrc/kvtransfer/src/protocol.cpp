#include <memory>
#include <stdexcept>
#include "protocol.h"

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
      // TODO
      throw std::runtime_error("RDMA Direct transport todo;");
#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    }
    default:throw std::runtime_error("Unknown transport type;");
  }
}
}