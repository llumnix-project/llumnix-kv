#include <stdexcept>
#include "channel.h"
#include "protocol/cuda_ipc.h"
#include "protocol/rdma_protocol.h"

namespace blade_llm {
bool IpcBlock::operator==(const IpcBlock &other) const {
  return src_offset == other.src_offset &&
      dst_offset == other.dst_offset &&
      length == other.length;
}

std::unique_ptr<IChannel> create_channel(Context *ctx) {
  auto type = ctx->transfer_type();
  switch (type) {
    case CUDA_IPC:
      return std::make_unique<CudaIpcChannel>(ctx);
    case RDMA_DIRECT:
#ifdef ENABLE_RDMA
      return std::make_unique<RDMAChannel>(ctx);
#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    default:throw std::runtime_error("Unknown transport type;");
  }
}
}
