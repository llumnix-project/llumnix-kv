#include <stdexcept>
#include "channel.h"

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
      //TODO: return std::make_unique<CudaIpcChannel>(ctx);
      throw std::runtime_error("cuda channel todo;");
    case RDMA_DIRECT:
#ifdef ENABLE_RDMA
      //TODO: return std::make_unique<RDMAChannel>(ctx);
      throw std::runtime_error("cuda channel todo;");
#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    default:throw std::runtime_error("Unknown transport type;");
  }
}
}
