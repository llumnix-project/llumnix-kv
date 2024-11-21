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

std::unique_ptr<IChannel> create_channel(Context *ctx, const TransferProtocol &proto) {
  if (!ctx->check_transfer_support(proto)) {
    throw std::runtime_error("unsupported transfer protocol: " + proto.to_string());
  }
  switch (proto.type) {
    case TransferProtocol::Kind::CUDA_IPC: {
      auto proto_ctx = ctx->get_protocol_ctx<CudaIpcContext>(proto);
      if (proto_ctx == nullptr) {
        throw std::runtime_error("cuda channel context not registered;");
      }
      return std::make_unique<CudaIpcChannel>(ctx->inst_name, ctx->worker_id, proto_ctx);
    }
    case TransferProtocol::Kind::RDMA_DIRECT: {
#ifdef ENABLE_RDMA
      auto proto_ctx = ctx->get_protocol_ctx<RDMAProtoContext>(proto);
      if (proto_ctx == nullptr) {
        throw std::runtime_error("RDMA channel context not registered;");
      }
      return std::make_unique<RDMAChannel>(ctx->inst_name, ctx->worker_id, proto_ctx->cli_barex_ctx());

#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    }
    default:throw std::runtime_error("Unknown transport protocol;");
  }
}
}
