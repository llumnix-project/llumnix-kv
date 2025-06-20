#include "server.h"
#include "protocol/cuda_ipc.h"
#include "protocol/rdma_protocol.h"

namespace blade_llm {

ITransferServer *create_transfer_server(const TransferProtocol &protocol) {
  switch (protocol.type) {
    case TransferProtocol::Kind::CUDA_IPC:return new CudaTransferServer();
    case TransferProtocol::Kind::RDMA_DIRECT:
#ifdef ENABLE_RDMA
      return new RDMAServer();
#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    default:throw std::runtime_error("Unknown transport protocol;");
  }
}
}
