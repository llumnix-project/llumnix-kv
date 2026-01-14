#include "server.h"
#include "protocol/rdma_protocol.h"
#include "protocol/tcp_channel.h"

namespace blade_llm {

ITransferServer *create_transfer_server(const TransferProtocol &protocol) {
  switch (protocol.type) {
    case TransferProtocol::Kind::TCP:return new TCPServer();
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
