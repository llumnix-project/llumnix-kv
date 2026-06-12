#include "server.h"
#include "protocol/rdma_channel.h"
#include "protocol/rdma_staged_server.h"
#include "protocol/tcp_channel.h"
#include "envcfg.h"

namespace blade_llm {

ITransferServer *create_transfer_server(const TransferProtocol &protocol) {
  switch (protocol.type) {
    case TransferProtocol::Kind::TCP:return new TCPServer();
    case TransferProtocol::Kind::RDMA_DIRECT:
#ifdef ENABLE_RDMA
      if (env_rdma_staged()) {
        return new RDMAStagedServer();
      }
      return new RDMAServer();
#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    default:throw std::runtime_error("Unknown transport protocol;");
  }
}
}
