#include "protocol.h"

namespace blade_llm {

TransferProtocol TransferProtocol::cuda_ipc() {
  return {TransferProtocol::Kind::CUDA_IPC};
}

TransferProtocol TransferProtocol::rdma_direct() {
  return {TransferProtocol::Kind::RDMA_DIRECT};
}

std::string TransferProtocol::to_string() const {
  switch (type) {
    case Kind::CUDA_IPC:return "CUDA_IPC";
    case Kind::RDMA_DIRECT:return "RDMA_DIRECT";
    default:return "UNKNOWN";
  }
}

void SupportTransferProtocols::set_support(const TransferProtocol &t) {
  protocols_ |= t.type;
}

void SupportTransferProtocols::set_support(TransferProtocol::Kind t) {
  protocols_ |= t;
}

bool SupportTransferProtocols::is_support(TransferProtocol::Kind t) const {
  return (protocols_ & t) > 0;
}

bool SupportTransferProtocols::is_support(const TransferProtocol &t) const {
  return (protocols_ & t.type) > 0;
}

std::vector<TransferProtocol> SupportTransferProtocols::as_vector() const {
  std::vector<TransferProtocol> v;
  if (is_support(TransferProtocol::Kind::CUDA_IPC)) {
    v.push_back(TransferProtocol::cuda_ipc());
  }
  if (is_support(TransferProtocol::Kind::RDMA_DIRECT)) {
    v.push_back(TransferProtocol::rdma_direct());
  }
  return v;
}

SupportTransferProtocols get_library_support_protocols() {
  SupportTransferProtocols s;
  s.set_support(TransferProtocol::Kind::CUDA_IPC);
#ifdef ENABLE_RDMA
  s.set_support(TransferProtocol::Kind::RDMA_DIRECT);
#endif
  return s;
}
}
