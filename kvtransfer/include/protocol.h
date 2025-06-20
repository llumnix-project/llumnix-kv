#ifndef KVTRANSFER_INCLUDE_PROTOCOL_H_
#define KVTRANSFER_INCLUDE_PROTOCOL_H_

#pragma once

#include <string>
#include <vector>

namespace blade_llm {

struct TransferProtocol {
 public:
  enum Kind {
    CUDA_IPC = 1,
    RDMA_DIRECT = 1U << 1
  };

  TransferProtocol(Kind p): type(p) {}
  static TransferProtocol cuda_ipc();
  static TransferProtocol rdma_direct();
  [[nodiscard]] std::string to_string() const;

  Kind type;
};

class SupportTransferProtocols {
 public:
  SupportTransferProtocols() = default;
  explicit SupportTransferProtocols(unsigned char p) : protocols_(p) {};

  void set_support(const TransferProtocol& t);
  void set_support(TransferProtocol::Kind t);
  [[nodiscard]] bool is_support(TransferProtocol::Kind t) const;
  [[nodiscard]] bool is_support(const TransferProtocol& t) const;
  [[nodiscard]] std::vector<TransferProtocol> as_vector() const;
  [[nodiscard]] unsigned char value() const { return protocols_; }
 private:
  unsigned char protocols_{0};
};

SupportTransferProtocols get_library_support_protocols();

}
#endif //KVTRANSFER_INCLUDE_PROTOCOL_H_
