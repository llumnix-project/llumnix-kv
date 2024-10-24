#ifndef KVTRANSFER_INCLUDE_PROTOCOL_H_
#define KVTRANSFER_INCLUDE_PROTOCOL_H_

#pragma once

#include <vector>
#include <memory>
#include "common.h"

namespace blade_llm {

enum TransferType {
  UNKNOWN = 0,
  CUDA_IPC = 1,
  RDMA_DIRECT = 2,
};

TransferType type_from(uint32_t v);

class IProtocolCtx {
 public:
  virtual ~IProtocolCtx() = default;
};

std::unique_ptr<IProtocolCtx> create_protocol_ctx(const WorkerInfo &info, TransferType type,
                                                  uint32_t layer_num_blocks,
                                                  const std::vector<uint64_t> &layer_addrs);
std::vector<TransferType> get_supported_transfer_types();

} // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_PROTOCOL_H_
