#ifndef KVTRANSFER_INCLUDE_SERVER_H_
#define KVTRANSFER_INCLUDE_SERVER_H_

#pragma once
#include <vector>
#include "common.h"
#include "context.h"

namespace blade_llm {

class ITransferService : public noncopyable {
 public:
  virtual void on_recv(const InstanceId&, WorkerId, const RequestId&, std::vector<uint32_t> &&block_ids) = 0;
  virtual ~ITransferService() = default;
};

class ITransferServer : public noncopyable {
 public:
  virtual void start_server(ITransferService *service, Context *ctx) = 0;
  virtual ~ITransferServer() = default;
};
ITransferServer *create_transfer_server(const TransferProtocol& protocol);
}

#endif //KVTRANSFER_INCLUDE_SERVER_H_
