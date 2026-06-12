#ifndef KVTRANSFER_INCLUDE_SERVER_H_
#define KVTRANSFER_INCLUDE_SERVER_H_

#pragma once
#include <vector>
#include "common.h"
#include "context.h"

namespace blade_llm {

class ITransferServer : public noncopyable {
 public:
  virtual void start_server(Context *ctx) = 0;
  virtual ~ITransferServer() = default;
};
ITransferServer *create_transfer_server(const TransferProtocol& protocol);
}

#endif //KVTRANSFER_INCLUDE_SERVER_H_
