#ifndef KVTRANSFER_INCLUDE_CHANNEL_H_
#define KVTRANSFER_INCLUDE_CHANNEL_H_


#pragma once

#include <vector>
#include <string>
#include "common.h"
#include "context.h"
#include "utils/iterator.h"

namespace blade_llm {

struct IpcBlock {
  size_t src_offset;
  size_t dst_offset;
  size_t length;
  IpcBlock(size_t s, size_t d, size_t l) : src_offset(s), dst_offset(d), length(l) {};
  bool operator==(const IpcBlock &other) const;
};

class IChannel {
 public:
  virtual void connect(const WorkerInfo &dst_info) = 0;
  virtual void send_data(size_t layer_index, const std::vector<IpcBlock> &data) = 0;
  virtual void send_notification(IIterator<const ReqSendTask *> *reqs) = 0;
  virtual void flush() = 0;
  virtual void close() {};
  virtual ~IChannel() = default;
};

using Channel = std::unique_ptr<IChannel>;
std::unique_ptr<IChannel> create_channel(Context *ctx, const TransferProtocol&);
}
#endif //KVTRANSFER_INCLUDE_CHANNEL_H_
