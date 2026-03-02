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

// src_offset, dst_offset, len
// return min_size, max_size, total_size, cnt
std::tuple<size_t, size_t, size_t, size_t> merge_interval(std::vector<IpcBlock> &input);


enum class TPKind {
  PEQD,
  PLTD,
  PGTD,
  UNKNOWN,
};


class IChannel {
 public:
  virtual void connect(const WorkerInfo &dst_info) = 0;

  // Here defines the communication protocol between KvSendStub.start_async and IChannel.
  // start_async first calls register_data to register data to send, along with data characteristics:
  // - kind: PEQD indicates data was produced when P tp size equals D tp size.
  // - data: register_data takes ownership; start_async must not access data afterwards. A cleaner
  //   interface would be register_data(std::vector<IpcBlock>&& data), std::vector<IpcBlock> flush();
  //   i.e. flush would return data back for the next iteration, but that's too cumbersome.
  // After each layer computation, send_data is called. send_data may be asynchronous -- its return
  // does not mean the transfer is complete. flush() blocks until all in-flight send_data calls finish.
  virtual void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) = 0;
  virtual void send_data(size_t layer_index) = 0;
  // out holds metrics collected during register_data and send_data, such as data characteristics
  // and send times. Prefer key=val,key2=val2 format for easy script processing.
  virtual void flush(std::string& out) = 0;
  virtual void send_notification(const std::vector<const ReqSendTask*>& reqs) = 0;

  virtual bool is_active() {
    return true;
  }

  virtual void close() {};
  virtual ~IChannel() = default;

  // ONLY FOR TEST
  void send_notification(IIterator<const ReqSendTask *> *reqs) {
    std::vector<const ReqSendTask*> vreqs;
    auto opt = reqs->next();
    while (opt.has_value()) {
      vreqs.emplace_back(opt.value());
      opt = reqs->next();
    }
    if (vreqs.empty()) {
      return ;
    }
    return this->send_notification(vreqs);
  }

};

using Channel = std::unique_ptr<IChannel>;
std::unique_ptr<IChannel> create_channel(Context *ctx, const TransferProtocol&);

class IChannelFactory {
public:
  virtual ~IChannelFactory() = default;
  virtual Channel create(const WorkerInfo& dst_info) = 0;
};

class ChannelFactory : public IChannelFactory {
  Context* const ctx_;  // onwner: kvclient
  std::optional<TransferProtocol> const proto_;
public:
  ChannelFactory(Context* ctx, std::optional<TransferProtocol> proto) noexcept:
    ctx_(ctx), proto_(proto) {}

  Channel create(const WorkerInfo& dst_info) override;
};

}
#endif //KVTRANSFER_INCLUDE_CHANNEL_H_
