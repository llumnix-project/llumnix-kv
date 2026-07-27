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

enum class IpcBlockOffset {
  SOURCE,
  DESTINATION,
};

// Validate a set of offsets against the registered GPU buffer before handing
// them to CUDA or RDMA. Unlike assert(), this remains active in release builds
// and emits the exact offending range before throwing.
void validate_ipc_block_bounds(
    const std::vector<IpcBlock>& blocks,
    size_t capacity,
    IpcBlockOffset offset_kind,
    const char* copy_path,
    size_t layer_idx,
    size_t tensor_idx,
    size_t length_scale = 1);

// src_offset, dst_offset, len
// return min_size, max_size, total_size, cnt
std::tuple<size_t, size_t, size_t, size_t> merge_interval(std::vector<IpcBlock> &input);

static inline std::ostream& operator<<(std::ostream& os, const IpcBlock& data) {
  os << "IpcBlock(" << "src=" << data.src_offset
     << ",dst=" << data.dst_offset
     << ",len=" << data.length << ")";
  return os;
}

enum class TPKind {
  PEQD,
  PLTD,
  PGTD,
  UNKNOWN,
};


class IChannel {
 public:
  virtual void connect(const WorkerInfo &dst_info) = 0;

  // Defines the protocol between KvSendStub.start_async and IChannel.
  // start_async first calls register_data to register data to send, passing extra data characteristics.
  // - kind: PEQD indicates data was produced when P tp_size == D tp_size.
  // - data: register_data takes ownership; start_async must not access data afterwards.
  //   register_data(std::vector<IpcBlock>&& data), std::vector<IpcBlock> flush(); i.e., flush
  //   should return data back for start_async to reuse in the next iteration, but that is overly complex.
  // After each layer finishes, send_data is called. send_data may be async: its return does not mean
  // data transfer is complete. flush() blocks until all in-flight send_data calls finish.
  virtual void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) = 0;
  virtual void send_data(size_t layer_index) = 0;
  // out stores metrics produced during register_data/send_data, e.g., data characteristics and timing.
  // out format should be key=val,key2=val2 for easy scripted processing.
  virtual void flush(std::string& out) = 0;

  virtual bool is_active() {
    return true;
  }

  virtual void close() {};
  virtual ~IChannel() = default;
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
