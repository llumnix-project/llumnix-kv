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

  // 这里定义了 KvSendStub.start_async 与 IChannel 的通信协议.
  // start_async 首先调用 register_data 注册待发送的数据, 此时会额外传递一些与数据特征有关的信息.
  // - kind, PEQD 表明 data 是在 P tp size eq D tp size 情形下产生.
  // - data, register_data taker owner of data, start_async 不可再访问 data 了. 更正确的接口定义是:
  //   register_data(std::vector<IpcBlock>&& data), std::vector<IpcBlock> flush(); 即 flush
  //   还需要把 data 再返回出来给 start_async 下一轮迭代使用, 但太麻烦了.
  // 之后在每个 layer 计算完毕之后调用 send_data 来发送数据. send_data 可能是异步的, 其返回并不意味数据
  // 传输完毕. flush() 会阻塞等待所有 in-flighting send_data 结束.
  virtual void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) = 0;
  virtual void send_data(size_t layer_index) = 0;
  // out 存放着 register_data, send_data 期间产生的一些 metric 信息. 比如数据特征, 发送时间等.
  // out 格式最好是 key=val,key2=val2 形式, 便于后期脚本化处理.
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
