#ifndef KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_
#define KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_

#pragma once
#include <memory>
#include <vector>
#include <thread>
#include "common.h"
#include "channel.h"
#include "step.h"

namespace blade_llm {

struct BatchSendTask {

  BatchSendTask() = default;
  BatchSendTask(std::vector<const RequestInfo *> &&ts) {
    tasks = std::make_shared<std::vector<const RequestInfo *>>(std::move(ts));
  };
  BatchSendTask(std::shared_ptr<Step> &s,
                std::vector<const RequestInfo *> &&ts) : step(s) {
    tasks = std::make_shared<std::vector<const RequestInfo *>>(std::move(ts));
  };

  std::shared_ptr<Step> step;
  std::shared_ptr<std::vector<const RequestInfo *>> tasks;
};

class ISendStub {
 public:
  virtual void connect(Context *ctx, std::unique_ptr<IChannel> &&ch, const WorkerInfo &dst_info) = 0;
  virtual void send_batch(const BatchSendTask &batch) = 0;
  virtual bool is_running() = 0;
  virtual ~ISendStub() = default;
};
using SendStub = std::unique_ptr<ISendStub>;
}
#endif //KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_
