#ifndef KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_
#define KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_

#pragma once
#include <memory>
#include <vector>
#include <thread>
#include "common.h"
#include "channel.h"
#include "step.h"
#include "utils/block_queue.h"

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
  virtual void connect(Context*, const WorkerInfo &dst) = 0;
  virtual void send_batch(const BatchSendTask &) = 0;
  virtual bool is_running() = 0;
  virtual ~ISendStub() = default;
};

using SendStub = std::unique_ptr<ISendStub>;

class ISendStubFactory {
 public:
  virtual SendStub create_stub(InstanceId, WorkerId, uint32_t start_layer, uint32_t num_layers) = 0;
  virtual ~ISendStubFactory() = default;
};
}
#endif //KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_
