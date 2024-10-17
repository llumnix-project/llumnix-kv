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
  explicit BatchSendTask(std::vector<const RequestInfo *> &&ts) {
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

class KvSendStub : public ISendStub, public noncopyable {
 public:
  KvSendStub(InstanceId dst_inst_id,
             WorkerId dst_worker_id,
             uint32_t start_layer,
             uint32_t num_layers,
             Channel &&channel) :
      dst_info_(dst_inst_id, dst_worker_id),
      start_layer_(start_layer),
      num_layers_(num_layers),
      ch_(std::move(channel)) {};
  KvSendStub(KvSendStub &&other) noexcept;
  void connect(Context *, const WorkerInfo &dst_info) override;
  void send_batch(const BatchSendTask &) override;
  bool is_running() override;
  ~KvSendStub() override;
 private:
  WorkerInfo dst_info_;
  uint32_t start_layer_{0};
  uint32_t num_layers_{0};
  std::unique_ptr<IChannel> ch_;
  std::atomic_bool is_running_{false};
  BlockingQueue<BatchSendTask> send_tasks_{};
  std::optional<std::thread> send_backend_;
};

class KvSendStubFactory : public ISendStubFactory {
 public:
  explicit KvSendStubFactory(Context *ctx) : ctx_(ctx) {}
  SendStub create_stub(InstanceId dst_inst_id,
                       WorkerId dst_worker_id,
                       uint32_t start_layer,
                       uint32_t num_layers) override;
 private:
  Context *ctx_;
};
}
#endif //KVTRANSFER_INCLUDE_UTILS_TX_STUB_H_
