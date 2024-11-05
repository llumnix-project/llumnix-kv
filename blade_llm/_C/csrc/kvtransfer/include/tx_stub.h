#ifndef KVTRANSFER_INCLUDE_TX_STUB_H_
#define KVTRANSFER_INCLUDE_TX_STUB_H_

#pragma once
#include <memory>
#include <vector>
#include <thread>
#include <optional>
#include "common.h"
#include "channel.h"
#include "step.h"
#include "naming.h"
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

enum StubState {
  INIT = 0,
  WORKING = 1,
  POISONED = 2,
  STOPPING = 3,
  DISCARD = 4
};

class ISendStub {
 public:
  virtual void start() = 0;
  virtual void send_batch(const BatchSendTask &) = 0;
  virtual StubState check_state() = 0;
  virtual void stop() = 0;
  virtual ~ISendStub() = default;
};

using SendStub = std::unique_ptr<ISendStub>;

class ISendStubFactory {
 public:
  virtual SendStub create_stub(InstanceId,
                               WorkerId,
                               uint32_t start_layer,
                               uint32_t num_layers,
                               std::optional<TransferProtocol>) = 0;
  virtual ~ISendStubFactory() = default;
};

class KvSendStub : public ISendStub, public noncopyable {
 public:
  KvSendStub(const WorkerInfo &dst_info,
             const WorkerInfo &src_info,
             uint32_t start_layer,
             uint32_t num_layers,
             Channel &&channel) :
      dst_info_(dst_info),
      src_info_(src_info),
      start_layer_(start_layer),
      num_layers_(num_layers),
      ch_(std::move(channel)) {};
  KvSendStub(KvSendStub &&other) noexcept;
  void start() override;
  void send_batch(const BatchSendTask &) override;
  StubState check_state() override;
  void stop() override;
  ~KvSendStub() override;
 private:
  void start_async();
  const WorkerInfo dst_info_;
  const WorkerInfo src_info_;
  uint32_t start_layer_{0};
  uint32_t num_layers_{0};
  std::unique_ptr<IChannel> ch_;
  BlockingQueue<BatchSendTask> send_tasks_{};
  std::atomic<StubState> state_{INIT};
  std::optional<std::thread> send_backend_;
};

class KvSendStubFactory : public ISendStubFactory, public noncopyable {
 public:
  KvSendStubFactory(Context *ctx, std::unique_ptr<INamingClient> &&naming) :
      ctx_(ctx),
      naming_(std::move(naming)) {}
  SendStub create_stub(InstanceId dst_inst_id,
                       WorkerId dst_worker_id,
                       uint32_t start_layer,
                       uint32_t num_layers,
                       std::optional<TransferProtocol> = std::nullopt) override;
 private:
  Context *ctx_;
  std::unique_ptr<INamingClient> naming_;
};
}
#endif //KVTRANSFER_INCLUDE_TX_STUB_H_
