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
#include "naming/eas_naming.h"

namespace blade_llm {

struct BatchSendTask {
  BatchSendTask() = default;

  BatchSendTask(std::shared_ptr<Step> s) noexcept:
    step(std::move(s)) {}

  BatchSendTask(const BatchSendTask&) = delete;

  BatchSendTask(BatchSendTask&& other) noexcept:
    step(std::move(other.step)),
    tasks(std::move(other.tasks)) {}

  BatchSendTask& operator=(BatchSendTask&& other) noexcept {
    this->step = std::move(other.step);
    this->tasks = std::move(other.tasks);
    return *this;
  }
public:
  std::shared_ptr<Step> step;
  std::vector<ReqSendTask> tasks;
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
  virtual void send_batch(BatchSendTask) = 0;
  virtual StubState check_state() = 0;
  virtual void stop() = 0;
  virtual ~ISendStub() = default;
};

using SendStub = std::unique_ptr<ISendStub>;

class ISendStubFactory {
 public:
  virtual SendStub create_stub(const InstanceId&,
                               WorkerId,
                               uint32_t start_layer,
                               uint32_t num_layers,
                               std::optional<TransferProtocol> ,
                               const std::optional<std::string> &) = 0;
  virtual ~ISendStubFactory() = default;
};

class KvSendStub : public ISendStub, public noncopyable {
 private:
  const InstanceId dstid_;
  const WorkerId dstworkerid_;
  const WorkerInfo src_info_;
  // OWNER: shared ownership with KvSendStubFactory
  std::shared_ptr<INamingWorkerClient> naming_ = nullptr;

  public:
  KvSendStub(InstanceId dstid, WorkerId dstworkerid,
             const WorkerInfo &src_info,
             uint32_t start_layer,
             uint32_t num_layers,
             std::unique_ptr<IChannelFactory> channel_factory,
             std::shared_ptr<INamingWorkerClient> naming) :
      dstid_(std::move(dstid)),
      dstworkerid_(dstworkerid),
      src_info_(src_info),
      naming_(std::move(naming)),
      start_layer_(start_layer),
      num_layers_(num_layers),
      channel_factory_(std::move(channel_factory)) {};
  KvSendStub(KvSendStub &&other) = delete;
  void start() override;
  void send_batch(BatchSendTask) override;
  StubState check_state() override;
  void stop() override;
  ~KvSendStub() override;
 private:
  void update_dst_info(WorkerInfo&& new_dst);
  void start_async();
  struct TaskContext;
 private:
  uint32_t const start_layer_{0};
  uint32_t const num_layers_{0};
  std::unique_ptr<IChannelFactory> channel_factory_;
  BlockingQueue<BatchSendTask> send_tasks_{};
  std::atomic<StubState> state_{INIT};
  std::optional<std::thread> send_backend_;
};

class KvSendStubFactory : public ISendStubFactory, public noncopyable {
 public:
  KvSendStubFactory(Context *ctx, GeneralNamingClient &&naming) :
      ctx_(ctx),
      naming_(std::move(naming)) {
    auto unique_naming_worker = naming_.create_naming_worker_client();
    naming_worker_ = std::shared_ptr<INamingWorkerClient>(std::move(unique_naming_worker));
  }

  SendStub create_stub(const InstanceId& dst_inst_name,
                       WorkerId dst_worker_id,
                       uint32_t start_layer,
                       uint32_t num_layers,
                       std::optional<TransferProtocol>,
                       const std::optional<std::string> &) override;
 private:
  Context *ctx_;
  std::shared_ptr<INamingWorkerClient> naming_worker_;
  GeneralNamingClient naming_;
};
}
#endif //KVTRANSFER_INCLUDE_TX_STUB_H_
