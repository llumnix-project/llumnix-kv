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

public:
  std::shared_ptr<Step> step;
  std::vector<ReqSendTask> tasks;
};

class ISendStub {
 protected:
  const InstanceId dstid_;
  const WorkerId dstworkerid_;

 public:
  ISendStub(InstanceId dstid, WorkerId dstworkerid) noexcept:
    dstid_(std::move(dstid)),
    dstworkerid_(dstworkerid) {}

  const auto& dstid() const noexcept {
    return this->dstid_;
  }

  auto dstworkerid() const noexcept {
    return this->dstworkerid_;
  }

  virtual void send_batch(BatchSendTask) = 0;
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
  const WorkerInfo src_info_;
  // OWNER: shared ownership with KvSendStubFactory
  std::shared_ptr<INamingWorkerClient> naming_ = nullptr;

  public:
  KvSendStub(InstanceId dstid, WorkerId dstworkerid,
             const WorkerInfo &src_info,
             uint32_t start_layer,
             uint32_t num_layers,
             std::unique_ptr<IChannelFactory> channel_factory,
             std::shared_ptr<INamingWorkerClient> naming);
  KvSendStub(KvSendStub &&other) = delete;
  ~KvSendStub();
  void send_batch(BatchSendTask) noexcept override;
 private:
  struct TaskContext;
 private:
  uint32_t const start_layer_{0};
  uint32_t const num_layers_{0};
  std::unique_ptr<IChannelFactory> channel_factory_;
  std::unique_ptr<TaskContext> taskctx_;
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
