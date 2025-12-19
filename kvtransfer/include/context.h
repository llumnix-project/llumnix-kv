#ifndef KVTRANSFER_INCLUDE_CONTEXT_H_
#define KVTRANSFER_INCLUDE_CONTEXT_H_

#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <cassert>
#include <unordered_map>
#include "common.h"
#include "protocol.h"


namespace blade_llm {
class ICUDABarrier {
 public:
  virtual ~ICUDABarrier() = default;
  virtual void wait(uint32_t layer_idx) = 0;
};

class CudaEventBarrier : public ICUDABarrier {
 public:
  explicit CudaEventBarrier(const std::vector<uint64_t> &events) : event_addrs_(events) {}
  void wait(uint32_t layer_idx) override;
 private:
  std::vector<uint64_t> event_addrs_;
};

class IProtocolContext;

class Context : noncopyable {
 public:
  const InstanceId inst_name;
  const WorkerId worker_id;

  Context(const InstanceId& inst_name, const WorkerId& worker_id);
  void set_tp(uint32_t tp_size, uint32_t worker_tp_rank);
  void set_block_params(std::vector<size_t> block_sizes, std::vector<size_t> token_sizes, uint32_t layer_num_blocks);
  void set_layer_info(uint8_t device_id, const std::vector<std::vector<LayerInfo>> &all_layer_infos);
  void set_cuda_barrier(std::unique_ptr<ICUDABarrier> &&);
  void register_protocol(std::unique_ptr<IProtocolContext> &&);

  WorkerInfo *worker_info_mutable();
  ICUDABarrier *cuda_barrier();
  [[nodiscard]] const WorkerInfo &worker_info() const;
  [[nodiscard]] const std::vector<std::vector<LayerInfo>> &all_layer_infos() const;
  [[nodiscard]] const std::vector<std::vector<uint64_t>> &layer_data_address() const;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] uint32_t num_layers() const;
  [[nodiscard]] uint32_t layer_num_blocks() const;
  [[nodiscard]] const SupportTransferProtocols& support_protocols() const;
  [[nodiscard]] bool check_transfer_support(TransferProtocol) const;
  [[nodiscard]] const std::vector<size_t> &block_sizes() const;
  template<class T>
  T* get_protocol_ctx(const TransferProtocol &t) {
    auto ret = protocol_ctxs_.find(t.type);
    if (ret == protocol_ctxs_.end()) {
      return nullptr;
    } else {
      return dynamic_cast<T *>(ret->second.get());
    }
  }

 private:
  int device_id_{-1};
  WorkerInfo worker_info_;
  SupportTransferProtocols transfer_protos_;
  std::vector<std::vector<LayerInfo>> all_layer_infos_;
  std::vector<std::vector<uint64_t>> layer_data_address_;
  std::unique_ptr<ICUDABarrier> cuda_barrier_;
  std::unordered_map<TransferProtocol::Kind, std::unique_ptr<IProtocolContext>> protocol_ctxs_;
};


class IProtocolContext {
 public:
  virtual bool check_support() = 0;
  virtual void init(Context *ctx) = 0;
  [[nodiscard]] virtual const TransferProtocol& protocol() const = 0;
  [[nodiscard]] bool is_initialized() const { return is_inited_; };
  virtual ~IProtocolContext() = default;
 protected:
  bool is_inited_{false};
};

}
#endif //KVTRANSFER_INCLUDE_CONTEXT_H_
