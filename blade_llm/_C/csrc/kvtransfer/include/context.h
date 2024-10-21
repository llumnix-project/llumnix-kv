#ifndef KVTRANSFER_INCLUDE_CONTEXT_H_
#define KVTRANSFER_INCLUDE_CONTEXT_H_

#pragma once

#include <vector>
#include <atomic>
#include <cassert>
#include "common.h"
#include "protocol.h"

#define MAX_WORKERS_PER_INST (8)

namespace blade_llm {
class ICUDABarrier {
 public:
  virtual ~ICUDABarrier() = default;
  virtual void wait(uint32_t layer_idx) = 0;
};

class CudaEventBarrier : public ICUDABarrier {
 public:
  explicit CudaEventBarrier(const std::vector<uint64_t> event_addr) : event_addrs_(event_addr) {}
  void wait(uint32_t layer_idx) override;
 private:
  std::vector<uint64_t> event_addrs_;
};

class Context : noncopyable {
 public:
  Context(InstanceId inst_id, WorkerId worker_id);
  void set_tp(uint32_t tp_size, uint32_t worker_tp_rank);
  void set_transfer_type(TransferType);
  void set_block_params(uint32_t block_size, uint32_t token_size, uint32_t layer_num_blocks);
  void set_layer_data_address(uint32_t device_id, const std::vector<uint64_t> &layers);
  void set_cuda_barrier(std::unique_ptr<ICUDABarrier> &&barrier);

  WorkerInfo *worker_info_mutable();
  ICUDABarrier *cuda_barrier();
  [[nodiscard]] const WorkerInfo &worker_info() const;
  [[nodiscard]] const std::vector<uint64_t> &layer_data_address() const;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] uint32_t num_layers() const;
  [[nodiscard]] uint32_t layer_num_blocks() const;
  [[nodiscard]] TransferType transfer_type() const;
  [[nodiscard]] size_t block_size() const;

  template<class T>
  T *protocol_ctx() {
    if (protocol_ctx_ == nullptr) {
      protocol_ctx_ = create_protocol_ctx(worker_info_, transfer_type_, layer_num_blocks_, layer_data_address_);
    }
    return dynamic_cast<T *>(protocol_ctx_.get());
  }

 private:
  WorkerInfo worker_info_;
  TransferType transfer_type_{};
  uint32_t num_layers_{};
  uint32_t layer_num_blocks_{};
  std::vector<uint64_t> layer_data_address_;
  std::unique_ptr<ICUDABarrier> cuda_barrier_;
  std::unique_ptr<IProtocolCtx> protocol_ctx_;
};
}
#endif //KVTRANSFER_INCLUDE_CONTEXT_H_
