#ifndef KVTRANSFER_INCLUDE_STEP_H_
#define KVTRANSFER_INCLUDE_STEP_H_

#pragma once
#include "context.h"
#include "utils/semaphore.h"

namespace blade_llm {
class Step {
 public:
  const size_t step_idx;
  Step(size_t i) : step_idx(i) {}
  void wait_layer_ready(uint32_t layer_i);
  void notify_layer_ready(uint32_t layer_i);
  void start_one();
  void finish_one();
  bool check_done();
 private:
  SyncSemaphore data_signal_;
  SyncSemaphore record_signal;
  std::atomic_size_t start_cnt_;
  std::atomic_size_t finish_cnt_;
};

class StepGuard {
 public:
  const uint32_t num_layers;
  StepGuard(uint32_t layers, ICUDABarrier*barrier, std::shared_ptr<Step> step) :
      num_layers(layers),
      step_(step),
      cu_barrier_(barrier) {
    assert(barrier != nullptr);
    assert(step != nullptr);
  }

  StepGuard(Context*ctx, std::shared_ptr<Step> step) :
      num_layers(ctx->num_layers()),
      step_(step),
      cu_barrier_(ctx->cuda_barrier()) {
    assert(step != nullptr);
  }
  size_t step_id() const;
  std::shared_ptr<Step>& step();
  void wait_layers();
  uint32_t ready_layers();
  void after_record_one();
  void after_record_all();
  void layer_ready_all();

 private:
  std::atomic_size_t ready_layers_{0};
  std::shared_ptr<Step> step_;
  SyncSemaphore record_signal_;
  ICUDABarrier *cu_barrier_;
};
}

#endif //KVTRANSFER_INCLUDE_STEP_H_
