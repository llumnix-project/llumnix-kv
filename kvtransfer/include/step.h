#ifndef KVTRANSFER_INCLUDE_STEP_H_
#define KVTRANSFER_INCLUDE_STEP_H_

#pragma once
#include <atomic>
#include <chrono>
#include <unordered_map>
#include "context.h"
#include "utils/semaphore.h"
#include "utils/timer.h"

namespace blade_llm {

constexpr inline std::memory_order drop_release(std::memory_order m) noexcept {
  return (m == std::memory_order_release ? std::memory_order_relaxed : (
      (m == std::memory_order_acq_rel || m == std::memory_order_seq_cst) ? std::memory_order_acquire : m));
}


template <typename T>
T atomic_fetch_max_explicit(std::atomic<T>* pv, typename std::atomic<T>::value_type v, std::memory_order m) noexcept {
  auto const mr = drop_release(m);
  auto t = pv->load(mr);
  while (std::max(v, t) != t) {
    if (pv->compare_exchange_weak(t, v, m, mr)) {
      break;
    }
  }
  return t;
}

class Step {
 public:
  const size_t step_idx;
  const Timepoint start_send_ts = SteadyClock::now();
  // sub_queue_time_us[i] = sub_submit_ts - sub_send_ts, appended in substep
  // submission order.
  // write by target mgr thread (do_submit)
  std::vector<int64_t> sub_queue_time_us;
  // write by python main thread
  Timepoint flush_send_ts;
  // write by wait layers thread.
  Timepoint wait_layers_start_ts;
  Timepoint wait_layers_end_ts;
 private:
  // write by multi send stub thread.
  // std::atomic<Timepoint> last_send_finish_ts_; does not compile with g++ 9.
  std::atomic<Timepoint> last_send_finish_ts_{Timepoint::min()};
 private:
  static_assert(std::atomic<Timepoint>::is_always_lock_free);
 public:
  explicit Step(size_t i) : step_idx(i) {}
  ~Step();

  Timepoint last_send_finish_ts() const noexcept {
    return this->last_send_finish_ts_.load(std::memory_order_relaxed);
  }

  void update_last_send_finish_ts(Timepoint val) noexcept {
    atomic_fetch_max_explicit(&this->last_send_finish_ts_, val, std::memory_order_relaxed);
  }

  void wait_layer_ready(uint32_t layer_i);
  uint32_t notify_layer_ready(uint32_t layer_i);
 private:
  SyncSemaphore data_signal_;
};

class StepGuard {
 public:
  const uint32_t num_layers;
 public:
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
  [[nodiscard]] size_t step_id() const;
  std::shared_ptr<Step>& step();
  void wait_layers();
  void after_record_one();
  void after_record_all();
  void layer_ready_all();
 private:
  std::shared_ptr<Step> step_;
  SyncSemaphore record_signal_;
  ICUDABarrier *cu_barrier_;
};
}

#endif //KVTRANSFER_INCLUDE_STEP_H_
