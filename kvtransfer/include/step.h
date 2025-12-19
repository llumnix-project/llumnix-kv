#ifndef KVTRANSFER_INCLUDE_STEP_H_
#define KVTRANSFER_INCLUDE_STEP_H_

#pragma once
#include <atomic>
#include <chrono>
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


template <typename T>
T atomic_fetch_min_explicit(std::atomic<T>* pv, typename std::atomic<T>::value_type v, std::memory_order m) noexcept {
  auto const mr = drop_release(m);
  auto t = pv->load(mr);
  while (std::min(v, t) != t) {
    if (pv->compare_exchange_weak(t, v, m, mr)) {
      break;
    }
  }
  return t;
}

class AtomicMinMaxTotal {
  std::atomic<uint64_t> min_{UINT64_MAX};
  std::atomic<uint64_t> max_{0};
  std::atomic<uint64_t> total_{0};
public:
  // may be called in multi-thread.
  void observe(uint64_t val) const noexcept {
    auto& self = *const_cast<AtomicMinMaxTotal*>(this);  // SAFETY: atomic
    atomic_fetch_min_explicit(&self.min_, val, std::memory_order_relaxed);
    atomic_fetch_max_explicit(&self.max_, val, std::memory_order_relaxed);
    std::atomic_fetch_add_explicit(&self.total_, val, std::memory_order_relaxed);
  }

  uint64_t min() const noexcept {
    return this->min_.load(std::memory_order_relaxed);
  }

  uint64_t max() const noexcept {
    return this->max_.load(std::memory_order_relaxed);
  }

  uint64_t total() const noexcept {
    return this->total_.load(std::memory_order_relaxed);
  }

  double avg(uint64_t cnt) const noexcept {
    return this->total() / double(cnt);
  }
};

struct StepMetrics {
  Timepoint start_send_ts;
public:
  uint64_t python_exec_ns = 0;
  uint64_t wait_layers_queue_ns = 0;
  uint64_t wait_layers_exec_ns = 0;

  // may be update by multi thread.
  std::atomic<uint64_t> send_stub_cnt_{0};
  AtomicMinMaxTotal send_queue_ns;
  AtomicMinMaxTotal send_wait_data_ns;
  AtomicMinMaxTotal send_non_overlay_ns;
  AtomicMinMaxTotal send_notification_ns;
  AtomicMinMaxTotal send_finished_req_n;
  AtomicMinMaxTotal send_data_size;  // per layer

public:
  void inc_send_stub() const noexcept {
    auto& self = *const_cast<StepMetrics*>(this);  // SAFETY: atomic
    self.send_stub_cnt_.fetch_add(1, std::memory_order_relaxed);
  }
};

class Step {
 public:
  const size_t step_idx;
  const Timepoint start_send_ts = SteadyClock::now();
  // write by target mgr thread
  Timepoint submit_ts;
  // write by python main thread
  Timepoint flush_send_ts;
  // write by wait layers thread.
  Timepoint wait_layers_start_ts;
  Timepoint wait_layers_end_ts;
 private:
  // write by multi send stub thread.
  // std::atomic<Timepoint> last_send_finish_ts_; 在 g++9 上编译报错.
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
