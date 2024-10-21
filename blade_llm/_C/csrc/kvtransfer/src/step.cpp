#include "step.h"

namespace blade_llm {
void Step::wait_layer_ready(uint32_t layer_i) {
  data_signal_.wait(layer_i);
};
void Step::notify_layer_ready(uint32_t layer_i) {
  data_signal_.release(layer_i);
}
void Step::start_one() {
  start_cnt_.fetch_add(1, std::memory_order_seq_cst);
}
void Step::finish_one() {
  finish_cnt_.fetch_add(1, std::memory_order_seq_cst);
};

bool Step::check_done() {
  auto s_cnt = start_cnt_.load(std::memory_order_relaxed);
  if (s_cnt > 0) {
    auto f_cnt = finish_cnt_.load(std::memory_order_relaxed);
    return s_cnt == f_cnt;
  }
  return false;
}

void StepGuard::wait_layers() {
  step_->notify_layer_ready(0);
  for(auto layer_i = 0; layer_i < num_layers; layer_i ++ ) {
    record_signal_.wait(layer_i);
    cu_barrier_->wait(layer_i);
    step_->notify_layer_ready(layer_i + 1);
    ready_layers_.fetch_add(1, std::memory_order_seq_cst);
  }
}

uint32_t StepGuard::ready_layers() {
  return ready_layers_.load(std::memory_order_relaxed);
}

void StepGuard::after_record_one() {
  record_signal_.release();
}
void StepGuard::after_record_all() {
  record_signal_.release(num_layers);
}
size_t StepGuard::step_id() const {
  return step_->step_idx;
}
std::shared_ptr<Step> &StepGuard::step() {
  return step_;
}
void StepGuard::layer_ready_all() {
  after_record_all();
  step_->notify_layer_ready(num_layers);
}
}