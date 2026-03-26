#include "step.h"
#include "thrid_party/logging.h"
#include <iostream>
#include <iomanip>

namespace blade_llm {

Step::~Step() noexcept {
  // When step is destructed, all related activities are finished; emit metrics here.
  auto& self = *this;
  const auto last_send_ts = self.last_send_finish_ts();
  LOG(INFO) << std::fixed << std::setprecision(3)
            << "StepMetrics. StepIdx=" << self.step_idx
            << ",SubmitQueueUs=" << elapse_us(self.start_send_ts, self.submit_ts)
            << ",PythonExecUs=" << elapse_us(self.start_send_ts, self.flush_send_ts)
            << ",WaitLayerQueueUs=" << elapse_us(self.start_send_ts, self.wait_layers_start_ts)
            << ",ForwardExecUs=" << elapse_us(self.wait_layers_start_ts, self.wait_layers_end_ts)
            << ",SendNonoverlapUs=" << ielapse_us(self.wait_layers_end_ts, last_send_ts)
            << ",LastSendFlushTs=" << last_send_ts.time_since_epoch().count();  // send stub id
}

void Step::wait_layer_ready(uint32_t layer_i) {
  data_signal_.wait(layer_i);
};
uint32_t Step::notify_layer_ready(uint32_t layer_i) {
  return data_signal_.release(layer_i);
}

void StepGuard::wait_layers() {
  auto val = step_->notify_layer_ready(0);
  if (val > 0) {
    assert(val == num_layers);
    return ;
  }
  for(uint32_t layer_i = 0; layer_i < num_layers; layer_i ++ ) {
    record_signal_.wait(layer_i);
    cu_barrier_->wait(layer_i);
    const auto next_layer = layer_i + 1;
    val = step_->notify_layer_ready(next_layer);
    if (val > next_layer) {
      assert(val == num_layers);
      return ;
    }
  }
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
}


}