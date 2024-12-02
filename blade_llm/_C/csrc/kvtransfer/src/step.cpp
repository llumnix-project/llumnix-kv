#include "step.h"
#include "thrid_party/logging.h"
#include <iostream>
#include <iomanip>

namespace blade_llm {


static std::ostream& operator<<(std::ostream& os, const StepMetrics& self) {
  auto US = [] (uint64_t ns) {
    return ns / 1000.0;
  };

  const auto send_stub_cnt = self.send_stub_cnt_.load(std::memory_order_relaxed);
  os << std::fixed << std::setprecision(3) << "PythonExecTimeUs=" << US(self.python_exec_ns)
     << ",WaitLayersQueueUs=" << US(self.wait_layers_queue_ns)
     << ",WaitLayersExecUs=" << US(self.wait_layers_exec_ns)
     << ",SendStubCnt=" << send_stub_cnt
     << ",SendQueueUs(min|max|avg)" << US(self.send_queue_ns.min()) << '|'
     << US(self.send_queue_ns.max()) << '|'
     << US(self.send_queue_ns.avg(send_stub_cnt))
     << ",SendWaitDataUs(min|max|avg)" << US(self.send_wait_data_ns.min()) << '|'
     << US(self.send_wait_data_ns.max()) << '|'
     << US(self.send_wait_data_ns.avg(send_stub_cnt))
     << ",SendNonOverlapUs(min|max|avg)" << US(self.send_non_overlay_ns.min()) << '|'
     << US(self.send_non_overlay_ns.max()) << '|'
     << US(self.send_non_overlay_ns.avg(send_stub_cnt))
     << ",SendNotifyUs(min|max|avg)" << US(self.send_notification_ns.min()) << '|'
     << US(self.send_notification_ns.max()) << '|'
     << US(self.send_notification_ns.avg(send_stub_cnt))
     << ",SendFinishedReq(min|max|total)" << self.send_finished_req_n.min() << '|'
     << self.send_finished_req_n.max() << '|'
     << self.send_finished_req_n.total()
     << ",SendDataSize(min|max|total)" << self.send_data_size.min() << '|'
     << self.send_data_size.max() << '|'
     << self.send_data_size.total();
  return os;
}

#if 0
Step::~Step() noexcept {
  // 当 step 析构时, 意味着 step 所有相关活动都结束了, 此时适合输出 metric.
  LOG(INFO) << "StepMetric:StepIdx=" << this->step_idx << ',' << this->metric;
}
#endif

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
  for(auto layer_i = 0; layer_i < num_layers; layer_i ++ ) {
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
  auto val = step_->notify_layer_ready(num_layers);
  assert(val == num_layers);
}

StepGuard::~StepGuard() {
  // 其析构执行时, 意味着 python main thread/wait layer thread 都已经结束
  // 对 StepGuard 的引用, 是时候输出 metric 了.
  LOG(INFO) << std::fixed << std::setprecision(3)
            << "StepGuardMetric. StepIdx=" << this->step_id()
            << ",PythonExecUs=" << this->python_exec_ns / 1000.0
            << ",WaitLayerQueueUs=" << this->wait_layers_queue_ns / 1000.0
            << ",WaitLayerExecUs=" << this->wait_layers_exec_ns / 1000.0;
}

}