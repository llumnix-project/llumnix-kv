#ifndef KVTRANSFER_INCLUDE_UTILS_TIMER_H_
#define KVTRANSFER_INCLUDE_UTILS_TIMER_H_

#pragma once
#include <chrono>

namespace blade_llm {

using SteadyClock = std::chrono::steady_clock;
using Timepoint = SteadyClock::time_point;

class TimeWatch {
 public:
  TimeWatch();
  size_t get_elapse_ms();
  size_t get_elapse_us();
  size_t get_elapse_ns();

  auto start_ts() const noexcept {
    return this->start;
  }
 private:
  Timepoint start;
};

static auto get_unix_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto elapse = std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch();
  return elapse.count();
}

} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_TIMER_H_
