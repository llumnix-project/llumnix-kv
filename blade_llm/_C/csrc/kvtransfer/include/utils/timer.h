#ifndef KVTRANSFER_INCLUDE_UTILS_TIMER_H_
#define KVTRANSFER_INCLUDE_UTILS_TIMER_H_

#pragma once
#include <chrono>

namespace blade_llm {
class TimeWatch {
 public:
  TimeWatch();
  size_t get_elapse_ms();
  size_t get_elapse_us();
 private:
  std::chrono::time_point<std::chrono::system_clock> start;
};

TimeWatch::TimeWatch() {
  start = std::chrono::system_clock::now();
}

size_t TimeWatch::get_elapse_ms() {
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return duration.count();
}

size_t TimeWatch::get_elapse_us() {
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return duration.count();
}
} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_TIMER_H_

