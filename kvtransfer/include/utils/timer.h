#ifndef KVTRANSFER_INCLUDE_UTILS_TIMER_H_
#define KVTRANSFER_INCLUDE_UTILS_TIMER_H_

#pragma once
#include <chrono>
#include <cassert>

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

static inline int64_t ielapse_us(Timepoint start, Timepoint end) {
  auto dur = end - start;
  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(dur);
  return dur_us.count();
}

static inline size_t elapse_us(Timepoint start, Timepoint end) {
  if (start >= end) {
    return 0;
  }
  int64_t ret = ielapse_us(start, end);
  assert(ret >= 0);
  return ret;
}

static inline size_t elapse_ns(Timepoint start, Timepoint end) {
  if (start >= end) {
    return 0;
  }
  auto dur = end - start;
  auto dur_ns = std::chrono::nanoseconds(dur);
  return dur_ns.count();
}

// Helper function to calculate elapsed time in microseconds for system_clock::time_point
static inline size_t elapse_us_system(
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end) {
  if (start >= end) {
    return 0;
  }
  auto dur = end - start;
  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(dur);
  int64_t ret = dur_us.count();
  assert(ret >= 0);
  return static_cast<size_t>(ret);
}

static inline std::chrono::system_clock::time_point microseconds_to_time_point(uint64_t us_timestamp) {
  int64_t us_signed = static_cast<int64_t>(us_timestamp);
  return std::chrono::system_clock::time_point(std::chrono::microseconds(us_signed));
}

// Convert system_clock::time_point to microseconds timestamp (since Unix epoch)
static inline uint64_t time_point_to_microseconds(std::chrono::system_clock::time_point tp) {
  auto dur = tp.time_since_epoch();
  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(dur);
  return static_cast<uint64_t>(dur_us.count());
}

static auto get_unix_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto elapse = std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch();
  return elapse.count();
}

} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_TIMER_H_
