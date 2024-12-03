#include "utils/timer.h"
namespace blade_llm {
TimeWatch::TimeWatch() {
  start = std::chrono::steady_clock::now();
}

size_t TimeWatch::get_elapse_ms() {
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return duration.count();
}

size_t TimeWatch::get_elapse_us() {
  auto end = std::chrono::steady_clock::now();
  return elapse_us(start, end);
}

size_t TimeWatch::get_elapse_ns() {
  auto end = std::chrono::steady_clock::now();
  return elapse_ns(start, end);
}

}
