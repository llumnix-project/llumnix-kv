
#ifndef KVTRANSFER_INCLUDE_UTILS_SEMAPHORE_H_
#define KVTRANSFER_INCLUDE_UTILS_SEMAPHORE_H_
#pragma once

#include <condition_variable>
#include <mutex>

namespace blade_llm {
static size_t const cacheline_size = 64;
typedef char cacheline_pad_t[cacheline_size];

class SyncSemaphore {
 public:
  SyncSemaphore() = default;
  void wait(uint32_t cond);
  void release();
  // return new value.
  uint32_t release(uint32_t cond);
 private:
  cacheline_pad_t pad0_{};
  uint32_t ready_{0};
  std::condition_variable cv_;
  cacheline_pad_t pad1_{};
  std::mutex c_mutex_;
  cacheline_pad_t pad2_{};
};
} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_SEMAPHORE_H_
