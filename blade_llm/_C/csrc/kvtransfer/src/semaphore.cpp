#include "utils/semaphore.h"
#include <assert.h>

namespace blade_llm {
void SyncSemaphore::wait(uint32_t cond) {
  std::unique_lock<std::mutex> lc(c_mutex_);
  if (ready_ <= cond) {
    cv_.wait(lc, [this, &cond] { return this->ready_ > cond; });
  }
}

void SyncSemaphore::release() {
  std::unique_lock<std::mutex> lc(c_mutex_);
  ready_++;
  cv_.notify_all();
}

uint32_t SyncSemaphore::release(uint32_t cond) {
  std::unique_lock<std::mutex> lc(c_mutex_);
  if (ready_ < cond) {
    ready_ = cond;
    cv_.notify_all();
  }
  return ready_;
}

}
