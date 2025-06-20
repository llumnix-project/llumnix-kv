#ifndef KVTRANSFER_INCLUDE_UTILS_BLOCK_QUEUE_H_
#define KVTRANSFER_INCLUDE_UTILS_BLOCK_QUEUE_H_

#pragma once
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <cassert>
#include "thrid_party/logging.h"

namespace blade_llm {
template<typename T>
class BlockingQueue {
 public:
  BlockingQueue() = default;

  void push(T &&t) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.emplace(std::move(t));
    cond_.notify_one();
  }

  void pop(T &t) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty() && !closed_.load(std::memory_order_relaxed)) {
      cond_.wait(lock);
    }

    if (!queue_.empty()) {
      t = std::move(queue_.front());
      queue_.pop();
    }
  }

  bool is_closed() {
    return closed_.load(std::memory_order_relaxed);
  }

  void close() {
    std::unique_lock<std::mutex> lock(mutex_);
    closed_.store(true, std::memory_order_release);
    cond_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<T> queue_;
  std::atomic_bool closed_{false};
};
} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_BLOCK_QUEUE_H_
