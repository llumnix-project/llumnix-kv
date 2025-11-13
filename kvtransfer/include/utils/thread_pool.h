#ifndef KVTRANSFER_INCLUDE_UTILS_THREAD_POOL_H_
#define KVTRANSFER_INCLUDE_UTILS_THREAD_POOL_H_

#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace blade_llm {

static constexpr int THD_NAME_MAX = 15;

static inline void set_thread_name(std::string& name) {
  if (name.size() > THD_NAME_MAX) {
    name.resize(THD_NAME_MAX);
  }
  int ret = pthread_setname_np(pthread_self(), name.c_str());
  (void)ret;
  return;
}

class ThreadPool {
 public:
  ThreadPool(size_t);
  template<class F, class... Args>
  auto spawn(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

  auto unsafe_size() const noexcept {
    return this->tasks_.size();
  }
 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()> > tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

inline ThreadPool::ThreadPool(size_t threads) : stop_(false) {
  for (size_t i = 0; i < threads; ++i)
    workers_.emplace_back(
        [this] {
          for (;;) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(this->queue_mutex_);
              this->condition_.wait(lock,
                                    [this] { return this->stop_ || !this->tasks_.empty(); });
              if (this->stop_ && this->tasks_.empty()) {
                return;
              }
              task = std::move(this->tasks_.front());
              this->tasks_.pop();
            }
            task();
          }
        }
    );
}

template<class F, class... Args>
auto ThreadPool::spawn(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;
  auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
  );
  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_)
      throw std::runtime_error("spawn on stopped ThreadPool");
    tasks_.emplace([task]() { (*task)(); });
  }
  condition_.notify_one();
  return res;
}

inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (std::thread &worker : workers_)
    worker.join();
}
} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_THREAD_POOL_H_
