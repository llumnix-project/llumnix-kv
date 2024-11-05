#include "utils/block_queue.h"
#include <gtest/gtest.h>
#include <thread>

using namespace blade_llm;

TEST(BlockQueueTest, TestBlockQueue) {
  BlockingQueue<std::string> q ;
  std::thread t([&]() {
    while (!q.is_closed()) {
      std::string s;
      q.pop(s);
      EXPECT_EQ(s, "hello");
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  q.push("hello");
  q.close();
  t.join();
}