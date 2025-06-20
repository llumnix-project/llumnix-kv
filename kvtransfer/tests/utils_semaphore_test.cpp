#include <gtest/gtest.h>
#include <thread>
#include "utils/semaphore.h"

using namespace blade_llm;

TEST(UtilsTest, TestSyncSemaphore) {
  SyncSemaphore signal;
  volatile int cond = 0;
  std::thread t0([&]() {
    cond ++;
    signal.release();
  });
  signal.wait(0);
  EXPECT_EQ(cond, 1);
  t0.join();

  int expect = 8;
  std::thread t1([&]() {
    cond = expect + 1;
    signal.release(cond);
  });
  signal.wait(expect);
  EXPECT_EQ(cond, expect + 1);
  t1.join();
}
