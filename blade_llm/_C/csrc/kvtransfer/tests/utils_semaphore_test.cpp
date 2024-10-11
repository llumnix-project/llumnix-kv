#include <gtest/gtest.h>
#include <thread>
#include "utils/semaphore.h"

using namespace blade_llm;

TEST(UtilsTest, TestSyncSemaphore) {
  SyncSemaphore signal;
  int cond = 0;
  int expect = 8;
  std::thread t([&]() {
    cond = expect + 1;
    signal.release(cond);
  });
  signal.wait(expect);
  EXPECT_EQ(cond, expect + 1);
  t.join();
}
