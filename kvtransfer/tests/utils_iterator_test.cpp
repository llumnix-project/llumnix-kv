
#include <gtest/gtest.h>
#include "utils/iterator.h"

using namespace blade_llm;

TEST(UtilsTest, TestIteratorSource) {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7};
  auto iter = Source<int>::from(nums);
  for (size_t i = 0; i < nums.size(); ++i) {
    auto opt = iter->next();
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(*opt.value(), i + 1);
    *opt.value() = 0;
  }
  for (size_t i = 0; i < nums.size(); ++i) {
    EXPECT_EQ(nums[i], 0);
  }
}

TEST(UtilsTest, TestIteratorFilter) {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7};
  auto src = Source<int>::from(nums);
  Filter<int *> filter(src, [](int *const &x) { return (*x) % 2 == 0; });
  for (size_t i = 0; i < 3; ++i) {
    auto opt = filter.next();
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(*opt.value(), 2 * i + 2);
  }
  EXPECT_EQ(7, nums.size());
}

TEST(UtilsTest, TestIteratorMap) {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7};
  auto src = Source<int>::from(nums);
  Map<int *, int> map(src, [](int *const &x) { return (*x) * 2; });
  for (size_t i = 0; i < nums.size(); ++i) {
    auto opt = map.next();
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), 2 * i + 2);
    EXPECT_EQ(nums[i], i + 1);
  }
}

TEST(UtilsTest, TestIteratorChain) {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7};
  Iterator<int *> src(Source<int>::from(nums));
  auto iter = src
      .filter([](int *const &x) { return *x % 2 == 0; })
      .map<int>([](int *const &x) { return (*x) * 2; });
  for (size_t i = 0; i < 3; ++i) {
    auto opt = iter->next();
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), (2 * i + 2) * 2);
  }
}

TEST(UtilsTest, TestIteratorItemCopy) {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7};
  auto iter = CopySource<int>::from(nums);
  for (size_t i = 0; i < nums.size(); ++i) {
    auto opt = iter->next();
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), i + 1);
    opt.value() = 0;
  }
  for (size_t i = 0; i < nums.size(); ++i) {
    EXPECT_EQ(nums[i], i + 1);
}
}
