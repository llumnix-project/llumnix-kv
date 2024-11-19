#include <gtest/gtest.h>
#include "utils/base64.h"

using namespace blade_llm;
TEST(UtilsTest, TestBase64) {
  std::vector<uint8_t> binary{0, 1, 2, 3, 4, 5, 4, 3, 2, 1};
  auto str = base64_encode(binary);
  auto decode = base64_decode(str);
  EXPECT_EQ(binary, decode);
}
