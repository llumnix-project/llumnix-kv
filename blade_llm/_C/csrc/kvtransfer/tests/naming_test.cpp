
#include <gtest/gtest.h>
#include "naming.h"
#include "naming/shm_naming.h"
#include "context.h"

using namespace blade_llm;

TEST(NamingTest, TestShmNaming) {
  const auto naming_url = "shm:shm_naming_test";
  auto n = create_shm_naming("shm_naming_test");
  auto info = n->get_worker_info(1, 1);
  EXPECT_FALSE(info.has_value());
  Context ctx(1, 1);
  ctx.set_tp(2, 0);
  ctx.set_block_params(512, 128, 8);
  n->register_worker(ctx.worker_info());
  connect_naming(naming_url);
  auto info_opt = naming()->get_worker_info(1, 1);
  EXPECT_TRUE(info_opt.has_value());
  EXPECT_EQ(1, info_opt->inst_id);
  EXPECT_EQ(1, info_opt->worker_id);
  EXPECT_EQ(2, info_opt->tp_size);
  EXPECT_EQ(0, info_opt->worker_tp_rank);
  EXPECT_EQ(512, info_opt->block_size);
  EXPECT_EQ(128, info_opt->token_size);
}