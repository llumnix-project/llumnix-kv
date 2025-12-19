#include "context.h"
#include <gtest/gtest.h>
#include "thrid_party/logging.h"

using namespace blade_llm;

TEST(ContextTest, TestContextBasicMethod) {
  Context ctx("1", 0);
  ctx.set_tp(2, 0);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    // LayerInfo: token size, block size, layer address
    {LayerInfo(256, KB, 0)}, {LayerInfo(256, KB, 1024)}, {LayerInfo(256, KB, 2048)}
  };

  ctx.set_layer_info(1, all_layer_infos);
  ctx.set_block_params({KB}, {256}, 1);
  const auto w_info = ctx.worker_info();
  EXPECT_EQ(w_info.tp_size, 2);
  EXPECT_EQ(w_info.worker_tp_rank, 0);
  EXPECT_EQ(w_info.inst_id, "1");
  EXPECT_EQ(w_info.worker_id, 0);
  EXPECT_EQ(w_info.block_sizes, std::vector<size_t>{KB});
  EXPECT_EQ(w_info.token_sizes, std::vector<size_t>{256});
  EXPECT_EQ(ctx.layer_num_blocks(), 1);
  EXPECT_EQ(ctx.num_layers(), 3);
  EXPECT_EQ(ctx.block_sizes(), std::vector<size_t>{KB});
  EXPECT_EQ(ctx.device_id(), 1);

  // assert failures cause program termination, use EXPECT_DEATH to test
  EXPECT_DEATH(ctx.set_block_params({KB}, {2* KB}, 1), ".*");
  EXPECT_DEATH(ctx.set_block_params({KB}, {300}, 1), ".*");
  EXPECT_THROW(ctx.set_cuda_barrier(nullptr), std::runtime_error);
}