#include "context.h"
#include <gtest/gtest.h>

using namespace blade_llm;

TEST(ContextTest, TestContextBasicMethod) {
  Context ctx(1, 0);
  ctx.set_tp(2, 0);
  ctx.set_transfer_type(TransferType::CUDA_IPC);
  ctx.set_layer_data_address(1, {0, 1024, 2048});
  ctx.set_block_params(KB, 256, 1);
  const auto w_info = ctx.worker_info();
  EXPECT_EQ(w_info.tp_size, 2);
  EXPECT_EQ(w_info.worker_tp_rank, 0);
  EXPECT_EQ(w_info.inst_id, 1);
  EXPECT_EQ(w_info.worker_id, 0);
  EXPECT_EQ(w_info.block_size, KB);
  EXPECT_EQ(w_info.token_size, 256);
  EXPECT_EQ(ctx.transfer_type(), TransferType::CUDA_IPC);
  EXPECT_EQ(ctx.layer_num_blocks(), 1);
  EXPECT_EQ(ctx.num_layers(), 3);
  EXPECT_EQ(ctx.block_size(), KB);
  EXPECT_EQ(ctx.device_id(), 1);

  EXPECT_THROW(ctx.set_block_params(KB, 2* KB, 1), std::runtime_error);
  EXPECT_THROW(ctx.set_block_params(KB, 300, 1), std::runtime_error);
  EXPECT_THROW(ctx.set_cuda_barrier(nullptr), std::runtime_error);
}
