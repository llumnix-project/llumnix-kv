#include "context.h"
#include "protocol/cuda_ipc.h"
#include <gtest/gtest.h>

using namespace blade_llm;

TEST(ContextTest, TestContextBasicMethod) {
  Context ctx("1", 0);
  ctx.set_tp(2, 0);
  ctx.set_layer_data_address(1, {0, 1024, 2048});
  ctx.set_block_params(KB, 256, 1);
  const auto w_info = ctx.worker_info();
  EXPECT_EQ(w_info.tp_size, 2);
  EXPECT_EQ(w_info.worker_tp_rank, 0);
  EXPECT_EQ(w_info.inst_id, "1");
  EXPECT_EQ(w_info.worker_id, 0);
  EXPECT_EQ(w_info.block_size, KB);
  EXPECT_EQ(w_info.token_size, 256);
  EXPECT_EQ(ctx.layer_num_blocks(), 1);
  EXPECT_EQ(ctx.num_layers(), 3);
  EXPECT_EQ(ctx.block_size(), KB);
  EXPECT_EQ(ctx.device_id(), 1);

  EXPECT_THROW(ctx.set_block_params(KB, 2* KB, 1), std::runtime_error);
  EXPECT_THROW(ctx.set_block_params(KB, 300, 1), std::runtime_error);
  EXPECT_THROW(ctx.set_cuda_barrier(nullptr), std::runtime_error);
}

TEST(ContextTest, TestCudaIpcContext) {
  Context ctx("1", 0);
  ctx.set_tp(2, 0);
  void *layer_0, *layer_1, *layer_2;
  cuda_malloc(&layer_0, KB);
  cuda_malloc(&layer_1, KB);
  cuda_malloc(&layer_2, KB);

  std::vector<uint64_t> layer_addrs(3);
  layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
  layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
  layer_addrs[2] = reinterpret_cast<uint64_t>(layer_2);
  ctx.set_layer_data_address(1, layer_addrs);
  ctx.set_block_params(KB, 256, 1);
  auto cuda_ctx = std::make_unique<CudaIpcContext>(ctx.device_id(), true);
  TransferProtocol proto = TransferProtocol::cuda_ipc();
  EXPECT_EQ(cuda_ctx->protocol().type, proto.type);
  EXPECT_TRUE(cuda_ctx->check_support());
  ctx.register_protocol(std::move(cuda_ctx));
  auto proto_ctx = ctx.get_protocol_ctx<CudaIpcContext>(proto);
  EXPECT_TRUE(proto_ctx != nullptr);
  EXPECT_EQ(proto_ctx->num_layers(), 3);
}



