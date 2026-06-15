#include <gtest/gtest.h>
#include "common.h"
#include "thrid_party/logging.h"

using namespace blade_llm;

TEST(CommonTest, TestWorkerInfo) {
  WorkerInfo wi("1234", 4);
  wi.engine_tp_size = 8;
  wi.worker_tp_rank = 2;
  wi.block_sizes = {1024};
  wi.token_sizes = {128};
  wi.layer_num_blocks = 4;
  wi.num_layers = 2;
  wi.transfer_protocols = 0x01;
  wi.attn_kernel_blk_ntpb = 64;
  wi.indexer_blk_ntpb = 32;
  wi.addr = "127.0.0.1:9900";

  auto str = wi.to_string();
  LOG(INFO) << "get worker info str: " << str;
  auto wii = WorkerInfo::from_string(str);
  EXPECT_EQ(wi.inst_id, wii.inst_id);
  EXPECT_EQ(wi.worker_id, wii.worker_id);
  EXPECT_EQ(wi.engine_tp_size, wii.engine_tp_size);
  EXPECT_EQ(wi.worker_tp_rank, wii.worker_tp_rank);
  EXPECT_EQ(wi.block_sizes, wii.block_sizes);
  EXPECT_EQ(wi.token_sizes, wii.token_sizes);
  EXPECT_EQ(wi.layer_num_blocks, wii.layer_num_blocks);
  EXPECT_EQ(wi.num_layers, wii.num_layers);
  EXPECT_EQ(wi.transfer_protocols, wii.transfer_protocols);
  EXPECT_EQ(wi.attn_kernel_blk_ntpb, wii.attn_kernel_blk_ntpb);
  EXPECT_EQ(wi.indexer_blk_ntpb, wii.indexer_blk_ntpb);
  EXPECT_EQ(wi.addr, wii.addr);
}
