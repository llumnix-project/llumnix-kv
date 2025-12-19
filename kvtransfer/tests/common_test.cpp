#include <gtest/gtest.h>
#include "common.h"
#include "thrid_party/logging.h"

using namespace blade_llm;

TEST(CommonTest, TestWorkerInfo) {
  WorkerInfo wi("1234", 4);
  wi.tp_size = 8;
  wi.worker_tp_rank = 2;
  wi.block_sizes = {1024};
  wi.token_sizes = {128};
  wi.layer_num_blocks = 4;
  wi.num_layers = 2;
  wi.transfer_protocols = 0x01;
  wi.addr = "127.0.0.1:9900";
  wi.other_info = std::vector<uint8_t>{1, 2, 3, 4, 5};

  auto binary = wi.to_bytes();
  auto wii = WorkerInfo::from_bytes(binary.data(), binary.size());
  EXPECT_EQ(wi.inst_id, wii.inst_id);
  EXPECT_EQ(wi.worker_id, wii.worker_id);
  EXPECT_EQ(wi.tp_size, wii.tp_size);
  EXPECT_EQ(wi.worker_tp_rank, wii.worker_tp_rank);
  EXPECT_EQ(wi.block_sizes, wii.block_sizes);
  EXPECT_EQ(wi.token_sizes, wii.token_sizes);
  EXPECT_EQ(wi.layer_num_blocks, wii.layer_num_blocks);
  EXPECT_EQ(wi.num_layers, wii.num_layers);
  EXPECT_EQ(wi.transfer_protocols, wii.transfer_protocols);
  EXPECT_EQ(wi.addr, wii.addr);
  EXPECT_EQ(wi.other_info, wii.other_info);

  auto str = wi.to_string();
  LOG(INFO) << "get worker info str: " << str;
  auto wiii = WorkerInfo::from_string(str);
  EXPECT_EQ(wi.inst_id, wiii.inst_id);
  EXPECT_EQ(wi.worker_id, wiii.worker_id);
  EXPECT_EQ(wi.tp_size, wiii.tp_size);
  EXPECT_EQ(wi.worker_tp_rank, wiii.worker_tp_rank);
  EXPECT_EQ(wi.block_sizes, wiii.block_sizes);
  EXPECT_EQ(wi.token_sizes, wiii.token_sizes);
  EXPECT_EQ(wi.layer_num_blocks, wiii.layer_num_blocks);
  EXPECT_EQ(wi.num_layers, wiii.num_layers);
  EXPECT_EQ(wi.transfer_protocols, wiii.transfer_protocols);
  EXPECT_EQ(wi.addr, wiii.addr);
  EXPECT_EQ(wi.other_info, wiii.other_info);
}
