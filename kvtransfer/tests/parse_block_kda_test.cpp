#include <gtest/gtest.h>

#include "parse_block_common.h"

namespace blade_llm {
namespace {

struct KdaParseFixture {
  WorkerInfo src{"src", 0};
  WorkerInfo dst{"dst", 0};
  std::shared_ptr<RequestInfo> request;

  KdaParseFixture(
      uint32_t p_tp,
      uint32_t p_rank,
      uint32_t d_tp,
      uint32_t d_rank,
      size_t p_block_size,
      size_t d_block_size,
      BlockIds src_blocks = {{1}},
      BlockIds dst_blocks = {{2}}) {
    src.engine_tp_size = p_tp;
    src.worker_tp_rank = p_rank;
    src.block_sizes = {p_block_size};
    src.token_sizes = {1};
    src.layer_num_blocks = 8;
    src.num_gdn_layers = 1;

    dst.engine_tp_size = d_tp;
    dst.worker_tp_rank = d_rank;
    dst.block_sizes = {d_block_size};
    dst.token_sizes = {1};
    dst.layer_num_blocks = 8;

    request = std::make_shared<RequestInfo>(
        "dst", 0, "kda-layout-test",
        std::move(src_blocks), std::move(dst_blocks));
  }

  void set_sd_layout(
      std::vector<size_t> conv_shape,
      std::vector<size_t> recurrent_shape,
    std::vector<size_t> channel_dims) {
    src.kda_conv_dim_first = false;
    src.conv_state_shape = std::move(conv_shape);
    src.ssm_state_shape = std::move(recurrent_shape);
    src.gdn_conv_channel_dims = std::move(channel_dims);
    src.gdn_conv_elem_size = 1;
    src.gdn_ssm_elem_size = 4;
  }

  void set_ds_layout(
      std::vector<size_t> conv_shape,
      std::vector<size_t> recurrent_shape,
      std::vector<size_t> channel_dims) {
    set_sd_layout(
        std::move(conv_shape),
        std::move(recurrent_shape),
        std::move(channel_dims));
    src.kda_conv_dim_first = true;
  }

  ReqSendTask last_task() const {
    return ReqSendTask(request, 0, 1, true);
  }
};

void expect_block(
    const IpcBlock& block,
    size_t src_offset,
    size_t dst_offset,
    size_t length) {
  EXPECT_EQ(block.src_offset, src_offset);
  EXPECT_EQ(block.dst_offset, dst_offset);
  EXPECT_EQ(block.length, length);
}

TEST(KdaParseBlockTest, PEqDCopiesTheFullPaddedPage) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/1,
      /*d_tp=*/2, /*d_rank=*/1,
      /*p_block_size=*/64, /*d_block_size=*/64);
  auto task = fixture.last_task();
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kda_block_send_p_eq_d(
      &fixture.src, &fixture.dst, {}, 2, 1, &task, send_blocks);

  ASSERT_EQ(send_blocks.size(), 1u);
  ASSERT_EQ(send_blocks.at(0).size(), 1u);
  expect_block(send_blocks.at(0).at(0), 64, 128, 64);
}

TEST(KdaParseBlockTest, PGtDMergesSdQkvRowsAndRecurrentHeads) {
  KdaParseFixture fixture(
      /*p_tp=*/4, /*p_rank=*/1,
      /*d_tp=*/2, /*d_rank=*/0,
      /*p_block_size=*/32, /*d_block_size=*/64);
  fixture.set_sd_layout(
      /*conv_shape=*/{8, 2, 6},
      /*recurrent_shape=*/{8, 1, 2, 2},
      /*channel_dims=*/{2, 2, 2});
  auto task = fixture.last_task();
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kda_block_send_p_gt_d(
      &fixture.src, &fixture.dst, {}, 0, 0, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 7u);
  expect_block(blocks.at(0), 32, 130, 2);  // row 0, Q
  expect_block(blocks.at(1), 34, 134, 2);  // row 0, K
  expect_block(blocks.at(2), 36, 138, 2);  // row 0, V
  expect_block(blocks.at(3), 38, 142, 2);  // row 1, Q
  expect_block(blocks.at(4), 40, 146, 2);  // row 1, K
  expect_block(blocks.at(5), 42, 150, 2);  // row 1, V
  expect_block(blocks.at(6), 44, 168, 16); // recurrent
}

TEST(KdaParseBlockTest, PGtDMergesDsQkvComponentsAndRecurrentHeads) {
  KdaParseFixture fixture(
      /*p_tp=*/4, /*p_rank=*/1,
      /*d_tp=*/2, /*d_rank=*/0,
      /*p_block_size=*/32, /*d_block_size=*/64);
  fixture.set_ds_layout(
      /*conv_shape=*/{8, 6, 2},
      /*recurrent_shape=*/{8, 1, 2, 2},
      /*channel_dims=*/{2, 2, 2});
  auto task = fixture.last_task();
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kda_block_send_p_gt_d(
      &fixture.src, &fixture.dst, {}, 0, 0, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 4u);
  expect_block(blocks.at(0), 32, 132, 4);  // Q
  expect_block(blocks.at(1), 36, 140, 4);  // K
  expect_block(blocks.at(2), 40, 148, 4);  // V
  expect_block(blocks.at(3), 44, 168, 16); // recurrent
}

TEST(KdaParseBlockTest, PLtDSplitsSdQkvRowsAndRecurrentHeads) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/0,
      /*d_tp=*/4, /*d_rank=*/1,
      /*p_block_size=*/64, /*d_block_size=*/32);
  fixture.set_sd_layout(
      /*conv_shape=*/{8, 2, 12},
      /*recurrent_shape=*/{8, 2, 2, 2},
      /*channel_dims=*/{4, 4, 4});
  auto task = fixture.last_task();
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kda_block_send_p_lt_d(
      &fixture.src, &fixture.dst, {}, 0, 0, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 7u);
  expect_block(blocks.at(0), 66, 64, 2);   // row 0, Q
  expect_block(blocks.at(1), 70, 66, 2);   // row 0, K
  expect_block(blocks.at(2), 74, 68, 2);   // row 0, V
  expect_block(blocks.at(3), 78, 70, 2);   // row 1, Q
  expect_block(blocks.at(4), 82, 72, 2);   // row 1, K
  expect_block(blocks.at(5), 86, 74, 2);   // row 1, V
  expect_block(blocks.at(6), 104, 76, 16); // recurrent
}

TEST(KdaParseBlockTest, PLtDSplitsDsQkvComponentsAndRecurrentHeads) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/0,
      /*d_tp=*/4, /*d_rank=*/1,
      /*p_block_size=*/64, /*d_block_size=*/32);
  fixture.set_ds_layout(
      /*conv_shape=*/{8, 12, 2},
      /*recurrent_shape=*/{8, 2, 2, 2},
      /*channel_dims=*/{4, 4, 4});
  auto task = fixture.last_task();
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kda_block_send_p_lt_d(
      &fixture.src, &fixture.dst, {}, 0, 0, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 4u);
  expect_block(blocks.at(0), 68, 64, 4);   // Q
  expect_block(blocks.at(1), 76, 68, 4);   // K
  expect_block(blocks.at(2), 84, 72, 4);   // V
  expect_block(blocks.at(3), 104, 76, 16); // recurrent
}

TEST(KdaParseBlockTest, DoesNotSendMutableStateBeforeRequestFinishes) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/0,
      /*d_tp=*/2, /*d_rank=*/0,
      /*p_block_size=*/64, /*d_block_size=*/64);
  ReqSendTask task(fixture.request, 0, 1, false);
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kda_block_send_p_eq_d(
      &fixture.src, &fixture.dst, {}, 2, 0, &task, send_blocks);

  ASSERT_EQ(send_blocks.size(), 1u);
  EXPECT_TRUE(send_blocks.at(0).empty());
}

void configure_kimi_mla(
    KdaParseFixture& fixture,
    uint32_t kernel_ntpb = 4,
    uint32_t manager_ntpb = 0) {
  if (manager_ntpb == 0) {
    manager_ntpb = kernel_ntpb;
  }
  fixture.src.attn_kernel_blk_ntpb = kernel_ntpb;
  fixture.dst.attn_kernel_blk_ntpb = kernel_ntpb;
  fixture.src.hybrid_attn_token_size = 2;
  fixture.dst.hybrid_attn_token_size = 2;
  ASSERT_EQ(fixture.src.block_sizes.at(0) % manager_ntpb, 0u);
  ASSERT_EQ(fixture.dst.block_sizes.at(0) % manager_ntpb, 0u);
  fixture.src.token_sizes.at(0) =
      fixture.src.block_sizes.at(0) / manager_ntpb;
  fixture.dst.token_sizes.at(0) =
      fixture.dst.block_sizes.at(0) / manager_ntpb;
  fixture.src.kda_page_stride = fixture.src.block_sizes.at(0);
  fixture.dst.kda_page_stride = fixture.dst.block_sizes.at(0);
}

TEST(KimiK3ParseBlockTest, PEqDCopiesMlaPrefixAndKdaPage) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/1,
      /*d_tp=*/2, /*d_rank=*/1,
      /*p_block_size=*/64, /*d_block_size=*/64,
      /*src_blocks=*/{{1}, {3}}, /*dst_blocks=*/{{2}, {4}});
  configure_kimi_mla(fixture);
  auto task = fixture.last_task();
  std::bitset<MAX_TP_SIZE> valid_ranks;
  valid_ranks.set(0).set(1);
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kimi_k3_mla_block_send_p_eq_d(
      &fixture.src, &fixture.dst, valid_ranks, 2, 1, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 2u);
  expect_block(blocks.at(0), 3 * 64, 4 * 64, 8);  // replicated MLA
  expect_block(blocks.at(1), 1 * 64, 2 * 64, 64); // full KDA page
}

TEST(KimiK3ParseBlockTest, PEqDUsesVllmManagerBlockGranularity) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/1,
      /*d_tp=*/2, /*d_rank=*/1,
      /*p_block_size=*/64, /*d_block_size=*/64,
      /*src_blocks=*/{{1}, {3, 5}}, /*dst_blocks=*/{{2}, {4, 6}});
  configure_kimi_mla(
      fixture, /*kernel_ntpb=*/2, /*manager_ntpb=*/4);
  std::bitset<MAX_TP_SIZE> valid_ranks;
  valid_ranks.set(0).set(1);

  // Completing one attention kernel block must not send half of the vLLM
  // manager block in equal-TP mode.
  ReqSendTask kernel_boundary_task(fixture.request, 0, 2, false);
  std::vector<std::vector<IpcBlock>> boundary_blocks;
  parse_kimi_k3_mla_block_send_p_eq_d(
      &fixture.src, &fixture.dst, valid_ranks, 2, 1,
      &kernel_boundary_task, boundary_blocks);
  ASSERT_EQ(boundary_blocks.size(), 1u);
  EXPECT_TRUE(boundary_blocks.at(0).empty());

  // At request completion, send the whole four-token manager block, including
  // its unused tail, plus every extra P-side block that vLLM pre-zeroed.
  ReqSendTask final_task(fixture.request, 2, 1, true);
  std::vector<std::vector<IpcBlock>> final_blocks;
  parse_kimi_k3_mla_block_send_p_eq_d(
      &fixture.src, &fixture.dst, valid_ranks, 2, 1,
      &final_task, final_blocks);
  const auto& blocks = final_blocks.at(0);
  ASSERT_EQ(blocks.size(), 3u);
  expect_block(blocks.at(0), 3 * 64, 4 * 64, 8);  // full vLLM MLA block
  expect_block(blocks.at(1), 5 * 64, 6 * 64, 8);  // extra zeroed MLA block
  expect_block(blocks.at(2), 1 * 64, 2 * 64, 64); // full KDA page
}

TEST(KimiK3ParseBlockTest, PGtDMergesKdaAndCopiesMlaFromRepresentative) {
  KdaParseFixture fixture(
      /*p_tp=*/4, /*p_rank=*/0,
      /*d_tp=*/2, /*d_rank=*/0,
      /*p_block_size=*/32, /*d_block_size=*/64,
      /*src_blocks=*/{{1}, {3}}, /*dst_blocks=*/{{2}, {4}});
  fixture.set_sd_layout(
      /*conv_shape=*/{8, 2, 6},
      /*recurrent_shape=*/{8, 1, 2, 2},
      /*channel_dims=*/{2, 2, 2});
  configure_kimi_mla(fixture);
  auto task = fixture.last_task();
  std::bitset<MAX_TP_SIZE> valid_ranks;
  valid_ranks.set(0).set(2);
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kimi_k3_mla_block_send_p_gt_d(
      &fixture.src, &fixture.dst, valid_ranks, 2, 0, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 9u);
  expect_block(blocks.at(0), 3 * 32, 4 * 64, 8); // replicated MLA
  // Fill the remaining 3 D tokens (6 bytes) from P request block 0.
  expect_block(blocks.at(1), 3 * 32, 4 * 64 + 2, 6);
  expect_block(blocks.back(), 32 + 12, 2 * 64 + 24, 16); // recurrent
}

TEST(KimiK3ParseBlockTest, PGtDSkipsReplicatedMlaOnNonRepresentative) {
  KdaParseFixture fixture(
      /*p_tp=*/4, /*p_rank=*/1,
      /*d_tp=*/2, /*d_rank=*/0,
      /*p_block_size=*/32, /*d_block_size=*/64,
      /*src_blocks=*/{{1}, {3}}, /*dst_blocks=*/{{2}, {4}});
  fixture.set_sd_layout(
      /*conv_shape=*/{8, 2, 6},
      /*recurrent_shape=*/{8, 1, 2, 2},
      /*channel_dims=*/{2, 2, 2});
  configure_kimi_mla(fixture);
  auto task = fixture.last_task();
  std::bitset<MAX_TP_SIZE> valid_ranks;
  valid_ranks.set(0).set(2);
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kimi_k3_mla_block_send_p_gt_d(
      &fixture.src, &fixture.dst, valid_ranks, 2, 1, &task, send_blocks);

  ASSERT_EQ(send_blocks.at(0).size(), 7u);
  for (const auto& block : send_blocks.at(0)) {
    EXPECT_NE(block.src_offset, 3 * 32);
  }
}

TEST(KimiK3ParseBlockTest, PLtDSplitsKdaAndFansOutMla) {
  KdaParseFixture fixture(
      /*p_tp=*/2, /*p_rank=*/0,
      /*d_tp=*/4, /*d_rank=*/1,
      /*p_block_size=*/64, /*d_block_size=*/32,
      /*src_blocks=*/{{1}, {3}}, /*dst_blocks=*/{{2}, {4}});
  fixture.set_sd_layout(
      /*conv_shape=*/{8, 2, 12},
      /*recurrent_shape=*/{8, 2, 2, 2},
      /*channel_dims=*/{4, 4, 4});
  configure_kimi_mla(fixture);
  auto task = fixture.last_task();
  std::bitset<MAX_TP_SIZE> valid_ranks;
  valid_ranks.set(0).set(1);
  std::vector<std::vector<IpcBlock>> send_blocks;

  parse_kimi_k3_mla_block_send_p_lt_d(
      &fixture.src, &fixture.dst, valid_ranks, 2, 0, &task, send_blocks);

  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 9u);
  expect_block(blocks.at(0), 3 * 64, 4 * 32, 8); // replicated MLA
  // Fill the remaining 3 D tokens (6 bytes) from P request block 0.
  expect_block(blocks.at(1), 3 * 64, 4 * 32 + 2, 6);
  expect_block(blocks.back(), 64 + 40, 2 * 32 + 12, 16); // recurrent
}

}  // namespace
}  // namespace blade_llm
