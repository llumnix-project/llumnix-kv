#include "parse_block_common.h"
#include "envcfg.h"
#include "thrid_party/logging.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

static void do_parse_block_aligned_send(
  size_t block_size,
  const ReqSendTask *task,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t ntpb,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;
  while (left_tokens > 0) {
    const auto blk_idx = wrote_tokens / ntpb;
    const auto token_idx = wrote_tokens % ntpb;
    assert(blk_idx < src_blocks.size() && blk_idx < dst_blocks.size());
    const size_t p_blk_off = src_blocks.at(blk_idx) * block_size;
    const size_t d_blk_off = dst_blocks.at(blk_idx) * block_size;
    auto tokens = std::min(ntpb - token_idx, left_tokens);
    size_t length = 0;
    if (tokens != ntpb){
      if (task->reach_last_token){
        length = block_size;
      } else if (token_idx == 0) {
        length = 0;
      } else {
        if (token_idx + tokens == ntpb) {
          length = block_size;
        } else {
          assert(token_idx + tokens < ntpb);
          length = 0;
        }
      }
    } else {
      length = block_size;
    }
    if (length > 0) {
      per_cache_send_blocks.emplace_back(p_blk_off, d_blk_off, length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

static void parse_hybrid_attn_block(
  uint32_t src_ntpb,
  uint32_t dst_ntpb,
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t attn_wrote_tokens,
  uint32_t attn_left_tokens,
  uint32_t attn_group_off,
  uint32_t src_attn_kernel_blk_ntpb,
  uint32_t dst_attn_kernel_blk_ntpb,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  const auto p_kernel_blk_size = src_attn_kernel_blk_ntpb * p_token_size;
  const auto d_kernel_blk_size = dst_attn_kernel_blk_ntpb * d_token_size;

  auto wrote_tokens = attn_wrote_tokens;
  auto left_tokens = attn_left_tokens;

  while (left_tokens > 0) {
    const auto p_blk_idx = wrote_tokens / src_ntpb;
    const auto p_kernel_blk_idx = (wrote_tokens - p_blk_idx * src_ntpb) / src_attn_kernel_blk_ntpb;
    const auto p_token_idx_base = wrote_tokens % src_attn_kernel_blk_ntpb;
    const auto d_blk_idx = wrote_tokens / dst_ntpb;
    const auto d_kernel_blk_idx = (wrote_tokens - d_blk_idx * dst_ntpb) / dst_attn_kernel_blk_ntpb;
    const auto d_token_idx_base = wrote_tokens % dst_attn_kernel_blk_ntpb;

    assert(p_blk_idx < src_blocks.size() && d_blk_idx < dst_blocks.size());
    const size_t p_blk_off = src_blocks.at(p_blk_idx) * p_block_size;
    const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;
    const size_t p_kernel_blk_off = p_blk_off + p_kernel_blk_idx * p_kernel_blk_size;
    const size_t d_kernel_blk_off = d_blk_off + d_kernel_blk_idx * d_kernel_blk_size;

    const auto tokens = std::min<uint32_t>(
      {src_attn_kernel_blk_ntpb - p_token_idx_base, dst_attn_kernel_blk_ntpb - d_token_idx_base, left_tokens}
    );
    const auto token_step_length = p_token_size / 2;
    assert(token_step_length * 2 == p_token_size);
    for (uint32_t idx = 0; idx < tokens; ++idx) {
      const uint32_t p_token_idx = p_token_idx_base + idx;
      const uint32_t d_token_idx = d_token_idx_base + idx;
      const size_t pk_token_off = p_kernel_blk_off + p_token_idx * token_step_length;
      const size_t pv_token_off = pk_token_off + p_kernel_blk_size/2;
      const size_t d_token_off = d_kernel_blk_off + d_token_idx * d_token_size / 2;
      const size_t dk_token_off = d_token_off + attn_group_off * token_step_length;
      const size_t dv_token_off = dk_token_off + d_kernel_blk_size/2;
      per_cache_send_blocks.emplace_back(pk_token_off, dk_token_off, token_step_length);
      per_cache_send_blocks.emplace_back(pv_token_off, dv_token_off, token_step_length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

static void parse_hybrid_indexer_block(
  size_t p_block_size,
  size_t d_block_size,
  uint32_t indexer_wrote_tokens,
  uint32_t indexer_left_tokens,
  const std::vector<uint32_t>& src_indexer_blocks,
  const std::vector<uint32_t>& dst_indexer_blocks,
  uint32_t p_indexer_ntpb,
  uint32_t d_indexer_ntpb,
  uint32_t indexer_token_size,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(indexer_token_size > 0);

  auto wrote_tokens = indexer_wrote_tokens;
  auto left_tokens = indexer_left_tokens;
  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / p_indexer_ntpb;
    auto src_token_idx = wrote_tokens % p_indexer_ntpb;
    auto dst_block_idx = wrote_tokens / d_indexer_ntpb;
    auto dst_token_idx = wrote_tokens % d_indexer_ntpb;
    assert(src_block_idx < src_indexer_blocks.size());
    assert(dst_block_idx < dst_indexer_blocks.size());
    auto tokens = std::min<uint32_t>({
      p_indexer_ntpb - src_token_idx, d_indexer_ntpb - dst_token_idx, left_tokens
    });
    auto src_offset = src_indexer_blocks.at(src_block_idx) * p_block_size
                    + src_token_idx * indexer_token_size;
    auto dst_offset = dst_indexer_blocks.at(dst_block_idx) * d_block_size
                    + dst_token_idx * indexer_token_size;
    auto length = tokens * indexer_token_size;
    per_cache_send_blocks.emplace_back(src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

static void do_parse_hybrid_block_send_p_gt_d(
  size_t p_token_size, size_t d_token_size,
  uint32_t src_ntpb,
  uint32_t dst_ntpb,
  uint32_t attn_group_n,
  uint32_t attn_group_off,
  uint32_t gdn_group_n,
  uint32_t gdn_group_off,
  const std::bitset<MAX_TP_SIZE>& validranks,
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  const size_t p_block_size = p_token_size * src_ntpb;
  const size_t d_block_size = d_token_size * dst_ntpb;
  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const auto p_indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  const auto d_indexer_ntpb = dst_worker_info->indexer_blk_ntpb;

  auto p_tp_rank = src_worker_info->worker_tp_rank;
  auto src_attn_kernel_blk_ntpb = src_worker_info->attn_kernel_blk_ntpb;
  auto dst_attn_kernel_blk_ntpb = dst_worker_info->attn_kernel_blk_ntpb;

  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;

  auto indexer_wrote_tokens = wrote_tokens;
  auto indexer_left_tokens = left_tokens;

  auto attn_wrote_tokens = wrote_tokens;
  auto attn_left_tokens = left_tokens;

  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  assert(num_gdn_layers < src_blocks.size());
  assert(num_gdn_layers < dst_blocks.size());
  assert(!src_worker_info->conv_state_shape.empty());
  assert(!src_worker_info->ssm_state_shape.empty());
  assert(!src_worker_info->gdn_conv_channel_dims.empty());

  if (task->reach_last_token){
    const auto &conv_state_shape = src_worker_info->conv_state_shape;
    const auto &ssm_state_shape = src_worker_info->ssm_state_shape;
    uint32_t gdn_conv_elem_size = src_worker_info->gdn_conv_elem_size;
    uint32_t gdn_ssm_elem_size = src_worker_info->gdn_ssm_elem_size;
    const auto &conv_channel_dims = src_worker_info->gdn_conv_channel_dims;

    const auto p_conv_dim_size = conv_state_shape.at(2) * gdn_conv_elem_size;
    const auto p_conv_block_size = conv_state_shape.at(1) * p_conv_dim_size;
    const auto p_ssm_block_size = ssm_state_shape.at(1) * ssm_state_shape.at(2) * ssm_state_shape.at(3) * gdn_ssm_elem_size;
    const auto d_conv_dim_size = gdn_group_n * p_conv_dim_size;

    for (size_t group_idx = 0; group_idx < num_gdn_layers; ++group_idx) {
      const auto& src_gdn_blks = src_blocks.at(group_idx);
      const auto& dst_gdn_blks = dst_blocks.at(group_idx);
      for (size_t gdn_blk_idx = 0; gdn_blk_idx < src_gdn_blks.size(); ++gdn_blk_idx) {
        const size_t p_gdn_off = src_gdn_blks.at(gdn_blk_idx) * p_block_size;
        const size_t d_gdn_off = dst_gdn_blks.at(gdn_blk_idx) * d_block_size;
        for (size_t conv_dim = 0; conv_dim < conv_state_shape.at(1); ++conv_dim) {
          size_t conv_dim_inner_offset = 0;
          for (size_t conv_channel_dim = 0; conv_channel_dim < conv_channel_dims.size(); ++conv_channel_dim) {
            const auto conv_step_length = conv_channel_dims.at(conv_channel_dim) * gdn_conv_elem_size;
            const size_t p_conv_off = p_gdn_off + conv_dim * p_conv_dim_size + conv_dim_inner_offset;
            const size_t d_conv_off = d_gdn_off
                      + conv_dim * d_conv_dim_size
                      + gdn_group_n * conv_dim_inner_offset
                      + gdn_group_off * conv_step_length;
            per_cache_send_blocks.emplace_back(p_conv_off, d_conv_off, conv_step_length);
            conv_dim_inner_offset += conv_step_length;
          }
          assert(conv_dim_inner_offset == p_conv_dim_size);
        }
        const size_t p_ssm_off = p_gdn_off + p_conv_block_size;
        const size_t d_ssm_off = d_gdn_off + gdn_group_n * p_conv_block_size + gdn_group_off * p_ssm_block_size;
        per_cache_send_blocks.emplace_back(p_ssm_off, d_ssm_off, p_ssm_block_size);
      }
    }
  }
  auto attn_group_offset = num_gdn_layers;
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());
  assert((p_indexer_ntpb == 0 && d_indexer_ntpb == 0) || (p_indexer_ntpb > 0 && d_indexer_ntpb > 0));

  if (p_indexer_ntpb > 0 and d_indexer_ntpb > 0) {
    if (p_tp_rank == 0) {
      const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
      const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
      parse_hybrid_indexer_block(
        p_block_size,
        d_block_size,
        indexer_wrote_tokens,
        indexer_left_tokens,
        src_indexer_blks,
        dst_indexer_blks,
        p_indexer_ntpb,
        d_indexer_ntpb,
        src_worker_info->hybrid_indexer_token_size,
        per_cache_send_blocks
      );
    }
    attn_group_offset += 1;
  }
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());
  if (validranks[p_tp_rank]) {
    const auto& src_attn_blks = src_blocks.at(attn_group_offset);
    const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);
    parse_hybrid_attn_block(
      src_ntpb,
      dst_ntpb,
      p_token_size,
      d_token_size,
      p_block_size,
      d_block_size,
      src_attn_blks,
      dst_attn_blks,
      attn_wrote_tokens,
      attn_left_tokens,
      attn_group_off,
      src_attn_kernel_blk_ntpb,
      dst_attn_kernel_blk_ntpb,
      per_cache_send_blocks
    );
  }
}

// QWEN3_NEXT_FLASH_CACHE_SHAPE: P == D (block aligned)
void vllm_parse_hybrid_block_send_p_eq_d_block_aligned(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  assert(src_worker_info->kvt_tp_size == dst_worker_info->kvt_tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  auto const& kv_token_sizes = src_worker_info->token_sizes;
  auto const& kv_block_sizes = src_worker_info->block_sizes;
  assert(kv_token_sizes == dst_worker_info->token_sizes);
  assert(kv_block_sizes == dst_worker_info->block_sizes);
  assert(kv_token_sizes.size() == kv_block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());
  assert(kv_token_sizes.size() == 1);
  assert(kv_block_sizes.size() == 1);
  send_blocks.resize(kv_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  size_t attn_group_offset = num_gdn_layers;

  auto attn_ntpb = kv_block_sizes.at(0) / kv_token_sizes.at(0);
  auto indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  assert(src_worker_info->attn_kernel_blk_ntpb == dst_worker_info->attn_kernel_blk_ntpb);
  assert(indexer_ntpb == dst_worker_info->indexer_blk_ntpb);

  if (indexer_ntpb > 0) {
    const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
    const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
    do_parse_block_aligned_send(
      kv_block_sizes.at(0),
      task,
      src_indexer_blks, dst_indexer_blks,
      indexer_ntpb, per_cache_send_blocks
    );
    attn_group_offset += 1;
  }
  const auto& src_attn_blks = src_blocks.at(attn_group_offset);
  const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);
  do_parse_block_aligned_send(
    kv_block_sizes.at(0),
    task,
    src_attn_blks, dst_attn_blks,
    attn_ntpb, per_cache_send_blocks
  );

  if (task->reach_last_token){
    for (size_t group_idx = 0; group_idx < num_gdn_layers; ++group_idx) {
      for (size_t block_idx = 0; block_idx < src_blocks.at(group_idx).size(); ++block_idx) {
        auto src_offset = src_blocks.at(group_idx).at(block_idx) * kv_block_sizes.at(0);
        auto dst_offset = dst_blocks.at(group_idx).at(block_idx) * kv_block_sizes.at(0);
        auto length = kv_block_sizes.at(0);
        per_cache_send_blocks.emplace_back(src_offset, dst_offset, length);
      }
    }
  }
}

// QWEN3_NEXT_FLASH_CACHE_SHAPE: P > D
void vllm_parse_hybrid_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  const auto & p_block_sizes = src_worker_info->block_sizes;
  const auto & d_block_sizes = dst_worker_info->block_sizes;
  const auto & p_token_sizes = src_worker_info->token_sizes;
  const auto & d_token_sizes = dst_worker_info->token_sizes;

  assert(src_worker_info->kvt_tp_size > dst_worker_info->kvt_tp_size);
  assert(src_worker_info->kvt_tp_size % dst_worker_info->kvt_tp_size == 0);

  const uint32_t p_origin_tp_size = src_worker_info->engine_tp_size;
  const uint32_t gdn_group_n = p_origin_tp_size / dst_worker_info->kvt_tp_size;

  assert(src_worker_info->worker_tp_rank / gdn_group_n == dst_worker_info->worker_tp_rank);
  const uint32_t gdn_group_off = src_worker_info->worker_tp_rank % gdn_group_n;
  const uint32_t attn_group_n = valid_ranks.count() / dst_worker_info->kvt_tp_size;
  const uint32_t attn_group_off = kvt_tp_rank % attn_group_n;

  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);
  assert(d_token_size == p_token_size * attn_group_n);

  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb * p_token_size == p_block_size);
  assert(dst_ntpb * d_token_size == d_block_size);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  do_parse_hybrid_block_send_p_gt_d(
    p_token_size, d_token_size,
    src_ntpb, dst_ntpb,
    attn_group_n, attn_group_off,
    gdn_group_n, gdn_group_off,
    valid_ranks,
    src_worker_info,
    dst_worker_info,
    task, per_cache_send_blocks
  );
}

}  // namespace blade_llm
