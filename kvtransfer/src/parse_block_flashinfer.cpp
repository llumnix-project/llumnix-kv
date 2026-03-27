#include "parse_block_common.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// storage shape in l20:(num_blocks, 2, num_kv_heads, block_size, head_dim)
static void parse_flashinfer_HND_block(
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  uint32_t ntpb,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  uint32_t attn_group_n,
  uint32_t attn_group_off,
  int num_kv_heads,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  while (left_tokens > 0) {
    auto block_idx = wrote_tokens / ntpb;
    auto token_idx = wrote_tokens % ntpb;
    auto p_blk_off = src_blocks.at(block_idx) * p_block_size;
    auto d_blk_off = dst_blocks.at(block_idx) * d_block_size;

    auto tokens = std::min(ntpb - token_idx, left_tokens);

    auto p_head_num = num_kv_heads;
    const auto p_head_size = p_block_size / 2 / p_head_num;
    const auto p_k_block_size = p_block_size / 2;
    const auto d_k_block_size = d_block_size / 2;

    for (auto head_idx = 0; head_idx < p_head_num; ++head_idx) {
      auto p_k_off = p_blk_off + head_idx * p_head_size;
      auto d_k_off = d_blk_off + attn_group_off * p_k_block_size + head_idx * p_head_size;
      auto p_v_off = p_blk_off + p_k_block_size;
      auto d_v_off = d_blk_off + d_k_block_size;
      auto length = p_head_size / ntpb * tokens;

      per_cache_send_blocks.emplace_back(p_k_off, d_k_off, length);
      per_cache_send_blocks.emplace_back(p_v_off, d_v_off, length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

// FLASHINFER_CACHE_SHAPE: P == D
void vllm_parse_flashinfer_block_send_p_eq_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  assert(p_info->kvt_tp_size == d_info->kvt_tp_size);

  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;

  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);

  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb == dst_ntpb);

  const auto &src_blocks = task->src_blocks().at(0);
  const auto &dst_blocks = task->dst_blocks().at(0);
  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;

  send_blocks.resize(1);
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  parse_flashinfer_HND_block(
    p_token_size, d_token_size,
    p_block_size, d_block_size,
    src_ntpb,
    src_blocks, dst_blocks,
    wrote_tokens, left_tokens,
    1, 0,
    p_info->num_kv_heads,
    per_cache_send_blocks
  );
}

// FLASHINFER_CACHE_SHAPE: P > D
void vllm_parse_flashinfer_block_send_p_gt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  if (!valid_ranks[p_info->worker_tp_rank]) {
    return;
  }
  const uint32_t attn_group_n = valid_ranks.count() / d_info->kvt_tp_size;
  const uint32_t attn_group_off = kvt_tp_rank % attn_group_n;
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;
  assert(p_info->kvt_tp_size > d_info->kvt_tp_size);
  assert(kvt_tp_rank / attn_group_n == d_info->worker_tp_rank);

  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);

  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb == dst_ntpb);

  const auto &src_blocks = task->src_blocks().at(0);
  const auto &dst_blocks = task->dst_blocks().at(0);
  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;

  send_blocks.resize(1);
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  parse_flashinfer_HND_block(
    p_token_size, d_token_size,
    p_block_size, d_block_size,
    src_ntpb,
    src_blocks, dst_blocks,
    wrote_tokens, left_tokens,
    attn_group_n,
    attn_group_off,
    p_info->num_kv_heads,
    per_cache_send_blocks
  );
}

}  // namespace blade_llm
