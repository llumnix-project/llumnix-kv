#include "parse_block_common.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// FLASH_CACHE_SHAPE (vllm): P == D
// cache shape: (2, num_blocks, block_size, num_kv_heads, head_dim)
void vllm_parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  assert(kvt_tp_size == dst_worker_info->engine_tp_size);

  auto const& kv_token_sizes = src_worker_info->token_sizes;
  auto const& kv_block_sizes = src_worker_info->block_sizes;

  // as tp size is same, token size should be same;
  assert(kv_token_sizes == dst_worker_info->token_sizes);
  assert(kv_block_sizes == dst_worker_info->block_sizes);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  assert(kv_token_sizes.size() == kv_block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());
  // Should only have one cache in one layer for FLASH_CACHE_SHAPE.
  assert(kv_token_sizes.size() == 1);
  assert(kv_block_sizes.size() == 1);
  auto &kv_token_size = kv_token_sizes.at(0);
  auto &kv_block_size = kv_block_sizes.at(0);
  size_t const k_token_size = kv_token_size / 2;
  size_t const k_block_size = kv_block_size / 2;
  assert(2 * k_token_size == kv_token_size);
  assert(2 * k_block_size == kv_block_size);

  size_t const src_layer_size = k_block_size * src_worker_info->layer_num_blocks;
  size_t const dst_layer_size = k_block_size * dst_worker_info->layer_num_blocks;
  // src_layer_size and dst_layer_size are not necessarily equal.

  send_blocks.resize(kv_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);
  size_t sb_idx = per_cache_send_blocks.size();  // sb: send block~
  do_parse_block_send_p_eq_d(
      k_block_size, k_token_size, task, bounds, per_cache_send_blocks);
  size_t const sb_end_idx = per_cache_send_blocks.size();

  // emplace_back v tensor block, which is non-contigious with k tensor
  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sb_end_idx - sb_idx);
  for (; sb_idx < sb_end_idx; ++sb_idx) {
    auto sb = per_cache_send_blocks.at(sb_idx);
    sb.src_offset += src_layer_size;
    sb.dst_offset += dst_layer_size;
    append_ipc_block_checked(
        per_cache_send_blocks, bounds, sb.src_offset, sb.dst_offset, sb.length);
  }
}

// FLASH_CACHE_SHAPE (vllm): P > D
// cache shape: (2, num_blocks, block_size, num_kv_heads, head_dim)
void vllm_parse_block_send_p_gt_d(
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
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;
  assert(kvt_tp_size > d_info->engine_tp_size);
  const uint32_t group_n = valid_ranks.count() / d_info->engine_tp_size;
  assert(group_n > 0);
  assert(kvt_tp_rank / group_n == d_info->worker_tp_rank);
  const uint32_t group_off = kvt_tp_rank % group_n;

  // Should only have one cache in one layer.
  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);

  assert(d_token_size == p_token_size * group_n);
  assert(d_token_size % 2 == 0);
  assert(p_token_size % 2 == 0);
  const size_t p_k_size = p_token_size / 2;
  const size_t d_k_size = d_token_size / 2;
  assert(d_k_size == p_k_size * group_n);
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_token_size;
  assert(ntpb * p_token_size == p_block_size);
  assert(ntpb * d_token_size == d_block_size);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(*p_info, *d_info, 0);
  size_t sb_idx = per_cache_send_blocks.size();  // sb: send block~
  parse_block_send_gt(
    p_k_size, p_k_size, d_k_size, ntpb, group_off,
    task->src_blocks()[0], task->dst_blocks()[0],
    task->seen_tokens, task->new_tokens, bounds, false,
    per_cache_send_blocks);
  size_t const sb_end_idx = per_cache_send_blocks.size();
  size_t const pk_layer_size = ntpb * p_k_size * p_info->layer_num_blocks;
  size_t const dk_layer_size = ntpb * d_k_size * d_info->layer_num_blocks;

  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sb_end_idx - sb_idx);
  for (; sb_idx < sb_end_idx; ++sb_idx) {
    auto sb = per_cache_send_blocks.at(sb_idx);
    sb.src_offset += pk_layer_size;
    sb.dst_offset += dk_layer_size;
    append_ipc_block_checked(
        per_cache_send_blocks, bounds, sb.src_offset, sb.dst_offset, sb.length);
  }
}

}  // namespace blade_llm
