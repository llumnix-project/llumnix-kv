#include "parse_block_common.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// =============================================================================
// DPSK specific helper functions
// =============================================================================

// Multi-tensor parse block send - only used by DPSK_V32_SPARSE_MLA_SHAPE.
// p_tp_size / d_tp_size are the *effective* TP sizes used to compute group_n.
// P<D path calls this helper with (p, d) swapped, so we accept TP sizes
// as explicit args instead of reading them from p_info/d_info.
static void do_multi_tensor_parse_block_send(
  const WorkerInfo *p_info,
  uint32_t p_tp_size,
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,
  uint32_t d_tp_size,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  const WorkerInfo *actual_src_info,
  const WorkerInfo *actual_dst_info,
  bool swap_offsets,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // for bladellm
  // cache shape [num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;

  assert(p_block_sizes.size() == d_block_sizes.size());
  assert(p_token_sizes.size() == d_token_sizes.size());
  assert(p_block_sizes.size() == p_token_sizes.size());

  send_blocks.resize(p_token_sizes.size());
  for(size_t cache_idx = 0; cache_idx < p_block_sizes.size(); cache_idx++){
    const size_t p_block_size = p_block_sizes.at(cache_idx);
    const size_t d_block_size = d_block_sizes.at(cache_idx);
    const size_t p_token_size = p_token_sizes.at(cache_idx);
    const size_t d_token_size = d_token_sizes.at(cache_idx);

    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(cache_idx);
    const auto bounds = make_ipc_block_bounds(
        *actual_src_info, *actual_dst_info, cache_idx);

    assert(p_tp_size > d_tp_size);
    assert((p_tp_size % d_tp_size) == 0);
    const uint32_t group_n = p_tp_size / d_tp_size;
    assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
    const uint32_t group_off = p_info->worker_tp_rank % group_n;
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

    const size_t sbsize = per_cache_send_blocks.size();
    // Same block id, different token_size/block_size
    parse_block_send_gt(
      p_token_size, p_k_size, d_token_size, ntpb, group_off,
      p_blocks, d_blocks, wrote_tokens, left_tokens,
      bounds, swap_offsets,
      per_cache_send_blocks
    );
    const size_t sbsize_after = per_cache_send_blocks.size();

    per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sbsize_after - sbsize);
    for (size_t idx = sbsize; idx < sbsize_after; ++idx) {
      const auto& sb = per_cache_send_blocks.at(idx);
      const size_t pv_token_off = sb.src_offset +
          (swap_offsets ? d_k_size : p_k_size);
      const size_t dv_token_off = sb.dst_offset +
          (swap_offsets ? p_k_size : d_k_size);
      assert(p_k_size == sb.length);
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          pv_token_off,
          dv_token_off,
          p_k_size);
    }
  }
}

// =============================================================================
// DPSK_V32_SPARSE_MLA_SHAPE: P == D
// Each layer contains two tensors: k tensor for select and mla tensor for attention.
// =============================================================================

void vllm_parse_block_send_multi_tensor_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // Skip ranks not selected by valid_ranks. Necessary for the "effective
  // P==D" case where the original P had more TP ranks than D and only a
  // subset (filtered by valid_ranks) should actually send.
  if (!valid_ranks[src_worker_info->worker_tp_rank]) {
    return;
  }

  // as tp size is same, token size should be same;
  assert(kvt_tp_size == dst_worker_info->engine_tp_size);
  // In the effective P==D case kvt_tp_rank is the dst rank this sender maps
  // to; for the genuine P==D case it equals src worker_tp_rank == dst rank.
  assert(kvt_tp_rank == dst_worker_info->worker_tp_rank);

  const auto &token_sizes = src_worker_info->token_sizes;
  const auto &block_sizes = src_worker_info->block_sizes;

  assert(token_sizes == dst_worker_info->token_sizes);
  assert(block_sizes == dst_worker_info->block_sizes);
  assert(token_sizes.size() == block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());

  // Each entry of send_blocks stores the block list for one cache tensor within the layer
  send_blocks.resize(token_sizes.size());
  for (size_t i = 0; i < token_sizes.size(); ++i) {
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(i);
    const auto bounds = make_ipc_block_bounds(
        *src_worker_info, *dst_worker_info, i);
    do_parse_block_send_p_eq_d(
      block_sizes.at(i), token_sizes.at(i),
      task, bounds, per_cache_send_blocks
    );
  }
}

// =============================================================================
// DPSK_V32_SPARSE_MLA_SHAPE: P > D
// =============================================================================

void vllm_parse_block_send_multi_tensor_p_gt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  if (p_info->token_sizes == d_info->token_sizes) {
    if (p_info->worker_tp_rank != 0) {
      return;
    }

    assert(d_info->worker_tp_rank == 0);
    // block size of PD Block should be the same
    assert(p_info->block_sizes == d_info->block_sizes);
    auto &token_sizes = p_info->token_sizes;
    auto &block_sizes = p_info->block_sizes;
    assert(token_sizes.size() == block_sizes.size());

    send_blocks.resize(token_sizes.size());
    for (size_t i = 0; i < token_sizes.size(); ++i) {
      std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(i);
      const auto bounds = make_ipc_block_bounds(*p_info, *d_info, i);
      do_parse_block_send_p_eq_d(
        block_sizes.at(i), token_sizes.at(i), task, bounds,
        per_cache_send_blocks
      );
    }
    return;
  }

  return do_multi_tensor_parse_block_send(
    p_info, kvt_tp_size, task->src_blocks().at(0),
    d_info, d_info->engine_tp_size, task->dst_blocks().at(0),
    task->seen_tokens,
    task->new_tokens,
    p_info,
    d_info,
    false,
    send_blocks);
}

// =============================================================================
// DPSK_V32_SPARSE_MLA_SHAPE: P < D
// =============================================================================

void vllm_parse_block_send_multi_tensor_p_lt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // P<D: swap (p, d) so the helper's "p side" is actually the larger TP (D).
  do_multi_tensor_parse_block_send(
    d_info, d_info->engine_tp_size, task->dst_blocks().at(0),
    p_info, kvt_tp_size, task->src_blocks().at(0),
    task->seen_tokens,
    task->new_tokens,
    p_info,
    d_info,
    true,
    send_blocks);
}

}  // namespace blade_llm
