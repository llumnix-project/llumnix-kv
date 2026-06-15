#include "parse_block_common.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// Internal helper for bladellm single-tensor case.
// p_tp_size / d_tp_size are the *effective* TP sizes used to compute group_n.
// In the canonical P>D call, p_tp_size = effective P kvt_tp_size and
// d_tp_size = dst.engine_tp_size. The P<D path calls this with the args
// swapped, so we accept tp sizes explicitly instead of reading from
// p_info/d_info.
static void do_parse_block_send_internal(
  const WorkerInfo *p_info,
  uint32_t p_tp_size,
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,
  uint32_t d_tp_size,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// RAGGED_FLASH_CACHE_SHAPE: P == D
void parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // as tp size is same, token size should be same;
  assert(kvt_tp_size == dst_worker_info->engine_tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);

  const auto &token_sizes = src_worker_info->token_sizes;
  const auto &block_sizes = src_worker_info->block_sizes;

  assert(token_sizes == dst_worker_info->token_sizes);
  assert(block_sizes == dst_worker_info->block_sizes);
  assert(token_sizes.size() == block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());
  // Should only have one cache in one layer for RAGGED_FLASH_CACHE_SHAPE.
  assert(token_sizes.size() == 1);
  assert(block_sizes.size() == 1);
  send_blocks.resize(token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
  do_parse_block_send_p_eq_d(
    block_sizes[0], token_sizes[0],
    task, per_cache_send_blocks
  );
}

// RAGGED_FLASH_CACHE_SHAPE: P > D (DPSK case, token sizes equal)
void parse_block_send_p_gt_d_dpsk(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  assert(p_info->token_sizes == d_info->token_sizes);
  // This means kvcache is identical across all P ranks.
  // Only the tp rank=0 worker needs to transfer kvcache.
  if (p_info->worker_tp_rank != 0) {
    return;
  }

  assert(d_info->worker_tp_rank == 0);
  // PD blocks must contain the same number of tokens
  assert(p_info->block_sizes == d_info->block_sizes);
  auto &token_sizes = p_info->token_sizes;
  auto &block_sizes = p_info->block_sizes;
  assert(token_sizes.size() == block_sizes.size());
  // Should only have one cache in one layer for RAGGED_FLASH_CACHE_SHAPE.
  assert(token_sizes.size() == 1);
  assert(block_sizes.size() == 1);

  send_blocks.resize(token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
  do_parse_block_send_p_eq_d(
    block_sizes[0], token_sizes[0], task, per_cache_send_blocks
  );
}

// Internal helper for bladellm single-tensor case.
// See the forward declaration above for the meaning of p_tp_size/d_tp_size.
static void do_parse_block_send_internal(
  const WorkerInfo *p_info,
  uint32_t p_tp_size,
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,
  uint32_t d_tp_size,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
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
  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);

  const size_t p_block_size = p_block_sizes.at(0);
  const size_t d_block_size = d_block_sizes.at(0);
  const size_t p_token_size = p_token_sizes.at(0);
  const size_t d_token_size = d_token_sizes.at(0);
  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);

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
    per_cache_send_blocks
  );
  const size_t sbsize_after = per_cache_send_blocks.size();

  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sbsize_after - sbsize);
  for (size_t idx = sbsize; idx < sbsize_after; ++idx) {
    const auto& sb = per_cache_send_blocks.at(idx);
    const size_t pv_token_off = sb.src_offset + p_k_size;
    const size_t dv_token_off = sb.dst_offset + d_k_size;
    assert(p_k_size == sb.length);
    per_cache_send_blocks.emplace_back(pv_token_off, dv_token_off, p_k_size);
  }
}

// RAGGED_FLASH_CACHE_SHAPE: P > D (bladellm case)
void parse_block_send_p_gt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  return do_parse_block_send_internal(
    p_info, kvt_tp_size, task->src_blocks()[0],
    d_info, d_info->engine_tp_size, task->dst_blocks()[0],
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
}

// RAGGED_FLASH_CACHE_SHAPE: P < D
void parse_block_send_p_lt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  size_t cache_idx = send_blocks.size();
  // P<D: swap (p, d) so the helper's "p side" is actually the larger TP (D).
  do_parse_block_send_internal(
    d_info, d_info->engine_tp_size, task->dst_blocks()[0],
    p_info, kvt_tp_size, task->src_blocks()[0],
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
  for (; cache_idx < send_blocks.size(); ++cache_idx) {
    auto &per_cache_sbs = send_blocks.at(cache_idx);
    for(size_t sb_idx = 0; sb_idx < per_cache_sbs.size(); ++sb_idx){
      std::swap(per_cache_sbs.at(sb_idx).src_offset, per_cache_sbs.at(sb_idx).dst_offset);
    }
  }
}

}  // namespace blade_llm
