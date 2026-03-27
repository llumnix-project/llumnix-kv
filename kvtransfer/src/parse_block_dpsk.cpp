#include "parse_block_common.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

static void do_multi_tensor_parse_block_send(
  const WorkerInfo *p_info,
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
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

    assert(p_info->kvt_tp_size > d_info->kvt_tp_size);
    assert((p_info->kvt_tp_size % d_info->kvt_tp_size) == 0);
    const uint32_t group_n = p_info->kvt_tp_size / d_info->kvt_tp_size;
    assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
    const uint32_t group_off = p_info->worker_tp_rank % group_n;
    assert(d_token_size == p_token_size * group_n);
    assert(d_token_size % 2 == 0);
    assert(p_token_size % 2 == 0);
    const size_t p_k_size = p_token_size / 2;
    const size_t d_k_size = d_token_size / 2;
    assert(d_k_size == p_k_size * group_n);
    const uint32_t ntpb = p_block_size / p_token_size;
    assert(ntpb * p_token_size == p_block_size);
    assert(ntpb * d_token_size == d_block_size);

    const size_t sbsize = per_cache_send_blocks.size();
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
}

// DPSK_V32_SPARSE_MLA_SHAPE: P == D
void vllm_parse_block_send_multi_tensor_p_eq_d(
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

  const auto &token_sizes = src_worker_info->token_sizes;
  const auto &block_sizes = src_worker_info->block_sizes;

  assert(token_sizes == dst_worker_info->token_sizes);
  assert(block_sizes == dst_worker_info->block_sizes);
  assert(token_sizes.size() == block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());

  send_blocks.resize(token_sizes.size());
  for (size_t i = 0; i < token_sizes.size(); ++i) {
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(i);
    do_parse_block_send_p_eq_d(
      block_sizes.at(i), token_sizes.at(i),
      task, per_cache_send_blocks
    );
  }
}

// DPSK_V32_SPARSE_MLA_SHAPE: P > D
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
    assert(p_info->block_sizes == d_info->block_sizes);
    auto &token_sizes = p_info->token_sizes;
    auto &block_sizes = p_info->block_sizes;
    assert(token_sizes.size() == block_sizes.size());

    send_blocks.resize(token_sizes.size());
    for (size_t i = 0; i < token_sizes.size(); ++i) {
      std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(i);
      do_parse_block_send_p_eq_d(
        block_sizes.at(i), token_sizes.at(i), task, per_cache_send_blocks
      );
    }
    return;
  }

  return do_multi_tensor_parse_block_send(
    p_info, task->src_blocks().at(0),
    d_info, task->dst_blocks().at(0),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
}

// DPSK_V32_SPARSE_MLA_SHAPE: P < D
void vllm_parse_block_send_multi_tensor_p_lt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  size_t cache_idx = send_blocks.size();
  do_multi_tensor_parse_block_send(
    d_info, task->dst_blocks().at(0),
    p_info, task->src_blocks().at(0),
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
