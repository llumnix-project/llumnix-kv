#ifndef KVTRANSFER_INCLUDE_PARSE_BLOCK_COMMON_H_
#define KVTRANSFER_INCLUDE_PARSE_BLOCK_COMMON_H_

#pragma once

#include "common.h"
#include "envcfg.h"
#include "channel.h"
#include <vector>

namespace blade_llm {

// ParseBlock function type: parse KV cache blocks for transfer.
//
// valid_ranks / kvt_tp_size / kvt_tp_rank are runtime per-destination
// values computed in KvSendStub::TaskContext::refresh_dst_info():
//   - valid_ranks: bitmask of source ranks that hold unique KV heads (used
//     for filtering when num_kv_heads < engine_tp_size and for mapping P
//     ranks to D ranks).
//   - kvt_tp_size: effective P TP after valid_ranks filtering, i.e.
//     valid_ranks.count(). This is what the parse_block_* implementations
//     should compare against `dst_worker_info->engine_tp_size`. NOTE that
//     for cache shapes carrying GDN state (QWEN3_NEXT_*), the
//     P==D/P>D parse_block selection happens in tx_stub.cpp based on
//     engine_tp_size; this parameter still carries the effective attn TP.
//   - kvt_tp_rank: the index of `src.worker_tp_rank` within `valid_ranks`
//     (i.e. how many bits are set below it). Used as the per-rank offset
//     inside the {p, d}-rank mapping group for attention parsing.
using ParseBlockFunc = void(*)(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// =============================================================================
// Common utility functions (shared across cache shapes)
// =============================================================================

// Basic parse for p_eq_d case (same tp size on both sides)
void do_parse_block_send_p_eq_d(
  size_t block_size,
  size_t token_size,
  const ReqSendTask *task,
  std::vector<IpcBlock> &per_cache_send_blocks
);

// Parse for p_gt_d case at token granularity
void parse_block_send_gt(
  size_t p_token_size,
  size_t p_k_size,
  size_t d_token_size,
  uint32_t ntpb,
  size_t group_off,
  const std::vector<uint32_t>& p_blocks,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &per_cache_send_blocks
);

// Fill a contiguous destination run [dst_off, dst_off+length) with zero data.
// `zero_src_off` must point at the start of a region that is entirely zeros and
// at least `max_seg` bytes long (typically the request's block 0, whose byte
// size equals the block size). Because the whole source region is zero, an
// arbitrarily long destination run can be satisfied by repeatedly copying from
// the same zero region; each emitted IpcBlock copies at most `max_seg` bytes so
// the source read never crosses the zero block boundary. length <= max_seg
// (the common case) emits exactly one IpcBlock.
void emit_zero_fill_segments(
  size_t zero_src_off,
  size_t dst_off,
  size_t length,
  size_t max_seg,
  std::vector<IpcBlock> &per_cache_send_blocks
);

// =============================================================================
// Cache shape specific parse block functions
// =============================================================================

// RAGGED_FLASH_CACHE_SHAPE
void parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_block_send_p_gt_d_dpsk(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_block_send_p_lt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// FLASH_CACHE_SHAPE (vllm)
void vllm_parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void vllm_parse_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// QWEN3_NEXT_FLASH_CACHE_SHAPE (vllm hybrid)
void vllm_parse_hybrid_block_send_p_eq_d_block_aligned(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void vllm_parse_hybrid_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// DPSK_V32_SPARSE_MLA_SHAPE
void vllm_parse_block_send_multi_tensor_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void vllm_parse_block_send_multi_tensor_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void vllm_parse_block_send_multi_tensor_p_lt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// FLASHINFER_CACHE_SHAPE
void vllm_parse_flashinfer_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void vllm_parse_flashinfer_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// QWEN3_NEXT_FLASHINFER_CACHE_SHAPE (vllm hybrid + flashinfer HND attn)
void vllm_parse_qwen3_next_flashinfer_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

}  // namespace blade_llm

#endif  // KVTRANSFER_INCLUDE_PARSE_BLOCK_COMMON_H_
