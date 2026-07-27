#ifndef KVTRANSFER_INCLUDE_PARSE_BLOCK_COMMON_H_
#define KVTRANSFER_INCLUDE_PARSE_BLOCK_COMMON_H_

#pragma once

#include "common.h"
#include "envcfg.h"
#include "channel.h"
#include "thrid_party/logging.h"
#include <limits>
#include <stdexcept>
#include <vector>

namespace blade_llm {

// Per-layer address ranges used while parse_block emits IpcBlocks.  Capacity
// is computed once for each cache tensor; every emitted range is checked
// before it is appended, avoiding a second O(number_of_ipc_blocks) pass.
struct IpcBlockBounds {
  size_t src_capacity;
  size_t dst_capacity;
  size_t src_length_scale;
  size_t cache_idx;
};

IpcBlockBounds make_ipc_block_bounds(
    const WorkerInfo& src_worker_info,
    const WorkerInfo& dst_worker_info,
    size_t cache_idx);

inline void append_ipc_block_checked(
    std::vector<IpcBlock>& blocks,
    const IpcBlockBounds& bounds,
    size_t src_offset,
    size_t dst_offset,
    size_t length) {
  const bool src_length_overflow = bounds.src_length_scale == 0 ||
      length > std::numeric_limits<size_t>::max() / bounds.src_length_scale;
  const size_t effective_src_length = src_length_overflow
      ? std::numeric_limits<size_t>::max()
      : length * bounds.src_length_scale;
  const bool src_invalid = length == 0 || src_length_overflow ||
      src_offset > bounds.src_capacity ||
      effective_src_length > bounds.src_capacity - src_offset;
  const bool dst_invalid = length == 0 ||
      dst_offset > bounds.dst_capacity ||
      length > bounds.dst_capacity - dst_offset;
  if (src_invalid || dst_invalid) {
    const bool source_failed = src_invalid;
    LOG(ERROR) << "KVT GPU buffer range validation failed"
               << ",copy_path=parse_block_generation"
               << ",layer_idx=0"
               << ",tensor_idx=" << bounds.cache_idx
               << ",block_idx=" << blocks.size()
               << ",offset_kind=" << (source_failed ? "src" : "dst")
               << ",src_offset=" << src_offset
               << ",dst_offset=" << dst_offset
               << ",length=" << length
               << ",length_scale="
               << (source_failed ? bounds.src_length_scale : 1)
               << ",effective_length="
               << (source_failed ? effective_src_length : length)
               << ",capacity="
               << (source_failed ? bounds.src_capacity : bounds.dst_capacity);
    throw std::out_of_range("KVT GPU buffer range validation failed");
  }
  blocks.emplace_back(src_offset, dst_offset, length);
}

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
  const IpcBlockBounds& bounds,
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
  const IpcBlockBounds& bounds,
  bool swap_offsets,
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
  const IpcBlockBounds& bounds,
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

void vllm_parse_hybrid_block_send_p_lt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// KDA state pages (community vLLM layout). These functions intentionally
// parse only the leading `num_gdn_layers` linear-attention block groups. They
// reuse GDN's state-shape metadata without calling or changing GDN helpers.
// They are exported now so the KDA layout and TP remapping can be tested
// independently; cache-shape registration and the hybrid attention wrapper
// are added with the vLLM connector integration.
void parse_kda_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_kda_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_kda_block_send_p_lt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

// KIMI_K3_MLA_CACHE_SHAPE: leading KDA groups + one replicated MLA group.
void parse_kimi_k3_mla_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_kimi_k3_mla_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

void parse_kimi_k3_mla_block_send_p_lt_d(
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

void vllm_parse_qwen3_next_flashinfer_block_send_p_lt_d(
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
