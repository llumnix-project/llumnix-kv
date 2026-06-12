#ifndef KVTRANSFER_SRC_PARSE_BLOCK_FLASHINFER_INTERNAL_H_
#define KVTRANSFER_SRC_PARSE_BLOCK_FLASHINFER_INTERNAL_H_

#pragma once

#include "parse_block_common.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace blade_llm {

// FLASHINFER HND attention block parser.
// Storage layout: (num_blocks, 2, num_kv_heads, block_size, head_dim).
// Splits each block by KV (K/V), then by head; per-head per-token slices are
// written to dst with the K side offset by attn_group_off.
// num_kv_heads: per-TP (per P-rank) count, NOT total.
void parse_flashinfer_HND_block(
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
);

// Hybrid (qwen3-next style) FlashInfer HND attention block parser.
// Storage layout per kernel block:
//   (num_kernel_blocks, 2, num_kv_heads, kernel_block_size, head_dim)
// Each block is reshaped into multiple kernel blocks; token idx is relative
// to a kernel block (src and dst may use different kernel_blk_ntpb). The
// destination block may merge attn_group_n TP groups along the K/V section.
void parse_hybrid_flashinfer_HND_block(
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  uint32_t p_ntpb,
  uint32_t d_ntpb,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  uint32_t attn_group_n,
  uint32_t attn_group_off,
  uint32_t src_attn_kernel_blk_ntpb,
  uint32_t dst_attn_kernel_blk_ntpb,
  int num_kv_heads,
  std::vector<IpcBlock> &per_cache_send_blocks
);

// Pad the unfilled tail of the last decode-side attn block (qwen3-next
// FlashInfer HND, P>D) using the request's block 0 (src_blocks[0]) data.
// total_tokens = seen_tokens + new_tokens (per-layer). No-op when the last
// dst block is already full. Block 0 is all zeros, so this simply emits one
// K and one V IpcBlock per kv-head covering each head's contiguous unfilled
// tail [filled, d_ntpb), sourcing zero data from block 0.
void fill_last_hybrid_flashinfer_HND_block(
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  uint32_t d_ntpb,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t total_tokens,
  uint32_t attn_group_off,
  int num_kv_heads,
  std::vector<IpcBlock> &per_cache_send_blocks
);

}  // namespace blade_llm

#endif  // KVTRANSFER_SRC_PARSE_BLOCK_FLASHINFER_INTERNAL_H_
