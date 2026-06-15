#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace blade_llm {

enum class CacheDim : uint8_t {
  BLOCK,     // num_gpu_blocks
  TOKEN,     // tokens_per_block
  KV,        // 2 (K and V)
  HEAD,      // num_kv_heads
  HEAD_DIM,  // head_dim
};

struct AttnLayoutDesc {
  // Dimension ordering from outermost to innermost in physical memory.
  // Determines all strides/offsets for AttnSpec generation.
  //
  // Layouts for known cache shapes:
  //   RAGGED_FLASH:  [BLOCK, TOKEN, KV, HEAD, HEAD_DIM]
  //   FLASH:         [KV, BLOCK, TOKEN, HEAD, HEAD_DIM]
  //   QWEN3_NEXT:    [BLOCK, KV, TOKEN, HEAD, HEAD_DIM]
  //   DPSK_V32:      [BLOCK, TOKEN, KV, HEAD, HEAD_DIM]
  //   FLASHINFER:    [BLOCK, KV, HEAD, TOKEN, HEAD_DIM]
  std::array<CacheDim, 5> dim_order;

  // Number of separate cache tensors per layer (>1 for DPSK_V32).
  // When 0, derived from WorkerInfo::token_sizes.size() at runtime.
  uint32_t num_tensors = 1;

  // P==D uses block-aligned transfer (whole-block granularity).
  bool block_aligned_peqd = false;

  // V component has head_stride=0 and no TP group offset on dst side.
  // Preserves FLASHINFER legacy behavior where V is head-invariant.
  bool v_head_invariant = false;

  // --- TP Policy flags ---

  // P>D: filter ranks by env_p_valid_ranks(); inactive ranks get spec.valid=false.
  bool use_valid_ranks = false;

  // When use_valid_ranks is true, group_n = valid_count / dst_tp
  // instead of src_tp / dst_tp.
  bool valid_ranks_affects_group_n = false;

  // P>D with identical token_sizes: only rank 0 sends (treated as P==D).
  bool rank0_only_when_same_sizes = false;

  // Force TPKind to PEQD regardless of TP configuration.
  // Used by DPSK_V32 where each rank has complete data (MLA architecture).
  bool force_peqd_tpkind = false;

  // Helper: find position of a dimension in dim_order.
  int pos(CacheDim d) const {
    for (int i = 0; i < 5; ++i) {
      if (dim_order[i] == d) return i;
    }
    return -1;
  }

  bool is_kv_outermost() const { return pos(CacheDim::KV) < pos(CacheDim::BLOCK); }
  bool is_kv_between_block_and_token() const {
    return pos(CacheDim::BLOCK) < pos(CacheDim::KV) &&
           pos(CacheDim::KV) < pos(CacheDim::TOKEN);
  }
  bool is_kv_inner_to_token() const { return pos(CacheDim::TOKEN) < pos(CacheDim::KV); }
  bool is_head_outer_to_token() const { return pos(CacheDim::HEAD) < pos(CacheDim::TOKEN); }
};

// Get the attention layout descriptor for a given cache shape.
// Returns a reference to a static descriptor; adding a new cache shape
// only requires adding a new entry here.
const AttnLayoutDesc& get_attn_layout(int cache_shape);

}  // namespace blade_llm
