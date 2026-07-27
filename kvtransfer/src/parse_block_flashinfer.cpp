#include "parse_block_common.h"
#include "parse_block_flashinfer_internal.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// =============================================================================
// FlashInfer specific helper functions
// =============================================================================

// need to split by head dimension, no need to distinguish p gt d or p eq d
// storage shape in l20:(num_blocks, 2, num_kv_heads, block_size, head_dim)
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
  const IpcBlockBounds& bounds,
  bool swap_offsets,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(num_kv_heads > 0);
  assert(p_token_size % (2 * static_cast<size_t>(num_kv_heads)) == 0);
  const size_t head_dim_size = p_token_size / 2 / static_cast<size_t>(num_kv_heads);
  const size_t p_head_section_size = static_cast<size_t>(ntpb) * head_dim_size;
  const size_t d_head_section_size = static_cast<size_t>(ntpb) * head_dim_size;
  const size_t p_kv_section_size = p_block_size / 2;
  const size_t d_kv_section_size = d_block_size / 2;
  const size_t per_group_kv_size = static_cast<size_t>(num_kv_heads) * d_head_section_size;
  assert(per_group_kv_size * attn_group_n == d_kv_section_size);

  while (left_tokens > 0) {
    auto block_idx = wrote_tokens / ntpb;
    auto token_idx = wrote_tokens % ntpb;
    auto p_blk_off = src_blocks.at(block_idx) * p_block_size;
    auto d_blk_off = dst_blocks.at(block_idx) * d_block_size;
    auto tokens = std::min(ntpb - token_idx, left_tokens);
    const auto length = static_cast<size_t>(tokens) * head_dim_size;

    for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
      const size_t pk_off = p_blk_off
                          + static_cast<size_t>(head_idx) * p_head_section_size
                          + static_cast<size_t>(token_idx) * head_dim_size;
      const size_t pv_off = pk_off + p_kv_section_size;
      const size_t dk_off = d_blk_off
                          + attn_group_off * per_group_kv_size
                          + static_cast<size_t>(head_idx) * d_head_section_size
                          + static_cast<size_t>(token_idx) * head_dim_size;
      const size_t dv_off = dk_off + d_kv_section_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          swap_offsets ? dk_off : pk_off,
          swap_offsets ? pk_off : dk_off, length);
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          swap_offsets ? dv_off : pv_off,
          swap_offsets ? pv_off : dv_off, length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

// used for qwen3-next's flashinfer attn block parsing
// kv cache stride per kernel block:
//   (num_kernel_blocks, 2, num_kv_heads, kernel_block_size, head_dim)
// Within one kernel block, K comes before V. Each head occupies a contiguous
// kbs * head_dim region in each section. In the destination kernel block,
// the K/V sections from attn_group_n TP groups are merged:
//   K: [K_group0(num_kv_heads * dst_kbs * head_dim), K_group1, ...],
// followed by V in the same layout.
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
  const IpcBlockBounds& bounds,
  bool swap_offsets,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // parse attention block
  const auto p_kernel_blk_size = src_attn_kernel_blk_ntpb * p_token_size;
  const auto d_kernel_blk_size = dst_attn_kernel_blk_ntpb * d_token_size;
  // K and V each occupy half of the kernel block.
  const auto p_kv_section_size = p_kernel_blk_size / 2;
  const auto d_kv_section_size = d_kernel_blk_size / 2;
  assert(p_kv_section_size * 2 == p_kernel_blk_size);
  assert(d_kv_section_size * 2 == d_kernel_blk_size);
  // p_token_size = 2 * num_kv_heads * head_dim; derive head_dim in bytes.
  assert(num_kv_heads > 0);
  assert(p_token_size % (2 * static_cast<size_t>(num_kv_heads)) == 0);
  const size_t head_dim_size = p_token_size / 2 / static_cast<size_t>(num_kv_heads);
  // Bytes occupied by each head in one K or V side of a kernel block.
  const auto p_head_section_size = src_attn_kernel_blk_ntpb * head_dim_size;
  const auto d_head_section_size = dst_attn_kernel_blk_ntpb * head_dim_size;
  // K (or V) bytes occupied by each TP group in the destination kernel block.
  const auto per_group_kv_size = static_cast<size_t>(num_kv_heads) * d_head_section_size;
  assert(per_group_kv_size * attn_group_n == d_kv_section_size);

  while (left_tokens > 0) {
    // The KV manager allocates blocks using the original block size, while
    // attn_kernel_blk_ntpb reshapes the layout within each block. token_idx
    // is therefore relative to a kernel block rather than the whole block.
    // kernel_blk_idx is the kernel-block ID within the block.
    const auto p_blk_idx = wrote_tokens / p_ntpb;
    const auto p_kernel_blk_idx = (wrote_tokens - p_blk_idx * p_ntpb) / src_attn_kernel_blk_ntpb;
    const auto p_token_idx_base = wrote_tokens % src_attn_kernel_blk_ntpb;
    const auto d_blk_idx = wrote_tokens / d_ntpb;
    const auto d_kernel_blk_idx = (wrote_tokens - d_blk_idx * d_ntpb) / dst_attn_kernel_blk_ntpb;
    const auto d_token_idx_base = wrote_tokens % dst_attn_kernel_blk_ntpb;

    assert(p_blk_idx < src_blocks.size() && d_blk_idx < dst_blocks.size());
    const size_t p_blk_off = src_blocks.at(p_blk_idx) * p_block_size;
    const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;
    const size_t p_kernel_blk_off = p_blk_off + p_kernel_blk_idx * p_kernel_blk_size;
    const size_t d_kernel_blk_off = d_blk_off + d_kernel_blk_idx * d_kernel_blk_size;

    const auto tokens = std::min<uint32_t>(
      {src_attn_kernel_blk_ntpb - p_token_idx_base, dst_attn_kernel_blk_ntpb - d_token_idx_base, left_tokens}
    );

    // In HND layout, token data within each head is physically contiguous
    // in both source and destination.
    const auto length = static_cast<size_t>(tokens) * head_dim_size;
    for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
      const size_t pk_off = p_kernel_blk_off
                          + static_cast<size_t>(head_idx) * p_head_section_size
                          + p_token_idx_base * head_dim_size;
      const size_t pv_off = pk_off + p_kv_section_size;
      const size_t dk_off = d_kernel_blk_off
                          + attn_group_off * per_group_kv_size
                          + static_cast<size_t>(head_idx) * d_head_section_size
                          + d_token_idx_base * head_dim_size;
      const size_t dv_off = dk_off + d_kv_section_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          swap_offsets ? dk_off : pk_off,
          swap_offsets ? pk_off : dk_off, length);
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          swap_offsets ? dv_off : pv_off,
          swap_offsets ? pv_off : dv_off, length);
    }

    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

// Fill unused tail slots in the final qwen3-next FlashInfer HND decode block
// for P>D. Token-granularity transfer writes only seen+new tokens, so the
// final destination block contains total % d_ntpb initialized tokens and
// leaves [filled, d_ntpb) uninitialized.
//
// Prefill block 0 (src_blocks[0]) contains only zeros. Since the goal is to
// clear the unfilled tail of the final decode block, token-by-token parsing
// is unnecessary: tokens are contiguous within each head in HND layout, so
// one IpcBlock for each head's K and V tail is sufficient. Every source points
// at zero data at the beginning of block 0; any equal-length region would work.
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
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  if (total_tokens == 0) {
    return;
  }
  const uint32_t filled = total_tokens % d_ntpb;
  if (filled == 0) {
    // The final destination block is full; no padding is needed.
    return;
  }
  const uint32_t remaining = d_ntpb - filled;
  const uint32_t d_blk_idx = total_tokens / d_ntpb;  // Final partially filled block.
  assert(d_blk_idx < dst_blocks.size());
  assert(num_kv_heads > 0);
  assert(p_token_size % (2 * static_cast<size_t>(num_kv_heads)) == 0);

  const size_t head_dim_size = p_token_size / 2 / static_cast<size_t>(num_kv_heads);
  const size_t d_head_section_size = static_cast<size_t>(d_ntpb) * head_dim_size;
  const size_t d_kv_section_size = static_cast<size_t>(d_ntpb) * d_token_size / 2;
  const size_t per_group_kv_size = static_cast<size_t>(num_kv_heads) * d_head_section_size;
  const size_t length = static_cast<size_t>(remaining) * head_dim_size;

  // Block 0 contains only zeros, so always use its beginning.
  const size_t zero_src_off = src_blocks.at(0) * p_block_size;
  const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;

  for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
    const size_t dk_off = d_blk_off
                        + attn_group_off * per_group_kv_size
                        + static_cast<size_t>(head_idx) * d_head_section_size
                        + filled * head_dim_size;
    const size_t dv_off = dk_off + d_kv_section_size;
    // Each K/V tail is contiguous and overwritten with zeros. If one segment
    // exceeds the size of block 0 under an extremely asymmetric configuration,
    // split it by p_block_size so every source segment remains within block 0.
    emit_zero_fill_segments(
        zero_src_off, dk_off, length, p_block_size, bounds,
        per_cache_send_blocks);
    emit_zero_fill_segments(
        zero_src_off, dv_off, length, p_block_size, bounds,
        per_cache_send_blocks);
  }
}

// =============================================================================
// FLASHINFER_CACHE_SHAPE: P == D
// =============================================================================

void vllm_parse_flashinfer_block_send_p_eq_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  assert(kvt_tp_size == d_info->engine_tp_size);

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

  const int per_tp_kv_heads = (p_info->num_kv_heads > 0 &&
                               static_cast<uint32_t>(p_info->num_kv_heads) >= p_info->engine_tp_size)
                             ? p_info->num_kv_heads / static_cast<int>(p_info->engine_tp_size)
                             : std::max(1, p_info->num_kv_heads);

  send_blocks.resize(1);
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(*p_info, *d_info, 0);
  parse_flashinfer_HND_block(
    p_token_size, d_token_size,
    p_block_size, d_block_size,
    src_ntpb,
    src_blocks, dst_blocks,
    wrote_tokens, left_tokens,
    1, // attn_group_n
    0, // attn_group_off
    per_tp_kv_heads,
    bounds,
    false,
    per_cache_send_blocks
  );
}

// =============================================================================
// FLASHINFER_CACHE_SHAPE: P > D
// =============================================================================

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
  const uint32_t attn_group_n = valid_ranks.count() / d_info->engine_tp_size;
  const uint32_t attn_group_off = kvt_tp_rank % attn_group_n;
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;
  assert(kvt_tp_size > d_info->engine_tp_size);
  assert(kvt_tp_rank / attn_group_n == d_info->worker_tp_rank);

  // Should only have one cache in one layer.
  // Block Layout:Block[[K1,K2,K3...], [V1,V2,V3...]]
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

  const int per_tp_kv_heads = (p_info->num_kv_heads > 0 &&
                               static_cast<uint32_t>(p_info->num_kv_heads) >= p_info->engine_tp_size)
                             ? p_info->num_kv_heads / static_cast<int>(p_info->engine_tp_size)
                             : std::max(1, p_info->num_kv_heads);

  send_blocks.resize(1);
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(*p_info, *d_info, 0);
  parse_flashinfer_HND_block(
    p_token_size, d_token_size,
    p_block_size, d_block_size,
    src_ntpb,
    src_blocks, dst_blocks,
    wrote_tokens, left_tokens,
    attn_group_n,
    attn_group_off,
    per_tp_kv_heads,
    bounds,
    false,
    per_cache_send_blocks
  );
}

}  // namespace blade_llm
