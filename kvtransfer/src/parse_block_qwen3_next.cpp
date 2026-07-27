#include "parse_block_common.h"
#include "parse_block_qwen3_next_internal.h"
#include "envcfg.h"
#include "thrid_party/logging.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// =============================================================================
// Qwen3-next specific helper functions
// =============================================================================

// Block-aligned parse (send full blocks only) - only used by qwen3_next
void do_parse_block_aligned_send(
  size_t block_size,
  const ReqSendTask *task,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t ntpb,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // already cached tokens
  auto wrote_tokens = task->seen_tokens;
  // tokens to be written, aligned with block size, could oversend in last block
  auto left_tokens = task->new_tokens;
  while (left_tokens > 0) {
    const auto blk_idx = wrote_tokens / ntpb;
    const auto token_idx = wrote_tokens % ntpb;
    assert(blk_idx < src_blocks.size() && blk_idx < dst_blocks.size());
    const size_t p_blk_off = src_blocks.at(blk_idx) * block_size;
    const size_t d_blk_off = dst_blocks.at(blk_idx) * block_size;
    // PD block size should be same
    auto tokens = std::min(ntpb - token_idx, left_tokens);
    size_t length = 0;
    // Transfer whole blocks.
    if (tokens != ntpb){
      // request's last block or chunked prefill first/last block
      if (task->reach_last_token){ // request's last block
        length = block_size;
      } else if (token_idx == 0) {
        // chunked prefill last block
        length = 0;
      } else { // chunked prefill first block
        if (token_idx + tokens == ntpb) {
          // The range reaches a block boundary; transfer the full block.
          length = block_size;
        } else {
          assert(token_idx + tokens < ntpb);
          // The range starts and ends inside a block and the request is not
          // finished, so defer the transfer to the next step.
          length = 0;
        }
      }
    } else { // tokens == ntpb, send full block
      length = block_size;
    }
    if (length > 0) {
      append_ipc_block_checked(
          per_cache_send_blocks, bounds, p_blk_off, d_blk_off, length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }

  // For hybrid models with speculative decoding, P allocates extra blocks for
  // the gamma+1 popped tokens but does not compute KV into them. These blocks
  // are zeroed on P-side. Send them to D so that D's corresponding blocks
  // (which may contain stale Mamba NaN data) get overwritten with zeros,
  // preventing NaN in attention during decode.
  if (task->reach_last_token) {
    const size_t next_blk = (wrote_tokens + ntpb - 1) / ntpb;
    assert(src_blocks.size() == dst_blocks.size());
    const size_t num_blks = src_blocks.size();
    for (size_t i = next_blk; i < num_blks; i++) {
      const size_t p_last_blk_off = src_blocks.at(i) * block_size;
      const size_t d_last_blk_off = dst_blocks.at(i) * block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          p_last_blk_off, d_last_blk_off, block_size);
    }
  }
}

// used for qwen3-next's flash attn block parsing
//
// attn_pack_size > 1 layout (hybrid attn+mamba, kv_cache_spec.pack_size > 1):
//   Each physical page contains attn_pack_size per-layer attn blocks
//   back-to-back:
//
//     page i = src_blocks[i] * p_block_size:
//       [ pack 0 block i : K(N tok) | V(N tok) ]   <- per-layer block bytes
//       [ pack 1 block i : K | V ]
//       ...
//       [ pack (attn_pack_size-1) block i : K | V ]
//
//   block_sizes[0] passed in from vllm equals attn_pack_size *
//   per_layer_block_bytes, while attn_kernel_blk_ntpb is divided by
//   attn_pack_size in context.cpp::set_block_params so it stays equal to the
//   per-layer reduced block size in tokens. The request's seen_tokens /
//   new_tokens are measured per-layer (because vllm allocates one block_id
//   per attn_pack_size attn layers).
//
// attn_pack_size == 1 keeps the legacy semantics where one vllm block may
// internally contain multiple kernel blocks via the
// BLLM_KVTRANS_ATTN_KERNEL_BLK_SIZE env override.
static void parse_hybrid_attn_block(
  uint32_t src_ntpb, // ntpb: number tokens per block
  uint32_t dst_ntpb,
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t attn_wrote_tokens,
  uint32_t attn_left_tokens,
  uint32_t attn_group_off,
  uint32_t src_attn_kernel_blk_ntpb,
  uint32_t dst_attn_kernel_blk_ntpb,
  uint32_t attn_pack_size,
  const IpcBlockBounds& bounds,
  bool swap_offsets,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // Per-layer (one pack member) block sizes in bytes.
  const auto p_kernel_blk_size = src_attn_kernel_blk_ntpb * p_token_size;
  const auto d_kernel_blk_size = dst_attn_kernel_blk_ntpb * d_token_size;

  auto wrote_tokens = attn_wrote_tokens;
  auto left_tokens = attn_left_tokens;
  const uint32_t pack = attn_pack_size == 0 ? 1u : attn_pack_size;

  while (left_tokens > 0) {
    uint32_t p_blk_idx = 0;
    uint32_t p_token_idx_base = 0;
    uint32_t d_blk_idx = 0;
    uint32_t d_token_idx_base = 0;
    // Byte offset to the "first" kernel block inside the page (used by the
    // legacy env-override path; always 0 when pack > 1 because the pack_idx
    // loop below applies the per-pack sub-offset).
    size_t p_kernel_blk_off_base = 0;
    size_t d_kernel_blk_off_base = 0;
    uint32_t num_sub_blocks = 1;
    uint32_t tokens_this_round = 0;

    if (pack == 1) {
      // Legacy layout: one vllm block may contain multiple kernel blocks
      // when attn_kernel_blk_ntpb < src_ntpb (env override). seen_tokens /
      // new_tokens are measured against src_ntpb (vllm block tokens).
      p_blk_idx = wrote_tokens / src_ntpb;
      const auto p_kernel_blk_idx =
          (wrote_tokens - p_blk_idx * src_ntpb) / src_attn_kernel_blk_ntpb;
      p_token_idx_base = wrote_tokens % src_attn_kernel_blk_ntpb;
      d_blk_idx = wrote_tokens / dst_ntpb;
      const auto d_kernel_blk_idx =
          (wrote_tokens - d_blk_idx * dst_ntpb) / dst_attn_kernel_blk_ntpb;
      d_token_idx_base = wrote_tokens % dst_attn_kernel_blk_ntpb;
      p_kernel_blk_off_base = p_kernel_blk_idx * p_kernel_blk_size;
      d_kernel_blk_off_base = d_kernel_blk_idx * d_kernel_blk_size;
      num_sub_blocks = 1;
      tokens_this_round = std::min<uint32_t>(
        {src_attn_kernel_blk_ntpb - p_token_idx_base,
         dst_attn_kernel_blk_ntpb - d_token_idx_base,
         left_tokens}
      );
    } else {
      // attn_pack_size > 1: one physical page packs `pack` attn layers'
      // worth of kernel blocks. Per-layer the vllm block holds
      // `src_ntpb / pack` tokens (= num_blocks_per_kv_block kernel blocks).
      // Layers interleave at kernel-block granularity inside the page:
      //   page = [L0K0 | L1K0 | ... | L(pack-1)K0 |
      //           L0K1 | L1K1 | ... | L(pack-1)K1 | ...]
      // seen_tokens / new_tokens count per-layer tokens.
      const uint32_t per_layer_src_vllm_ntpb = src_ntpb / pack;
      const uint32_t per_layer_dst_vllm_ntpb = dst_ntpb / pack;
      assert(per_layer_src_vllm_ntpb * pack == src_ntpb);
      assert(per_layer_dst_vllm_ntpb * pack == dst_ntpb);
      assert(per_layer_src_vllm_ntpb % src_attn_kernel_blk_ntpb == 0);
      assert(per_layer_dst_vllm_ntpb % dst_attn_kernel_blk_ntpb == 0);

      p_blk_idx = wrote_tokens / per_layer_src_vllm_ntpb;
      d_blk_idx = wrote_tokens / per_layer_dst_vllm_ntpb;
      const uint32_t p_kernel_in_vllm =
          (wrote_tokens % per_layer_src_vllm_ntpb) / src_attn_kernel_blk_ntpb;
      const uint32_t d_kernel_in_vllm =
          (wrote_tokens % per_layer_dst_vllm_ntpb) / dst_attn_kernel_blk_ntpb;
      p_token_idx_base = wrote_tokens % src_attn_kernel_blk_ntpb;
      d_token_idx_base = wrote_tokens % dst_attn_kernel_blk_ntpb;
      // Within the page, kernel blocks of all `pack` layers at index k are
      // grouped before kernel blocks at index k+1. Layer `sub`'s kernel
      // block #k therefore lives at offset (k * pack + sub) * kernel_size.
      p_kernel_blk_off_base = p_kernel_in_vllm * pack * p_kernel_blk_size;
      d_kernel_blk_off_base = d_kernel_in_vllm * pack * d_kernel_blk_size;
      num_sub_blocks = pack;
      tokens_this_round = std::min<uint32_t>(
        {src_attn_kernel_blk_ntpb - p_token_idx_base,
         dst_attn_kernel_blk_ntpb - d_token_idx_base,
         left_tokens}
      );
    }

    assert(p_blk_idx < src_blocks.size() && d_blk_idx < dst_blocks.size());
    const size_t p_blk_off = src_blocks.at(p_blk_idx) * p_block_size;
    const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;

    // kv is not contiguous within a kernel/per-layer block: K then V.
    const auto p_token_step_length = p_token_size / 2;
    const auto d_token_step_length = d_token_size / 2;
    assert(p_token_step_length * 2 == p_token_size);
    assert(d_token_step_length * 2 == d_token_size);

    // Emit transfers for each pack member's sub-block within this physical
    // page. attn_pack_size == 1 collapses to a single iteration whose
    // sub-offset is 0, equivalent to the pre-pack behavior.
    for (uint32_t sub = 0; sub < num_sub_blocks; ++sub) {
      const size_t p_kernel_blk_off =
          p_blk_off + p_kernel_blk_off_base + sub * p_kernel_blk_size;
      const size_t d_kernel_blk_off =
          d_blk_off + d_kernel_blk_off_base + sub * d_kernel_blk_size;
      for (uint32_t idx = 0; idx < tokens_this_round; ++idx) {
        const uint32_t p_token_idx = p_token_idx_base + idx;
        const uint32_t d_token_idx = d_token_idx_base + idx;
        const size_t pk_token_off =
            p_kernel_blk_off + p_token_idx * p_token_step_length;
        const size_t pv_token_off = pk_token_off + p_kernel_blk_size / 2;
        const size_t d_token_off =
            d_kernel_blk_off + d_token_idx * d_token_step_length;
        const size_t dk_token_off =
            d_token_off + attn_group_off * p_token_step_length;
        const size_t dv_token_off = dk_token_off + d_kernel_blk_size / 2;
        append_ipc_block_checked(
            per_cache_send_blocks, bounds,
            swap_offsets ? dk_token_off : pk_token_off,
            swap_offsets ? pk_token_off : dk_token_off,
            p_token_step_length);
        append_ipc_block_checked(
            per_cache_send_blocks, bounds,
            swap_offsets ? dv_token_off : pv_token_off,
            swap_offsets ? pv_token_off : dv_token_off,
            p_token_step_length);
      }
    }
    wrote_tokens += tokens_this_round;
    left_tokens -= tokens_this_round;
  }
}

// Fill unused tail slots in the final qwen3-next FlashAttention decode block
// for P>D. After token-granularity transfer, only
// total % per_layer_dst_vllm_ntpb tokens have been written to the final
// per-layer destination vLLM block, so its tail may contain stale data.
//
// Request block 0 (src_blocks[0]) contains only zeros. Since the goal is only
// to clear the tail, every padding packet uses zero data from the beginning of
// block 0 and does not need a per-token source offset. In FlashAttention layout,
// destination tokens have gaps across TP groups and are not contiguous, so the
// destination still uses parse_hybrid_attn_block's pack, kernel-block, and K/V
// formulas to emit token-granularity packets.
static void fill_last_hybrid_attn_block(
  uint32_t src_ntpb,
  uint32_t dst_ntpb,
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t total_tokens,
  uint32_t attn_group_off,
  uint32_t src_attn_kernel_blk_ntpb,
  uint32_t dst_attn_kernel_blk_ntpb,
  uint32_t attn_pack_size,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  if (total_tokens == 0) {
    return;
  }
  const uint32_t pack = attn_pack_size == 0 ? 1u : attn_pack_size;
  const uint32_t per_layer_src_vllm_ntpb = (pack == 1) ? src_ntpb : src_ntpb / pack;
  const uint32_t per_layer_dst_vllm_ntpb = (pack == 1) ? dst_ntpb : dst_ntpb / pack;
  assert(per_layer_src_vllm_ntpb * pack == src_ntpb);
  assert(per_layer_dst_vllm_ntpb * pack == dst_ntpb);

  const uint32_t filled = total_tokens % per_layer_dst_vllm_ntpb;
  if (filled == 0) {
    // The final destination block is full; no padding is needed.
    return;
  }
  uint32_t to_fill = per_layer_dst_vllm_ntpb - filled;

  const auto d_kernel_blk_size = dst_attn_kernel_blk_ntpb * d_token_size;
  // KV is non-contiguous within a kernel/per-layer block: K comes before V.
  const auto p_token_step_length = p_token_size / 2;
  const auto d_token_step_length = d_token_size / 2;
  assert(p_token_step_length * 2 == p_token_size);
  assert(d_token_step_length * 2 == d_token_size);

  // Block 0 contains only zeros, so every padding packet starts from block 0.
  const size_t zero_src_off = src_blocks.at(0) * p_block_size;

  // Fill the destination sequentially from the first unused slot in the
  // final block.
  uint32_t dst_pos = total_tokens;
  uint32_t src_pos = 0;
  while (to_fill > 0) {
    uint32_t p_blk_idx = 0;
    uint32_t d_blk_idx = 0;
    uint32_t p_token_idx_base = 0;
    uint32_t d_token_idx_base = 0;
    size_t d_kernel_blk_off_base = 0;
    uint32_t num_sub_blocks = 1;

    if (pack == 1) {
      p_blk_idx = src_pos / src_ntpb;
      p_token_idx_base = src_pos % src_attn_kernel_blk_ntpb;
      d_blk_idx = dst_pos / dst_ntpb;
      const auto d_kernel_blk_idx =
          (dst_pos - d_blk_idx * dst_ntpb) / dst_attn_kernel_blk_ntpb;
      d_token_idx_base = dst_pos % dst_attn_kernel_blk_ntpb;
      d_kernel_blk_off_base = d_kernel_blk_idx * d_kernel_blk_size;
      num_sub_blocks = 1;
    } else {
      p_blk_idx = src_pos / per_layer_src_vllm_ntpb;
      d_blk_idx = dst_pos / per_layer_dst_vllm_ntpb;
      const uint32_t d_kernel_in_vllm =
          (dst_pos % per_layer_dst_vllm_ntpb) / dst_attn_kernel_blk_ntpb;
      p_token_idx_base = src_pos % src_attn_kernel_blk_ntpb;
      d_token_idx_base = dst_pos % dst_attn_kernel_blk_ntpb;
      d_kernel_blk_off_base = d_kernel_in_vllm * pack * d_kernel_blk_size;
      num_sub_blocks = pack;
    }

    // Padding data comes from request block 0.
    assert(p_blk_idx == 0);
    assert(d_blk_idx < dst_blocks.size());
    const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;

    const auto tokens = std::min<uint32_t>(
      {src_attn_kernel_blk_ntpb - p_token_idx_base,
       dst_attn_kernel_blk_ntpb - d_token_idx_base, to_fill}
    );

    for (uint32_t sub = 0; sub < num_sub_blocks; ++sub) {
      const size_t d_kernel_blk_off =
          d_blk_off + d_kernel_blk_off_base + sub * d_kernel_blk_size;
      for (uint32_t idx = 0; idx < tokens; ++idx) {
        const uint32_t d_token_idx = d_token_idx_base + idx;
        const size_t d_token_off =
            d_kernel_blk_off + d_token_idx * d_token_step_length;
        const size_t dk_token_off =
            d_token_off + attn_group_off * p_token_step_length;
        const size_t dv_token_off = dk_token_off + d_kernel_blk_size / 2;
        // Overwrite unused K/V slots with zero data from block 0. Destination
        // tokens are non-contiguous, so emit token-granularity packets. If a
        // segment exceeds block 0, split it by p_block_size to keep each source
        // segment in bounds.
        emit_zero_fill_segments(
            zero_src_off, dk_token_off, p_token_step_length, p_block_size,
            bounds,
            per_cache_send_blocks);
        emit_zero_fill_segments(
            zero_src_off, dv_token_off, p_token_step_length, p_block_size,
            bounds,
            per_cache_send_blocks);
      }
    }
    src_pos += tokens;
    dst_pos += tokens;
    to_fill -= tokens;
  }
}

void parse_hybrid_indexer_block(
  size_t p_block_size,
  size_t d_block_size,
  uint32_t indexer_wrote_tokens,
  uint32_t indexer_left_tokens,
  const std::vector<uint32_t>& src_indexer_blocks,
  const std::vector<uint32_t>& dst_indexer_blocks,
  uint32_t p_indexer_ntpb,
  uint32_t d_indexer_ntpb,
  uint32_t indexer_token_size,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(indexer_token_size > 0);

  auto wrote_tokens = indexer_wrote_tokens;
  auto left_tokens = indexer_left_tokens;
  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / p_indexer_ntpb;
    auto src_token_idx = wrote_tokens % p_indexer_ntpb;
    auto dst_block_idx = wrote_tokens / d_indexer_ntpb;
    auto dst_token_idx = wrote_tokens % d_indexer_ntpb;
    assert(src_block_idx < src_indexer_blocks.size());
    assert(dst_block_idx < dst_indexer_blocks.size());
    auto tokens = std::min<uint32_t>({
      p_indexer_ntpb - src_token_idx, d_indexer_ntpb - dst_token_idx, left_tokens
    });
    auto src_offset = src_indexer_blocks.at(src_block_idx) * p_block_size
                    + src_token_idx * indexer_token_size;
    auto dst_offset = dst_indexer_blocks.at(dst_block_idx) * d_block_size
                    + dst_token_idx * indexer_token_size;
    auto length = tokens * indexer_token_size;
    append_ipc_block_checked(
        per_cache_send_blocks, bounds, src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

// GDN block emit for P==D: send the entire padded GDN block as-is.
void parse_hybrid_gdn_block_send_p_eq_d(
  size_t block_size,
  size_t num_gdn_layers,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  for (size_t group_idx = 0; group_idx < num_gdn_layers; ++group_idx) {
    assert(group_idx < src_blocks.size() && group_idx < dst_blocks.size());
    const auto& src_grp = src_blocks.at(group_idx);
    const auto& dst_grp = dst_blocks.at(group_idx);
    assert(src_grp.size() == dst_grp.size());
    for (size_t block_idx = 0; block_idx < src_grp.size(); ++block_idx) {
      auto src_offset = src_grp.at(block_idx) * block_size;
      auto dst_offset = dst_grp.at(block_idx) * block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds, src_offset, dst_offset, block_size);
    }
  }
}

// GDN block emit for P>D: split conv state by channel dims, append ssm state,
// adjust dst offsets by gdn_group_n / gdn_group_off.
void parse_hybrid_gdn_block_send_p_gt_d(
  size_t p_block_size,
  size_t d_block_size,
  size_t num_gdn_layers,
  uint32_t gdn_group_n,
  uint32_t gdn_group_off,
  const WorkerInfo *src_worker_info,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(num_gdn_layers <= src_blocks.size());
  assert(num_gdn_layers <= dst_blocks.size());
  assert(!src_worker_info->conv_state_shape.empty());
  assert(!src_worker_info->ssm_state_shape.empty());
  assert(!src_worker_info->gdn_conv_channel_dims.empty());

  const auto &conv_state_shape = src_worker_info->conv_state_shape;
  const auto &ssm_state_shape = src_worker_info->ssm_state_shape;
  uint32_t gdn_conv_elem_size = src_worker_info->gdn_conv_elem_size;
  uint32_t gdn_ssm_elem_size = src_worker_info->gdn_ssm_elem_size;
  const auto &conv_channel_dims = src_worker_info->gdn_conv_channel_dims;

  const auto p_conv_dim_size = conv_state_shape.at(2) * gdn_conv_elem_size;
  const auto p_conv_block_size = conv_state_shape.at(1) * p_conv_dim_size;
  const auto p_ssm_block_size = ssm_state_shape.at(1) * ssm_state_shape.at(2)
                              * ssm_state_shape.at(3) * gdn_ssm_elem_size;
  const auto d_conv_dim_size = gdn_group_n * p_conv_dim_size;

  for (size_t group_idx = 0; group_idx < num_gdn_layers; ++group_idx) {
    const auto& src_gdn_blks = src_blocks.at(group_idx);
    const auto& dst_gdn_blks = dst_blocks.at(group_idx);
    for (size_t gdn_blk_idx = 0; gdn_blk_idx < src_gdn_blks.size(); ++gdn_blk_idx) {
      // GDN Block:[[Conv][SSM][Padding]]
      const size_t p_gdn_off = src_gdn_blks.at(gdn_blk_idx) * p_block_size;
      const size_t d_gdn_off = dst_gdn_blks.at(gdn_blk_idx) * d_block_size;
      // conv state is split along the last dimension, need to split into conv_state_shape[1] small packets
      for (size_t conv_dim = 0; conv_dim < conv_state_shape.at(1); ++conv_dim) {
        // conv dim will be split into q/k/v
        size_t conv_dim_inner_offset = 0;
        for (size_t conv_channel_dim = 0; conv_channel_dim < conv_channel_dims.size(); ++conv_channel_dim) {
          const auto conv_step_length = conv_channel_dims.at(conv_channel_dim) * gdn_conv_elem_size;
          const size_t p_conv_off = p_gdn_off + conv_dim * p_conv_dim_size + conv_dim_inner_offset;
          const size_t d_conv_off = d_gdn_off
                    + conv_dim * d_conv_dim_size
                    + gdn_group_n * conv_dim_inner_offset
                    + gdn_group_off * conv_step_length;
          append_ipc_block_checked(
              per_cache_send_blocks, bounds,
              p_conv_off, d_conv_off, conv_step_length);
          conv_dim_inner_offset += conv_step_length;
        }
        assert(conv_dim_inner_offset == p_conv_dim_size);
      }
      const size_t p_ssm_off = p_gdn_off + p_conv_block_size;
      const size_t d_ssm_off = d_gdn_off + gdn_group_n * p_conv_block_size + gdn_group_off * p_ssm_block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          p_ssm_off, d_ssm_off, p_ssm_block_size);
    }
  }
}


void parse_hybrid_short_conv_block_send_p_eq_d(
  size_t block_size,
  size_t num_ple_layers,
  size_t ple_block_group,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  if (num_ple_layers == 0) {
    return;
  }
  for (size_t i = 0; i < num_ple_layers; ++i) {
    const size_t group_idx = ple_block_group + i;
    assert(group_idx < src_blocks.size() && group_idx < dst_blocks.size());
    const auto& src_grp = src_blocks.at(group_idx);
    const auto& dst_grp = dst_blocks.at(group_idx);
    assert(src_grp.size() == dst_grp.size());
    for (size_t block_idx = 0; block_idx < src_grp.size(); ++block_idx) {
      const auto src_offset = src_grp.at(block_idx) * block_size;
      const auto dst_offset = dst_grp.at(block_idx) * block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds, src_offset, dst_offset, block_size);
    }
  }
}

void parse_hybrid_short_conv_block_send_p_gt_d(
  size_t p_block_size,
  size_t d_block_size,
  size_t num_ple_layers,
  size_t ple_block_group,
  uint32_t gdn_group_n,
  uint32_t gdn_group_off,
  const WorkerInfo *src_worker_info,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  if (num_ple_layers == 0) {
    return;
  }
  assert(!src_worker_info->ple_conv_state_shape.empty());
  assert(gdn_group_n > 0);
  assert(gdn_group_off < gdn_group_n);

  // Qwen4 PLE keeps a complete short-conv state on every TP rank. Unlike
  // GDN, its channel dimension is not TP-sharded, so only one representative
  // P rank in each P>D subgroup should copy the complete state to D.
  if (gdn_group_off != 0) {
    return;
  }

  const auto &conv_state_shape = src_worker_info->ple_conv_state_shape;
  assert(conv_state_shape.size() == 3);
  const uint32_t conv_elem_size = src_worker_info->ple_conv_elem_size;
  const auto conv_state_size = conv_state_shape.at(1)
                             * conv_state_shape.at(2)
                             * conv_elem_size;
  assert(conv_state_size <= p_block_size);
  assert(conv_state_size <= d_block_size);

  for (size_t i = 0; i < num_ple_layers; ++i) {
    const size_t group_idx = ple_block_group + i;
    assert(group_idx < src_blocks.size() && group_idx < dst_blocks.size());
    const auto& src_ple_blks = src_blocks.at(group_idx);
    const auto& dst_ple_blks = dst_blocks.at(group_idx);
    assert(src_ple_blks.size() == dst_ple_blks.size());
    for (size_t ple_blk_idx = 0; ple_blk_idx < src_ple_blks.size(); ++ple_blk_idx) {
      const size_t p_ple_off = src_ple_blks.at(ple_blk_idx) * p_block_size;
      const size_t d_ple_off = dst_ple_blks.at(ple_blk_idx) * d_block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          p_ple_off, d_ple_off, conv_state_size);
    }
  }
}

// GDN block emit for P<D: mirror of p_gt_d. P holds gdn_group_n = d_tp/p_tp
// D ranks' worth of GDN state per rank; this destination D rank receives the
// gdn_group_off-th slice. Uses only the src (P, larger) worker info shapes and
// derives D's smaller per-rank sizes, so it does not depend on the parsed dst
// worker info carrying GDN shapes. Caller invokes only on reach_last_token.
void parse_hybrid_gdn_block_send_p_lt_d(
  size_t p_block_size,
  size_t d_block_size,
  size_t num_gdn_layers,
  uint32_t gdn_group_n,
  uint32_t gdn_group_off,
  const WorkerInfo *src_worker_info,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(num_gdn_layers <= src_blocks.size());
  assert(num_gdn_layers <= dst_blocks.size());
  assert(!src_worker_info->conv_state_shape.empty());
  assert(!src_worker_info->ssm_state_shape.empty());
  assert(!src_worker_info->gdn_conv_channel_dims.empty());
  assert(gdn_group_n > 0);

  const auto &conv_state_shape = src_worker_info->conv_state_shape;
  const auto &ssm_state_shape = src_worker_info->ssm_state_shape;
  uint32_t gdn_conv_elem_size = src_worker_info->gdn_conv_elem_size;
  uint32_t gdn_ssm_elem_size = src_worker_info->gdn_ssm_elem_size;
  const auto &conv_channel_dims = src_worker_info->gdn_conv_channel_dims;

  // P (src) is the larger side here.
  const auto p_conv_dim_size = conv_state_shape.at(2) * gdn_conv_elem_size;
  const auto p_conv_block_size = conv_state_shape.at(1) * p_conv_dim_size;
  const auto p_ssm_block_size = ssm_state_shape.at(1) * ssm_state_shape.at(2)
                              * ssm_state_shape.at(3) * gdn_ssm_elem_size;
  assert(p_conv_dim_size % gdn_group_n == 0);
  assert(p_ssm_block_size % gdn_group_n == 0);
  const auto d_conv_dim_size = p_conv_dim_size / gdn_group_n;
  const auto d_conv_block_size = conv_state_shape.at(1) * d_conv_dim_size;
  const auto d_ssm_block_size = p_ssm_block_size / gdn_group_n;

  for (size_t group_idx = 0; group_idx < num_gdn_layers; ++group_idx) {
    const auto& src_gdn_blks = src_blocks.at(group_idx);
    const auto& dst_gdn_blks = dst_blocks.at(group_idx);
    for (size_t gdn_blk_idx = 0; gdn_blk_idx < src_gdn_blks.size(); ++gdn_blk_idx) {
      // GDN Block:[[Conv][SSM][Padding]]
      const size_t p_gdn_off = src_gdn_blks.at(gdn_blk_idx) * p_block_size;
      const size_t d_gdn_off = dst_gdn_blks.at(gdn_blk_idx) * d_block_size;
      for (size_t conv_dim = 0; conv_dim < conv_state_shape.at(1); ++conv_dim) {
        // Each q/k/v channel segment on P is gdn_group_n times the D segment;
        // send this D rank's gdn_group_off-th slice of every segment.
        size_t p_inner_offset = 0;  // offset within P's (large) conv row
        size_t d_inner_offset = 0;  // offset within D's (small) conv row
        for (size_t c = 0; c < conv_channel_dims.size(); ++c) {
          const auto p_step = conv_channel_dims.at(c) * gdn_conv_elem_size;
          assert(p_step % gdn_group_n == 0);
          const auto d_step = p_step / gdn_group_n;
          const size_t p_conv_off = p_gdn_off
                    + conv_dim * p_conv_dim_size
                    + p_inner_offset
                    + gdn_group_off * d_step;
          const size_t d_conv_off = d_gdn_off
                    + conv_dim * d_conv_dim_size
                    + d_inner_offset;
          append_ipc_block_checked(
              per_cache_send_blocks, bounds, p_conv_off, d_conv_off, d_step);
          p_inner_offset += p_step;
          d_inner_offset += d_step;
        }
        assert(p_inner_offset == p_conv_dim_size);
        assert(d_inner_offset == d_conv_dim_size);
      }
      const size_t p_ssm_off = p_gdn_off + p_conv_block_size
                             + gdn_group_off * d_ssm_block_size;
      const size_t d_ssm_off = d_gdn_off + d_conv_block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          p_ssm_off, d_ssm_off, d_ssm_block_size);
    }
  }
}

// PLE short-conv block emit for P<D. Qwen4 PLE state is replicated across TP
// ranks, so every destination D rank receives the complete state from its
// mapped P rank. Caller invokes only on reach_last_token.
void parse_hybrid_short_conv_block_send_p_lt_d(
  size_t p_block_size,
  size_t d_block_size,
  size_t num_ple_layers,
  size_t ple_block_group,
  uint32_t gdn_group_n,
  uint32_t gdn_group_off,
  const WorkerInfo *src_worker_info,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  if (num_ple_layers == 0) {
    return;
  }
  assert(!src_worker_info->ple_conv_state_shape.empty());
  assert(gdn_group_n > 0);
  assert(gdn_group_off < gdn_group_n);
  const auto &conv_state_shape = src_worker_info->ple_conv_state_shape;
  assert(conv_state_shape.size() == 3);
  const uint32_t conv_elem_size = src_worker_info->ple_conv_elem_size;
  const auto conv_state_size = conv_state_shape.at(1)
                             * conv_state_shape.at(2)
                             * conv_elem_size;
  assert(conv_state_size <= p_block_size);
  assert(conv_state_size <= d_block_size);

  for (size_t i = 0; i < num_ple_layers; ++i) {
    const size_t group_idx = ple_block_group + i;
    assert(group_idx < src_blocks.size() && group_idx < dst_blocks.size());
    const auto& src_ple_blks = src_blocks.at(group_idx);
    const auto& dst_ple_blks = dst_blocks.at(group_idx);
    assert(src_ple_blks.size() == dst_ple_blks.size());
    for (size_t ple_blk_idx = 0; ple_blk_idx < src_ple_blks.size(); ++ple_blk_idx) {
      const size_t p_ple_off = src_ple_blks.at(ple_blk_idx) * p_block_size;
      const size_t d_ple_off = dst_ple_blks.at(ple_blk_idx) * d_block_size;
      append_ipc_block_checked(
          per_cache_send_blocks, bounds,
          p_ple_off, d_ple_off, conv_state_size);
    }
  }
}

// NOTE(llx) Qwen3-next block parsing special case:
// - The last three blocks are reserved for GDN blocks
// - KVT length calculation must account for these additional blocks: (new_tokens + 3 * ntpb)
// - During chunked prefill processing, GDN blocks transfer occurs only upon request completion
static void do_parse_hybrid_block_send_p_gt_d(
  size_t p_token_size, size_t d_token_size,
  uint32_t src_ntpb, // ntpb: number tokens per block
  uint32_t dst_ntpb,
  uint32_t attn_group_n,
  uint32_t attn_group_off, // attn offset in pd rank mapping group
  uint32_t gdn_group_n,
  uint32_t gdn_group_off, // gdn offset in pd rank mapping group
  const std::bitset<MAX_TP_SIZE>& validranks,
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // Next's p block size might not equal to d block size
  const size_t p_block_size = p_token_size * src_ntpb;
  const size_t d_block_size = d_token_size * dst_ntpb;
  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const auto p_indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  const auto d_indexer_ntpb = dst_worker_info->indexer_blk_ntpb;

  auto p_tp_rank = src_worker_info->worker_tp_rank;
  auto src_attn_kernel_blk_ntpb = src_worker_info->attn_kernel_blk_ntpb;
  auto dst_attn_kernel_blk_ntpb = dst_worker_info->attn_kernel_blk_ntpb;
  const uint32_t src_attn_pack_size =
      src_worker_info->attn_pack_size == 0 ? 1u : src_worker_info->attn_pack_size;
  const uint32_t dst_attn_pack_size =
      dst_worker_info->attn_pack_size == 0 ? 1u : dst_worker_info->attn_pack_size;
  // PD must agree on attn_pack_size when packing is enabled.
  assert(src_attn_pack_size == dst_attn_pack_size);

  // already cached tokens
  auto wrote_tokens = task->seen_tokens;
  // tokens to be written, aligned with block size, could oversend in last block
  auto left_tokens = task->new_tokens;

  auto indexer_wrote_tokens = wrote_tokens;
  auto indexer_left_tokens = left_tokens;

  auto attn_wrote_tokens = wrote_tokens;
  auto attn_left_tokens = left_tokens;

  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  assert(num_gdn_layers < src_blocks.size());
  assert(num_gdn_layers < dst_blocks.size());

  //  need to recalculate the offset of gdn, because pd block size is different
  // GDN blocks will be update in-place, so send it when prefill is finished
  if (task->reach_last_token) {
    parse_hybrid_gdn_block_send_p_gt_d(
      p_block_size, d_block_size, num_gdn_layers,
      gdn_group_n, gdn_group_off,
      src_worker_info, src_blocks, dst_blocks,
      bounds,
      per_cache_send_blocks
    );
    parse_hybrid_short_conv_block_send_p_gt_d(
      p_block_size, d_block_size,
      src_worker_info->num_ple_layers,
      src_worker_info->ple_block_group,
      gdn_group_n, gdn_group_off,
      src_worker_info, src_blocks, dst_blocks,
      bounds,
      per_cache_send_blocks
    );
  }
  auto attn_group_offset = num_gdn_layers;
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());
  assert((p_indexer_ntpb == 0 && d_indexer_ntpb == 0) || (p_indexer_ntpb > 0 && d_indexer_ntpb > 0));

  if (p_indexer_ntpb > 0 and d_indexer_ntpb > 0) {
    // Indexer cache is replicated within each D rank. In P>D, choose the
    // first P rank in every P-rank subgroup that maps to a D rank, so each
    // D worker receives one complete indexer cache copy.
    if (p_tp_rank % gdn_group_n == 0) {
      const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
      const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
      parse_hybrid_indexer_block(
        p_block_size,
        d_block_size,
        indexer_wrote_tokens,
        indexer_left_tokens,
        src_indexer_blks,
        dst_indexer_blks,
        p_indexer_ntpb,
        d_indexer_ntpb,
        src_worker_info->hybrid_indexer_token_size,
        bounds,
        per_cache_send_blocks
      );
    }
    attn_group_offset += 1;
  }
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());
  if (validranks[p_tp_rank]) {
    const auto& src_attn_blks = src_blocks.at(attn_group_offset);
    const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);
    parse_hybrid_attn_block(
      src_ntpb,
      dst_ntpb,
      p_token_size,
      d_token_size,
      p_block_size,
      d_block_size,
      src_attn_blks,
      dst_attn_blks,
      attn_wrote_tokens,
      attn_left_tokens,
      attn_group_off,
      src_attn_kernel_blk_ntpb,
      dst_attn_kernel_blk_ntpb,
      src_attn_pack_size,
      bounds,
      false,
      per_cache_send_blocks
    );
    // When the request completes, fill the unused tail of the final decode
    // attention block with block 0.
    if (env_pad_last_attn_block() && task->reach_last_token) {
      fill_last_hybrid_attn_block(
        src_ntpb,
        dst_ntpb,
        p_token_size,
        d_token_size,
        p_block_size,
        d_block_size,
        src_attn_blks,
        dst_attn_blks,
        wrote_tokens + left_tokens,
        attn_group_off,
        src_attn_kernel_blk_ntpb,
        dst_attn_kernel_blk_ntpb,
        src_attn_pack_size,
        bounds,
        per_cache_send_blocks
      );
    }
  }
}

// =============================================================================
// QWEN3_NEXT_FLASH_CACHE_SHAPE: P == D (block aligned)
// =============================================================================

void vllm_parse_hybrid_block_send_p_eq_d_block_aligned(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // QWEN3_NEXT selects parse_block by engine_tp_size (GDN grouping is
  // engine-tp-based). When this P==D path is taken, engine_tp_size on both
  // sides must match -- which also implies the effective kvt_tp_size == D
  // (since p_tp == d_tp means valid_ranks selects all P ranks).
  assert(src_worker_info->engine_tp_size == dst_worker_info->engine_tp_size);
  assert(kvt_tp_size == dst_worker_info->engine_tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  auto const& kv_token_sizes = src_worker_info->token_sizes;
  auto const& kv_block_sizes = src_worker_info->block_sizes;
  assert(kv_token_sizes == dst_worker_info->token_sizes);
  assert(kv_block_sizes == dst_worker_info->block_sizes);
  assert(kv_token_sizes.size() == kv_block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());
  // Should only have one cache in one layer for QWEN3_NEXT_FLASH_CACHE_SHAPE.
  assert(kv_token_sizes.size() == 1);
  assert(kv_block_sizes.size() == 1);
  send_blocks.resize(kv_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);
  // LOG(INFO) << "vllm_parse_hybrid_block_send_p_eq_d_block_aligned: kv_block_sizes[0]: " << kv_block_sizes.at(0)
  //     << " kv_token_sizes[0]: " << kv_token_sizes.at(0) << " task->new_tokens: " << task->new_tokens
  //     << " task->seen_tokens: " << task->seen_tokens << " task->src_blocks.size(): " << task->src_blocks().size()
  //     << " task->dst_blocks.size(): " << task->dst_blocks().size();

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  size_t attn_group_offset = num_gdn_layers;

  // For P==D block-aligned send, each iteration transfers one full physical
  // page worth of bytes (length == block_size). vllm allocates src_blocks at
  // per-layer block granularity and task->seen_tokens / new_tokens are
  // measured per-layer, so the loop iterator must step per per-layer vllm
  // block (= block_size / token_size / attn_pack_size). attn_kernel_blk_ntpb
  // may be smaller (e.g. when BLLM_KVTRANS_ATTN_KERNEL_BLK_SIZE forces a
  // sub-block size), which would over-iterate and walk off src_blocks.
  const uint32_t pack = src_worker_info->attn_pack_size == 0
                            ? 1u
                            : src_worker_info->attn_pack_size;
  assert((kv_block_sizes.at(0) / kv_token_sizes.at(0)) % pack == 0);
  auto attn_ntpb = kv_block_sizes.at(0) / kv_token_sizes.at(0) / pack;
  auto indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  assert(src_worker_info->attn_kernel_blk_ntpb == dst_worker_info->attn_kernel_blk_ntpb);
  assert(indexer_ntpb == dst_worker_info->indexer_blk_ntpb);
  assert(src_worker_info->attn_pack_size == dst_worker_info->attn_pack_size);

  if (indexer_ntpb > 0) {
    const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
    const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
    // Parse Indexer cache block and full attn block
    do_parse_block_aligned_send(
      kv_block_sizes.at(0),
      task,
      src_indexer_blks, dst_indexer_blks,
      indexer_ntpb, bounds, per_cache_send_blocks
    );
    attn_group_offset += 1;
  }
  const auto& src_attn_blks = src_blocks.at(attn_group_offset);
  const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);
  do_parse_block_aligned_send(
    kv_block_sizes.at(0),
    task,
    src_attn_blks, dst_attn_blks,
    attn_ntpb, bounds, per_cache_send_blocks
  );

  if (task->reach_last_token) {
    // Now gdn layers are before indexer and attention layers
    parse_hybrid_gdn_block_send_p_eq_d(
      kv_block_sizes.at(0), num_gdn_layers,
      src_blocks, dst_blocks, bounds, per_cache_send_blocks
    );
    parse_hybrid_short_conv_block_send_p_eq_d(
      kv_block_sizes.at(0),
      src_worker_info->num_ple_layers,
      src_worker_info->ple_block_group,
      src_blocks, dst_blocks, bounds, per_cache_send_blocks
    );
  }
}

// =============================================================================
// QWEN3_NEXT_FLASH_CACHE_SHAPE: P > D
// =============================================================================

// Parse block using shape QWEN3_NEXT_FLASH_CACHE_SHAPE in vllm
// Qwen3-next model's attn block and mamba block is block continous and padded
// in same block size, so kvt will parse it using vllm's token_size and block_size
void vllm_parse_hybrid_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  const auto & p_block_sizes = src_worker_info->block_sizes;
  const auto & d_block_sizes = dst_worker_info->block_sizes;
  const auto & p_token_sizes = src_worker_info->token_sizes;
  const auto & d_token_sizes = dst_worker_info->token_sizes;

  // GDN grouping must use engine_tp_size on both sides: GDN is unaffected
  // by valid_ranks (which is num_kv_heads-driven and only applies to attn).
  assert(src_worker_info->engine_tp_size > dst_worker_info->engine_tp_size);
  assert(src_worker_info->engine_tp_size % dst_worker_info->engine_tp_size == 0);
  const uint32_t p_origin_tp_size = src_worker_info->engine_tp_size;
  const uint32_t gdn_group_n = p_origin_tp_size / dst_worker_info->engine_tp_size;

  assert(src_worker_info->worker_tp_rank / gdn_group_n == dst_worker_info->worker_tp_rank);
  const uint32_t gdn_group_off = src_worker_info->worker_tp_rank % gdn_group_n;
  const uint32_t attn_group_n = valid_ranks.count() / dst_worker_info->engine_tp_size;
  const uint32_t attn_group_off = kvt_tp_rank % attn_group_n;

  // Should only have one cache in one layer.
  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);
  assert(d_token_size == p_token_size * attn_group_n);

  // ntpb: number tokens per block, pd block size should align
  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb * p_token_size == p_block_size);
  assert(dst_ntpb * d_token_size == d_block_size);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);
  do_parse_hybrid_block_send_p_gt_d(
    p_token_size, d_token_size,
    src_ntpb, dst_ntpb,
    attn_group_n, attn_group_off,
    gdn_group_n, gdn_group_off,
    valid_ranks,
    src_worker_info,
    dst_worker_info,
    task, bounds, per_cache_send_blocks
  );
}

// Zero-fill the tail of the D-side attention blocks for the sliced P<D
// regimes (token-granularity attn transfer does not cover the padded tail nor
// the spec-decode trailing zero blocks, unlike the block-aligned copy).
// Fills every dst token slot in [total_tokens, dst_blocks.size() * per-layer
// block tokens). Unlike P>D, the dst K/V rows have no TP-group holes here, so
// each (kernel block, K/V) tail is one contiguous segment. Source is the
// request's block 0 (all zeros); emit_zero_fill_segments splits segments
// longer than p_block_size so the src read never leaves block 0.
static void fill_tail_hybrid_attn_block_p_lt_d(
  size_t p_block_size,
  size_t d_block_size,
  size_t d_token_size,
  uint32_t dst_ntpb,
  uint32_t dst_attn_kernel_blk_ntpb,
  uint32_t attn_pack_size,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t total_tokens,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  const uint32_t pack = attn_pack_size == 0 ? 1u : attn_pack_size;
  const uint32_t per_layer_dst_vllm_ntpb = (pack == 1) ? dst_ntpb : dst_ntpb / pack;
  assert(per_layer_dst_vllm_ntpb * pack == dst_ntpb);
  assert(per_layer_dst_vllm_ntpb % dst_attn_kernel_blk_ntpb == 0);
  const size_t d_kernel_blk_size = dst_attn_kernel_blk_ntpb * d_token_size;
  const size_t d_token_step = d_token_size / 2;
  assert(d_token_step * 2 == d_token_size);

  const uint32_t end_tokens =
      static_cast<uint32_t>(dst_blocks.size()) * per_layer_dst_vllm_ntpb;
  if (total_tokens >= end_tokens) {
    return;
  }
  const size_t zero_src_off = src_blocks.at(0) * p_block_size;

  uint32_t dst_pos = total_tokens;
  while (dst_pos < end_tokens) {
    const uint32_t d_blk_idx = dst_pos / per_layer_dst_vllm_ntpb;
    const uint32_t d_kernel_in_vllm =
        (dst_pos % per_layer_dst_vllm_ntpb) / dst_attn_kernel_blk_ntpb;
    const uint32_t d_token_idx = dst_pos % dst_attn_kernel_blk_ntpb;
    const uint32_t tokens = std::min<uint32_t>(
        dst_attn_kernel_blk_ntpb - d_token_idx, end_tokens - dst_pos);
    assert(d_blk_idx < dst_blocks.size());
    const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;
    // pack > 1: layer `sub`'s kernel block #k lives at (k * pack + sub) *
    // kernel_size within the page; pack == 1 collapses to k * kernel_size.
    const size_t d_kernel_off_base =
        static_cast<size_t>(d_kernel_in_vllm) * pack * d_kernel_blk_size;
    const size_t length = static_cast<size_t>(tokens) * d_token_step;
    for (uint32_t sub = 0; sub < pack; ++sub) {
      const size_t d_kernel_blk_off =
          d_blk_off + d_kernel_off_base + sub * d_kernel_blk_size;
      const size_t dk_off = d_kernel_blk_off + d_token_idx * d_token_step;
      const size_t dv_off = dk_off + d_kernel_blk_size / 2;
      emit_zero_fill_segments(
          zero_src_off, dk_off, length, p_block_size, bounds,
          per_cache_send_blocks);
      emit_zero_fill_segments(
          zero_src_off, dv_off, length, p_block_size, bounds,
          per_cache_send_blocks);
    }
    dst_pos += tokens;
  }
}

// =============================================================================
// QWEN3_NEXT_FLASH_CACHE_SHAPE: P < D
// =============================================================================
//
// One P rank fans out to gdn_group_n = d_tp / p_tp D ranks (the caller in
// tx_stub issues one stub per D rank). Attention/indexer handling depends on
// the head regime:
//   - num_kv_heads <= p_tp (replicated): P and D hold identical per-rank
//     physical blocks; reuse the block-aligned full-block copy (which also
//     zero-fills the padded tail on reach_last_token).
//   - num_kv_heads > p_tp (mixed p_tp < heads < d_tp, or fully head-split
//     heads >= d_tp): the P row is attn_group_n = p_token/d_token times the D
//     row; each fan-out target receives its attn_group_off-th head sub-slice
//     of every token row (mirror of P>D where the offset is on the dst side).
//     The indexer stays replicated (full copy per target) and the padded tail
//     plus spec-decode trailing blocks are zero-filled explicitly.
// GDN/short-conv are split by engine_tp in all regimes, so this D rank gets
// the gdn_group_off-th slice of P's state.
void vllm_parse_hybrid_block_send_p_lt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  const uint32_t p_engine_tp = src_worker_info->engine_tp_size;
  const uint32_t d_engine_tp = dst_worker_info->engine_tp_size;
  assert(p_engine_tp < d_engine_tp);
  assert(d_engine_tp % p_engine_tp == 0);
  const uint32_t gdn_group_n = d_engine_tp / p_engine_tp;
  const uint32_t gdn_group_off = dst_worker_info->worker_tp_rank % gdn_group_n;
  assert(dst_worker_info->worker_tp_rank / gdn_group_n
         == src_worker_info->worker_tp_rank);

  const auto &p_token_sizes = src_worker_info->token_sizes;
  const auto &p_block_sizes = src_worker_info->block_sizes;
  const auto &d_token_sizes = dst_worker_info->token_sizes;
  const auto &d_block_sizes = dst_worker_info->block_sizes;
  assert(p_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  size_t attn_group_offset = num_gdn_layers;

  const uint32_t pack = src_worker_info->attn_pack_size == 0
                            ? 1u
                            : src_worker_info->attn_pack_size;
  const auto p_indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  const auto d_indexer_ntpb = dst_worker_info->indexer_blk_ntpb;
  assert((p_indexer_ntpb == 0 && d_indexer_ntpb == 0) ||
         (p_indexer_ntpb > 0 && d_indexer_ntpb > 0));
  assert(src_worker_info->attn_pack_size == dst_worker_info->attn_pack_size);

  if (p_token_size == d_token_size) {
    // Replicated-head regime (num_kv_heads <= p_tp): attn/indexer per-rank
    // layouts are identical.
    assert(p_block_size == d_block_size);
    assert((p_block_size / p_token_size) % pack == 0);
    const auto attn_ntpb = p_block_size / p_token_size / pack;
    assert(src_worker_info->attn_kernel_blk_ntpb
           == dst_worker_info->attn_kernel_blk_ntpb);
    assert(p_indexer_ntpb == d_indexer_ntpb);

    // Indexer: replicated, every fan-out target gets a full copy.
    if (p_indexer_ntpb > 0) {
      const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
      const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
      do_parse_block_aligned_send(
        p_block_size, task,
        src_indexer_blks, dst_indexer_blks,
        p_indexer_ntpb, bounds, per_cache_send_blocks
      );
      attn_group_offset += 1;
    }
    // Attention: replicated, full-block copy (also zero-fills padded tail).
    const auto& src_attn_blks = src_blocks.at(attn_group_offset);
    const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);
    do_parse_block_aligned_send(
      p_block_size, task,
      src_attn_blks, dst_attn_blks,
      attn_ntpb, bounds, per_cache_send_blocks
    );
  } else {
    // Sliced regime (num_kv_heads > p_tp): the P token row is attn_group_n
    // times the D row. Head sub-slicing:
    //   attn_group_n = p_token/d_token  (heads/p_tp in mixed, d_tp/p_tp in
    //                                    fully head-split)
    //   replicas     = D ranks sharing the same head slice
    //                = d_tp / (p_tp * attn_group_n)  (d_tp/heads in mixed, 1
    //                                                 in fully head-split)
    //   attn_group_off = (d_rank / replicas) % attn_group_n
    assert(p_token_size > d_token_size);
    assert(p_token_size % d_token_size == 0);
    const uint32_t attn_group_n =
        static_cast<uint32_t>(p_token_size / d_token_size);
    assert(d_engine_tp % (p_engine_tp * attn_group_n) == 0);
    const uint32_t replicas = d_engine_tp / (p_engine_tp * attn_group_n);
    const uint32_t attn_group_off =
        (dst_worker_info->worker_tp_rank / replicas) % attn_group_n;
    assert(src_worker_info->num_kv_heads <= 0 ||
           static_cast<uint32_t>(src_worker_info->num_kv_heads) % p_engine_tp
               == 0);

    const uint32_t src_ntpb = p_block_size / p_token_size;
    const uint32_t dst_ntpb = d_block_size / d_token_size;
    assert(src_ntpb * p_token_size == p_block_size);
    assert(dst_ntpb * d_token_size == d_block_size);

    // Indexer: replicated across TP ranks, every fan-out target gets a full
    // copy. Block byte sizes differ between P and D, so use the generic
    // token-granularity indexer parse instead of the block-aligned copy.
    if (p_indexer_ntpb > 0) {
      const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
      const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
      parse_hybrid_indexer_block(
        p_block_size, d_block_size,
        task->seen_tokens, task->new_tokens,
        src_indexer_blks, dst_indexer_blks,
        p_indexer_ntpb, d_indexer_ntpb,
        src_worker_info->hybrid_indexer_token_size,
        bounds,
        per_cache_send_blocks
      );
      attn_group_offset += 1;
    }
    assert(attn_group_offset < src_blocks.size());
    assert(attn_group_offset < dst_blocks.size());
    const auto& src_attn_blks = src_blocks.at(attn_group_offset);
    const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);

    // Attention: P<D is the exact mirror of P>D -- reuse parse_hybrid_attn_
    // block with (p, d) swapped so the "wide" side (here P) receives the
    // group offset, then swap src/dst offsets of the emitted records.
    parse_hybrid_attn_block(
      dst_ntpb,               // "src" ntpb  <- actual D
      src_ntpb,               // "dst" ntpb  <- actual P
      d_token_size, p_token_size,
      d_block_size, p_block_size,
      dst_attn_blks, src_attn_blks,
      task->seen_tokens, task->new_tokens,
      attn_group_off,
      dst_worker_info->attn_kernel_blk_ntpb,
      src_worker_info->attn_kernel_blk_ntpb,
      pack,
      bounds,
      true,
      per_cache_send_blocks
    );

    // Token-granularity transfer covers neither the padded tail of the last
    // dst block nor the spec-decode trailing zero blocks (the block-aligned
    // copy in the replicated branch does both implicitly). Zero-fill them
    // from the request's block 0 on request completion.
    if (env_pad_last_attn_block() && task->reach_last_token) {
      fill_tail_hybrid_attn_block_p_lt_d(
        p_block_size, d_block_size, d_token_size,
        dst_ntpb,
        dst_worker_info->attn_kernel_blk_ntpb,
        pack,
        src_attn_blks, dst_attn_blks,
        task->seen_tokens + task->new_tokens,
        bounds,
        per_cache_send_blocks
      );
    }
  }

  // GDN / short-conv: engine-tp sub-slice, emitted on request completion.
  if (task->reach_last_token) {
    parse_hybrid_gdn_block_send_p_lt_d(
      p_block_size, d_block_size, num_gdn_layers,
      gdn_group_n, gdn_group_off,
      src_worker_info, src_blocks, dst_blocks, bounds,
      per_cache_send_blocks
    );
    parse_hybrid_short_conv_block_send_p_lt_d(
      p_block_size, d_block_size,
      src_worker_info->num_ple_layers,
      src_worker_info->ple_block_group,
      gdn_group_n, gdn_group_off,
      src_worker_info, src_blocks, dst_blocks, bounds,
      per_cache_send_blocks
    );
  }
}

}  // namespace blade_llm
