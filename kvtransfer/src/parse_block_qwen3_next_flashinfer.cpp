#include "parse_block_common.h"
#include "parse_block_flashinfer_internal.h"
#include "parse_block_qwen3_next_internal.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

namespace {

int per_tp_kv_heads(const WorkerInfo *info) noexcept {
  return std::max(
      1,
      info->num_kv_heads / static_cast<int>(info->engine_tp_size));
}

}  // namespace

// =============================================================================
// QWEN3_NEXT_FLASHINFER_CACHE_SHAPE
//
// Hybrid model (GDN + optional indexer + attention) where the attention layer
// is stored in FlashInfer HND physical layout:
//   (num_blocks, 2, num_kv_heads, block_size, head_dim)
//
// BlockIds groups (same as QWEN3_NEXT):
//   [gdn_layer_0, ..., gdn_layer_{N-1}, indexer?, attn]
// =============================================================================

// =============================================================================
// QWEN3_NEXT_FLASHINFER_CACHE_SHAPE: P == D
// =============================================================================

void vllm_parse_qwen3_next_flashinfer_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // QWEN3_NEXT_FLASHINFER selects parse_block by engine_tp_size for the same
  // reason as QWEN3_NEXT_FLASH (GDN grouping is engine-tp-based).
  assert(src_worker_info->engine_tp_size == dst_worker_info->engine_tp_size);
  assert(kvt_tp_size == dst_worker_info->engine_tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);

  const auto &kv_token_sizes = src_worker_info->token_sizes;
  const auto &kv_block_sizes = src_worker_info->block_sizes;
  assert(kv_token_sizes == dst_worker_info->token_sizes);
  assert(kv_block_sizes == dst_worker_info->block_sizes);
  // Should only have one cache in one layer for the hybrid+flashinfer shape.
  assert(kv_token_sizes.size() == 1);
  assert(kv_block_sizes.size() == 1);

  const auto p_token_size = kv_token_sizes.at(0);
  const auto d_token_size = dst_worker_info->token_sizes.at(0);
  const auto p_block_size = kv_block_sizes.at(0);
  const auto d_block_size = dst_worker_info->block_sizes.at(0);
  const uint32_t ntpb = p_block_size / p_token_size;
  assert(ntpb * p_token_size == p_block_size);
  // FlashInfer attn does not use sub-block kernel reshape, so kernel_blk == ntpb.
  assert(src_worker_info->attn_kernel_blk_ntpb == ntpb);
  assert(dst_worker_info->attn_kernel_blk_ntpb == ntpb);

  send_blocks.resize(kv_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  size_t attn_group_offset = num_gdn_layers;
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());

  const auto indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  assert(indexer_ntpb == dst_worker_info->indexer_blk_ntpb);

  // 1. Indexer (if present): block-aligned send (rank-replicated cache).
  if (indexer_ntpb > 0) {
    const auto& src_indexer_blks = src_blocks.at(attn_group_offset);
    const auto& dst_indexer_blks = dst_blocks.at(attn_group_offset);
    do_parse_block_aligned_send(
      p_block_size, task,
      src_indexer_blks, dst_indexer_blks,
      indexer_ntpb, bounds, per_cache_send_blocks
    );
    attn_group_offset += 1;
  }
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());

  // 2. Attention: P==D, src/dst physical layouts are identical, so we just
  //    emit full-block transfers (block-aligned send) instead of splitting by
  //    head/token. This mirrors qwen3_next P==D's attn handling and avoids any
  //    HND-layout-specific offset math.
  const auto &src_attn_blks = src_blocks.at(attn_group_offset);
  const auto &dst_attn_blks = dst_blocks.at(attn_group_offset);
  do_parse_block_aligned_send(
    p_block_size, task,
    src_attn_blks, dst_attn_blks,
    ntpb, bounds, per_cache_send_blocks
  );

  // 3. GDN: emit only on the last token (after attn, mirroring the existing
  // QWEN3_NEXT P==D ordering so downstream send queue layout matches).
  if (task->reach_last_token) {
    parse_hybrid_gdn_block_send_p_eq_d(
      p_block_size, num_gdn_layers,
      src_blocks, dst_blocks, bounds, per_cache_send_blocks
    );
    parse_hybrid_short_conv_block_send_p_eq_d(
      p_block_size,
      src_worker_info->num_ple_layers,
      src_worker_info->ple_block_group,
      src_blocks, dst_blocks, bounds, per_cache_send_blocks
    );
  }
}

// =============================================================================
// QWEN3_NEXT_FLASHINFER_CACHE_SHAPE: P > D
// =============================================================================

void vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // GDN groups are derived from engine_tp_size on both sides (GDN is
  // num_kv_heads-oblivious). Attention groups are derived from
  // valid_ranks.count() (effective P kvt_tp_size) / dst engine_tp_size.
  const uint32_t p_engine_tp = src_worker_info->engine_tp_size;
  const uint32_t d_engine_tp = dst_worker_info->engine_tp_size;
  assert(p_engine_tp > d_engine_tp);
  assert(p_engine_tp % d_engine_tp == 0);
  const uint32_t gdn_group_n = p_engine_tp / d_engine_tp;
  const uint32_t gdn_group_off = src_worker_info->worker_tp_rank % gdn_group_n;
  assert(src_worker_info->worker_tp_rank / gdn_group_n == dst_worker_info->worker_tp_rank);

  const uint32_t attn_group_n = valid_ranks.count() / d_engine_tp;
  const uint32_t attn_group_off = (attn_group_n > 0) ? (kvt_tp_rank % attn_group_n) : 0;

  const auto &p_block_sizes = src_worker_info->block_sizes;
  const auto &d_block_sizes = dst_worker_info->block_sizes;
  const auto &p_token_sizes = src_worker_info->token_sizes;
  const auto &d_token_sizes = dst_worker_info->token_sizes;
  // Should only have one cache in one layer.
  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);

  const auto p_token_size = p_token_sizes.at(0);
  const auto d_token_size = d_token_sizes.at(0);
  const auto p_block_size = p_block_sizes.at(0);
  const auto d_block_size = d_block_sizes.at(0);
  const int src_heads_per_rank = per_tp_kv_heads(src_worker_info);

  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb * p_token_size == p_block_size);
  assert(dst_ntpb * d_token_size == d_block_size);
  // FlashInfer page size matches kv-manager block size; both sides agree.
  assert(src_worker_info->attn_kernel_blk_ntpb == src_ntpb);
  assert(dst_worker_info->attn_kernel_blk_ntpb == dst_ntpb);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  assert(num_gdn_layers < src_blocks.size());
  assert(num_gdn_layers < dst_blocks.size());

  const auto p_indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  const auto d_indexer_ntpb = dst_worker_info->indexer_blk_ntpb;
  assert((p_indexer_ntpb == 0 && d_indexer_ntpb == 0) ||
         (p_indexer_ntpb > 0 && d_indexer_ntpb > 0));

  // 1. GDN: emit only on the last token, sliced by gdn_group_n / gdn_group_off.
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

  // 2. Indexer (if present): one representative P rank per D-rank
  // subgroup sends the replicated indexer cache.
  size_t attn_group_offset = num_gdn_layers;
  if (p_indexer_ntpb > 0 && d_indexer_ntpb > 0) {
    if (src_worker_info->worker_tp_rank % gdn_group_n == 0) {
      assert(attn_group_offset < src_blocks.size());
      assert(attn_group_offset < dst_blocks.size());
      const auto &src_indexer_blks = src_blocks.at(attn_group_offset);
      const auto &dst_indexer_blks = dst_blocks.at(attn_group_offset);
      parse_hybrid_indexer_block(
        p_block_size, d_block_size,
        task->seen_tokens, task->new_tokens,
        src_indexer_blks, dst_indexer_blks,
        p_indexer_ntpb, d_indexer_ntpb,
        src_worker_info->hybrid_indexer_token_size,
        bounds,
        per_cache_send_blocks
      );
    }
    attn_group_offset += 1;
  }
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());

  // 3. Attention: FlashInfer HND parse, only ranks in valid_ranks send.
  if (valid_ranks[src_worker_info->worker_tp_rank]) {
    const auto &src_attn_blks = src_blocks.at(attn_group_offset);
    const auto &dst_attn_blks = dst_blocks.at(attn_group_offset);
    parse_hybrid_flashinfer_HND_block(
      p_token_size, d_token_size,
      p_block_size, d_block_size,
      src_ntpb,
      dst_ntpb,
      src_attn_blks, dst_attn_blks,
      task->seen_tokens, task->new_tokens,
      attn_group_n,
      attn_group_off,
      src_worker_info->attn_kernel_blk_ntpb,
      dst_worker_info->attn_kernel_blk_ntpb,
      src_heads_per_rank,
      bounds,
      false,
      per_cache_send_blocks
    );
    // When the request completes, fill the unused tail of the final decode
    // attention block with block 0.
    if (env_pad_last_attn_block() && task->reach_last_token) {
      fill_last_hybrid_flashinfer_HND_block(
        p_token_size, d_token_size,
        p_block_size, d_block_size,
        dst_ntpb,
        src_attn_blks, dst_attn_blks,
        task->seen_tokens + task->new_tokens,
        attn_group_off,
        src_heads_per_rank,
        bounds,
        per_cache_send_blocks
      );
    }
  }
}

// Zero-fill the tail of the D-side HND attention blocks for the sliced P<D
// regimes. Fills every dst token slot in [total_tokens,
// dst_blocks.size() * dst_ntpb). In HND layout each dst head section's tail is
// one contiguous segment per block, with no TP-group holes (the whole dst row
// belongs to this rank). Source is the request's block 0 (all zeros).
static void fill_tail_hybrid_flashinfer_HND_block_p_lt_d(
  size_t p_block_size,
  size_t d_block_size,
  uint32_t dst_ntpb,
  size_t head_dim_size,
  int dst_heads_per_rank,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t total_tokens,
  const IpcBlockBounds& bounds,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  const uint32_t end_tokens =
      static_cast<uint32_t>(dst_blocks.size()) * dst_ntpb;
  if (total_tokens >= end_tokens) {
    return;
  }
  const size_t d_head_section_size =
      static_cast<size_t>(dst_ntpb) * head_dim_size;
  const size_t d_kv_section_size = d_block_size / 2;
  const size_t zero_src_off = src_blocks.at(0) * p_block_size;

  uint32_t dst_pos = total_tokens;
  while (dst_pos < end_tokens) {
    const uint32_t d_blk_idx = dst_pos / dst_ntpb;
    const uint32_t d_token_idx = dst_pos % dst_ntpb;
    const uint32_t tokens =
        std::min<uint32_t>(dst_ntpb - d_token_idx, end_tokens - dst_pos);
    assert(d_blk_idx < dst_blocks.size());
    const size_t d_blk_off = dst_blocks.at(d_blk_idx) * d_block_size;
    const size_t length = static_cast<size_t>(tokens) * head_dim_size;
    for (int head_idx = 0; head_idx < dst_heads_per_rank; ++head_idx) {
      const size_t dk_off = d_blk_off
                          + static_cast<size_t>(head_idx) * d_head_section_size
                          + d_token_idx * head_dim_size;
      const size_t dv_off = dk_off + d_kv_section_size;
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
// QWEN3_NEXT_FLASHINFER_CACHE_SHAPE: P < D
// =============================================================================
//
// One P rank fans out to gdn_group_n = d_tp / p_tp D ranks. Attention/indexer
// handling depends on the head regime:
//   - num_kv_heads <= p_tp (replicated): attention and indexer
//     caches are replicated across TP ranks, so P and D per-rank blocks are
//     physically identical: reuse the block-aligned full-block copy
//   - num_kv_heads > p_tp (mixed p_tp < heads < d_tp, or fully head-split
//     heads >= d_tp): the P rank holds attn_group_n = p_token/d_token times
//     the D rank's heads; each fan-out target receives its attn_group_off-th
//     head-section sub-slice (mirror of P>D where the offset is on the dst
//     side). The indexer stays replicated and the padded tail plus spec-decode
//     trailing blocks are zero-filled explicitly.
// GDN/short-conv are engine-tp sub-sliced in all regimes; this D rank
// receives the gdn_group_off-th slice.
void vllm_parse_qwen3_next_flashinfer_block_send_p_lt_d(
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
  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb * p_token_size == p_block_size);
  assert(dst_ntpb * d_token_size == d_block_size);
  // FlashInfer attn does not use sub-block kernel reshape, kernel_blk == ntpb.
  assert(src_worker_info->attn_kernel_blk_ntpb == src_ntpb);
  assert(dst_worker_info->attn_kernel_blk_ntpb == dst_ntpb);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  const size_t num_gdn_layers = src_worker_info->num_gdn_layers;
  size_t attn_group_offset = num_gdn_layers;
  assert(attn_group_offset < src_blocks.size());
  assert(attn_group_offset < dst_blocks.size());

  const auto p_indexer_ntpb = src_worker_info->indexer_blk_ntpb;
  const auto d_indexer_ntpb = dst_worker_info->indexer_blk_ntpb;
  assert((p_indexer_ntpb == 0 && d_indexer_ntpb == 0) ||
         (p_indexer_ntpb > 0 && d_indexer_ntpb > 0));

  if (p_token_size == d_token_size) {
    // Replicated-head regime (num_kv_heads <= p_tp): attn/indexer per-rank
    // layouts are identical.
    assert(p_block_size == d_block_size);
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
    assert(attn_group_offset < src_blocks.size());
    assert(attn_group_offset < dst_blocks.size());

    // Attention: replicated, full-block copy (also zero-fills padded tail).
    const auto& src_attn_blks = src_blocks.at(attn_group_offset);
    const auto& dst_attn_blks = dst_blocks.at(attn_group_offset);
    do_parse_block_aligned_send(
      p_block_size, task,
      src_attn_blks, dst_attn_blks,
      src_ntpb, bounds, per_cache_send_blocks
    );
  } else {
    // Sliced regime (num_kv_heads > p_tp): the P rank holds attn_group_n
    // times the D rank's head sections. Head sub-slicing:
    //   attn_group_n = p_token/d_token  (heads/p_tp in mixed, d_tp/p_tp in
    //                                    fully head-split)
    //   replicas     = D ranks sharing the same head slice
    //                = d_tp / (p_tp * attn_group_n)
    //   attn_group_off = (d_rank / replicas) % attn_group_n
    assert(p_token_size > d_token_size);
    assert(p_token_size % d_token_size == 0);
    const uint32_t attn_group_n =
        static_cast<uint32_t>(p_token_size / d_token_size);
    assert(d_engine_tp % (p_engine_tp * attn_group_n) == 0);
    const uint32_t replicas = d_engine_tp / (p_engine_tp * attn_group_n);
    const uint32_t attn_group_off =
        (dst_worker_info->worker_tp_rank / replicas) % attn_group_n;

    // Heads held by one D rank: heads/d_tp when fully split, 1 replicated
    // head in the mixed regime. Needed for the HND head-section math.
    const int num_kv_heads = src_worker_info->num_kv_heads;
    assert(num_kv_heads > 0);
    const int dst_heads_per_rank =
        (static_cast<uint32_t>(num_kv_heads) >= d_engine_tp)
            ? num_kv_heads / static_cast<int>(d_engine_tp)
            : 1;
    assert(d_token_size % (2 * static_cast<size_t>(dst_heads_per_rank)) == 0);
    const size_t head_dim_size =
        d_token_size / 2 / static_cast<size_t>(dst_heads_per_rank);

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

    // Attention: P<D is the exact mirror of P>D -- reuse the HND parse with
    // (p, d) swapped so the "wide" side (here P) receives the group offset,
    // then swap src/dst offsets of the emitted records.
    parse_hybrid_flashinfer_HND_block(
      d_token_size, p_token_size,
      d_block_size, p_block_size,
      dst_ntpb,               // "src" ntpb  <- actual D
      src_ntpb,               // "dst" ntpb  <- actual P
      dst_attn_blks, src_attn_blks,
      task->seen_tokens, task->new_tokens,
      attn_group_n,
      attn_group_off,
      dst_worker_info->attn_kernel_blk_ntpb,
      src_worker_info->attn_kernel_blk_ntpb,
      dst_heads_per_rank,
      bounds,
      true,
      per_cache_send_blocks
    );

    // Token-granularity transfer covers neither the padded tail of the last
    // dst block nor the spec-decode trailing zero blocks (the block-aligned
    // copy in the replicated branch does both implicitly). Zero-fill them
    // from the request's block 0 on request completion.
    if (env_pad_last_attn_block() && task->reach_last_token) {
      fill_tail_hybrid_flashinfer_HND_block_p_lt_d(
        p_block_size, d_block_size,
        dst_ntpb,
        head_dim_size,
        dst_heads_per_rank,
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
