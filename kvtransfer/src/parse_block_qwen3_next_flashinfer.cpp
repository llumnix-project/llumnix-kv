#include "parse_block_common.h"
#include "parse_block_flashinfer_internal.h"
#include "parse_block_qwen3_next_internal.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

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
      indexer_ntpb, per_cache_send_blocks
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
    ntpb, per_cache_send_blocks
  );

  // 3. GDN: emit only on the last token (after attn, mirroring the existing
  // QWEN3_NEXT P==D ordering so downstream send queue layout matches).
  if (task->reach_last_token) {
    parse_hybrid_gdn_block_send_p_eq_d(
      p_block_size, num_gdn_layers,
      src_blocks, dst_blocks, per_cache_send_blocks
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

  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  assert(src_ntpb * p_token_size == p_block_size);
  assert(dst_ntpb * d_token_size == d_block_size);
  // FlashInfer page size matches kv-manager block size; both sides agree.
  assert(src_worker_info->attn_kernel_blk_ntpb == src_ntpb);
  assert(dst_worker_info->attn_kernel_blk_ntpb == dst_ntpb);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks.at(0);

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
      per_cache_send_blocks
    );
  }

  // 2. Indexer (if present): rank 0 only, token-granularity copy.
  size_t attn_group_offset = num_gdn_layers;
  if (p_indexer_ntpb > 0 && d_indexer_ntpb > 0) {
    if (src_worker_info->worker_tp_rank == 0) {
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
      src_worker_info->num_kv_heads,
      per_cache_send_blocks
    );
    // On request completion, fill the tail of the last unfilled decode attn block with block 0.
    if (env_pad_last_attn_block() && task->reach_last_token) {
      fill_last_hybrid_flashinfer_HND_block(
        p_token_size, d_token_size,
        p_block_size, d_block_size,
        dst_ntpb,
        src_attn_blks, dst_attn_blks,
        task->seen_tokens + task->new_tokens,
        attn_group_off,
        src_worker_info->num_kv_heads,
        per_cache_send_blocks
      );
    }
  }
}

}  // namespace blade_llm
