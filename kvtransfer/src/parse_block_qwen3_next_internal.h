#ifndef KVTRANSFER_SRC_PARSE_BLOCK_QWEN3_NEXT_INTERNAL_H_
#define KVTRANSFER_SRC_PARSE_BLOCK_QWEN3_NEXT_INTERNAL_H_

#pragma once

#include "parse_block_common.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace blade_llm {

// Block-aligned send: only emit IpcBlocks at full-block granularity, deferring
// partial chunked-prefill blocks to the next step (unless reach_last_token).
// Used for qwen3_next P==D attention/indexer transfer.
void do_parse_block_aligned_send(
  size_t block_size,
  const ReqSendTask *task,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t ntpb,
  std::vector<IpcBlock> &per_cache_send_blocks
);

// Token-granularity indexer block parse (used by qwen3_next P>D, indexer cache
// is replicated across TP ranks so only rank 0 sends).
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
  std::vector<IpcBlock> &per_cache_send_blocks
);

// GDN block emit for P==D: send the entire padded GDN block as-is.
// Iterates each gdn group's block list and pushes one full-block IpcBlock.
// Caller should only invoke when task->reach_last_token is true.
void parse_hybrid_gdn_block_send_p_eq_d(
  size_t block_size,
  size_t num_gdn_layers,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  std::vector<IpcBlock> &per_cache_send_blocks
);

// GDN block emit for P>D: split conv state along channel dims and append ssm
// state, with dst offsets adjusted by gdn_group_n / gdn_group_off.
// Caller should only invoke when task->reach_last_token is true.
void parse_hybrid_gdn_block_send_p_gt_d(
  size_t p_block_size,
  size_t d_block_size,
  size_t num_gdn_layers,
  uint32_t gdn_group_n,
  uint32_t gdn_group_off,
  const WorkerInfo *src_worker_info,
  const std::vector<std::vector<uint32_t>>& src_blocks,
  const std::vector<std::vector<uint32_t>>& dst_blocks,
  std::vector<IpcBlock> &per_cache_send_blocks
);

}  // namespace blade_llm

#endif  // KVTRANSFER_SRC_PARSE_BLOCK_QWEN3_NEXT_INTERNAL_H_
