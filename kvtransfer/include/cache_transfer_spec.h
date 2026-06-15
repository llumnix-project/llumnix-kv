#pragma once

#include <vector>
#include <optional>
#include <variant>
#include <cstdint>
#include <cstddef>
#include <cassert>

#include "common.h"
#include "channel.h"
#include "attn_layout.h"

namespace blade_llm {

// Describes one contiguous copy stream per token for the K component
// (or merged KV when v is nullopt).
// For each token, the offset is computed as:
//   offset = blocks[block_idx] * block_stride
//          + token_idx * token_stride
//          + base_offset
//          + global_offset
//          + head_iter * head_stride
//
// When v is set, a second stream is generated for V using VSpec's
// offsets but sharing block_stride, token_stride, bytes_per_token,
// and head_num with K.
struct AttnSpec {
  size_t src_block_stride;
  size_t dst_block_stride;
  size_t src_token_stride;
  size_t dst_token_stride;
  size_t src_base_offset = 0;
  size_t dst_base_offset = 0;
  size_t src_global_offset = 0;
  size_t dst_global_offset = 0;
  size_t bytes_per_token;

  // For head-major layouts (like FLASHINFER): iterate over heads with
  // per-head stride offsets.
  uint32_t head_num = 1;
  size_t src_head_stride = 0;
  size_t dst_head_stride = 0;

  // V component offsets when K and V are separate.
  // Shares block_stride, token_stride, bytes_per_token, head_num with K.
  struct VSpec {
    size_t src_base_offset = 0;
    size_t dst_base_offset = 0;
    size_t src_global_offset = 0;
    size_t dst_global_offset = 0;
    size_t src_head_stride = 0;
    size_t dst_head_stride = 0;
  };
  std::optional<VSpec> v;
};

// One contiguous segment within a GDN block.
struct GdnSegment {
  size_t src_offset_in_block;
  size_t dst_offset_in_block;
  size_t length;
};

// GDN blocks are transferred as whole segments, not per-token.
// Only sent when reach_last_token is true.
// num_groups: number of GDN layers; each group in BlockIds has its own block list.
struct GdnSpec {
  uint32_t num_groups;
  size_t src_block_stride;
  size_t dst_block_stride;
  std::vector<GdnSegment> segments;
};

// Token-level transfer component (attention or indexer).
// Carries its own block_group index for looking up block IDs.
struct AttnComponent {
  AttnSpec spec;
  uint32_t block_group = 0;
  uint32_t src_ntpb = 0;
  uint32_t dst_ntpb = 0;
  bool block_aligned = false;
  // When true (qwen3_next P>D fullattn), and reach_last_token, pad the unfilled
  // tail of the last decode-side block using the request's block 0 data so the
  // decode kernel never reads uninitialized KV in the padded tail.
  bool fill_last_block = false;
};

// GDN transfer component.
// Iterates groups 0..spec.num_groups-1 from BlockIds.
// Only processed when reach_last_token is true.
struct GdnComponent {
  GdnSpec spec;
};

using TransferComponent = std::variant<AttnComponent, GdnComponent>;

// Full specification for one cache tensor's transfer pattern.
// Components are stored in build-time order: GDN -> Indexer -> FullAttn.
struct CacheTransferSpec {
  std::vector<TransferComponent> components;
  bool valid = true;
};

// Complete transfer plan: one spec per cache tensor, plus TPKind.
struct TransferPlan {
  std::vector<CacheTransferSpec> specs;
  TPKind tpkind = TPKind::UNKNOWN;
};

// Generate IpcBlocks for a single cache tensor spec.
void generate_ipc_blocks(
    const CacheTransferSpec& spec,
    const BlockIds& src_blocks,
    const BlockIds& dst_blocks,
    uint32_t wrote_tokens,
    uint32_t left_tokens,
    bool reach_last_token,
    std::vector<IpcBlock>& out);

// Generate IpcBlocks for all cache tensors in a plan.
void generate_all_ipc_blocks(
    const TransferPlan& plan,
    const ReqSendTask* task,
    std::vector<std::vector<IpcBlock>>& send_blocks);

TransferPlan build_transfer_plan(
    int cache_shape,
    const WorkerInfo& src_info,
    const WorkerInfo& dst_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank);

}  // namespace blade_llm
