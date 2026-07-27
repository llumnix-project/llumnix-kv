#include "parse_block_common.h"

#include <algorithm>
#include <cassert>

namespace blade_llm {
namespace {

void parse_replicated_mla_blocks(
    const WorkerInfo& src,
    const WorkerInfo& dst,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    const ReqSendTask& task,
    std::vector<std::vector<IpcBlock>>& send_blocks) {
  // For P>D only one representative P rank feeds each D rank. For P==D and
  // P<D compute_valid_ranks_pd keeps every sender needed by the rank mapping.
  if (!valid_ranks.test(src.worker_tp_rank)) {
    return;
  }

  assert(src.block_sizes.size() == 1);
  assert(dst.block_sizes.size() == 1);
  assert(src.token_sizes.size() == 1);
  assert(dst.token_sizes.size() == 1);
  assert(src.hybrid_attn_token_size > 0);
  assert(src.attn_kernel_blk_ntpb > 0);
  assert(src.attn_kernel_blk_ntpb == dst.attn_kernel_blk_ntpb);
  assert(src.attn_pack_size == 1);
  assert(dst.attn_pack_size == 1);
  assert(src.hybrid_attn_token_size == dst.hybrid_attn_token_size);

  const size_t group_idx = src.num_gdn_layers;
  const auto& src_groups = task.src_blocks();
  const auto& dst_groups = task.dst_blocks();
  assert(src_groups.size() == group_idx + 1);
  assert(dst_groups.size() == group_idx + 1);
  const auto& src_blocks = src_groups.at(group_idx);
  const auto& dst_blocks = dst_groups.at(group_idx);

  const size_t src_logical_block_size = src.block_sizes.at(0);
  const size_t dst_logical_block_size = dst.block_sizes.at(0);
  const size_t src_registered_token_size = src.token_sizes.at(0);
  const size_t dst_registered_token_size = dst.token_sizes.at(0);
  assert(src_registered_token_size > 0);
  assert(dst_registered_token_size > 0);
  assert(src_logical_block_size % src_registered_token_size == 0);
  assert(dst_logical_block_size % dst_registered_token_size == 0);
  const uint32_t src_manager_ntpb =
      src_logical_block_size / src_registered_token_size;
  const uint32_t dst_manager_ntpb =
      dst_logical_block_size / dst_registered_token_size;
  const uint32_t kernel_ntpb = src.attn_kernel_blk_ntpb;
  assert(src_manager_ntpb > 0);
  assert(dst_manager_ntpb > 0);
  assert(src_manager_ntpb % kernel_ntpb == 0);
  assert(dst_manager_ntpb % kernel_ntpb == 0);
  const size_t kernel_mla_page_size =
      static_cast<size_t>(kernel_ntpb) * src.hybrid_attn_token_size;
  const size_t src_physical_page_size =
      src.kda_page_stride != 0
          ? src.kda_page_stride
          : src_logical_block_size;
  const size_t dst_physical_page_size =
      dst.kda_page_stride != 0
          ? dst.kda_page_stride
          : dst_logical_block_size;
  assert(static_cast<size_t>(src_manager_ntpb)
             * src.hybrid_attn_token_size <= src_physical_page_size);
  assert(static_cast<size_t>(dst_manager_ntpb)
             * src.hybrid_attn_token_size <= dst_physical_page_size);

  send_blocks.resize(1);
  auto& blocks = send_blocks.at(0);
  const auto bounds = make_ipc_block_bounds(src, dst, 0);

  // Equal TP has a one-to-one vLLM manager-block mapping. Follow the same
  // block-aligned policy as do_parse_block_aligned_send: defer an incomplete
  // block, but on request completion send every remaining allocated block.
  // Copy only the real MLA prefix; the rest of each physical page is KDA state.
  if (src.engine_tp_size == dst.engine_tp_size) {
    assert(src_manager_ntpb == dst_manager_ntpb);
    assert(src_blocks.size() == dst_blocks.size());
    const size_t first_block = task.seen_tokens / src_manager_ntpb;
    const size_t end_block = task.reach_last_token
        ? src_blocks.size()
        : task.end_tokens() / src_manager_ntpb;
    assert(first_block <= end_block);
    assert(end_block <= src_blocks.size());
    const size_t mla_manager_block_size =
        static_cast<size_t>(src_manager_ntpb) * src.hybrid_attn_token_size;
    for (size_t block_idx = first_block; block_idx < end_block; ++block_idx) {
      append_ipc_block_checked(
          blocks, bounds,
          static_cast<size_t>(src_blocks.at(block_idx)) *
              src_physical_page_size,
          static_cast<size_t>(dst_blocks.at(block_idx)) *
              dst_physical_page_size,
          mla_manager_block_size);
    }
    return;
  }

  // Unequal TP may use different vLLM manager block sizes. Map token
  // positions through each side's manager block and transfer kernel pages.
  uint32_t wrote_tokens = task.seen_tokens;
  uint32_t left_tokens = task.new_tokens;
  while (left_tokens > 0) {
    // Request block IDs index vLLM manager blocks, not KVT kernel blocks.
    // Resolve the manager block first, then add the kernel-block offset within
    // its physical KDA-sized page.
    const uint32_t src_block_idx = wrote_tokens / src_manager_ntpb;
    const uint32_t dst_block_idx = wrote_tokens / dst_manager_ntpb;
    const uint32_t src_token_in_manager =
        wrote_tokens % src_manager_ntpb;
    const uint32_t dst_token_in_manager =
        wrote_tokens % dst_manager_ntpb;
    const uint32_t src_kernel_idx =
        src_token_in_manager / kernel_ntpb;
    const uint32_t dst_kernel_idx =
        dst_token_in_manager / kernel_ntpb;
    const uint32_t token_idx = wrote_tokens % kernel_ntpb;
    assert(src_block_idx < src_blocks.size());
    assert(dst_block_idx < dst_blocks.size());
    const uint32_t tokens =
        std::min(kernel_ntpb - token_idx, left_tokens);

    bool emit = tokens == kernel_ntpb;
    if (!emit && task.reach_last_token) {
      emit = true;
    } else if (!emit && token_idx + tokens == kernel_ntpb) {
      emit = true;
    }
    if (emit) {
      append_ipc_block_checked(
          blocks, bounds,
          static_cast<size_t>(src_blocks.at(src_block_idx))
                  * src_physical_page_size
              + static_cast<size_t>(src_kernel_idx) * kernel_mla_page_size,
          static_cast<size_t>(dst_blocks.at(dst_block_idx))
                  * dst_physical_page_size
              + static_cast<size_t>(dst_kernel_idx) * kernel_mla_page_size,
          kernel_mla_page_size);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }

  // With unequal engine TP sizes, only the MLA bytes for computed tokens are
  // guaranteed to be emitted above. Mirror the Qwen3-Next behavior and fill
  // the rest of the allocated D-side MLA token region from the first P-side
  // request block. This prevents stale data in the unused tail from surviving
  // a P/D TP remap. The fill is intentionally limited to the real MLA prefix
  // of each KDA-sized physical page; KDA padding is not model state.
  if (!env_pad_last_attn_block() || !task.reach_last_token) {
    return;
  }

  const size_t total_tokens =
      static_cast<size_t>(task.seen_tokens) + task.new_tokens;
  const size_t dst_capacity_tokens =
      dst_blocks.size() * static_cast<size_t>(dst_manager_ntpb);
  if (total_tokens >= dst_capacity_tokens) {
    return;
  }
  assert(!src_blocks.empty());
  const size_t fill_src_off =
      static_cast<size_t>(src_blocks.at(0)) * src_physical_page_size;
  size_t dst_pos = total_tokens;
  while (dst_pos < dst_capacity_tokens) {
    const size_t dst_block_idx = dst_pos / dst_manager_ntpb;
    const size_t token_in_block = dst_pos % dst_manager_ntpb;
    const size_t tokens = std::min(
        static_cast<size_t>(dst_manager_ntpb) - token_in_block,
        dst_capacity_tokens - dst_pos);
    const size_t dst_off =
        static_cast<size_t>(dst_blocks.at(dst_block_idx)) *
            dst_physical_page_size +
        token_in_block * src.hybrid_attn_token_size;
    emit_zero_fill_segments(
        fill_src_off, dst_off, tokens * src.hybrid_attn_token_size,
        src_physical_page_size, bounds, blocks);
    dst_pos += tokens;
  }
}

}  // namespace

void parse_kimi_k3_mla_block_send_p_eq_d(
    const WorkerInfo *src_worker_info,
    const WorkerInfo *dst_worker_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank,
    const ReqSendTask *task,
    std::vector<std::vector<IpcBlock>> &send_blocks) {
  assert(src_worker_info->engine_tp_size == dst_worker_info->engine_tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  parse_replicated_mla_blocks(
      *src_worker_info, *dst_worker_info, valid_ranks, *task, send_blocks);
  parse_kda_block_send_p_eq_d(
      src_worker_info, dst_worker_info, valid_ranks, kvt_tp_size,
      kvt_tp_rank, task, send_blocks);
}

void parse_kimi_k3_mla_block_send_p_gt_d(
    const WorkerInfo *src_worker_info,
    const WorkerInfo *dst_worker_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank,
    const ReqSendTask *task,
    std::vector<std::vector<IpcBlock>> &send_blocks) {
  assert(src_worker_info->engine_tp_size > dst_worker_info->engine_tp_size);
  parse_replicated_mla_blocks(
      *src_worker_info, *dst_worker_info, valid_ranks, *task, send_blocks);
  parse_kda_block_send_p_gt_d(
      src_worker_info, dst_worker_info, valid_ranks, kvt_tp_size,
      kvt_tp_rank, task, send_blocks);
}

void parse_kimi_k3_mla_block_send_p_lt_d(
    const WorkerInfo *src_worker_info,
    const WorkerInfo *dst_worker_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank,
    const ReqSendTask *task,
    std::vector<std::vector<IpcBlock>> &send_blocks) {
  assert(src_worker_info->engine_tp_size < dst_worker_info->engine_tp_size);
  parse_replicated_mla_blocks(
      *src_worker_info, *dst_worker_info, valid_ranks, *task, send_blocks);
  parse_kda_block_send_p_lt_d(
      src_worker_info, dst_worker_info, valid_ranks, kvt_tp_size,
      kvt_tp_rank, task, send_blocks);
}

}  // namespace blade_llm
