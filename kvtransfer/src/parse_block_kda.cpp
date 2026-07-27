#include "parse_block_common.h"

#include <numeric>

namespace blade_llm {
namespace {

struct KdaStateLayout {
  size_t state_len;
  size_t qkv_channels;
  size_t conv_size;
  size_t recurrent_size;
};

size_t get_kda_page_stride(const WorkerInfo& worker_info) {
  assert(worker_info.block_sizes.size() == 1);
  return worker_info.kda_page_stride != 0
             ? worker_info.kda_page_stride
             : worker_info.block_sizes.at(0);
}

KdaStateLayout get_kda_state_layout(const WorkerInfo& worker_info) {
  const auto& conv_shape = worker_info.conv_state_shape;
  const auto& recurrent_shape = worker_info.ssm_state_shape;
  const auto& channel_dims = worker_info.gdn_conv_channel_dims;

  // Reuse the shapes already populated for GDN-style recurrent states. The
  // leading dimension is num_blocks; one parsed page contains only dims 1+.
  assert(conv_shape.size() == 3);
  assert(recurrent_shape.size() == 4);
  assert(channel_dims.size() == 3);
  assert(worker_info.gdn_conv_elem_size > 0);
  assert(worker_info.gdn_ssm_elem_size > 0);

  const size_t state_len = worker_info.kda_conv_dim_first
                               ? conv_shape.at(2)
                               : conv_shape.at(1);
  const size_t qkv_channels = worker_info.kda_conv_dim_first
                                  ? conv_shape.at(1)
                                  : conv_shape.at(2);
  assert(state_len > 0);
  assert(qkv_channels > 0);
  assert(std::accumulate(channel_dims.begin(), channel_dims.end(), size_t{0})
         == qkv_channels);

  return KdaStateLayout{
      state_len,
      qkv_channels,
      state_len * qkv_channels * worker_info.gdn_conv_elem_size,
      recurrent_shape.at(1) * recurrent_shape.at(2) * recurrent_shape.at(3)
          * worker_info.gdn_ssm_elem_size};
}

void check_kda_block_groups(
    size_t num_linear_attention_layers,
    const std::vector<std::vector<uint32_t>>& src_blocks,
    const std::vector<std::vector<uint32_t>>& dst_blocks) {
  assert(num_linear_attention_layers > 0);
  assert(num_linear_attention_layers <= src_blocks.size());
  assert(num_linear_attention_layers <= dst_blocks.size());
  for (size_t group_idx = 0;
       group_idx < num_linear_attention_layers; ++group_idx) {
    assert(src_blocks.at(group_idx).size()
           == dst_blocks.at(group_idx).size());
  }
}

void emit_kda_block_p_gt_d(
    size_t p_block_offset,
    size_t d_block_offset,
    uint32_t group_n,
    uint32_t group_off,
    const WorkerInfo& src_worker_info,
    const KdaStateLayout& p_layout,
    const IpcBlockBounds& bounds,
    std::vector<IpcBlock>& blocks) {
  const size_t elem_size = src_worker_info.gdn_conv_elem_size;
  const auto& channel_dims = src_worker_info.gdn_conv_channel_dims;

  if (src_worker_info.kda_conv_dim_first) {
    // DS: [Q channels][K channels][V channels], with every channel holding
    // one contiguous state_len row.
    size_t channel_inner_offset = 0;
    for (const size_t channel_dim : channel_dims) {
      const size_t component_size = channel_dim * p_layout.state_len * elem_size;
      append_ipc_block_checked(
          blocks, bounds,
          p_block_offset + channel_inner_offset,
          d_block_offset + group_n * channel_inner_offset
              + group_off * component_size,
          component_size);
      channel_inner_offset += component_size;
    }
    assert(channel_inner_offset == p_layout.conv_size);
  } else {
    // SD: repeat the Q/K/V placement for every short-conv history row.
    const size_t p_row_size = p_layout.qkv_channels * elem_size;
    const size_t d_row_size = group_n * p_row_size;
    for (size_t state_idx = 0; state_idx < p_layout.state_len; ++state_idx) {
      size_t row_inner_offset = 0;
      for (const size_t channel_dim : channel_dims) {
        const size_t component_size = channel_dim * elem_size;
        append_ipc_block_checked(
            blocks, bounds,
            p_block_offset + state_idx * p_row_size + row_inner_offset,
            d_block_offset + state_idx * d_row_size
                + group_n * row_inner_offset
                + group_off * component_size,
            component_size);
        row_inner_offset += component_size;
      }
      assert(row_inner_offset == p_row_size);
    }
  }

  // recurrent_state is [local_heads, value_dim, key_dim], so TP shards are
  // contiguous along the first dimension.
  append_ipc_block_checked(
      blocks, bounds,
      p_block_offset + p_layout.conv_size,
      d_block_offset + group_n * p_layout.conv_size
          + group_off * p_layout.recurrent_size,
      p_layout.recurrent_size);
}

void emit_kda_block_p_lt_d(
    size_t p_block_offset,
    size_t d_block_offset,
    uint32_t group_n,
    uint32_t group_off,
    const WorkerInfo& src_worker_info,
    const KdaStateLayout& p_layout,
    const IpcBlockBounds& bounds,
    std::vector<IpcBlock>& blocks) {
  const size_t elem_size = src_worker_info.gdn_conv_elem_size;
  const auto& channel_dims = src_worker_info.gdn_conv_channel_dims;

  assert(p_layout.recurrent_size % group_n == 0);
  const size_t d_recurrent_size = p_layout.recurrent_size / group_n;
  size_t d_conv_size = 0;

  if (src_worker_info.kda_conv_dim_first) {
    size_t p_inner_offset = 0;
    size_t d_inner_offset = 0;
    for (const size_t p_channel_dim : channel_dims) {
      assert(p_channel_dim % group_n == 0);
      const size_t d_channel_dim = p_channel_dim / group_n;
      const size_t component_size =
          d_channel_dim * p_layout.state_len * elem_size;
      append_ipc_block_checked(
          blocks, bounds,
          p_block_offset + p_inner_offset + group_off * component_size,
          d_block_offset + d_inner_offset,
          component_size);
      p_inner_offset += group_n * component_size;
      d_inner_offset += component_size;
    }
    assert(p_inner_offset == p_layout.conv_size);
    d_conv_size = d_inner_offset;
  } else {
    const size_t p_row_size = p_layout.qkv_channels * elem_size;
    assert(p_row_size % group_n == 0);
    const size_t d_row_size = p_row_size / group_n;
    for (size_t state_idx = 0; state_idx < p_layout.state_len; ++state_idx) {
      size_t p_row_inner_offset = 0;
      size_t d_row_inner_offset = 0;
      for (const size_t p_channel_dim : channel_dims) {
        assert(p_channel_dim % group_n == 0);
        const size_t component_size =
            (p_channel_dim / group_n) * elem_size;
        append_ipc_block_checked(
            blocks, bounds,
            p_block_offset + state_idx * p_row_size + p_row_inner_offset
                + group_off * component_size,
            d_block_offset + state_idx * d_row_size + d_row_inner_offset,
            component_size);
        p_row_inner_offset += group_n * component_size;
        d_row_inner_offset += component_size;
      }
      assert(p_row_inner_offset == p_row_size);
      assert(d_row_inner_offset == d_row_size);
    }
    d_conv_size = p_layout.conv_size / group_n;
  }

  append_ipc_block_checked(
      blocks, bounds,
      p_block_offset + p_layout.conv_size + group_off * d_recurrent_size,
      d_block_offset + d_conv_size,
      d_recurrent_size);
}

void prepare_kda_parse(
    const WorkerInfo& src_worker_info,
    const WorkerInfo& dst_worker_info,
    const ReqSendTask& task,
    std::vector<std::vector<IpcBlock>>& send_blocks) {
  assert(src_worker_info.block_sizes.size() == 1);
  assert(dst_worker_info.block_sizes.size() == 1);
  check_kda_block_groups(
      src_worker_info.num_gdn_layers, task.src_blocks(), task.dst_blocks());
  send_blocks.resize(1);
}

}  // namespace

void parse_kda_block_send_p_eq_d(
    const WorkerInfo *src_worker_info,
    const WorkerInfo *dst_worker_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank,
    const ReqSendTask *task,
    std::vector<std::vector<IpcBlock>> &send_blocks) {
  prepare_kda_parse(*src_worker_info, *dst_worker_info, *task, send_blocks);
  assert(src_worker_info->engine_tp_size == dst_worker_info->engine_tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  const size_t src_page_stride = get_kda_page_stride(*src_worker_info);
  const size_t dst_page_stride = get_kda_page_stride(*dst_worker_info);
  assert(src_page_stride == dst_page_stride);
  if (!task->reach_last_token) {
    return;
  }

  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);
  auto& blocks = send_blocks.at(0);
  for (size_t group_idx = 0;
       group_idx < src_worker_info->num_gdn_layers; ++group_idx) {
    const auto& src_group = task->src_blocks().at(group_idx);
    const auto& dst_group = task->dst_blocks().at(group_idx);
    for (size_t block_idx = 0; block_idx < src_group.size(); ++block_idx) {
      append_ipc_block_checked(
          blocks, bounds,
          src_group.at(block_idx) * src_page_stride,
          dst_group.at(block_idx) * dst_page_stride,
          src_page_stride);
    }
  }
}

void parse_kda_block_send_p_gt_d(
    const WorkerInfo *src_worker_info,
    const WorkerInfo *dst_worker_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank,
    const ReqSendTask *task,
    std::vector<std::vector<IpcBlock>> &send_blocks) {
  prepare_kda_parse(*src_worker_info, *dst_worker_info, *task, send_blocks);
  const uint32_t p_tp = src_worker_info->engine_tp_size;
  const uint32_t d_tp = dst_worker_info->engine_tp_size;
  assert(p_tp > d_tp);
  assert(p_tp % d_tp == 0);
  const uint32_t group_n = p_tp / d_tp;
  const uint32_t group_off = src_worker_info->worker_tp_rank % group_n;
  assert(src_worker_info->worker_tp_rank / group_n
         == dst_worker_info->worker_tp_rank);
  if (!task->reach_last_token) {
    return;
  }

  const size_t p_block_size = get_kda_page_stride(*src_worker_info);
  const size_t d_block_size = get_kda_page_stride(*dst_worker_info);
  const auto p_layout = get_kda_state_layout(*src_worker_info);
  assert(group_n * (p_layout.conv_size + p_layout.recurrent_size)
         <= d_block_size);
  assert(p_layout.conv_size + p_layout.recurrent_size <= p_block_size);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);
  auto& blocks = send_blocks.at(0);

  for (size_t group_idx = 0;
       group_idx < src_worker_info->num_gdn_layers; ++group_idx) {
    const auto& src_group = task->src_blocks().at(group_idx);
    const auto& dst_group = task->dst_blocks().at(group_idx);
    for (size_t block_idx = 0; block_idx < src_group.size(); ++block_idx) {
      emit_kda_block_p_gt_d(
          src_group.at(block_idx) * p_block_size,
          dst_group.at(block_idx) * d_block_size,
          group_n, group_off, *src_worker_info, p_layout, bounds, blocks);
    }
  }
}

void parse_kda_block_send_p_lt_d(
    const WorkerInfo *src_worker_info,
    const WorkerInfo *dst_worker_info,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank,
    const ReqSendTask *task,
    std::vector<std::vector<IpcBlock>> &send_blocks) {
  prepare_kda_parse(*src_worker_info, *dst_worker_info, *task, send_blocks);
  const uint32_t p_tp = src_worker_info->engine_tp_size;
  const uint32_t d_tp = dst_worker_info->engine_tp_size;
  assert(p_tp < d_tp);
  assert(d_tp % p_tp == 0);
  const uint32_t group_n = d_tp / p_tp;
  const uint32_t group_off = dst_worker_info->worker_tp_rank % group_n;
  assert(dst_worker_info->worker_tp_rank / group_n
         == src_worker_info->worker_tp_rank);
  if (!task->reach_last_token) {
    return;
  }

  const size_t p_block_size = get_kda_page_stride(*src_worker_info);
  const size_t d_block_size = get_kda_page_stride(*dst_worker_info);
  const auto p_layout = get_kda_state_layout(*src_worker_info);
  assert(p_layout.conv_size % group_n == 0);
  assert(p_layout.recurrent_size % group_n == 0);
  assert(p_layout.conv_size + p_layout.recurrent_size <= p_block_size);
  assert((p_layout.conv_size + p_layout.recurrent_size) / group_n
         <= d_block_size);
  const auto bounds = make_ipc_block_bounds(
      *src_worker_info, *dst_worker_info, 0);
  auto& blocks = send_blocks.at(0);

  for (size_t group_idx = 0;
       group_idx < src_worker_info->num_gdn_layers; ++group_idx) {
    const auto& src_group = task->src_blocks().at(group_idx);
    const auto& dst_group = task->dst_blocks().at(group_idx);
    for (size_t block_idx = 0; block_idx < src_group.size(); ++block_idx) {
      emit_kda_block_p_lt_d(
          src_group.at(block_idx) * p_block_size,
          dst_group.at(block_idx) * d_block_size,
          group_n, group_off, *src_worker_info, p_layout, bounds, blocks);
    }
  }
}

}  // namespace blade_llm
