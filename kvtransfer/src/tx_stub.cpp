#include "tx_stub.h"
#include "utils/timer.h"
#include "thrid_party/logging.h"
#include "utils/socket_helper.h"
#include "channel.h"
#include "envcfg.h"
#include "naming/fake_naming.h"
#include <string.h>
#include <unistd.h>
#include "fault_inject.h"

namespace blade_llm {

// Can only used for gdn block parsing
// rename after
static void parse_hybrid_attn_block(
  int p_base_blk_idx,
  int d_base_blk_idx,
  uint32_t src_ntpb, // ntpb: number tokens per block
  uint32_t dst_ntpb, 
  size_t p_token_size,
  size_t d_token_size,
  size_t p_block_size,
  size_t d_block_size,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  uint32_t attn_group_off,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  while (left_tokens > 0) {
    const auto p_blk_idx = p_base_blk_idx + wrote_tokens / src_ntpb;
    const auto p_token_idx_base = wrote_tokens % src_ntpb;
    const auto d_blk_idx = d_base_blk_idx + wrote_tokens / dst_ntpb;
    const auto d_token_idx_base = wrote_tokens % dst_ntpb;

    assert(p_blk_idx < src_blocks.size() && d_blk_idx < dst_blocks.size());
    const size_t p_blk_off = src_blocks[p_blk_idx] * p_block_size;
    const size_t d_blk_off = dst_blocks[d_blk_idx] * d_block_size;

    const auto tokens = std::min<uint32_t>(
      {src_ntpb - p_token_idx_base, dst_ntpb - d_token_idx_base, left_tokens}
    );
    // kv分开，除以2
    const auto token_step_length = p_token_size / 2;
    assert(token_step_length * 2 == p_token_size);
    for (uint32_t idx = 0; idx < tokens; ++idx) { // 这里按照p_token_size拆成小包
      const uint32_t p_token_idx = p_token_idx_base + idx;
      const uint32_t d_token_idx = d_token_idx_base + idx;
      const size_t pk_token_off = p_blk_off + p_token_idx * token_step_length;
      const size_t pv_token_off = pk_token_off + p_block_size/2;
      const size_t d_token_off = d_blk_off + d_token_idx * d_token_size / 2;
      const size_t dk_token_off = d_token_off + attn_group_off * token_step_length;
      const size_t dv_token_off = dk_token_off + d_block_size/2;
      // K
      per_cache_send_blocks.emplace_back(pk_token_off, dk_token_off, token_step_length);
      // V
      per_cache_send_blocks.emplace_back(pv_token_off, dv_token_off, token_step_length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return;
}

// actual parsing flashinfer block
static void parse_flashinfer_block_p_eq_d(
  size_t token_size,
  size_t block_size,
  uint32_t ntpb,
  const std::vector<uint32_t>& src_blocks,
  const std::vector<uint32_t>& dst_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &per_cache_send_blocks,
  const bool reach_last_token
){
  while (left_tokens > 0) {
    const auto blk_idx = wrote_tokens / ntpb;
    const auto token_idx = wrote_tokens % ntpb;

    assert(blk_idx < src_blocks.size() && blk_idx < dst_blocks.size());

    const size_t p_blk_off = src_blocks[blk_idx] * block_size;
    const size_t d_blk_off = dst_blocks[blk_idx] * block_size;

    // PD block size should be same
    auto tokens = std::min(ntpb - token_idx, left_tokens);
    size_t length = 0;
    // 按照block传
    if (tokens != ntpb){
      // request's last block or chunked prefill first/last block
      if (reach_last_token){ // request's last block
        length = block_size;
      } else if (token_idx == 0) {
        // chunked prefill last block
        length = 0;
      } else { // chunked prefill first block
        if (token_idx + tokens == ntpb) {
          // 达到了block的边界，整个block都传输
          length = block_size;
        } else {
          assert(token_idx + tokens < ntpb);
          // 开头在block中间，结束的token也在block中间，并且请求未完成，顺延到下个step传输
          length = 0;
        }
      }
    } else { // tokens == ntpb, send full block
      length = block_size;
    }
    if (length > 0) {
      per_cache_send_blocks.emplace_back(p_blk_off, d_blk_off, length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return;
}

// 需要按照head维度进行分割，不需要区分p gt d还是p eq d
static void parse_flashinfer_HND_block(
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
  std::vector<IpcBlock> &per_cache_send_blocks
){ // (num_blocks, 2, num_kv_heads, block_size, head_size)
  while (left_tokens > 0) {
    auto block_idx = wrote_tokens / ntpb;
    auto token_idx = wrote_tokens % ntpb;
    auto p_blk_off = src_blocks[block_idx] * p_block_size;
    auto d_blk_off = dst_blocks[block_idx] * d_block_size;

    auto tokens = std::min(ntpb - token_idx, left_tokens);

    auto p_head_num = env_attn_head_num();
    const auto p_head_size = p_block_size / 2 / p_head_num;
    const auto p_k_block_size = p_block_size / 2;
    const auto d_k_block_size = d_block_size / 2;

    // 如果按token粒度发送就需要按照head进行分割
    for (auto head_idx = 0; head_idx < p_head_num; ++head_idx) {
      auto p_k_off = p_blk_off + head_idx * p_head_size;
      auto d_k_off = d_blk_off + attn_group_off * p_k_block_size + head_idx * p_head_size;
      auto p_v_off = p_blk_off + p_k_block_size;
      auto d_v_off = d_blk_off + d_k_block_size;
      auto length = p_head_size / ntpb * tokens;

      per_cache_send_blocks.emplace_back(p_k_off, d_k_off, length);
      per_cache_send_blocks.emplace_back(p_v_off, d_v_off, length);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return ;
}

static void do_parse_block_send_p_eq_d(
  size_t block_size, size_t token_size,
  const ReqSendTask *task,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // ntpb: number tokens per block
  uint32_t src_ntpb = block_size / token_size;
  uint32_t dst_ntpb = src_ntpb;
  assert(src_ntpb * token_size == block_size);

  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;
  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  // assert(src_blocks.size() == dst_blocks.size());
  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + left_tokens / src_ntpb + 3);

  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / src_ntpb;
    auto src_token_idx = wrote_tokens % src_ntpb;
    assert(src_block_idx < src_blocks.size());
    size_t src_offset = src_blocks[src_block_idx] * block_size
        + src_token_idx * token_size;
    auto tokens = std::min(src_ntpb - src_token_idx, left_tokens);
    auto dst_block_idx = wrote_tokens / dst_ntpb;
    auto dst_token_idx = wrote_tokens % dst_ntpb;
    assert(dst_block_idx < dst_blocks.size());
    size_t dst_offset = dst_blocks[dst_block_idx] * block_size
        + dst_token_idx * token_size;
    size_t length = tokens * token_size;
    per_cache_send_blocks.emplace_back(src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return ;
}

// NOTE(llx) Qwen3-next block parsing special case:
// - The last three blocks are reserved for GDN blocks
// - KVT length calculation must account for these additional blocks: (new_tokens + 3 * ntpb)
// - During chunked prefill processing, GDN blocks transfer occurs only upon request completion
static void do_parse_hybrid_block_send_p_eq_d(
  size_t block_size, size_t token_size,
  const ReqSendTask *task,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // ntpb: number tokens per block
  uint32_t src_ntpb = block_size / token_size;
  uint32_t dst_ntpb = src_ntpb;
  assert(src_ntpb * token_size == block_size);

  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  // already cached tokens
  auto wrote_tokens = task->seen_tokens;
  // tokens to be written, aligned with block size, could oversend in last block
  auto left_tokens = task->new_tokens;
  // assert(src_blocks.size() == dst_blocks.size());
  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + left_tokens / src_ntpb + 6);
  // GDN block parsing only happens when reaching last token(finish chunked prefill)
  if (task->reach_last_token){
    left_tokens = src_ntpb * src_blocks.size() - wrote_tokens;
  }

  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / src_ntpb;
    auto src_token_idx = wrote_tokens % src_ntpb;
    assert(src_block_idx < src_blocks.size());

    auto tokens = std::min<uint32_t>(src_ntpb - src_token_idx, left_tokens);

    auto dst_block_idx = wrote_tokens / dst_ntpb;
    auto dst_token_idx = wrote_tokens % dst_ntpb;
    assert(dst_block_idx < dst_blocks.size());
    // align with kv block size
    size_t src_offset = src_blocks[src_block_idx] * block_size
      + src_token_idx * token_size;
    size_t dst_offset = dst_blocks[dst_block_idx] * block_size
      + dst_token_idx * token_size;
    size_t length = tokens * token_size;
    per_cache_send_blocks.emplace_back(src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return ;
}

static void do_parse_hybrid_block_send_p_gt_d(
  size_t p_token_size, size_t d_token_size,
  uint32_t src_ntpb, // ntpb: number tokens per block
  uint32_t dst_ntpb, 
  uint32_t attn_group_n,
  uint32_t attn_group_off, // attn offset in pd rank mapping group
  uint32_t gdn_group_n,
  uint32_t gdn_group_off, // gdn offset in pd rank mapping group
  const std::bitset<MAX_TP_SIZE>& validranks,
  uint32_t p_tp_rank,
  const ReqSendTask *task,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // Next's p block size might not equal to d block size
  const size_t p_block_size = p_token_size * src_ntpb;
  const size_t d_block_size = d_token_size * dst_ntpb;
  const auto &src_blocks = task->src_blocks();
  const auto &dst_blocks = task->dst_blocks();
  // already cached tokens
  auto wrote_tokens = task->seen_tokens;
  // tokens to be written, aligned with block size, could oversend in last block
  auto left_tokens = task->new_tokens;

  const auto p_gdn_blks_num = env_gdn_block_num();
  const auto d_gdn_blks_num = p_gdn_blks_num;
  // 现在develop修改了排列顺序，先 GDN Block 再 Attn Block
  const auto p_gdn_blk_idx = 0;
  const auto d_gdn_blk_idx = 0;
  // 这里要重新算gdn的offset，因为pd block size不一样
  // 讲道理，现在是先算GDN，不过反正传的数据量也不多，先等最后一起传，后面再说
  if (task->reach_last_token){
    std::vector<size_t> conv_state_shape = *env_conv_state_shape();
    std::vector<size_t> ssm_state_shape = *env_ssm_state_shape();
    uint32_t gdn_element_size = env_gdn_element_size();
    // TODO: Not elegant, refactor this later
    const auto p_conv_block_size = conv_state_shape[1] * conv_state_shape[2] * gdn_element_size;
    const auto p_ssm_block_size = ssm_state_shape[1] * ssm_state_shape[2] * ssm_state_shape[3] * gdn_element_size;
    for (auto i = 0; i < p_gdn_blks_num; ++i) {
      const size_t p_blk_idx = p_gdn_blk_idx + i;
      const size_t d_blk_idx = d_gdn_blk_idx + i;
      // GDN Block:[[Conv][SSM][Padding]]
      const size_t p_gdn_off = src_blocks[p_blk_idx] * p_block_size;
      const size_t d_gdn_off = dst_blocks[d_blk_idx] * d_block_size;
      // conv state在最后一维分割，需要分成conv_state_shape[1]个小包
      const auto conv_step_length = conv_state_shape[2] * gdn_element_size;
      for (size_t conv_dim = 0; conv_dim < conv_state_shape[1]; ++conv_dim) {
        const size_t p_conv_off = p_gdn_off + conv_dim * conv_step_length;
        const size_t d_conv_off = d_gdn_off + (gdn_group_off + conv_dim * gdn_group_n)* conv_step_length;
        per_cache_send_blocks.emplace_back(p_conv_off, d_conv_off, conv_step_length);
      }
      const size_t p_ssm_off = p_gdn_off + p_conv_block_size;
      const size_t d_ssm_off = d_gdn_off + gdn_group_n * p_conv_block_size + gdn_group_off * p_ssm_block_size;
      per_cache_send_blocks.emplace_back(p_ssm_off, d_ssm_off, p_ssm_block_size);
    }
  }
  if (!validranks[p_tp_rank]) {
    // kv 复制 case: 仅有效 worker tp rank 参与attn发送.
    return ;
  }
  parse_hybrid_attn_block(
    p_gdn_blks_num,
    d_gdn_blks_num,
    src_ntpb,
    dst_ntpb,
    p_token_size,
    d_token_size,
    p_block_size,
    d_block_size,
    src_blocks,
    dst_blocks,
    wrote_tokens, 
    left_tokens, 
    attn_group_off,
    per_cache_send_blocks
  );
  return ;
}

// RAGGED_FLASH_CACHE_SHAPE
// Now this method also support Dpsk in vllm
static void parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  // as tp size is same, token size should be same;
  assert(src_worker_info->tp_size == dst_worker_info->tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);

  const auto &token_sizes = src_worker_info->token_sizes;
  const auto &block_sizes = src_worker_info->block_sizes;

  assert(token_sizes == dst_worker_info->token_sizes);
  assert(block_sizes == dst_worker_info->block_sizes);
  assert(token_sizes.size() == block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());
  // Should only have one cache in one layer for RAGGED_FLASH_CACHE_SHAPE.
  assert(token_sizes.size() == 1);
  assert(block_sizes.size() == 1);
  send_blocks.resize(token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
  do_parse_block_send_p_eq_d(
    block_sizes[0], token_sizes[0],
    task, per_cache_send_blocks
  );
  return;
}

static void vllm_parse_block_send_multi_tensor_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  // as tp size is same, token size should be same;
  assert(src_worker_info->tp_size == dst_worker_info->tp_size);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);

  const auto &token_sizes = src_worker_info->token_sizes;
  const auto &block_sizes = src_worker_info->block_sizes;

  assert(token_sizes == dst_worker_info->token_sizes);
  assert(block_sizes == dst_worker_info->block_sizes);
  assert(token_sizes.size() == block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());

  // Each entry of send_blocks stores the block list for one cache tensor within the layer
  send_blocks.resize(token_sizes.size());
  for (size_t i = 0; i < token_sizes.size(); ++i) {
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[i];
    do_parse_block_send_p_eq_d(
      block_sizes[i], token_sizes[i],
      task, per_cache_send_blocks
    );
  }
  return;
}

// FLASH_CACHE_SHAPE
static void vllm_parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  assert(src_worker_info->tp_size == dst_worker_info->tp_size);

  auto const& kv_token_sizes = src_worker_info->token_sizes;
  auto const& kv_block_sizes = src_worker_info->block_sizes;

  // as tp size is same, token size should be same;
  assert(kv_token_sizes == dst_worker_info->token_sizes);
  assert(kv_block_sizes == dst_worker_info->block_sizes);
  assert(src_worker_info->worker_tp_rank == dst_worker_info->worker_tp_rank);
  assert(kv_token_sizes.size() == kv_block_sizes.size());
  assert(dst_worker_info->token_sizes.size() == dst_worker_info->block_sizes.size());
  // Should only have one cache in one layer for FLASH_CACHE_SHAPE.
  assert(kv_token_sizes.size() == 1);
  assert(kv_block_sizes.size() == 1);
  auto &kv_token_size = kv_token_sizes[0];
  auto &kv_block_size = kv_block_sizes[0];
  size_t const k_token_size = kv_token_size / 2;
  size_t const k_block_size = kv_block_size / 2;
  assert(2 * k_token_size == kv_token_size);
  assert(2 * k_block_size == kv_block_size);

  size_t const src_layer_size = k_block_size * src_worker_info->layer_num_blocks;
  size_t const dst_layer_size = k_block_size * dst_worker_info->layer_num_blocks;
  // src_layer_size, dst_layer_size 并不一定要相等.

  send_blocks.resize(kv_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
  size_t sb_idx = per_cache_send_blocks.size();  // sb: send block~
  do_parse_block_send_p_eq_d(k_block_size, k_token_size, task, per_cache_send_blocks);
  size_t const sb_end_idx = per_cache_send_blocks.size();

  // emplace_back v tensor block, which is non-contigious with k tensor
  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sb_end_idx - sb_idx);
  for (; sb_idx < sb_end_idx; ++sb_idx) {
    auto sb = per_cache_send_blocks[sb_idx];
    sb.src_offset += src_layer_size;
    sb.dst_offset += dst_layer_size;
    per_cache_send_blocks.emplace_back(std::move(sb));
  }
  return ;
}

// Parse block using shape QWEN3_NEXT_FLASH_CACHE_SHAPE in vllm
// Qwen3-next model's attn block and mamba block is block continous and padded
// in same block size, so kvt will parse it using vllm's token_size and block_size
static void vllm_parse_hybrid_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
    // as tp size is same, token size/block size should be same;
    assert(src_worker_info->tp_size == dst_worker_info->tp_size);
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
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
    do_parse_hybrid_block_send_p_eq_d(
      kv_block_sizes[0], kv_token_sizes[0],
      task, per_cache_send_blocks
    );
    return ;
}

static void vllm_parse_hybrid_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
    const auto & p_block_sizes = src_worker_info->block_sizes;
    const auto & d_block_sizes = dst_worker_info->block_sizes;
    const auto & p_token_sizes = src_worker_info->token_sizes;
    const auto & d_token_sizes = dst_worker_info->token_sizes;

    assert(src_worker_info->tp_size > dst_worker_info->tp_size);
    assert(src_worker_info->tp_size % dst_worker_info->tp_size == 0);

    // GDN不会受到head < tp size的困扰
    const uint32_t p_origin_tp_size = env_origin_p_tp_size();
    const uint32_t gdn_group_n = p_origin_tp_size / dst_worker_info->tp_size;
    assert(src_worker_info->worker_tp_rank / gdn_group_n == dst_worker_info->worker_tp_rank);
    const uint32_t gdn_group_off = src_worker_info->worker_tp_rank % gdn_group_n;
    auto const validranks = env_p_valid_ranks();
    uint32_t const worker_tp_rank = (validranks << (validranks.size() - src_worker_info->worker_tp_rank)).count();
    const uint32_t attn_group_n = validranks.count() / dst_worker_info->tp_size;
    const uint32_t attn_group_off = worker_tp_rank % attn_group_n;

    // Should only have one cache in one layer.
    assert(p_token_sizes.size() == 1);
    assert(d_token_sizes.size() == 1);
    assert(p_block_sizes.size() == 1);
    assert(d_block_sizes.size() == 1);
    const auto p_token_size = p_token_sizes[0];
    const auto d_token_size = d_token_sizes[0];
    const auto p_block_size = p_block_sizes[0];
    const auto d_block_size = d_block_sizes[0];

    assert(d_token_size == p_token_size * attn_group_n);

    // ntpb: number tokens per block, pd block size should align
    const uint32_t src_ntpb = p_block_size / p_token_size;
    const uint32_t dst_ntpb = d_block_size / d_token_size;
    assert(src_ntpb * p_token_size == p_block_size);
    assert(dst_ntpb * d_token_size == d_block_size);

    send_blocks.resize(p_token_sizes.size());
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
    size_t sb_idx = per_cache_send_blocks.size();  // sb: send block~
    do_parse_hybrid_block_send_p_gt_d(
      p_token_size, d_token_size,
      src_ntpb, dst_ntpb,
      attn_group_n, attn_group_off,
      gdn_group_n, gdn_group_off,
      validranks,
      src_worker_info->worker_tp_rank,
      task, per_cache_send_blocks
    );
    return;
  }


// This is for vllm
static void parse_block_send_gt(
  size_t p_token_size, size_t p_k_size, size_t d_token_size,
  uint32_t ntpb, size_t group_off,
  const std::vector<uint32_t>& p_blocks,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(p_k_size == p_token_size || p_k_size * 2 == p_token_size);
  const size_t p_block_size = p_token_size * ntpb;
  const size_t d_block_size = d_token_size * ntpb;

  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + left_tokens);
  while (left_tokens > 0) {
    const uint32_t block_idx = wrote_tokens / ntpb;
    assert(block_idx < p_blocks.size() && block_idx < d_blocks.size());
    const uint32_t token_idx_base = wrote_tokens % ntpb;
    const size_t p_blk_off = p_blocks[block_idx] * p_block_size;
    const size_t d_blk_off = d_blocks[block_idx] * d_block_size;
    const uint32_t tokens = std::min(ntpb - token_idx_base, left_tokens);

    for (uint32_t idx = 0; idx < tokens; ++idx) {
      const uint32_t token_idx = token_idx_base + idx;
      const size_t pk_token_off = p_blk_off + token_idx * p_token_size;
      const size_t d_token_off = d_blk_off + token_idx * d_token_size;
      const size_t dk_token_off = d_token_off + group_off * p_k_size;
      per_cache_send_blocks.emplace_back(pk_token_off, dk_token_off, p_k_size);
    }
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
  return;
}

static void do_parse_block_send(
  const WorkerInfo *p_info,  // src
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,  // dst
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  // for bladellm
  // cache shape [num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;

  assert(p_block_sizes.size() == d_block_sizes.size());
  assert(p_token_sizes.size() == d_token_sizes.size());
  assert(p_block_sizes.size() == p_token_sizes.size());
  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);

  const size_t p_block_size = p_block_sizes[0];
  const size_t d_block_size = d_block_sizes[0];
  const size_t p_token_size = p_token_sizes[0];
  const size_t d_token_size = d_token_sizes[0];
  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];

  assert(p_info->tp_size > d_info->tp_size);
  assert((p_info->tp_size % d_info->tp_size) == 0);
  const uint32_t group_n = p_info->tp_size / d_info->tp_size;
  assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
  const uint32_t group_off = p_info->worker_tp_rank % group_n;
  assert(d_token_size == p_token_size * group_n);
  assert(d_token_size % 2 == 0);
  assert(p_token_size % 2 == 0);
  const size_t p_k_size = p_token_size / 2;
  const size_t d_k_size = d_token_size / 2;
  assert(d_k_size == p_k_size * group_n);
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_token_size;
  assert(ntpb * p_token_size == p_block_size);
  assert(ntpb * d_token_size == d_block_size);

  const size_t sbsize = per_cache_send_blocks.size();
  // Same block id, different token_size/block_size
  parse_block_send_gt(
    p_token_size, p_k_size, d_token_size, ntpb, group_off,
    p_blocks, d_blocks, wrote_tokens, left_tokens,
    per_cache_send_blocks
  );
  const size_t sbsize_after = per_cache_send_blocks.size();

  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sbsize_after - sbsize);
  for (size_t idx = sbsize; idx < sbsize_after; ++idx) {
    const auto& sb = per_cache_send_blocks[idx];
    const size_t pv_token_off = sb.src_offset + p_k_size;
    const size_t dv_token_off = sb.dst_offset + d_k_size;
    assert(p_k_size == sb.length);
    per_cache_send_blocks.emplace_back(pv_token_off, dv_token_off, p_k_size);
  }
  return;
}

static void do_multi_tensor_parse_block_send(
  const WorkerInfo *p_info,  // src
  const std::vector<uint32_t>& p_blocks,
  const WorkerInfo *d_info,  // dst
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  // for bladellm
  // cache shape [num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;

  assert(p_block_sizes.size() == d_block_sizes.size());
  assert(p_token_sizes.size() == d_token_sizes.size());
  assert(p_block_sizes.size() == p_token_sizes.size());

  send_blocks.resize(p_token_sizes.size());
  for(size_t cache_idx = 0; cache_idx < p_block_sizes.size(); cache_idx++){
    const size_t p_block_size = p_block_sizes[cache_idx];
    const size_t d_block_size = d_block_sizes[cache_idx];
    const size_t p_token_size = p_token_sizes[cache_idx];
    const size_t d_token_size = d_token_sizes[cache_idx];

    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[cache_idx];

    assert(p_info->tp_size > d_info->tp_size);
    assert((p_info->tp_size % d_info->tp_size) == 0);
    const uint32_t group_n = p_info->tp_size / d_info->tp_size;
    assert(p_info->worker_tp_rank / group_n == d_info->worker_tp_rank);
    const uint32_t group_off = p_info->worker_tp_rank % group_n;
    assert(d_token_size == p_token_size * group_n);
    assert(d_token_size % 2 == 0);
    assert(p_token_size % 2 == 0);
    const size_t p_k_size = p_token_size / 2;
    const size_t d_k_size = d_token_size / 2;
    assert(d_k_size == p_k_size * group_n);
    // ntpb: number tokens per block
    const uint32_t ntpb = p_block_size / p_token_size;
    assert(ntpb * p_token_size == p_block_size);
    assert(ntpb * d_token_size == d_block_size);

    const size_t sbsize = per_cache_send_blocks.size();
    // Same block id, different token_size/block_size
    parse_block_send_gt(
      p_token_size, p_k_size, d_token_size, ntpb, group_off,
      p_blocks, d_blocks, wrote_tokens, left_tokens,
      per_cache_send_blocks
    );
    const size_t sbsize_after = per_cache_send_blocks.size();

    per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sbsize_after - sbsize);
    for (size_t idx = sbsize; idx < sbsize_after; ++idx) {
      const auto& sb = per_cache_send_blocks[idx];
      const size_t pv_token_off = sb.src_offset + p_k_size;
      const size_t dv_token_off = sb.dst_offset + d_k_size;
      assert(p_k_size == sb.length);
      per_cache_send_blocks.emplace_back(pv_token_off, dv_token_off, p_k_size);
    }
  }
  return;
}

// FLASH_CACHE_SHAPE
// (2, num_blocks, block_size, num_kv_heads, head_dim)
static void vllm_parse_block_send_p_gt_d(
  const WorkerInfo *p_info,  // prefill
  const WorkerInfo *d_info,  // decode
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  auto const validranks = env_p_valid_ranks();
  if (!validranks[p_info->worker_tp_rank]) {
    // kv 复制 case: 仅有效 worker tp rank 参与发送.
    return ;
  }
  uint32_t const worker_tp_rank = (validranks << (validranks.size() - p_info->worker_tp_rank)).count();
  const auto & p_block_sizes = p_info->block_sizes;
  const auto & d_block_sizes = d_info->block_sizes;
  const auto & p_token_sizes = p_info->token_sizes;
  const auto & d_token_sizes = d_info->token_sizes;
  assert(p_info->tp_size > d_info->tp_size);
  assert((p_info->tp_size % d_info->tp_size) == 0);
  const uint32_t group_n = p_info->tp_size / d_info->tp_size;
  assert(worker_tp_rank / group_n == d_info->worker_tp_rank);
  const uint32_t group_off = worker_tp_rank % group_n;

  // Should only have one cache in one layer.
  assert(p_token_sizes.size() == 1);
  assert(d_token_sizes.size() == 1);
  assert(p_block_sizes.size() == 1);
  assert(d_block_sizes.size() == 1);
  const auto p_token_size = p_token_sizes[0];
  const auto d_token_size = d_token_sizes[0];
  const auto p_block_size = p_block_sizes[0];
  const auto d_block_size = d_block_sizes[0];

  assert(d_token_size == p_token_size * group_n);
  assert(d_token_size % 2 == 0);
  assert(p_token_size % 2 == 0);
  const size_t p_k_size = p_token_size / 2;
  const size_t d_k_size = d_token_size / 2;
  assert(d_k_size == p_k_size * group_n);
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_token_size;
  assert(ntpb * p_token_size == p_block_size);
  assert(ntpb * d_token_size == d_block_size);

  send_blocks.resize(p_token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
  size_t sb_idx = per_cache_send_blocks.size();  // sb: send block~
  parse_block_send_gt(
    p_k_size, p_k_size, d_k_size, ntpb, group_off,
    task->src_blocks(), task->dst_blocks(),
    task->seen_tokens, task->new_tokens, per_cache_send_blocks);
  size_t const sb_end_idx = per_cache_send_blocks.size();
  size_t const pk_layer_size = ntpb * p_k_size * p_info->layer_num_blocks;
  size_t const dk_layer_size = ntpb * d_k_size * d_info->layer_num_blocks;

  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + sb_end_idx - sb_idx);
  for (; sb_idx < sb_end_idx; ++sb_idx) {
    auto sb = per_cache_send_blocks[sb_idx];
    sb.src_offset += pk_layer_size;
    sb.dst_offset += dk_layer_size;
    per_cache_send_blocks.emplace_back(std::move(sb));
  }
  return;
}

static void parse_block_send_p_gt_d_dpsk(
  const WorkerInfo *p_info,  // src
  const WorkerInfo *d_info,  // dst
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  assert(p_info->token_sizes == d_info->token_sizes);
  // 此时表明 kvcache 在各个 P rank 之间是完全一样的.
  // 只需要 tp rank=0 的 worker 传输 kvcache 即可.
  // 后续这里可以让所有 worker 都参与进来, 每个 worker 传 1 部分.
  if (p_info->worker_tp_rank != 0) {
    return ;
  }

  assert(d_info->worker_tp_rank == 0);
  // PD Block 包含的token数需要一致
  assert(p_info->block_sizes == d_info->block_sizes);
  auto &token_sizes = p_info->token_sizes;
  auto &block_sizes = p_info->block_sizes;
  assert(token_sizes.size() == block_sizes.size());
  // Should only have one cache in one layer for RAGGED_FLASH_CACHE_SHAPE.
  assert(token_sizes.size() == 1);
  assert(block_sizes.size() == 1);

  send_blocks.resize(token_sizes.size());
  std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
  do_parse_block_send_p_eq_d(
    block_sizes[0], token_sizes[0], task, per_cache_send_blocks
  );
  return;
}

// This is for bladellm/vllm dpsk
static void parse_block_send_p_gt_d(
  const WorkerInfo *p_info,  // src
  const WorkerInfo *d_info,  // dst
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {

  return do_parse_block_send(
    p_info, task->src_blocks(),
    d_info, task->dst_blocks(),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
}

static void vllm_parse_block_send_multi_tensor_p_gt_d(
  const WorkerInfo *p_info,  // src
  const WorkerInfo *d_info,  // dst
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  if (p_info->token_sizes == d_info->token_sizes) {
    // 此时表明 kvcache 在各个 P rank 之间是完全一样的.
    // 只需要 tp rank=0 的 worker 传输 kvcache 即可.
    // 后续这里可以让所有 worker 都参与进来, 每个 worker 传 1 部分.
    if (p_info->worker_tp_rank != 0) {
      return ;
    }

    assert(d_info->worker_tp_rank == 0);
    // PD Block 包含的token数需要一致
    assert(p_info->block_sizes == d_info->block_sizes);
    auto &token_sizes = p_info->token_sizes;
    auto &block_sizes = p_info->block_sizes;
    assert(token_sizes.size() == block_sizes.size());

    send_blocks.resize(token_sizes.size());
    for (size_t i = 0; i < token_sizes.size(); ++i) {
      std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[i];
      do_parse_block_send_p_eq_d(
        block_sizes[i], token_sizes[i], task, per_cache_send_blocks
      );
    }
    return;
  }

  return do_multi_tensor_parse_block_send(
    p_info, task->src_blocks(),
    d_info, task->dst_blocks(),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
}

static void vllm_parse_block_send_multi_tensor_p_lt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  size_t cache_idx = send_blocks.size();
  do_multi_tensor_parse_block_send(
    d_info, task->dst_blocks(),
    p_info, task->src_blocks(),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
  for (; cache_idx < send_blocks.size(); ++cache_idx) {
    auto &per_cache_sbs = send_blocks[cache_idx];
    for(size_t sb_idx = 0; sb_idx < per_cache_sbs.size(); ++sb_idx){
      std::swap(per_cache_sbs[sb_idx].src_offset, per_cache_sbs[sb_idx].dst_offset);
    }
  }
  return ;
}

static void vllm_parse_flashinfer_block_send_p_eq_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
    assert(p_info->tp_size == d_info->tp_size);

    const auto & p_block_sizes = p_info->block_sizes;
    const auto & d_block_sizes = d_info->block_sizes;
    const auto & p_token_sizes = p_info->token_sizes;
    const auto & d_token_sizes = d_info->token_sizes;

    assert(p_token_sizes.size() == 1);
    assert(d_token_sizes.size() == 1);
    assert(p_block_sizes.size() == 1);
    assert(d_block_sizes.size() == 1);
    const auto p_token_size = p_token_sizes[0];
    const auto d_token_size = d_token_sizes[0];
    const auto p_block_size = p_block_sizes[0];
    const auto d_block_size = d_block_sizes[0];

    const uint32_t src_ntpb = p_block_size / p_token_size;
    const uint32_t dst_ntpb = d_block_size / d_token_size;
    assert(src_ntpb == dst_ntpb);

    const auto &src_blocks = task->src_blocks();
    const auto &dst_blocks = task->dst_blocks();
    auto wrote_tokens = task->seen_tokens;
    auto left_tokens = task->new_tokens;

    send_blocks.resize(1);
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
    parse_flashinfer_HND_block(
      p_token_size, d_token_size,
      p_block_size, d_block_size,
      src_ntpb,
      src_blocks, dst_blocks,
      wrote_tokens, left_tokens,
      1, // attn_group_n
      0, // attn_group_off
      per_cache_send_blocks
    );
    return;
}

static void vllm_parse_flashinfer_block_send_p_gt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
    auto const validranks = env_p_valid_ranks();
    if (!validranks[p_info->worker_tp_rank]) {
      return ;
    }
    uint32_t const worker_tp_rank = (validranks << (validranks.size() - p_info->worker_tp_rank)).count();
    const uint32_t attn_group_n = validranks.count() / d_info->tp_size;
    const uint32_t attn_group_off = worker_tp_rank % attn_group_n;
    const auto & p_block_sizes = p_info->block_sizes;
    const auto & d_block_sizes = d_info->block_sizes;
    const auto & p_token_sizes = p_info->token_sizes;
    const auto & d_token_sizes = d_info->token_sizes;
    assert(p_info->tp_size > d_info->tp_size);
    assert((p_info->tp_size % d_info->tp_size) == 0);
    assert(worker_tp_rank / attn_group_n == d_info->worker_tp_rank);

    // Should only have one cache in one layer.
    // Block Layout:Block[[K1,K2,K3...], [V1,V2,V3...]]
    assert(p_token_sizes.size() == 1);
    assert(d_token_sizes.size() == 1);
    assert(p_block_sizes.size() == 1);
    assert(d_block_sizes.size() == 1);
    const auto p_token_size = p_token_sizes[0];
    const auto d_token_size = d_token_sizes[0];
    const auto p_block_size = p_block_sizes[0];
    const auto d_block_size = d_block_sizes[0];

    const uint32_t src_ntpb = p_block_size / p_token_size;
    const uint32_t dst_ntpb = d_block_size / d_token_size;
    assert(src_ntpb == dst_ntpb);

    const auto &src_blocks = task->src_blocks();
    const auto &dst_blocks = task->dst_blocks();
    auto wrote_tokens = task->seen_tokens;
    auto left_tokens = task->new_tokens;

    send_blocks.resize(1);
    std::vector<IpcBlock> &per_cache_send_blocks = send_blocks[0];
    parse_flashinfer_HND_block(
      p_token_size, d_token_size,
      p_block_size, d_block_size,
      src_ntpb,
      src_blocks, dst_blocks,
      wrote_tokens, left_tokens,
      attn_group_n,
      attn_group_off,
      per_cache_send_blocks
    );
    return;
}

static void parse_block_send_p_lt_d(
  const WorkerInfo *p_info,
  const WorkerInfo *d_info,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks) {
  size_t cache_idx = send_blocks.size();
  do_parse_block_send(
    d_info, task->dst_blocks(),
    p_info, task->src_blocks(),
    task->seen_tokens,
    task->new_tokens,
    send_blocks);
  for (; cache_idx < send_blocks.size(); ++cache_idx) {
    auto &per_cache_sbs = send_blocks[cache_idx];
    for(size_t sb_idx = 0; sb_idx < per_cache_sbs.size(); ++sb_idx){
      std::swap(per_cache_sbs[sb_idx].src_offset, per_cache_sbs[sb_idx].dst_offset);
    }
  }
  return ;
}

using ParseBlockFunc = decltype(parse_block_send_p_lt_d);

static bool same_dst(const WorkerInfo& l, const WorkerInfo& r) noexcept {
  return l.inst_id == r.inst_id &&
         l.worker_id == r.worker_id &&
         l.tp_size == r.tp_size &&
         l.worker_tp_rank == r.worker_tp_rank &&
         l.block_sizes == r.block_sizes &&
         l.token_sizes == r.token_sizes &&
         l.layer_num_blocks == r.layer_num_blocks &&
         l.num_layers == r.num_layers &&
         l.transfer_protocols == r.transfer_protocols;
}


static char* expand_vec(std::vector<char>& buf, size_t s) {
  auto old_s = buf.size();
  buf.resize(old_s + s);
  return buf.data() + old_s;
}

// see vllm vllm/v1/hybrid_connector/__init__.py
static constexpr uint32_t SEND_DONE_REQ = 0x20181219ul;
static constexpr uint32_t SAVE_DONE2_REQ = 0x20181223ul;
static constexpr uint32_t SEND_DONE_RESP = 0x91218102ul;

struct KvSendStub::TaskContext {
  KvSendStub* const stub = nullptr;

  // runtime state
  std::vector<char> send_done_buf;
  std::optional<FdGuard> send_done_sock;
  const sockaddr_in* const send_done_addr = env_send_done_addr();  // OWNER: GLOBAL
  std::string flush_out_buf;
  std::vector<std::vector<IpcBlock>> send_blocks;
  std::vector<const ReqSendTask *> send_req;
  std::vector<const ReqSendTask *> finished_req;
  TPKind tpkind = TPKind::UNKNOWN;
  ParseBlockFunc* parse_block = nullptr;
  std::optional<WorkerInfo> dstinfo;
  Channel ch;
  uint64_t wait_time_us = 0;
  uint64_t wait_and_send_us = 0;
  uint64_t send_notify_us = 0;
  Timepoint iter_start_ts;
  Timepoint send_finish_ts;
public:
  TaskContext(KvSendStub* s) noexcept
    : stub(s) {
  }

  void do_task(BatchSendTask& batch) noexcept {
    auto& self = *this;
    auto const step_idx = batch.step->step_idx;
    const auto& dst_id = self.stub->dstid_;
    const auto dst_worker_id = self.stub->dstworkerid_;

    // 用于提高 https://aone.alibaba-inc.com/v2/project/664220/req/60815172 复现概率.
    fault_inject_sleep(300 * 1000);

    self.iter_start_ts = SteadyClock::now();  // iterator begin;
    auto const queue_us = elapse_us(batch.step->start_send_ts, self.iter_start_ts);

    // prepare
    assert(self.flush_out_buf.empty());
    assert(self.send_req.empty());
    assert(self.finished_req.empty());
    assert(self.send_blocks.empty());
    assert(!batch.tasks.empty());
    for (const auto& task : batch.tasks) {
      assert(task.dst_inst_id() == dst_id);
      assert(task.dst_worker_id() == dst_worker_id);
      if (task.reach_last_token) {
        self.finished_req.emplace_back(&task);
      }
      if (task.new_tokens <= 0) {
        continue;
      }

      if (task.state() == ReqState::FAILED) {
        continue;
      }
      assert(task.state() == ReqState::INPROCESS);
      assert(task.new_tokens > 0);
      self.send_req.emplace_back(&task);
    }

    if (!self.send_req.empty()) {
      try {
        self.try_create_channel();
        const auto& srcinfo = self.stub->src_info_;
        assert(self.parse_block != nullptr);
        assert(self.tpkind != TPKind::UNKNOWN);
        for (auto* task : self.send_req) {
          self.parse_block(&srcinfo, &self.dstinfo.value(), task, self.send_blocks);
        }
        if (!self.send_blocks.empty() && !self.send_blocks[0].empty()) {
#ifndef NDEBUG
          for (const auto& sb : self.send_blocks) {
            assert(!sb.empty());
          }
#endif
          self.do_send(batch);
        }
      } catch (std::exception& ex) {
        LOG(ERROR) << "KVT tx_stub fail to send data. DstId=" << dst_id
                  << ",DstWorkerId=" << dst_worker_id
                  << ",StepIdx=" << step_idx
                  << ',' << self.flush_out_buf
                  << ",ex=" << ex.what();
        self.ch.reset();
        for (const auto& task : batch.tasks) {
          assert(task.state() == ReqState::INPROCESS || task.state() == ReqState::FAILED);
          task.set_state(ReqState::FAILED);
        }
      }
    }

    for (auto task : self.finished_req) {
      assert(task->reach_last_token);
      if (task->state() == ReqState::INPROCESS) {
        auto final_state = ReqState::OK;
        if (task->new_tokens <= 0) {
          final_state = ReqState::FAILED;
        }
        task->set_state(final_state);
      }

      LOG(INFO) << "KVT tx_stub. DstId=" << dst_id << ",DstWorkerId=" << dst_worker_id
                << ",StepIdx=" << step_idx
                << ",State=" << static_cast<int>(task->state())
                << ",FinishedReq=" << task->req_id();
    }

    self.send_notify_us = 0;
    if (!self.finished_req.empty()) {
      self.send_done(self.finished_req);
      auto send_notify_end_ts = SteadyClock::now();
      self.send_notify_us = elapse_us(self.send_finish_ts, send_notify_end_ts);
      self.send_finish_ts = send_notify_end_ts;
    }

    batch.step->update_last_send_finish_ts(self.send_finish_ts);
    LOG(INFO) << "SendStubMetrics. DstId=" << dst_id << ",DstWorkerId=" << dst_worker_id
              << ",StepIdx=" << step_idx << ",FinishedReqSize=" << self.finished_req.size()
              << "," << self.flush_out_buf
              << ",QueueUs=" << queue_us
              << ",WaitUs=" << self.wait_time_us
              << ",WaitAndSendUs=" << self.wait_and_send_us
              << ",SendNotifyUs=" << self.send_notify_us
              << ",SendFinishTs=" << self.send_finish_ts.time_since_epoch().count();  //  send stub id
    return ;
  }

  void clear() noexcept {
    auto& self = *this;
    self.flush_out_buf.clear();
    self.send_blocks.clear();
    self.send_blocks.shrink_to_fit();
    self.send_req.clear();
    self.finished_req.clear();
  }
private:
  // send self.send_done_buf to self.send_done_sock
  void do_rpc_send_done(int respsize) {
    auto& self = *this;

    if (!self.send_done_sock.has_value()) {
      int sock = socket(AF_INET, SOCK_STREAM, 0);
      if (sock == -1) {
        int errno_bak = errno;
        auto errmsg = std::string("rpc send done: create socket error. errno=");
        errmsg += std::to_string(errno_bak);
        throw std::runtime_error(std::move(errmsg));
      }
      self.send_done_sock.emplace(sock);

      auto* addr = reinterpret_cast<const sockaddr*>(self.send_done_addr);
      int sysok = connect(sock, addr, sizeof(*self.send_done_addr));
      if (sysok == -1) {
        int errno_bak = errno;
        auto errmsg = std::string("rpc send done: connect error. errno=");
        errmsg += std::to_string(errno_bak);
        throw std::runtime_error(std::move(errmsg));
      }
    }
    int sock = self.send_done_sock->fd();

    const auto& sbuf = self.send_done_buf;
    int sysok = write_sock(sock, sbuf.data(), sbuf.size());
    if (sysok == -1) {
      int errno_bak = errno;
      auto errmsg = std::string("rpc send done: write error. errno=");
      errmsg += std::to_string(errno_bak);
      throw std::runtime_error(std::move(errmsg));
    }

    char respbuf[16];
    assert(respsize < int(sizeof(respbuf)));
    sysok = read_sock(sock, respbuf, respsize);
    if (sysok == -1) {
      int errno_bak = errno;
      auto errmsg = std::string("rpc send done: read error. errno=");
      errmsg += std::to_string(errno_bak);
      throw std::runtime_error(std::move(errmsg));
    }
    if (sysok != respsize) {
      auto errmsg = std::string("rpc send done: invalid response");
      throw std::runtime_error(std::move(errmsg));
    }
    return ;
  }

  void rpc_send_done(const std::vector<const ReqSendTask*>& reqs) {
    auto& self = *this;

    // see vllm disagg.py
    int respsize = 4;
    uint32_t header = SEND_DONE_REQ;
    uint32_t worker_tp_rank = self.stub->src_info_.worker_tp_rank;
    static constexpr uint32_t HAS_VER2 = 0x40000000u;
    static constexpr uint32_t CODE_INTERNALERROR = 500;
    assert(worker_tp_rank < 0xffff);
    worker_tp_rank |= HAS_VER2;
    uint32_t num_req = reqs.size();

    // len('cfb0aa74-6752-9bcd-879e-19d1d1cf368b-ee5d8cdc-57d1-478f-a1d7-5b0c05bd8fb3')
    // == 73 一个典型的 dash reqid 长度.
    self.send_done_buf.reserve((4 + 4 + 4 + num_req * (4 + 73 + 4 + 4)) * 2ul);

    self.send_done_buf.resize(4 + 4 + 4);
    memcpy(self.send_done_buf.data() + 0, &header, 4);
    memcpy(self.send_done_buf.data() + 4, &worker_tp_rank, 4);
    memcpy(self.send_done_buf.data() + 8, &num_req, 4);
    for (const auto* req : reqs) {
      assert(req->reach_last_token);
      uint32_t code = (req->state() == ReqState::OK) ? 0 : CODE_INTERNALERROR;
      uint32_t plen = req->end_tokens();
      const std::string& reqid = req->req_id();
      uint32_t rlen = reqid.size();
      char* dst = expand_vec(self.send_done_buf, 4 + 4 + 4 + rlen);
      memcpy(dst, &plen, 4);
      memcpy(dst + 4, &code, 4);
      memcpy(dst + 8, &rlen, 4);
      memcpy(dst + 12, reqid.data(), rlen);
    }

    if (env_send_done_head_kind() == SEND_SAVE_DONE_HEAD_KIND) {
      size_t const reqsize = self.send_done_buf.size();
      char* dst = expand_vec(self.send_done_buf, reqsize);
      memcpy(dst, self.send_done_buf.data(), reqsize);
      uint32_t save_header = SAVE_DONE2_REQ;
      memcpy(dst, &save_header, sizeof(uint32_t));
      respsize = 8;
    }

    // 考虑到我们使用的是一个长链接, 其可能已经失效了, 这里会在需要的时候,
    // 进行重试, 重新创建一个连接. 当然, 一次就好~
    try {
      self.do_rpc_send_done(respsize);
      return ;
    } catch (const std::exception& ex) {
      LOG(ERROR) << "do_rpc_send_done failed. ex=" << ex.what();
      self.send_done_sock.reset();
    }

    try {
      self.do_rpc_send_done(respsize);
      return ;
    } catch (const std::exception& ex) {
      LOG(ERROR) << "do_rpc_send_done failed. ex=" << ex.what();
      self.send_done_sock.reset();
      // throw;
    }
    return ;
}

  void send_done(const std::vector<const ReqSendTask*>& reqs) {
    auto& self = *this;
    if (self.send_done_addr == nullptr) {
      // channel send done, used in bladellm
      if (self.ch) {
        self.ch->send_notification(self.finished_req);
      }
    } else {
      // rpc send done, used in vllm
      self.rpc_send_done(self.finished_req);
    }
    return;
  }

  void do_send(BatchSendTask& batch) {
    auto& self = *this;
    const auto start_layer = self.stub->start_layer_;
    const auto num_layers = self.stub->num_layers_;

    self.ch->register_data(self.send_blocks, self.tpkind);
    self.wait_time_us = 0;
    for (auto i = start_layer; i < num_layers; ++i) {
      TimeWatch wait_start;
      batch.step->wait_layer_ready(i);
      self.wait_time_us += wait_start.get_elapse_us();
      // NOTE: write 可能是异步的! write 返回并不意味着数据发送了!
      self.ch->send_data(i);
    }
    fault_inject_throw();
    self.ch->flush(self.flush_out_buf);
    self.send_finish_ts = SteadyClock::now();
    self.wait_and_send_us = elapse_us(self.iter_start_ts, self.send_finish_ts);
    return ;
  }

  void refresh_dst_info() {
    auto& self = *this;
    const auto& dstid = self.stub->dstid_;
    const auto dstwid = self.stub->dstworkerid_;
    auto dstinfoopt = self.stub->naming_->get_worker_info(dstid, dstwid);
    if (!dstinfoopt) {
      if (self.dstinfo.has_value()) {
        LOG(WARNING) << "refresh_dst_info:use origdstinfo=" << self.dstinfo->to_string();
        return;
      }
      LOG(ERROR) << "refresh_dst_info:origdstinfo=none dstid=" << dstid
                 << " dstwid=" << dstwid;
      throw std::runtime_error("can't find target worker");
    }
    if (self.dstinfo.has_value() && !same_dst(*self.dstinfo, *dstinfoopt)) {
      LOG(WARNING) << "TaskContext.dst_info: invalid dst: orig_dst_info="
                  << self.dstinfo->to_string()
                  << "bad_dst_info=" << dstinfoopt->to_string();
      return;
    }
    if (self.dstinfo.has_value()) {
      self.dstinfo->addr = std::move(dstinfoopt->addr);
      self.dstinfo->other_info = std::move(dstinfoopt->other_info);
      return;
    }
    self.dstinfo = dstinfoopt;

    const auto& srcinfo = self.stub->src_info_;
    assert(self.parse_block == nullptr);
    assert(self.tpkind == TPKind::UNKNOWN);
    const int cache_shape = env_cache_shape();

    if (cache_shape == RAGGED_FLASH_CACHE_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = parse_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        if (srcinfo.token_sizes == self.dstinfo->token_sizes) {
          self.parse_block = parse_block_send_p_gt_d_dpsk;
          self.tpkind = TPKind::PEQD;
        } else {
          self.parse_block = parse_block_send_p_gt_d;
          self.tpkind = TPKind::PGTD;
        }
        return;
      }
      self.parse_block = parse_block_send_p_lt_d;
      self.tpkind = TPKind::PLTD;
      return;
    }
    if (cache_shape == FLASH_CACHE_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    if (cache_shape == QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_hybrid_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_hybrid_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    if (cache_shape == DPSK_V32_SPARSE_MLA_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_block_send_multi_tensor_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_block_send_multi_tensor_p_gt_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      self.parse_block = vllm_parse_block_send_multi_tensor_p_lt_d;
      self.tpkind = TPKind::PEQD;
      return;
    }
    if (cache_shape == FLASHINFER_CACHE_SHAPE) {
      if (srcinfo.tp_size == self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_flashinfer_block_send_p_eq_d;
        self.tpkind = TPKind::PEQD;
        return;
      }
      if (srcinfo.tp_size > self.dstinfo->tp_size) {
        self.parse_block = vllm_parse_flashinfer_block_send_p_gt_d;
        self.tpkind = TPKind::PGTD;
        return;
      }
    }
    throw std::runtime_error("unsupported cache_shape");
  }

  void try_create_channel() {
    auto& self = *this;
    if (self.ch && self.ch->is_active()) {
      assert(self.dstinfo.has_value());
      return ;
    }
    self.refresh_dst_info();
    assert(self.dstinfo.has_value());
    self.ch = self.stub->channel_factory_->create(*self.dstinfo);
    LOG(INFO) << "create channel. dst=" << self.dstinfo->to_string()
              << ";ch=" << self.ch.get();
    return ;
  }
};

void KvSendStub::send_batch(BatchSendTask batch) noexcept {
  assert(!batch.tasks.empty());
  auto& ctx = *this->taskctx_;
  ctx.do_task(batch);
  ctx.clear();
  return;
}

KvSendStub::~KvSendStub() {}

KvSendStub::KvSendStub(InstanceId dstid, WorkerId dstworkerid,
             const WorkerInfo &src_info,
             uint32_t start_layer,
             uint32_t num_layers,
             std::unique_ptr<IChannelFactory> channel_factory,
             std::shared_ptr<INamingWorkerClient> naming) :
      ISendStub(std::move(dstid), dstworkerid),
      src_info_(src_info),
      naming_(std::move(naming)),
      start_layer_(start_layer),
      num_layers_(num_layers),
      channel_factory_(std::move(channel_factory)),
      taskctx_(std::make_unique<TaskContext>(this)) {};

SendStub KvSendStubFactory::create_stub(const InstanceId& dst_inst_name,
                                        WorkerId dst_worker_id,
                                        uint32_t start_layer,
                                        uint32_t num_layers,
                                        std::optional<TransferProtocol> proto_opt,
                                        const std::optional<std::string> &dst_info) {
  auto channel_factory = std::make_unique<ChannelFactory>(ctx_, proto_opt);
  std::shared_ptr<INamingWorkerClient> naming_worker;
  if (dst_info) {
    naming_worker = std::make_shared<FakeNamingWorkerClient>(dst_info.value());
  } else {
    naming_worker = naming_worker_;
  }
  return std::make_unique<KvSendStub>(dst_inst_name, dst_worker_id, ctx_->worker_info(),
                                      start_layer, num_layers,
                                      std::move(channel_factory),
                                      std::move(naming_worker));
}

}
