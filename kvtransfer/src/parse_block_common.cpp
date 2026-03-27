#include "parse_block_common.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

void do_parse_block_send_p_eq_d(
  size_t block_size,
  size_t token_size,
  const ReqSendTask *task,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  // ntpb: number tokens per block
  uint32_t src_ntpb = block_size / token_size;
  uint32_t dst_ntpb = src_ntpb;
  assert(src_ntpb * token_size == block_size);

  auto wrote_tokens = task->seen_tokens;
  auto left_tokens = task->new_tokens;
  const auto &src_blocks = task->src_blocks()[0];
  const auto &dst_blocks = task->dst_blocks()[0];
  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + left_tokens / src_ntpb + 3);

  while (left_tokens > 0) {
    auto src_block_idx = wrote_tokens / src_ntpb;
    auto src_token_idx = wrote_tokens % src_ntpb;
    assert(src_block_idx < src_blocks.size());
    size_t src_offset = src_blocks.at(src_block_idx) * block_size
        + src_token_idx * token_size;
    auto tokens = std::min(src_ntpb - src_token_idx, left_tokens);
    auto dst_block_idx = wrote_tokens / dst_ntpb;
    auto dst_token_idx = wrote_tokens % dst_ntpb;
    assert(dst_block_idx < dst_blocks.size());
    size_t dst_offset = dst_blocks.at(dst_block_idx) * block_size
        + dst_token_idx * token_size;
    size_t length = tokens * token_size;
    per_cache_send_blocks.emplace_back(src_offset, dst_offset, length);
    wrote_tokens += tokens;
    left_tokens -= tokens;
  }
}

void parse_block_send_gt(
  size_t p_token_size,
  size_t p_k_size,
  size_t d_token_size,
  uint32_t ntpb,
  size_t group_off,
  const std::vector<uint32_t>& p_blocks,
  const std::vector<uint32_t>& d_blocks,
  uint32_t wrote_tokens,
  uint32_t left_tokens,
  std::vector<IpcBlock> &per_cache_send_blocks
) {
  assert(p_k_size == p_token_size || p_k_size * 2 == p_token_size);
  const size_t p_block_size = p_token_size * ntpb;
  const size_t d_block_size = d_token_size * ntpb;

  auto cast2fp8 = env_bf162fp8_conversion();
  if (cast2fp8) {
    p_k_size = p_k_size / 2;
  }

  per_cache_send_blocks.reserve(per_cache_send_blocks.size() + left_tokens);
  while (left_tokens > 0) {
    const uint32_t block_idx = wrote_tokens / ntpb;
    assert(block_idx < p_blocks.size() && block_idx < d_blocks.size());
    const uint32_t token_idx_base = wrote_tokens % ntpb;
    const size_t p_blk_off = p_blocks.at(block_idx) * p_block_size;
    const size_t d_blk_off = d_blocks.at(block_idx) * d_block_size;
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
}

}  // namespace blade_llm
