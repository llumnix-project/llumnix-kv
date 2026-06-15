#include "cache_transfer_spec.h"
#include "attn_layout.h"
#include "envcfg.h"
#include "thrid_party/logging.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace blade_llm {

// ============================================================
// Generic IpcBlock generation algorithm
// ============================================================

// Generate IpcBlocks for a single K-style block copy path (also reused by V).
static void generate_k_block(
    const AttnSpec& comp,
    size_t base_off_src, size_t base_off_dst,
    size_t global_off_src, size_t global_off_dst,
    size_t head_stride_src, size_t head_stride_dst,
    uint32_t src_ntpb, uint32_t dst_ntpb,
    const std::vector<uint32_t>& src_blocks,
    const std::vector<uint32_t>& dst_blocks,
    uint32_t wrote_tokens,
    uint32_t left_tokens,
    std::vector<IpcBlock>& out)
{
  uint32_t done = 0;

  while (done < left_tokens) {
    const uint32_t cur = wrote_tokens + done;
    const uint32_t src_bidx = cur / src_ntpb;
    const uint32_t src_tidx = cur % src_ntpb;
    const uint32_t dst_bidx = cur / dst_ntpb;
    const uint32_t dst_tidx = cur % dst_ntpb;

    assert(src_bidx < src_blocks.size() && dst_bidx < dst_blocks.size());

    const uint32_t tokens = std::min(
      {src_ntpb - src_tidx, dst_ntpb - dst_tidx, left_tokens - done}
    );

    for (uint32_t r = 0; r < comp.head_num; ++r) {
      const size_t src_roff = r * head_stride_src;
      const size_t dst_roff = r * head_stride_dst;

      for (uint32_t i = 0; i < tokens; ++i) {
        const size_t src_off =
            src_blocks[src_bidx] * comp.src_block_stride
            + (src_tidx + i) * comp.src_token_stride
            + base_off_src + global_off_src + src_roff;
        const size_t dst_off =
            dst_blocks[dst_bidx] * comp.dst_block_stride
            + (dst_tidx + i) * comp.dst_token_stride
            + base_off_dst + global_off_dst + dst_roff;
        out.emplace_back(src_off, dst_off, comp.bytes_per_token);
      }
    }
    done += tokens;
  }
}

// Pad the unfilled tail of the last decode-side block using block 0 data.
// Mirrors generate_k_block but with diverging src/dst cursors: the src cursor
// is anchored to the request's block 0 (src_blocks[0]) starting at token 0,
// while the dst cursor starts at the first empty slot of the last dst block.
static void generate_k_block_fill(
    const AttnSpec& comp,
    size_t base_off_src, size_t base_off_dst,
    size_t global_off_src, size_t global_off_dst,
    size_t head_stride_src, size_t head_stride_dst,
    uint32_t src_ntpb, uint32_t dst_ntpb,
    const std::vector<uint32_t>& src_blocks,
    const std::vector<uint32_t>& dst_blocks,
    uint32_t total_tokens,
    std::vector<IpcBlock>& out)
{
  if (total_tokens == 0) return;
  const uint32_t filled = total_tokens % dst_ntpb;
  if (filled == 0) return;  // last dst block already full
  const uint32_t to_fill = dst_ntpb - filled;

  uint32_t src_pos = 0;
  uint32_t dst_pos = total_tokens;
  uint32_t done = 0;
  while (done < to_fill) {
    const uint32_t src_bidx = src_pos / src_ntpb;
    const uint32_t src_tidx = src_pos % src_ntpb;
    const uint32_t dst_bidx = dst_pos / dst_ntpb;
    const uint32_t dst_tidx = dst_pos % dst_ntpb;
    assert(src_bidx == 0);  // fill from block 0
    assert(dst_bidx < dst_blocks.size());

    const uint32_t tokens = std::min(
      {src_ntpb - src_tidx, dst_ntpb - dst_tidx, to_fill - done}
    );

    for (uint32_t r = 0; r < comp.head_num; ++r) {
      const size_t src_roff = r * head_stride_src;
      const size_t dst_roff = r * head_stride_dst;
      for (uint32_t i = 0; i < tokens; ++i) {
        const size_t src_off =
            src_blocks[0] * comp.src_block_stride
            + (src_tidx + i) * comp.src_token_stride
            + base_off_src + global_off_src + src_roff;
        const size_t dst_off =
            dst_blocks[dst_bidx] * comp.dst_block_stride
            + (dst_tidx + i) * comp.dst_token_stride
            + base_off_dst + global_off_dst + dst_roff;
        out.emplace_back(src_off, dst_off, comp.bytes_per_token);
      }
    }
    src_pos += tokens;
    dst_pos += tokens;
    done += tokens;
  }
}

// Fill the last dst block's tail for an AttnSpec: K stream first, then V.
static void generate_attn_fill(
    const AttnSpec& comp,
    uint32_t src_ntpb, uint32_t dst_ntpb,
    const std::vector<uint32_t>& src_blocks,
    const std::vector<uint32_t>& dst_blocks,
    uint32_t total_tokens,
    std::vector<IpcBlock>& out)
{
  generate_k_block_fill(
      comp,
      comp.src_base_offset, comp.dst_base_offset,
      comp.src_global_offset, comp.dst_global_offset,
      comp.src_head_stride, comp.dst_head_stride,
      src_ntpb, dst_ntpb,
      src_blocks, dst_blocks,
      total_tokens, out);

  if (comp.v) {
    generate_k_block_fill(
        comp,
        comp.v->src_base_offset, comp.v->dst_base_offset,
        comp.v->src_global_offset, comp.v->dst_global_offset,
        comp.v->src_head_stride, comp.v->dst_head_stride,
        src_ntpb, dst_ntpb,
        src_blocks, dst_blocks,
        total_tokens, out);
  }
}

// Generate IpcBlocks for an AttnSpec: K stream first, then V stream if present.
static void generate_attn(
    const AttnSpec& comp,
    uint32_t src_ntpb, uint32_t dst_ntpb,
    const std::vector<uint32_t>& src_blocks,
    const std::vector<uint32_t>& dst_blocks,
    uint32_t wrote_tokens,
    uint32_t left_tokens,
    std::vector<IpcBlock>& out)
{
  generate_k_block(
      comp,
      comp.src_base_offset, comp.dst_base_offset,
      comp.src_global_offset, comp.dst_global_offset,
      comp.src_head_stride, comp.dst_head_stride,
      src_ntpb, dst_ntpb,
      src_blocks, dst_blocks,
      wrote_tokens, left_tokens, out);

  if (comp.v) {
    // V part reuses the same block generation routine with V-specific offsets.
    generate_k_block(
        comp,
        comp.v->src_base_offset, comp.v->dst_base_offset,
        comp.v->src_global_offset, comp.v->dst_global_offset,
        comp.v->src_head_stride, comp.v->dst_head_stride,
        src_ntpb, dst_ntpb,
        src_blocks, dst_blocks,
        wrote_tokens, left_tokens, out);
  }
}

static void generate_block_aligned(
    const AttnComponent& attn,
    const std::vector<uint32_t>& src_blocks,
    const std::vector<uint32_t>& dst_blocks,
    uint32_t wrote_tokens,
    uint32_t left_tokens,
    bool reach_last_token,
    std::vector<IpcBlock>& out)
{
  const auto& comp = attn.spec;
  const uint32_t ntpb = attn.src_ntpb;
  assert(attn.src_ntpb == attn.dst_ntpb);

  uint32_t done = 0;
  while (done < left_tokens) {
    const uint32_t cur = wrote_tokens + done;
    const uint32_t blk_idx = cur / ntpb;
    const uint32_t token_idx = cur % ntpb;
    assert(blk_idx < src_blocks.size() && blk_idx < dst_blocks.size());

    const size_t p_blk_off = src_blocks[blk_idx] * comp.src_block_stride;
    const size_t d_blk_off = dst_blocks[blk_idx] * comp.dst_block_stride;
    const uint32_t tokens = std::min(ntpb - token_idx, left_tokens - done);

    size_t length = 0;
    if (tokens != ntpb) {
      if (reach_last_token) {
        length = comp.src_block_stride;
      } else if (token_idx == 0) {
        length = 0;
      } else {
        if (token_idx + tokens == ntpb) {
          length = comp.src_block_stride;
        } else {
          assert(token_idx + tokens < ntpb);
          length = 0;
        }
      }
    } else {
      length = comp.src_block_stride;
    }
    if (length > 0) {
      out.emplace_back(p_blk_off, d_blk_off, length);
    }
    done += tokens;
  }
}

static void generate_gdn(
    const GdnSpec& gdn,
    const BlockIds& src_blocks,
    const BlockIds& dst_blocks,
    std::vector<IpcBlock>& out)
{
  for (uint32_t g = 0; g < gdn.num_groups; ++g) {
    assert(g < src_blocks.size() && g < dst_blocks.size());
    const auto& src_grp = src_blocks[g];
    const auto& dst_grp = dst_blocks[g];
    assert(src_grp.size() == dst_grp.size());
    for (size_t i = 0; i < src_grp.size(); ++i) {
      const size_t p_blk_off = src_grp[i] * gdn.src_block_stride;
      const size_t d_blk_off = dst_grp[i] * gdn.dst_block_stride;
      for (const auto& seg : gdn.segments) {
        out.emplace_back(
          p_blk_off + seg.src_offset_in_block,
          d_blk_off + seg.dst_offset_in_block,
          seg.length
        );
      }
    }
  }
}

void generate_ipc_blocks(
    const CacheTransferSpec& spec,
    const BlockIds& src_blocks,
    const BlockIds& dst_blocks,
    uint32_t wrote_tokens,
    uint32_t left_tokens,
    bool reach_last_token,
    std::vector<IpcBlock>& out)
{
  if (!spec.valid) return;

  for (const auto& comp : spec.components) {
    if (auto* attn = std::get_if<AttnComponent>(&comp)) {
      assert(attn->block_group < src_blocks.size());
      assert(attn->block_group < dst_blocks.size());
      const auto& src_blks = src_blocks[attn->block_group];
      const auto& dst_blks = dst_blocks[attn->block_group];
      if (attn->block_aligned) {
        generate_block_aligned(*attn, src_blks, dst_blks,
                               wrote_tokens, left_tokens, reach_last_token, out);
      } else {
        generate_attn(attn->spec, attn->src_ntpb, attn->dst_ntpb,
                      src_blks, dst_blks,
                      wrote_tokens, left_tokens, out);
        // On request completion, fill the tail of the last unfilled decode block with block 0.
        if (attn->fill_last_block && reach_last_token) {
          generate_attn_fill(attn->spec, attn->src_ntpb, attn->dst_ntpb,
                             src_blks, dst_blks,
                             wrote_tokens + left_tokens, out);
        }
      }
    } else if (auto* gdn = std::get_if<GdnComponent>(&comp)) {
      if (reach_last_token) {
        generate_gdn(gdn->spec, src_blocks, dst_blocks, out);
      }
    }
  }
}

void generate_all_ipc_blocks(
    const TransferPlan& plan,
    const ReqSendTask* task,
    std::vector<std::vector<IpcBlock>>& send_blocks)
{
  send_blocks.resize(plan.specs.size());
  for (size_t i = 0; i < plan.specs.size(); ++i) {
    generate_ipc_blocks(
      plan.specs[i],
      task->src_blocks(), task->dst_blocks(),
      task->seen_tokens, task->new_tokens,
      task->reach_last_token,
      send_blocks[i]
    );
  }
}

// ============================================================
// Factory: build TransferPlan from cache shape + worker info
// ============================================================

// ---- Layout registry ----

const AttnLayoutDesc& get_attn_layout(int cache_shape) {
  using D = CacheDim;
  static const std::unordered_map<int, AttnLayoutDesc> registry = {
    {RAGGED_FLASH_CACHE_SHAPE, {
      /*dim_order=*/ {D::BLOCK, D::TOKEN, D::KV, D::HEAD, D::HEAD_DIM},
      /*num_tensors=*/ 1,
      /*block_aligned_peqd=*/ false,
      /*v_head_invariant=*/ false,
      /*use_valid_ranks=*/ false,
      /*valid_ranks_affects_group_n=*/ false,
      /*rank0_only_when_same_sizes=*/ true,
      /*force_peqd_tpkind=*/ false,
    }},
    {FLASH_CACHE_SHAPE, {
      /*dim_order=*/ {D::KV, D::BLOCK, D::TOKEN, D::HEAD, D::HEAD_DIM},
      /*num_tensors=*/ 1,
      /*block_aligned_peqd=*/ false,
      /*v_head_invariant=*/ false,
      /*use_valid_ranks=*/ true,
      /*valid_ranks_affects_group_n=*/ false,
      /*rank0_only_when_same_sizes=*/ false,
      /*force_peqd_tpkind=*/ false,
    }},
    {QWEN3_NEXT_FLASH_CACHE_SHAPE, {
      /*dim_order=*/ {D::BLOCK, D::KV, D::TOKEN, D::HEAD, D::HEAD_DIM},
      /*num_tensors=*/ 1,
      /*block_aligned_peqd=*/ true,
      /*v_head_invariant=*/ false,
      /*use_valid_ranks=*/ true,
      /*valid_ranks_affects_group_n=*/ true,
      /*rank0_only_when_same_sizes=*/ false,
      /*force_peqd_tpkind=*/ false,
    }},
    {DPSK_V32_SPARSE_MLA_SHAPE, {
      /*dim_order=*/ {D::BLOCK, D::TOKEN, D::KV, D::HEAD, D::HEAD_DIM},
      /*num_tensors=*/ 0,  // derived from WorkerInfo::token_sizes.size()
      /*block_aligned_peqd=*/ false,
      /*v_head_invariant=*/ false,
      /*use_valid_ranks=*/ false,
      /*valid_ranks_affects_group_n=*/ false,
      /*rank0_only_when_same_sizes=*/ true,
      /*force_peqd_tpkind=*/ true,
    }},
    {FLASHINFER_CACHE_SHAPE, {
      /*dim_order=*/ {D::BLOCK, D::KV, D::HEAD, D::TOKEN, D::HEAD_DIM},
      /*num_tensors=*/ 1,
      /*block_aligned_peqd=*/ false,
      /*v_head_invariant=*/ true,
      /*use_valid_ranks=*/ true,
      /*valid_ranks_affects_group_n=*/ true,
      /*rank0_only_when_same_sizes=*/ false,
      /*force_peqd_tpkind=*/ false,
    }},
    // TODO(llx)
    // {TURBOQUANT_CACHE_SHAPE, {
    //   /*dim_order=*/ {D::KV, D::BLOCK, D::TOKEN, D::HEAD, D::HEAD_DIM},
    //   /*num_tensors=*/ 1,
    //   /*block_aligned_peqd=*/ false,
    //   /*v_head_invariant=*/ false,
    //   /*use_valid_ranks=*/ true,
    //   /*valid_ranks_affects_group_n=*/ false,
    //   /*rank0_only_when_same_sizes=*/ false,
    //   /*force_peqd_tpkind=*/ false,
    // }},
  };
  auto it = registry.find(cache_shape);
  if (it == registry.end()) {
    throw std::runtime_error("unsupported cache_shape: " + std::to_string(cache_shape));
  }
  return it->second;
}

// ---- GDN builders (unchanged, hybrid-model specific) ----

static GdnSpec build_gdn_peqd(
    size_t block_size, uint32_t ntpb, size_t token_size,
    const WorkerInfo& w) {
  GdnSpec gdn;
  gdn.num_groups = w.num_gdn_layers;
  gdn.src_block_stride = block_size;
  gdn.dst_block_stride = block_size;
  gdn.segments.push_back({0, 0, ntpb * token_size});
  return gdn;
}

static GdnSpec build_gdn_pgtd(
    size_t p_block_size, size_t d_block_size,
    uint32_t gdn_group_n, uint32_t gdn_group_off,
    const WorkerInfo& w)
{
  RTASSERT(!w.conv_state_shape.empty() && w.conv_state_shape.size() >= 3);
  RTASSERT(!w.ssm_state_shape.empty() && w.ssm_state_shape.size() >= 4);
  RTASSERT(!w.gdn_conv_channel_dims.empty());

  const auto& conv_state_shape = w.conv_state_shape;
  const auto& ssm_state_shape = w.ssm_state_shape;
  uint32_t gdn_conv_elem_size = w.gdn_conv_elem_size;
  uint32_t gdn_ssm_elem_size = w.gdn_ssm_elem_size;

  const auto& conv_channel_dims = w.gdn_conv_channel_dims;

  const auto p_conv_dim_size = conv_state_shape[2] * gdn_conv_elem_size;
  const auto p_conv_block_size = conv_state_shape[1] * p_conv_dim_size;
  const auto p_ssm_block_size =
      ssm_state_shape[1] * ssm_state_shape[2] * ssm_state_shape[3] * gdn_ssm_elem_size;
  const auto d_conv_dim_size = gdn_group_n * p_conv_dim_size;

  GdnSpec gdn;
  gdn.num_groups = w.num_gdn_layers;
  gdn.src_block_stride = p_block_size;
  gdn.dst_block_stride = d_block_size;

  for (size_t conv_dim = 0; conv_dim < conv_state_shape[1]; ++conv_dim) {
    size_t conv_dim_inner_offset = 0;
    for (size_t i = 0; i < conv_channel_dims.size(); ++i) {
      const auto conv_step_length = conv_channel_dims[i] * gdn_conv_elem_size;
      size_t p_conv_off = conv_dim * p_conv_dim_size + conv_dim_inner_offset;
      size_t d_conv_off = conv_dim * d_conv_dim_size
                  + gdn_group_n * conv_dim_inner_offset
                  + gdn_group_off * conv_step_length;
      gdn.segments.push_back({p_conv_off, d_conv_off, conv_step_length});
      conv_dim_inner_offset += conv_step_length;
    }
    assert(conv_dim_inner_offset == p_conv_dim_size);
  }

  size_t p_ssm_off = p_conv_block_size;
  size_t d_ssm_off = gdn_group_n * p_conv_block_size + gdn_group_off * p_ssm_block_size;
  gdn.segments.push_back({p_ssm_off, d_ssm_off, p_ssm_block_size});

  return gdn;
}

// ============================================================
// Unified attn spec builder (layout-driven)
// ============================================================

// Build an AttnComponent for one cache tensor from the layout descriptor.
// group_off=0 for P==D.
static AttnComponent build_attn_spec_from_layout(
    const AttnLayoutDesc& layout,
    size_t p_block_size, size_t p_token_size,
    size_t d_block_size, size_t d_token_size,
    uint32_t group_off,
    uint32_t layer_num_blocks_src,
    uint32_t layer_num_blocks_dst,
    uint32_t head_num)
{
  const size_t p_k_token = p_token_size / 2;
  const size_t d_k_token = d_token_size / 2;
  const uint32_t src_ntpb = p_block_size / p_token_size;
  const uint32_t dst_ntpb = d_block_size / d_token_size;
  const size_t p_k_block = src_ntpb * p_k_token;
  const size_t d_k_block = dst_ntpb * d_k_token;

  const bool is_peqd = (group_off == 0 &&
                        p_block_size == d_block_size &&
                        p_token_size == d_token_size);

  const bool head_outer = layout.is_head_outer_to_token();
  const bool kv_outermost = layout.is_kv_outermost();
  const bool kv_between = layout.is_kv_between_block_and_token();
  const bool kv_inner = layout.is_kv_inner_to_token();

  // For P==D with block_aligned layout, use a single merged component
  if (is_peqd && layout.block_aligned_peqd) {
    AttnComponent result;
    result.src_ntpb = src_ntpb;
    result.dst_ntpb = dst_ntpb;
    result.block_aligned = true;
    AttnSpec attn{};
    attn.src_block_stride = p_block_size;
    attn.dst_block_stride = d_block_size;
    attn.src_token_stride = p_token_size;
    attn.dst_token_stride = d_token_size;
    attn.bytes_per_token = p_token_size;
    result.spec = attn;
    return result;
  }

  // For P==D without head-outer or kv-outermost, merge KV into single component
  if (is_peqd && !head_outer && !kv_outermost) {
    AttnComponent result;
    result.src_ntpb = src_ntpb;
    result.dst_ntpb = dst_ntpb;
    AttnSpec attn{};
    attn.src_block_stride = p_block_size;
    attn.dst_block_stride = d_block_size;
    attn.src_token_stride = p_token_size;
    attn.dst_token_stride = d_token_size;
    attn.bytes_per_token = p_token_size;
    result.spec = attn;
    return result;
  }

  // Compute effective bytes_per_token (may be halved by bf16->fp8, only for P>D)
  auto effective_p_k_token = p_k_token;
  if (!is_peqd && env_bf162fp8_conversion()) {
    effective_p_k_token = p_k_token / 2;
  }

  AttnComponent result;
  result.src_ntpb = src_ntpb;
  result.dst_ntpb = dst_ntpb;

  if (head_outer) {
    // HEAD is outer to TOKEN: need repeat per head (FLASHINFER-style)
    assert(head_num > 0);
    const size_t p_head_size = p_k_block / head_num;
    const size_t per_token_per_head = p_head_size / src_ntpb;
    const size_t p_kv_stride = p_k_block;
    const size_t d_kv_stride = d_k_block;

    AttnSpec& spec = result.spec;
    spec.src_block_stride = p_block_size;
    spec.dst_block_stride = d_block_size;
    spec.src_token_stride = per_token_per_head;
    spec.dst_token_stride = per_token_per_head;
    spec.src_base_offset = 0;
    spec.dst_base_offset = group_off * p_kv_stride;
    spec.bytes_per_token = per_token_per_head;
    spec.head_num = head_num;
    spec.src_head_stride = p_head_size;
    spec.dst_head_stride = p_head_size;

    AttnSpec::VSpec vs{};
    vs.src_base_offset = p_kv_stride;
    vs.src_global_offset = 0;
    vs.dst_global_offset = 0;
    if (layout.v_head_invariant) {
      vs.dst_base_offset = d_kv_stride;
      vs.src_head_stride = 0;
      vs.dst_head_stride = 0;
    } else {
      vs.dst_base_offset = group_off * p_kv_stride + d_kv_stride;
      vs.src_head_stride = p_head_size;
      vs.dst_head_stride = p_head_size;
    }
    spec.v = vs;

  } else if (kv_outermost) {
    // KV is outermost: K and V in separate layer regions (FLASH-style)
    const size_t src_layer_size = p_k_block * layer_num_blocks_src;
    const size_t dst_layer_size = d_k_block * layer_num_blocks_dst;

    AttnSpec& spec = result.spec;
    spec.src_block_stride = p_k_block;
    spec.dst_block_stride = d_k_block;
    spec.src_token_stride = p_k_token;
    spec.dst_token_stride = d_k_token;
    spec.src_base_offset = 0;
    spec.dst_base_offset = group_off * effective_p_k_token;
    spec.src_global_offset = 0;
    spec.dst_global_offset = 0;
    spec.bytes_per_token = effective_p_k_token;

    AttnSpec::VSpec vs{};
    vs.src_base_offset = spec.src_base_offset;
    vs.dst_base_offset = spec.dst_base_offset;
    vs.src_global_offset = src_layer_size;
    vs.dst_global_offset = dst_layer_size;
    spec.v = vs;

  } else if (kv_between) {
    // KV between BLOCK and TOKEN: K/V in separate block halves (QWEN3_NEXT-style)
    AttnSpec& spec = result.spec;
    spec.src_block_stride = p_block_size;
    spec.dst_block_stride = d_block_size;
    spec.src_token_stride = p_k_token;
    spec.dst_token_stride = d_k_token;
    spec.src_base_offset = 0;
    spec.dst_base_offset = group_off * effective_p_k_token;
    spec.bytes_per_token = effective_p_k_token;

    AttnSpec::VSpec vs{};
    vs.src_base_offset = p_k_block;
    vs.dst_base_offset = group_off * effective_p_k_token + d_k_block;
    spec.v = vs;

  } else {
    // KV inner to TOKEN: K/V interleaved per token (RAGGED_FLASH-style)
    assert(kv_inner);
    AttnSpec& spec = result.spec;
    spec.src_block_stride = p_block_size;
    spec.dst_block_stride = d_block_size;
    spec.src_token_stride = p_token_size;
    spec.dst_token_stride = d_token_size;
    spec.src_base_offset = 0;
    spec.dst_base_offset = group_off * effective_p_k_token;
    spec.bytes_per_token = effective_p_k_token;

    AttnSpec::VSpec vs{};
    vs.src_base_offset = p_k_token;
    vs.dst_base_offset = group_off * effective_p_k_token + d_k_token;
    spec.v = vs;
  }

  return result;
}

// ============================================================
// Generic build_transfer_plan (no per-shape branches)
// ============================================================

TransferPlan build_transfer_plan(
    int cache_shape,
    const WorkerInfo& src,
    const WorkerInfo& dst,
    const std::bitset<MAX_TP_SIZE>& valid_ranks,
    uint32_t kvt_tp_size,
    uint32_t kvt_tp_rank)
{
  const auto& layout = get_attn_layout(cache_shape);
  TransferPlan plan;

  const uint32_t num_tensors = (layout.num_tensors > 0)
      ? layout.num_tensors
      : static_cast<uint32_t>(src.token_sizes.size());

  const uint32_t num_gdn_layers = src.num_gdn_layers;
  const bool has_gdn = (num_gdn_layers > 0);
  const uint32_t p_indexer_ntpb = src.indexer_blk_ntpb;
  const uint32_t d_indexer_ntpb = dst.indexer_blk_ntpb;
  const bool has_indexer = (p_indexer_ntpb > 0);

  uint32_t head_num = 0;
  if (layout.is_head_outer_to_token()) {
    if (src.num_kv_heads <= 0) {
      throw std::runtime_error(
          "cache_transfer_spec: num_kv_heads must be set on WorkerInfo for this layout");
    }
    head_num = static_cast<uint32_t>(src.num_kv_heads);
  }

  // P-side TP value used for selecting between P==D and P>D layouts.
  // For shapes carrying GDN, this is engine_tp_size (GDN grouping is
  // engine-tp-based); otherwise the effective kvt_tp_size after valid_ranks
  // filtering. Mirrors selection_p_tp() in tx_stub.cpp.
  const uint32_t p_tp_for_selection = has_gdn ? src.engine_tp_size : kvt_tp_size;
  const uint32_t d_tp = dst.engine_tp_size;

  // ---- P == D ----
  if (p_tp_for_selection == d_tp) {
    plan.tpkind = TPKind::PEQD;
    assert(src.token_sizes == dst.token_sizes);
    assert(src.block_sizes == dst.block_sizes);

    for (uint32_t t = 0; t < num_tensors; ++t) {
      CacheTransferSpec spec;

      auto fullattn = build_attn_spec_from_layout(
        layout,
        src.block_sizes[t], src.token_sizes[t],
        dst.block_sizes[t], dst.token_sizes[t],
        /*group_off=*/ 0,
        src.layer_num_blocks, dst.layer_num_blocks,
        head_num);

      if (has_gdn) {
        const auto ntpb = src.block_sizes[t] / src.token_sizes[t];

        // GDN first
        GdnComponent gdn_comp;
        gdn_comp.spec = build_gdn_peqd(src.block_sizes[t], ntpb, src.token_sizes[t], src);
        spec.components.push_back(std::move(gdn_comp));

        // Indexer second (if present)
        if (has_indexer) {
          assert(p_indexer_ntpb == d_indexer_ntpb);
          AttnComponent idx;
          idx.block_group = num_gdn_layers;
          idx.src_ntpb = p_indexer_ntpb;
          idx.dst_ntpb = d_indexer_ntpb;
          idx.block_aligned = true;
          AttnSpec attn{};
          attn.src_block_stride = src.block_sizes[t];
          attn.dst_block_stride = src.block_sizes[t];
          attn.bytes_per_token = 0;
          idx.spec = attn;
          spec.components.push_back(std::move(idx));
        }

        // FullAttn last
        fullattn.block_group = num_gdn_layers + (has_indexer ? 1 : 0);
      } else if (num_tensors > 1) {
        fullattn.block_group = t;
      }

      spec.components.push_back(std::move(fullattn));
      plan.specs.push_back(std::move(spec));
    }
    return plan;
  }

  // ---- P > D ----
  if (p_tp_for_selection > d_tp) {
    // rank0_only_when_same_sizes: P>D but identical token_sizes -> only rank 0 sends as P==D
    if (layout.rank0_only_when_same_sizes && src.token_sizes == dst.token_sizes) {
      plan.tpkind = TPKind::PEQD;
      assert(src.block_sizes == dst.block_sizes);
      if (src.worker_tp_rank == 0) {
        for (uint32_t t = 0; t < num_tensors; ++t) {
          CacheTransferSpec spec;
          auto fullattn = build_attn_spec_from_layout(
            layout,
            src.block_sizes[t], src.token_sizes[t],
            dst.block_sizes[t], dst.token_sizes[t],
            /*group_off=*/ 0,
            src.layer_num_blocks, dst.layer_num_blocks,
            head_num);
          if (num_tensors > 1) {
            fullattn.block_group = t;
          }
          spec.components.push_back(std::move(fullattn));
          plan.specs.push_back(std::move(spec));
        }
      }
      return plan;
    }

    plan.tpkind = layout.force_peqd_tpkind ? TPKind::PEQD : TPKind::PGTD;

    // Compute attn group_off (may use valid_ranks filtering)
    uint32_t group_off = 0;
    bool sends_attn = true;
    if (layout.use_valid_ranks) {
      if (!valid_ranks[src.worker_tp_rank]) {
        sends_attn = false;
      }
      uint32_t const effective_rank = kvt_tp_rank;
      uint32_t group_n;
      if (layout.valid_ranks_affects_group_n) {
        group_n = valid_ranks.count() / d_tp;
      } else {
        group_n = kvt_tp_size / d_tp;
      }
      group_off = effective_rank % group_n;
    } else {
      const uint32_t group_n = kvt_tp_size / d_tp;
      group_off = src.worker_tp_rank % group_n;
    }

    // If this rank is filtered out, return inactive spec
    if (!sends_attn && !has_gdn && !has_indexer) {
      CacheTransferSpec spec;
      spec.valid = false;
      plan.specs.push_back(std::move(spec));
      return plan;
    }

    for (uint32_t t = 0; t < num_tensors; ++t) {
      CacheTransferSpec spec;
      const size_t p_block_size = src.block_sizes[t];
      const size_t p_token_size = src.token_sizes[t];
      const size_t d_block_size = dst.block_sizes[t];
      const size_t d_token_size = dst.token_sizes[t];

      // Attach GDN (uses origin_p_tp_size for group computation, independent of valid_ranks)
      if (has_gdn) {
        RTASSERT(src.engine_tp_size > 0);
        const uint32_t gdn_group_n = src.engine_tp_size / d_tp;
        const uint32_t gdn_group_off = src.worker_tp_rank % gdn_group_n;

        uint32_t attn_group_offset = num_gdn_layers;

        // GDN first
        GdnComponent gdn_comp;
        gdn_comp.spec = build_gdn_pgtd(
            p_block_size, d_block_size, gdn_group_n, gdn_group_off, src);
        spec.components.push_back(std::move(gdn_comp));

        // Indexer second (only rank 0 sends)
        if (has_indexer) {
          assert((p_indexer_ntpb == 0 && d_indexer_ntpb == 0) ||
                 (p_indexer_ntpb > 0 && d_indexer_ntpb > 0));
          if (src.worker_tp_rank == 0) {
            const uint32_t indexer_token_size = src.hybrid_indexer_token_size;
            RTASSERT(indexer_token_size > 0);
            AttnComponent idx;
            idx.block_group = attn_group_offset;
            idx.src_ntpb = p_indexer_ntpb;
            idx.dst_ntpb = d_indexer_ntpb;
            AttnSpec attn{};
            attn.src_block_stride = p_block_size;
            attn.dst_block_stride = d_block_size;
            attn.src_token_stride = static_cast<size_t>(indexer_token_size);
            attn.dst_token_stride = static_cast<size_t>(indexer_token_size);
            attn.bytes_per_token = static_cast<size_t>(indexer_token_size);
            idx.spec = attn;
            spec.components.push_back(std::move(idx));
          }
          attn_group_offset += 1;
        }

        // FullAttn last
        if (sends_attn) {
          auto fullattn = build_attn_spec_from_layout(
            layout,
            p_block_size, p_token_size,
            d_block_size, d_token_size,
            group_off,
            src.layer_num_blocks, dst.layer_num_blocks,
            head_num);
          fullattn.block_group = attn_group_offset;
          // qwen3_next (has_gdn) P>D: on completion, fill the last
          // decode attn block tail empty slots (consistent with legacy parse_block path).
          fullattn.fill_last_block = env_pad_last_attn_block();
          spec.components.push_back(std::move(fullattn));
        }
      } else {
        // No GDN: just the fullattn component
        if (sends_attn) {
          auto fullattn = build_attn_spec_from_layout(
            layout,
            p_block_size, p_token_size,
            d_block_size, d_token_size,
            group_off,
            src.layer_num_blocks, dst.layer_num_blocks,
            head_num);
          if (num_tensors > 1) {
            fullattn.block_group = t;
          }
          spec.components.push_back(std::move(fullattn));
        }
      }

      plan.specs.push_back(std::move(spec));
    }
    return plan;
  }

  throw std::runtime_error("cache_shape " + std::to_string(cache_shape) + ": P TP size < D TP size not supported");
}

}  // namespace blade_llm
