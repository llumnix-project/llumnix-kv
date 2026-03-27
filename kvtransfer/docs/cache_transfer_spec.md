# CacheTransferSpec Refactoring Documentation

## 1. Background

`tx_stub.cpp` originally contained multiple hand-written static `parse_block_send_*` functions. Each targeted a specific cache shape and P/D TP size configuration and manually computed `IpcBlock` `src_offset`, `dst_offset`, and `length`.

These functions included:

| Function | Cache Shape | TP configuration |
|----------|-------------|------------------|
| `parse_block_send_p_eq_d` | RAGGED_FLASH | P==D |
| `parse_block_send_p_gt_d` | RAGGED_FLASH | P>D |
| `parse_block_send_p_gt_d_dpsk` | RAGGED_FLASH (DPSK) | P>D (same `token_sizes`) |
| `vllm_parse_block_send_p_eq_d` | FLASH | P==D |
| `vllm_parse_block_send_p_gt_d` | FLASH | P>D |
| `vllm_parse_hybrid_block_send_p_eq_d_block_aligned` | QWEN3_NEXT | P==D (block-granularity send) |
| `vllm_parse_hybrid_block_send_p_gt_d` | QWEN3_NEXT | P>D |
| `vllm_parse_block_send_multi_tensor_p_eq_d` | DPSK_V32 | P==D |
| `vllm_parse_block_send_multi_tensor_p_gt_d` | DPSK_V32 | P>D |
| `vllm_parse_flashinfer_block_send_p_eq_d` | FLASHINFER | P==D |
| `vllm_parse_flashinfer_block_send_p_gt_d` | FLASHINFER | P>D |

The refactor abstracts offset computation into two layers:

1. **Layout description layer** (`AttnLayoutDesc`): Each cache shape describes its memory layout via dimension order (`dim_order`) and boolean flags, registered in a static layout table.
2. **Unified build layer** (`build_attn_spec_from_layout`): From the layout description and P/D TP sizes, it derives all `AttnSpec` stride/offset parameters automatically, without per-shape build functions.

`build_transfer_plan` handles P==D and P>D uniformly from the layout description (P<D is not considered for now) and produces a `TransferPlan`. Adding a cache shape only requires a new row in the layout registry.

---

## 2. Data structures

All structures are defined in `cache_transfer_spec.h` and `attn_layout.h`.

### 2.1 AttnSpec

Describes a single token-level K (or fused KV) copy stream, optionally with V offset parameters. Per-token src and dst offsets are computed as:

```
offset = blocks[block_idx] * block_stride
       + token_idx * token_stride
       + base_offset
       + global_offset
       + head_iter * head_stride
```

K component fields:

| Field | Meaning |
|-------|---------|
| `src_block_stride` / `dst_block_stride` | Stride from block index to byte offset (usually `block_size`) |
| `src_token_stride` / `dst_token_stride` | Stride from token index to byte offset (usually `token_size`) |
| `src_base_offset` / `dst_base_offset` | Fixed offset within a block (e.g. `group_off` for P>D, or K/V partition offset) |
| `src_global_offset` / `dst_global_offset` | Global offset (e.g. V layer offset for `FLASH_CACHE_SHAPE`) |
| `bytes_per_token` | Bytes copied per token |
| `head_num` | Number of KV heads (used by FLASHINFER head-major layout; default 1) |
| `src_head_stride` / `dst_head_stride` | Byte stride between adjacent heads (= ntpb * head_dim * dtype_size) |

V component (`AttnSpec::VSpec`, optional):

When K and V are transferred separately, `v` carries V’s offset parameters. V shares `block_stride`, `token_stride`, `bytes_per_token`, and `head_num` with K; only the following may differ:

| Field | Meaning |
|-------|---------|
| `src_base_offset` / `dst_base_offset` | Fixed V offset within a block |
| `src_global_offset` / `dst_global_offset` | V global offset (FLASH layout locates V via `layer_size`) |
| `src_head_stride` / `dst_head_stride` | V head stride (0 when FLASHINFER `v_head_invariant`) |

`v` settings by layout:

| Scenario | `v` | Notes |
|----------|-----|-------|
| P==D fused KV (RAGGED_FLASH, DPSK_V32) | `nullopt` | `bytes_per_token = token_size`; K+V transferred together |
| P==D block-aligned (QWEN3_NEXT) | `nullopt` | Whole-block transfer |
| P==D kv_outermost (FLASH) | set | V located via `global_offset = layer_size` |
| P==D head_outer (FLASHINFER) | set | V differs in `base_offset`, `head_stride` |
| P>D, all layouts | set | K and V must be transferred separately |

### 2.2 GdnSegment / GdnSpec

Used for GDN block transfer in Qwen3/3.5-Next. GDN is not token-granular; it transfers Conv/SSM caches.

```
GdnSegment:
  src_offset_in_block  -- src offset within block
  dst_offset_in_block  -- dst offset within block
  length               -- bytes to send

GdnSpec:
  num_groups           -- number of GDN layers (= WorkerInfo::num_gdn_layers); each has its own block list in BlockIds
  src_block_stride     -- src block size (bytes)
  dst_block_stride     -- dst block size (bytes)
  segments             -- segment list (1 segment for P==D; multiple conv + ssm segments for P>D)
```

GDN blocks are transferred only when `reach_last_token == true` (chunked prefill complete). `generate_gdn` iterates by group, using `src_blocks[g]` and `dst_blocks[g]` per group.

### 2.3 AttnComponent / GdnComponent

Token-level and GDN transfer components are the basic building blocks inside `CacheTransferSpec`.

```cpp
struct AttnComponent {
  AttnSpec spec;             // Token-level K (+ optional V) copy description
  uint32_t block_group = 0;  // Group index in BlockIds
  uint32_t src_ntpb = 0;     // Tokens per block on src
  uint32_t dst_ntpb = 0;     // Tokens per block on dst
  bool block_aligned = false; // Whole-block transfer granularity
};

struct GdnComponent {
  GdnSpec spec;  // GDN iterates BlockIds groups 0..num_groups-1 via num_groups
};
```

`AttnComponent` is used for attention and indexer:

- **fullattn**: Main attention transfer; `spec` may include VSpec
- **indexer**: Indexer cache for QWEN3_NEXT hybrid models; `spec` is K-only (`v = nullopt`)

### 2.4 CacheTransferSpec

Transfer specification for a single cache tensor: an ordered list of components; order is fixed at build time.

```cpp
using TransferComponent = std::variant<AttnComponent, GdnComponent>;

struct CacheTransferSpec {
  std::vector<TransferComponent> components;  // Build order: GDN -> Indexer -> FullAttn
  bool valid = true;                          // Whether this rank participates
};
```

Component order (as built by `build_transfer_plan`):

1. **GdnComponent** (handled only when `reach_last_token`)
2. **AttnComponent** for indexer (if present)
3. **AttnComponent** for fullattn

### 2.5 TransferPlan

```cpp
struct TransferPlan {
  std::vector<CacheTransferSpec> specs;  // One spec per cache tensor
  TPKind tpkind;                         // PEQD / PGTD
};
```

For single-tensor cache shapes (RAGGED_FLASH, FLASH, QWEN3_NEXT, FLASHINFER), `specs` has one element.
For multi-tensor cache shapes (DPSK_V32_SPARSE_MLA), `specs` has multiple elements, one per cache tensor.

### 2.6 CacheDim enum

Defined in `attn_layout.h`; the five cache tensor dimensions:

| Value | Meaning |
|-------|---------|
| `BLOCK` | num_gpu_blocks |
| `TOKEN` | tokens_per_block |
| `KV` | 2 (K and V) |
| `HEAD` | num_kv_heads |
| `HEAD_DIM` | head_dim |

### 2.7 AttnLayoutDesc

Defined in `attn_layout.h`; describes memory layout for one cache shape. `dim_order` and boolean flags drive the unified spec build.

| Field | Meaning |
|-------|---------|
| `dim_order` | Outermost-to-innermost dimension order (array of 5 `CacheDim` values) |
| `num_tensors` | Cache tensors per layer (DPSK_V32 uses 0; resolved at runtime from `WorkerInfo::token_sizes.size()`) |
| `block_aligned_peqd` | P==D block-aligned transfer (whole-block granularity) |
| `v_head_invariant` | V has `head_stride = 0`; no TP group offset on dst (FLASHINFER-specific) |
| `use_valid_ranks` | P>D: filter invalid ranks via `WorkerInfo::valid_ranks` |
| `valid_ranks_affects_group_n` | When `use_valid_ranks`, `group_n = valid_count / dst_tp` (not `src_tp / dst_tp`) |
| `rank0_only_when_same_sizes` | P>D with identical `token_sizes`: only rank 0 sends (treated like P==D) |
| `force_peqd_tpkind` | Force `TPKind` to PEQD (DPSK_V32 MLA) |

Helper methods (derived from relative positions of KV/HEAD/TOKEN/BLOCK in `dim_order`):

| Method | Meaning | Cache shape |
|--------|---------|-------------|
| `is_kv_outermost()` | KV dimension before BLOCK | FLASH |
| `is_kv_between_block_and_token()` | KV between BLOCK and TOKEN | QWEN3_NEXT |
| `is_kv_inner_to_token()` | KV after TOKEN | RAGGED_FLASH, DPSK_V32 |
| `is_head_outer_to_token()` | HEAD before TOKEN | FLASHINFER |

Registered layout configuration per cache shape:

| Cache shape | dim_order | Key flags |
|-------------|-----------|-----------|
| RAGGED_FLASH | `[BLOCK, TOKEN, KV, HEAD, HEAD_DIM]` | `rank0_only_when_same_sizes` |
| FLASH | `[KV, BLOCK, TOKEN, HEAD, HEAD_DIM]` | `use_valid_ranks` |
| QWEN3_NEXT | `[BLOCK, KV, TOKEN, HEAD, HEAD_DIM]` | `block_aligned_peqd`, `use_valid_ranks`, `valid_ranks_affects_group_n` |
| DPSK_V32 | `[BLOCK, TOKEN, KV, HEAD, HEAD_DIM]` | `num_tensors=0` (runtime), `rank0_only_when_same_sizes`, `force_peqd_tpkind` |
| FLASHINFER | `[BLOCK, KV, HEAD, TOKEN, HEAD_DIM]` | `v_head_invariant`, `use_valid_ranks`, `valid_ranks_affects_group_n` |

---

## 3. IpcBlock generation

Implemented in `cache_transfer_spec.cpp`.

### 3.1 Entry points

`generate_all_ipc_blocks(plan, task, send_blocks)` walks all specs in the plan and builds the `IpcBlock` list per cache tensor.

```
generate_ipc_blocks(spec, src_blocks, dst_blocks, wrote_tokens, left_tokens, reach_last_token, out)
```

`src_blocks` and `dst_blocks` are `BlockIds` (`vector<vector<uint32_t>>`); the outer vector is cache groups (e.g. QWEN3_NEXT GDN, indexer, attn).

`generate_ipc_blocks` walks `spec.components` in build order:

```
for comp in spec.components:
    match comp:
        AttnComponent -> generate_attn() or generate_block_aligned()
        GdnComponent  -> generate_gdn() (only when reach_last_token)
```

Each component uses its `block_group` index to select the block list from `BlockIds`.

### 3.2 Token-granularity transfer (`generate_attn`)

Handles K and V (if `VSpec` is present):

1. **K**: Offsets from main `AttnSpec` fields (`generate_k_block`)
2. **V** (if `spec.v`): `VSpec` `base_offset` / `global_offset` / `head_stride`, sharing K’s `block_stride` / `token_stride` / `bytes_per_token` / `head_num` (same `generate_k_block`)

Per stream:

1. From `wrote_tokens + done`, compute current block index and in-block offset.
2. Compute how many tokens can be handled contiguously (min of src block bound, dst block bound, remaining tokens).
3. For each head (`head_num` iterations), emit one `IpcBlock` per token with `length = bytes_per_token`. Adjacent `IpcBlock` merging happens later in `register_data`.

Output order: all K `IpcBlock`s first, then all V `IpcBlock`s.

### 3.3 Block-granularity transfer (`generate_block_aligned`)

Used when `AttnComponent.block_aligned = true`. Whole block is the transfer unit; send decision follows chunked prefill semantics:

| Condition | Send policy |
|-----------|-------------|
| `tokens == ntpb` (full block) | Send whole block |
| `reach_last_token` (last chunk of request) | Send whole block |
| `token_idx == 0` (tail of chunked prefill, more steps follow) | Do not send; defer |
| `token_idx + tokens == ntpb` (hit block boundary) | Send whole block |
| Otherwise (mid-block, request not done) | Do not send; defer |

### 3.4 GDN transfer (`generate_gdn`)

Only when `reach_last_token == true`. Iterate GDN by group with `src_blocks[g]` and `dst_blocks[g]`:

```
for g in 0..num_groups:
    for block_idx in 0..src_blocks[g].size():
        for segment in segments:
            out.emplace_back(
                src_blocks[g][block_idx] * src_block_stride + segment.src_offset,
                dst_blocks[g][block_idx] * dst_block_stride + segment.dst_offset,
                segment.length
            )
```

---

## 4. Build logic

### 4.1 Layout registry

`get_attn_layout(cache_shape)` looks up `AttnLayoutDesc` in a static table. New shapes only need a registry entry; no new build function. The table lives in `cache_transfer_spec.cpp`.

### 4.2 Unified attention spec build (`build_attn_spec_from_layout`)

`build_attn_spec_from_layout` infers KV dimension relationships from `AttnLayoutDesc::dim_order` and returns an `AttnComponent`:

```
build_attn_spec_from_layout(layout, p_block_size, p_token_size,
                            d_block_size, d_token_size, group_off,
                            layer_num_blocks_src, layer_num_blocks_dst, head_num)
```

Parameters:

| Parameter | Meaning |
|-----------|---------|
| `layout` | `AttnLayoutDesc` for the cache shape |
| `p_block_size` / `p_token_size` | Src block/token sizes in bytes |
| `d_block_size` / `d_token_size` | Dst block/token sizes in bytes |
| `group_off` | Offset within TP group (0 for P==D) |
| `layer_num_blocks_src` / `layer_num_blocks_dst` | Blocks per layer (FLASH: `layer_size`) |
| `head_num` | Head count (FLASHINFER head-outer only) |

Key intermediates:

```
p_k_token = p_token_size / 2          // K bytes per token
d_k_token = d_token_size / 2
src_ntpb  = p_block_size / p_token_size
dst_ntpb  = d_block_size / d_token_size
p_k_block = src_ntpb * p_k_token      // K bytes per block on src
d_k_block = dst_ntpb * d_k_token
```

When `env_bf162fp8_conversion()` and not P==D, `effective_p_k_token = p_k_token / 2`.

Component strategy (chosen from `dim_order`):

| Condition | Strategy | Cache shape |
|-----------|----------|-------------|
| P==D and `block_aligned_peqd` | Single block-aligned component, `block_aligned=true`, `v = nullopt` | QWEN3_NEXT |
| P==D, not head_outer, not kv_outermost | Single fused-KV component (`bytes_per_token = token_size`), `v = nullopt` | RAGGED_FLASH, DPSK_V32 |
| `is_head_outer_to_token()` | K uses `head_num` and `head_stride`; V differs via `VSpec` `base_offset` and `head_stride` (0 if `v_head_invariant`) | FLASHINFER |
| `is_kv_outermost()` | K `block_stride = k_block_size`; V via `VSpec.global_offset = layer_size` | FLASH |
| `is_kv_between_block_and_token()` | K `block_stride = block_size`; V via `VSpec.base_offset = k_block_size` | QWEN3_NEXT (P>D) |
| `is_kv_inner_to_token()` | K `token_stride = token_size`; V via `VSpec.base_offset = k_token_size` | RAGGED_FLASH (P>D) |

### 4.3 `build_transfer_plan`

`build_transfer_plan(cache_shape, src, dst)` is the entry: builds `TransferPlan` from cache shape and P/D worker info. There are no per-shape branches inside; behavior is driven by `AttnLayoutDesc` flags.

Components are appended to `CacheTransferSpec.components` in this order:

1. **GdnComponent** (only when `reach_last_token`)
2. **AttnComponent** for indexer (if any)
3. **AttnComponent** for fullattn

**P==D path:**

1. Per tensor, `build_attn_spec_from_layout` with `group_off=0` → fullattn `AttnComponent`
2. If `has_gdn`: append `GdnComponent` (`build_gdn_peqd`)
3. If `has_indexer`: append `AttnComponent` (`block_aligned=true`, `block_group = num_gdn_layers`)
4. Set fullattn `block_group = num_gdn_layers + (has_indexer ? 1 : 0)`

**P>D path:**

1. **`rank0_only_when_same_sizes` fast path**: If `layout.rank0_only_when_same_sizes` and P/D `token_sizes` match, only rank 0 sends as P==D; other ranks get an empty plan (`tpkind=PEQD`)
2. **Compute `group_off`**:
   - If `use_valid_ranks`: filter with `src.valid_ranks`; effective rank uses `src.kvt_tp_rank`; if `valid_ranks_affects_group_n`, `group_n = valid_count / dst_tp`
   - Else: `group_n = src_tp / dst_tp`, `group_off = rank % group_n`
3. If filtered out with no GDN/indexer, spec has `valid=false`
4. If `has_gdn`: append `GdnComponent` (`build_gdn_pgtd`)
5. If `has_indexer` and rank 0: append indexer `AttnComponent` (`block_aligned=false`, `src.hybrid_indexer_token_size` as token stride)
6. If `sends_attn`: `build_attn_spec_from_layout` → fullattn `AttnComponent`, set `block_group`

### 4.4 Memory layout per cache shape

#### RAGGED_FLASH_CACHE_SHAPE

Layout: `[num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]`

<p align="center">
  <img src="./imgs/ragged_flash_cache_shape_layout.png" width="60%" />
</p>

K and V are interleaved per token; `token_size = 2 * num_kv_heads_per_rank * head_dim * dtype_size`.

- **P==D**: `is_kv_inner_to_token()`, not head_outer, not kv_outermost → fused KV, `v = nullopt`, `bytes_per_token = token_size`; can coalesce to large block-level `IpcBlock`s.
- **P>D**: `is_kv_inner_to_token()` → set `v`; V at `v.base_offset = k_token_size`.
- **DPSK** (P>D, same `token_sizes`): `rank0_only_when_same_sizes=true`; only rank 0 sends.

#### FLASH_CACHE_SHAPE

Layout: `(2, num_blocks, block_size, num_kv_heads, head_dim)`

<p align="center">
  <img src="./imgs/flash_cache_shape_layout.png" width="60%" />
</p>

K and V live in separate layer regions: K at offset 0, V at `layer_size`.

- **P==D**: `is_kv_outermost()` → set `v`; V via `v.global_offset = layer_size`. Both streams can be merged.
- **P>D**: `is_kv_outermost()` → like P==D but `dst_base_offset` uses `group_off`. `use_valid_ranks=true` filters ranks. Supports `env_bf162fp8_conversion()` half conversion.

#### QWEN3_NEXT_FLASH_CACHE_SHAPE

Hybrid model; `BlockIds` use multiple groups:

- **Groups 0..num_gdn_layers-1**: GDN layers, one block list per group (conv state + ssm state)
- **Group num_gdn_layers** (optional): indexer blocks when `indexer_blk_ntpb > 0`
- **Group num_gdn_layers + (indexer?1:0)**: Attention blocks

<p align="center">
  <img src="./imgs/qwen3_next_gdn_layout.png" width="60%" />
</p>

<p align="center">
  <img src="./imgs/qwen3_next_attn_layout.png" width="60%" />
</p>

`CacheTransferSpec.components` order: `[GdnComponent, AttnComponent(indexer), AttnComponent(fullattn)]`

- **P==D (block-aligned)**: `block_aligned_peqd=true` → fullattn `block_aligned=true`; attention blocks whole-block. GDN via `GdnComponent` whole-block at `reach_last_token`. Indexer `block_aligned=true`.
- **P>D**: `is_kv_between_block_and_token()` → fullattn `v`; V at `v.base_offset = k_block_size`. GDN via `GdnComponent` (`build_gdn_pgtd`) conv/ssm segments. Indexer rank 0 only (`block_aligned=false`). `use_valid_ranks=true` + `valid_ranks_affects_group_n=true` affect attn grouping; GDN uses `src.engine_tp_size / dst.kvt_tp_size`. Conv sub-channel offsets:

  ```
  p_conv_dim_size = conv_state_shape[2] * gdn_conv_elem_size
  d_conv_dim_size = gdn_group_n * p_conv_dim_size
  conv_step_length = conv_channel_dims[i] * gdn_conv_elem_size

  p_conv_off = conv_dim * p_conv_dim_size + conv_dim_inner_offset
  d_conv_off = conv_dim * d_conv_dim_size + gdn_group_n * conv_dim_inner_offset + gdn_group_off * conv_step_length
  ```

#### DPSK_V32_SPARSE_MLA_SHAPE

Multiple cache tensors per layer; `token_sizes` and `block_sizes` are vectors. `num_tensors=0`; count from `WorkerInfo::token_sizes.size()` at runtime.

- **P==D**: Per tensor, `build_attn_spec_from_layout` (`is_kv_inner_to_token()` → fused KV, `v = nullopt`).
- **P>D, same `token_sizes` (DPSK)**: `rank0_only_when_same_sizes=true`; rank 0 sends all tensors.
- **P>D, different `token_sizes`**: Per tensor `build_attn_spec_from_layout` (`is_kv_inner_to_token()` → set `v`). `force_peqd_tpkind=true` keeps `TPKind` as PEQD.

#### FLASHINFER_CACHE_SHAPE

<p align="center">
  <img src="./imgs/FLASHINFER_CACHE_SHAPE_LAYOUT.png" width="60%" />
</p>

Layout: `(num_blocks, 2, num_kv_heads, block_size, head_dim)`

Head-major (`is_head_outer_to_token()`); tokens contiguous per head. Iterate by `head_num`.

- **K cache**: `head_stride = head_size`; each step advances to the next head.
- **V cache**: `v_head_invariant=true` → `VSpec.head_stride = 0`; no `group_off` on dst.
- **P>D**: `use_valid_ranks=true` + `valid_ranks_affects_group_n=true` filter ranks and affect `group_n`. `group_off` applies to dst K offset.

### 4.5 Helper build functions

| Function | Role |
|----------|------|
| `build_attn_spec_from_layout(...)` | Build `AttnComponent` from `AttnLayoutDesc` (K + optional VSpec) |
| `build_gdn_peqd(block_size, ntpb, token_size, worker_info)` | P==D GDN spec (whole-block; `num_groups` from `worker_info.num_gdn_layers`) |
| `build_gdn_pgtd(p_block_size, d_block_size, gdn_group_n, gdn_group_off, worker_info)` | P>D GDN spec (conv/ssm shapes, elem size, channel dims from `worker_info`) |

---

## 5. Changes in `tx_stub.cpp`

**Build phase** (`refresh_dst_info`):

`tx_stub` uses a **dual dispatch** path:

1. When `BLLM_KVTRANS_TX_PARSE_MODE` enables `cache_spec`:
   - `self.transfer_plan = build_transfer_plan(...)`
   - Send path uses `generate_all_ipc_blocks(...)`
2. Otherwise (default):
   - Legacy `parse_block_common` + `parse_block_*` dispatch
   - Parse function pointers for `RAGGED/FLASH/QWEN3_NEXT/DPSK/FLASHINFER`

**Execute phase** (`do_task`):

- If `transfer_plan` is set: `generate_all_ipc_blocks(self.transfer_plan.value(), ...)`
- Else: `self.parse_block(...)`

**TPKind**:

- `register_data` uses one rule: if `transfer_plan` exists, use `transfer_plan->tpkind`; else `self.tpkind`.
