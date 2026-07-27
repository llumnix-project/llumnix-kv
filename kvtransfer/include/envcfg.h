#pragma once
#include <arpa/inet.h>
#include <sys/socket.h>
#include <vector>
#include <bitset>

namespace blade_llm {

int env_heap_prof();

int env_send_parallel();
int env_ctx_tpsize();
int env_conn_tpsize();
int env_h2d_sync_tpsize();
int env_fsnaming_keepalive_interval_s();
int env_fsnaming_tolerate_interval_s();
// Tx stub send failure rate: (RETURN - 1) / 100.
int env_debug_tx_failrate();
int env_debug_tx_delay_ms();

// RETURN addr points to global storage and must not be modified.
// NULL means the send-done mechanism is disabled.
const struct sockaddr_in* env_send_done_addr();

// sync with kv_transfer_impl._get_layer_num_blocks
// (num_blocks, block_size, 2, num_kv_heads, head_dim)
constexpr int RAGGED_FLASH_CACHE_SHAPE = 1;
// (2, num_blocks, block_size, num_kv_heads, head_dim)
constexpr int FLASH_CACHE_SHAPE = 2;
// shape: (2, num_blocks, block_size, num_kv_heads, head_dim)
// stride:(num_blocks, 2, block_size, num_kv_heads, head_dim)
constexpr int QWEN3_NEXT_FLASH_CACHE_SHAPE = 3;
// Each layer contains two 3D tensors:
//   - indexer cache (for sparse top-k selection):
//       (num_blocks, block_size, head_size)
//   - main MLA cache (for attention):
//       (num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
// Compatible with both FlashAttention sparse MLA (Hopper, FLASH_MLA_SPARSE)
// and FlashInfer sparse MLA (Blackwell, FLASHINFER_MLA_SPARSE)
constexpr int DPSK_V32_SPARSE_MLA_SHAPE = 4;
// storage shape in l20:(num_blocks, 2, num_kv_heads, block_size, head_dim)
constexpr int FLASHINFER_CACHE_SHAPE = 5;
// TURBOQUANT cache shape, current only support hybrid model+turboquant
constexpr int TURBOQUANT_CACHE_SHAPE = 6;
// Qwen3-next hybrid model with FlashInfer HND attn layout.
// GDN/indexer follow QWEN3_NEXT layout; attn uses FLASHINFER HND:
// (num_blocks, 2, num_kv_heads, block_size, head_dim)
constexpr int QWEN3_NEXT_FLASHINFER_CACHE_SHAPE = 7;
// Kimi K3 hybrid layout: leading KDA state block groups followed by one
// rank-replicated MLA block group. Physical pages are sized by KDA and may
// differ across P/D TP; only the real MLA prefix is copied for the MLA group.
constexpr int KIMI_K3_MLA_CACHE_SHAPE = 8;

int env_cache_shape();

constexpr int SEND_DONE_HEAD_KIND = 1;
constexpr int SEND_SAVE_DONE_HEAD_KIND = 2;
int env_send_done_head_kind();

// RETURN addr points to global storage and must not be modified.
// Example: "4096,8000;8192,1000;" preallocates 8000 4096-byte buffers and
// 1000 8192-byte buffers.
const std::vector<std::pair<uint64_t, int>>* env_reserve();

int env_rpc_timeout_s();

int env_crc();

int env_port_base();

int env_send_tpsize();

int env_txstub_cap();

int env_waitlayer_tpsize();

int env_shrink_tpsize();

static constexpr size_t MAX_TP_SIZE = 16;

int env_attn_kernel_blk_size();

size_t env_kernel_copy_max_block_num() noexcept;

// SM usage ratio for copy kernel (0.0 to 1.0)
// Default: 1.0 means use all SMs
// Set to 0.5 to use 50% of SMs, etc.
double env_kernel_copy_sm_usage() noexcept;

// Enable BF16 to FP8 conversion during D2H transfer
// 0 = disabled (default), 1 = enabled
bool env_bf162fp8_conversion() noexcept;

// For qwen3_next P>D token-granularity attn transfer: fill the unfilled tail
// slots of the last decode-side attn block using the request's block 0 data,
// so the decode kernel never reads uninitialized KV in the padded tail.
// 0 = disabled, 1 = enabled (default).
bool env_pad_last_attn_block() noexcept;

// RDMA staged send: D2H to CPU buffer, then Send via RDMA Send/Recv
// 0 = disabled (default, use GDR WriteBatch), 1 = enabled (staged D2H + Send)
bool env_rdma_staged() noexcept;

const char* env_nic_name() noexcept;


}  // namespace blade_llm {
