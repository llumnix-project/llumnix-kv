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
// tx stub send failure rate: (RETURN - 1) / 100
int env_debug_tx_failrate();
int env_debug_tx_delay_ms();

// RETURN addr points to some global memory; do not modify.
// NULL means the send-done mechanism is not used.
const struct sockaddr_in* env_send_done_addr();

// sync with kv_transfer_impl._get_layer_num_blocks
// (num_blocks, block_size, 2, num_kv_heads, head_dim)
constexpr int RAGGED_FLASH_CACHE_SHAPE = 1;
// (2, num_blocks, block_size, num_kv_heads, head_dim)
constexpr int FLASH_CACHE_SHAPE = 2;
// shape: (2, num_blocks, block_size, num_kv_heads, head_dim)
// stride:(num_blocks, 2, block_size, num_kv_heads, head_dim)
constexpr int QWEN3_NEXT_FLASH_CACHE_SHAPE = 3;
// Each layer contains two tensors: 
// k tensor for select and mla tensor for attention.
constexpr int DPSK_V32_SPARSE_MLA_SHAPE = 4;
// storage shape in l20:(num_blocks, 2, num_kv_heads, block_size, head_dim)
constexpr int FLASHINFER_CACHE_SHAPE = 5;

int env_cache_shape();

constexpr int SEND_DONE_HEAD_KIND = 1;
constexpr int SEND_SAVE_DONE_HEAD_KIND = 2;
int env_send_done_head_kind();

// RETURN addr points to some global memory; do not modify.
// Example: "4096,8000;8192,1000;" means pre-allocate 8000 chunks of 4096 bytes and 1000 chunks of 8192 bytes.
const std::vector<std::pair<uint64_t, int>>* env_reserve();

int env_rpc_timeout_s();

int env_crc();

int env_port_base();

int env_send_tpsize();

int env_txstub_cap();

int env_waitlayer_tpsize();

int env_shrink_tpsize();

static constexpr size_t MAX_TP_SIZE = 64;
std::bitset<MAX_TP_SIZE> env_p_valid_ranks() noexcept;

int env_attn_head_num();

int env_gdn_element_size();

int env_gdn_block_num();

uint32_t env_origin_p_tp_size() noexcept;

size_t env_kernel_copy_max_block_num() noexcept;

// SM usage ratio for copy kernel (0.0 to 1.0)
// Default: 1.0 means use all SMs
// Set to 0.5 to use 50% of SMs, etc.
double env_kernel_copy_sm_usage() noexcept;

// Enable BF16 to FP8 conversion during D2H transfer
// 0 = disabled (default), 1 = enabled
bool env_bf162fp8_conversion() noexcept;

const std::vector<size_t>* env_ssm_state_shape();
const std::vector<size_t>* env_conv_state_shape();

}  // namespace blade_llm {