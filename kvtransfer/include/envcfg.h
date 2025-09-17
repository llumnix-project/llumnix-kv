#pragma once
#include <arpa/inet.h>
#include <sys/socket.h>
#include <vector>

namespace blade_llm {

int env_send_parallel();
int env_ctx_tpsize();
int env_conn_tpsize();
int env_fsnaming_keepalive_interval_s();
int env_fsnaming_tolerate_interval_s();
// tx stub 发送失败率: (RETURN - 1) / 100
int env_debug_tx_failrate();
int env_debug_tx_delay_ms();

// RETURN addr 指向着某处全局空间, 不要修改.
// NULL 意味着不使用 send done 机制.
const struct sockaddr_in* env_send_done_addr();

// sync with kv_transfer_impl._get_layer_num_blocks
// (num_blocks, block_size, 2, num_kv_heads, head_dim)
constexpr int RAGGED_FLASH_CACHE_SHAPE = 1;
// (2, num_blocks, block_size, num_kv_heads, head_dim)
constexpr int FLASH_CACHE_SHAPE = 2;
int env_cache_shape();

constexpr int SEND_DONE_HEAD_KIND = 1;
constexpr int SEND_SAVE_DONE_HEAD_KIND = 2;
int env_send_done_head_kind();

// RETURN addr 指向着某处全局空间, 不要修改.
// 示例: "4096,8000;8192,1000;" 意味着预先分配 8000 个 4096byte 的内存, 1000 个 8192 byte 的内存.
const std::vector<std::pair<uint64_t, int>>* env_reserve();


}  // namespace blade_llm {