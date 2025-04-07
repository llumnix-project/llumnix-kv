#pragma once
#include <arpa/inet.h>
#include <sys/socket.h>

namespace blade_llm {

int env_send_parallel();
int env_ctx_tpsize();
int env_conn_tpsize();
int env_fsnaming_keepalive_interval_s();
int env_fsnaming_tolerate_interval_s();
// tx stub 发送失败率: (RETURN - 1) / 100
int env_debug_tx_failrate();

// RETURN addr 指向着某处全局空间, 不要修改.
const struct sockaddr_in* env_send_done_addr();

// sync with kv_transfer_impl._get_layer_num_blocks
// (num_blocks, block_size, 2, num_kv_heads, head_dim)
constexpr int RAGGED_FLASH_CACHE_SHAPE = 1;
// (2, num_blocks, block_size, num_kv_heads, head_dim)
constexpr int FLASH_CACHE_SHAPE = 2;
int env_cache_shape();

}  // namespace blade_llm {