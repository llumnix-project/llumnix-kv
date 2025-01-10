#pragma once

namespace blade_llm {

int env_send_parallel();
int env_ctx_tpsize();
int env_conn_tpsize();
int env_fsnaming_keepalive_interval_s();
int env_fsnaming_tolerate_interval_s();
// tx stub 发送失败率: (RETURN - 1) / 100
int env_debug_tx_failrate();

}  // namespace blade_llm {