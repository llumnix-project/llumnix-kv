#pragma once

namespace blade_llm {

int env_send_parallel();
int env_ctx_tpsize();
int env_conn_tpsize();
int env_fsnaming_keepalive_interval_s();
int env_fsnaming_tolerate_interval_s();

}  // namespace blade_llm {