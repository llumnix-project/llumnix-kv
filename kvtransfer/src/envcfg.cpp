
#include "envcfg.h"
#include "common.h"
#include <mutex>
#include <assert.h>
#include <string>
#include <string.h>
#include <stdlib.h>

namespace blade_llm {

static int env2posint(const char* env, int def) {
  assert(def > 0);
  const char *valstr = getenv(env);
  if (valstr == nullptr) {
    return def;
  }
  int ret = atoi(valstr);
  if (ret <= 0) {
    return def;
  }
  return ret;
}

int env_send_parallel() {
  static constexpr int DEFVAL = 1;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_RDMA_SP", DEFVAL);
  });
  return val;
}

int env_ctx_tpsize() {
  static constexpr int DEFVAL = 4;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_CTX_TPSIZE", DEFVAL);
  });
  return val;
}


int env_conn_tpsize() {
  static constexpr int DEFVAL = 2;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_CONN_TPSIZE", DEFVAL);
  });
  return val;
}

int env_fsnaming_keepalive_interval_s() {
  static constexpr int DEFVAL = 3;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_FSNAMING_KEEPALIVE_S", DEFVAL);
  });
  return val;
}

int env_fsnaming_tolerate_interval_s() {
  static constexpr int DEFVAL = 9;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_FSNAMING_TOLERATE_S", DEFVAL);
  });
  return val;
}

int env_debug_tx_failrate() {
  static constexpr int DEFVAL = 1;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_DEBUG_TX_FAILRATE", DEFVAL);
  });
  return val;
}

int env_debug_tx_delay_ms() {
  static constexpr int DEFVAL = 0;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_DEBUG_TX_DELAYMS", DEFVAL);
  });
  return val;
}


// input: IP:PORT
static void parse_addr(struct sockaddr_in* addr, const char* input) {
  auto addr_str = std::string(input);
  size_t colon_pos = addr_str.find(':');
  RTCHECK(colon_pos != std::string::npos);

  std::string ip_str = addr_str.substr(0, colon_pos);
  std::string port_str = addr_str.substr(colon_pos + 1);
  int port = std::stoi(port_str);

  memset(addr, 0, sizeof(*addr));
  addr->sin_family = AF_INET;
  addr->sin_port = htons(port);

  auto inet_ok = inet_pton(AF_INET, ip_str.c_str(), &addr->sin_addr);
  RTCHECK(inet_ok == 1);
  return ;
}

const struct sockaddr_in* env_send_done_addr() {
  static struct sockaddr_in* addr_ptr = nullptr;
  static struct sockaddr_in addr;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* addr_env = getenv("BLLM_KVTRANS_SEND_DONE_ADDR");
    if (addr_env == nullptr) {
      return ;
    }
    parse_addr(&addr, addr_env);
    addr_ptr = &addr;
    return ;
  });
  return addr_ptr;
}

int env_cache_shape() {
  static constexpr int DEFVAL = RAGGED_FLASH_CACHE_SHAPE;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_CACHE_SHAPE", DEFVAL);
  });
  return val;
}


int env_send_done_head_kind() {
  static constexpr int DEFVAL = SEND_SAVE_DONE_HEAD_KIND;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_SDH_KIND", DEFVAL);
  });
  return val;
}

}  // namespace blade_llm {