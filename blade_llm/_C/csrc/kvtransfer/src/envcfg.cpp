
#include "envcfg.h"
#include <mutex>
#include <assert.h>

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


}  // namespace blade_llm {