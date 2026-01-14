
#include "envcfg.h"
#include "common.h"
#include <mutex>
#include <assert.h>
#include <string>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

namespace blade_llm {

static int env2posint(const char* env, int def) {
  const char *valstr = getenv(env);
  fprintf(stdout, "kvtenv: env=%s val=%s def=%d\n", env, (valstr == nullptr ? "NULL" : valstr), def);
  fflush(stdout);
  if (valstr == nullptr) {
    return def;
  }
  int ret = atoi(valstr);
  if (ret <= 0) {
    return def;
  }
  return ret;
}

int env_heap_prof() {
  static constexpr int DEFVAL = 0;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_HEAP_PROF", DEFVAL);
  });
  return val;
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
  static constexpr int DEFVAL = 4;
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
  static constexpr int DEFVAL = 1;
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
  RTASSERT(colon_pos != std::string::npos);

  std::string ip_str = addr_str.substr(0, colon_pos);
  std::string port_str = addr_str.substr(colon_pos + 1);
  int port = std::stoi(port_str);

  memset(addr, 0, sizeof(*addr));
  addr->sin_family = AF_INET;
  addr->sin_port = htons(port);

  auto inet_ok = inet_pton(AF_INET, ip_str.c_str(), &addr->sin_addr);
  RTASSERT(inet_ok == 1);
  return ;
}

const struct sockaddr_in* env_send_done_addr() {
  static struct sockaddr_in* addr_ptr = nullptr;
  static struct sockaddr_in addr;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* addr_env = getenv("BLLM_KVTRANS_SEND_DONE_ADDR");
    fprintf(stdout, "kvtenv: env=BLLM_KVTRANS_SEND_DONE_ADDR val=%s\n", (addr_env == nullptr ? "NULL" : addr_env));
    fflush(stdout);
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

static void parse_reserve(std::vector<std::pair<uint64_t, int>>* out, const char* inputval) {
  auto input = std::string(inputval);
  size_t start = 0;
  size_t end = 0;
  while ((end = input.find(';', start)) != std::string::npos) {
    assert(end >= start);
    auto part = input.substr(start, end - start);
    start = end + 1;
    auto idx = part.find(',');
    if (idx == std::string::npos) {
      continue;
    }
    uint64_t mrsize = std::stoul(part.substr(0, idx - 0));
    int mrcnt = std::stoi(part.substr(idx + 1));
    out->emplace_back(mrsize, mrcnt);
  }
  return;
}

const std::vector<std::pair<uint64_t, int>>* env_reserve() {
  static std::vector<std::pair<uint64_t, int>> val;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* valenv = getenv("BLLM_KVTRANS_RESERVE");
    fprintf(stdout, "kvtenv: env=BLLM_KVTRANS_RESERVE val=%s\n", (valenv == nullptr ? "NULL" : valenv));
    fflush(stdout);
    if (valenv == nullptr) {
      return;
    }
    try {
      parse_reserve(&val, valenv);
    } catch (const std::exception& e) {
      fprintf(stderr, "env_reserve: bad input=%s;ex=%s\n", valenv, e.what());
    }
  });
  return &val;
}

int env_rpc_timeout_s() {
  static constexpr int DEFVAL = 7;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_RPC_TIMEOUT_S", DEFVAL);
  });
  return val;
}

int env_crc() {
  static constexpr int DEFVAL = 0;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_CRC", DEFVAL);
  });
  return val;
}

int env_port_base() {
  static constexpr int DEFVAL = 31218;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_PORT_BASE", DEFVAL);
  });
  return val;
}


int env_send_tpsize() {
  static constexpr int DEFVAL = 16;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_SEND_TPSIZE", DEFVAL);
  });
  return val;
}


int env_txstub_cap() {
  static constexpr int DEFVAL = 640;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_TXSTUB_CAP", DEFVAL);
  });
  return val;
}


int env_waitlayer_tpsize() {
  static constexpr int DEFVAL = 2;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_WAITLAYER_TPSIZE", DEFVAL);
  });
  return val;
}


int env_shrink_tpsize() {
  static constexpr int DEFVAL = 1;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_SHRINK_TPSIZE", DEFVAL);
  });
  return val;
}

std::bitset<MAX_TP_SIZE> env_p_valid_ranks() noexcept {
  static std::bitset<MAX_TP_SIZE> val("1111111111111111111111111111111111111111111111111111111111111111");
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* valenv = getenv("BLLM_KVTRANS_P_VALID_RANKS");
    fprintf(stdout, "kvtenv: env=BLLM_KVTRANS_P_VALID_RANKS val=%s\n", (valenv == nullptr ? "NULL" : valenv));
    fflush(stdout);
    if (valenv == nullptr) {
      return;
    }
    val = std::bitset<MAX_TP_SIZE>(valenv);
  });
  return val;
}

int env_attn_head_num(){
  static constexpr int DEFVAL = -1;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_ATTN_HEAD_NUM", DEFVAL);
  });
  return val;
}

int env_gdn_element_size() {
  static constexpr int DEFVAL = 1;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("GDN_ELEMENT_SIZE", DEFVAL);
  });
  return val;
}

static void parse_tensor_shape(std::vector<size_t>* out, const char* inputval) {
  if (inputval == nullptr) {
    return;
  }
  auto input = std::string(inputval);
  size_t start = 0;
  size_t end = 0;
  while ((end = input.find(',', start)) != std::string::npos) {
    assert(end >= start);
    auto part = input.substr(start, end - start);
    start = end + 1;
    if (!part.empty()) {
      try {
        size_t dim = std::stoull(part);
        out->push_back(dim);
      } catch (const std::exception& e) {
        fprintf(stderr, "parse_tensor_shape: failed to parse dimension '%s': %s\n", part.c_str(), e.what());
      }
    }
  }
  if (start < input.length()) {
    auto part = input.substr(start);
    if (!part.empty()) {
      try {
        size_t dim = std::stoull(part);
        out->push_back(dim);
      } catch (const std::exception& e) {
        fprintf(stderr, "parse_tensor_shape: failed to parse dimension '%s': %s\n", part.c_str(), e.what());
      }
    }
  }
  return;
}

const std::vector<size_t>* env_ssm_state_shape() {
  static std::vector<size_t> val;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* valenv = getenv("QWEN3_NEXT_SSM_SHAPE");
    fprintf(stdout, "kvtenv: env=QWEN3_NEXT_SSM_SHAPE val=%s\n", (valenv == nullptr ? "NULL" : valenv));
    fflush(stdout);
    if (valenv == nullptr) {
      return;
    }
    try {
      parse_tensor_shape(&val, valenv);
    } catch (const std::exception& e) {
      fprintf(stderr, "env_ssm_state_shape: bad input=%s;ex=%s\n", valenv, e.what());
    }
  });
  return &val;
}

const std::vector<size_t>* env_conv_state_shape() {
  static std::vector<size_t> val;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* valenv = getenv("QWEN3_NEXT_CONV_SHAPE");
    fprintf(stdout, "kvtenv: env=QWEN3_NEXT_CONV_SHAPE val=%s\n", (valenv == nullptr ? "NULL" : valenv));
    fflush(stdout);
    if (valenv == nullptr) {
      return;
    }
    try {
      parse_tensor_shape(&val, valenv);
    } catch (const std::exception& e) {
      fprintf(stderr, "env_conv_state_shape: bad input=%s;ex=%s\n", valenv, e.what());
    }
  });
  return &val;
}

int env_gdn_block_num() {
  static constexpr int DEFVAL = 3;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_GDN_BLOCK_NUM", DEFVAL);
  });
  return val;
}

uint32_t env_origin_p_tp_size() noexcept {
  static uint32_t val = 0;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    auto* valenv = getenv("BLLM_KVTRANS_ORIGIN_P_TP_SIZE");
    if (valenv != nullptr) {
      val = static_cast<uint32_t>(atoi(valenv));
    }
  });
  return val;
}

}  // namespace blade_llm {