#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include "naming/tcpstore_naming.h"

#ifdef ENABLE_TORCH

namespace blade_llm {

// url 格式 '//ip:port'
static std::pair<std::string, int> parse_url(const std::string& url) {
  auto pos = url.find(':', 2);
  if (pos == std::string::npos) {
    throw std::runtime_error("bad url: " + url);
  }
  assert(pos >= 2);
  auto ip = url.substr(2, pos - 2);
  assert(url.c_str()[pos] == ':');
  int port = atoi(&url.c_str()[pos + 1]);
  if (port <= 0) {
    throw std::runtime_error("bad port: " + url);
  }
  return make_pair(ip, port);
}

// url: //ip:port
void TCPStoreNaming::connect(const Schema& schema, const std::string &url)  {
  if(schema != TCP_NAMING_SCHEMA) {
    throw std::runtime_error("unknown schema: " + schema);
  }
  auto& self = *this;
  auto [addr, port] = parse_url(url);
  c10d::TCPStoreOptions opts;
  opts.port = port;
  self.tcp_store_.emplace(addr, opts);
}

void TCPStoreNaming::store(const std::string &key, const std::string &v) {
  auto& self = *this;
  std::vector<uint8_t> value(v.begin(),v.end());
  assert(self.tcp_store_);
  auto real_key = inst_id + "/" + key;
  self.tcp_store_->set(real_key, value);
}

std::optional<std::string> TCPStoreNaming::get(const InstanceId& inst, const std::string &k) {
  auto& self = *this;
  assert(self.tcp_store_);
  auto real_key = inst + "/" + k;
  auto val = self.tcp_store_->get(real_key);
  if (val.empty()) {
    return std::nullopt;
  } else {
    return std::string(val.begin(), val.end());
  }
}

std::vector<std::string> TCPStoreNaming::search(const InstanceId &inst, const std::string &prefix) {
  throw std::runtime_error("unsupported search operation of tcpstore");
}

void TCPStoreNaming::remove(const std::string &key) {
  auto real_key = inst_id + "/" + key;
  auto& self = *this;
  assert(self.tcp_store_);
  self.tcp_store_->deleteKey(real_key);
}

const std::vector<std::string>&  TCPStoreNaming::list() {
  throw std::runtime_error("unsupported list operation of tcpstore");
}

}  // namespace blade_llm {

#endif // USE_TORCH_TCP_STORE
