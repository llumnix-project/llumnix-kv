#include <cstdlib>
#include "naming/tcpstore_naming.h"

#ifdef ENABLE_RDMA

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
bool TCPStoreNaming::init(const std::string &url)  {
  auto& self = *this;
  auto [addr, port] = parse_url(url);
  c10d::TCPStoreOptions opts;
  opts.port = port;
  self.tcp_store_.emplace(addr, opts);
  return true;
}

std::string TCPStoreNaming::get_key(uint32_t inst_id, uint32_t worker_id) {
  // optimize with fmt::fmt/absel::StrCat
  // T/W 是 TCPStoreNaming/WorkerInfo/ 的缩写, 节省点空间==
  return "T/W/" + std::to_string(inst_id) + "/" + std::to_string(worker_id);
}

std::vector<uint8_t> TCPStoreNaming::to_value(const WorkerInfo& info) {
  std::vector<uint8_t> ret;
  static_assert(std::is_standard_layout_v<WorkerInfo>);
  // WorkerInfo 是 POD 类型, 可以安全 memcpy.
  ret.resize(sizeof(WorkerInfo));
  memcpy(ret.data(), &info, sizeof(WorkerInfo));
  return ret;
}

WorkerInfo TCPStoreNaming::from_value(const std::vector<uint8_t>& input) {
  WorkerInfo wi;
  assert(input.size() == sizeof(wi));
  memcpy(&wi, input.data(), input.size());
  return wi;
}

bool TCPStoreNaming::register_worker(const WorkerInfo& worker_info) {
  auto& self = *this;
  auto key = get_key(worker_info.inst_id, worker_info.worker_id);
  assert(self.tcp_store_);
  self.tcp_store_->set(key, to_value(worker_info));
  return true;
}

std::optional<WorkerInfo> TCPStoreNaming::get_worker_info(uint32_t inst_id, uint32_t worker_id) {
  auto& self = *this;
  assert(self.tcp_store_);
  auto val = self.tcp_store_->get(get_key(inst_id, worker_id));
  if (val.size() != sizeof(WorkerInfo)) {
    return std::nullopt;
  }
  return from_value(val);
}

}  // namespace blade_llm {

#endif // ENABLE_RDMA