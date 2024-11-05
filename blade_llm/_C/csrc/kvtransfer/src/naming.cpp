#include <stdexcept>
#include <mutex>
#include "naming.h"

namespace blade_llm {
bool NamingManager::register_factory(std::unique_ptr<INamingClientFactory> &&factory) {
  auto schema = factory->get_schema();
  std::unique_lock<std::shared_mutex> w_lock(shared_mutex_);
  auto ret = factories_.try_emplace(schema, std::move(factory));
  return ret.second;
}

void NamingManager::remove_factory(const Schema &schema) {
  std::unique_lock<std::shared_mutex> w_lock(shared_mutex_);
  factories_.erase(schema);
}

std::unique_ptr<INamingClient> NamingManager::connect_naming(const std::string &url) {
  auto pos = url.find(SCHEMA_DELIMITER);
  if (pos == std::string::npos) {
    throw std::runtime_error("unrecognized naming url;");
  }
  auto schema = url.substr(0, pos);
  auto content = url.substr(pos + 1, url.size());
  std::shared_lock<std::shared_mutex> r_lock(shared_mutex_);
  auto f = factories_.find(schema);
  if (f != factories_.end()) {
    auto client = f->second->create();
    client->connect(schema, content);
    return client;
  } else {
    throw std::runtime_error("unknown naming schema: " + schema);
  }
}
}
