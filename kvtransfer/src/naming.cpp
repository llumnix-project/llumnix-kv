#include <stdexcept>
#include <mutex>
#include <sstream>
#include "naming.h"
#include "naming/eas_naming.h"
#include "naming/filesys_naming.h"
#include "thrid_party/logging.h"
namespace blade_llm {
GeneralNamingClient NamingManager::connect_naming(const InstanceId& myname, const std::string &url) {
  {
    std::shared_lock<std::shared_mutex> r_lock(shared_mutex_);
    auto ret = naming_clients_.find(url);
    if (ret != naming_clients_.end()) {
      return ret->second;
    }
  }

  auto pos = url.find(SCHEMA_DELIMITER);
  if (pos == std::string::npos) {
    throw std::runtime_error("unrecognized naming url;");
  }
  auto schema = url.substr(0, pos);
  auto content = url.substr(pos + 1, url.size());
  std::unique_lock<std::shared_mutex> w_lock(shared_mutex_);
  {
    auto ret = naming_clients_.find(url);
    if (ret != naming_clients_.end()) {
      return ret->second;
    }
  }

  auto f = factories_.find(schema);
  if (f != factories_.end()) {
    auto client = f->second->create(myname);
    client->connect(schema, content);
    auto res = naming_clients_.insert(std::make_pair(url, GeneralNamingClient(std::move(client))));
    return res.first->second;
  } else {
    std::stringstream ss;
    ss << "unknown naming schema: " << schema << ",supports: [";
    for(const auto &p: factories_) {
      ss << p.first << ",";
    }
    ss << "];";
    LOG(INFO) << std::move(ss).str();
    return GeneralNamingClient();
  }
}
NamingManager::NamingManager() {
  if (factories_.empty()) {
    factories_.emplace(EAS_NAMING_SCHEMA, std::make_unique<EASNamingClientFactory>());
    factories_.emplace(FILESYS_NAMING_SCHEMA, std::make_unique<FileSysNamingClientFactory>());
  }
}

std::unique_ptr<INamingWorkerClient> GeneralNamingClient::create_naming_worker_client() {
  auto ptr = client_.get();
  return std::make_unique<WorkerNamingClient>(ptr);
}
}
