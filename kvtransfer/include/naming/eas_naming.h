#ifndef KVTRANSFER_INCLUDE_NAMING_EAS_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_EAS_NAMING_H_

#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <memory>
#include <atomic>
#include "naming.h"
#include "utils/http_helper.h"
#include "utils/timer.h"

namespace blade_llm {

#define EAS_MAX_VALUE_LENGTH (2048)

const static std::string EAS_NAMING_SCHEMA = "eas";

class EASNamingCache : noncopyable {
 public:
  const std::string pod_name;
  EASNamingCache(const std::string name);
  void update(const std::string &str);
  std::optional<std::string> get_value(const std::string &key) const;
  std::vector<std::string> search_value(const std::string &prefix) const;
  uint32_t get_version() const;
  void set_outdated();
  bool is_outdated() const;
 private:
  mutable std::shared_mutex update_mutex_;
  std::atomic_bool is_outdated_{false};
  long version_{-1};
  std::unordered_map<std::string, std::string> kv_;
};

static const std::string EAS_POST_API{"/api/message"};
static const std::string EAS_GET_API{"/api/messages"};
static const std::string EAS_LIST_API{"/api/instances"};
static const std::string EAS_INSTANCE_KEY{"__instance__"};
// https://project.aone.alibaba-inc.com/v2/project/664220/req/76900958
// REAL VERSION KEY: __timestamp__
static const std::string EAS_VERSION_KEY{"__timestamp__"};
static const std::string EAS_LOCAL_ENDPOINT{"http://127.0.0.1:9900"};
static const std::string EAS_POST_PATH = EAS_LOCAL_ENDPOINT + EAS_POST_API;

class EASNamingClient : public INamingClient, noncopyable {
 public:
  EASNamingClient(const std::string &pod_name, std::unique_ptr<IHttpClient> &&client) :
      INamingClient(pod_name),
      pod_name_(pod_name),
      http_client_(std::move(client)) {};
  Schema get_schema() override {
    return EAS_NAMING_SCHEMA;
  }
  void connect(const Schema &schema, const std::string &path) override;
  void store(const std::string &k, const std::string &v) override;
  void remove(const std::string &key) override;
  std::optional<std::string> get(const InstanceId&, const std::string &k) override;
  std::vector<std::string> search(const InstanceId &, const std::string &prefix) override;
  const std::vector<std::string> &list() override;
 private:
  void load();
  std::optional<EASNamingCache *> get_pod_kv(const std::string &pod_name);

  uint32_t cache_expire_seconds_{1};
  std::string pod_name_;
  std::string service_endpoint_;
  std::string get_kv_path_;
  std::string list_instance_path_;
  std::unique_ptr<IHttpClient> http_client_;
  std::unordered_map<std::string, EASNamingCache> pods_;
  std::vector<std::string> instance_list_;
  std::shared_mutex m_mutex_;
  TimeWatch time_watch_;
};

class EASNamingClientFactory : public INamingClientFactory, noncopyable {
 public:
  const Schema &get_schema() override {
    return EAS_NAMING_SCHEMA;
  }
  std::unique_ptr<INamingClient> create(const InstanceId& inst_name) override {
    return std::make_unique<EASNamingClient>(inst_name, std::make_unique<CurlHttpClient>());
  };
};

} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_NAMING_EAS_NAMING_H_
