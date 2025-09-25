#include <sstream>
#include "naming/eas_naming.h"
#include "thrid_party/json.h"
#include "thrid_party/logging.h"

namespace blade_llm {

EASNamingCache::EASNamingCache(const std::string name) :
    pod_name(name),
    version_(-1) {}

std::optional<std::string> EASNamingCache::get_value(const std::string &key) const {
  std::shared_lock<std::shared_mutex> lock(update_mutex_);
  auto iter = kv_.find(key);
  if (iter != kv_.end()) {
    return iter->second;
  }
  return std::nullopt;
}

std::vector<std::string> EASNamingCache::search_value(const std::string &prefix) const {
  std::shared_lock<std::shared_mutex> lock(update_mutex_);
  std::vector<std::string> result;
  for (const auto & [k, v] : kv_) {
    if (k.find(prefix) == 0) {
      result.push_back(v);
    }
  }
  return result;
}

uint32_t EASNamingCache::get_version() const {
  std::shared_lock<std::shared_mutex> lock(update_mutex_);
  return version_;
}

void EASNamingCache::set_outdated() {
  is_outdated_ .store(true, std::memory_order_release);
}

bool EASNamingCache::is_outdated() const {
  return is_outdated_.load(std::memory_order_relaxed);
}
void EASNamingCache::update(const std::string &str) {
  std::unique_lock<std::shared_mutex> lock(update_mutex_);
  if (is_outdated()) {
    is_outdated_.store(false, std::memory_order_release);
    kv_.clear();
    struct json_value_s *inst = json_parse(str.c_str(), str.length());
    struct json_object_s *inst_obj = json_value_as_object(inst);
    struct json_object_element_s *kv = inst_obj->start;
    while (kv != nullptr) {
      std::string key(kv->name->string, kv->name->string_size);
      struct json_string_s *v = json_value_as_string(kv->value);
      std::string value(v->string, v->string_size);
      if (key == EAS_VERSION_KEY) {
        auto latest = std::stoi(value);
        LOG(INFO) << "KVT: eas naming: update " << pod_name << " from version: " << version_ << " to " << latest;
        version_ = latest;
      } else {
        kv_[key] = std::move(value);
      }
      kv = kv->next;
    }
  }
}

void EASNamingClient::connect(const Schema &schema, const std::string &path) {
  if (schema != EAS_NAMING_SCHEMA) {
    throw std::runtime_error("unknown schema of eas naming client: " + schema);
  }
  auto t = getenv("EAS_NAMING_CACHE_EXPIRE_TIME");
  if (t != nullptr) {
    auto time = std::stoi(t);
    if (time > 0) {
      cache_expire_seconds_ = time;
      LOG(INFO) << "KVT: eas naming update cache expire time to " << cache_expire_seconds_ << " mill seconds;";
    }
  }

  service_endpoint_ = path;
  get_kv_path_ = service_endpoint_ + EAS_GET_API;
  list_instance_path_ = service_endpoint_ + EAS_LIST_API;
  LOG(INFO) << "KVT: set eas naming endpoint : " << service_endpoint_;
}

void EASNamingClient::store(const std::string &k, const std::string &v) {
  if (v.size() >= EAS_MAX_VALUE_LENGTH) {
    throw std::runtime_error("too large value size of eas naming;");
  }
  std::stringstream ss;
  ss << R"({")" << k << R"(":")" << v << R"("})";
  auto value = std::move(ss).str();
  LOG(INFO) << "KVT: eas naming store " << value << " size=" << value.size() << " to " << pod_name_;
  http_client_->post(EAS_POST_PATH, value);
}

std::optional<std::string> EASNamingClient::get(const InstanceId& inst, const std::string &k) {
  load();
  auto pod_opt = get_pod_kv(inst);
  if (pod_opt.has_value()) {
    return pod_opt.value()->get_value(k);
  }
  return std::nullopt;
}

std::vector<std::string> EASNamingClient::search(const InstanceId& inst, const std::string &prefix) {
  load();
  auto pod_opt = get_pod_kv(inst);
  if (pod_opt.has_value()) {
    return pod_opt.value()->search_value(prefix);
  }
  return {};
}

void EASNamingClient::remove(const std::string &key) {
  throw std::runtime_error("eas naming not support remove;");
}

const std::vector<std::string> &EASNamingClient::list() {
  load();
  return instance_list_;
}

inline void into_map(json_object_s *object, std::unordered_map<std::string, std::string> &pair) {
  struct json_object_element_s *kv = object->start;
  while (kv != nullptr) {
    std::string key(kv->name->string, kv->name->string_size);
    struct json_string_s *v = json_value_as_string(kv->value);
    pair[key] = { v->string, v->string_size };
    kv = kv->next;
  }
}

void EASNamingClient::load() {
  {
    std::shared_lock<std::shared_mutex> lock(m_mutex_);
    if (!pods_.empty() && time_watch_.get_elapse_ms() < cache_expire_seconds_) {
      return;
    }
  }

  std::unique_lock<std::shared_mutex> lock(m_mutex_);
  auto elapse_since_load = time_watch_.get_elapse_ms();
  if (instance_list_.empty() || elapse_since_load > cache_expire_seconds_) {
    LOG(INFO) << "KVT: eas naming: try to re-flush instances ...";
    auto resp = http_client_->get(list_instance_path_);
    if (resp.has_value()) {
      time_watch_ = TimeWatch();
      struct json_value_s *root = json_parse(resp->c_str(), resp->length());
      auto object = json_value_as_object(root);
      std::unordered_map<std::string, uint32_t> instance_latest;
      auto entry = object->start;
      while (entry != nullptr) {
        std::string pod_name(entry->name->string, entry->name->string_size);
        auto pod_info = json_value_as_object(entry->value);
        auto pod_info_kv = pod_info->start;
        while (pod_info_kv != nullptr) {
          std::string key(pod_info_kv->name->string, pod_info_kv->name->string_size);
          if (key == EAS_VERSION_KEY) {
            break;
          }
          pod_info_kv = pod_info_kv->next;
        }
        if (pod_info_kv != nullptr) {
          auto value = json_value_as_string(pod_info_kv->value);
          std::string version_str(value->string, value->string_size);
          auto version = std::stoi(version_str);
          instance_latest[pod_name] = version;
        }
        entry = entry->next;
      }

      for(const auto& pod_name: instance_list_) {
        auto exist = instance_latest.find(pod_name);
        if (exist == instance_latest.end()) {
          LOG(INFO) << "remove pod: " << pod_name;
          pods_.erase(pod_name);
        }
      }

      for(const auto&[pod_name, v] : instance_latest) {
        auto exist = pods_.find(pod_name);
        if (exist == pods_.end()) {
          LOG(INFO) << "detect new pod: " << pod_name;
          auto ret = pods_.insert(std::make_pair(pod_name, pod_name));
          ret.first->second.set_outdated();
        } else {
          auto pre_v = exist->second.get_version();
          if (pre_v < v) {
            LOG(INFO) << "detect new version of pod: " << pod_name << ", new version=" << v;
            exist->second.set_outdated();
          }
        }
      }

      instance_list_.clear();
      for(auto&[k, v] : pods_) {
        instance_list_.push_back(k);
      }
    }
  }
}
std::optional<EASNamingCache *> EASNamingClient::get_pod_kv(const std::string &pod_name) {
  std::shared_lock<std::shared_mutex> lock(m_mutex_);
  auto f = pods_.find(pod_name);
  if (f != pods_.end()) {
    if (f->second.is_outdated()) {
      auto pod_path = get_kv_path_ + "/" + pod_name;
      auto resp = http_client_->get(pod_path);
      if (resp.has_value()) {
        f->second.update(resp.value());
      } else {
        return std::nullopt;
      }
    }
    return &f->second;
  }
  return std::nullopt;
}
}
