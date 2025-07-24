#include "naming/filesys_naming.h"
#include "thrid_party/logging.h"
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "envcfg.h"

namespace blade_llm {

void FileSysNaming::connect(const Schema &schema, const std::string &path) {
  if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
    naming_path_ = path;
    instance_path_ = naming_path_ / inst_id;
    if (!std::filesystem::exists(instance_path_)) {
      std::filesystem::create_directory(instance_path_);
      write_file("_timestamp_", std::to_string(get_unix_timestamp()));
      periodic_task_ = std::thread([this]() {
        while (!this->stop_.load(std::memory_order_relaxed)) {
          auto sleep_dur = std::chrono::seconds(env_fsnaming_keepalive_interval_s());
          std::this_thread::sleep_for(sleep_dur);
          write_file("_timestamp_", std::to_string(get_unix_timestamp()));
        }
      });
    }
  } else {
    throw std::runtime_error("parent directory not exist: " + path);
  }
}

void FileSysNaming::store(const std::string &k, const std::string &v) {
  auto f = k.find('/');
  if (f != std::string::npos) {
    throw std::runtime_error("unsupported key format: " + k);
  }
  write_file(k, v);
}

std::optional<std::string> FileSysNaming::get(const InstanceId& inst_n, const std::string &k) {
  auto inst_path = naming_path_ / inst_n;
  if (std::filesystem::exists(inst_path) && std::filesystem::is_directory(inst_path)) {
    auto full_path = inst_path / k;
    std::ifstream in(full_path);
    if (in) {
      std::string content;
      std::getline(in, content);
      in.close();
      if (!content.empty()) {
        return content;
      }
    }
  }
  return std::nullopt;
}

std::vector<std::string> FileSysNaming::search(const InstanceId &inst_n, const std::string &prefix) {
  std::vector<std::string> result;
  auto inst_path = naming_path_ / inst_n;
  if (std::filesystem::exists(inst_path) && std::filesystem::is_directory(inst_path)) {
    for (const auto &entry : std::filesystem::directory_iterator(inst_path)) {
      if (entry.path().filename().string().find(prefix) == 0) {
        if (std::ifstream in{entry.path()}) {
          std::string content;
          std::getline(in, content);
          in.close();
          if (!content.empty()) {
            result.push_back(std::move(content));
          }
        }
      }
    }
  }
  return result;
}

void FileSysNaming::remove(const std::string &key) {
  auto full_path = instance_path_ / key;
  if (std::remove(full_path.c_str()) != 0) {
    throw std::runtime_error("failed to remove file: " + key);
  }
}

const std::vector<std::string> & FileSysNaming::list() {
  std::unique_lock<std::mutex> lock(list_mutex_);
  auto elapse = timer_.get_elapse_ms();
  if (list_cache_.empty() || elapse > 2000) {
    auto full_path = naming_path_;
    list_cache_.clear();
    if (std::filesystem::exists(full_path) && std::filesystem::is_directory(full_path)) {
      auto now = get_unix_timestamp();
      for (const auto &entry : std::filesystem::directory_iterator(full_path)) {
        if (entry.is_directory()) {
          InstanceId inst_name = entry.path().filename();
          auto opt = get(inst_name, "_timestamp_");
          if (opt.has_value()) {
            try {
              auto last_report = std::stol(opt.value());
              const auto tolerate_s = env_fsnaming_tolerate_interval_s();
              bool good_d = last_report >= now || now - last_report < tolerate_s;
              if (good_d) {
                list_cache_.push_back(inst_name);
              }
            } catch (const std::exception & e) {
              LOG(WARNING) << "invalid timestamp: " << opt.value() << ", " << e.what();
            }
          }
        }
      }
    }
    timer_ = TimeWatch();
  }
  return list_cache_;
}

void FileSysNaming::write_file(const std::string &path, const std::string &content) {
  std::string full_path = instance_path_ / path;

  // 先写入临时文件, 之后通过 rename 原子性机制确保其他进程不会看到 full_path 中间状态.
  // 目前在 XPU EAS 环境中观测到, prefill 读取 D _timestamp_ 文件时, 读取到空.
  // pre-cxx11 abi 下 std::string .data() 返回的指针有效性存疑, 这里使用 vector 更安全.
  auto temp_path = std::vector<char>(full_path.begin(), full_path.end());
  // .tmpXXXXXX
  temp_path.emplace_back('.');
  temp_path.emplace_back('t');
  temp_path.emplace_back('m');
  temp_path.emplace_back('p');
  temp_path.emplace_back('X');
  temp_path.emplace_back('X');
  temp_path.emplace_back('X');
  temp_path.emplace_back('X');
  temp_path.emplace_back('X');
  temp_path.emplace_back('X');
  temp_path.emplace_back('\0');
  int fd = mkstemp(temp_path.data());
  if (fd == -1) {
    throw std::runtime_error("FileSysNaming.write_file: mkstemp failed. err=" + std::string(strerror(errno)));
  }
  ::close(fd);

  auto tmp_out = std::ofstream(temp_path.data());
  if (!tmp_out.is_open()) {
    auto errmsg = std::string("FileSysNaming.write_file: open temp failed. path=");
    errmsg += temp_path.data();
    throw std::runtime_error(errmsg);
  }
  tmp_out << content << std::endl;
  tmp_out.flush();
  tmp_out.close();

  int sysret = rename(temp_path.data(), full_path.c_str());
  if (sysret != 0) {
    throw std::runtime_error("FileSysNaming.write_file: rename failed. err=" + std::string(strerror(errno)));
  }

  LOG(INFO) << "KVT file naming: store key:[" << full_path << ", " << content << "];";
  return ;
}
void FileSysNaming::create_dir(const std::string &path) {
  std::filesystem::create_directory(path);
}
}
