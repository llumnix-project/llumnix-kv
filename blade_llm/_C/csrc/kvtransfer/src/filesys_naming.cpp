#include "naming/filesys_naming.h"
#include "thrid_party/logging.h"
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <thread>


namespace blade_llm {

void FileSysNaming::connect(const Schema &schema, const std::string &path) {
  if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
    naming_path_ = path;
    instance_path_ = naming_path_ / inst_id;
    if (!std::filesystem::exists(instance_path_)) {
      std::filesystem::create_directory(instance_path_);
      auto time_file = instance_path_ / "_timestamp_";
      {
        std::ofstream report(time_file, std::ios::out);
        report << get_unix_timestamp() << std::endl;
        report.close();
      }
      periodic_task_ = std::thread([time_file, this]() {
        while (!this->stop_.load(std::memory_order_relaxed)) {
          std::this_thread::sleep_for(std::chrono::seconds(3));
          std::ofstream report(time_file, std::ios::out | std::ios::trunc);
          report << get_unix_timestamp() << std::endl;
          report.close();
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
              if (last_report >= now || now - last_report < 6) {
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
  auto full_path = instance_path_ / path;
  auto mode = std::ios::out;
  if (std::filesystem::exists(full_path)) {
    mode |= std::ios::trunc;
  }
  std::ofstream out(full_path, mode);
  if (out.is_open()) {
    out << content << std::endl;
    out.flush();
    out.close();
    LOG(INFO) << "KVT file naming: store key:[" << full_path << ", " << content << "];";
  } else {
    throw std::runtime_error("failed to open file: " + full_path.string());
  }
}
void FileSysNaming::create_dir(const std::string &path) {
  std::filesystem::create_directory(path);
}
}
