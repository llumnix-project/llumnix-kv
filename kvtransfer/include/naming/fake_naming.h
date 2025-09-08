#ifndef KVTRANSFER_INCLUDE_NAMING_FAKE_NAMING_H_
#define KVTRANSFER_INCLUDE_NAMING_FAKE_NAMING_H_

#include <cassert>
#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <optional>
#include "naming.h"
#include "utils/http_helper.h"
#include "utils/timer.h"
namespace blade_llm {
class FakeNamingWorkerClient : public INamingWorkerClient {
    WorkerInfo dst_info_;
  public:
    FakeNamingWorkerClient(const std::string& dst_info_str): dst_info_(WorkerInfo::from_string(dst_info_str)) {}
  
    void register_worker(const WorkerInfo &worker_info) override {
      throw std::runtime_error("fake naming does not support register worker");
    }
  
    std::optional<WorkerInfo> get_worker_info(const InstanceId &, WorkerId) override {
        return this->dst_info_;
    }
  };
}
#endif