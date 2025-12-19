#ifndef KVTRANSFER_INCLUDE_CLIENT_H_
#define KVTRANSFER_INCLUDE_CLIENT_H_

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <queue>
#include "common.h"
#include "context.h"
#include "tx_stub.h"
#include "utils/semaphore.h"
#include "utils/thread_pool.h"
#include "envcfg.h"
#include "error.h"
#include <list>

namespace accl {
namespace barex {
class XThreadpool;
}
}

namespace blade_llm {

// 这里 value 长度一般是 p_info->tp_size / d_info->tp_size; 约 1/2/4.
using StepTasks = std::unordered_map<InstanceId, std::vector<std::pair<WorkerId, BatchSendTask>>>;

struct ReqMeta {
  InstanceId dst_inst;
  WorkerId dst_worker = 0;
  RequestId reqid;
  uint32_t seen_tokens = 0;
  uint32_t new_tokens = 0;
  std::vector<uint32_t> src_block_ids;
  std::vector<uint32_t> dst_block_ids;
  std::optional<std::string> dst_worker_info;
};

class KvTransferClient : public noncopyable {

 public:
  static std::unique_ptr<KvTransferClient> create(std::unique_ptr<Context> &&,
                                                  const std::vector<TransferProtocol> &,
                                                  std::unique_ptr<ISendStubFactory> &&f);

  KvTransferClient(std::unique_ptr<Context> &&, std::unique_ptr<ISendStubFactory> &&);

  void add_target(const InstanceId &,
                  const WorkerId &,
                  uint32_t start_layer,
                  uint32_t num_layers,
                  std::optional<TransferProtocol> = std::nullopt,
                  const std::optional<std::string> &dst_worker_info = std::nullopt);
  void remove_target(const InstanceId &, const WorkerId &);

  // ReqMeta 会被 moved.
  // thread safe
  void start_req_send(std::vector<ReqMeta>& metas);

  void submit_req_send(InstanceId dst_inst,
                       WorkerId dst_worker,
                       RequestId reqid,
                       uint32_t seen_tokens,
                       uint32_t new_tokens,
                       bool has_last_token,
                       std::vector<uint32_t> src_block_ids,
                       std::vector<uint32_t> dst_block_ids,
                       std::optional<std::string> dst_worker_info = std::nullopt);

  void submit_req_send(InstanceId dst_inst,
                       WorkerId dst_worker,
                       RequestId r,
                       uint32_t new_tokens,
                       bool has_last_token,
                       std::vector<uint32_t> src_block_ids,
                       std::vector<uint32_t> dst_block_ids) {
    return submit_req_send(std::move(dst_inst), dst_worker,
                           std::move(r), 0, new_tokens, has_last_token,
                           std::move(src_block_ids),
                           std::move(dst_block_ids));
  }

  void submit_delta_send(const RequestId &,
                         uint32_t seen_tokens,
                         uint32_t new_tokens,
                         bool has_last_token);

  size_t start_send();
  void notify_event_record(size_t step_id);
  void flush_send(size_t step_id);

  ReqState check_transfer_done(const RequestId &);
  Context *context() { return ctx_.get(); };
  void enable_auto_connect() { auto_connect_ = true; }

  // only for test
  size_t target_size() const noexcept {
    return this->mgr_.size();
  }

  // only for test
  void target_try_shrink(size_t cap) {
    return this->mgr_.try_shrink_for_test(cap);
  }
 private:
  struct Target {
    std::atomic<int> inflycnt {0};
    int thread_hint = 0;
    SendStub stub;
  public:
    Target(SendStub s) noexcept:
      stub(std::move(s)) {}

    SendStub release_stub() noexcept {
      SendStub tmp;
      this->stub.swap(tmp);
      return tmp;
    }
  };

  // 所有操作状态变更都发生在 mgr_thd_ 中.
  class TargetMgr {
    // OWNER: GLOBAL
    Context* const ctx_ = nullptr;
    ThreadPool mgr_thd_{1};

    std::unique_ptr<ISendStubFactory> stub_factory_;
    // head is newer~
    std::list<Target> targets_;
    std::unordered_map<InstanceId, std::vector<std::optional<std::list<Target>::iterator>>> target_map_;
    accl::barex::XThreadpool* target_thdpool_ = nullptr;
    int thread_hint_ = 0;
  public:
    TargetMgr(Context* ctx, std::unique_ptr<ISendStubFactory> s);

    // only for test
    size_t size() const noexcept {
      // thread unsafe!
      return this->targets_.size();
    }

    // thread safe!
    void submit(std::shared_ptr<Step> step, StepTasks tasks);

    void try_shrink_for_test(size_t cap) {
      return this->try_shrink(cap);
    }

  private:
    // RETURN self.targets_.end() means not found
    std::list<Target>::iterator peek(const InstanceId& inst_id, WorkerId worker_id);

    void shrink(size_t cap);

    void try_shrink(size_t cap);

    int try_pop_map(const InstanceId& inst_id, WorkerId worker_id);

    Target* create(const InstanceId& inst_id, WorkerId worker_id,
                   uint32_t start_layer, uint32_t num_layers,
                   const std::optional<std::string>& worker_info);

    Target* create_or_get(const RequestInfo& req);

    Target* get(const InstanceId& inst_id, WorkerId worker_id);

    void do_submit(std::shared_ptr<Step>& step, StepTasks& tasks);

    TargetMgr(TargetMgr&&) = delete;
    TargetMgr(const TargetMgr&) = delete;
  };
 private:
  const bool auto_remove_req_;
  bool auto_connect_{false};
  size_t step_id_{0};
  std::atomic<size_t> fast_step_id_{UINT64_MAX};
  static_assert(std::atomic<size_t>::is_always_lock_free);
  std::unique_ptr<Context> ctx_;
  std::unordered_map<RequestId, std::vector<std::shared_ptr<RequestInfo>>> reqs_;
  // 用来临时暂存 submit_req_send/submit_delta_send 创建的 send task.
  // start_send() 会清空该字段.
  StepTasks targets_tasks_buf_;
  std::shared_ptr<StepGuard> last_step_guard_;  // may be null.
  ThreadPool single_thd_;
  TargetMgr mgr_;
};
}
#endif //KVTRANSFER_INCLUDE_CLIENT_H_
