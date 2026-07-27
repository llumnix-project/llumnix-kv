#ifndef KVTRANSFER_INCLUDE_CLIENT_H_
#define KVTRANSFER_INCLUDE_CLIENT_H_

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <queue>
#include <mutex>
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

struct ReqMeta {
  InstanceId dst_inst;
  WorkerId dst_worker = 0;
  RequestId reqid;
  uint32_t seen_tokens = 0;
  uint32_t new_tokens = 0;
  BlockIds src_block_ids;
  BlockIds dst_block_ids;
  std::optional<WorkerInfo> dst_worker_info;
};

// Metadata for a specific step/substep.
struct StepMetas {
  size_t stepid;
  uint32_t substepid;
  std::vector<ReqMeta> metas;
};

// StepTasks contains substep-related information.
struct StepTasks {
  // Maps InstanceId to [(WorkerId, BatchSendTask)].
  std::unordered_map<InstanceId, std::vector<std::pair<WorkerId, BatchSendTask>>> tasks;
  // Substep identifier.
  size_t stepid = 0;
  uint32_t substepid = 0;
  // sub_send_ts: submission time of the substep task.
  const Timepoint send_ts;

public:
  StepTasks() : send_ts(SteadyClock::now()) {}
  explicit StepTasks(Timepoint ts) : send_ts(ts) {}

  StepTasks(StepTasks&&) = default;
  StepTasks(const StepTasks&) = delete;

  bool empty() const noexcept { return tasks.empty(); }
  void swap(StepTasks& other) noexcept {
    tasks.swap(other.tasks);
    std::swap(stepid, other.stepid);
    std::swap(substepid, other.substepid);
    // send_ts is const, cannot swap
  }
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

  // ReqMeta is moved.
  // thread safe
  void start_req_send(std::vector<ReqMeta>& metas, size_t stepid, uint32_t substepid);

  void submit_req_send(InstanceId dst_inst,
                       WorkerId dst_worker,
                       RequestId reqid,
                       uint32_t seen_tokens,
                       uint32_t new_tokens,
                       bool has_last_token,
                       BlockIds src_block_ids,
                       BlockIds dst_block_ids,
                       std::optional<std::string> dst_worker_info,
                       size_t stepid,
                       uint32_t substepid);

  void submit_delta_send(const RequestId &,
                         uint32_t seen_tokens,
                         uint32_t new_tokens,
                         bool has_last_token,
                         size_t stepid,
                         uint32_t substepid);

  // start_send begins a step.
  // stepid: step identifier supplied by the caller.
  // sched_tokens: number of tokens scheduled in this step, used for optimization.
  //   - sched_tokens == 0: skip Step/StepGuard creation and return EMPTY_STEP_ID.
  //   - sched_tokens > 0: create Step/StepGuard even if targets_tasks_buf_ is empty.
  size_t start_send(size_t stepid, size_t sched_tokens);

  void notify_event_record(size_t step_id);
  void flush_send(size_t step_id);

  // start_send_substep appends send tasks to the current step; it is called
  // from the disagg worker thread.
  // metas: nonfreeze_metas with has_freeze=false; has_last_token may be false.
  void start_send_substep(size_t stepid, uint32_t substepid, std::vector<ReqMeta>& metas);

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

  // All operation state transitions happen in mgr_thd_.
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
                   const std::optional<WorkerInfo>& worker_info);

    Target* create_or_get(const RequestInfo& req);

    Target* get(const InstanceId& inst_id, WorkerId worker_id);

    void do_submit(std::shared_ptr<Step>& step, StepTasks& tasks);

    TargetMgr(TargetMgr&&) = delete;
    TargetMgr(const TargetMgr&) = delete;
  };
 private:
  bool auto_connect_{false};
  std::atomic<size_t> fast_step_id_{UINT64_MAX};
  static_assert(std::atomic<size_t>::is_always_lock_free);
  std::unique_ptr<Context> ctx_;
  std::unordered_map<RequestId, std::vector<std::shared_ptr<RequestInfo>>> reqs_;
  // Temporarily stores send tasks created by submit_req_send/submit_delta_send.
  // start_send() clears this field. Semantically this is the substep with
  // substepid=0.
  StepTasks targets_tasks_buf_;
  ThreadPool single_thd_;
  TargetMgr mgr_;

  // ========== Thread coordination state ==========
  // lock protects the following fields.
  mutable std::mutex coord_lock_;
  size_t coord_step_id_{0};
  std::shared_ptr<StepGuard> last_step_guard_;
  std::vector<StepMetas> pending_step_metas_;
};

}
#endif //KVTRANSFER_INCLUDE_CLIENT_H_
