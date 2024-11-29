
#ifndef KVTRANSFER_INCLUDE_COMMON_H_
#define KVTRANSFER_INCLUDE_COMMON_H_
#pragma once

#include <cstdint>
#include <atomic>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>

#define MAX_OTHER_INFO_LEN (8192)
#define MAX_ADDRESS_LEN (64)
#define INVALID_INST_WORKER_ID (UINT32_MAX)
#define MAX_INSTANCE_NAME_LEN (256)
#define KB (1ULL << 10ULL) // 1KB


#define RTCHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "Runtime error: Assertion failed in %s on line %d: %s\n", __FILE__, __LINE__, #expr); \
        abort(); \
    } \
} while (0)


namespace blade_llm {

class noncopyable {
 public:
  noncopyable(const noncopyable &) = delete;
  void operator=(const noncopyable &) = delete;

 protected:
  noncopyable() = default;
  ~noncopyable() = default;
};

typedef std::string InstanceId;
typedef uint32_t WorkerId;
typedef std::string RequestId;

struct WorkerInfo {
  WorkerId worker_id;
  uint32_t tp_size;
  uint32_t worker_tp_rank;
  uint32_t block_size;
  uint32_t token_size;
  uint32_t layer_num_blocks{1};
  uint32_t num_layers{1};
  uint8_t transfer_protocols{0};
  InstanceId inst_id;
  std::string addr;
  std::vector<uint8_t> other_info;

  WorkerInfo() :
      worker_id(INVALID_INST_WORKER_ID),
      tp_size(0),
      worker_tp_rank(0),
      block_size(0),
      token_size(0) {};

  WorkerInfo(const InstanceId& id, const WorkerId &w_id) :
      inst_id(id),
      worker_id(w_id),
      tp_size(1),
      worker_tp_rank(0),
      block_size(16 * KB),
      token_size(KB) {};

  WorkerInfo(InstanceId&& inst_id,
             WorkerId worker_id,
             uint32_t tp_size,
             uint32_t worker_tp_rank,
             uint32_t block_size,
             uint32_t token_size,
             uint32_t layer_num_blocks,
             uint32_t num_layers,
             uint32_t protocols) :
      inst_id(std::move(inst_id)),
      worker_id(worker_id),
      tp_size(tp_size),
      worker_tp_rank(worker_tp_rank),
      block_size(block_size),
      token_size(token_size),
      layer_num_blocks(layer_num_blocks),
      num_layers(num_layers),
      transfer_protocols(protocols) {};

  static WorkerInfo from_bytes(const unsigned char *src, size_t length);
  static WorkerInfo from_string(const std::string& src);
  [[nodiscard]] std::vector<uint8_t> to_bytes() const;
  [[nodiscard]] std::string to_string() const;
};

class RequestInfo;
class ReqSendTask {
  const RequestInfo* const req_ = nullptr;
public:
  const uint32_t seen_tokens{0};
  const uint32_t new_tokens{0};  // always gt 0.
  const bool reach_last_token{false};
public:
  ReqSendTask(RequestInfo* req, uint32_t seen_, uint32_t new_, bool last_):
    req_(req),
    seen_tokens(seen_),
    new_tokens(new_),
    reach_last_token(last_) {}

  void set_transfer_done() const;

  const auto& req_id() const noexcept;
  const auto& dst_blocks() const noexcept;
  const auto& src_blocks() const noexcept;
  auto dst_inst_id() const noexcept;
  auto dst_worker_id() const noexcept;

  // FOR GTEST
  bool operator==(const ReqSendTask& other) const {
    return req_ == other.req_ &&
           seen_tokens == other.seen_tokens &&
           new_tokens == other.new_tokens &&
           reach_last_token == other.reach_last_token;
  }

  friend inline std::ostream& operator<<(std::ostream& os, const ReqSendTask& task);
};

// FOR GTEST
inline std::ostream& operator<<(std::ostream& os, const ReqSendTask& task) {
  os << "ReqSendTask(seen_tokens: " << task.seen_tokens
      << ", new_tokens: " << task.new_tokens
      << ", reach_last_token: " << std::boolalpha << task.reach_last_token
      << ", req_: " << task.req_ << ")";
  return os;
}

// RequestInfo 只能在 python main thread 使用.
class RequestInfo {
 public:
  const InstanceId dst_inst_id;
  const WorkerId dst_worker_id;
  const RequestId req_id;
  const std::vector<uint32_t> src_blocks;
  const std::vector<uint32_t> dst_blocks;
 private:
  mutable std::atomic_bool is_all_transferred_{false};
#ifndef NDEBUG
  uint32_t last_seen_ = 0;
  bool has_last_ = false;
#endif

 public:
  RequestInfo(const InstanceId& dst_inst_id_,
              WorkerId dst_worker_id_,
              RequestId req_id_,
              std::vector<uint32_t> src_blocks_,
              std::vector<uint32_t> dst_blocks_):
      dst_inst_id(dst_inst_id_),
      dst_worker_id(dst_worker_id_),
      req_id(std::move(req_id_)),
      src_blocks(std::move(src_blocks_)),
      dst_blocks(std::move(dst_blocks_)) {};

  RequestInfo(const RequestInfo&) = delete;
  RequestInfo(RequestInfo&&) = delete;

  void update_send(uint32_t seen, uint32_t new_tokens, bool has_last) {
#ifndef NDEBUG
    assert(!this->has_last_);
    assert(this->last_seen_ == seen);
    assert(new_tokens > 0);
    this->last_seen_ += new_tokens;
    this->has_last_ = has_last;
#endif
    return ;
  }

  bool is_all_transferred() const {
    return is_all_transferred_.load(std::memory_order_acquire);
  }

 private:
  friend class ReqSendTask;

  void set_transfer_done() const {
    is_all_transferred_.store(true, std::memory_order_release);
  }
};

inline void ReqSendTask::set_transfer_done() const {
  assert(this->reach_last_token);
  this->req_->set_transfer_done();
  // DO NOT ACCESS req_! THIS MAY BE FREED!
}

inline const auto& ReqSendTask::req_id() const noexcept {
  return this->req_->req_id;
}

inline const auto& ReqSendTask::dst_blocks() const noexcept {
  return this->req_->dst_blocks;
}

inline const auto& ReqSendTask::src_blocks() const noexcept {
  return this->req_->src_blocks;
}

inline auto ReqSendTask::dst_inst_id() const noexcept {
  return this->req_->dst_inst_id;
}

inline auto ReqSendTask::dst_worker_id() const noexcept {
  return this->req_->dst_worker_id;
}


} // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_COMMON_H_
