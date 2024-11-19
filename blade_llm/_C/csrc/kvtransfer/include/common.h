
#ifndef KVTRANSFER_INCLUDE_COMMON_H_
#define KVTRANSFER_INCLUDE_COMMON_H_
#pragma once

#include <cstdint>
#include <atomic>
#include <string>
#include <vector>

#define MAX_OTHER_INFO_LEN (8192)
#define MAX_ADDRESS_LEN (64)
#define INVALID_INST_WORKER_ID (UINT32_MAX)
#define KB (1ULL << 10ULL) // 1KB

namespace blade_llm {

class noncopyable {
 public:
  noncopyable(const noncopyable &) = delete;
  void operator=(const noncopyable &) = delete;

 protected:
  noncopyable() = default;
  ~noncopyable() = default;
};

typedef uint64_t InstanceId;
typedef std::string InstanceName;
typedef uint32_t WorkerId;
typedef std::string RequestId;

struct WorkerInfo {
  InstanceId inst_id;
  WorkerId worker_id;
  uint32_t tp_size;
  uint32_t worker_tp_rank;
  uint32_t block_size;
  uint32_t token_size;
  uint32_t layer_num_blocks{1};
  uint32_t num_layers{1};
  uint8_t transfer_protocols{0};
  std::string addr;
  std::vector<uint8_t> other_info;

  WorkerInfo() :
      inst_id(INVALID_INST_WORKER_ID),
      worker_id(INVALID_INST_WORKER_ID),
      tp_size(0),
      worker_tp_rank(0),
      block_size(0),
      token_size(0) {};

  WorkerInfo(const InstanceId &i_id, const WorkerId &w_id) :
      inst_id(i_id),
      worker_id(w_id),
      tp_size(1),
      worker_tp_rank(0),
      block_size(16 * KB),
      token_size(KB) {};

  WorkerInfo(InstanceId inst_id,
             WorkerId worker_id,
             uint32_t tp_size,
             uint32_t worker_tp_rank,
             uint32_t block_size,
             uint32_t token_size,
             uint32_t layer_num_blocks,
             uint32_t num_layers,
             uint32_t protocols) :
      inst_id(inst_id),
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

class RequestInfo {
 public:
  const InstanceId dst_inst_id;
  const WorkerId dst_worker_id;
  const RequestId req_id;

  RequestInfo(InstanceId dst_inst_id,
              WorkerId dst_worker_id,
              const RequestId &req_id,
              const std::vector<uint32_t> &src_blocks,
              const std::vector<uint32_t> &dst_blocks) :
      dst_inst_id(dst_inst_id),
      dst_worker_id(dst_worker_id),
      is_all_transferred_(false),
      req_id(req_id),
      src_blocks_(src_blocks),
      dst_blocks_(dst_blocks) {};
  RequestInfo(RequestInfo &&other) noexcept:
      dst_inst_id(other.dst_inst_id),
      dst_worker_id(other.dst_worker_id),
      req_id(other.req_id),
      src_blocks_(other.src_blocks_),
      dst_blocks_(other.dst_blocks_),
      seen_tokens_(other.seen_tokens_),
      new_tokens_(other.new_tokens_),
      reach_last_token_(other.reach_last_token_) {
    is_all_transferred_ = other.is_all_transferred_.load(std::memory_order_relaxed);
  }
  RequestInfo &set_seen_tokens(uint32_t s_token);
  RequestInfo &add_new_tokens(uint32_t n_token, bool has_last);

  void clear_new_tokens() const;
  void set_transfer_done() const;
  [[nodiscard]] const std::vector<uint32_t> &src_blocks() const;
  [[nodiscard]] const std::vector<uint32_t> &dst_blocks() const;
  [[nodiscard]] uint32_t seen_tokens() const;
  [[nodiscard]] uint32_t new_tokens() const;
  [[nodiscard]] bool reach_last_token() const;
  [[nodiscard]] bool is_all_transferred() const;

 private:
  uint32_t seen_tokens_{0};
  uint32_t new_tokens_{0};
  bool reach_last_token_{false};
  std::vector<uint32_t> src_blocks_;
  std::vector<uint32_t> dst_blocks_;
  std::atomic_bool is_all_transferred_{false};
};
} // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_COMMON_H_
