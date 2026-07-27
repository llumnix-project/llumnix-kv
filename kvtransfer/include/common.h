
#ifndef KVTRANSFER_INCLUDE_COMMON_H_
#define KVTRANSFER_INCLUDE_COMMON_H_
#pragma once

#include <cstdint>
#include <atomic>
#include <bitset>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <unistd.h>
#include <memory>
#include <optional>

#include "thrid_party/logging.h"
#include "envcfg.h"

#define MAX_ADDRESS_LEN (64)
#define INVALID_INST_WORKER_ID (UINT32_MAX)
#define MAX_INSTANCE_NAME_LEN (256)
#define KB (1ULL << 10ULL) // 1KB

class FdGuard {
  int const fd_;
 public:
  FdGuard(int fd) noexcept: fd_(fd) {}
  ~FdGuard() {
    ::close(this->fd_);
  }
  int fd() const noexcept {
    return this->fd_;
  }
};


#define RTASSERT(expr) do { \
    if (!(expr)) { \
        std::cerr << "AssertFailed: loc=" << __FILE__ << ':' << __LINE__ \
                  << ";errno=" << errno \
                  << ";expr=" << #expr << std::endl; \
        abort(); \
    } \
} while (0)

#define RTASSERT_EQ(left, right) do { \
    const auto& left_val = (left);  \
    const auto& right_val = (right); \
    if (left_val != right_val) { \
        std::cerr << "AssertFailed: loc=" << __FILE__ << ':' << __LINE__ \
           << ";errno=" << errno \
           << ";left=" << #left << ";leftval=" << left_val \
           << ";right=" << #right << ";rightval=" << right_val << std::endl; \
        abort(); \
    } \
} while (0)


#define RTCHECK_EQ(left, right) do { \
    const auto& left_val = (left);  \
    const auto& right_val = (right); \
    if (left_val != right_val) { \
        std::stringstream ss; \
        ss << "AssertFailed: loc=" << __FILE__ << ':' << __LINE__ \
           << ";errno=" << errno \
           << ";left=" << #left << ";leftval=" << left_val \
           << ";right=" << #right << ";rightval=" << right_val; \
        throw std::runtime_error(std::move(ss).str()); \
    } \
} while (0)


#define RTCHECK(expr) do { \
    if (!(expr)) { \
        std::stringstream ss; \
        ss << "AssertFailed: loc=" << __FILE__ << ':' << __LINE__ \
           << ";errno=" << errno \
           << ";expr=" << #expr; \
        throw std::runtime_error(std::move(ss).str()); \
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
using BlockIds = std::vector<std::vector<uint32_t>>;

struct LayerInfo {
  size_t token_size;
  size_t block_size;
  uint64_t layer_addr;

  LayerInfo(
    size_t token_size_,
    size_t block_size_,
    uint64_t layer_addr_) :
  token_size(token_size_),
  block_size(block_size_),
  layer_addr(layer_addr_) {}
};

struct WorkerInfo {
  InstanceId inst_id;
  WorkerId worker_id;
  // engine_tp_size: the engine's real tensor-parallel size for this worker.
  // Serialized across P/D so both sides agree on the model's TP shape.
  // The per-destination "effective" TP used for KV transfer (after
  // valid_ranks filtering by num_kv_heads) is NOT stored here -- it lives
  // in the sender's TaskContext and is passed to parse_block as the
  // kvt_tp_size / kvt_tp_rank function parameters.
  uint32_t engine_tp_size{0};
  uint32_t worker_tp_rank;
  std::vector<size_t> block_sizes;
  std::vector<size_t> token_sizes;
  uint32_t layer_num_blocks{1};
  uint32_t num_layers{1};
  uint8_t transfer_protocols{0};
  uint32_t attn_kernel_blk_ntpb{0};
  uint32_t indexer_blk_ntpb{0};
  // Number of consecutive FullAttention layers packed into one physical
  // KV-cache page in hybrid models (mamba+attn). Always 1 unless the engine
  // is configured with cache_config.attn_pack_size > 1. Kept in the
  // serialized form so the destination side can adapt block offsets in
  // P>D parsing.
  uint32_t attn_pack_size{1};
  std::string addr; // ip

  // local-only, not serialized
  int num_kv_heads{-1};
  uint32_t num_gdn_layers{3};
  uint32_t hybrid_indexer_token_size{0};
  uint32_t gdn_conv_elem_size{1};
  uint32_t gdn_ssm_elem_size{1};
  std::vector<size_t> conv_state_shape;
  std::vector<size_t> ssm_state_shape;
  std::vector<size_t> gdn_conv_channel_dims;
  uint32_t num_ple_layers{0};
  uint32_t ple_block_group{0};
  uint32_t ple_conv_elem_size{1};
  std::vector<size_t> ple_conv_state_shape;
  std::vector<size_t> ple_conv_channel_dims;

  // KDA reuses the GDN state metadata above because both page layouts are
  // [conv][recurrent][padding]. This KDA-only flag is the one extra piece of
  // metadata needed for community vLLM's optional DS conv layout. Existing
  // GDN parsing remains SD-only and does not read this field.
  bool kda_conv_dim_first{false};

  // Real per-token byte size of the rank-replicated MLA cache stored at the
  // beginning of a KDA-sized physical page. Local-only, not serialized.
  uint32_t hybrid_attn_token_size{0};

  // Physical byte stride between KDA blocks. Kimi K3 may expose the same
  // backing allocation through smaller logical MLA blocks, so block_sizes[0]
  // is not necessarily the correct stride for recurrent-state block IDs.
  // Serialized so a P sender can address the D allocation correctly when
  // P/D tensor-parallel sizes differ.
  uint64_t kda_page_stride{0};

  // todo
  WorkerInfo() :
      worker_id(INVALID_INST_WORKER_ID),
      worker_tp_rank(0),
      block_sizes({}),
      token_sizes({}) {};

  WorkerInfo(const InstanceId& id, const WorkerId &w_id) :
      inst_id(id),
      worker_id(w_id),
      engine_tp_size(1),
      worker_tp_rank(0),
      block_sizes({16 * KB}),
      token_sizes({KB}) {};

  WorkerInfo(InstanceId&& inst_id_,
             WorkerId worker_id_,
             uint32_t tp_size_,
             uint32_t worker_tp_rank_,
             std::vector<size_t> block_sizes_,
             std::vector<size_t> token_sizes_,
             uint32_t layer_num_blocks_,
             uint32_t num_layers_,
             uint32_t protocols) :
      inst_id(std::move(inst_id_)),
      worker_id(worker_id_),
      engine_tp_size(tp_size_),
      worker_tp_rank(worker_tp_rank_),
      block_sizes(block_sizes_),
      token_sizes(token_sizes_),
      layer_num_blocks(layer_num_blocks_),
      num_layers(num_layers_),
      transfer_protocols(protocols) {};

  static WorkerInfo from_string(const std::string& src);
  [[nodiscard]] std::string to_string() const;
};


static constexpr inline int ReqStateInprocess = 0;
static constexpr inline int ReqStateOk = 1;
static constexpr inline int ReqStateFailed = 2;
enum class ReqState : int {
  INPROCESS = ReqStateInprocess,
  // final state
  OK = ReqStateOk,
  FAILED = ReqStateFailed,
};
class RequestInfo;
class ReqSendTask {
  const std::shared_ptr<RequestInfo> req_ = nullptr;
public:
  const uint32_t seen_tokens{0};
  const uint32_t new_tokens{0};  // always gt 0.
  const bool reach_last_token{false};
public:
  ReqSendTask(std::shared_ptr<RequestInfo> req, uint32_t seen_, uint32_t new_, bool last_):
    req_(std::move(req)),
    seen_tokens(seen_),
    new_tokens(new_),
    reach_last_token(last_) {}

  void set_state(ReqState) const noexcept;
  ReqState state() const noexcept;

  const auto& req_id() const noexcept;
  const auto& dst_blocks() const noexcept;
  const auto& src_blocks() const noexcept;
  auto dst_inst_id() const noexcept;
  auto dst_worker_id() const noexcept;
  uint32_t end_tokens() const noexcept {
    return this->seen_tokens + this->new_tokens;
  }

  const RequestInfo& req() const noexcept {
    return *req_;
  }

  // FOR GTEST
  bool operator==(const ReqSendTask& other) const {
    return req_ == other.req_ &&
           seen_tokens == other.seen_tokens &&
           new_tokens == other.new_tokens &&
           reach_last_token == other.reach_last_token;
  }

  friend inline std::ostream& operator<<(std::ostream& os, const ReqSendTask& task);
};

// RequestInfo may only be used on the Python main thread.
class RequestInfo {
 public:
  const InstanceId dst_inst_id;
  const WorkerId dst_worker_id;
  const std::optional<WorkerInfo> dst_worker_info;
  const RequestId req_id;
  // src_blocks/dst_blocks are passed in from vLLM.
  // The outer vector represents KV cache groups, each with its own block IDs.
  // Different cache tensors must use the same block IDs. QSA shares tensors,
  // so each layer contains only one cache tensor.
  const BlockIds src_blocks;
  const BlockIds dst_blocks;
 private:
  static_assert(std::atomic<ReqState>::is_always_lock_free);
  mutable std::atomic<ReqState> state_{ReqState::INPROCESS};
#ifndef NDEBUG
  uint32_t last_seen_ = UINT32_MAX;
  bool has_last_ = false;
#endif

 public:
  // only for test
  RequestInfo(InstanceId dst_inst_id_,
              WorkerId dst_worker_id_,
              RequestId req_id_,
              BlockIds src_blocks_,
              BlockIds dst_blocks_):
      RequestInfo(std::move(dst_inst_id_), dst_worker_id_,
                  std::nullopt,
                  std::move(req_id_),
                  std::move(src_blocks_),
                  std::move(dst_blocks_)) {};

  RequestInfo(InstanceId dst_inst_id_,
              WorkerId dst_worker_id_,
              std::optional<WorkerInfo> dst_worker_info_,
              RequestId req_id_,
              BlockIds src_blocks_,
              BlockIds dst_blocks_):
      dst_inst_id(std::move(dst_inst_id_)),
      dst_worker_id(dst_worker_id_),
      dst_worker_info(std::move(dst_worker_info_)),
      req_id(std::move(req_id_)),
      src_blocks(std::move(src_blocks_)),
      dst_blocks(std::move(dst_blocks_)) {};

  RequestInfo(const RequestInfo&) = delete;
  RequestInfo(RequestInfo&&) = delete;

  void update_send(uint32_t seen, uint32_t new_tokens, bool has_last) {
#ifndef NDEBUG
    assert(!this->has_last_);
    if (this->last_seen_ == UINT32_MAX) {
      this->last_seen_ = seen;
    }
    assert(this->last_seen_ == seen);
    // assert(new_tokens >= 0);
    this->last_seen_ += new_tokens;
    this->has_last_ = has_last;
#endif
    return ;
  }

  ReqState state() const noexcept {
    return this->state_.load(std::memory_order_acquire);
  }

 private:
  friend class ReqSendTask;

  void set_state(ReqState state) const noexcept {
    assert(state != ReqState::INPROCESS);
    this->state_.store(state, std::memory_order_release);
  }
};


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

inline void ReqSendTask::set_state(ReqState s) const noexcept {
  return this->req_->set_state(s);
}

inline ReqState ReqSendTask::state() const noexcept {
  return this->req_->state();
}

inline static void print_block_ids(std::ostream& os, const BlockIds& bids) {
  os << "[";
  for (size_t i = 0; i < bids.size(); ++i) {
    os << "[";
    for (size_t j = 0; j < bids[i].size(); ++j) {
      os << bids[i][j];
      if (j + 1 < bids[i].size()) os << ',';
    }
    os << "]";
    if (i + 1 < bids.size()) os << ',';
  }
  os << "]";
}

inline std::ostream& operator<<(std::ostream& os, const ReqSendTask& task) {
  os << "ReqSendTask(seen_tokens: " << task.seen_tokens
      << ", new_tokens: " << task.new_tokens
      << ", reach_last_token: " << std::boolalpha << task.reach_last_token
      << ", reqid: " << task.req_id()
      << ", srcblks: ";
  print_block_ids(os, task.src_blocks());
  os << ", dstblks: ";
  print_block_ids(os, task.dst_blocks());
  os << ", req_: " << task.req_.get() << ")";
  return os;
}


} // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_COMMON_H_
