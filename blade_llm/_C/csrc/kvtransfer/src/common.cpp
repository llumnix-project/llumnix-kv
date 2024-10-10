
#include "common.h"

namespace blade_llm {
RequestInfo &RequestInfo::set_seen_tokens(uint32_t s_token) {
  seen_tokens_ = s_token;
  return *this;
}

RequestInfo &RequestInfo::add_new_tokens(uint32_t n_token, bool has_last) {
  new_tokens_ += n_token;
  reach_last_token_ = has_last;
  return *this;
}

void RequestInfo::clear_new_tokens() const {
  const_cast<RequestInfo *>(this)->new_tokens_ = 0;
}

void RequestInfo::set_transfer_done() const {
  const_cast<RequestInfo *>(this)->is_all_transferred_.store(true, std::memory_order_release);
}

const std::vector<uint32_t> &RequestInfo::src_blocks() const {
  return src_blocks_;
}

const std::vector<uint32_t> &RequestInfo::dst_blocks() const {
  return dst_blocks_;
}

uint32_t RequestInfo::seen_tokens() const {
  return seen_tokens_;
}

uint32_t RequestInfo::new_tokens() const {
  return new_tokens_;
}

bool RequestInfo::reach_last_token() const {
  return reach_last_token_;
}

bool RequestInfo::is_all_transferred() const {
  return reach_last_token_ && is_all_transferred_.load(std::memory_order_relaxed);
}
}