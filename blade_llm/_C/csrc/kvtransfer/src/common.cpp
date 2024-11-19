#include "common.h"
#include <cstring>
#include <cassert>
#include <endian.h>
#include <stdexcept>
#include <sstream>
#include "utils/base64.h"

namespace blade_llm {

std::vector<uint8_t> WorkerInfo::to_bytes() const {
  uint32_t tmp[10];
  tmp[0] = htobe32(worker_id);
  tmp[1] = htobe32(tp_size);
  tmp[2] = htobe32(worker_tp_rank);
  tmp[3] = htobe32(block_size);
  tmp[4] = htobe32(token_size);
  tmp[5] = htobe32(layer_num_blocks);
  tmp[6] = htobe32(num_layers);
  tmp[7] = htobe32(transfer_protocols);
  tmp[8] = htobe32(addr.size());
  tmp[9] = htobe32(other_info.size());
  auto length = sizeof(uint32_t) * 10 + sizeof(InstanceId) + addr.size() + other_info.size();
  std::vector<uint8_t> ret(length);
  auto dst = ret.data();
  // TODO : string instance id
  auto inst_id_t = htobe64(inst_id);
  memcpy(dst, &inst_id_t, sizeof(inst_id_t));
  dst += sizeof(inst_id_t);
  memcpy(dst, tmp, sizeof(uint32_t) * 10);
  dst += sizeof(uint32_t) * 10;
  if (!addr.empty()) {
    memcpy(dst, addr.data(), addr.size());
    dst += addr.size();
  }
  if (!other_info.empty()) {
    memcpy(dst, other_info.data(), other_info.size());
  }
  return ret;
}
WorkerInfo WorkerInfo::from_bytes(const unsigned char *src, size_t length) {
  auto at_least_size = sizeof(uint32_t) * 10 + sizeof(InstanceId);
  if (length < at_least_size) {
    throw std::runtime_error("invalid worker info binary;");
  }

  WorkerInfo wi;
  uint64_t inst_id_t;
  memcpy(&inst_id_t, src, sizeof(inst_id_t));
  src += sizeof(inst_id_t);
  wi.inst_id = be64toh(inst_id_t);
  uint32_t tmp[10];
  memcpy(&tmp[0], src, sizeof(uint32_t) * 10);
  src += sizeof(uint32_t) * 10;
  wi.worker_id = be32toh(tmp[0]);
  wi.tp_size = be32toh(tmp[1]);
  wi.worker_tp_rank = be32toh(tmp[2]);
  wi.block_size = be32toh(tmp[3]);
  wi.token_size = be32toh(tmp[4]);
  wi.layer_num_blocks = be32toh(tmp[5]);
  wi.num_layers = be32toh(tmp[6]);
  wi.transfer_protocols = (uint8_t)be32toh(tmp[7]);
  auto addr_len = be32toh(tmp[8]);
  auto other_info_len = be32toh(tmp[9]);
  if (addr_len > 0) {
    at_least_size += addr_len;
    if (length < at_least_size) {
      throw std::runtime_error("invalid worker info binary;");
    }
    wi.addr.resize(addr_len);
    memcpy(wi.addr.data(), src, addr_len);
    src += addr_len;
  }

  if (other_info_len > 0) {
    at_least_size += other_info_len;
    if (length < at_least_size) {
      throw std::runtime_error("invalid worker info binary;");
    }
    wi.other_info.resize(other_info_len);
    memcpy(wi.other_info.data(), src, other_info_len);
  }
  return wi;
}
std::string WorkerInfo::to_string() const {
  std::stringstream ss;
  ss << inst_id << ","
     << worker_id << ","
     << tp_size << ","
     << worker_tp_rank << ","
     << block_size << ","
     << token_size << ","
     << layer_num_blocks << ","
     << num_layers << ","
     << (uint32_t)transfer_protocols;
  if (!addr.empty()) {
    ss << "," << addr ;
  }
  if (!other_info.empty()) {
    ss << "," << base64_encode(other_info);
  }
  return ss.str();
}
WorkerInfo WorkerInfo::from_string(const std::string &src) {
  WorkerInfo w;
  std::vector<std::string> tmp;
  tmp.reserve(11);

  size_t start = 0;
  size_t end = src.find(',');
  while (end != std::string::npos) {
    tmp.emplace_back(src.substr(start, end - start));
    start = end + 1;
    end = src.find(',', start);
  }
  tmp.emplace_back(src.substr(start));

  if (tmp.size() < 9) {
    throw std::runtime_error("invalid worker info string;");
  }

  w.inst_id = stoul(tmp[0]);
  w.worker_id = stoul(tmp[1]);
  w.tp_size = stoul(tmp[2]);
  w.worker_tp_rank = stoul(tmp[3]);
  w.block_size = stoul(tmp[4]);
  w.token_size = stoul(tmp[5]);
  w.layer_num_blocks = stoul(tmp[6]);
  w.num_layers = stoul(tmp[7]);
  w.transfer_protocols = stoul(tmp[8]);
  if (tmp.size() > 9) {
    w.addr = tmp[9];
  }
  if (tmp.size() > 10) {
    w.other_info = base64_decode(tmp[10]);
  }

  return w;
}

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