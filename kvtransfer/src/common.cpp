#include "common.h"
#include <cstring>
#include <endian.h>
#include <stdexcept>
#include <sstream>
#include "utils/base64.h"
#include "thrid_party/logging.h"

namespace blade_llm {

static std::string join_sizes(const std::vector<size_t>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {
      ss << '|';
    }
    ss << v[i];
  }
  return ss.str();
}

static std::vector<size_t> parse_sizes(const std::string& s) {
  std::vector<size_t> out;
  size_t start = 0;
  while (true) {
    auto pos = s.find('|', start);
    auto token = (pos == std::string::npos)
                   ? s.substr(start)
                   : s.substr(start, pos - start);
    if (!token.empty())
      out.push_back(std::stoul(token));
    if (pos == std::string::npos) break;
    start = pos + 1;
  }
  return out;
}

// TODO(llx): now only support one block_size/token_size
std::vector<uint8_t> WorkerInfo::to_bytes() const {
  RTASSERT(!block_sizes.empty());
  RTASSERT(!token_sizes.empty());
  auto block_size = block_sizes[0];
  auto token_size = token_sizes[0];
  RTASSERT_EQ(block_size, uint32_t(block_size));
  RTASSERT_EQ(token_size, uint32_t(token_size));
  uint32_t tmp[11];
  tmp[0] = htobe32(worker_id);
  tmp[1] = htobe32(tp_size);
  tmp[2] = htobe32(worker_tp_rank);
  // use uint32_t to change to bytes
  tmp[3] = htobe32(static_cast<uint32_t>(block_size));
  tmp[4] = htobe32(static_cast<uint32_t>(token_size));
  tmp[5] = htobe32(layer_num_blocks);
  tmp[6] = htobe32(num_layers);
  tmp[7] = htobe32(transfer_protocols);
  tmp[8] = htobe32(inst_id.size());
  if (inst_id.size() == 0 || inst_id.size() > MAX_INSTANCE_NAME_LEN) {
    throw std::runtime_error("invalid worker instance name with size: " + std::to_string(inst_id.size()));
  }
  tmp[9] = htobe32(addr.size());

  if (other_info.size() > MAX_OTHER_INFO_LEN) {
    throw std::runtime_error("invalid worker info, too large other info;");
  }
  tmp[10] = htobe32(other_info.size());

  auto length = sizeof(uint32_t) * 11 + inst_id.size() + addr.size() + other_info.size();
  std::vector<uint8_t> ret(length);
  auto dst = ret.data();
  memcpy(dst, tmp, sizeof(uint32_t) * 11);
  dst += sizeof(uint32_t) * 11;

  memcpy(dst, inst_id.data(), inst_id.size());
  dst += inst_id.size();
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
  auto at_least_size = sizeof(uint32_t) * 11 + 1;
  if (length < at_least_size) {
    throw std::runtime_error("invalid worker info binary;");
  }

  WorkerInfo wi;
  uint32_t tmp[11];
  memcpy(&tmp[0], src, sizeof(uint32_t) * 11);
  src += sizeof(uint32_t) * 11;

  wi.worker_id = be32toh(tmp[0]);
  wi.tp_size = be32toh(tmp[1]);
  wi.worker_tp_rank = be32toh(tmp[2]);

  wi.block_sizes.clear();
  wi.block_sizes.push_back(be32toh(tmp[3]));
  wi.token_sizes.clear();
  wi.token_sizes.push_back(be32toh(tmp[4]));

  wi.layer_num_blocks = be32toh(tmp[5]);
  wi.num_layers = be32toh(tmp[6]);
  wi.transfer_protocols = (uint8_t)be32toh(tmp[7]);
  auto inst_name_len = be32toh(tmp[8]);
  auto addr_len = be32toh(tmp[9]);
  auto other_info_len = be32toh(tmp[10]);

  if (inst_name_len > 0 && inst_name_len < MAX_INSTANCE_NAME_LEN) {
    at_least_size += inst_name_len - 1;
    if (length < at_least_size) {
      throw std::runtime_error("invalid worker info binary;");
    }
    wi.inst_id.resize(inst_name_len);
    memcpy(wi.inst_id.data(), src, inst_name_len);
    src += inst_name_len;
  } else {
    throw std::runtime_error("invalid worker binary, too large instance name;");
  }

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
    if (other_info_len > MAX_OTHER_INFO_LEN) {
      throw std::runtime_error("invalid worker binary, too large other info;");
    }
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
  if (inst_id.size() > MAX_INSTANCE_NAME_LEN) {
    throw std::runtime_error("invalid worker instance name, too large;");
  }
  std::stringstream ss;
  ss << inst_id << ","
     << worker_id << ","
     << tp_size << ","
     << worker_tp_rank << ","
     << join_sizes(block_sizes) << ","
     << join_sizes(token_sizes) << ","
     << layer_num_blocks << ","
     << num_layers << ","
     << (uint32_t)transfer_protocols;
  if (!addr.empty()) {
    if (addr.size() > MAX_ADDRESS_LEN) {
      throw std::runtime_error("invalid worker address, too large;");
    }
    ss << "," << addr ;
  }
  if (!other_info.empty()) {
    if (other_info.size() > MAX_OTHER_INFO_LEN) {
      throw std::runtime_error("invalid worker other info, too large;");
    }
    ss << "," << base64_encode(other_info);
  }
  return std::move(ss).str();
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

  w.inst_id = tmp[0];
  w.worker_id = stoul(tmp[1]);
  w.tp_size = stoul(tmp[2]);
  w.worker_tp_rank = stoul(tmp[3]);
  w.block_sizes = parse_sizes(tmp[4]);
  w.token_sizes = parse_sizes(tmp[5]);
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
}
