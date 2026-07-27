#include "common.h"
#include <cstring>
#include <stdexcept>
#include <sstream>
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
// Helper: a serialized token is numeric iff it is non-empty and contains
// only ASCII digits. Used by from_string to distinguish a trailing IPv4/IPv6
// address from a numeric attn_pack_size in older/newer payloads.
static bool is_numeric_token(const std::string& s) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

std::string WorkerInfo::to_string() const {
  if (inst_id.size() > MAX_INSTANCE_NAME_LEN) {
    throw std::runtime_error("invalid worker instance name, too large;");
  }
  std::stringstream ss;
  ss << inst_id << ","
     << worker_id << ","
     << engine_tp_size << ","
     << worker_tp_rank << ","
     << join_sizes(block_sizes) << ","
     << join_sizes(token_sizes) << ","
     << layer_num_blocks << ","
     << num_layers << ","
     << (uint32_t)transfer_protocols << ","
     << attn_kernel_blk_ntpb << ","
     << indexer_blk_ntpb << ","
     << attn_pack_size << ","
     << kda_page_stride;
  if (!addr.empty()) {
    if (addr.size() > MAX_ADDRESS_LEN) {
      throw std::runtime_error("invalid worker address, too large;");
    }
    ss << "," << addr ;
  }
  return ss.str();
}

WorkerInfo WorkerInfo::from_string(const std::string &src) {
  WorkerInfo w;
  std::vector<std::string> tmp;
  tmp.reserve(14);

  size_t start = 0;
  size_t end = src.find(',');
  while (end != std::string::npos) {
    tmp.emplace_back(src.substr(start, end - start));
    start = end + 1;
    end = src.find(',', start);
  }
  tmp.emplace_back(src.substr(start));

  // Backward compatibility:
  //   - 11 fields: legacy payload without attn_pack_size and without addr
  //   - 12 fields: either legacy {..., indexer_blk_ntpb, addr} OR new
  //                {..., indexer_blk_ntpb, attn_pack_size}
  //   - 13 fields: either {..., attn_pack_size, addr} OR
  //                {..., attn_pack_size, kda_page_stride}
  //   - 14 fields: {..., attn_pack_size, kda_page_stride, addr}
  if (tmp.size() < 11 || tmp.size() > 14) {
    throw std::runtime_error("invalid worker info string;");
  }

  w.inst_id = tmp[0];
  w.worker_id = stoul(tmp[1]);
  // wire slot[2] is the engine_tp_size (kept at the same byte position as
  // the legacy "kvt_tp_size" slot since at advertisement time those two
  // values are identical -- old senders wrote engine_tp_size into this slot).
  w.engine_tp_size = stoul(tmp[2]);
  w.worker_tp_rank = stoul(tmp[3]);
  w.block_sizes = parse_sizes(tmp[4]);
  w.token_sizes = parse_sizes(tmp[5]);
  w.layer_num_blocks = stoul(tmp[6]);
  w.num_layers = stoul(tmp[7]);
  w.transfer_protocols = stoul(tmp[8]);
  w.attn_kernel_blk_ntpb = stoul(tmp[9]);
  w.indexer_blk_ntpb = stoul(tmp[10]);
  // Default for missing attn_pack_size or older payloads.
  w.attn_pack_size = 1;
  w.kda_page_stride = 0;
  if (tmp.size() == 11) {
    // legacy without addr, nothing more to parse
  } else if (tmp.size() == 12) {
    // Distinguish legacy {addr} payload from new {attn_pack_size} payload.
    if (is_numeric_token(tmp[11])) {
      w.attn_pack_size = stoul(tmp[11]);
    } else {
      w.addr = tmp[11];
    }
  } else if (tmp.size() == 13) {
    w.attn_pack_size = stoul(tmp[11]);
    if (is_numeric_token(tmp[12])) {
      w.kda_page_stride = stoull(tmp[12]);
    } else {
      w.addr = tmp[12];
    }
  } else {
    // 14 fields: attn_pack_size, kda_page_stride, then addr
    w.attn_pack_size = stoul(tmp[11]);
    w.kda_page_stride = stoull(tmp[12]);
    w.addr = tmp[13];
  }
  if (w.attn_pack_size == 0) {
    w.attn_pack_size = 1;
  }
  return w;
}
}
