#include <mutex>
#include <algorithm>
#include "rx_stub.h"
#include "thrid_party/logging.h"

namespace blade_llm {

void ReqRecvTask::set_dst_blocks(const std::vector<uint32_t> &block_ids) {
  dst_blocks_ = block_ids;
  std::sort(dst_blocks_.begin(), dst_blocks_.end());
}

const std::vector<uint32_t> &ReqRecvTask::dst_blocks() const {
  return dst_blocks_;
}

void KvRecvStub::on_recv(const RequestId &req_id, std::vector<uint32_t> &&dst_block_ids) {
  LOG(INFO) << "KVT rx_stub: recv all kv of request(" << req_id <<
            ") from (" << src_inst_id << ":" << src_worker_id << ").";
  std::sort(dst_block_ids.begin(), dst_block_ids.end());
  std::unique_lock<std::shared_mutex> lock(task_m_);
  auto [_, ok] = recv_tasks_.try_emplace(req_id, src_inst_id, src_worker_id, req_id, std::move(dst_block_ids));
  assert(ok);
}


static bool operator==(const std::vector<uint32_t>& lhs, const std::vector<uint32_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }

  return true;
}

static void print_vec(std::ostream& os, const std::vector<uint32_t>& vec) {
  os << "[";
  for (std::size_t i = 0; i < vec.size(); ++i) {
    os << vec[i] << ',';
  }
  os << "]";
}

static std::ostream& operator<<(std::ostream& os, const std::vector<uint32_t>& vec) {
  print_vec(os, vec);
  return os;
}

static std::ostream& operator<<(std::ostream&& os, const std::vector<uint32_t>& vec) {
  print_vec(os, vec);
  return os;
}

bool KvRecvStub::check_recv_done(const RequestId &req_id, const std::vector<uint32_t> &blocks) {
  std::shared_lock<std::shared_mutex> lock(task_m_);
  auto task = recv_tasks_.find(req_id);
  if (task == recv_tasks_.end()) {
    return false;
  }
  const auto &recv_blocks = task->second.dst_blocks();
  if (blocks == recv_blocks) {
    return true;
  }

  LOG(ERROR) << "KVT check_recv_done:unpexcted blocks: expected=" << blocks
             << ",received=" << recv_blocks
             << ",src_inst_id=" << src_inst_id
             << ",src_worker_id=" << src_worker_id
             << ",req_id=" << req_id;
  throw KVTransferException(ErrorKind::UNEXPECTED_REQ_RECV, "unexpected request blocks;");
}

void KvRecvStub::earse(const RequestId &req_id) {
  std::unique_lock<std::shared_mutex> lock(task_m_);
  recv_tasks_.erase(req_id);
}
}
