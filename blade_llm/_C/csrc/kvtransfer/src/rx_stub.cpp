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

KvRecvStub::KvRecvStub(KvRecvStub &&other) noexcept:
    src_inst_id(other.src_inst_id),
    src_worker_id(other.src_worker_id),
    recv_tasks_(std::move(other.recv_tasks_)) {}

void KvRecvStub::on_recv(const RequestId &req_id, std::vector<uint32_t> &&dst_block_ids) {
  LOG(INFO) << "KVT rx_stub: recv all kv of request(" << req_id <<
            ") from (" << src_inst_id << ":" << src_worker_id << ").";
  std::sort(dst_block_ids.begin(), dst_block_ids.end());
  std::unique_lock<std::shared_mutex> lock(task_m_);
  auto [_, ok] = recv_tasks_.try_emplace(req_id, src_inst_id, src_worker_id, req_id, std::move(dst_block_ids));
  assert(ok);
}

Result<bool> KvRecvStub::check_recv_done(const RequestId &req_id, const std::vector<uint32_t> &blocks) {
  size_t expect_blocks = blocks.size();
  std::shared_lock<std::shared_mutex> lock(task_m_);
  auto task = recv_tasks_.find(req_id);
  if (task != recv_tasks_.end()) {
    const auto &recv_blocks = task->second.dst_blocks();
    if (recv_blocks.size() < expect_blocks) {
      LOG(ERROR) << "KVT rx_stub: recv " << recv_blocks.size() << ", but expect "
                 << expect_blocks << " blocks of request: " << req_id;
      return Result<bool>::error(UNEXPECTED_REQ_RECV, "unexpected request blocks;");
    }
    for (auto i = 0; i < expect_blocks; i++) {
      if (recv_blocks[i] != blocks[i]) {
        LOG(ERROR) << "KVT rx_stub: expect block[" << i << "] = " << blocks[i] <<
                   ", but get block[" << i << "] = " << recv_blocks[i] << " of request " << req_id;
        return Result<bool>::error(UNEXPECTED_REQ_RECV, "unexpected request blocks;");
      }
    }
    return {true};
  } else {
    return {false};
  }
}

void KvRecvStub::earse(const RequestId &req_id) {
  std::unique_lock<std::shared_mutex> lock(task_m_);
  recv_tasks_.erase(req_id);
}
}
