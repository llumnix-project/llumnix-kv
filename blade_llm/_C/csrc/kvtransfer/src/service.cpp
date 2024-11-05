#include <mutex>
#include "service.h"
#include "logging.h"

namespace blade_llm {
KvTransferService::KvTransferService(std::unique_ptr<Context> &&ctx) :
    ctx_(std::move(ctx)) {}

KvRecvStub &KvTransferService::get_or_create_conn(InstanceId src_inst_id,
                                                  WorkerId src_worker_id) {
  {
    std::shared_lock<std::shared_mutex> rlock(conn_m_);
    if (auto f = recv_conns_.find(src_inst_id);
        f != recv_conns_.end()) {
      if (f->second.size() > src_worker_id) {
        if (f->second[src_worker_id].has_value()) {
          return f->second[src_worker_id].value();
        }
      }
    }
  }
  std::unique_lock<std::shared_mutex> wlock(conn_m_);
  while (recv_conns_[src_inst_id].size() <= src_worker_id) {
    recv_conns_[src_inst_id].emplace_back(std::nullopt);
  }
  auto &src = recv_conns_[src_inst_id];
  if (!src[src_worker_id].has_value()) {
    src[src_worker_id].emplace(src_inst_id, src_worker_id);
  }
  LOG(INFO) << "KVT service: create rx_stub from worker("
            << src_inst_id << ":" << src_worker_id << ")";
  return src[src_worker_id].value();
}

Result<bool> KvTransferService::submit_recv(InstanceId src_inst_id,
                                            WorkerId src_worker_id,
                                            const RequestId &req_id,
                                            const std::vector<uint32_t> &dst_block_ids) {
  if (dst_block_ids.empty()) {
    return Result<bool>::error(ErrorCode::INVALID_REQUEST_PARAM, "receive blocks can't be empty;");
  }
  auto [r, _] = reqs_.try_emplace(req_id);
  r->second.emplace_back(src_inst_id, src_worker_id, req_id)
      .set_dst_blocks(dst_block_ids);
  LOG(INFO) << "KVT service: accept request(" << req_id << ") recv from worker("
            << src_inst_id << "," << src_worker_id << ")";
  return {true};
}

Result<bool> KvTransferService::check_recv_done(const RequestId &req_id) {
  auto f = reqs_.find(req_id);
  if (f == reqs_.end()) {
    LOG(ERROR) << "request " << req_id << " not submit to recv;";
    return Result<bool>::error(REQUEST_NOT_FOUND, "receive of request not submit;");
  }
  for (const auto &r : f->second) {
    bool is_done = false;
    std::shared_lock<std::shared_mutex> rlock(conn_m_);
    if (auto inst = recv_conns_.find(r.src_inst_id);
        inst != recv_conns_.end()) {
      if (inst->second.size() > r.src_worker_id) {
        if (inst->second[r.src_worker_id].has_value()) {
          auto ret = inst->second[r.src_worker_id]->check_recv_done(r.req_id, r.dst_blocks());
          if (ret.is_ok()) {
            is_done = ret.ok();
          } else {
            return ret;
          }
        }
      }
    }
    if (!is_done) {
      return false;
    }
  }
  auto fa = reqs_.find(req_id);
  for (const auto &r : fa->second) {
    std::shared_lock<std::shared_mutex> rlock(conn_m_);
    recv_conns_[r.src_inst_id][r.src_worker_id]->earse(r.req_id);
  }
  reqs_.erase(req_id);
  return true;
}

void KvTransferService::on_recv(InstanceId src_inst_id,
                                WorkerId src_worker_id,
                                const RequestId &req_id,
                                std::vector<uint32_t> &&dst_block_ids) {
  get_or_create_conn(src_inst_id, src_worker_id)
      .on_recv(req_id, std::move(dst_block_ids));
}
}
