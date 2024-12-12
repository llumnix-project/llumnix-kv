#include <mutex>
#include "service.h"
#include "thrid_party/logging.h"

namespace blade_llm {
KvTransferService::KvTransferService(std::unique_ptr<Context> &&ctx) :
    ctx_(std::move(ctx)) {}

KvRecvStub* KvTransferService::try_get_conn(const InstanceId& src_inst_id,
                                            WorkerId src_worker_id) {
  auto& self = *this;
  std::shared_lock<std::shared_mutex> rlock(self.conn_m_);
  auto iter = self.recv_conns_.find(src_inst_id);
  if (iter == self.recv_conns_.end()) {
    return nullptr;
  }
  auto& stubs = iter->second;
  if (stubs.size() <= src_worker_id) {
    return nullptr;
  }
  return stubs[src_worker_id].get();  // may be nullptr
}

KvRecvStub &KvTransferService::get_or_create_conn(const InstanceId &src_inst_id,
                                                  WorkerId src_worker_id) {
  KvRecvStub* ret = try_get_conn(src_inst_id, src_worker_id);
  if (ret != nullptr) {
    return *ret;
  }

  {
    std::unique_lock<std::shared_mutex> wlock(conn_m_);
    auto& src = recv_conns_[src_inst_id];
    if (src.size() <= src_worker_id) {
      src.resize(src_worker_id + 1);
    }
    assert(src.size() > src_worker_id);

    if (!src[src_worker_id]) {
      src[src_worker_id] = std::make_unique<KvRecvStub>(src_inst_id, src_worker_id);
    }
    ret = src[src_worker_id].get();
  }
  LOG(INFO) << "KVT service: create rx_stub from worker("
            << src_inst_id << ":" << src_worker_id << ")";
  return *ret;
}

void KvTransferService::submit_recv(const InstanceId &src_inst_id,
                                    WorkerId src_worker_id,
                                    const RequestId &req_id,
                                    const std::vector<uint32_t> &dst_block_ids) {
  if (dst_block_ids.empty()) {
    throw KVTransferException(ErrorKind::INVALID_REQUEST_PARAM, "receive blocks can't be empty;");
  }
  auto [r, _] = reqs_.try_emplace(req_id);
  r->second.emplace_back(src_inst_id, src_worker_id, req_id)
      .set_dst_blocks(dst_block_ids);
  LOG(INFO) << "KVT service: accept request(" << req_id << ") recv from worker("
            << src_inst_id << "," << src_worker_id << ")";
}

bool KvTransferService::check_recv_done(const RequestId &req_id) {
  auto reqiter = reqs_.find(req_id);
  if (reqiter == reqs_.end()) {
    LOG(ERROR) << "request " << req_id << " not submit to recv;";
    throw KVTransferException(ErrorKind::REQUEST_NOT_FOUND, "receive of request not submit;");
  }

  for (const auto &r : reqiter->second) {
    auto* recv_stub = try_get_conn(r.src_inst_id, r.src_worker_id);
    if (recv_stub == nullptr) {
      return false;
    }
    auto const is_done = recv_stub->check_recv_done(r.req_id, r.dst_blocks());
    if (!is_done) {
      return false;
    }
  }
  // found it!

  for (const auto& r : reqiter->second) {
    auto* recv_stub = try_get_conn(r.src_inst_id, r.src_worker_id);
    assert(recv_stub != nullptr);
    recv_stub->earse(r.req_id);
  }
  reqs_.erase(reqiter);
  return true;
}

void KvTransferService::on_recv(const InstanceId &src_inst_id,
                                WorkerId src_worker_id,
                                const RequestId &req_id,
                                std::vector<uint32_t> &&dst_block_ids) {
  get_or_create_conn(src_inst_id, src_worker_id)
      .on_recv(req_id, std::move(dst_block_ids));
}
}
