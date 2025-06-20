#ifndef KVTRANSFER_INCLUDE_RX_STUB_H_
#define KVTRANSFER_INCLUDE_RX_STUB_H_

#pragma once
#include <vector>
#include <string>
#include <shared_mutex>
#include <unordered_map>
#include <algorithm>
#include "common.h"
#include "error.h"

namespace blade_llm {
class ReqRecvTask : public noncopyable {
 public:
  const RequestId req_id;
  const InstanceId src_inst_id;
  const WorkerId src_worker_id;

  ReqRecvTask(const InstanceId& src_inst_id_, WorkerId src_worker_id_, const RequestId &req_id_) :
      req_id(req_id_),
      src_inst_id(src_inst_id_),
      src_worker_id(src_worker_id_) {};
  ReqRecvTask(const InstanceId& src_inst_id_,
              WorkerId src_worker_id_,
              const RequestId &req_id_,
              std::vector<uint32_t> &&blocks) :
      req_id(req_id_),
      src_inst_id(src_inst_id_),
      src_worker_id(src_worker_id_),
      dst_blocks_(std::move(blocks)) {};
  ReqRecvTask(ReqRecvTask &&other) :
      req_id(other.req_id),
      src_inst_id(other.src_inst_id),
      src_worker_id(other.src_worker_id),
      dst_blocks_(std::move(other.dst_blocks_)) {};
  void set_dst_blocks(const std::vector<uint32_t> &block_ids);
  const std::vector<uint32_t> &dst_blocks() const;

 private:
  std::vector<uint32_t> dst_blocks_;
};

class KvRecvStub : public noncopyable {
 public:
  const InstanceId src_inst_id;
  const WorkerId src_worker_id;
  KvRecvStub(const InstanceId& inst_id, WorkerId worker_id) :
      src_inst_id(inst_id),
      src_worker_id(worker_id) {};

  KvRecvStub(const KvRecvStub&) = delete;
  KvRecvStub(KvRecvStub &&other) = delete;

  // thread safe
  void on_recv(const RequestId&, std::vector<uint32_t> &&block_ids);
  bool check_recv_done(const RequestId&, const std::vector<uint32_t> &block_ids);
  void earse(const RequestId&);
 private:
  std::shared_mutex task_m_;
  std::unordered_map<RequestId, ReqRecvTask> recv_tasks_;
};
}
#endif //KVTRANSFER_INCLUDE_RX_STUB_H_
