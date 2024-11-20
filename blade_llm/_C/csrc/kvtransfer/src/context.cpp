#include <cassert>
#include <stdexcept>
#include "context.h"
#include "utils/cuda_helper.h"
#include "thrid_party/logging.h"

namespace blade_llm {

class NoCudaBarrier : public ICUDABarrier {
 public:
  NoCudaBarrier() = default;
  void wait(uint32_t layer_idx) override {
    // do nothing;
  }
};

void CudaEventBarrier::wait(uint32_t layer_idx) {
  if (layer_idx < event_addrs_.size()) {
    cuda_wait_event(event_addrs_[layer_idx]);
  }
}

Context::Context(const InstanceName &inst, const WorkerId &worker) :
    inst_name(inst),
    inst_id(Context::get_inst_id(inst)),
    worker_id(worker),
    cuda_barrier_(std::make_unique<NoCudaBarrier>()) {
  worker_info_ = WorkerInfo(inst_id, worker_id);
};

Context::Context(const InstanceName &name, const InstanceId &id, const WorkerId &worker_id) :
    inst_name(name),
    inst_id(id),
    worker_id(worker_id),
    worker_info_(id, worker_id),
    cuda_barrier_(std::make_unique<NoCudaBarrier>()) {};

void Context::set_tp(uint32_t tp_size, uint32_t worker_tp_rank) {
  worker_info_.tp_size = tp_size;
  worker_info_.worker_tp_rank = worker_tp_rank;
}

void Context::set_block_params(uint32_t block_size, uint32_t token_size, uint32_t layer_num_blocks) {
  if (block_size < token_size || block_size % token_size != 0) {
    throw std::runtime_error("block_size must be a multiple of token_size");
  }
  worker_info_.block_size = block_size;
  worker_info_.token_size = token_size;
  worker_info_.layer_num_blocks = layer_num_blocks;
}

void Context::set_layer_data_address(uint8_t device_id, const std::vector<uint64_t> &layers) {
  device_id_ = (int) device_id;
  worker_info_.num_layers = layers.size();
  layer_data_address_ = layers;
}

void Context::set_cuda_barrier(std::unique_ptr<ICUDABarrier> &&barrier) {
  if (barrier == nullptr) {
    throw std::runtime_error("cuda barrier can't be null;");
  }
  cuda_barrier_ = std::move(barrier);
}

WorkerInfo *Context::worker_info_mutable() {
  return &worker_info_;
}

ICUDABarrier *Context::cuda_barrier() {
  assert(cuda_barrier_ != nullptr);
  return cuda_barrier_.get();
}

const WorkerInfo &Context::worker_info() const {
  return worker_info_;
}
const std::vector<uint64_t> &Context::layer_data_address() const {
  return layer_data_address_;
}

int Context::device_id() const {
  return device_id_;
}

uint32_t Context::num_layers() const {
  return worker_info_.num_layers;
}

uint32_t Context::layer_num_blocks() const {
  return worker_info_.layer_num_blocks;
}

size_t Context::block_size() const {
  return worker_info_.block_size;
}
const SupportTransferProtocols &Context::support_protocols() const {
  return transfer_protos_;
}

bool Context::check_transfer_support(TransferProtocol t) const {
  return transfer_protos_.is_support(t);
}
void Context::register_protocol(std::unique_ptr<IProtocolContext> &&ctx) {
  if (!ctx->check_support()) {
    throw std::runtime_error("register unsupported protocol: " + ctx->protocol().to_string());
  }
  if (!ctx->is_initialized()) {
    try {
      ctx->init(this);
    } catch (const std::exception &e) {
      throw std::runtime_error("init protocol context failed: " + std::string(e.what()));
    }
    LOG(INFO) << "KVT: init " << ctx->protocol().to_string() << " protocol context successfully";
  }
  auto p = ctx->protocol();
  auto ret = protocol_ctxs_.emplace(p.type, std::move(ctx));
  assert(ret.second);
  transfer_protos_.set_support(p);
}
InstanceId Context::get_inst_id(const InstanceName &name) {
  return std::hash<InstanceName>{}(name);
}
}
