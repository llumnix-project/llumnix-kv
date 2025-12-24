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
  cuda_wait_event(event_addrs_.at(layer_idx));
}

Context::Context(const InstanceId &inst, const WorkerId &worker) :
    inst_name(inst),
    worker_id(worker),
    cuda_barrier_(std::make_unique<NoCudaBarrier>()) {
  worker_info_ = WorkerInfo(inst, worker_id);
};

void Context::set_tp(uint32_t tp_size, uint32_t worker_tp_rank) {
  worker_info_.tp_size = tp_size;
  worker_info_.worker_tp_rank = worker_tp_rank;
}

void Context::set_block_params(std::vector<size_t> block_sizes, std::vector<size_t> token_sizes, uint32_t layer_num_blocks) {
  RTASSERT(block_sizes.size() == token_sizes.size());
  for (size_t i = 0; i < block_sizes.size(); i++) {
    auto block_size = block_sizes[i];
    auto token_size = token_sizes[i];
    RTASSERT(block_size % token_size == 0);
  }
  worker_info_.block_sizes = block_sizes;
  worker_info_.token_sizes = token_sizes;
  worker_info_.layer_num_blocks = layer_num_blocks;
}


void Context::set_layer_info(uint8_t device_id, const std::vector<std::vector<LayerInfo>> &all_layer_infos) {
  device_id_ = (int) device_id;
  worker_info_.num_layers = all_layer_infos.size();
  all_layer_infos_ = all_layer_infos;

  // Extract all layer addresses from all_layer_infos_
  layer_data_address_.clear();
  for (const auto &layer_infos : all_layer_infos_) {
    std::vector<uint64_t> layer_cache_address;
    for (const auto &info : layer_infos) {
      layer_cache_address.push_back(info.layer_addr);
    }
    layer_data_address_.push_back(std::move(layer_cache_address));
  }
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

const std::vector<std::vector<LayerInfo>> &Context::all_layer_infos() const {
  return all_layer_infos_;
}

const std::vector<std::vector<uint64_t>> &Context::layer_data_address() const {
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

const std::vector<size_t> &Context::block_sizes() const {
  return worker_info_.block_sizes;
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
}
