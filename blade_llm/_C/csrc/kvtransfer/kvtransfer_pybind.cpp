#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "client.h"
#include "service.h"
#include "server.h"
#include "naming.h"
#include "protocol.h"

namespace blade_llm {

static KvTransferClient *KV_CLIENT = nullptr;
static ITransferServer *KV_SERVER = nullptr;
static KvTransferService *KV_SERVICE = nullptr;
static thread_local std::optional<Error> ERROR_OPT = std::nullopt;

static std::vector<std::string> LIBRARY_SUPPORT_TRANSFER_TYP;

const std::vector<std::string>& support_transfer_types() {
  if (LIBRARY_SUPPORT_TRANSFER_TYP.empty()) {
    auto supported = get_supported_transfer_types();
    for(auto t: supported) {
      switch (t) {
        case CUDA_IPC:
          LIBRARY_SUPPORT_TRANSFER_TYP.push_back("CUDA_IPC");
          break;
        case RDMA_DIRECT:
          LIBRARY_SUPPORT_TRANSFER_TYP.push_back("RDMA_DIRECT");
          break;
        default:break;
      }
    }
  }
  return LIBRARY_SUPPORT_TRANSFER_TYP;
}

void init_kv_transfer_client(uint32_t inst_id,
                             uint32_t tp_size,
                             uint32_t worker_id,
                             uint32_t worker_tp_rank,
                             uint32_t device_id,
                             uint32_t block_size,
                             uint32_t token_size,
                             uint32_t layer_num_blocks,
                             uint32_t transfer_type,
                             const std::string &naming_url,
                             const std::vector<uint64_t> &events,
                             const std::vector<uint64_t> &layers) {

  if (KV_CLIENT == nullptr) {
    assert(block_size % token_size == 0);
    connect_naming(naming_url);
    auto context = std::make_unique<Context>(inst_id, worker_id);
    context->set_tp(tp_size, worker_tp_rank);
    context->set_block_params(block_size, token_size, layer_num_blocks);
    context->set_layer_data_address(device_id, layers);
    context->set_cuda_barrier(std::make_unique<CudaEventBarrier>(events));
    context->set_transfer_type(type_from(transfer_type));
    KV_CLIENT = new KvTransferClient(std::move(context));
    // disable auto connect after python runtime ready;
    KV_CLIENT->enable_auto_connect();
  }
}

void add_target(uint32_t inst_id, uint32_t worker_id, uint32_t start_layer, uint32_t num_layers) {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->add_target(inst_id, worker_id, start_layer, num_layers);
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv client is not initialized"));
  }
}

void submit_req_send(uint32_t dst_inst_id,
                     uint32_t dst_worker_id,
                     const std::string &req_id,
                     uint32_t new_tokens,
                     bool has_last_token,
                     const std::vector<uint32_t> &src_block_ids,
                     const std::vector<uint32_t> &dst_block_ids) {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->submit_req_send(dst_inst_id,
                                          dst_worker_id,
                                          req_id,
                                          new_tokens,
                                          has_last_token,
                                          src_block_ids,
                                          dst_block_ids);
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv client is not initialized"));
  }
}

void submit_delta_send(const std::string &req_id,
                       uint32_t seen_tokens,
                       uint32_t new_tokens,
                       bool has_last_token) {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->submit_delta_send(req_id,
                                            seen_tokens,
                                            new_tokens,
                                            has_last_token);
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  }
}

void start_send() {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->start_send();
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv client is not initialized"));
  }
}

void notify_event_record() {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->notify_event_record();
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv client is not initialized"));
  }
};

void flush_send() {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->flush_send();
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv client is not initialized"));
  }
}

bool check_transfer_done(const std::string &req_id) {
  if (KV_CLIENT != nullptr) {
    auto ret = KV_CLIENT->check_transfer_done(req_id);
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    } else {
      return ret.ok();
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv client is not initialized"));
  }
  return false;
}

void init_kv_transfer_server(uint32_t inst_id,
                             uint32_t tp_size,
                             uint32_t worker_id,
                             uint32_t worker_tp_rank,
                             uint32_t device_id,
                             uint32_t block_size,
                             uint32_t token_size,
                             uint32_t layer_num_blocks,
                             uint32_t transfer_type,
                             const std::string &naming_url,
                             const std::vector<uint64_t> &layers) {
  if (KV_SERVER == nullptr) {
    connect_naming(naming_url);
    auto context = std::make_unique<Context>(inst_id, worker_id);
    context->set_tp(tp_size, worker_tp_rank);
    context->set_block_params(block_size, token_size, layer_num_blocks);
    context->set_layer_data_address(device_id, layers);
    context->set_transfer_type(type_from(transfer_type));

    KV_SERVER = create_transfer_server(context->transfer_type());
    KV_SERVICE = new KvTransferService(std::move(context));
    auto ctx = KV_SERVICE->get_context();
    KV_SERVER->start_server(KV_SERVICE, ctx);
    naming()->register_worker(ctx->worker_info());
  }
}

void submit_req_recv(uint32_t src_inst_id,
                     uint32_t src_worker_id,
                     const std::string &req_id,
                     const std::vector<uint32_t> &dst_block_ids) {

  if (KV_SERVICE != nullptr) {
    auto ret = KV_SERVICE->submit_recv(src_inst_id,
                                       src_worker_id,
                                       req_id,
                                       dst_block_ids);
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv service is not start"));
  }
}

bool check_recv_done(const std::string &req_id) {
  if (KV_SERVICE != nullptr) {
    auto ret =  KV_SERVICE->check_recv_done(req_id);
    if (ret.is_err()) {
      ERROR_OPT.emplace(std::move(ret.err()));
    } else {
      return ret.ok();
    }
  } else {
    ERROR_OPT.emplace(Error(ErrorCode::INVALID_OPERATION, "kv service is not start"));
  }
  return false;
}

std::string check_error() {
  if (ERROR_OPT.has_value()) {
    auto str = ERROR_OPT.value().to_string();
    ERROR_OPT.reset();
    return str;
  } else {
    return "";
  }
}
}

PYBIND11_MODULE(kvtransfer_ops, m) {
m.def("support_transfer_types", &blade_llm::support_transfer_types, "get supported transfer types");
// client
m.def("init_kv_transfer_client", &blade_llm::init_kv_transfer_client, "init kv transfer client;");
m.def("add_target", &blade_llm::add_target, "add target to kv client;");
m.def("submit_req_send", &blade_llm::submit_req_send, "submit kv send to kv client;");
m.def("submit_delta_send", &blade_llm::submit_delta_send, "submit kv token to kv client;");
m.def("start_send", &blade_llm::start_send, "start to send submitted kv data;");
m.def("notify_event_record", &blade_llm::notify_event_record, "record kv send events;");
m.def("flush_send", &blade_llm::flush_send, "check if all kv send tasks are done;");
m.def("check_transfer_done", &blade_llm::check_transfer_done, "check if all kv data of a request are sent;");
// server
m.def("init_kv_transfer_server", &blade_llm::init_kv_transfer_server, "init kv transfer server;");
m.def("submit_req_recv", &blade_llm::submit_req_recv, "submit kv recv task to kv server;");
m.def("check_recv_done", &blade_llm::check_recv_done, "check if all kv data of a request are received;");
// common
m.def("check_error", &blade_llm::check_error, "check error message;");
}
