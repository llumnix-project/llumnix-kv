#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include "client.h"
#include "service.h"
#include "server.h"
#include "naming.h"
#include "protocol.h"
#include "thrid_party/logging.h"

namespace py = pybind11;

namespace blade_llm {

static std::unique_ptr<KvTransferClient> KV_CLIENT = nullptr;
static ITransferServer *KV_SERVER = nullptr;
static KvTransferService *KV_SERVICE = nullptr;
static std::vector<TransferProtocol> LIBRARY_SUPPORT_TRANSFER_PROTOCOLS;
static NamingManager *NAMING_MANAGER = new NamingManager();

const std::vector<TransferProtocol> &support_transfer_protocols() {
  if (LIBRARY_SUPPORT_TRANSFER_PROTOCOLS.empty()) {
    auto supported = get_library_support_protocols();
    LIBRARY_SUPPORT_TRANSFER_PROTOCOLS = supported.as_vector();
  }
  return LIBRARY_SUPPORT_TRANSFER_PROTOCOLS;
}

GeneralNamingClient connect_naming(const InstanceId &name, const std::string &url) {
  return NAMING_MANAGER->connect_naming(name, url);
}

void init_kv_transfer_client(const std::string &inst_name,
                             uint32_t tp_size,
                             uint32_t worker_id,
                             uint32_t worker_tp_rank,
                             uint32_t device_id,
                             uint32_t block_size,
                             uint32_t token_size,
                             uint32_t layer_num_blocks,
                             const std::string &naming_url,
                             const std::vector<uint64_t> &events,
                             const std::vector<uint64_t> &layers,
                             const std::vector<TransferProtocol> &protocols) {

  if (KV_CLIENT == nullptr) {
    assert(block_size % token_size == 0);
    LOG(INFO) << "KVT: init kv client for worker(" << inst_name << ":" << worker_id << ") at " << inst_name;
    auto context = std::make_unique<Context>(inst_name, worker_id);
    context->set_tp(tp_size, worker_tp_rank);
    context->set_block_params(block_size, token_size, layer_num_blocks);
    context->set_layer_data_address(device_id, layers);
    context->set_cuda_barrier(std::make_unique<CudaEventBarrier>(events));
    auto naming_client = NAMING_MANAGER->connect_naming(inst_name, naming_url);
    auto stub_factory = std::make_unique<KvSendStubFactory>(context.get(), std::move(naming_client));
    KV_CLIENT = KvTransferClient::create(std::move(context), protocols, std::move(stub_factory));
    // disable auto connect after python runtime ready;
    KV_CLIENT->enable_auto_connect();
    auto* ctx = KV_CLIENT->context();
    auto* worker_info = ctx->worker_info_mutable();
    worker_info->transfer_protocols = ctx->support_protocols().value();
    LOG(INFO) << "init_kv_transfer_client. worker_info=" << worker_info->to_string();
  }
}

void add_target(const std::string &inst_name,
                uint32_t worker_id,
                uint32_t start_layer,
                uint32_t num_layers,
                std::optional<TransferProtocol> protocol) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->add_target(inst_name, worker_id, start_layer, num_layers, protocol);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

void submit_req_send2(const std::string &dst_inst_name,
                      uint32_t dst_worker_id,
                      const std::string &req_id,
                      uint32_t seen_tokens,
                      uint32_t new_tokens,
                      bool has_last_token,
                      std::vector<uint32_t> src_block_ids,
                      std::vector<uint32_t> dst_block_ids,
                      std::optional<std::string> dst_worker_info = std::nullopt) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->submit_req_send(dst_inst_name,
                               dst_worker_id,
                               req_id,
                               seen_tokens,
                               new_tokens,
                               has_last_token,
                               std::move(src_block_ids),
                               std::move(dst_block_ids),
                               std::move(dst_worker_info));
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

void submit_req_send(const std::string &dst_inst_name,
                     uint32_t dst_worker_id,
                     const std::string &req_id,
                     uint32_t new_tokens,
                     bool has_last_token,
                     std::vector<uint32_t> src_block_ids,
                     std::vector<uint32_t> dst_block_ids) {
  return submit_req_send2(dst_inst_name, dst_worker_id,
                          req_id, 0, new_tokens, has_last_token,
                          std::move(src_block_ids),
                          std::move(dst_block_ids));
}

void submit_delta_send(const std::string &req_id,
                       uint32_t seen_tokens,
                       uint32_t new_tokens,
                       bool has_last_token) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->submit_delta_send(req_id, seen_tokens, new_tokens, has_last_token);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

size_t start_send() {
  if (KV_CLIENT == nullptr) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
  return KV_CLIENT->start_send();
}

void notify_event_record(size_t step_id) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->notify_event_record(step_id);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
};

void flush_send(size_t step_id) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->flush_send(step_id);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

bool check_transfer_done(const std::string &req_id) {
  if (KV_CLIENT == nullptr) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
  auto rs = KV_CLIENT->check_transfer_done(req_id);
  if (rs == ReqState::FAILED) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "send failed");
  }
  return rs == ReqState::OK;
}

void init_kv_transfer_server(const std::string &inst_name,
                             uint32_t tp_size,
                             uint32_t worker_id,
                             uint32_t worker_tp_rank,
                             uint32_t device_id,
                             uint32_t block_size,
                             uint32_t token_size,
                             uint32_t layer_num_blocks,
                             const std::string &naming_url,
                             const std::vector<uint64_t> &layers,
                             const std::vector<TransferProtocol> &protocols) {
  if (KV_SERVER == nullptr) {
    auto context = std::make_unique<Context>(inst_name, worker_id);
    context->set_tp(tp_size, worker_tp_rank);
    context->set_block_params(block_size, token_size, layer_num_blocks);
    context->set_layer_data_address(device_id, layers);

    if (protocols.size() > 1) {
      throw std::runtime_error("multi-protocols server not support temporarily");
    }

    KV_SERVICE = new KvTransferService(std::move(context));
    auto ctx = KV_SERVICE->get_context();
    if (protocols.empty()) {
      auto supported = get_library_support_protocols();
      if (supported.is_support(TransferProtocol::Kind::RDMA_DIRECT)) {
        auto p = TransferProtocol::rdma_direct();
        try {
          KV_SERVER = create_transfer_server(p);
          KV_SERVER->start_server(KV_SERVICE, ctx);
          LOG(INFO) << "KVT: start kvtransfer server with protocol: " + p.to_string();
        } catch (const std::exception &e) {
          KV_SERVER = nullptr;
          LOG(INFO) << "KVT: transfer protocol: " + p.to_string() + " not support, try next ...";
        }
      } else if (supported.is_support(TransferProtocol::Kind::CUDA_IPC)) {
        auto p = TransferProtocol::cuda_ipc();
        try {
          KV_SERVER = create_transfer_server(p);
          KV_SERVER->start_server(KV_SERVICE, ctx);
          LOG(INFO) << "KVT: start kvtransfer server with protocol: " + p.to_string();
        } catch (const std::exception &e) {
          KV_SERVER = nullptr;
          LOG(INFO) << "KVT: transfer protocol: " + p.to_string() + " not support.";
        }
      }

      if (KV_SERVER == nullptr) {
        throw std::runtime_error("start kvtransfer server failed, because no transfer protocol support");
      }
    } else {
      KV_SERVER = create_transfer_server(protocols[0]);
      KV_SERVER->start_server(KV_SERVICE, ctx);
    }
    auto* worker_info = ctx->worker_info_mutable();
    worker_info->transfer_protocols = ctx->support_protocols().value();
    LOG(INFO) << "init_kv_transfer_server. worker_info=" << worker_info->to_string();
  }
}

void submit_req_recv(const std::string &src_inst_name,
                     uint32_t src_worker_id,
                     const std::string &req_id,
                     const std::vector<uint32_t> &dst_block_ids) {

  if (KV_SERVICE != nullptr) {
    KV_SERVICE->submit_recv(src_inst_name, src_worker_id, req_id, dst_block_ids);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv service is not start");
  }
}

bool check_recv_done(const std::string &req_id) {
  if (KV_SERVICE != nullptr) {
    return KV_SERVICE->check_recv_done(req_id);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv service is not start");
  }
}

// empty 意味着不处于 kvt 环境下.
std::string current_worker_info(const std::string& kind = "any") {
    Context* ctx = nullptr;

    if (kind == "client" && KV_CLIENT) {
        ctx = KV_CLIENT->context();
    } else if (kind == "server" && KV_SERVICE) {
        ctx = KV_SERVICE->get_context();
    } else if (kind == "any") {
        if (KV_CLIENT) {
            ctx = KV_CLIENT->context();
        } else if (KV_SERVICE) {
            ctx = KV_SERVICE->get_context();
        }
    }
    if (ctx) {
      return ctx->worker_info().to_string();
    }
    return {};
}

#ifdef ENABLE_TORCH

PyObject* alloc_phy_cont_mem(size_t size, PyObject* device);

static py::object alloc_phy_cont_mem_w(size_t size, py::handle device) {
  auto* res = alloc_phy_cont_mem(size, device.ptr());
  return py::reinterpret_steal<py::object>(py::handle(res));
}

#endif  // ENABLE_TORCH

}  // namespace blade_llm

PYBIND11_MODULE(kvtransfer_ops, m) {
  // class
  py::register_exception<blade_llm::KVTransferException>(m, "KVTransferError", PyExc_RuntimeError);
  py::class_<blade_llm::TransferProtocol> protocol(m, "TransferProtocol");
  protocol.def(py::init<blade_llm::TransferProtocol::Kind>())
      .def_readwrite("type", &blade_llm::TransferProtocol::type)
      .def("to_string", &blade_llm::TransferProtocol::to_string)
      .def("__str__", &blade_llm::TransferProtocol::to_string)
      .def("__repr__", &blade_llm::TransferProtocol::to_string);
  py::enum_<blade_llm::TransferProtocol::Kind>(protocol, "Kind")
      .value("CUDA_IPC", blade_llm::TransferProtocol::Kind::CUDA_IPC)
      .value("RDMA_DIRECT", blade_llm::TransferProtocol::Kind::RDMA_DIRECT)
      .export_values();

  py::class_<blade_llm::GeneralNamingClient> naming_client(m, "NamingClient");
  naming_client.def(py::init<>())
      .def("connect", &blade_llm::GeneralNamingClient::connect, "connect to naming service;")
      .def("get", &blade_llm::GeneralNamingClient::get, "get key from naming service;")
      .def("search", &blade_llm::GeneralNamingClient::search, "search key from naming service;")
      .def("store", &blade_llm::GeneralNamingClient::store, "get key from naming service;")
      .def("remove", &blade_llm::GeneralNamingClient::remove, "remove key from naming service;")
      .def("list", &blade_llm::GeneralNamingClient::list, "list keys from naming service;");

#ifdef ENABLE_TORCH
  m.def("alloc_phy_cont_mem", &blade_llm::alloc_phy_cont_mem_w, "alloca physical contiguous memory");
#endif  // ENABLE_TORCH

  m.def("connect_naming", &blade_llm::connect_naming, "connect to naming service;");
  // client
  m.def("init_kv_transfer_client", &blade_llm::init_kv_transfer_client, "init kv transfer client;");
  m.def("add_target", &blade_llm::add_target, "add target to kv client;");
  m.def("submit_req_send", &blade_llm::submit_req_send, "submit kv send to kv client;");
  m.def("submit_req_send2", &blade_llm::submit_req_send2, "submit kv send to kv client;");
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
  m.def("current_worker_info", &blade_llm::current_worker_info, "get current worker info;", py::arg("kind") = "any");
  m.def("lib_support_transfer_protocols", &blade_llm::support_transfer_protocols, "get supported transfer types");
}
