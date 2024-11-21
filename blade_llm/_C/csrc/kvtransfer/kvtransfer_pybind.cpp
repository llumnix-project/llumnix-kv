#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include "client.h"
#include "service.h"
#include "server.h"
#include "naming.h"
#include "protocol.h"
#include "thrid_party/logging.h"

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

void submit_req_send(const std::string &dst_inst_name,
                     uint32_t dst_worker_id,
                     const std::string &req_id,
                     uint32_t new_tokens,
                     bool has_last_token,
                     std::vector<uint32_t> src_block_ids,
                     std::vector<uint32_t> dst_block_ids) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->submit_req_send(dst_inst_name,
                               dst_worker_id,
                               req_id,
                               new_tokens,
                               has_last_token,
                               std::move(src_block_ids),
                               std::move(dst_block_ids));
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
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

void start_send() {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->start_send();
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

void notify_event_record() {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->notify_event_record();
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
};

void flush_send() {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->flush_send();
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

bool check_transfer_done(const std::string &req_id) {
  if (KV_CLIENT != nullptr) {
    return KV_CLIENT->check_transfer_done(req_id);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
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
    auto naming_client = connect_naming(inst_name, naming_url);
    LOG(INFO) << "KVT: init kv server for worker(" << inst_name << ":" << worker_id << ") at " << inst_name;
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
    auto worker_info = ctx->worker_info();
    worker_info.transfer_protocols = ctx->support_protocols().value();
    auto naming_worker_client = naming_client.create_naming_worker_client();

    int retry_times = 3;
    while (retry_times > 0) {
      try {
        naming_worker_client->register_worker(worker_info);
        break;
      } catch (const std::exception &e) {
        LOG(WARNING) << "KVT: register worker failed: " << e.what() << ", will try later.";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      retry_times--;
    }
    if (retry_times <= 0) {
      LOG(ERROR) << "KVT: can't register worker after retry;";
      throw std::runtime_error("register worker failed");
    }
  }
}

void submit_req_recv(const std::string &src_inst_name,
                     uint32_t src_worker_id,
                     const std::string &req_id,
                     const std::vector<uint32_t> &dst_block_ids) {

  if (KV_SERVICE != nullptr) {
    LOG(INFO) << "KVT: submit recv request: " << req_id << " from "
              << src_inst_name << ":" << src_worker_id <<";";
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
}

namespace py = pybind11;
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
      .def("store", &blade_llm::GeneralNamingClient::store, "get key from naming service;")
      .def("remove", &blade_llm::GeneralNamingClient::remove, "remove key from naming service;")
      .def("list", &blade_llm::GeneralNamingClient::list, "list keys from naming service;");

  m.def("connect_naming", &blade_llm::connect_naming, "connect to naming service;");
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
  m.def("lib_support_transfer_protocols", &blade_llm::support_transfer_protocols, "get supported transfer types");
}
