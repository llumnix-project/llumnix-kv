#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <cstdlib>

#include "client.h"
#include "server.h"
#include "naming.h"
#include "protocol.h"
#include "thrid_party/logging.h"
#include "envcfg.h"
#include "tx_stub.h"

namespace py = pybind11;

namespace blade_llm {

void monitor_init();

static std::unique_ptr<KvTransferClient> KV_CLIENT = nullptr;
static ITransferServer *KV_SERVER = nullptr;
static std::unique_ptr<Context> KV_SERVER_CTX = nullptr;
static std::vector<TransferProtocol> LIBRARY_SUPPORT_TRANSFER_PROTOCOLS;
static NamingManager *NAMING_MANAGER = new NamingManager();

const std::vector<TransferProtocol> &support_transfer_protocols() {
  if (LIBRARY_SUPPORT_TRANSFER_PROTOCOLS.empty()) {
    auto supported = get_library_support_protocols();
    LIBRARY_SUPPORT_TRANSFER_PROTOCOLS = supported.as_vector();
  }
  return LIBRARY_SUPPORT_TRANSFER_PROTOCOLS;
}

static std::optional<WorkerInfo> parse_py_worker_info(const py::handle& worker_info) {
  if (worker_info.is_none()) {
    return std::nullopt;
  }
  if (py::isinstance<py::str>(worker_info)) {
    return WorkerInfo::from_string(worker_info.cast<std::string>());
  }
  throw py::type_error("dst_worker_info must be None or str");
}

static py::object get_py_worker_info(const std::optional<WorkerInfo>& worker_info) {
  if (!worker_info) {
    return py::none();
  }
  return py::str(worker_info->to_string());
}

GeneralNamingClient connect_naming(const InstanceId &name, const std::string &url) {
  return NAMING_MANAGER->connect_naming(name, url);
}

static std::unique_ptr<Context> create_context(
    const std::string &inst_name,
    uint32_t tp_size,
    uint32_t worker_id,
    uint32_t worker_tp_rank,
    uint32_t device_id,
    const std::vector<size_t> &block_sizes,
    const std::vector<size_t> &token_sizes,
    uint32_t layer_num_blocks,
    uint32_t indexer_blk_ntpb,
    const std::vector<std::vector<uint64_t>> &layers,
    int num_kv_heads,
    uint32_t num_gdn_layers,
    uint32_t hybrid_indexer_token_size,
    uint32_t gdn_conv_elem_size,
    uint32_t gdn_ssm_elem_size,
    std::vector<size_t> conv_state_shape,
    std::vector<size_t> ssm_state_shape,
    std::vector<size_t> gdn_conv_channel_dims,
    uint32_t attn_pack_size) {

  assert(block_sizes.size() == token_sizes.size());
  for (size_t i = 0; i < block_sizes.size(); i++) {
    assert(block_sizes[i] % token_sizes[i] == 0);
  }

  auto context = std::make_unique<Context>(inst_name, worker_id);
  context->set_tp(tp_size, worker_tp_rank);

  // attn_pack_size must be written BEFORE set_block_params so the
  // attn_kernel_blk_ntpb auto-compute path can divide block_size/token_size
  // by it (see context.cpp::set_block_params).
  auto* wi = context->worker_info_mutable();
  wi->attn_pack_size = attn_pack_size == 0 ? 1 : attn_pack_size;

  std::vector<std::vector<LayerInfo>> all_layer_infos;
  for (auto layer: layers) {
    std::vector<LayerInfo> layer_infos;
    for (auto tensor_addr: layer) {
      auto layer_info = LayerInfo(token_sizes[layer_infos.size()], block_sizes[layer_infos.size()], tensor_addr);
      layer_infos.emplace_back(std::move(layer_info));
    }
    all_layer_infos.emplace_back(std::move(layer_infos));
  }

  context->set_block_params(block_sizes, token_sizes, layer_num_blocks, indexer_blk_ntpb);
  context->set_layer_info(device_id, all_layer_infos);

  wi->num_kv_heads = num_kv_heads;
  wi->num_gdn_layers = num_gdn_layers;
  wi->hybrid_indexer_token_size = hybrid_indexer_token_size;
  wi->gdn_conv_elem_size = gdn_conv_elem_size;
  wi->gdn_ssm_elem_size = gdn_ssm_elem_size;
  wi->conv_state_shape = std::move(conv_state_shape);
  wi->ssm_state_shape = std::move(ssm_state_shape);
  wi->gdn_conv_channel_dims = std::move(gdn_conv_channel_dims);

  return context;
}

void init_kv_transfer_client(const std::string &inst_name,
                             uint32_t tp_size,
                             uint32_t worker_id,
                             uint32_t worker_tp_rank,
                             uint32_t device_id,
                             std::vector<size_t> block_sizes,
                             std::vector<size_t> token_sizes,
                             uint32_t layer_num_blocks,
                             const std::string &naming_url,
                             const std::vector<uint64_t> &events,
                             const std::vector<std::vector<uint64_t>> &layers,
                             const std::vector<TransferProtocol> &protocols,
                             int num_kv_heads,
                             uint32_t num_gdn_layers,
                             uint32_t indexer_blk_ntpb,
                             uint32_t hybrid_indexer_token_size,
                             uint32_t gdn_conv_elem_size,
                             uint32_t gdn_ssm_elem_size,
                             std::vector<size_t> conv_state_shape,
                             std::vector<size_t> ssm_state_shape,
                             std::vector<size_t> gdn_conv_channel_dims,
                             uint32_t attn_pack_size) {

  if (KV_CLIENT == nullptr) {
    // Note: valid_ranks, kvt_tp_size (effective) and kvt_tp_rank are
    // computed dynamically per-destination in KvSendStub::TaskContext
    // (depend on dst's engine_tp_size + num_kv_heads).

    LOG(INFO) << "KVT: init kv client for worker(" << inst_name << ":" << worker_id << ") at " << inst_name;
    auto context = create_context(inst_name, tp_size, worker_id, worker_tp_rank,
                                  device_id, block_sizes, token_sizes,
                                  layer_num_blocks, indexer_blk_ntpb, layers,
                                  num_kv_heads, num_gdn_layers,
                                  hybrid_indexer_token_size,
                                  gdn_conv_elem_size, gdn_ssm_elem_size,
                                  std::move(conv_state_shape),
                                  std::move(ssm_state_shape),
                                  std::move(gdn_conv_channel_dims),
                                  attn_pack_size);

    context->set_cuda_barrier(std::make_unique<CudaEventBarrier>(events));

    auto naming_client = NAMING_MANAGER->connect_naming(inst_name, naming_url);
    auto stub_factory = std::make_unique<KvSendStubFactory>(context.get(), std::move(naming_client));
    KV_CLIENT = KvTransferClient::create(std::move(context), protocols, std::move(stub_factory));
    KV_CLIENT->enable_auto_connect();
    auto* ctx = KV_CLIENT->context();
    auto* worker_info = ctx->worker_info_mutable();
    worker_info->transfer_protocols = ctx->support_protocols().value();
    LOG(INFO) << "init_kv_transfer_client. worker_info=" << worker_info->to_string();
  }
  monitor_init();
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

void start_req_send(std::vector<ReqMeta>& metas, size_t stepid, uint32_t substepid) {
  RTASSERT(KV_CLIENT != nullptr);
  KV_CLIENT->start_req_send(metas, stepid, substepid);
}

void submit_req_send2(std::string dst_inst_name,
                      uint32_t dst_worker_id,
                      std::string req_id,
                      uint32_t seen_tokens,
                      uint32_t new_tokens,
                      bool has_last_token,
                      BlockIds src_block_ids,
                      BlockIds dst_block_ids,
                      std::optional<std::string> dst_worker_info,
                      size_t stepid,
                      uint32_t substepid) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->submit_req_send(std::move(dst_inst_name),
                               dst_worker_id,
                               std::move(req_id),
                               seen_tokens,
                               new_tokens,
                               has_last_token,
                               std::move(src_block_ids),
                               std::move(dst_block_ids),
                               std::move(dst_worker_info),
                               stepid,
                               substepid);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

void submit_delta_send(const std::string &req_id,
                       uint32_t seen_tokens,
                       uint32_t new_tokens,
                       bool has_last_token,
                       size_t stepid,
                       uint32_t substepid) {
  if (KV_CLIENT != nullptr) {
    KV_CLIENT->submit_delta_send(req_id, seen_tokens, new_tokens, has_last_token, stepid, substepid);
  } else {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
}

size_t start_send(size_t stepid, size_t sched_tokens) {
  if (KV_CLIENT == nullptr) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
  return KV_CLIENT->start_send(stepid, sched_tokens);
}

void start_send_substep(size_t stepid, uint32_t substepid, std::vector<ReqMeta>& metas) {
  if (KV_CLIENT == nullptr) {
    throw KVTransferException(ErrorKind::INVALID_OPERATION, "kv client is not initialized");
  }
  KV_CLIENT->start_send_substep(stepid, substepid, metas);
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

void init_kv_transfer_server(const std::string &inst_name,
                             uint32_t tp_size,
                             uint32_t worker_id,
                             uint32_t worker_tp_rank,
                             uint32_t device_id,
                             std::vector<size_t> block_sizes,
                             std::vector<size_t> token_sizes,
                             uint32_t layer_num_blocks,
                             const std::string &naming_url,
                             const std::vector<std::vector<uint64_t>> &layers,
                             const std::vector<TransferProtocol> &protocols,
                             int num_kv_heads,
                             uint32_t num_gdn_layers,
                             uint32_t indexer_blk_ntpb,
                             uint32_t hybrid_indexer_token_size,
                             uint32_t gdn_conv_elem_size,
                             uint32_t gdn_ssm_elem_size,
                             std::vector<size_t> conv_state_shape,
                             std::vector<size_t> ssm_state_shape,
                             std::vector<size_t> gdn_conv_channel_dims,
                             uint32_t attn_pack_size) {
  if (KV_SERVER == nullptr) {
    auto context = create_context(inst_name, tp_size, worker_id, worker_tp_rank,
                                  device_id, block_sizes, token_sizes,
                                  layer_num_blocks, indexer_blk_ntpb, layers,
                                  num_kv_heads, num_gdn_layers,
                                  hybrid_indexer_token_size,
                                  gdn_conv_elem_size, gdn_ssm_elem_size,
                                  std::move(conv_state_shape),
                                  std::move(ssm_state_shape),
                                  std::move(gdn_conv_channel_dims),
                                  attn_pack_size);

    if (protocols.size() > 1) {
      throw std::runtime_error("multi-protocols server not support temporarily");
    }

    KV_SERVER_CTX = std::move(context);
    auto ctx = KV_SERVER_CTX.get();
    if (protocols.empty()) {
      auto supported = get_library_support_protocols();
      if (supported.is_support(TransferProtocol::Kind::RDMA_DIRECT)) {
        auto p = TransferProtocol::rdma_direct();
        try {
          KV_SERVER = create_transfer_server(p);
          KV_SERVER->start_server(ctx);
          LOG(INFO) << "KVT: start kvtransfer server with protocol: " + p.to_string();
        } catch (const std::exception &e) {
          KV_SERVER = nullptr;
          LOG(INFO) << "KVT: transfer protocol: " + p.to_string() + " not support, try next ...";
        }
      } else if (supported.is_support(TransferProtocol::Kind::TCP)) {
        auto p = TransferProtocol::tcp();
        try {
          KV_SERVER = create_transfer_server(p);
          KV_SERVER->start_server(ctx);
          LOG(INFO) << "KVT: start kvtransfer server with protocol: " + p.to_string();
        } catch (const std::exception &e) {
          KV_SERVER = nullptr;
          LOG(INFO) << "KVT: transfer protocol: " + p.to_string() + " not support, try next ...";
        }
      }
      if (KV_SERVER == nullptr) {
        throw std::runtime_error("start kvtransfer server failed, because no transfer protocol support");
      }
    } else {
      KV_SERVER = create_transfer_server(protocols[0]);
      KV_SERVER->start_server(ctx);
    }
    auto* worker_info = ctx->worker_info_mutable();
    worker_info->transfer_protocols = ctx->support_protocols().value();
    LOG(INFO) << "init_kv_transfer_server. worker_info=" << worker_info->to_string();
  }
  monitor_init();
}

// Empty means not running in a kvt environment.
std::string current_worker_info(const std::string& kind = "any") {
    Context* ctx = nullptr;

    if (kind == "client" && KV_CLIENT) {
        ctx = KV_CLIENT->context();
    } else if (kind == "server" && KV_SERVER_CTX) {
        ctx = KV_SERVER_CTX.get();
    } else if (kind == "any") {
        if (KV_CLIENT) {
            ctx = KV_CLIENT->context();
        } else if (KV_SERVER_CTX) {
            ctx = KV_SERVER_CTX.get();
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
      .value("RDMA_DIRECT", blade_llm::TransferProtocol::Kind::RDMA_DIRECT)
      .value("TCP", blade_llm::TransferProtocol::Kind::TCP)
      .export_values();

  py::class_<blade_llm::ReqMeta>(m, "ReqMeta")
      .def(py::init<>())  // default constructor
      .def_readwrite("dst_inst", &blade_llm::ReqMeta::dst_inst)
      .def_readwrite("dst_worker", &blade_llm::ReqMeta::dst_worker)
      .def_readwrite("reqid", &blade_llm::ReqMeta::reqid)
      .def_readwrite("seen_tokens", &blade_llm::ReqMeta::seen_tokens)
      .def_readwrite("new_tokens", &blade_llm::ReqMeta::new_tokens)
      .def_readwrite("src_block_ids", &blade_llm::ReqMeta::src_block_ids)
      .def_readwrite("dst_block_ids", &blade_llm::ReqMeta::dst_block_ids)
      .def_property(
          "dst_worker_info",
          [](const blade_llm::ReqMeta& req_meta) {
            return blade_llm::get_py_worker_info(req_meta.dst_worker_info);
          },
          [](blade_llm::ReqMeta& req_meta, const py::object& worker_info) {
            req_meta.dst_worker_info = blade_llm::parse_py_worker_info(worker_info);
          });

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
  m.def("init_kv_transfer_client", &blade_llm::init_kv_transfer_client, "init kv transfer client;",
      py::arg("inst_name"), py::arg("tp_size"), py::arg("worker_id"),
      py::arg("worker_tp_rank"), py::arg("device_id"),
      py::arg("block_sizes"), py::arg("token_sizes"),
      py::arg("layer_num_blocks"), py::arg("naming_url"),
      py::arg("events"), py::arg("layers"), py::arg("protocols"),
      py::arg("num_kv_heads") = -1,
      py::arg("num_gdn_layers") = 3,
      py::arg("indexer_blk_ntpb") = 0,
      py::arg("hybrid_indexer_token_size") = 0,
      py::arg("gdn_conv_elem_size") = 1,
      py::arg("gdn_ssm_elem_size") = 1,
      py::arg("conv_state_shape") = std::vector<size_t>{},
      py::arg("ssm_state_shape") = std::vector<size_t>{},
      py::arg("gdn_conv_channel_dims") = std::vector<size_t>{},
      py::arg("attn_pack_size") = 1);
  m.def("add_target", &blade_llm::add_target, "add target to kv client;");
  m.def("submit_req_send2", &blade_llm::submit_req_send2, "submit kv send to kv client;");
  m.def("start_req_send", &blade_llm::start_req_send, "submit kv send to kv client;");
  m.def("submit_delta_send", &blade_llm::submit_delta_send, "submit kv token to kv client;");
  m.def("start_send", &blade_llm::start_send, "start to send submitted kv data;");
  m.def("start_send_substep", &blade_llm::start_send_substep, "submit bypass substep metas;");
  m.def("notify_event_record", &blade_llm::notify_event_record, "record kv send events;");
  m.def("flush_send", &blade_llm::flush_send, "check if all kv send tasks are done;");
  // server
  m.def("init_kv_transfer_server", &blade_llm::init_kv_transfer_server, "init kv transfer server;",
      py::arg("inst_name"), py::arg("tp_size"), py::arg("worker_id"),
      py::arg("worker_tp_rank"), py::arg("device_id"),
      py::arg("block_sizes"), py::arg("token_sizes"),
      py::arg("layer_num_blocks"), py::arg("naming_url"),
      py::arg("layers"), py::arg("protocols"),
      py::arg("num_kv_heads") = -1,
      py::arg("num_gdn_layers") = 3,
      py::arg("indexer_blk_ntpb") = 0,
      py::arg("hybrid_indexer_token_size") = 0,
      py::arg("gdn_conv_elem_size") = 1,
      py::arg("gdn_ssm_elem_size") = 1,
      py::arg("conv_state_shape") = std::vector<size_t>{},
      py::arg("ssm_state_shape") = std::vector<size_t>{},
      py::arg("gdn_conv_channel_dims") = std::vector<size_t>{},
      py::arg("attn_pack_size") = 1);
  // common
  m.def("current_worker_info", &blade_llm::current_worker_info, "get current worker info;", py::arg("kind") = "any");
  m.def("lib_support_transfer_protocols", &blade_llm::support_transfer_protocols, "get supported transfer types");
}
