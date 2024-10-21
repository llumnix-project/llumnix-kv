#include <sys/un.h>
#include <stdexcept>
#include <cassert>
#include "naming.h"
#include "protocol/cuda_ipc.h"
#include "utils/cuda_helper.h"
#include "utils/socket_helper.h"
#include "logging.h"

#define MAX_REQ_ID_LENGTH (255)
#define SOCK_PATH "/tmp/kvt-ipc-%d-%d.sock"

namespace blade_llm {
CudaIpcWrite::CudaIpcWrite(Context *ctx) : is_connected_(false) {
  auto &src_layers = ctx->layer_data_address();
  src_ptr_.resize(src_layers.size());
  for (auto i = 0; i < src_layers.size(); ++i) {
    src_ptr_[i] = reinterpret_cast<const void *>(src_layers[i]);
  }
}

void CudaIpcWrite::init(const cudaIpcHandles *handles) {
  auto num_layers = src_ptr_.size();
  dst_ipc_ptr_.resize(num_layers);
  cuda_open_ipc_handle(handles, num_layers, dst_ipc_ptr_.data());
  is_connected_ = true;
}

void CudaIpcWrite::write(uint32_t layer_idx, const std::vector<IpcBlock> &data) {
  if (!is_connected_) {
    throw std::runtime_error("cuda ipc not connect;");
  }

  for (auto &block : data) {
    const char *src = (const char *) src_ptr_[layer_idx] + block.src_offset;
    char *dst = (char *) dst_ipc_ptr_[layer_idx] + block.dst_offset;
    cuda_d2d_mem_copy(dst, src, block.length);
  }
}

void CudaIpcWrite::close() {
  if (is_connected_) {
    for (auto &i : dst_ipc_ptr_) {
      cuda_close_ipc_handle(i);
    }
    is_connected_ = false;
  }
}

CudaIpcWrite::~CudaIpcWrite() noexcept {
  close();
}

SocketWriter::SocketWriter(const WorkerInfo &src_info) :
    src_inst_id_(src_info.inst_id),
    src_worker_id_(src_info.worker_id) {};

void SocketWriter::connect(const WorkerInfo &dst_info) {
  if (socket_fd_ == -1) {
    dst_inst_id_ = dst_info.inst_id;
    dst_worker_id_ = dst_info.worker_id;
    auto path = dst_info.addr_url;
    if (!try_connect_uds(path, &socket_fd_)) {
      throw std::runtime_error("fail to connect uds server on target worker;");
    }
    write_sock(socket_fd_, (char *) &src_inst_id_, sizeof(uint32_t));
    write_sock(socket_fd_, (char *) &src_worker_id_, sizeof(uint32_t));
  }
}

void SocketWriter::write(const RequestId &req_id, const std::vector<uint32_t> &block_ids) {
  if (socket_fd_ == -1) {
    throw std::runtime_error("socket writer not connected;");
  }
  auto req_len = req_id.length();
  assert(req_len < 255);
  uint8_t req_len_byte = req_len;

  write_sock(socket_fd_, (char *) &req_len_byte, sizeof(uint8_t));
  write_sock(socket_fd_, req_id.data(), req_len);
  uint32_t num_blocks = block_ids.size();
  write_sock(socket_fd_, (char *) &num_blocks, sizeof(uint32_t));
  write_sock(socket_fd_, (char *) block_ids.data(), num_blocks * sizeof(uint32_t));
}

void SocketWriter::close() {
  if (socket_fd_ != -1) {
    close_sock(socket_fd_);
    socket_fd_ = -1;
  }
}

SocketWriter::~SocketWriter() {
  close();
}

void CudaIpcChannel::connect(const blade_llm::WorkerInfo &dst_info) {
  cudaIpcHandles ipc_handles;
  static_assert(sizeof(ipc_handles) <= MAX_OTHER_INFO_LEN);
  memcpy(ipc_handles.buf, dst_info.other_info, sizeof(ipc_handles));
  data_writer_.init(&ipc_handles);
  notify_writer_.connect(dst_info);
}
void CudaIpcChannel::send_data(size_t layer_index,
                               const std::vector<IpcBlock> &data) {
  data_writer_.write(layer_index, data);
}
void CudaIpcChannel::send_notification(IIterator<const RequestInfo *> *reqs) {
  for (;;) {
    auto v = reqs->next();
    if (v.has_value()) {
      auto req = v.value();
      notify_writer_.write(req->req_id, req->dst_blocks());
    } else {
      break;
    }
  }
}

void CudaIpcChannel::flush() { /* do nothing  */ }
void CudaIpcChannel::close() {
  data_writer_.close();
  notify_writer_.close();
}
CudaIpcChannel::~CudaIpcChannel() {
  data_writer_.close();
  notify_writer_.close();
}

void recv_transfer_done(InstanceId inst_id,
                        WorkerId worker_id,
                        int sock_fd,
                        ITransferService *service,
                        const std::atomic_bool *shutdown) {
  char buf[MAX_REQ_ID_LENGTH + sizeof(uint32_t)];
  uint8_t req_id_len;
  while (!shutdown->load(std::memory_order_relaxed)) {
    auto read = try_read(sock_fd, (char *) &req_id_len, sizeof(uint8_t), 10);
    if (read > 0) {
      read_sock(sock_fd, buf, req_id_len + sizeof(uint32_t));
      std::string req_id(buf, req_id_len);
      uint32_t num_blocks;
      memcpy(&num_blocks, buf + req_id_len, sizeof(uint32_t));
      LOG(INFO) << "KVT cuda_ipc server: receive transfer finish notification of req: "
                << req_id << " with " << num_blocks << " blocks;";

      std::vector<uint32_t> block_ids(num_blocks);
      auto size = num_blocks * sizeof(uint32_t);
      auto read_len = read_sock(sock_fd, (char *) block_ids.data(), size);
      if (read_len != size) {
        LOG(ERROR) << "KVT cuda_ipc server: unexpected message size of notification of req"
                   << req_id << ", expect " << size
                   << "bytes, but only read" << read_len << " bytes";
      } else {
        service->on_recv(inst_id, worker_id, req_id, std::move(block_ids));
      }
    }
  }
  LOG(INFO) << "KVT cuda_ipc server: connection from (" << inst_id << ", " << worker_id << ") exit;";
  close_sock(sock_fd);
}

CudaTransferServer::CudaTransferServer(): pool_(8) {};

void CudaTransferServer::start_server(ITransferService *service, Context *ctx) {
  if (service == nullptr) {
    throw std::runtime_error("KvTransferService should not be null;");
  }

  auto *worker_info = ctx->worker_info_mutable();
  if (!cuda_check_ipc_support(ctx->device_id())) {
    throw std::runtime_error("cuda ipc not support; ");
  }

  service_ = service;
  inst_id_ = worker_info->inst_id;
  worker_id_ = worker_info->worker_id;
  auto layers = ctx->layer_data_address();
  if (layers.empty()) {
    throw std::runtime_error("layer data address should not be empty;");
  }
  cudaIpcHandles ipc_handles;
  cuda_create_ipc_handles(layers.data(), layers.size(), &ipc_handles);
  static_assert(sizeof(ipc_handles) <= MAX_OTHER_INFO_LEN);
  memcpy(worker_info->other_info, ipc_handles.buf, sizeof(ipc_handles));
  char* sock_path = worker_info->addr_url;
  sprintf(sock_path, SOCK_PATH, inst_id_, worker_id_);
  start_uds_server(sock_path, &socket_fd_);
  pool_.spawn(&CudaTransferServer::handle_connect_reqs, this);
  LOG(INFO) << "CudaTransferServer at " << inst_id_ << ":" << worker_id_ << " started; ";
}

void CudaTransferServer::handle_connect_reqs() {
  while (!shutdown_.load(std::memory_order_relaxed)) {
    LOG(INFO) << "KVT cuda_ipc server: (" << inst_id_ << ":" << worker_id_ << ") wait connection ...; ";
    int client_sock = wait_conn(socket_fd_, 2);
    if (client_sock != -1) {
      uint32_t src_inst_id;
      uint32_t src_worker_id;

      if (read_sock(client_sock, (char *) &src_inst_id, sizeof(uint32_t)) != sizeof(uint32_t)) {
        LOG(WARNING) << "KVT cuda_ipc server: accept unknown connection;";
        continue;
      }
      if (read_sock(client_sock, (char *) &src_worker_id, sizeof(uint32_t)) != sizeof(uint32_t)) {
        LOG(WARNING) << "KVT cuda_ipc server: accept unknown connection;";
        continue;
      }
      if (src_worker_id >= MAX_WORKERS_PER_INST) {
        LOG(WARNING) << "KVT cuda_ipc server: accept connection from invalid worker: ("
                     << src_inst_id << ":" << src_worker_id << "), discard.";
        continue;
      }

      LOG(INFO) << "KVT cuda_ipc server: uds server: accept connection from ("
                << src_inst_id << ":" << src_worker_id << ")";
      pool_.spawn(&recv_transfer_done, src_inst_id, src_worker_id, client_sock, service_, &shutdown_);
    }
  }
  LOG(INFO) << "KVT cuda_ipc server: uds server exit ...";
}
void CudaTransferServer::shutdown() {
  shutdown_.store(true, std::memory_order_release);
  if (socket_fd_ != -1) {
    close_sock(socket_fd_);
  }
  socket_fd_ = -1;
}
CudaTransferServer::~CudaTransferServer() {
  shutdown();
}
} // namespace blade_llm