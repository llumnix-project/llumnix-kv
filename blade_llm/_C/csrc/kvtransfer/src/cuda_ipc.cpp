#include <sys/un.h>
#include <stdexcept>
#include <cassert>
#include "naming.h"
#include "protocol/cuda_ipc.h"
#include "utils/cuda_helper.h"
#include "utils/socket_helper.h"
#include "thrid_party/logging.h"

#define MAX_REQ_ID_LENGTH (255)

namespace blade_llm {


bool write_handshake(int fd, const InstanceId& name, WorkerId id) {
  if (fd == -1) {
    LOG(ERROR) << "write_hand_shake: socket not connected;";
    return false;
  }

  uint32_t inst_name_len = name.length();
  if (inst_name_len == 0 || inst_name_len > MAX_INSTANCE_NAME_LEN) {
    throw std::runtime_error("invalid instance name length: " + std::to_string(inst_name_len));
  }
  uint64_t head = (uint64_t) name.length() << 32 | id;
  write_sock(fd, (char*)&head, sizeof(head));
  write_sock(fd, name.data(), inst_name_len);
  return true;
}

bool read_handshake(int fd, InstanceId& name, WorkerId& id) {
  if (fd == -1) {
    LOG(ERROR) << "req_hand_shake: socket not connected;";
    return false;
  }

  uint64_t head;
  auto ret = read_sock(fd, (char*)&head, sizeof(head));
  if (ret < 0 || ret != sizeof(head)) {
    LOG(ERROR) << "unexpected eof of socket;";
    return false;
  }

  id = head & 0xffffffff;
  uint32_t name_len = head >> 32;
  if (name_len == 0 || name_len > MAX_INSTANCE_NAME_LEN) {
    throw std::runtime_error("invalid request id length: " + std::to_string(name_len));
  }
  name.resize(name_len);
  ret = read_sock(fd, name.data(), name_len);
  if (ret < 0 || ret != int(name_len)) {
    LOG(ERROR) << "unexpected eof of socket;";
    return false;
  }
  return true;
}

bool write_req(int fd, const RequestId & req_id, const std::vector<uint32_t> & block_ids) {
  if (fd == -1) {
    LOG(ERROR) << "write_req: socket not connected;";
    return false;
  }
  uint32_t req_len = req_id.length();
  if (req_len == 0 || req_len > MAX_REQ_ID_LENGTH) {
    throw std::runtime_error("invalid request id length: " + std::to_string(req_len));
  }
  uint32_t num_blocks = block_ids.size();
  uint64_t head = (uint64_t) req_len << 32 | num_blocks;
  // only write to local through unix domain socket instead network;
  write_sock(fd, (char*)&head, sizeof(head));
  write_sock(fd, req_id.data(), req_len);
  write_sock(fd, (char *) block_ids.data(), num_blocks * sizeof(uint32_t));
  return true;
}

bool read_req(int fd, RequestId& req_id, std::vector<uint32_t> & blocks) {
  if (fd == -1) {
    LOG(ERROR) << "req_req: socket not connected;";
    return false;
  }

  uint64_t head;
  auto ret = read_sock(fd, (char*)&head, sizeof(head));
  if (ret < 0 || ret != sizeof(head)) {
    LOG(ERROR) << "unexpected eof of socket;";
    return false;
  }

  uint32_t req_len = head >> 32;
  if (req_len == 0 || req_len > MAX_REQ_ID_LENGTH) {
    throw std::runtime_error("invalid request id length: " + std::to_string(req_len));
  }
  req_id.resize(req_len);
  ret = read_sock(fd, req_id.data(), req_len);
  if (ret < 0 || ret != int(req_len)) {
    LOG(ERROR) << "unexpected eof of socket;";
    return false;
  }
  uint32_t num_blocks = head & 0xffffffff;
  blocks.resize(num_blocks);
  ret = read_sock(fd, (char *) blocks.data(), num_blocks * sizeof(uint32_t));
  if (ret < 0 || uint32_t(ret) != num_blocks * sizeof(uint32_t)) {
    LOG(ERROR) << "unexpected eof of socket;";
    return false;
  }
  return true;
}

void CudaIpcWrite::init(const cudaIpcHandles *handles) {
  auto num_layers = ctx_->num_layers();
  dst_ipc_ptr_.resize(num_layers);
  cuda_open_ipc_handle(handles, num_layers, dst_ipc_ptr_.data());
  is_connected_ = true;
}

void CudaIpcWrite::write(uint32_t layer_idx, const std::vector<IpcBlock> &data) {
  if (!is_connected_) {
    throw std::runtime_error("cuda ipc not connect;");
  }

  for (auto &block : data) {
    const char *src_ptr = (const char *) ctx_->get_layer_ptr(layer_idx);
    const char *src = src_ptr + block.src_offset;
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

void SocketWriter::connect(const WorkerInfo &dst_info) {
  if (socket_fd_ == -1) {
    dst_inst_id_ = dst_info.inst_id;
    dst_worker_id_ = dst_info.worker_id;
    const auto& path = dst_info.addr;
    if (!try_connect_uds(path.c_str(), &socket_fd_)) {
      throw std::runtime_error("fail to connect uds server on target worker;");
    }
    if (!write_handshake(socket_fd_, src_inst_id_, src_worker_id_)) {
      throw std::runtime_error("fail to establish connection to " + dst_inst_id_ + ", " + std::to_string(dst_worker_id_));
    }
  }
}

void SocketWriter::write(const RequestId &req_id, const std::vector<uint32_t> &block_ids) {
  write_req(socket_fd_, req_id, block_ids);
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
  if (sizeof(ipc_handles) > dst_info.other_info.size()) {
    throw std::runtime_error("invalid cuda ipc handles format;");
  }
  memcpy(ipc_handles.buf, dst_info.other_info.data(), sizeof(ipc_handles));
  data_writer_.init(&ipc_handles);
  notify_writer_.connect(dst_info);
}

void CudaIpcChannel::register_data(std::vector<IpcBlock>& data, TPKind) {
  auto& self = *this;
  assert(self.data_ == nullptr);
  self.data_ = &data;
}

void CudaIpcChannel::send_data(size_t layer_index) {
  data_writer_.write(layer_index, *data_);
}

void CudaIpcChannel::flush(std::string&) {
  this->data_ = nullptr;
}

void CudaIpcChannel::send_notification(const std::vector<const ReqSendTask*>& reqs) {
  for (const auto* req : reqs) {
    notify_writer_.write(req->req_id(), req->dst_blocks());
  }
}

void CudaIpcChannel::close() {
  data_writer_.close();
  notify_writer_.close();
}
CudaIpcChannel::~CudaIpcChannel() {
  data_writer_.close();
  notify_writer_.close();
}

void recv_transfer_done(const InstanceId& inst_id,
                        WorkerId worker_id,
                        int sock_fd,
                        ITransferService *service,
                        const std::atomic_bool *shutdown) {
  while (!shutdown->load(std::memory_order_relaxed)) {
    std::string req_id;
    std::vector<uint32_t> block_ids;
    auto ret = read_req(sock_fd, req_id, block_ids);
    if (ret) {
      service->on_recv(inst_id, worker_id, req_id, std::move(block_ids));
    } else {
      break;
    }
  }
  LOG(INFO) << "KVT cuda_ipc server: connection from (" << inst_id << ", " << worker_id << ") exit;";
  close_sock(sock_fd);
}

CudaTransferServer::CudaTransferServer() : pool_(8) {};

void CudaTransferServer::start_server(ITransferService *service, Context *ctx) {
  if (service == nullptr) {
    throw std::runtime_error("KvTransferService should not be null;");
  }

  auto proto_ctx = std::make_unique<CudaIpcContext>(ctx->device_id(), true);
  if (!proto_ctx->check_support()) {
    throw std::runtime_error("can't start cuda transfer server as cuda_ipc protocol not support;");
  }
  ctx->register_protocol(std::move(proto_ctx));
  TransferProtocol proto = TransferProtocol::cuda_ipc();
  auto *worker_info = ctx->worker_info_mutable();
  service_ = service;
  inst_id_ = worker_info->inst_id;
  worker_id_ = worker_info->worker_id;

  auto cuda_ctx = ctx->get_protocol_ctx<CudaIpcContext>(proto);
  if (cuda_ctx == nullptr) {
    throw std::runtime_error("cuda ipc context not found;");
  }
  const auto &ipc_handles = cuda_ctx->get_ipc_handles();
  worker_info->other_info.resize(sizeof(ipc_handles));
  memcpy(worker_info->other_info.data(), ipc_handles.buf, sizeof(ipc_handles));
  std::string sock_path("/tmp/kvt-ipc-" + inst_id_ + "-" + std::to_string(worker_id_) + ".sock");
  start_uds_server(sock_path.c_str(), &socket_fd_);
  worker_info->addr = sock_path;
  pool_.spawn(&CudaTransferServer::handle_connect_reqs, this);
  LOG(INFO) << "KVT: CudaTransferServer of (" << inst_id_ << ":" << worker_id_ << ") started at "
            << worker_info->addr;
}

void CudaTransferServer::handle_connect_reqs() {
  while (!shutdown_.load(std::memory_order_relaxed)) {
    LOG(INFO) << "KVT: cuda_ipc server: (" << inst_id_ << ":" << worker_id_ << ") wait new connection ...; ";
    int client_sock = wait_conn(socket_fd_, 2);
    if (client_sock != -1) {
      InstanceId src_inst_id;
      uint32_t src_worker_id;
      if (read_handshake(client_sock, src_inst_id, src_worker_id)) {
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

void CudaIpcContext::init(Context *ctx) {
  auto &layer_addrs = ctx->layer_data_address();
  src_layer_ptrs_.resize(layer_addrs.size());
  for (size_t i = 0; i < layer_addrs.size(); ++i) {
    src_layer_ptrs_[i] = reinterpret_cast<const void *>(layer_addrs[i]);
  }
  cuda_create_ipc_handles(layer_addrs.data(), layer_addrs.size(), &ipc_handles_);
  is_inited_ = true;
}

bool CudaIpcContext::check_support() {
  if (is_server_) {
    if (!is_cuda_ipc_supported_) {
      is_cuda_ipc_supported_ = cuda_check_ipc_support(device_id_);
      return is_cuda_ipc_supported_;
    }
  }
  return true;
}

const TransferProtocol &CudaIpcContext::protocol() const {
  return protocol_;
}
const void *CudaIpcContext::get_layer_ptr(uint32_t layer_idx) const {
  if (layer_idx > src_layer_ptrs_.size()) {
    LOG(FATAL) << "KVT cuda_ipc: invalid layer index: " << layer_idx;
  }
  return src_layer_ptrs_[layer_idx];
}
const cudaIpcHandles &CudaIpcContext::get_ipc_handles() const {
  return ipc_handles_;
}
size_t CudaIpcContext::num_layers() const {
  return src_layer_ptrs_.size();
}
} // namespace blade_llm
