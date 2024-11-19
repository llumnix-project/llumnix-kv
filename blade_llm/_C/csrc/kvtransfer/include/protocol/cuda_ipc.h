#ifndef KVTRANSFER_INCLUDE_PROTOCOL_CUDA_IPC_H_
#define KVTRANSFER_INCLUDE_PROTOCOL_CUDA_IPC_H_

#pragma once

#include "channel.h"
#include "server.h"
#include "utils/cuda_helper.h"
#include "utils/thread_pool.h"

namespace blade_llm {

class CudaIpcContext : public IProtocolContext {
 public:
  explicit CudaIpcContext(int device, bool is_server = false) : device_id_(device), is_server_(is_server) {}
  void init(Context *) override;
  bool check_support() override;
  [[nodiscard]] const TransferProtocol &protocol() const override;
  [[nodiscard]] const void *get_layer_ptr(uint32_t layer_idx) const;
  [[nodiscard]] const cudaIpcHandles &get_ipc_handles() const;
  [[nodiscard]] size_t num_layers() const;
 private:
  bool is_server_;
  int device_id_{0};
  bool is_cuda_ipc_supported_{false};
  std::vector<const void *> src_layer_ptrs_;
  cudaIpcHandles ipc_handles_;
  TransferProtocol protocol_{TransferProtocol::Kind::CUDA_IPC};
};

class CudaIpcWrite : public noncopyable {
 public:
  explicit CudaIpcWrite(CudaIpcContext *ctx) : ctx_(ctx), is_connected_(false) {
    if (ctx_ == nullptr) {
      throw std::runtime_error("cuda context not registered;");
    }
  };

  void init(const cudaIpcHandles *);
  void write(uint32_t layer_idx, const std::vector<IpcBlock> &data);
  void close();
  ~CudaIpcWrite() noexcept;
 private:
  bool is_connected_;
  std::vector<void *> dst_ipc_ptr_;
  CudaIpcContext *ctx_;
};

class SocketWriter : public noncopyable {
 public:
  SocketWriter(InstanceId inst_id, WorkerId worker_id) :
      src_inst_id_(inst_id),
      src_worker_id_(worker_id) {};
  void connect(const WorkerInfo &dst);
  void write(const RequestId &, const std::vector<uint32_t> &block_ids);
  void close();
  ~SocketWriter();
 private:
  int socket_fd_{-1};
  InstanceId src_inst_id_;
  WorkerId src_worker_id_;
  InstanceId dst_inst_id_{INVALID_INST_WORKER_ID};
  WorkerId dst_worker_id_{INVALID_INST_WORKER_ID};
};

class CudaIpcChannel : public IChannel, public noncopyable {
 public:
  CudaIpcChannel(InstanceId inst_id, WorkerId worker_id, CudaIpcContext *ctx) :
      data_writer_(ctx),
      notify_writer_(inst_id, worker_id) {};
  void connect(const WorkerInfo &dst) override;
  void send_data(size_t layer_index, const std::vector<IpcBlock> &data) override;
  void send_notification(IIterator<const ReqSendTask *> *reqs) override;
  void flush() override;
  void close() override;
  ~CudaIpcChannel() override;

 private:
  CudaIpcWrite data_writer_;
  SocketWriter notify_writer_;
};

class CudaTransferServer : public ITransferServer {
 public:
  CudaTransferServer();
  void start_server(ITransferService *service, Context *ctx) override;
  void handle_connect_reqs();
  void shutdown();
  ~CudaTransferServer() override;

 private:
  InstanceId inst_id_;
  WorkerId worker_id_;
  int socket_fd_{-1};
  ThreadPool pool_;
  ITransferService *service_{nullptr};
  std::atomic_bool shutdown_{false};
};
}
#endif //KVTRANSFER_INCLUDE_PROTOCOL_CUDA_IPC_H_
