#ifndef KVTRANSFER_INCLUDE_PROTOCOL_CUDA_IPC_H_
#define KVTRANSFER_INCLUDE_PROTOCOL_CUDA_IPC_H_

#pragma once

#include "channel.h"
#include "server.h"
#include "utils/cuda_helper.h"
#include "utils/thread_pool.h"

namespace blade_llm {
class CudaIpcWrite : public noncopyable {
 public:
  explicit CudaIpcWrite(Context *);
  void init(const cudaIpcHandles *);
  void write(uint32_t layer_idx, const std::vector<IpcBlock> &data);
  void close();
  ~CudaIpcWrite() noexcept;
 private:
  bool is_connected_;
  std::vector<void *> dst_ipc_ptr_;
  std::vector<const void *> src_ptr_;
};

class SocketWriter : public noncopyable {
 public:
  explicit SocketWriter(const WorkerInfo &src);
  void connect(const WorkerInfo &dst);
  void write(const RequestId &, const std::vector<uint32_t> &block_ids);
  void close();
  ~SocketWriter();
 private:
  int socket_fd_{-1};
  uint32_t src_inst_id_;
  uint32_t src_worker_id_;
  uint32_t dst_inst_id_{INVALID_INST_WORKER_ID};
  uint32_t dst_worker_id_{INVALID_INST_WORKER_ID};
};

class CudaIpcChannel : public IChannel, public noncopyable {
 public:
  explicit CudaIpcChannel(Context *ctx) :
      data_writer_(ctx),
      notify_writer_(ctx->worker_info()) {};
  void connect(const WorkerInfo &dst) override;
  void send_data(size_t layer_index, const std::vector<IpcBlock> &data) override;
  void send_notification(IIterator<const RequestInfo *> *reqs) override;
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
