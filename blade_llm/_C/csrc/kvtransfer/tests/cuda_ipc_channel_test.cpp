
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "context.h"
#include "naming/shm_naming.h"
#include "protocol/cuda_ipc.h"
#include "utils/cuda_helper.h"
#include "utils/shm_helper.h"
#include "utils/socket_helper.h"
#include "utils/iterator.h"
#include "thrid_party/logging.h"

#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#define sync_file "cuda_write_test"
#define shm_naming_file "cuda_ipc_test"
#define sock_file "/tmp/sock_write_test.sock"

using namespace std;

using namespace blade_llm;

typedef struct cudaWriteSyncInfo_st {
  int barrier;
  cudaIpcHandles handles;
} cudaWriteSyncInfo;

TEST(CudaIpcTest, TestCudaWrite) {
  {
    sharedMemoryInfo shm_info;
    volatile cudaWriteSyncInfo* sync_info;
    auto ret = sharedMemoryCreate(sync_file, sizeof(*sync_info), &shm_info);
    EXPECT_EQ(0, ret);
    sync_info = (volatile cudaWriteSyncInfo *) shm_info.addr;
    memset((void *) sync_info, 0, sizeof(*sync_info));
  }

  auto pid = fork();
  if (pid > 0) {
    // parent process;
    cuda_set_device(0);
    void *layer_0, *layer_1;
    cuda_malloc(&layer_0, 4 * KB);
    cuda_malloc(&layer_1, 4 * KB);
    cudaMemset(layer_0, 0, 4 * KB);
    cudaMemset(layer_1, 0, 4 * KB);
    std::vector<uint64_t> layer_addrs(2);
    layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
    layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
    EXPECT_TRUE(cuda_check_ipc_support(0));

    cudaIpcHandles handles;
    cuda_create_ipc_handles(layer_addrs.data(), layer_addrs.size(), &handles);
    sharedMemoryInfo shm_info_p;
    volatile cudaWriteSyncInfo* sync_info;
    sharedMemoryOpen(sync_file, sizeof(*sync_info), &shm_info_p);
    sync_info = (volatile cudaWriteSyncInfo*) shm_info_p.addr;

    memcpy((void *)sync_info->handles.buf, handles.buf, sizeof(cudaIpcHandles));
    cpu_atomic_add32(&sync_info->barrier, 1);
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
      char written[128];
      cuda_d2h_mem_copy(written, layer_0, 128);
      for (char i : written) {
        EXPECT_EQ(i, 11);
      }
      cuda_d2h_mem_copy(written, ((char*)layer_1 + 128), 128);
      for(char i: written) {
        EXPECT_EQ(i, 12);
      }
    } else {
      LOG(ERROR) << "child process exit abnormally";
      EXPECT_FALSE(true);
    }
    sharedMemoryClose(&shm_info_p);
    cudaFree(layer_0);
    cudaFree(layer_1);
  } else {
    // child process;
    cuda_set_device(0);
    void *layer_0, *layer_1;
    cuda_malloc(&layer_0, 4 * KB);
    cuda_malloc(&layer_1, 4 * KB);
    cudaMemset(layer_0, 11, 4 * KB);
    cudaMemset(layer_1, 12, 4 * KB);
    std::vector<uint64_t> layer_addrs(2);
    layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
    layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
    Context ctx("0", 0, 1);
    ctx.set_block_params(4 * KB, KB, 1);
    ctx.set_layer_data_address(0, layer_addrs);
    auto cuda_ctx = std::make_unique<CudaIpcContext>(ctx.device_id());
    EXPECT_TRUE(cuda_ctx->check_support());
    ctx.register_protocol(std::move(cuda_ctx));
    auto proto_ctx = ctx.get_protocol_ctx<CudaIpcContext>(TransferProtocol::Kind::CUDA_IPC);
    EXPECT_TRUE(proto_ctx != nullptr);
    CudaIpcWrite cu_writer(proto_ctx);
    sharedMemoryInfo shm_info_c;
    volatile cudaWriteSyncInfo* sync_info;
    sharedMemoryOpen(sync_file, sizeof(*sync_info), &shm_info_c);
    sync_info = (volatile cudaWriteSyncInfo*) shm_info_c.addr;
    volatile int* barrier = &sync_info->barrier;
    while(*barrier < 1);
    // copy data from local to remote through ipc handle;
    cudaIpcHandles handles;
    memcpy(handles.buf, (void *)sync_info->handles.buf, sizeof(cudaIpcHandles));
    cu_writer.init(&handles);
    cu_writer.write(0, {{0, 0, 128}});
    cu_writer.write(1, {{0, 128, 128}});
    cu_writer.close();
    cuda_free(layer_0);
    cuda_free(layer_1);
  }
}

TEST(CudaIpcTest, TestSocketWrite) {
  int server_sock;
  start_uds_server(sock_file, &server_sock);
  EXPECT_TRUE(server_sock != -1);
  std::thread t([&]() {
    WorkerInfo src(1, 4);
    SocketWriter sock_writer(1, 4);

    WorkerInfo dst(0, 0);
    dst.addr = sock_file;
    sock_writer.connect(dst);

    sock_writer.write("test_req", {0, 2, 4, 6});
    sock_writer.close();
  });

  int conn_sock = -1;
  size_t retry_cnt = 0;
  while (retry_cnt < 3 && conn_sock == -1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    conn_sock = wait_conn(server_sock, 100);
    retry_cnt++;
  }
  EXPECT_TRUE(conn_sock != -1);
  uint32_t inst_id, worker_id;
  read_sock(conn_sock, (char *)&inst_id, sizeof(inst_id));
  read_sock(conn_sock, (char *)&worker_id, sizeof(worker_id));
  EXPECT_EQ(inst_id, 1);
  EXPECT_EQ(worker_id, 4);

  uint8_t req_id_len;
  read_sock(conn_sock, (char *)&req_id_len, sizeof(req_id_len));
  EXPECT_EQ(req_id_len, 8);
  char req_id[req_id_len + 1];
  memset(req_id, 0, req_id_len + 1);
  read_sock(conn_sock, req_id, req_id_len);
  EXPECT_STREQ(req_id, "test_req");
  uint32_t num_blocks;
  read_sock(conn_sock, (char *)&num_blocks, sizeof(num_blocks));
  EXPECT_EQ(num_blocks, 4);

  for (uint32_t i = 0; i < num_blocks; i++) {
    uint32_t block_id;
    read_sock(conn_sock, (char *)&block_id, sizeof(block_id));
    EXPECT_EQ(block_id, i * 2);
  }
  t.join();
  close_sock(server_sock);
}

class MockTransferService : public ITransferService {
 public:
  MOCK_METHOD(void, on_recv, (InstanceId, uint32_t, (const std::string&), (std::vector<uint32_t> &&)), (override));
};

TEST(CudaIpcTest, TestTransfer) {
  auto n_server = ShmNamingServer(shm_naming_file);
  n_server.start();
  auto pid = fork();
  if (pid > 0) {
    // parent process as server side;
    LOG(INFO) << "cuda_set_device(0)";
    cuda_set_device(0);
    LOG(INFO) << "cuda_set_device(0) ok";
    void *layer_0, *layer_1;
    cuda_malloc(&layer_0, 4 * KB);
    cuda_malloc(&layer_1, 4 * KB);
    cudaMemset(layer_0, 0, 4 * KB);
    cudaMemset(layer_1, 0, 4 * KB);
    std::vector<uint64_t> layer_addrs(2);
    layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
    layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
    Context ctx("0", 0, 0);
    ctx.set_layer_data_address(0, layer_addrs);
    ctx.set_block_params(KB, 128, 4);
    CudaTransferServer server;
    MockTransferService service;
    atomic_bool recv_done{false};
    EXPECT_CALL(service, on_recv).WillOnce([&recv_done] (uint32_t id1, uint32_t id2, const string& rid, vector<uint32_t>&& b) {
      EXPECT_EQ(id1, 1);
      EXPECT_EQ(id2, 0);
      EXPECT_EQ(rid, "REQ-0001");
      EXPECT_EQ(b.size(), 2);
      EXPECT_EQ(b[0], 0);
      EXPECT_EQ(b[1], 1);
      recv_done.store(true, std::memory_order_release);
    });
    server.start_server(&service, &ctx);
    ShmNamingClient naming;
    naming.connect(shm_naming_file);
    naming.register_worker(ctx.worker_info());

    for (int i = 0; i < 128; ++i) {
      if (!recv_done.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } else {
        break;
      }
    }
    EXPECT_TRUE(recv_done.load(std::memory_order_relaxed));
    if (recv_done.load(std::memory_order_relaxed)) {
      char written[128];
      cuda_d2h_mem_copy(written, layer_0, 128);
      for (char i : written) {
        EXPECT_EQ(i, 11);
      }
      cuda_d2h_mem_copy(written, layer_1, 128);
      for (char i : written) {
        EXPECT_EQ(i, 12);
      }
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status)) {
      LOG(ERROR) << "child process exit abnormally";
      EXPECT_FALSE(true);
    }
    server.shutdown();
    cudaFree(layer_0);
    cudaFree(layer_1);
  } else {
    // child process as client side;
    LOG(INFO) << "cuda_set_device(1)";
    cuda_set_device(1);
    LOG(INFO) << "cuda_set_device(1) ok";
    void *layer_0, *layer_1;
    cuda_malloc(&layer_0, 4 * KB);
    cuda_malloc(&layer_1, 4 * KB);
    cudaMemset(layer_0, 11, 4 * KB);
    cudaMemset(layer_1, 12, 4 * KB);
    std::vector<uint64_t> layer_addrs(2);
    layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
    layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
    Context ctx("1", 1, 0);
    ctx.set_block_params(KB, 128, 4);
    ctx.set_layer_data_address(1, layer_addrs);
    auto cuda_ctx = std::make_unique<CudaIpcContext>(ctx.device_id());
    EXPECT_TRUE(cuda_ctx->check_support());
    ctx.register_protocol(std::move(cuda_ctx));
    TransferProtocol proto = TransferProtocol::cuda_ipc();
    auto channel = create_channel(&ctx, proto);
    ShmNamingClient naming;
    naming.connect(shm_naming_file);
    auto info_opt = naming.get_worker_info("0", 0);
    for(auto i=0;i < 128; i ++) {
      if (!info_opt.has_value()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        info_opt = naming.get_worker_info("0", 0);
      } else {
        LOG(INFO) << "get worker info, addr =  " << info_opt->addr;
        break;
      }
    }
    EXPECT_TRUE(info_opt.has_value());
    channel->connect(info_opt.value());
    LOG(INFO) << "test cuda server connected;";
    RequestInfo req_info(0, 0, "REQ-0001", {0, 1}, {0, 1});
    ReqSendTask req_task(&req_info, 0, 1, true);
    channel->send_data(0, {{0, 0, 128}});
    channel->send_data(1, {{0, 0, 128}});
    channel->flush();
    auto iter = Source<const ReqSendTask>::from(&req_task, 1);
    channel->send_notification(iter.get());
    channel->close();
    LOG(INFO) << "channel process exit ...";
    cuda_free(layer_0);
    cuda_free(layer_1);
  }
}
