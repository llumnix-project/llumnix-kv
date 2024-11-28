#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "client.h"
#include "channel.h"
#include "thrid_party/logging.h"
#include "utils/cuda_helper.h"
#include "naming.h"
#include "naming/shm_naming.h"

using namespace blade_llm;
#define TEST_REQ_ID "TEST_REQ_ID_007"

__global__ void clientTestKernel(char *ptr, int sz, char val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("thread%d: blockIdx.x=%d,blockDim.x=%d, gridDim.x=%d;\n", idx, blockIdx.x, blockDim.x, gridDim.x);
  int stride = gridDim.x * blockDim.x;
  for (; idx < sz; idx += stride) {
    ptr[idx] = val;
  }
}

class FakeChannel: public IChannel {
 public:
  const std::vector<uint64_t> src_layer;
  const std::vector<uint64_t> dst_layer;
  std::queue<std::string> *notifies;
  FakeChannel(Context *ctx, const std::vector<uint64_t> &dst, std::queue<std::string> *n) :
    src_layer(ctx->layer_data_address()), dst_layer(dst), notifies(n) {}
  void connect(const WorkerInfo &dst_info) override {};
  void send_data(size_t layer_index, const std::vector<IpcBlock> &data) override {
    for(const auto& b: data) {
      auto src_layer_ptr = reinterpret_cast<const char *>(src_layer[layer_index]);
      auto src_ptr = src_layer_ptr + b.src_offset;
      auto dst_layer_ptr = reinterpret_cast<char *>(dst_layer[layer_index]);
      auto dst_ptr = dst_layer_ptr + b.dst_offset;
      cuda_d2h_mem_copy(dst_ptr, src_ptr, b.length);
    }
  };
  void send_notification(const std::vector<const ReqSendTask*>& reqs) override {
    for (const auto* req : reqs) {
      notifies->push(req->req_id());
    }
  };
  void flush() override {};
  void close() override {};
};

class FakeSendStubFactory : public ISendStubFactory {
 public:
  const std::vector<uint64_t> dst_layer;
  Context *ctx;
  std::queue<std::string> *notifies;

  FakeSendStubFactory(Context *ctx, const std::vector<uint64_t> &dst, std::queue<std::string> &n):
    ctx(ctx), dst_layer(dst), notifies(&n) {}
  SendStub create_stub(const InstanceId& i, WorkerId w, uint32_t start_layer, uint32_t num_layers,
                       std::optional<TransferProtocol> p) override {
    LOG(INFO) << "Create SendStub";
    auto channel = std::make_unique<FakeChannel>(ctx, dst_layer, notifies);
    WorkerInfo dst_info(i, w);
    dst_info.tp_size = 1;
    dst_info.worker_tp_rank = 0;
    dst_info.block_size =  16 * KB;
    dst_info.token_size = KB;
    return std::make_unique<KvSendStub>(dst_info, ctx->worker_info(), start_layer, num_layers, std::move(channel));
  }
};

TEST(KVTransferClientTest, TestKernelSyncAndDataTransfer) {
  size_t num_layers = 2;
  cudaSetDevice(0);
  std::vector<cudaEvent_t> events(num_layers);
  std::vector<uint64_t> event_addrs;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  for (size_t i = 0; i < num_layers; ++i) {
    cudaEventCreate(&events[i]);
    auto addr = reinterpret_cast<uintptr_t>(events[i]);
    event_addrs.push_back(addr);
  }

  auto cu_barrier = std::make_unique<CudaEventBarrier>(event_addrs);
  int blocks = 1;
  int threads = 128;

  size_t data_size = blocks * threads * (1 << 14); // 128 * 16 KB
  void *layer_0, *layer_1;
  cudaMalloc(&layer_0, data_size);
  cudaMalloc(&layer_1, data_size);
  cudaMemset(layer_0, 0, data_size);
  cudaMemset(layer_1, 0, data_size);
  std::vector<uint64_t> device_layer_addrs;
  device_layer_addrs.push_back(reinterpret_cast<uint64_t>(layer_0));
  device_layer_addrs.push_back(reinterpret_cast<uint64_t>(layer_1));

  auto * host_layer_0 = malloc(data_size);
  auto * host_layer_1 = malloc(data_size);
  memset(host_layer_0, 0, data_size);
  memset(host_layer_1, 0, data_size);
  std::vector<uint64_t> host_layer_addrs;
  host_layer_addrs.push_back(reinterpret_cast<uint64_t>(host_layer_0));
  host_layer_addrs.push_back(reinterpret_cast<uint64_t>(host_layer_1));
  std::vector<uint32_t> dst_blocks{4, 5, 6, 7};

  auto ctx = std::make_unique<Context>("1",  0);
  ctx->set_tp(1, 0);
  ctx->set_layer_data_address(0, device_layer_addrs);
  auto block_size = 16 * KB;
  ctx->set_block_params(block_size, KB, 128);
  ctx->set_cuda_barrier(std::move(cu_barrier));

  std::queue<RequestId> notifies;
  auto f = std::make_unique<FakeSendStubFactory>(ctx.get(), host_layer_addrs, notifies);
  KvTransferClient client(std::move(ctx), std::move(f));

  client.add_target("0", 0, 0, 2);
  client.submit_req_send("0", 0, TEST_REQ_ID, 16 * 4, true, {0, 1, 2, 3}, dst_blocks);
  auto zyidx33 = client.start_send();
  std::this_thread::sleep_for(std::chrono::milliseconds (100)); // let start_send to run;
  // mock layer 0
  clientTestKernel<<<blocks, threads, 0, stream>>>((char *)(layer_0), data_size,  10);
  cudaEventRecord(events[0], stream);
  client.notify_event_record(zyidx33);
  // mock layer 1
  clientTestKernel<<<blocks, threads, 0, stream>>>((char *)(layer_1), data_size,  20);
  cudaEventRecord(events[1], stream);
  client.notify_event_record(zyidx33);
  cudaStreamSynchronize(stream);
  LOG(INFO) << "cuda stream synced;";
  client.flush_send(zyidx33);

  int cnt = 0;
  while(cnt < 20 && notifies.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    cnt ++;
  }
  EXPECT_FALSE(notifies.empty());
  auto req_id = notifies.front();
  EXPECT_TRUE(req_id == TEST_REQ_ID);
  {
    char *ptr = (char *) host_layer_0;
    for (auto bid : dst_blocks) {
      auto offset = bid * block_size;
      auto check_p = ptr + offset;
      uint32_t sum = 0;
      for (auto i = 0; i < block_size; ++i) {
        sum += check_p[i];
      }
      EXPECT_EQ(sum, 10 * block_size);
    }
  }
  {
    char *ptr = (char *) host_layer_1;
    for (auto bid : dst_blocks) {
      auto offset = bid * block_size;
      auto check_p = ptr + offset;
      uint32_t sum = 0;
      for (auto i = 0; i < block_size; ++i) {
        sum += check_p[i];
      }
      EXPECT_EQ(sum, 20 * block_size);
    }
  }
  auto done_ret = client.check_transfer_done(TEST_REQ_ID);
  EXPECT_TRUE(done_ret);

  client.remove_target("0", 0);
  LOG(INFO) << "finish";
  cuda_free(layer_0);
  cuda_free(layer_1);
  free(host_layer_0);
  free(host_layer_1);
}
