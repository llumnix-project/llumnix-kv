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
  const std::vector<std::vector<uint64_t>> src_layers;
  const std::vector<std::vector<uint64_t>> dst_layers;
  std::queue<std::string> *notifies;
  std::vector<std::vector<IpcBlock>>* data_;

  FakeChannel(Context *ctx, const std::vector<std::vector<uint64_t>> &dst, std::queue<std::string> *n) :
    src_layers(ctx->layer_data_address()), dst_layers(dst), notifies(n) {}
  void connect(const WorkerInfo &dst_info) override {};
  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind) override {
    this->data_ = &data;
  }
  void send_data(size_t layer_index) override {
    for(size_t tensor_index = 0; tensor_index < this->data_->size(); ++tensor_index) {
      const auto& src_layer = src_layers[layer_index];
      const auto& dst_layer = dst_layers[layer_index];
      const auto& tensor_data = (*this->data_)[tensor_index];
      for(const auto& b: tensor_data) {
        auto src_layer_ptr = reinterpret_cast<const char *>(src_layer[tensor_index]);
        auto src_ptr = src_layer_ptr + b.src_offset;
        auto dst_layer_ptr = reinterpret_cast<char *>(dst_layer[tensor_index]);
        auto dst_ptr = dst_layer_ptr + b.dst_offset;
        cuda_d2h_mem_copy(dst_ptr, src_ptr, b.length);
      }
    }
  };
  void send_notification(const std::vector<const ReqSendTask*>& reqs) override {
    for (const auto* req : reqs) {
      notifies->push(req->req_id());
    }
  };
  void flush(std::string&) override {};
  void close() override {};
};

class FakeChannelFactory : public IChannelFactory {
  Context* const ctx_;
  std::vector<std::vector<uint64_t>> dst_layers_;
  std::queue<blade_llm::RequestId>* notifies_;
public:
  FakeChannelFactory(Context *ctx, const std::vector<std::vector<uint64_t>> &dst, std::queue<blade_llm::RequestId>* n):
    ctx_(ctx), dst_layers_(dst), notifies_(n) {}

  Channel create(const WorkerInfo&) override {
    return std::make_unique<FakeChannel>(ctx_, dst_layers_, notifies_);
  }
};

class FakeNamingWorkerClient : public INamingWorkerClient {
public:
  void register_worker(const WorkerInfo &worker_info) override {
    throw std::runtime_error("biubiu~");
  }

  std::optional<WorkerInfo> get_worker_info(const InstanceId &id, WorkerId wid) override {
    LOG(INFO) << "fake get_worker_info: id=" << id << " wid=" << wid;
    WorkerInfo dst_info(id, wid);
    dst_info.tp_size = 1;
    dst_info.worker_tp_rank = 0;
    dst_info.block_sizes = {16 * KB};
    dst_info.token_sizes = {KB};
    return dst_info;
  }
};

class FakeSendStubFactory : public ISendStubFactory {
 public:
  const std::vector<std::vector<uint64_t>> dst_layers;
  Context *ctx;
  std::queue<std::string> *notifies;
  std::shared_ptr<FakeNamingWorkerClient> naming_ = std::make_shared<FakeNamingWorkerClient>();

  FakeSendStubFactory(Context *ctx, const std::vector<std::vector<uint64_t>> &dst, std::queue<std::string> &n):
    ctx(ctx), dst_layers(dst), notifies(&n) {}
  SendStub create_stub(const InstanceId& i, WorkerId w, uint32_t start_layer, uint32_t num_layers,
                       std::optional<TransferProtocol> p, const std::optional<std::string> &) override {
    LOG(INFO) << "Create SendStub";
    auto cf = std::make_unique<FakeChannelFactory>(ctx, dst_layers, notifies);
    return std::make_unique<KvSendStub>(i, w, ctx->worker_info(), start_layer, num_layers, std::move(cf), naming_);
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
  std::vector<std::vector<uint64_t>> device_layer_addrs;
  std::vector<uint64_t> layer_0_addrs = {reinterpret_cast<uint64_t>(layer_0)};
  std::vector<uint64_t> layer_1_addrs = {reinterpret_cast<uint64_t>(layer_1)};
  device_layer_addrs.push_back(layer_0_addrs);
  device_layer_addrs.push_back(layer_1_addrs);

  auto * host_layer_0 = malloc(data_size);
  auto * host_layer_1 = malloc(data_size);
  memset(host_layer_0, 0, data_size);
  memset(host_layer_1, 0, data_size);
  std::vector<std::vector<uint64_t>> host_layer_addrs;
  std::vector<uint64_t> host_layer_0_addrs = {reinterpret_cast<uint64_t>(host_layer_0)};
  std::vector<uint64_t> host_layer_1_addrs = {reinterpret_cast<uint64_t>(host_layer_1)};
  host_layer_addrs.push_back(host_layer_0_addrs);
  host_layer_addrs.push_back(host_layer_1_addrs);
  std::vector<uint32_t> dst_blocks{4, 5, 6, 7};

  auto ctx = std::make_unique<Context>("1",  0);
  ctx->set_tp(1, 0);

  std::vector<std::vector<LayerInfo>> all_layer_infos;
  for (auto layer: device_layer_addrs){ // each layer
    std::vector<LayerInfo> layer_infos;
    for(auto tensor_addr: layer){ // each cache tensor of each layer
      auto layer_info = LayerInfo(KB, 16 * KB, tensor_addr);
      layer_infos.emplace_back(std::move(layer_info));
    }
    all_layer_infos.emplace_back(std::move(layer_infos));
  }
  ctx->set_layer_info(0, all_layer_infos);
  uint32_t block_size = 16 * KB;
  ctx->set_block_params({block_size}, {KB}, 128);
  ctx->set_cuda_barrier(std::move(cu_barrier));

  std::queue<RequestId> notifies;
  auto f = std::make_unique<FakeSendStubFactory>(ctx.get(), host_layer_addrs, notifies);
  KvTransferClient client(std::move(ctx), std::move(f));

  // client.add_target("0", 0, 0, 2);
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
  EXPECT_TRUE(done_ret == ReqState::OK);

  // client.remove_target("0", 0);
  LOG(INFO) << "finish";
  cuda_free(layer_0);
  cuda_free(layer_1);
  free(host_layer_0);
  free(host_layer_1);
}
