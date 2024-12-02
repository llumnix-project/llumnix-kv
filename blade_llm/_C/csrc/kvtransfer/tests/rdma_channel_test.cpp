#include <cassert>
#include <gtest/gtest.h>
#include "utils/cuda_helper.h"
#include "utils/block_queue.h"
#include "protocol/rdma_protocol.h"
#include "naming/shm_naming.h"
#include "thrid_party/logging.h"

#define NAMING_FILE "rdma_test"
#ifdef ENABLE_RDMA

using namespace blade_llm;

struct ReqNotification {

  ReqNotification() = default;

  ReqNotification(const InstanceId& src_inst_id,
                  uint32_t src_worker_id,
                  const std::string &req_id,
                  std::vector<uint32_t> &&dst_block_ids) :
      src_inst_id(src_inst_id),
      src_worker_id(src_worker_id),
      req_id(req_id),
      dst_block_ids(std::move(dst_block_ids)) {};

  ReqNotification(ReqNotification &&other) noexcept:
      src_inst_id(std::move(other.src_inst_id)),
      src_worker_id(other.src_worker_id),
      req_id(std::move(other.req_id)),
      dst_block_ids(std::move(other.dst_block_ids)) {};

  ReqNotification &operator=(ReqNotification &&other) noexcept {
    src_inst_id = std::move(other.src_inst_id);
    src_worker_id = other.src_worker_id;
    req_id = std::move(other.req_id);
    dst_block_ids = std::move(other.dst_block_ids);
    return *this;
  }

  InstanceId src_inst_id;
  uint32_t src_worker_id{};
  std::string req_id;
  std::vector<uint32_t> dst_block_ids;
};

class FakeTransferService : public ITransferService {
 public:
  FakeTransferService() = default;

  void on_recv(const InstanceId& src_inst_id,
               WorkerId src_worker_id,
               const RequestId &req_id,
               std::vector<uint32_t> &&dst_block_ids) override {

    ReqNotification rn(src_inst_id, src_worker_id, req_id, std::move(dst_block_ids));
    queue_.push(std::move(rn));
  }

  ReqNotification pop() {
    ReqNotification rn;
    queue_.pop(rn);
    return rn;
  }

 private:
  BlockingQueue<ReqNotification> queue_;
};


TEST(RDMAChannelTest, TestSerdeReqNofitfication) {
  ReqNotification rn("##############1", 1, "test_rdma", {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048});
  auto size = get_encode_size(rn.src_inst_id, rn.src_worker_id, rn.req_id, rn.dst_block_ids);
  auto buf = static_cast<char*>(malloc(size));
  encode_notification(buf, rn.src_inst_id, rn.src_worker_id, rn.req_id, rn.dst_block_ids);
  ReqNotification d_rn;
  auto ret = decode_notification(buf, size, d_rn.src_inst_id, d_rn.src_worker_id, d_rn.req_id, d_rn.dst_block_ids);
  EXPECT_TRUE(ret);
  EXPECT_EQ(rn.src_inst_id, d_rn.src_inst_id);
  EXPECT_EQ(rn.src_worker_id, d_rn.src_worker_id);
  EXPECT_EQ(rn.req_id, d_rn.req_id);
  EXPECT_EQ(rn.dst_block_ids.size(), d_rn.dst_block_ids.size());
  for (size_t i = 0; i < rn.dst_block_ids.size(); i++) {
    EXPECT_EQ(rn.dst_block_ids[i], d_rn.dst_block_ids[i]);
  }
  free(buf);
}


TEST(RDMAChannelTest, TestTransefer) {
  auto test_rdma_ctx = RDMAProtoContext::client_context("test", 1);
  if (!test_rdma_ctx->check_support()) {
    LOG(INFO) << "rdma not support on this node;";
    return;
  }
  LOG(INFO) << "start to test rdma channel;";
  auto n_server = ShmNamingServer(NAMING_FILE);
  n_server.start();
  auto pid = fork();
  if (pid > 0) {
    // parent process, receive part;
    cuda_set_device(0);
    void *layer_0, *layer_1;
    auto layer_size = 16 * KB;
    auto block_size = 4 * KB;
    auto token_size = KB;
    cuda_malloc(&layer_0, layer_size);
    cuda_malloc(&layer_1, layer_size);
    cudaMemset(layer_0, 0, layer_size);
    cudaMemset(layer_1, 0, layer_size);
    std::vector<uint64_t> layer_addrs(2);
    layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
    layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
    ShmNamingClient naming;
    naming.connect(NAMING_FILE);

    Context ctx("0", 0);
    ctx.set_layer_data_address(0, layer_addrs);
    ctx.set_block_params(block_size, token_size, 4);
    RDMAServer server;
    FakeTransferService service;
    server.start_server(&service, &ctx);
    naming.register_worker(ctx.worker_info());
    LOG(INFO) << "rdma server: started...";
    auto rn = service.pop();
    EXPECT_EQ(rn.src_inst_id, "1");
    EXPECT_EQ(rn.src_worker_id, 0);
    EXPECT_EQ(rn.req_id, "test_rdma");
    EXPECT_EQ(rn.dst_block_ids.size(), 2);
    EXPECT_EQ(rn.dst_block_ids[0], 0);
    EXPECT_EQ(rn.dst_block_ids[1], 1);

    char *host_buf = new char[token_size * 3];
    cuda_d2h_mem_copy(host_buf, layer_0, token_size * 3);
    for (auto i = 0; i < token_size; i++) {
      EXPECT_EQ(host_buf[i], 7);
    }
    for (auto i = token_size; i < 2 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 0);
    }
    for (auto i = 2 * token_size; i < 3 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 7);
    }

    cuda_d2h_mem_copy(host_buf, layer_1, token_size * 3);
    for (auto i = 0; i < token_size; i++) {
      EXPECT_EQ(host_buf[i], 8);
    }
    for (auto i = token_size; i < 2 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 0);
    }
    for (auto i = 2 * token_size; i < 3 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 8);
    }
    LOG(INFO) << "rdma server: all test pass, do clean ...";
    delete[] host_buf;
    cudaFree(layer_0);
    cudaFree(layer_1);
  } else {
    // child process, send part;
    assert(pid == 0);
    void *layer_0, *layer_1;
    auto layer_size = 16 * KB;
    auto block_size = 4 * KB;
    auto token_size = KB;
    cuda_malloc(&layer_0, layer_size);
    cuda_malloc(&layer_1, layer_size);
    cudaMemset(layer_0, 7, layer_size);
    cudaMemset(layer_1, 8, layer_size);
    std::vector<uint64_t> layer_addrs(2);
    layer_addrs[0] = reinterpret_cast<uint64_t>(layer_0);
    layer_addrs[1] = reinterpret_cast<uint64_t>(layer_1);
    ShmNamingClient naming;
    naming.connect(NAMING_FILE);

    Context ctx("1", 0);
    ctx.set_layer_data_address(0, layer_addrs);
    ctx.set_block_params(block_size, token_size, 4);
    auto proto_ctx = RDMAProtoContext::client_context("KVTClient", 4);
    auto proto = proto_ctx->protocol();
    ctx.register_protocol(std::move(proto_ctx));
    auto rdma_ctx = ctx.get_protocol_ctx<RDMAProtoContext>(proto);
    RDMAChannel channel(ctx.inst_name, ctx.worker_id, rdma_ctx->cli_barex_ctx());
    auto dst_info = naming.get_worker_info("0", 0);
    size_t retry_cnt = 3;
    while (!dst_info.has_value() && retry_cnt > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      dst_info = naming.get_worker_info("0", 0);
      retry_cnt--;
    }
    EXPECT_TRUE(dst_info.has_value());
    auto &info = dst_info.value();
    LOG(INFO) << "fetch dst worker info: token_size=" << info.token_size << ", block_size=" << info.block_size;
    channel.connect(dst_info.value());
    std::vector<uint32_t> blocks{0, 1};
    std::vector<const ReqSendTask *> reqs;
    RequestInfo r("0", 0, "test_rdma", blocks, blocks);
    ReqSendTask t(&r, 0, 1, true);
    reqs.push_back(&t);
    std::vector<IpcBlock> data{{0, 0, token_size}, {2 * token_size, 2 * token_size, token_size}};
    channel.register_data(data, TPKind::PEQD);
    channel.send_data(0);
    channel.send_data(1);
    std::string out;
    channel.flush(out);
    auto nf = CopySource<const ReqSendTask *>::from(reqs);
    channel.send_notification(nf.get());
    channel.close();
    cudaFree(layer_0);
    cudaFree(layer_1);
    LOG(INFO) << "rdma channel:  send finish, exit;";
  }
}

#endif // ENABLE_RDMA
