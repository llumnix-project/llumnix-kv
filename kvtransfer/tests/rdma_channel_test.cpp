#include <cassert>
#include <gtest/gtest.h>
#include "utils/cuda_helper.h"
#include "utils/block_queue.h"
#include "protocol/rdma_channel.h"
#include "naming/shm_naming.h"
#include "thrid_party/logging.h"

#define NAMING_FILE "rdma_test"
#ifdef ENABLE_RDMA

using namespace blade_llm;

TEST(RDMAChannelTest, TestTransfer) {
  auto test_rdma_ctx = BarexProtoContext::client_context("test", TransferProtocol::Kind::RDMA_DIRECT);
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
    ShmNamingClient naming;
    naming.connect(NAMING_FILE);

    Context ctx("0", 0);
    std::vector<std::vector<LayerInfo>> all_layer_infos = {
      {LayerInfo(token_size, block_size, reinterpret_cast<uint64_t>(layer_0))},
      {LayerInfo(token_size, block_size, reinterpret_cast<uint64_t>(layer_1))}
    };
    ctx.set_layer_info(0, all_layer_infos);
    ctx.set_block_params({block_size}, {token_size}, 4);
    RDMAServer server;
    server.start_server(&ctx);
    naming.register_worker(ctx.worker_info());
    LOG(INFO) << "rdma server: started...";

    // Wait for transfer to complete (using event-based mechanism now)
    std::this_thread::sleep_for(std::chrono::seconds(2));

    char *host_buf = new char[token_size * 3];
    cuda_d2h_mem_copy(host_buf, layer_0, token_size * 3);
    for (size_t i = 0; i < token_size; i++) {
      EXPECT_EQ(host_buf[i], 7);
    }
    for (size_t i = token_size; i < 2 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 0);
    }
    for (size_t i = 2 * token_size; i < 3 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 7);
    }

    cuda_d2h_mem_copy(host_buf, layer_1, token_size * 3);
    for (size_t i = 0; i < token_size; i++) {
      EXPECT_EQ(host_buf[i], 8);
    }
    for (size_t i = token_size; i < 2 * token_size; ++i) {
      EXPECT_EQ(host_buf[i], 0);
    }
    for (size_t i = 2 * token_size; i < 3 * token_size; ++i) {
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
    ShmNamingClient naming;
    naming.connect(NAMING_FILE);

    Context ctx("1", 0);
    std::vector<std::vector<LayerInfo>> all_layer_infos = {
      {LayerInfo(token_size, block_size, reinterpret_cast<uint64_t>(layer_0))},
      {LayerInfo(token_size, block_size, reinterpret_cast<uint64_t>(layer_1))}
    };
    ctx.set_layer_info(0, all_layer_infos);
    ctx.set_block_params({block_size}, {token_size}, 4);
    auto proto_ctx = BarexProtoContext::client_context("KVTClient", TransferProtocol::Kind::RDMA_DIRECT);
    auto proto = proto_ctx->protocol();
    ctx.register_protocol(std::move(proto_ctx));
    auto rdma_ctx = ctx.get_protocol_ctx<BarexProtoContext>(proto);
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
    LOG(INFO) << "fetch dst worker info: token_sizes=" << info.token_sizes[0] << ", block_sizes=" << info.block_sizes[0];
    channel.connect(dst_info.value());
    std::vector<uint32_t> blocks{0, 1};
    std::vector<const ReqSendTask *> reqs;
    auto rp = std::make_shared<RequestInfo>("0", 0, "test_rdma", blocks, blocks);
    ReqSendTask t(rp, 0, 1, true);
    reqs.push_back(&t);
    std::vector<std::vector<IpcBlock>> data{
      {IpcBlock(0, 0, token_size), IpcBlock(2 * token_size, 2 * token_size, token_size)}
    };
    channel.register_data(data, TPKind::PEQD);
    channel.send_data(0);
    channel.send_data(1);

    std::string out;
    channel.flush(out);

    channel.close();
    cudaFree(layer_0);
    cudaFree(layer_1);
    LOG(INFO) << "rdma channel:  send finish, exit;";
  }
}

#endif // ENABLE_RDMA
