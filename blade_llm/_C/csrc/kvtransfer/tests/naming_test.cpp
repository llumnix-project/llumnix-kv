
#include <gtest/gtest.h>
#include "naming.h"
#include "naming/shm_naming.h"
#include "context.h"
#include "logging.h"

using namespace blade_llm;

TEST(NamingTest, TestShmNaming) {
  const std::string PATH = "shm_naming_test_x";
  ShmNamingServer n_server(PATH);
  n_server.start();

  ShmNamingClient n_client;
  n_client.connect(SHARE_MEMORY_NAMING_SCHEMA, PATH);
  auto info = n_client.get_worker_info(1, 1);
  EXPECT_FALSE(info.has_value());
  Context ctx(1, 1);
  ctx.set_tp(2, 0);
  ctx.set_block_params(512, 128, 8);
  SupportTransferProtocols protocols;
  protocols.set_support(TransferProtocol::Kind::CUDA_IPC);
  protocols.set_support(TransferProtocol::Kind::RDMA_DIRECT);
  auto worker_info = ctx.worker_info();
  worker_info.transfer_protocols = protocols.value();
  n_client.register_worker(worker_info);

  NamingManager naming_manager;
  naming_manager.register_factory(std::make_unique<ShmNamingClientFactory>());
  auto n_client2 = naming_manager.connect_naming(n_server.url);
  auto info_opt = n_client2->get_worker_info(1, 1);
  EXPECT_TRUE(info_opt.has_value());
  EXPECT_EQ(worker_info.inst_id, info_opt->inst_id);
  EXPECT_EQ(worker_info.worker_id, info_opt->worker_id);
  EXPECT_EQ(worker_info.tp_size, info_opt->tp_size);
  EXPECT_EQ(worker_info.worker_tp_rank, info_opt->worker_tp_rank);
  EXPECT_EQ(worker_info.block_size, info_opt->block_size);
  EXPECT_EQ(worker_info.token_size, info_opt->token_size);
  EXPECT_EQ(worker_info.transfer_protocols, info_opt->transfer_protocols);
  SupportTransferProtocols check_protocol(info_opt->transfer_protocols);
  EXPECT_TRUE(check_protocol.is_support(TransferProtocol::Kind::CUDA_IPC));
  EXPECT_TRUE(check_protocol.is_support(TransferProtocol::Kind::RDMA_DIRECT));
}
