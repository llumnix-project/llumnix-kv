#include <gtest/gtest.h>
#include "protocol.h"

using namespace blade_llm;

TEST(ProtocolTest, TestProtocalSupportCheck) {
  SupportTransferProtocols support;
  support.set_support(TransferProtocol::Kind::CUDA_IPC);
  EXPECT_TRUE(support.is_support(TransferProtocol::Kind::CUDA_IPC));
  EXPECT_TRUE(support.is_support(TransferProtocol::cuda_ipc()));
  EXPECT_FALSE(support.is_support(TransferProtocol::Kind::RDMA_DIRECT));
  support.set_support(TransferProtocol::Kind::RDMA_DIRECT);
  EXPECT_TRUE(support.is_support(TransferProtocol::Kind::CUDA_IPC));
  EXPECT_TRUE(support.is_support(TransferProtocol::cuda_ipc()));
  EXPECT_TRUE(support.is_support(TransferProtocol::Kind::RDMA_DIRECT));
  EXPECT_TRUE(support.is_support(TransferProtocol::rdma_direct()));

  auto v = support.value();
  SupportTransferProtocols check(v);
  EXPECT_TRUE(check.is_support(TransferProtocol::Kind::CUDA_IPC));
  EXPECT_TRUE(check.is_support(TransferProtocol::cuda_ipc()));
  EXPECT_TRUE(check.is_support(TransferProtocol::Kind::RDMA_DIRECT));
  EXPECT_TRUE(check.is_support(TransferProtocol::rdma_direct()));
}
