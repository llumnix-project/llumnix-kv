#include <gtest/gtest.h>
#include "service.h"

using namespace blade_llm;

TEST(KVTransferServiceTest, TestSubmitReq) {
  auto ctx = std::make_unique<Context>("1", 1, 1);
  KvTransferService service(std::move(ctx));
  auto ret0 = service.submit_recv(0, 0, "test_req", {1, 2, 3});
  EXPECT_TRUE(ret0.is_ok());
  auto ret1 = service.check_recv_done("test_req");
  EXPECT_TRUE(ret1.is_ok());
  EXPECT_FALSE(ret1.ok());

  auto ret3 = service.submit_recv(0, 0, "test_req_000", {});
  EXPECT_TRUE(ret3.is_err());
}

TEST(KVTransferServiceTest, TestReqReceive) {
  auto ctx = std::make_unique<Context>("1", 1, 1);
  KvTransferService service(std::move(ctx));
  auto ret0 = service.submit_recv(0, 0, "REQ_0001", {1, 2, 3});
  EXPECT_TRUE(ret0.is_ok());
  service.on_recv(0, 0, "REQ_0001", {1, 2, 3});
  auto ret1 = service.check_recv_done("REQ_0001");
  EXPECT_TRUE(ret1.is_ok());
  EXPECT_TRUE(ret1.ok());

  service.submit_recv(0, 0, "REQ_0002", {1, 2, 3});
  service.on_recv(0, 0, "REQ_0002", {3, 2, 1});
  auto ret3 = service.check_recv_done("REQ_0002");
  EXPECT_TRUE(ret3.is_ok());
  EXPECT_TRUE(ret3.ok());

  service.submit_recv(0, 0, "REQ_0003", {1, 2, 3});
  service.on_recv(0, 0, "REQ_0003", {2, 1});
  auto ret4 = service.check_recv_done("REQ_0003");
  EXPECT_TRUE(ret4.is_err());

  // Receive before submit;
  service.on_recv(0, 0, "REQ_0004", {3, 2, 1});
  service.submit_recv(0, 0, "REQ_0004", {3, 2, 1});
  auto ret5 = service.check_recv_done("REQ_0004");
  EXPECT_TRUE(ret5.is_ok());
  EXPECT_TRUE(ret5.ok());

  auto ret6 = service.check_recv_done("REQ_0005");
  EXPECT_TRUE(ret6.is_err());
}

TEST(KVTransferServiceTest, TestReqMultiSourceReceive) {
  auto ctx = std::make_unique<Context>("1", 1, 1);
  KvTransferService service(std::move(ctx));
  service.submit_recv(0, 0, "REQ_0001", {1, 2, 3});
  service.submit_recv(1, 0, "REQ_0001", {4, 5, 6});
  service.on_recv(0, 0, "REQ_0001", {1, 2, 3});
  auto ret0 = service.check_recv_done("REQ_0001");
  EXPECT_TRUE(ret0.is_ok());
  EXPECT_FALSE(ret0.ok());
  service.on_recv(1, 0, "REQ_0001", {4, 5, 6});
  auto ret1 = service.check_recv_done("REQ_0001");
  EXPECT_TRUE(ret1.is_ok());
  EXPECT_TRUE(ret1.ok());
}
