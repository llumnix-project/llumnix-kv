#include <gtest/gtest.h>
#include "service.h"

using namespace blade_llm;

TEST(KVTransferServiceTest, TestSubmitReq) {
  auto ctx = std::make_unique<Context>("1",  1);
  KvTransferService service(std::move(ctx));
  service.submit_recv("0", 0, "test_req", {1, 2, 3});
  auto ret1 = service.check_recv_done("test_req");
  EXPECT_FALSE(ret1);
  EXPECT_THROW(service.submit_recv("0", 0, "test_req_000", {}), KVTransferException);
}

TEST(KVTransferServiceTest, TestReqReceive) {
  auto ctx = std::make_unique<Context>("1", 1);
  KvTransferService service(std::move(ctx));
  service.submit_recv("0", 0, "REQ_0001", {1, 2, 3});
  service.on_recv("0", 0, "REQ_0001", {1, 2, 3});
  auto ret1 = service.check_recv_done("REQ_0001");
  EXPECT_TRUE(ret1);

  service.submit_recv("0", 0, "REQ_0002", {1, 2, 3});
  service.on_recv("0", 0, "REQ_0002", {3, 2, 1});
  auto ret3 = service.check_recv_done("REQ_0002");
  EXPECT_TRUE(ret3);

  service.submit_recv("0", 0, "REQ_0003", {1, 2, 3});
  service.on_recv("0", 0, "REQ_0003", {2, 1});
  EXPECT_THROW(service.check_recv_done("REQ_0003"), KVTransferException);

  // Receive before submit;
  service.on_recv("0", 0, "REQ_0004", {3, 2, 1});
  service.submit_recv("0", 0, "REQ_0004", {3, 2, 1});
  auto ret5 = service.check_recv_done("REQ_0004");
  EXPECT_TRUE(ret5);

  EXPECT_THROW(service.check_recv_done("REQ_0005"), KVTransferException);
}

TEST(KVTransferServiceTest, TestReqMultiSourceReceive) {
  auto ctx = std::make_unique<Context>("1", 1);
  KvTransferService service(std::move(ctx));
  service.submit_recv("0", 0, "REQ_0001", {1, 2, 3});
  service.submit_recv("1", 0, "REQ_0001", {4, 5, 6});
  service.on_recv("0", 0, "REQ_0001", {1, 2, 3});
  auto ret0 = service.check_recv_done("REQ_0001");
  EXPECT_FALSE(ret0);
  service.on_recv("1", 0, "REQ_0001", {4, 5, 6});
  auto ret1 = service.check_recv_done("REQ_0001");
  EXPECT_TRUE(ret1);
}
