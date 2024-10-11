#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "client.h"
#include "logging.h"

using ::testing::Return;
using ::testing::_;
using ::testing::Field;
using ::testing::Eq;
using ::testing::WhenSorted;
using ::testing::ElementsAre;

using namespace blade_llm;

MATCHER_P(batchCheck, expect, "unexpected batch") {
  EXPECT_EQ(arg.tasks->size(), expect.size());
  auto tasks = arg.tasks;
  std::sort(tasks->begin(),
            tasks->end(),
            [](const RequestInfo *a, const RequestInfo *b) { return a->req_id < b->req_id; });
  for (auto i = 0; i < expect.size(); ++i) {
    const RequestInfo *a = (*tasks)[i];
    const RequestInfo *b = expect[i];
    EXPECT_EQ(a->req_id, b->req_id);
    EXPECT_EQ(a->dst_inst_id, b->dst_inst_id);
    EXPECT_EQ(a->dst_worker_id, b->dst_worker_id);
    EXPECT_EQ(a->src_blocks(), b->src_blocks());
    EXPECT_EQ(a->dst_blocks(), b->dst_blocks());
    EXPECT_EQ(a->seen_tokens(), b->seen_tokens());
    EXPECT_EQ(a->new_tokens(), b->new_tokens());
    EXPECT_EQ(a->reach_last_token(), b->reach_last_token());
  }
  return true;
}

class MockSendStub : public ISendStub {
 public:
  MockSendStub() = default;
  MOCK_METHOD(void, connect, (Context *, const WorkerInfo&), (override));
  MOCK_METHOD(bool, is_running, (), (override));
  MOCK_METHOD(void, send_batch, (const BatchSendTask&), (override));
};

class ProxyStub : public ISendStub {
 public:
  InstanceId dst_inst;
  WorkerId dst_worker;
  MockSendStub *stub;
  ProxyStub(InstanceId i, WorkerId w, MockSendStub *s) : dst_inst(i), dst_worker(w), stub(s) {}
  void connect(Context *ctx, const WorkerInfo &dst_info) override {
    stub->connect(ctx, dst_info);
  }
  bool is_running() override {
    return stub->is_running();
  }
  void send_batch(const BatchSendTask &batch) override {
    stub->send_batch(batch);
  }
};

class FakeStubFactory : public ISendStubFactory {
 public:
  std::vector<std::unique_ptr<ProxyStub>> stubs;

  FakeStubFactory() = default;
  SendStub create_stub(InstanceId dst_inst, WorkerId dst_worker, uint32_t, uint32_t) override {
    for (auto i = 0; i < stubs.size(); ++i) {
      if (stubs[i] != nullptr) {
        if (stubs[i]->dst_inst == dst_inst && stubs[i]->dst_worker == dst_worker) {
          return std::move(stubs[i]);
        }
      }
    }
    return nullptr;
  }
};


TEST(KVTransferClientTest, SendTo1) {
  auto ctx = std::make_unique<Context>(1, 1);
  RequestInfo req1(2, 1, "REQ00000001", {0, 1}, {0, 1});
  req1.add_new_tokens(1, false);
  std::vector<const RequestInfo *> expect_reqs{&req1};
  MockSendStub stub;
  EXPECT_CALL(stub, is_running())
      .Times(4)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(stub, send_batch(batchCheck(expect_reqs))).Times(2);
  auto factory = std::make_unique<FakeStubFactory>();
  factory->stubs.push_back(std::make_unique<ProxyStub>(2, 1, &stub));

  KvTransferClient client(std::move(ctx), std::move(factory));
  client.add_target(2, 1, 0, 2);
  {
    auto ret = client.submit_req_send(3, 1, req1.req_id,
                                      req1.new_tokens(), req1.reach_last_token(),
                                      req1.src_blocks(), req1.dst_blocks());
    EXPECT_TRUE(ret.is_err());
    EXPECT_EQ(ret.err().code, ErrorCode::TARGET_DISCONNECTED);
  }
  {
    auto ret = client.submit_req_send(2, 1, req1.req_id,
                                      req1.new_tokens(), req1.reach_last_token(),
                                      req1.src_blocks(), req1.dst_blocks());
    EXPECT_TRUE(ret.is_ok());
    client.start_send();
    client.flush_send();
  }
  {
    req1.set_seen_tokens(1).add_new_tokens(1, true);
    auto ret = client.submit_delta_send(req1.req_id, 1,
                                        1, req1.reach_last_token());
    EXPECT_TRUE(ret.is_ok());
    client.start_send();
    client.flush_send();
  }
}

TEST(KVTransferClientTest, SendTo2) {
  auto ctx = std::make_unique<Context>(1, 1);
  RequestInfo req0(3, 1, "REQ00000000", {0, 1}, {0, 1});
  req0.add_new_tokens(1, false);
  RequestInfo req1(2, 1, "REQ00000001", {2, 3}, {2, 3});
  req1.add_new_tokens(1, false);
  std::vector<const RequestInfo *> expect_reqs{&req0, &req1};

  MockSendStub stub0;
  EXPECT_CALL(stub0, is_running())
      .Times(4)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(stub0, send_batch(batchCheck(expect_reqs))).Times(2);
  MockSendStub stub1;
  EXPECT_CALL(stub1, is_running())
      .Times(4)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(stub1, send_batch(batchCheck(expect_reqs))).Times(2);
  auto factory = std::make_unique<FakeStubFactory>();
  factory->stubs.push_back(std::make_unique<ProxyStub>(2, 1, &stub0));
  factory->stubs.push_back(std::make_unique<ProxyStub>(3, 1, &stub1));

  KvTransferClient client(std::move(ctx),  std::move(factory));
  client.add_target(2, 1, 0, 2);
  client.add_target(3, 1, 0, 2);
  {
    auto ret = client.submit_req_send(3, 1, req0.req_id,
                                      req0.new_tokens(), req0.reach_last_token(),
                                      req0.src_blocks(), req0.dst_blocks());
    EXPECT_TRUE(ret.ok());
  }
  {
    auto ret = client.submit_req_send(2, 1, req1.req_id,
                                      req1.new_tokens(), req1.reach_last_token(),
                                      req1.src_blocks(), req1.dst_blocks());
    EXPECT_TRUE(ret.is_ok());
  }
  client.start_send();
  client.flush_send();

  req1.set_seen_tokens(1).add_new_tokens(1, true);
  req0.set_seen_tokens(1).add_new_tokens(1, true);
  {
    auto ret = client.submit_delta_send(req0.req_id, 1,
                                        1, req0.reach_last_token());
    EXPECT_TRUE(ret.is_ok());
  }
  {
    auto ret = client.submit_delta_send(req1.req_id, 1,
                                        1, req1.reach_last_token());
    EXPECT_TRUE(ret.is_ok());
  }
  client.start_send();
  client.flush_send();
}

TEST(KVTransferClientTest, SendToPP2) {
  auto ctx = std::make_unique<Context>(1, 1);
  RequestInfo req0(3, 1, "REQ00000001", {0, 1}, {0, 1});
  req0.add_new_tokens(1, false);
  RequestInfo req1(2, 1, "REQ00000001", {0, 1}, {2, 3});
  req1.add_new_tokens(1, false);
  std::vector<const RequestInfo *> expect_reqs{&req0, &req1};

  MockSendStub stub0;
  EXPECT_CALL(stub0, is_running())
      .Times(4)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(stub0, send_batch(batchCheck(expect_reqs))).Times(2);
  MockSendStub stub1;
  EXPECT_CALL(stub1, is_running())
      .Times(4)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(stub1, send_batch(batchCheck(expect_reqs))).Times(2);

  auto factory = std::make_unique<FakeStubFactory>();
  factory->stubs.push_back(std::make_unique<ProxyStub>(2, 1, &stub0));
  factory->stubs.push_back(std::make_unique<ProxyStub>(3, 1, &stub1));

  KvTransferClient client(std::move(ctx),  std::move(factory));
  client.add_target(2, 1, 0, 2);
  client.add_target(3, 1, 0, 2);
  {
    auto ret = client.submit_req_send(3, 1, req0.req_id,
                                      req0.new_tokens(), req0.reach_last_token(),
                                      req0.src_blocks(), req0.dst_blocks());
    EXPECT_TRUE(ret.ok());
  }
  {
    auto ret = client.submit_req_send(2, 1, req1.req_id,
                                      req1.new_tokens(), req1.reach_last_token(),
                                      req1.src_blocks(), req1.dst_blocks());
    EXPECT_TRUE(ret.is_ok());
  }
  client.start_send();
  client.flush_send();

  req1.set_seen_tokens(1).add_new_tokens(1, true);
  req0.set_seen_tokens(1).add_new_tokens(1, true);
  {
    auto ret = client.submit_delta_send(req0.req_id, 1,
                                        1, req0.reach_last_token());
    EXPECT_TRUE(ret.is_ok());
  }
  client.start_send();
  client.flush_send();
}
