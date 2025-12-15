#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "client.h"
#include "thrid_party/logging.h"

using ::testing::Return;
using ::testing::Field;
using ::testing::Eq;
using ::testing::WhenSorted;
using ::testing::ElementsAre;

using namespace blade_llm;

struct TestRequestInfo : public RequestInfo,
                         public std::enable_shared_from_this<TestRequestInfo> {
  std::vector<ReqSendTask> task_;
public:
  using RequestInfo::RequestInfo;

  void add_send_task(uint32_t seen, uint32_t new_tokens, bool has_last) {
    auto& self = *this;
    self.update_send(seen, new_tokens, has_last);
    self.task_.emplace_back(this->shared_from_this(), seen, new_tokens, has_last);
  }

  void pop_tasks(std::vector<ReqSendTask>& out) {
    out.reserve(out.size() + this->task_.size());
    for (auto&& task : this->task_) {
      out.emplace_back(std::move(task));
    }
    this->task_.clear();
  }
};

MATCHER_P(batchCheck, expect_req_info, "unexpected batch") {
  auto expect = std::vector<ReqSendTask>();
  for (auto* r : expect_req_info) {
    r->pop_tasks(expect);
  }
  auto tasks = std::vector<const ReqSendTask*>();
  for (const auto& t : arg.tasks) {
    if (t.dst_inst_id() == expect_req_info[0]->dst_inst_id &&
        t.dst_worker_id() == expect_req_info[0]->dst_worker_id) {
      tasks.emplace_back(&t);
    }
  }
  EXPECT_EQ(tasks.size(), expect.size());
  std::stable_sort(tasks.begin(), tasks.end(),
            [](const ReqSendTask *a, const ReqSendTask *b) { return a->req_id() < b->req_id(); });
  for (size_t i = 0; i < expect.size(); ++i) {
    const ReqSendTask *a = tasks[i];
    const ReqSendTask *b = &expect[i];
    EXPECT_EQ(a->req_id(), b->req_id());
    EXPECT_EQ(a->dst_inst_id(), b->dst_inst_id());
    EXPECT_EQ(a->dst_worker_id(), b->dst_worker_id());
    EXPECT_EQ(a->src_blocks(), b->src_blocks());
    EXPECT_EQ(a->dst_blocks(), b->dst_blocks());
    EXPECT_EQ(a->seen_tokens, b->seen_tokens);
    EXPECT_EQ(a->new_tokens, b->new_tokens);
    EXPECT_EQ(a->reach_last_token, b->reach_last_token);
  }
  return true;
}

class MockSendStub : public ISendStub {
 public:
  MockSendStub(): ISendStub("MockSendStubDstId", 20231105) {}
  MOCK_METHOD(void, send_batch, (BatchSendTask), (override));
};

class ProxyStub : public ISendStub {
 public:
  InstanceId dst_inst;
  WorkerId dst_worker;
  ISendStub *stub;
  ProxyStub(const InstanceId &i, WorkerId w, ISendStub *s) :
      ISendStub(i, w),
      dst_inst(i),
      dst_worker(w),
      stub(s),
      info_(i, w) {}

  void send_batch(BatchSendTask batch) override {
    stub->send_batch(std::move(batch));
  }

 private:
  WorkerInfo info_;
};

class FakeStubFactory : public ISendStubFactory {
 public:
  std::unordered_map<std::string, ISendStub*> stubs;

  FakeStubFactory() = default;
  SendStub create_stub(const InstanceId &dst_inst, WorkerId dst_worker, uint32_t, uint32_t,
                       std::optional<TransferProtocol> p, const std::optional<std::string> &) override {
    auto iter = stubs.find(dst_inst + "-" + std::to_string(dst_worker));
    if (iter == stubs.end()) {
      RTASSERT(false);
      return nullptr;
    }
    return std::make_unique<ProxyStub>(dst_inst, dst_worker, iter->second);
  }
};

TEST(KVTransferClientTest, SendTo1) {
  auto ctx = std::make_unique<Context>("1", 1);
  auto req1p = std::make_shared<TestRequestInfo>("2", 1, "REQ00000001", std::vector<uint32_t>{0, 1}, std::vector<uint32_t>{0, 1});
  auto& req1 = *req1p;
  req1.add_send_task(0, 1, false);
  std::vector<TestRequestInfo*> expect_reqs{&req1};
  MockSendStub stub;

  EXPECT_CALL(stub, send_batch(batchCheck(expect_reqs))).Times(2);
  auto factory = std::make_unique<FakeStubFactory>();
  factory->stubs.emplace("2-1", &stub);

  KvTransferClient client(std::move(ctx), std::move(factory));
  // client.add_target("2", 1, 0, 2);
  // {
  //   EXPECT_THROW(client.submit_req_send("3", 1, req1.req_id, 1, false,
  //                                       req1.src_blocks, req1.dst_blocks), KVTransferException);
  // }
  {
    client.submit_req_send("2", 1, req1.req_id, 1, false,
                           req1.src_blocks, req1.dst_blocks);
    auto idx = client.start_send();
    client.target_try_shrink(0);
    usleep(10 * 1000);
    client.target_try_shrink(0);
    usleep(10 * 1000);
    client.target_try_shrink(0);
    client.notify_event_record(idx);
    client.flush_send(idx);
  }
  while (client.target_size() > 0) {
    usleep(100 * 1000);
    client.target_try_shrink(0);
  }

  {
    EXPECT_THROW(client.submit_delta_send("UNKNOWN_REQ_ID", 1, 1, false), KVTransferException);
    req1.add_send_task(1, 1, false);
    req1.add_send_task(2, 0, true);
    client.submit_delta_send(req1.req_id, 1, 1, false);
    client.submit_delta_send(req1.req_id, 2, 0, true);
    auto idx = client.start_send();
    client.target_try_shrink(0);
    usleep(10 * 1000);
    client.target_try_shrink(0);
    usleep(10 * 1000);
    client.target_try_shrink(0);
    client.notify_event_record(idx);
    client.flush_send(idx);
  }
  while (client.target_size() > 0) {
    usleep(100 * 1000);
    client.target_try_shrink(0);
  }
}

TEST(KVTransferClientTest, SendTo2) {
  auto ctx = std::make_unique<Context>("1", 1);
  auto req0p = std::make_shared<TestRequestInfo>("3", 1, "REQ00000000", std::vector<uint32_t>{0, 1}, std::vector<uint32_t>{0, 1});
  auto& req0 = *req0p;
  req0.add_send_task(0, 1, false);
  auto req1p = std::make_shared<TestRequestInfo>("2", 1, "REQ00000001", std::vector<uint32_t>{2, 3}, std::vector<uint32_t>{2, 3});
  auto& req1 = *req1p;
  req1.add_send_task(0, 1, false);
  std::vector<TestRequestInfo*> expect_reqs0{&req1};
  std::vector<TestRequestInfo*> expect_reqs1{&req0};

  MockSendStub stub0;
  EXPECT_CALL(stub0, send_batch(batchCheck(expect_reqs0))).Times(2);
  MockSendStub stub1;
  EXPECT_CALL(stub1, send_batch(batchCheck(expect_reqs1))).Times(2);
  auto factory = std::make_unique<FakeStubFactory>();
  factory->stubs.emplace("2-1", &stub0);
  factory->stubs.emplace("3-1", &stub1);

  KvTransferClient client(std::move(ctx), std::move(factory));
  // client.add_target("2", 1, 0, 2);
  // client.add_target("3", 1, 0, 2);
  client.submit_req_send("3", 1, req0.req_id,
                         1, false,
                         req0.src_blocks, req0.dst_blocks);
  client.submit_req_send("2", 1, req1.req_id,
                         1, false,
                         req1.src_blocks, req1.dst_blocks);
  auto idx35 = client.start_send();
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  client.flush_send(idx35);
  while (client.target_size() > 0) {
    usleep(100 * 1000);
    client.target_try_shrink(0);
  }

  req1.add_send_task(1, 1, false);
  req1.add_send_task(2, 0, true);
  req0.add_send_task(1, 1, false);
  req0.add_send_task(2, 0, true);
  client.submit_delta_send(req0.req_id, 1, 1, false);
  client.submit_delta_send(req0.req_id, 2, 0, true);
  client.submit_delta_send(req1.req_id, 1, 1, false);
  client.submit_delta_send(req1.req_id, 2, 0, true);
  auto idx33 = client.start_send();
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  client.notify_event_record(idx33);
  client.flush_send(idx33);
  while (client.target_size() > 0) {
    usleep(100 * 1000);
    client.target_try_shrink(0);
  }
}

TEST(KVTransferClientTest, SendToPP2) {
  auto ctx = std::make_unique<Context>("1", 1);
  auto req0p = std::make_shared<TestRequestInfo>("3", 1, "REQ00000000", std::vector<uint32_t>{0, 1}, std::vector<uint32_t>{0, 1});
  auto req1p = std::make_shared<TestRequestInfo>("2", 1, "REQ00000001", std::vector<uint32_t>{0, 1}, std::vector<uint32_t>{2, 3});
  auto& req0 = *req0p;
  auto& req1 = *req1p;
  req0.add_send_task(0, 1, false);
  req1.add_send_task(0, 1, false);
  std::vector<TestRequestInfo*> expect_reqs0{&req1};
  std::vector<TestRequestInfo*> expect_reqs1{&req0};

  MockSendStub stub0;
  EXPECT_CALL(stub0, send_batch(batchCheck(expect_reqs0))).Times(2);
  MockSendStub stub1;
  EXPECT_CALL(stub1, send_batch(batchCheck(expect_reqs1))).Times(2);

  auto factory = std::make_unique<FakeStubFactory>();
  factory->stubs.emplace("2-1", &stub0);
  factory->stubs.emplace("3-1", &stub1);

  KvTransferClient client(std::move(ctx), std::move(factory));
  // client.add_target("2", 1, 0, 2);
  // client.add_target("3", 1, 0, 2);
  client.submit_req_send("3", 1, req0.req_id,
                         1, false,
                         req0.src_blocks, req0.dst_blocks);
  client.submit_req_send("2", 1, req1.req_id,
                         1, false,
                         req1.src_blocks, req1.dst_blocks);
  auto idx36 = client.start_send();
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  client.flush_send(idx36);
  while (client.target_size() > 0) {
    usleep(100 * 1000);
    client.target_try_shrink(0);
  }

  req1.add_send_task(1, 1, true);
  req0.add_send_task(1, 1, true);
  client.submit_delta_send(req1.req_id, 1, 1, true);
  client.submit_delta_send(req0.req_id, 1, 1, true);
  auto zyidx34 = client.start_send();
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  usleep(10 * 1000);
  client.target_try_shrink(0);
  client.notify_event_record(zyidx34);
  client.flush_send(zyidx34);
  while (client.target_size() > 0) {
    usleep(100 * 1000);
    client.target_try_shrink(0);
  }
}
