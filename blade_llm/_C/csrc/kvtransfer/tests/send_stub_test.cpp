#include <queue>
#include <iostream>
#include <limits>
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

namespace blade_llm {
class Message {
 public:
  struct DataEntry {
    uint32_t layer_idx;
    size_t src_offset;
    size_t dst_offset;
    size_t length;
  };

  uint32_t dst_inst_id;
  uint32_t dst_worker_id;
  std::optional<std::string> req_id;
  std::optional<DataEntry> data{};
  Message(uint32_t inst_id, uint32_t worker_id) :
      dst_inst_id(inst_id), dst_worker_id(worker_id) {};

  Message(uint32_t inst_id, uint32_t worker_id, const std::string &id) :
      dst_inst_id(inst_id), dst_worker_id(worker_id), req_id(id) {};

  void set_data(uint32_t layer_idx, size_t src_offset, size_t dst_offset, size_t length) {
    data = DataEntry{layer_idx, src_offset, dst_offset, length};
  }
  void set_req_id(const std::string &id) {
    req_id = id;
  }
};

class FakeChannel : public IChannel {
 public:
  uint32_t dst_inst_id;
  uint32_t dst_worker_id;
  std::queue<Message> q;

  FakeChannel() : q(), dst_inst_id(0), dst_worker_id(INVALID_INST_WORKER_ID) {};
  void connect(const WorkerInfo &info) override {
    dst_inst_id = info.inst_id;
    dst_worker_id = info.worker_id;
  }
  void send_data(size_t layer_idx, const std::vector<IpcBlock> &data) override {
    for (const auto &[src_offset, dst_offset, length] : data) {
      if (length > 0) {
        Message msg(dst_inst_id, dst_worker_id);
        msg.set_data(layer_idx, src_offset, dst_offset, length);
        q.push(msg);
      }
    }
  }
  void flush() override {
    q.emplace(dst_inst_id, dst_worker_id);
  }

  void send_notification(IIterator<const RequestInfo *> *reqs) override {
    auto opt = reqs->next();
    while (opt.has_value()) {
      auto r = opt.value();
      auto req_id = r->req_id;
      q.emplace(dst_inst_id, dst_worker_id, req_id);
      opt = reqs->next();
    }
  }
};

class ProxyChannel : public IChannel {
 public:
  explicit ProxyChannel(IChannel *ch) : ch_(ch) {}
  void connect(const WorkerInfo &dst_info) override {
    ch_->connect(dst_info);
  }
  void send_data(size_t layer_idx, const std::vector<IpcBlock> &data) override {
    LOG(INFO) << "send " << data.size() << " blocks";
    ch_->send_data(layer_idx, data);
  }
  void flush() override {
    ch_->flush();
  }
  void send_notification(IIterator<const RequestInfo *> *reqs) override {
    ch_->send_notification(reqs);
  }
 private:
  IChannel *ch_;
};

TEST(SendStubTest, UseFakeChannel) {
  uint32_t bs = 16 * KB;
  uint32_t ts = KB;
  WorkerInfo src_info(0, 0);
  src_info.block_size = bs;
  src_info.token_size = ts;

  WorkerInfo dst_info(1, 0);
  dst_info.block_size = bs;
  dst_info.token_size = ts;

  auto fbc = std::make_shared<FakeChannel>();
  auto q = &fbc->q;
  Context ctx(0, 1);
  ctx.set_block_params(bs, ts, 8);
  ctx.set_layer_data_address(0, {0, 8 * bs});
  uint32_t num_layers = 2;
  auto tx = KvSendStub(1, 0, 0, num_layers, std::make_unique<ProxyChannel>(fbc.get()));
  tx.connect(&ctx, dst_info);
  {
    std::vector<const RequestInfo *> reqs;
    // the first send;
    RequestInfo req0(1, 0, "req_id00000000000000000000000000", {0, 1, 2}, {4, 5, 6});
    req0.add_new_tokens(8, false);
    reqs.push_back(&req0);

    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0, std::move(reqs));
    tx.send_batch(task);
    step_0->notify_layer_ready(num_layers);
    while (!step_0->check_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(req0.is_all_transferred());
    EXPECT_EQ(q->size(), 3);
    uint32_t layer = 0;
    while (layer < 2) {
      auto msg = q->front();
      q->pop();
      EXPECT_EQ(msg.dst_inst_id, 1);
      EXPECT_EQ(msg.dst_worker_id, 0);
      EXPECT_TRUE(msg.data.has_value());
      EXPECT_EQ(msg.data.value().layer_idx, layer);
      EXPECT_EQ(msg.data.value().src_offset, 0);
      EXPECT_EQ(msg.data.value().dst_offset, 4 * 16 * KB);
      EXPECT_EQ(msg.data.value().length, 8 * KB);
      layer++;
    }
    auto flush = q->front();
    q->pop();
    EXPECT_EQ(flush.dst_inst_id, 1);
    EXPECT_EQ(flush.dst_worker_id, 0);
  }
  {
    // the second send;
    std::vector<const RequestInfo *> reqs;
    RequestInfo req1(1, 0, "req_id00000000000000000000000001", {0, 1, 2}, {4, 5, 6});
    req1.add_new_tokens(17, false);
    reqs.push_back(&req1);
    auto step_1 = std::make_shared<Step>(1);
    BatchSendTask task(step_1, std::move(reqs));
    tx.send_batch(task);
    step_1->notify_layer_ready(num_layers);
    while (!step_1->check_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(req1.is_all_transferred());
    EXPECT_EQ(q->size(), 3); // because req1 has 17 tokens need two continuous blocks, can be merged;
    {
      auto b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 0);
      EXPECT_EQ(b1.data.value().src_offset, 0);
      EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB);
      EXPECT_EQ(b1.data.value().length, 17 * KB);
      auto b2 = q->front();
      q->pop();
      EXPECT_TRUE(b2.data.has_value());
      EXPECT_EQ(b2.data.value().layer_idx, 1);
      EXPECT_EQ(b2.data.value().src_offset, 0);
      EXPECT_EQ(b2.data.value().dst_offset, 4 * 16 * KB);
      EXPECT_EQ(b2.data.value().length, 17 * KB);
    }
    auto flush = q->front();
    q->pop();
    EXPECT_EQ(flush.dst_inst_id, 1);
    EXPECT_EQ(flush.dst_worker_id, 0);
  }
  {
    // the third send
    std::vector<const RequestInfo *> reqs;
    RequestInfo req3(1, 0, "req_id00000000000000000000000002", {3, 4, 5}, {7, 8, 9});
    req3.set_seen_tokens(17);
    req3.add_new_tokens(16, true);
    reqs.push_back(&req3);
    auto step_2 = std::make_shared<Step>(2);
    BatchSendTask task(step_2, std::move(reqs));
    tx.send_batch(task);
    step_2->notify_layer_ready(num_layers);
    while (!step_2->check_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(req3.is_all_transferred());
    EXPECT_EQ(q->size(), 4);
    {
      auto b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 0);
      EXPECT_EQ(b1.data.value().src_offset, 4 * bs + ts); // 4
      EXPECT_EQ(b1.data.value().dst_offset, 8 * bs + ts);
      EXPECT_EQ(b1.data.value().length, 16 * ts);
      auto b2 = q->front();
      q->pop();
      EXPECT_TRUE(b2.data.has_value());
      EXPECT_EQ(b2.data.value().layer_idx, 1);
      EXPECT_EQ(b2.data.value().src_offset, 4 * bs + ts);
      EXPECT_EQ(b2.data.value().dst_offset, 8 * bs + ts);
      EXPECT_EQ(b2.data.value().length, 16 * ts);
      auto flush = q->front();
      q->pop();
      EXPECT_EQ(flush.dst_inst_id, 1);
      EXPECT_EQ(flush.dst_worker_id, 0);
      auto b3 = q->front();
      q->pop();
      EXPECT_TRUE(b3.req_id.has_value());
      EXPECT_EQ(b3.req_id.value(), "req_id00000000000000000000000002");
    }
  }
}

class MockChannel : public IChannel {
 public:
  MockChannel() = default;
  MOCK_METHOD(void, connect, (const WorkerInfo &dst_info), (override));
  MOCK_METHOD(void, flush, (), (override));
  MOCK_METHOD(void, send_data, (size_t layer_idx, (const std::vector<IpcBlock> &data)), (override));
  MOCK_METHOD(void, send_notification, (IIterator<const RequestInfo *> * reqs), (override));
};

TEST(KvSendStubTest, UseMockChannel) {
  uint32_t bs = 16 * KB;
  uint32_t ts = KB;
  WorkerInfo src_info(0, 0);
  src_info.block_size = bs;
  src_info.token_size = ts;

  WorkerInfo dst_info(1, 0);
  dst_info.block_size = bs;
  dst_info.token_size = ts;

  Context ctx(0, 1);
  ctx.set_block_params(bs, ts, 8);
  ctx.set_layer_data_address(0, {0, 8 * bs});
  uint32_t num_layers = 2;

  std::vector<const RequestInfo *> reqs;
  RequestInfo req0(1, 0, "req_id00000000000000000000000000", {0, 1, 2}, {4, 5, 6});
  req0.add_new_tokens(8, false);
  reqs.push_back(&req0);
  IpcBlock expect_data(0, 4 * bs, 8 * ts);

  MockChannel channel;
  EXPECT_CALL(channel, connect(Field(&WorkerInfo::inst_id, Eq(1)))).Times(1);
  EXPECT_CALL(channel, send_data(Eq(0), ElementsAre(expect_data))).Times(1);
  EXPECT_CALL(channel, send_data(Eq(1), ElementsAre(expect_data))).Times(1);
  EXPECT_CALL(channel, flush()).Times(1);
  EXPECT_CALL(channel, send_notification(_)).Times(0);

  auto tx = KvSendStub(1, 0, 0, num_layers, std::make_unique<ProxyChannel>(&channel));
  tx.connect(&ctx, dst_info);
  auto step_0 = std::make_shared<Step>(0);
  BatchSendTask task(step_0, std::move(reqs));
  tx.send_batch(task);
  step_0->notify_layer_ready(num_layers);
  while (!step_0->check_done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

std::tuple<size_t, size_t, size_t, size_t> merge_interval(std::vector<IpcBlock> &input);

TEST(KvSendStubTest, MergeIntervalTest) {
  {
    std::vector<IpcBlock> data;
    auto [min_size, max_size, total_size, cnt] = merge_interval(data);
    EXPECT_EQ(min_size, std::numeric_limits<size_t>::max());
    EXPECT_EQ(max_size, 0);
    EXPECT_EQ(total_size, 0);
    EXPECT_EQ(cnt, 0);
  }

  {
    std::vector<IpcBlock> edata{{1, 1, 1}};
    std::vector<IpcBlock> data{{1, 1, 1}};
    auto [min_size, max_size, total_size, cnt] = merge_interval(data);
    EXPECT_EQ(min_size, 1);
    EXPECT_EQ(max_size, 1);
    EXPECT_EQ(total_size, 1);
    EXPECT_EQ(data, edata);
    EXPECT_EQ(cnt, 1);
  }

  {
    std::vector<IpcBlock> edata{{1, 1, 1}, {3, 3, 2}};
    std::vector<IpcBlock> data{{1, 1, 1}, {3, 3, 2}};
    auto [min_size, max_size, total_size, cnt] = merge_interval(data);
    EXPECT_EQ(min_size, 1);
    EXPECT_EQ(max_size, 2);
    EXPECT_EQ(total_size, 3);
    EXPECT_EQ(data, edata);
    EXPECT_EQ(cnt, 2);
  }

  {
    std::vector<IpcBlock> edata{{1, 1, 3}, {3, 3, 0}};
    std::vector<IpcBlock> data{{1, 1, 2}, {3, 3, 1}};
    auto [min_size, max_size, total_size, cnt] = merge_interval(data);
    EXPECT_EQ(min_size, 3);
    EXPECT_EQ(max_size, 3);
    EXPECT_EQ(total_size, 3);
    EXPECT_EQ(data, edata);
    EXPECT_EQ(cnt, 1);
  }

  {
    std::vector<IpcBlock> edata{{1, 1, 3}, {3, 3, 0}, {4, 5, 2}};
    std::vector<IpcBlock> data{{1, 1, 2}, {3, 3, 1}, {4, 5, 2}};
    auto [min_size, max_size, total_size, cnt] = merge_interval(data);
    EXPECT_EQ(min_size, 2);
    EXPECT_EQ(max_size, 3);
    EXPECT_EQ(total_size, 5);
    EXPECT_EQ(data, edata);
    EXPECT_EQ(cnt, 2);
  }
}
}
