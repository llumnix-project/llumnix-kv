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

  public:
    bool operator==(const DataEntry& other) const {
      return layer_idx == other.layer_idx &&
             src_offset == other.src_offset &&
             dst_offset == other.dst_offset &&
             length == other.length;
    }
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

  bool operator==(const Message& other) const {
    return dst_inst_id == other.dst_inst_id &&
           dst_worker_id == other.dst_worker_id &&
           req_id == other.req_id &&
           data == other.data;
  }
};

std::ostream& operator<<(std::ostream& os, const Message& msg) {
  os << "Message { ";
  os << "dst_inst_id: " << msg.dst_inst_id << ", ";
  os << "dst_worker_id: " << msg.dst_worker_id;

  if (msg.req_id) {
    os << ", req_id: " << *msg.req_id;
  } else {
    os << ", req_id: nullopt";
  }

  if (msg.data) {
    const Message::DataEntry& dataEntry = *msg.data;
    os << ", data: { ";
    os << "layer_idx: " << dataEntry.layer_idx << ", ";
    os << "src_offset: " << dataEntry.src_offset << ", ";
    os << "dst_offset: " << dataEntry.dst_offset << ", ";
    os << "length: " << dataEntry.length << " }";
  } else {
    os << ", data: nullopt";
  }

  os << " }";
  return os;
}

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

template <typename T>
static std::vector<T> queue2vec(std::queue<T>& input) {
  auto output = std::vector<T>();
  output.reserve(input.size());
  while (!input.empty()) {
    output.emplace_back(std::move(input.front()));
    input.pop();
  }
  return output;
}

static void test_parse_block_generate(int p_rank, int d_rank) {
  int kv_heads = 16;
  int num_layers = 2;
  int head_dim = 256;
  int ntpb = 64;

  auto p_info = WorkerInfo(0, 0);
  p_info.tp_size = 16;
  p_info.worker_tp_rank = p_rank;
  p_info.token_size = 2 * (kv_heads / p_info.tp_size) * head_dim * sizeof(uint16_t);
  p_info.block_size = p_info.token_size * ntpb;
  std::cout << "p_info.token_size=" << p_info.token_size << " p_info.block_size=" << p_info.block_size << std::endl;

  auto d_info = WorkerInfo(1, 0);
  d_info.tp_size = 4;
  d_info.worker_tp_rank = d_rank;
  d_info.token_size = 2 * (kv_heads / d_info.tp_size) * head_dim * sizeof(uint16_t);
  d_info.block_size = d_info.token_size * ntpb;
  std::cout << "d_info.token_size=" << d_info.token_size << " d_info.block_size=" << d_info.block_size << std::endl;

  auto fbc = std::make_unique<FakeChannel>();
  fbc->connect(d_info);
  auto& fbcq = fbc->q;
  auto tx = KvSendStub(d_info, p_info, 0, num_layers, std::move(fbc));
  tx.start();
  EXPECT_EQ(tx.check_state(), StubState::WORKING);

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

    auto actual_q = queue2vec(fbcq);
    auto expect_q = std::vector<Message>();
    if (d_rank == 0) {
      if (p_rank == 0) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 0,    1048576, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 512,  1050624, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1024, 1052672, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1536, 1054720, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2048, 1056768, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2560, 1058816, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3072, 1060864, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3584, 1062912, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4096, 1064960, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4608, 1067008, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5120, 1069056, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5632, 1071104, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6144, 1073152, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6656, 1075200, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7168, 1077248, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7680, 1079296, 512);
        }
      } else if (p_rank == 1) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 0,    1049088, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 512,  1051136, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1024, 1053184, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1536, 1055232, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2048, 1057280, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2560, 1059328, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3072, 1061376, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3584, 1063424, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4096, 1065472, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4608, 1067520, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5120, 1069568, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5632, 1071616, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6144, 1073664, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6656, 1075712, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7168, 1077760, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7680, 1079808, 512);
        }
      } else if (p_rank == 2) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 0,    1049600, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 512,  1051648, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1024, 1053696, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1536, 1055744, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2048, 1057792, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2560, 1059840, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3072, 1061888, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3584, 1063936, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4096, 1065984, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4608, 1068032, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5120, 1070080, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5632, 1072128, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6144, 1074176, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6656, 1076224, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7168, 1078272, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7680, 1080320, 512);
        }
      } else {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 0,    1050112, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 512,  1052160, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1024, 1054208, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1536, 1056256, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2048, 1058304, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2560, 1060352, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3072, 1062400, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3584, 1064448, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4096, 1066496, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4608, 1068544, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5120, 1070592, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5632, 1072640, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6144, 1074688, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6656, 1076736, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7168, 1078784, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7680, 1080832, 512);
        }
      }
    }
    expect_q.emplace_back(1, 0);  // flush
    EXPECT_EQ(actual_q, expect_q);
  }
}

TEST(SendStubTest, ParseBlockSendPGtD) {
  test_parse_block_generate(0, 0);
  test_parse_block_generate(1, 0);
  test_parse_block_generate(2, 0);
  test_parse_block_generate(3, 0);
}


static void dgtp_test_parse_block_generate(int p_rank, int d_rank) {
  int kv_heads = 16;
  int num_layers = 2;
  int head_dim = 256;
  int ntpb = 64;

  auto p_info = WorkerInfo(0, 0);
  p_info.tp_size = 4;
  p_info.worker_tp_rank = p_rank;
  p_info.token_size = 2 * (kv_heads / p_info.tp_size) * head_dim * sizeof(uint16_t);
  p_info.block_size = p_info.token_size * ntpb;
  std::cout << "p_info.token_size=" << p_info.token_size << " p_info.block_size=" << p_info.block_size << std::endl;

  auto d_info = WorkerInfo(1, 0);
  d_info.tp_size = 16;
  d_info.worker_tp_rank = d_rank;
  d_info.token_size = 2 * (kv_heads / d_info.tp_size) * head_dim * sizeof(uint16_t);
  d_info.block_size = d_info.token_size * ntpb;
  std::cout << "d_info.token_size=" << d_info.token_size << " d_info.block_size=" << d_info.block_size << std::endl;

  auto fbc = std::make_unique<FakeChannel>();
  fbc->connect(d_info);
  auto& fbcq = fbc->q;
  auto tx = KvSendStub(d_info, p_info, 0, num_layers, std::move(fbc));
  tx.start();
  EXPECT_EQ(tx.check_state(), StubState::WORKING);

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

    auto actual_q = queue2vec(fbcq);
    auto expect_q = std::vector<Message>();
    if (p_rank == 0) {
      if (d_rank == 0) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 0,     262144, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2048,  262656, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4096,  263168, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6144,  263680, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 8192,  264192, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 10240, 264704, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 12288, 265216, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 14336, 265728, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 16384, 266240, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 18432, 266752, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 20480, 267264, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 22528, 267776, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 24576, 268288, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 26624, 268800, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 28672, 269312, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 30720, 269824, 512);
        }
      } else if (d_rank == 1) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 512,   262144, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 2560,  262656, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 4608,  263168, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 6656,  263680, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 8704,  264192, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 10752, 264704, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 12800, 265216, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 14848, 265728, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 16896, 266240, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 18944, 266752, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 20992, 267264, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 23040, 267776, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 25088, 268288, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 27136, 268800, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 29184, 269312, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 31232, 269824, 512);
        }
      } else if (d_rank == 2) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1024,  262144, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3072,  262656, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5120,  263168, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7168,  263680, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 9216,  264192, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 11264, 264704, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 13312, 265216, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 15360, 265728, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 17408, 266240, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 19456, 266752, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 21504, 267264, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 23552, 267776, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 25600, 268288, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 27648, 268800, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 29696, 269312, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 31744, 269824, 512);
        }
      } else {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 1536,  262144, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 3584,  262656, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 5632,  263168, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 7680,  263680, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 9728,  264192, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 11776, 264704, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 13824, 265216, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 15872, 265728, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 17920, 266240, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 19968, 266752, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 22016, 267264, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 24064, 267776, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 26112, 268288, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 28160, 268800, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 30208, 269312, 512);
          expect_q.emplace_back(1, 0);
          expect_q.back().set_data(layer_idx, 32256, 269824, 512);
        }
      }
    }
    expect_q.emplace_back(1, 0);  // flush
    EXPECT_EQ(actual_q, expect_q);
  }
}

TEST(SendStubTest, ParseBlockSendDGtP) {
  dgtp_test_parse_block_generate(0, 0);
  dgtp_test_parse_block_generate(0, 1);
  dgtp_test_parse_block_generate(0, 2);
  dgtp_test_parse_block_generate(0, 3);
}

TEST(SendStubTest, ParseBlockSendPEqD) {
  uint32_t bs = 16 * KB;
  uint32_t ts = KB;
  WorkerInfo src_info(0, 0);
  src_info.block_size = bs;
  src_info.token_size = ts;

  WorkerInfo dst_info(1, 0);
  dst_info.block_size = bs;
  dst_info.token_size = ts;

  auto fbc = std::make_shared<FakeChannel>();
  fbc->connect(dst_info);
  auto q = &fbc->q;
  Context ctx(0, 1);
  ctx.set_block_params(bs, ts, 8);
  ctx.set_layer_data_address(0, {0, 8 * bs});
  uint32_t num_layers = 2;
  auto tx = KvSendStub(dst_info, src_info, 0, num_layers, std::make_unique<ProxyChannel>(fbc.get()));
  tx.start();
  EXPECT_EQ(tx.check_state(), StubState::WORKING);
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

TEST(SendStubTest, UseMockChannel) {
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
  EXPECT_CALL(channel, send_data(Eq(0), ElementsAre(expect_data))).Times(1);
  EXPECT_CALL(channel, send_data(Eq(1), ElementsAre(expect_data))).Times(1);
  EXPECT_CALL(channel, flush()).Times(1);
  EXPECT_CALL(channel, send_notification(_)).Times(0);

  auto tx = KvSendStub(dst_info, src_info, 0, num_layers, std::make_unique<ProxyChannel>(&channel));
  tx.start();
  auto step_0 = std::make_shared<Step>(0);
  BatchSendTask task(step_0, std::move(reqs));
  tx.send_batch(task);
  step_0->notify_layer_ready(num_layers);
  while (!step_0->check_done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

std::tuple<size_t, size_t, size_t, size_t> merge_interval(std::vector<IpcBlock> &input);

TEST(SendStubTest, MergeIntervalTest) {
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
