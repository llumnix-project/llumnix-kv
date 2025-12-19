#include <queue>
#include <iostream>
#include <limits>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "client.h"
#include "envcfg.h"
#include "thrid_party/logging.h"

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

  InstanceId dst_inst_id;
  uint32_t dst_worker_id;
  std::optional<std::string> req_id;
  std::optional<DataEntry> data{};
  Message(const InstanceId& inst_id, uint32_t worker_id) :
      dst_inst_id(inst_id), dst_worker_id(worker_id) {};

  Message(const InstanceId& inst_id, uint32_t worker_id, const std::string &id) :
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
  InstanceId dst_inst_id;
  uint32_t dst_worker_id;
  std::queue<Message> q;
  std::shared_ptr<std::atomic<int>> flush_cnt = std::make_shared<std::atomic<int>>(0);
  std::vector<std::vector<IpcBlock>>* data_;

  FakeChannel() : dst_inst_id("0"), dst_worker_id(INVALID_INST_WORKER_ID), q() {};
  void connect(const WorkerInfo &info) override {
    dst_inst_id = info.inst_id;
    dst_worker_id = info.worker_id;
  }

  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind) override {
    this->data_ = &data;
    for (auto &tensor_data : data) {
      merge_interval(tensor_data);
    }
  }

  void send_data(size_t layer_idx) override {
    for (const auto &tensor_data : *this->data_) {
      for (const auto &[src_offset, dst_offset, length] : tensor_data) {
        if (length > 0) {
          Message msg(dst_inst_id, dst_worker_id);
          msg.set_data(layer_idx, src_offset, dst_offset, length);
          q.push(msg);
        }
      }
    }
  }
  void flush(std::string&) override {
    q.emplace(dst_inst_id, dst_worker_id);
    flush_cnt->fetch_add(1);
  }

  void send_notification(const std::vector<const ReqSendTask*>& reqs) override {
    for (const auto* r : reqs) {
      auto req_id = r->req_id();
      q.emplace(dst_inst_id, dst_worker_id, req_id);
    }
  }
};

class ProxyChannel : public IChannel {
 public:
  explicit ProxyChannel(IChannel *ch) : ch_(ch) {}
  void connect(const WorkerInfo &dst_info) override {
    ch_->connect(dst_info);
  }
  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind k) override {
    ch_->register_data(data, k);
  }
  void send_data(size_t layer_idx) override {
    ch_->send_data(layer_idx);
  }
  void flush(std::string& o) override {
    ch_->flush(o);
  }
  void send_notification(const std::vector<const ReqSendTask*>& reqs) override {
    ch_->send_notification(reqs);
  }
 private:
  IChannel *ch_;
};


class FakeNamingWorkerClient : public INamingWorkerClient {
  int kind_ = 0;
  int kind3_times_ = 0;
  int first_time_ = true;
  WorkerInfo dst_info_;
public:
  FakeNamingWorkerClient(int k, WorkerInfo d): kind_(k), dst_info_(d) {}

  void register_worker(const WorkerInfo &worker_info) override {
    throw std::runtime_error("biubiu~");
  }

  std::optional<WorkerInfo> get_worker_info(const InstanceId &, WorkerId) override {
    if (this->kind_ != 3 && this->first_time_) {
      this->first_time_ = false;
      return this->dst_info_;
    }
    if (this->kind_ == 0) {
      return std::nullopt;
    }
    if (this->kind_ == 1) {
      return WorkerInfo("DO-NOT-EXISTS-WORKER-ID", 1);
    }
    if (this->kind_ == 2) {
      return this->dst_info_;
    }
    if (kind3_times_ == 0) {
      kind3_times_ += 1;
      throw std::runtime_error("biubiubiu~");
    }
    return this->dst_info_;
  }
};

class FakeChannelFactory : public IChannelFactory {
  IChannel* ch_;
public:
  FakeChannelFactory(IChannel* ch) : ch_(ch) {}

  Channel create(const WorkerInfo& dst_info) override {
    return std::make_unique<ProxyChannel>(ch_);
  }
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

  auto p_info = WorkerInfo("0", 0);
  p_info.tp_size = 16;
  p_info.worker_tp_rank = p_rank;
  p_info.token_sizes = {2 * (kv_heads / p_info.tp_size) * head_dim * sizeof(uint16_t)};
  p_info.block_sizes = {p_info.token_sizes[0] * ntpb};
  std::cout << "p_info.token_size=" << p_info.token_sizes[0] << " p_info.block_size=" << p_info.block_sizes[0] << std::endl;

  auto d_info = WorkerInfo("1", 0);
  d_info.tp_size = 4;
  d_info.worker_tp_rank = d_rank;
  d_info.token_sizes = {2 * (kv_heads / d_info.tp_size) * head_dim * sizeof(uint16_t)};
  d_info.block_sizes = {d_info.token_sizes[0] * ntpb};
  std::cout << "d_info.token_size=" << d_info.token_sizes[0] << " d_info.block_size=" << d_info.block_sizes[0] << std::endl;

  auto fbc = std::make_unique<FakeChannel>();
  fbc->connect(d_info);
  auto& fbcq = fbc->q;
  auto flush_cnt = fbc->flush_cnt;
  auto naming = std::make_shared<FakeNamingWorkerClient>(std::min(p_rank, 2), d_info);
  auto tx = KvSendStub("1", 0, p_info, 0, num_layers,
                       std::make_unique<FakeChannelFactory>(fbc.get()),
                       naming);

  {
    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0);

    std::vector<RequestInfo *> reqs;
    // the first send;
    auto req0p = std::make_shared<RequestInfo>("1", 0, "req_id00000000000000000000000000", std::vector<uint32_t>{0, 1, 2}, std::vector<uint32_t>{4, 5, 6});
    RequestInfo& req0 = *req0p;
    task.tasks.emplace_back(req0p, 0, 8, false);
    reqs.push_back(&req0);

    step_0->notify_layer_ready(num_layers);
    tx.send_batch(std::move(task));
    while (flush_cnt->load() < 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(req0.state() == ReqState::OK);

    auto actual_q = queue2vec(fbcq);
    auto expect_q = std::vector<Message>();
    if (d_rank == 0) {
      if (p_rank == 0) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 0,    1048576, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 512,  1050624, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1024, 1052672, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1536, 1054720, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2048, 1056768, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2560, 1058816, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3072, 1060864, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3584, 1062912, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4096, 1064960, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4608, 1067008, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5120, 1069056, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5632, 1071104, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6144, 1073152, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6656, 1075200, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7168, 1077248, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7680, 1079296, 512);
        }
      } else if (p_rank == 1) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 0,    1049088, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 512,  1051136, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1024, 1053184, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1536, 1055232, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2048, 1057280, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2560, 1059328, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3072, 1061376, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3584, 1063424, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4096, 1065472, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4608, 1067520, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5120, 1069568, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5632, 1071616, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6144, 1073664, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6656, 1075712, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7168, 1077760, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7680, 1079808, 512);
        }
      } else if (p_rank == 2) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 0,    1049600, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 512,  1051648, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1024, 1053696, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1536, 1055744, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2048, 1057792, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2560, 1059840, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3072, 1061888, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3584, 1063936, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4096, 1065984, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4608, 1068032, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5120, 1070080, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5632, 1072128, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6144, 1074176, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6656, 1076224, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7168, 1078272, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7680, 1080320, 512);
        }
      } else {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 0,    1050112, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 512,  1052160, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1024, 1054208, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1536, 1056256, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2048, 1058304, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2560, 1060352, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3072, 1062400, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3584, 1064448, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4096, 1066496, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4608, 1068544, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5120, 1070592, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5632, 1072640, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6144, 1074688, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6656, 1076736, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7168, 1078784, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7680, 1080832, 512);
        }
      }
    }
    expect_q.emplace_back("1", 0);  // flush
    EXPECT_EQ(actual_q, expect_q);
  }
}

TEST(SendStubTest, ParseBlockSendPGtD) {
  if (env_cache_shape() != RAGGED_FLASH_CACHE_SHAPE) {
    return ;
  }
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

  auto p_info = WorkerInfo("0", 0);
  p_info.tp_size = 4;
  p_info.worker_tp_rank = p_rank;
  p_info.token_sizes = {2 * (kv_heads / p_info.tp_size) * head_dim * sizeof(uint16_t)};
  p_info.block_sizes = {p_info.token_sizes[0] * ntpb};
  std::cout << "p_info.token_size=" << p_info.token_sizes[0] << " p_info.block_size=" << p_info.block_sizes[0] << std::endl;

  auto d_info = WorkerInfo("1", 0);
  d_info.tp_size = 16;
  d_info.worker_tp_rank = d_rank;
  d_info.token_sizes = {2 * (kv_heads / d_info.tp_size) * head_dim * sizeof(uint16_t)};
  d_info.block_sizes = {d_info.token_sizes[0] * ntpb};
  std::cout << "d_info.token_size=" << d_info.token_sizes[0] << " d_info.block_size=" << d_info.block_sizes[0] << std::endl;

  auto fbc = std::make_unique<FakeChannel>();
  fbc->connect(d_info);
  auto& fbcq = fbc->q;
  auto flush_cnt = fbc->flush_cnt;
  auto naming = std::make_shared<FakeNamingWorkerClient>(2, d_info);
  auto tx = KvSendStub("1", 0, p_info, 0, num_layers, std::make_unique<FakeChannelFactory>(fbc.get()), naming);

  {
    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0);

    std::vector<RequestInfo *> reqs;
    // the first send;
    auto req0p = std::make_shared<RequestInfo>("1", 0, "req_id00000000000000000000000000", std::vector<uint32_t>{0, 1, 2}, std::vector<uint32_t>{4, 5, 6});
    auto& req0 = *req0p;
    task.tasks.emplace_back(req0p, 0, 8, false);
    reqs.push_back(&req0);

    step_0->notify_layer_ready(num_layers);
    tx.send_batch(std::move(task));
    while (flush_cnt->load() < 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(req0.state() == ReqState::OK);

    auto actual_q = queue2vec(fbcq);
    auto expect_q = std::vector<Message>();
    if (p_rank == 0) {
      if (d_rank == 0) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 0,     262144, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2048,  262656, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4096,  263168, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6144,  263680, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 8192,  264192, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 10240, 264704, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 12288, 265216, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 14336, 265728, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 16384, 266240, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 18432, 266752, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 20480, 267264, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 22528, 267776, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 24576, 268288, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 26624, 268800, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 28672, 269312, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 30720, 269824, 512);
        }
      } else if (d_rank == 1) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 512,   262144, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 2560,  262656, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 4608,  263168, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 6656,  263680, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 8704,  264192, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 10752, 264704, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 12800, 265216, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 14848, 265728, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 16896, 266240, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 18944, 266752, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 20992, 267264, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 23040, 267776, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 25088, 268288, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 27136, 268800, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 29184, 269312, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 31232, 269824, 512);
        }
      } else if (d_rank == 2) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1024,  262144, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3072,  262656, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5120,  263168, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7168,  263680, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 9216,  264192, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 11264, 264704, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 13312, 265216, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 15360, 265728, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 17408, 266240, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 19456, 266752, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 21504, 267264, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 23552, 267776, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 25600, 268288, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 27648, 268800, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 29696, 269312, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 31744, 269824, 512);
        }
      } else {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 1536,  262144, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 3584,  262656, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 5632,  263168, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 7680,  263680, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 9728,  264192, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 11776, 264704, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 13824, 265216, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 15872, 265728, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 17920, 266240, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 19968, 266752, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 22016, 267264, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 24064, 267776, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 26112, 268288, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 28160, 268800, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 30208, 269312, 512);
          expect_q.emplace_back("1", 0);
          expect_q.back().set_data(layer_idx, 32256, 269824, 512);
        }
      }
    }
    expect_q.emplace_back("1", 0);  // flush
    EXPECT_EQ(actual_q, expect_q);
  }
}

TEST(SendStubTest, ParseBlockSendDGtP) {
  if (env_cache_shape() != RAGGED_FLASH_CACHE_SHAPE) {
    return ;
  }
  dgtp_test_parse_block_generate(0, 0);
  dgtp_test_parse_block_generate(0, 1);
  dgtp_test_parse_block_generate(0, 2);
  dgtp_test_parse_block_generate(0, 3);
}

TEST(SendStubTest, ParseBlockSendPEqD) {
  uint32_t bs = 16 * KB; // block byte size
  uint32_t ts = KB;  // token byte size
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};
  src_info.layer_num_blocks = 8;
  size_t src_half_layer_size = src_info.layer_num_blocks * bs / 2;

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};
  dst_info.layer_num_blocks = 16;
  size_t dst_half_layer_size = dst_info.layer_num_blocks * bs / 2;

  auto fbc = std::make_shared<FakeChannel>();
  fbc->connect(dst_info);
  auto flush_cnt = fbc->flush_cnt;
  auto q = &fbc->q;
  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 8);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 8 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;
  auto naming = std::make_shared<FakeNamingWorkerClient>(2, dst_info);
  auto tx = KvSendStub("1", 0, src_info, 0, num_layers, std::make_unique<FakeChannelFactory>(fbc.get()), naming);
  {
    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0);

    std::vector<RequestInfo *> reqs;
    // the first send;
    auto req0p = std::make_shared<RequestInfo>(
      /*InstanceId*/ "1", /*WorkerId*/ 0, 
      /*RequestId*/ "req_id00000000000000000000000000", 
      /*src_blocks*/ std::vector<uint32_t>{0, 1, 2}, 
      /*dst_blocks*/ std::vector<uint32_t>{4, 5, 6}
    );
    auto& req0 = *req0p;
    task.tasks.emplace_back(
      /*RequestInfo*/ req0p, 
      /*seen_tokens*/ 0, 
      /*new_tokens*/ 8, 
      /*reach_last_token*/ false
    );
    reqs.push_back(&req0);

    step_0->notify_layer_ready(num_layers);
    tx.send_batch(std::move(task));
    while (flush_cnt->load() < 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(req0.state() == ReqState::OK);
    if (env_cache_shape() == RAGGED_FLASH_CACHE_SHAPE) {
      EXPECT_EQ(q->size(), 3);
      uint32_t layer = 0;
      while (layer < 2) {
        auto msg = q->front();
        q->pop();
        EXPECT_EQ(msg.dst_inst_id, "1");
        EXPECT_EQ(msg.dst_worker_id, 0);
        EXPECT_TRUE(msg.data.has_value());
        EXPECT_EQ(msg.data.value().layer_idx, layer);
        EXPECT_EQ(msg.data.value().src_offset, 0);
        EXPECT_EQ(msg.data.value().dst_offset, 4 * 16 * KB);
        EXPECT_EQ(msg.data.value().length, 8 * KB);
        layer++;
      }
    } else if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE){
      // KV tensors are continous
      EXPECT_EQ(q->size(), 3);
      auto layer0 = q->front();
      q->pop();
      auto layer1 = q->front();
      q->pop();
      EXPECT_EQ(layer0.dst_inst_id, "1");
      EXPECT_EQ(layer0.dst_worker_id, 0);
      EXPECT_TRUE(layer0.data.has_value());
      EXPECT_EQ(layer0.data.value().layer_idx, 0);
      EXPECT_EQ(layer0.data.value().src_offset, 0);
      EXPECT_EQ(layer0.data.value().dst_offset, 4 * bs);
      EXPECT_EQ(layer0.data.value().length, 8 * KB);

      EXPECT_EQ(layer1.dst_inst_id, "1");
      EXPECT_EQ(layer1.dst_worker_id, 0);
      EXPECT_TRUE(layer1.data.has_value());
      EXPECT_EQ(layer1.data.value().layer_idx, 1);
      EXPECT_EQ(layer1.data.value().src_offset, 0);
      EXPECT_EQ(layer1.data.value().dst_offset, 4 * bs);
      EXPECT_EQ(layer1.data.value().length, 8 * KB);
    }else { // FLASH_CACHE_SHAPE
      // 为啥是5?
      EXPECT_EQ(q->size(), 5);
      auto layer0_k = q->front();
      q->pop();
      auto layer0_v = q->front();
      q->pop();
      auto layer1_k = q->front();
      q->pop();
      auto layer1_v = q->front();
      q->pop();
      EXPECT_EQ(layer0_k.dst_inst_id, "1");
      EXPECT_EQ(layer0_k.dst_worker_id, 0);
      EXPECT_TRUE(layer0_k.data.has_value());
      EXPECT_EQ(layer0_k.data.value().layer_idx, 0);
      EXPECT_EQ(layer0_k.data.value().src_offset, 0);
      EXPECT_EQ(layer0_k.data.value().dst_offset, 4 * bs / 2);
      EXPECT_EQ(layer0_k.data.value().length, 8 * KB / 2);

      EXPECT_EQ(layer0_v.dst_inst_id, "1");
      EXPECT_EQ(layer0_v.dst_worker_id, 0);
      EXPECT_TRUE(layer0_v.data.has_value());
      EXPECT_EQ(layer0_v.data.value().layer_idx, 0);
      EXPECT_EQ(layer0_v.data.value().src_offset, 0 + src_half_layer_size);
      EXPECT_EQ(layer0_v.data.value().dst_offset, 4 * bs / 2 + dst_half_layer_size);
      EXPECT_EQ(layer0_v.data.value().length, 8 * KB / 2);

      EXPECT_EQ(layer1_k.dst_inst_id, "1");
      EXPECT_EQ(layer1_k.dst_worker_id, 0);
      EXPECT_TRUE(layer1_k.data.has_value());
      EXPECT_EQ(layer1_k.data.value().layer_idx, 1);
      EXPECT_EQ(layer1_k.data.value().src_offset, 0);
      EXPECT_EQ(layer1_k.data.value().dst_offset, 4 * bs / 2);
      EXPECT_EQ(layer1_k.data.value().length, 8 * KB / 2);

      EXPECT_EQ(layer1_v.dst_inst_id, "1");
      EXPECT_EQ(layer1_v.dst_worker_id, 0);
      EXPECT_TRUE(layer1_v.data.has_value());
      EXPECT_EQ(layer1_v.data.value().layer_idx, 1);
      EXPECT_EQ(layer1_v.data.value().src_offset, 0 + src_half_layer_size);
      EXPECT_EQ(layer1_v.data.value().dst_offset, 4 * bs / 2 + dst_half_layer_size);
      EXPECT_EQ(layer1_v.data.value().length, 8 * KB / 2);
    }
    auto flush = q->front();
    q->pop();
    EXPECT_EQ(flush.dst_inst_id, "1");
    EXPECT_EQ(flush.dst_worker_id, 0);
  }
  {
    auto step_1 = std::make_shared<Step>(1);
    BatchSendTask task(step_1);

    // the second send;
    std::vector<RequestInfo *> reqs;
    auto req1p = std::make_shared<RequestInfo>(
      /*InstanceId*/ "1", /*WorkerId*/ 0, 
      /*RequestId*/ "req_id00000000000000000000000001", 
      /*src_blocks*/ std::vector<uint32_t>{0, 1, 2}, 
      /*dst_blocks*/ std::vector<uint32_t>{4, 5, 6});
    auto& req1 = *req1p;
    task.tasks.emplace_back(
      /*RequestInfo*/ req1p, 
      /*seen_tokens*/ 0, 
      /*new_tokens*/ 17, 
      /*reach_last_token*/ false
    );
    reqs.push_back(&req1);
    step_1->notify_layer_ready(num_layers);
    tx.send_batch(std::move(task));
    while (flush_cnt->load() < 2) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(req1.state() == ReqState::OK);
    if (env_cache_shape() == RAGGED_FLASH_CACHE_SHAPE) {
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
    } else if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      EXPECT_EQ(q->size(), 3); // because req1 has 17 tokens need two continuous blocks, can be merged;
      {
        auto b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 0);
        EXPECT_EQ(b1.data.value().src_offset, 0);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB);
        EXPECT_EQ(b1.data.value().length, 17 * KB);

        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 1);
        EXPECT_EQ(b1.data.value().src_offset, 0);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB);
        EXPECT_EQ(b1.data.value().length, 17 * KB);
      }
    }else {// FLASH_CACHE_SHAPE
      EXPECT_EQ(q->size(), 5); // because req1 has 17 tokens need two continuous blocks, can be merged;
      {
        auto b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 0);
        EXPECT_EQ(b1.data.value().src_offset, 0);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB / 2);
        EXPECT_EQ(b1.data.value().length, 17 * KB / 2);

        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 0);
        EXPECT_EQ(b1.data.value().src_offset, 0 + src_half_layer_size);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB / 2 + dst_half_layer_size);
        EXPECT_EQ(b1.data.value().length, 17 * KB / 2);

        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 1);
        EXPECT_EQ(b1.data.value().src_offset, 0);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB / 2);
        EXPECT_EQ(b1.data.value().length, 17 * KB / 2);

        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 1);
        EXPECT_EQ(b1.data.value().src_offset, 0 + src_half_layer_size);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB / 2 + dst_half_layer_size);
        EXPECT_EQ(b1.data.value().length, 17 * KB / 2);
      }
    }
    auto flush = q->front();
    q->pop();
    EXPECT_EQ(flush.dst_inst_id, "1");
    EXPECT_EQ(flush.dst_worker_id, 0);
  }
  {
    auto step_2 = std::make_shared<Step>(2);
    BatchSendTask task(step_2);

    // the third send
    std::vector<RequestInfo *> reqs;
    auto req3p = std::make_shared<RequestInfo>(
      /*InstanceId*/ "1", /*WorkerId*/ 0, 
      /*RequestId*/ "req_id00000000000000000000000002", 
      /*src_blocks*/ std::vector<uint32_t>{3, 4, 5}, 
      /*dst_blocks*/ std::vector<uint32_t>{7, 8, 9});
    auto& req3 = *req3p;
    task.tasks.emplace_back(
      /*RequestInfo*/ req3p, 
      /*seen_tokens*/ 17, 
      /*new_tokens*/ 16, 
      /*reach_last_token*/ true
    );
    reqs.push_back(&req3);
    step_2->notify_layer_ready(num_layers);
    tx.send_batch(std::move(task));
    while (flush_cnt->load() < 3) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(req3.state() == ReqState::OK);
    if (env_cache_shape() == RAGGED_FLASH_CACHE_SHAPE) {
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
      }
    } else if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE)
    {
      auto b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 0);
      EXPECT_EQ(b1.data.value().src_offset, (4 * bs + ts));
      EXPECT_EQ(b1.data.value().dst_offset, (8 * bs + ts));
      EXPECT_EQ(b1.data.value().length, (3 * bs - 17 * ts));

      b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 1);
      EXPECT_EQ(b1.data.value().src_offset, (4 * bs + ts));
      EXPECT_EQ(b1.data.value().dst_offset, (8 * bs + ts));
      EXPECT_EQ(b1.data.value().length, (3 * bs - 17 * ts));
    } else { // FLASH_CACHE_SHAPE
      EXPECT_EQ(q->size(), 4 + 2);
      {
        auto b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 0);
        EXPECT_EQ(b1.data.value().src_offset, (4 * bs + ts) / 2); // 4
        EXPECT_EQ(b1.data.value().dst_offset, (8 * bs + ts) / 2);
        EXPECT_EQ(b1.data.value().length, 16 * ts / 2);
        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 0);
        EXPECT_EQ(b1.data.value().src_offset, (4 * bs + ts) / 2 + src_half_layer_size); // 4
        EXPECT_EQ(b1.data.value().dst_offset, (8 * bs + ts) / 2 + dst_half_layer_size);
        EXPECT_EQ(b1.data.value().length, 16 * ts / 2);

        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 1);
        EXPECT_EQ(b1.data.value().src_offset, (4 * bs + ts) / 2); // 4
        EXPECT_EQ(b1.data.value().dst_offset, (8 * bs + ts) / 2);
        EXPECT_EQ(b1.data.value().length, 16 * ts / 2);
        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 1);
        EXPECT_EQ(b1.data.value().src_offset, (4 * bs + ts) / 2 + src_half_layer_size); // 4
        EXPECT_EQ(b1.data.value().dst_offset, (8 * bs + ts) / 2 + dst_half_layer_size);
        EXPECT_EQ(b1.data.value().length, 16 * ts / 2);
      }
    }
    auto flush = q->front();
    q->pop();
    EXPECT_EQ(flush.dst_inst_id, "1");
    EXPECT_EQ(flush.dst_worker_id, 0);
    auto b3 = q->front();
    q->pop();
    EXPECT_TRUE(b3.req_id.has_value());
    EXPECT_EQ(b3.req_id.value(), "req_id00000000000000000000000002");
  }
  {
    // the fourth send, only test hybrid block parsing method 
    // when reach_last_token is True, kvt should send all blocks even tokens are not enough
    if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE){
      auto step_3 = std::make_shared<Step>(3);
      BatchSendTask task(step_3);

      std::vector<RequestInfo *> reqs;
      auto req4p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000003", 
        /*src_blocks*/ std::vector<uint32_t>{3, 4, 5, 6}, 
        /*dst_blocks*/ std::vector<uint32_t>{7, 8, 9, 10});
      auto& req4 = *req4p;
      task.tasks.emplace_back(
        /*RequestInfo*/ req4p, 
        /*seen_tokens*/ 1, 
        /*new_tokens*/ 3, 
        /*reach_last_token*/ true
      );

      reqs.push_back(&req4);
      step_3->notify_layer_ready(num_layers);
      tx.send_batch(std::move(task));
      EXPECT_TRUE(req4.state() == ReqState::OK);

      auto b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 0);
      EXPECT_EQ(b1.data.value().src_offset, (3 * bs + ts));
      EXPECT_EQ(b1.data.value().dst_offset, (7 * bs + ts));
      EXPECT_EQ(b1.data.value().length, (4 * bs - ts));

      b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 1);
      EXPECT_EQ(b1.data.value().src_offset, (3 * bs + ts));
      EXPECT_EQ(b1.data.value().dst_offset, (7 * bs + ts));
      EXPECT_EQ(b1.data.value().length, (4 * bs - ts));
    }
  }
}

class MockChannel : public IChannel {
 public:
  MockChannel() = default;
  MOCK_METHOD(void, connect, (const WorkerInfo &dst_info), (override));
  MOCK_METHOD(void, register_data, ((std::vector<std::vector<IpcBlock>>& data), TPKind kind), (override));
  MOCK_METHOD(void, flush, (std::string& out), (override));
  MOCK_METHOD(void, send_data, (size_t layer_idx), (override));
  MOCK_METHOD(void, send_notification, (const std::vector<const ReqSendTask*>& reqs), (override));
};

TEST(SendStubTest, UseMockChannel) {
  if (env_cache_shape() != RAGGED_FLASH_CACHE_SHAPE) {
    return ;
  }
  uint32_t bs = 16 * KB;
  uint32_t ts = KB;
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};

  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 8);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 8 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;

  auto step_0 = std::make_shared<Step>(0);
  BatchSendTask task(step_0);
  std::vector<RequestInfo *> reqs;
  auto req0p = std::make_shared<RequestInfo>("1", 0, "req_id00000000000000000000000000", std::vector<uint32_t>{0, 1, 2}, std::vector<uint32_t>{4, 5, 6});
  auto& req0 = *req0p;
  task.tasks.emplace_back(req0p, 0, 8, false);
  reqs.push_back(&req0);
  IpcBlock expect_data(0, 4 * bs, 8 * ts);
  std::vector<std::vector<IpcBlock>> expect_datas = {{expect_data}};

  MockChannel channel;
  EXPECT_CALL(channel, register_data(ElementsAre(ElementsAre(expect_data)), Eq(TPKind::PEQD))).Times(1);
  EXPECT_CALL(channel, send_data(Eq(0))).Times(1);
  EXPECT_CALL(channel, send_data(Eq(1))).Times(1);
  EXPECT_CALL(channel, flush(_)).Times(1);
  EXPECT_CALL(channel, send_notification(_)).Times(0);

  {
    auto naming = std::make_shared<FakeNamingWorkerClient>(2, dst_info);
    auto tx = KvSendStub("1", 0, src_info, 0, num_layers, std::make_unique<FakeChannelFactory>(&channel), naming);
    step_0->notify_layer_ready(num_layers);
    tx.send_batch(std::move(task));
    usleep(3 * 1000 * 1000);  // 3s
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

  {
    // prompt cache 可能会生成重叠的区间.
    std::vector<IpcBlock> edata{{1, 1, 2}, {1, 3, 3}, {2, 4, 0}};
    std::vector<IpcBlock> data{{1, 1, 2}, {1, 3, 1}, {2, 4, 2}};
    auto [min_size, max_size, total_size, cnt] = merge_interval(data);
    EXPECT_EQ(min_size, 2);
    EXPECT_EQ(max_size, 3);
    EXPECT_EQ(total_size, 5);
    EXPECT_EQ(data, edata);
    EXPECT_EQ(cnt, 2);
  }
}


class ProxyChannelFactory : public IChannelFactory {
  IChannelFactory* factory_;
public:
  ProxyChannelFactory(IChannelFactory* f): factory_(f) {}

  Channel create(const WorkerInfo& dst_info) override {
    return factory_->create(dst_info);
  }
};

static constexpr uint32_t bs = 16 * KB;
static constexpr uint32_t ts = KB;


// FaultTolerantTestChannel
class FTTC : public IChannel {
  int ch_id_ = 0;
  bool* dead_ = nullptr;
  bool* register_data_called_ = nullptr;
  bool* send_data_called_ = nullptr;
  bool* flush_called_ = nullptr;
  bool* send_notification_called_ = nullptr;
public:
  FTTC(int ch_id, bool* dead, bool* register_data_called, bool* send_data_called, bool* flush_called, bool* send_notification_called):
    ch_id_(ch_id),
    dead_(dead),
    register_data_called_(register_data_called),
    send_data_called_(send_data_called),
    flush_called_(flush_called),
    send_notification_called_(send_notification_called) {}

  ~FTTC() {
    *dead_ = true;
  }

  void connect(const WorkerInfo &dst_info) {}
  void register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) override {
    *register_data_called_ = true;
    if (ch_id_ == 0) {
    } else if (ch_id_ == 1) {
      IpcBlock expect_data(0, 4 * bs, 8 * ts);
      EXPECT_EQ(data.size(), 1);
      EXPECT_EQ(data[0].size(), 1);
      EXPECT_EQ(data[0][0], expect_data);
    } else {
      EXPECT_TRUE(false);
    }
  }

  void send_data(size_t layer_index) override {
    *send_data_called_ = true;
    if (ch_id_ == 0) {
      if (layer_index == 1) {
        throw std::runtime_error("biubiu");
      }
    } else if (ch_id_ == 1) {
    } else {
      EXPECT_TRUE(false);
    }
  }

  void flush(std::string& out) override {
    *flush_called_ = true;
  }

  void send_notification(const std::vector<const ReqSendTask*>& reqs) override {
    *send_notification_called_ = true;
    EXPECT_EQ(ch_id_, 1);
    EXPECT_EQ(reqs.size(), 2);
    EXPECT_EQ(reqs[0]->req_id(), "2");
    EXPECT_EQ(reqs[0]->state(), ReqState::FAILED);
    EXPECT_EQ(reqs[1]->req_id(), "3");
    EXPECT_EQ(reqs[1]->state(), ReqState::OK);
  }
};

// FaultTolerantTestChannelFactory
struct FTTCF: public IChannelFactory {
  int ch_id_ = 0;
  bool dead_ch_[2] = {false, false};
  bool register_data_called_[2] = {false, false};
  bool send_data_called_[2] = {false, false};
  bool flush_called_[2] = {false, false};
  bool send_notification_called_[2] = {false, false};
public:
  FTTCF() {}

  Channel create(const WorkerInfo& dst_info) override {
    auto ch_id = ch_id_++;
    EXPECT_TRUE(ch_id == 0 || ch_id == 1);
    return std::make_unique<FTTC>(ch_id,
      &dead_ch_[ch_id],
      &register_data_called_[ch_id],
      &send_data_called_[ch_id],
      &flush_called_[ch_id],
      &send_notification_called_[ch_id]);
  }
};


TEST(SendStubTest, FaultTolerantTest) {
  if (env_cache_shape() != RAGGED_FLASH_CACHE_SHAPE) {
    return ;
  }
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};

  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 8);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 8 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;

  FTTCF fttcf;
  auto naming = std::make_shared<FakeNamingWorkerClient>(2, dst_info);
  auto tx = std::make_unique<KvSendStub>("1", 0, src_info, 0, num_layers, std::make_unique<ProxyChannelFactory>(&fttcf), naming);

  auto req1p = std::make_shared<RequestInfo>("1", 0, "1", std::vector<uint32_t>{10, 11, 12}, std::vector<uint32_t>{14, 15, 16});
  auto& req1 = *req1p;

  auto req2p = std::make_shared<RequestInfo>("1", 0, "2", std::vector<uint32_t>{17, 18, 19}, std::vector<uint32_t>{20, 21, 22});
  auto& req2 = *req2p;

  auto req3p = std::make_shared<RequestInfo>("1", 0, "3", std::vector<uint32_t>{0, 1, 2}, std::vector<uint32_t>{4, 5, 6});
  auto& req3 = *req3p;
  {
    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0);
    task.tasks.emplace_back(req1p, 0, 8, true);
    task.tasks.emplace_back(req2p, 0, 8, false);
    step_0->notify_layer_ready(num_layers);
    tx->send_batch(std::move(task));

    while (req1.state() == ReqState::INPROCESS) {
      usleep(10 * 1000);
    }
    while (req2.state() == ReqState::INPROCESS) {
      usleep(10 * 1000);
    }
    EXPECT_EQ(req1.state(), ReqState::FAILED);
    EXPECT_EQ(req2.state(), ReqState::FAILED);
    while (!fttcf.dead_ch_[0]) {
      usleep(10 * 1000);
    }
    EXPECT_TRUE(fttcf.register_data_called_[0]);
    EXPECT_TRUE(fttcf.send_data_called_[0]);
    EXPECT_FALSE(fttcf.flush_called_[0]);
    EXPECT_FALSE(fttcf.send_notification_called_[0]);

    auto step_1 = std::make_shared<Step>(1);
    BatchSendTask task1(step_1);
    task1.tasks.emplace_back(req1p, 0, 8, true);
    task1.tasks.emplace_back(req2p, 0, 8, false);
    tx->send_batch(std::move(task1));
  }

  {
    auto step_2 = std::make_shared<Step>(2);
    BatchSendTask task2(step_2);
    task2.tasks.emplace_back(req2p, 8, 16, true);
    task2.tasks.emplace_back(req3p, 0, 8, true);
    step_2->notify_layer_ready(num_layers);
    tx->send_batch(std::move(task2));

    while (req3.state() == ReqState::INPROCESS) {
      usleep(10 * 1000);
    }
    EXPECT_EQ(req1.state(), ReqState::FAILED);
    EXPECT_EQ(req2.state(), ReqState::FAILED);
    EXPECT_EQ(req3.state(), ReqState::OK);
    EXPECT_TRUE(fttcf.register_data_called_[1]);
    EXPECT_TRUE(fttcf.send_data_called_[1]);
    EXPECT_TRUE(fttcf.flush_called_[1]);
    EXPECT_TRUE(fttcf.send_notification_called_[1]);
    EXPECT_FALSE(fttcf.dead_ch_[1]);
  }
  tx.reset();
  while (!fttcf.dead_ch_[1]) {
    usleep(10 * 1000);
  }
}


TEST(SendStubTest, CreateChannelFaultTolerantTest) {
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};

  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 8);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 8 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;

  FTTCF fttcf;
  auto naming = std::make_shared<FakeNamingWorkerClient>(3, dst_info);
  auto tx = std::make_unique<KvSendStub>("1", 0, src_info, 0, num_layers, std::make_unique<ProxyChannelFactory>(&fttcf), naming);

  auto req1p = std::make_shared<RequestInfo>("1", 0, "1", std::vector<uint32_t>{10, 11, 12}, std::vector<uint32_t>{14, 15, 16});
  auto& req1 = *req1p;

  auto req2p = std::make_shared<RequestInfo>("1", 0, "2", std::vector<uint32_t>{17, 18, 19}, std::vector<uint32_t>{20, 21, 22});
  auto& req2 = *req2p;

  auto req3p = std::make_shared<RequestInfo>("1", 0, "3", std::vector<uint32_t>{0, 1, 2}, std::vector<uint32_t>{4, 5, 6});
  auto& req3 = *req3p;
  {
    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0);
    task.tasks.emplace_back(req1p, 0, 8, true);
    task.tasks.emplace_back(req2p, 0, 8, false);
    step_0->notify_layer_ready(num_layers);
    tx->send_batch(std::move(task));

    while (req1.state() == ReqState::INPROCESS) {
      usleep(10 * 1000);
    }
    while (req2.state() == ReqState::INPROCESS) {
      usleep(10 * 1000);
    }
    EXPECT_EQ(req1.state(), ReqState::FAILED);
    EXPECT_EQ(req2.state(), ReqState::FAILED);
    EXPECT_FALSE(fttcf.register_data_called_[0]);
    EXPECT_FALSE(fttcf.send_data_called_[0]);
    EXPECT_FALSE(fttcf.flush_called_[0]);
    EXPECT_FALSE(fttcf.send_notification_called_[0]);

    auto step_1 = std::make_shared<Step>(1);
    BatchSendTask task1(step_1);
    task1.tasks.emplace_back(req1p, 0, 8, true);
    task1.tasks.emplace_back(req2p, 0, 8, false);
    tx->send_batch(std::move(task1));
  }

  {
    auto step_2 = std::make_shared<Step>(2);
    BatchSendTask task2(step_2);
    task2.tasks.emplace_back(req2p, 8, 16, true);
    task2.tasks.emplace_back(req3p, 0, 8, true);
    step_2->notify_layer_ready(num_layers);
    tx->send_batch(std::move(task2));

    while (req3.state() == ReqState::INPROCESS) {
      usleep(10 * 1000);
    }
    EXPECT_EQ(req1.state(), ReqState::FAILED);
    EXPECT_EQ(req2.state(), ReqState::FAILED);
    EXPECT_EQ(req3.state(), ReqState::FAILED);
    EXPECT_TRUE(fttcf.register_data_called_[0]);
    EXPECT_TRUE(fttcf.send_data_called_[0]);
    EXPECT_FALSE(fttcf.flush_called_[0]);
    EXPECT_FALSE(fttcf.send_notification_called_[0]);
  }
  tx.reset();
  while (!fttcf.dead_ch_[0]) {
    usleep(10 * 1000);
  }
}


}
