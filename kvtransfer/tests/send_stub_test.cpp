#include <queue>
#include <iostream>
#include <limits>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "client.h"
#include "envcfg.h"
#include "tx_stub.h"
#include "parse_block_common.h"
#include "../src/parse_block_qwen3_next_internal.h"
#include <cstdlib>
#include "thrid_party/logging.h"

using ::testing::Return;
using ::testing::_;
using ::testing::Field;
using ::testing::Eq;
using ::testing::WhenSorted;
using ::testing::ElementsAre;

namespace blade_llm {

TEST(Qwen4PleParseBlockTest, PEqDCopiesFullPaddedBlock) {
  constexpr size_t block_size = 64;
  const std::vector<std::vector<uint32_t>> src_blocks{{}, {}, {2}};
  const std::vector<std::vector<uint32_t>> dst_blocks{{}, {}, {5}};
  const IpcBlockBounds bounds{/*src_capacity=*/1024,
                              /*dst_capacity=*/1024,
                              /*src_length_scale=*/1,
                              /*cache_idx=*/0};
  std::vector<IpcBlock> blocks;

  parse_hybrid_short_conv_block_send_p_eq_d(
      block_size, /*num_ple_layers=*/1, /*ple_block_group=*/2,
      src_blocks, dst_blocks, bounds, blocks);

  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks[0].src_offset, 2u * block_size);
  EXPECT_EQ(blocks[0].dst_offset, 5u * block_size);
  EXPECT_EQ(blocks[0].length, block_size);
}

TEST(Qwen4PleParseBlockTest, PGtDCopiesOnceFromRepresentativeRank) {
  constexpr size_t p_block_size = 64;
  constexpr size_t d_block_size = 96;
  WorkerInfo src("0", 0);
  src.ple_conv_state_shape = {32, 3, 4};
  src.ple_conv_elem_size = 2;
  const std::vector<std::vector<uint32_t>> src_blocks{{}, {}, {2}};
  const std::vector<std::vector<uint32_t>> dst_blocks{{}, {}, {5}};
  const IpcBlockBounds bounds{/*src_capacity=*/1024,
                              /*dst_capacity=*/1024,
                              /*src_length_scale=*/1,
                              /*cache_idx=*/0};

  std::vector<IpcBlock> representative_blocks;
  parse_hybrid_short_conv_block_send_p_gt_d(
      p_block_size, d_block_size,
      /*num_ple_layers=*/1, /*ple_block_group=*/2,
      /*gdn_group_n=*/2, /*gdn_group_off=*/0,
      &src, src_blocks, dst_blocks, bounds, representative_blocks);

  ASSERT_EQ(representative_blocks.size(), 1u);
  EXPECT_EQ(representative_blocks[0].src_offset, 2u * p_block_size);
  EXPECT_EQ(representative_blocks[0].dst_offset, 5u * d_block_size);
  EXPECT_EQ(representative_blocks[0].length, 3u * 4u * 2u);

  std::vector<IpcBlock> duplicate_blocks;
  parse_hybrid_short_conv_block_send_p_gt_d(
      p_block_size, d_block_size,
      /*num_ple_layers=*/1, /*ple_block_group=*/2,
      /*gdn_group_n=*/2, /*gdn_group_off=*/1,
      &src, src_blocks, dst_blocks, bounds, duplicate_blocks);
  EXPECT_TRUE(duplicate_blocks.empty());
}

TEST(Qwen4PleParseBlockTest, PLtDFansOutCompleteState) {
  constexpr size_t p_block_size = 96;
  constexpr size_t d_block_size = 64;
  WorkerInfo src("0", 0);
  src.ple_conv_state_shape = {32, 3, 4};
  src.ple_conv_elem_size = 2;
  const std::vector<std::vector<uint32_t>> src_blocks{{}, {}, {2}};
  const std::vector<std::vector<uint32_t>> dst_blocks{{}, {}, {5}};
  const IpcBlockBounds bounds{/*src_capacity=*/1024,
                              /*dst_capacity=*/1024,
                              /*src_length_scale=*/1,
                              /*cache_idx=*/0};

  for (uint32_t group_off = 0; group_off < 2; ++group_off) {
    std::vector<IpcBlock> blocks;
    parse_hybrid_short_conv_block_send_p_lt_d(
        p_block_size, d_block_size,
        /*num_ple_layers=*/1, /*ple_block_group=*/2,
        /*gdn_group_n=*/2, group_off,
        &src, src_blocks, dst_blocks, bounds, blocks);

    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].src_offset, 2u * p_block_size);
    EXPECT_EQ(blocks[0].dst_offset, 5u * d_block_size);
    EXPECT_EQ(blocks[0].length, 3u * 4u * 2u);
  }
}

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
  p_info.engine_tp_size = 16;
  p_info.worker_tp_rank = p_rank;
  p_info.token_sizes = {2 * (kv_heads / p_info.engine_tp_size) * head_dim * sizeof(uint16_t)};
  p_info.block_sizes = {p_info.token_sizes[0] * ntpb};
  // Synthetic requests below use block ids up to 6. Keep the advertised
  // allocation consistent with those ids so production range validation is
  // exercised against a realistic capacity.
  p_info.layer_num_blocks = 8;
  std::cout << "p_info.token_size=" << p_info.token_sizes[0] << " p_info.block_size=" << p_info.block_sizes[0] << std::endl;

  auto d_info = WorkerInfo("1", 0);
  d_info.engine_tp_size = 4;
  d_info.worker_tp_rank = d_rank;
  d_info.token_sizes = {2 * (kv_heads / d_info.engine_tp_size) * head_dim * sizeof(uint16_t)};
  d_info.block_sizes = {d_info.token_sizes[0] * ntpb};
  d_info.layer_num_blocks = 8;
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
    auto req0p = std::make_shared<RequestInfo>("1", 0, "req_id00000000000000000000000000", BlockIds{{0, 1, 2}}, BlockIds{{4, 5, 6}});
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
  p_info.engine_tp_size = 4;
  p_info.worker_tp_rank = p_rank;
  p_info.token_sizes = {2 * (kv_heads / p_info.engine_tp_size) * head_dim * sizeof(uint16_t)};
  p_info.block_sizes = {p_info.token_sizes[0] * ntpb};
  // Synthetic requests below use block ids up to 6.
  p_info.layer_num_blocks = 8;
  std::cout << "p_info.token_size=" << p_info.token_sizes[0] << " p_info.block_size=" << p_info.block_sizes[0] << std::endl;

  auto d_info = WorkerInfo("1", 0);
  d_info.engine_tp_size = 16;
  d_info.worker_tp_rank = d_rank;
  d_info.token_sizes = {2 * (kv_heads / d_info.engine_tp_size) * head_dim * sizeof(uint16_t)};
  d_info.block_sizes = {d_info.token_sizes[0] * ntpb};
  d_info.layer_num_blocks = 8;
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
    auto req0p = std::make_shared<RequestInfo>("1", 0, "req_id00000000000000000000000000", BlockIds{{0, 1, 2}}, BlockIds{{4, 5, 6}});
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
  // QWEN3_NEXT_FLASHINFER has stricter preconditions (attn_kernel_blk_ntpb,
  // conv/ssm state shapes, etc.) that this generic P==D test does not set
  // up. Coverage for that shape lives in the dedicated
  // Qwen3NextFlashinferPEqD test below.
  if (env_cache_shape() == QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  uint32_t bs = 16 * KB; // block byte size
  uint32_t ts = KB;  // token byte size
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};
  // Qwen3 Next synthetic block groups below use source block id 8.
  src_info.layer_num_blocks = 16;
  // The generic P==D path here means engine_tp_size on both sides should
  // be equal. We pick 1 explicitly so selection_p_tp == dst.engine_tp_size
  // and we land on the P==D branch (matches the old default of kvt_tp_size=1
  // that this test used to rely on implicitly).
  src_info.engine_tp_size = 1;
  size_t src_half_layer_size = src_info.layer_num_blocks * bs / 2;

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};
  dst_info.engine_tp_size = 1;
  dst_info.layer_num_blocks = 16;
  size_t dst_half_layer_size = dst_info.layer_num_blocks * bs / 2;

  auto fbc = std::make_shared<FakeChannel>();
  fbc->connect(dst_info);
  auto flush_cnt = fbc->flush_cnt;
  auto q = &fbc->q;
  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 16);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 16 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;
  auto naming = std::make_shared<FakeNamingWorkerClient>(2, dst_info);
  auto tx = KvSendStub("1", 0, src_info, 0, num_layers, std::make_unique<FakeChannelFactory>(fbc.get()), naming);
  {
    auto step_0 = std::make_shared<Step>(0);
    BatchSendTask task(step_0);

    std::vector<RequestInfo *> reqs;
    std::shared_ptr<RequestInfo> req0p;
    if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      // BLLM_KVTRANS_NUM_GDN_LAYERS default is 3
      // BlockIds layout: [gdn_group_0, gdn_group_1, gdn_group_2, attn_group]
      req0p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000000", 
        /*src_blocks*/ BlockIds{{6}, {7}, {8}, {0, 1, 2}}, 
        /*dst_blocks*/ BlockIds{{10}, {11}, {12}, {4, 5, 6}});
      // after parse block, ther will be no data to send
      // we need to manually increment the flush count 
      // in case of waiting for flush
      flush_cnt->fetch_add(1);
    } else {
      req0p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000000", 
        /*src_blocks*/ BlockIds{{0, 1, 2}}, 
        /*dst_blocks*/ BlockIds{{4, 5, 6}});
    }
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
      // new_tokens < block_size, no data to send
      EXPECT_EQ(q->size(), 0);
    }else { // FLASH_CACHE_SHAPE
      // Why is this 5?
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
    if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      auto flush = q->front();
      q->pop();
      EXPECT_EQ(flush.dst_inst_id, "1");
      EXPECT_EQ(flush.dst_worker_id, 0);
    }
  }
  {
    auto step_1 = std::make_shared<Step>(1);
    BatchSendTask task(step_1);

    // the second send;
    std::vector<RequestInfo *> reqs;
    std::shared_ptr<RequestInfo> req1p;
    if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      // BLLM_KVTRANS_NUM_GDN_LAYERS default is 3
      // BlockIds layout: [gdn_group_0, gdn_group_1, gdn_group_2, attn_group]
      req1p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000001", 
        /*src_blocks*/ BlockIds{{6}, {7}, {8}, {0, 1, 2}}, 
        /*dst_blocks*/ BlockIds{{10}, {11}, {12}, {4, 5, 6}});
    } else {
      req1p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000001", 
        /*src_blocks*/ BlockIds{{0, 1, 2}}, 
        /*dst_blocks*/ BlockIds{{4, 5, 6}});
    }
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
      // req1 has 17 tokens, one total block and one partial block,
      // only send the total block
      EXPECT_EQ(q->size(), 3); // flush_msg/layer0/layer1
      {
        auto b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 0);
        EXPECT_EQ(b1.data.value().src_offset, 0);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB);
        EXPECT_EQ(b1.data.value().length, 16 * KB);

        b1 = q->front();
        q->pop();
        EXPECT_TRUE(b1.data.has_value());
        EXPECT_EQ(b1.data.value().layer_idx, 1);
        EXPECT_EQ(b1.data.value().src_offset, 0);
        EXPECT_EQ(b1.data.value().dst_offset, 4 * 16 * KB);
        EXPECT_EQ(b1.data.value().length, 16 * KB);
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
    std::shared_ptr<RequestInfo> req3p;
    if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE) {
      // BLLM_KVTRANS_NUM_GDN_LAYERS default is 3
      // BlockIds layout: [gdn_group_0, gdn_group_1, gdn_group_2, attn_group]
      req3p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000002", 
        /*src_blocks*/ BlockIds{{6}, {7}, {8}, {3, 4, 5}}, 
        /*dst_blocks*/ BlockIds{{10}, {11}, {12}, {7, 8, 9}});
    } else {
      req3p = std::make_shared<RequestInfo>(
        /*InstanceId*/ "1", /*WorkerId*/ 0, 
        /*RequestId*/ "req_id00000000000000000000000002", 
        /*src_blocks*/ BlockIds{{3, 4, 5}}, 
        /*dst_blocks*/ BlockIds{{7, 8, 9}});
    }

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
      EXPECT_EQ(q->size(), 3);
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
      // req3 reached last token, send all blocks
      auto b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 0);
      // block aligned, no partial block
      EXPECT_EQ(b1.data.value().src_offset, (4 * bs));
      EXPECT_EQ(b1.data.value().dst_offset, (8 * bs));
      EXPECT_EQ(b1.data.value().length, (5 * bs));

      b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 1);
      EXPECT_EQ(b1.data.value().src_offset, (4 * bs));
      EXPECT_EQ(b1.data.value().dst_offset, (8 * bs));
      EXPECT_EQ(b1.data.value().length, (5 * bs));
    } else { // FLASH_CACHE_SHAPE
      EXPECT_EQ(q->size(), 4 + 1);
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
        /*src_blocks*/ BlockIds{{3}, {4}, {5}, {6}}, 
        /*dst_blocks*/ BlockIds{{7}, {8}, {9}, {10}});
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
      EXPECT_EQ(b1.data.value().src_offset, (3 * bs));
      EXPECT_EQ(b1.data.value().dst_offset, (7 * bs));
      EXPECT_EQ(b1.data.value().length, (4 * bs));

      b1 = q->front();
      q->pop();
      EXPECT_TRUE(b1.data.has_value());
      EXPECT_EQ(b1.data.value().layer_idx, 1);
      EXPECT_EQ(b1.data.value().src_offset, (3 * bs));
      EXPECT_EQ(b1.data.value().dst_offset, (7 * bs));
      EXPECT_EQ(b1.data.value().length, (4 * bs));
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
};

TEST(SendStubTest, GeneratedIpcBlockBoundsAreCheckedBeforeAppend) {
  const IpcBlockBounds bounds{
      /*src_capacity=*/1024,
      /*dst_capacity=*/2048,
      /*src_length_scale=*/2,
      /*cache_idx=*/0};
  std::vector<IpcBlock> blocks;

  append_ipc_block_checked(blocks, bounds, 0, 1024, 512);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_THROW(
      append_ipc_block_checked(blocks, bounds, 1, 0, 512),
      std::out_of_range);
  EXPECT_THROW(
      append_ipc_block_checked(blocks, bounds, 0, 1537, 512),
      std::out_of_range);
  EXPECT_EQ(blocks.size(), 1u);
}

TEST(SendStubTest, UseMockChannel) {
  if (env_cache_shape() != RAGGED_FLASH_CACHE_SHAPE) {
    return ;
  }
  uint32_t bs = 16 * KB;
  uint32_t ts = KB;
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};
  src_info.layer_num_blocks = 8;

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};
  dst_info.layer_num_blocks = 8;

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
  auto req0p = std::make_shared<RequestInfo>("1", 0, "req_id00000000000000000000000000", BlockIds{{0, 1, 2}}, BlockIds{{4, 5, 6}});
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
    // Prompt caching may produce overlapping ranges.
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
public:
  FTTC(int ch_id, bool* dead, bool* register_data_called, bool* send_data_called, bool* flush_called):
    ch_id_(ch_id),
    dead_(dead),
    register_data_called_(register_data_called),
    send_data_called_(send_data_called),
    flush_called_(flush_called) {}

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
};

// FaultTolerantTestChannelFactory
struct FTTCF: public IChannelFactory {
  int ch_id_ = 0;
  bool dead_ch_[2] = {false, false};
  bool register_data_called_[2] = {false, false};
  bool send_data_called_[2] = {false, false};
  bool flush_called_[2] = {false, false};
public:
  FTTCF() {}

  Channel create(const WorkerInfo& dst_info) override {
    auto ch_id = ch_id_++;
    EXPECT_TRUE(ch_id == 0 || ch_id == 1);
    return std::make_unique<FTTC>(ch_id,
      &dead_ch_[ch_id],
      &register_data_called_[ch_id],
      &send_data_called_[ch_id],
      &flush_called_[ch_id]);
  }
};


TEST(SendStubTest, FaultTolerantTest) {
  if (env_cache_shape() != RAGGED_FLASH_CACHE_SHAPE) {
    return ;
  }
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};
  // Fault-injection requests use synthetic block ids up to 22.
  src_info.layer_num_blocks = 32;

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};
  dst_info.layer_num_blocks = 32;

  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 32);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 32 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;

  FTTCF fttcf;
  auto naming = std::make_shared<FakeNamingWorkerClient>(2, dst_info);
  auto tx = std::make_unique<KvSendStub>("1", 0, src_info, 0, num_layers, std::make_unique<ProxyChannelFactory>(&fttcf), naming);

  auto req1p = std::make_shared<RequestInfo>("1", 0, "1", BlockIds{{10, 11, 12}}, BlockIds{{14, 15, 16}});
  auto& req1 = *req1p;

  auto req2p = std::make_shared<RequestInfo>("1", 0, "2", BlockIds{{17, 18, 19}}, BlockIds{{20, 21, 22}});
  auto& req2 = *req2p;

  auto req3p = std::make_shared<RequestInfo>("1", 0, "3", BlockIds{{0, 1, 2}}, BlockIds{{4, 5, 6}});
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
    EXPECT_FALSE(fttcf.dead_ch_[1]);
  }
  tx.reset();
  while (!fttcf.dead_ch_[1]) {
    usleep(10 * 1000);
  }
}


TEST(SendStubTest, CreateChannelFaultTolerantTest) {
  // Both QWEN3_NEXT_FLASH and QWEN3_NEXT_FLASHINFER require shape-specific
  // WorkerInfo fields (GDN state shapes, attn_kernel_blk_ntpb, ...) that
  // this generic fault-tolerance test does not set up.
  if (env_cache_shape() == QWEN3_NEXT_FLASH_CACHE_SHAPE ||
      env_cache_shape() == QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) {
    return ;
  }
  WorkerInfo src_info("0", 0);
  src_info.block_sizes = {bs};
  src_info.token_sizes = {ts};
  src_info.engine_tp_size = 1;
  // Fault-injection requests use synthetic block ids up to 22.
  src_info.layer_num_blocks = 32;

  WorkerInfo dst_info("1", 0);
  dst_info.block_sizes = {bs};
  dst_info.token_sizes = {ts};
  dst_info.engine_tp_size = 1;
  dst_info.layer_num_blocks = 32;

  Context ctx("0", 1);
  ctx.set_block_params({bs}, {ts}, 32);
  std::vector<std::vector<LayerInfo>> all_layer_infos = {
    {LayerInfo(ts, bs, 0)},
    {LayerInfo(ts, bs, 32 * bs)}
  };
  ctx.set_layer_info(0, all_layer_infos);
  uint32_t num_layers = 2;

  FTTCF fttcf;
  auto naming = std::make_shared<FakeNamingWorkerClient>(3, dst_info);
  auto tx = std::make_unique<KvSendStub>("1", 0, src_info, 0, num_layers, std::make_unique<ProxyChannelFactory>(&fttcf), naming);

  auto req1p = std::make_shared<RequestInfo>("1", 0, "1", BlockIds{{10, 11, 12}}, BlockIds{{14, 15, 16}});
  auto& req1 = *req1p;

  auto req2p = std::make_shared<RequestInfo>("1", 0, "2", BlockIds{{17, 18, 19}}, BlockIds{{20, 21, 22}});
  auto& req2 = *req2p;

  auto req3p = std::make_shared<RequestInfo>("1", 0, "3", BlockIds{{0, 1, 2}}, BlockIds{{4, 5, 6}});
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
  }
  tx.reset();
  while (!fttcf.dead_ch_[0]) {
    usleep(10 * 1000);
  }
}


TEST(ComputeValidRanksTest, AllRanksValidWhenKvHeadsGeqPtp) {
  // When d_tp <= num_kv_heads, use original logic
  auto vr = compute_valid_ranks_pd(4, 2, 8);  // p_tp=4, d_tp=2, num_kv_heads=8
  EXPECT_EQ(vr.count(), 4u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(vr[i]);
  }
}

// =============================================================================
// QWEN3_NEXT_FLASHINFER_CACHE_SHAPE tests
// =============================================================================

TEST(SendStubTest, Qwen3NextFlashinferPEqD) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  const int kv_heads = 4;
  const int head_dim = 64;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);
  const size_t token_size =
      static_cast<size_t>(2 * kv_heads * head_dim * dtype_size);
  const size_t block_size = token_size * ntpb;

  WorkerInfo src("0", 0);
  src.engine_tp_size = 2;
  src.worker_tp_rank = 0;
  src.num_kv_heads = kv_heads;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.token_sizes = {token_size};
  src.block_sizes = {block_size};
  src.layer_num_blocks = 16;

  WorkerInfo dst("1", 0);
  dst.engine_tp_size = 2;
  dst.worker_tp_rank = 0;
  dst.num_kv_heads = kv_heads;
  dst.num_gdn_layers = 1;
  dst.indexer_blk_ntpb = 0;
  dst.attn_kernel_blk_ntpb = ntpb;
  dst.token_sizes = {token_size};
  dst.block_sizes = {block_size};
  dst.layer_num_blocks = 16;

  // BlockIds: [gdn_layer_0, attn]
  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{6}, {0, 1}},
      BlockIds{{10}, {4, 5}});
  // Full first block (16 tokens) + reach_last_token to flush GDN.
  ReqSendTask task(req, 0, ntpb, true);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_eq_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);

  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);
  // Attn: 1 block * 4 heads * 2 (K + V) = 8 records (FlashInfer HND).
  // GDN: 1 layer * 1 block = 1 full-block record.
  ASSERT_EQ(blocks.size(), 9u);

  const size_t p_head_size = block_size / 2 / kv_heads;
  const size_t p_k_block_size = block_size / 2;
  const size_t per_token_per_head = p_head_size / ntpb;
  const size_t length = per_token_per_head * ntpb;

  // First 8 records are attn: pairs of (K, V) per head.
  for (int h = 0; h < kv_heads; ++h) {
    const auto& k = blocks[h * 2];
    const auto& v = blocks[h * 2 + 1];
    EXPECT_EQ(k.length, length);
    EXPECT_EQ(v.length, length);
    // src_block id = 0 -> p_blk_off = 0
    EXPECT_EQ(k.src_offset, static_cast<size_t>(h) * p_head_size);
    EXPECT_EQ(k.dst_offset, 4u * block_size + static_cast<size_t>(h) * p_head_size);
    EXPECT_EQ(v.src_offset, p_k_block_size);
    EXPECT_EQ(v.dst_offset, 4u * block_size + p_k_block_size);
  }

  // Last record is GDN block: src 6 -> dst 10, full block.
  const auto& gdn = blocks.back();
  EXPECT_EQ(gdn.src_offset, 6u * block_size);
  EXPECT_EQ(gdn.dst_offset, 10u * block_size);
  EXPECT_EQ(gdn.length, block_size);
}

TEST(SendStubTest, Qwen3NextFlashinferPGtD) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  const int kv_heads = 4;
  const int head_dim = 64;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);

  // P engine_tp_size=4; D engine_tp_size=2 -> GDN selects P>D.
  WorkerInfo src("0", 0);
  src.engine_tp_size = 4;
  src.worker_tp_rank = 0;
  src.num_kv_heads = kv_heads;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.hybrid_indexer_token_size = 0;
  // conv channel_dims sum must equal conv_state_shape[2] * gdn_conv_elem_size.
  src.conv_state_shape = {1, 3, 4};
  src.ssm_state_shape = {1, 1, 2, 2};
  src.gdn_conv_channel_dims = {2, 2};
  src.gdn_conv_elem_size = 1;
  src.gdn_ssm_elem_size = 1;
  // FlashInfer per-rank attn token: 2 * (kv_heads/p_kvt_tp) * head_dim * dtype.
  src.token_sizes = {static_cast<size_t>(
      2 * (kv_heads / static_cast<int>(src.engine_tp_size)) * head_dim * dtype_size)};
  src.block_sizes = {src.token_sizes[0] * ntpb};
  src.layer_num_blocks = 256;

  WorkerInfo dst("1", 0);
  dst.engine_tp_size = 2;
  dst.worker_tp_rank = 0;
  dst.num_kv_heads = kv_heads;
  dst.num_gdn_layers = 1;
  dst.indexer_blk_ntpb = 0;
  dst.attn_kernel_blk_ntpb = ntpb;
  dst.hybrid_indexer_token_size = 0;
  dst.conv_state_shape = src.conv_state_shape;
  dst.ssm_state_shape = src.ssm_state_shape;
  dst.gdn_conv_channel_dims = src.gdn_conv_channel_dims;
  dst.gdn_conv_elem_size = src.gdn_conv_elem_size;
  dst.gdn_ssm_elem_size = src.gdn_ssm_elem_size;
  dst.token_sizes = {static_cast<size_t>(
      2 * (kv_heads / static_cast<int>(dst.engine_tp_size)) * head_dim * dtype_size)};
  dst.block_sizes = {dst.token_sizes[0] * ntpb};
  dst.layer_num_blocks = 256;

  // BlockIds: [gdn_layer_0, attn]
  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {0, 1}},
      BlockIds{{200}, {10, 11}});
  // Reach last token to flush GDN; send 16 attn tokens.
  ReqSendTask task(req, 0, ntpb, true);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);

  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  // GDN P>D records:
  //   conv: conv_state_shape[1] * conv_channel_dims.size() = 3 * 2 = 6 records.
  //   ssm:  1 record.
  //   Total per gdn block = 7. With 1 gdn layer * 1 block = 7 records.
  // Attn records (FlashInfer HND): 1 block * (kv_heads/p_kvt_tp) heads * 2 KV
  //   = 1 * 1 * 2 = 2 records.
  // Order: GDN first, then indexer (none), then attn.
  ASSERT_EQ(blocks.size(), 7u + 2u);

  const size_t p_block_size = src.block_sizes[0];
  const size_t d_block_size = dst.block_sizes[0];
  const size_t p_token_size = src.token_sizes[0];
  const size_t d_token_size = dst.token_sizes[0];
  const int p_head_num = kv_heads / static_cast<int>(src.engine_tp_size);
  const size_t p_head_size = p_block_size / 2 / p_head_num;
  const size_t p_k_block_size = p_block_size / 2;
  const size_t d_k_block_size = d_block_size / 2;

  // attn_group_n = valid_ranks.count() / dst.engine_tp_size = 4 / 2 = 2.
  // attn_group_off = kvt_tp_rank % attn_group_n = 0 (worker_tp_rank=0 maps to slot 0).
  const uint32_t expected_attn_group_off = 0;

  // GDN records occupy [0, 7). Skip-validate full GDN length sum equals
  // expected raw conv+ssm bytes (3*(2+2)*1 + 1*1*2*2*1 = 12 + 4 = 16).
  size_t gdn_total = 0;
  for (size_t i = 0; i < 7; ++i) gdn_total += blocks[i].length;
  EXPECT_EQ(gdn_total, 16u);

  // Attn K + V (1 head per p rank, full block).
  const auto& k_rec = blocks[7];
  const auto& v_rec = blocks[8];
  const size_t length = (p_head_size / ntpb) * ntpb;
  EXPECT_EQ(k_rec.length, length);
  EXPECT_EQ(v_rec.length, length);
  // src_blocks attn is {0, 1}; first block id is 0 -> p_blk_off = 0.
  EXPECT_EQ(k_rec.src_offset, 0u);
  EXPECT_EQ(k_rec.dst_offset,
            10u * d_block_size + expected_attn_group_off * p_k_block_size + 0u);
  EXPECT_EQ(v_rec.src_offset, p_k_block_size);
  EXPECT_EQ(v_rec.dst_offset, 10u * d_block_size + d_k_block_size);
  // Suppress unused-warning when token sizes change in the future.
  (void)p_token_size; (void)d_token_size;
}

TEST(SendStubTest, Qwen3NextFlashinferPGtD_ReplicatedKvHeadsPartialBlock) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  // Qwen4 replicated-head case: total KV heads (2) is smaller than P TP (4),
  // so vLLM allocates one physical KV head on every P rank. P>D parsing must
  // use that per-rank head count rather than splitting the physical head into
  // two artificial half-width heads.
  const int total_kv_heads = 2;
  const int head_dim = 64;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);
  const int src_heads_per_rank = 1;
  const int dst_heads_per_rank = 1;

  WorkerInfo src("0", 0);
  src.engine_tp_size = 4;
  src.worker_tp_rank = 0;
  src.num_kv_heads = total_kv_heads;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.token_sizes = {static_cast<size_t>(
      2 * src_heads_per_rank * head_dim * dtype_size)};
  src.block_sizes = {src.token_sizes[0] * ntpb};
  src.layer_num_blocks = 256;

  WorkerInfo dst("1", 0);
  dst.engine_tp_size = 2;
  dst.worker_tp_rank = 0;
  dst.num_kv_heads = total_kv_heads;
  dst.num_gdn_layers = 1;
  dst.indexer_blk_ntpb = 0;
  dst.attn_kernel_blk_ntpb = ntpb;
  dst.token_sizes = {static_cast<size_t>(
      2 * dst_heads_per_rank * head_dim * dtype_size)};
  dst.block_sizes = {dst.token_sizes[0] * ntpb};
  dst.layer_num_blocks = 256;

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {0}},
      BlockIds{{200}, {10}});
  constexpr uint32_t partial_tokens = 5;
  ReqSendTask task(
      req, 0, partial_tokens, /*reach_last_token=*/false);

  auto valid_ranks = compute_valid_ranks_pd(
      src.engine_tp_size, dst.engine_tp_size, total_kv_heads);
  ASSERT_TRUE(valid_ranks[0]);
  ASSERT_TRUE(valid_ranks[2]);
  ASSERT_EQ(valid_ranks.count(), 2u);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);

  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);
  ASSERT_EQ(blocks.size(), 2u);  // one physical head, K and V

  const size_t head_dim_bytes = head_dim * dtype_size;
  const size_t expected_length = partial_tokens * head_dim_bytes;
  const size_t src_kv_section = src.block_sizes[0] / 2;
  const size_t dst_kv_section = dst.block_sizes[0] / 2;
  const size_t dst_block_offset = 10u * dst.block_sizes[0];

  EXPECT_EQ(blocks[0].src_offset, 0u);
  EXPECT_EQ(blocks[0].dst_offset, dst_block_offset);
  EXPECT_EQ(blocks[0].length, expected_length);
  EXPECT_EQ(blocks[1].src_offset, src_kv_section);
  EXPECT_EQ(blocks[1].dst_offset, dst_block_offset + dst_kv_section);
  EXPECT_EQ(blocks[1].length, expected_length);
}

// =============================================================================
// Fill-last-decode-block tests (qwen3_next P>D, token-granularity attn).
// BLLM_KVTRANS_PAD_LAST_ATTN_BLOCK defaults to enabled, so the parse functions
// append fill records that pad the unfilled tail of the last decode block with
// the request's block 0 data.
// =============================================================================

namespace {
// Build the qwen3_next FLASH P>D src/dst WorkerInfo used by the fill tests.
void make_qwen3_next_flash_pgtd_workers(WorkerInfo& src, WorkerInfo& dst) {
  const int kv_heads = 8;
  const int head_dim = 128;
  const int ntpb = 16;

  src.engine_tp_size = 4;
  src.worker_tp_rank = 0;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.hybrid_indexer_token_size = 0;
  src.num_kv_heads = kv_heads;
  src.conv_state_shape = {1, 3, 4};
  src.ssm_state_shape = {1, 1, 2, 2};
  src.gdn_conv_channel_dims = {2, 2};
  src.gdn_conv_elem_size = 1;
  src.gdn_ssm_elem_size = 1;
  src.token_sizes = {size_t(2 * (kv_heads / src.engine_tp_size) * head_dim * sizeof(uint16_t))};
  src.block_sizes = {src.token_sizes[0] * ntpb};
  // Tests below deliberately use sparse synthetic block ids up to 101.
  // Keep WorkerInfo capacity consistent now that parse_block validates each
  // emitted range before appending it.
  src.layer_num_blocks = 256;

  dst.engine_tp_size = 2;
  dst.worker_tp_rank = 0;
  dst.num_gdn_layers = 1;
  dst.indexer_blk_ntpb = 0;
  dst.attn_kernel_blk_ntpb = ntpb;
  dst.hybrid_indexer_token_size = 0;
  dst.num_kv_heads = kv_heads;
  dst.conv_state_shape = src.conv_state_shape;
  dst.ssm_state_shape = src.ssm_state_shape;
  dst.gdn_conv_channel_dims = src.gdn_conv_channel_dims;
  dst.gdn_conv_elem_size = src.gdn_conv_elem_size;
  dst.gdn_ssm_elem_size = src.gdn_ssm_elem_size;
  dst.token_sizes = {size_t(2 * (kv_heads / dst.engine_tp_size) * head_dim * sizeof(uint16_t))};
  dst.block_sizes = {dst.token_sizes[0] * ntpb};
  // Destination fixtures use sparse synthetic block ids up to 201.
  dst.layer_num_blocks = 256;
}
}  // namespace


TEST(SendStubTest, Qwen3NextFlashPGtD_IndexerRepresentativePerDRank) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  auto run_for_rank = [](uint32_t p_rank) {
    WorkerInfo src("0", 0);
    WorkerInfo dst("1", 0);
    make_qwen3_next_flash_pgtd_workers(src, dst);
    src.worker_tp_rank = p_rank;
    dst.worker_tp_rank = p_rank / 2;
    src.indexer_blk_ntpb = 4;
    dst.indexer_blk_ntpb = 8;
    src.hybrid_indexer_token_size = 32;
    dst.hybrid_indexer_token_size = 32;

    auto valid_ranks = compute_valid_ranks_pd(
        src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
    uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
    uint32_t eff_tp_rank = static_cast<uint32_t>(
        (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

    auto req = std::make_shared<RequestInfo>(
        "1", 0, "req0",
        BlockIds{{100}, {30, 31}, {0}},
        BlockIds{{200}, {40}, {10}});
    ReqSendTask task(req, 1, 5, /*reach_last_token=*/false);

    std::vector<std::vector<IpcBlock>> send_blocks;
    vllm_parse_hybrid_block_send_p_gt_d(
        &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
    return std::make_tuple(src, dst, send_blocks);
  };

  // P ranks 0/1 map to D rank 0, P ranks 2/3 map to D rank 1. The first
  // rank in each subgroup must send the replicated indexer cache.
  auto [rep_src, rep_dst, rep_send_blocks] = run_for_rank(2);
  ASSERT_EQ(rep_send_blocks.size(), 1u);
  const auto& rep_blocks = rep_send_blocks.at(0);
  ASSERT_GE(rep_blocks.size(), 2u);

  const size_t p_block_size = rep_src.block_sizes[0];
  const size_t d_block_size = rep_dst.block_sizes[0];
  EXPECT_EQ(rep_blocks[0].src_offset, 30u * p_block_size + 1u * 32u);
  EXPECT_EQ(rep_blocks[0].dst_offset, 40u * d_block_size + 1u * 32u);
  EXPECT_EQ(rep_blocks[0].length, 3u * 32u);
  EXPECT_EQ(rep_blocks[1].src_offset, 31u * p_block_size);
  EXPECT_EQ(rep_blocks[1].dst_offset, 40u * d_block_size + 4u * 32u);
  EXPECT_EQ(rep_blocks[1].length, 2u * 32u);

  auto [nonrep_src, nonrep_dst, nonrep_send_blocks] = run_for_rank(1);
  ASSERT_EQ(nonrep_send_blocks.size(), 1u);
  const auto& nonrep_blocks = nonrep_send_blocks.at(0);
  EXPECT_EQ(nonrep_blocks.size() + 2u, rep_blocks.size());
  const size_t nonrep_p_block_size = nonrep_src.block_sizes[0];
  const size_t nonrep_d_block_size = nonrep_dst.block_sizes[0];
  for (const auto& block : nonrep_blocks) {
    EXPECT_FALSE(block.src_offset == 30u * nonrep_p_block_size + 1u * 32u &&
                 block.dst_offset == 40u * nonrep_d_block_size + 1u * 32u &&
                 block.length == 3u * 32u);
  }
}

TEST(SendStubTest, Qwen3NextFlashPGtD_FillLastBlock) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flash_pgtd_workers(src, dst);

  const size_t p_block_size = src.block_sizes[0];
  const size_t d_block_size = dst.block_sizes[0];
  const size_t p_token_step = src.token_sizes[0] / 2;
  const size_t d_token_step = dst.token_sizes[0] / 2;
  const uint32_t dst_ntpb = static_cast<uint32_t>(d_block_size / dst.token_sizes[0]);  // 16

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  // BlockIds: [gdn_layer_0 (2 blocks), attn (3 blocks)].
  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100, 101}, {0, 1, 2}},
      BlockIds{{200, 201}, {10, 11, 12}});
  // total = 5 + 18 = 23; 23 % 16 = 7 filled, 9 to fill in the last dst block.
  ReqSendTask task(req, 5, 18, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_hybrid_block_send_p_gt_d(&src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  // GDN: 2 blocks * (3*2 conv + 1 ssm) = 14 records.
  // Attn: 18 tokens * 2 (K,V) = 36 records.
  // Fill: 9 tokens * 2 (K,V) = 18 records.
  const uint32_t total = 23;
  const uint32_t filled = total % dst_ntpb;  // 7
  const uint32_t to_fill = dst_ntpb - filled; // 9
  ASSERT_EQ(blocks.size(), 14u + 36u + to_fill * 2u);

  // The fill records are the suffix; block 0 is all zeros, so every fill packet
  // (both K and V) sources from the same zero region at block 0 start (offset 0).
  // The first fill token zeroes the last dst block (id 11) at token slot `filled`.
  const size_t fill_start = blocks.size() - to_fill * 2u;
  const auto& fk = blocks[fill_start];      // K
  const auto& fv = blocks[fill_start + 1];  // V
  const size_t last_d_blk_off = 11u * d_block_size;
  const size_t d_kernel_half = (dst.attn_kernel_blk_ntpb * dst.token_sizes[0]) / 2;
  EXPECT_EQ(fk.src_offset, 0u);
  EXPECT_EQ(fk.dst_offset, last_d_blk_off + filled * d_token_step);
  EXPECT_EQ(fk.length, p_token_step);
  EXPECT_EQ(fv.src_offset, 0u);
  EXPECT_EQ(fv.dst_offset, last_d_blk_off + filled * d_token_step + d_kernel_half);
  EXPECT_EQ(fv.length, p_token_step);
}

TEST(SendStubTest, Qwen3NextFlashPGtD_NoFillWhenAligned) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flash_pgtd_workers(src, dst);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100, 101}, {0, 1, 2}},
      BlockIds{{200, 201}, {10, 11, 12}});
  // total = 0 + 16 = 16, exactly one full dst block -> no fill.
  ReqSendTask task(req, 0, 16, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_hybrid_block_send_p_gt_d(&src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  // GDN 14 + attn (16 tokens * 2) = 32; no fill.
  EXPECT_EQ(send_blocks.at(0).size(), 14u + 32u);
}

TEST(SendStubTest, Qwen3NextFlashPGtD_NoFillWhenNotLastToken) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flash_pgtd_workers(src, dst);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100, 101}, {0, 1, 2}},
      BlockIds{{200, 201}, {10, 11, 12}});
  // Partial last block but reach_last_token=false -> no GDN and no fill.
  ReqSendTask task(req, 5, 18, /*reach_last_token=*/false);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_hybrid_block_send_p_gt_d(&src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  // Attn only: 18 tokens * 2 = 36 records.
  EXPECT_EQ(send_blocks.at(0).size(), 36u);
}

namespace {
// Build the qwen3_next FLASHINFER P>D src/dst WorkerInfo used by fill tests.
void make_qwen3_next_flashinfer_pgtd_workers(WorkerInfo& src, WorkerInfo& dst) {
  const int kv_heads = 4;
  const int head_dim = 64;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);

  src.engine_tp_size = 4;
  src.worker_tp_rank = 0;
  src.num_kv_heads = kv_heads;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.hybrid_indexer_token_size = 0;
  src.conv_state_shape = {1, 3, 4};
  src.ssm_state_shape = {1, 1, 2, 2};
  src.gdn_conv_channel_dims = {2, 2};
  src.gdn_conv_elem_size = 1;
  src.gdn_ssm_elem_size = 1;
  src.token_sizes = {static_cast<size_t>(
      2 * (kv_heads / static_cast<int>(src.engine_tp_size)) * head_dim * dtype_size)};
  src.block_sizes = {src.token_sizes[0] * ntpb};
  src.layer_num_blocks = 256;

  dst.engine_tp_size = 2;
  dst.worker_tp_rank = 0;
  dst.num_kv_heads = kv_heads;
  dst.num_gdn_layers = 1;
  dst.indexer_blk_ntpb = 0;
  dst.attn_kernel_blk_ntpb = ntpb;
  dst.hybrid_indexer_token_size = 0;
  dst.conv_state_shape = src.conv_state_shape;
  dst.ssm_state_shape = src.ssm_state_shape;
  dst.gdn_conv_channel_dims = src.gdn_conv_channel_dims;
  dst.gdn_conv_elem_size = src.gdn_conv_elem_size;
  dst.gdn_ssm_elem_size = src.gdn_ssm_elem_size;
  dst.token_sizes = {static_cast<size_t>(
      2 * (kv_heads / static_cast<int>(dst.engine_tp_size)) * head_dim * dtype_size)};
  dst.block_sizes = {dst.token_sizes[0] * ntpb};
  dst.layer_num_blocks = 256;
}
}  // namespace


TEST(SendStubTest, Qwen3NextFlashinferPGtD_IndexerRepresentativePerDRank) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  auto run_for_rank = [](uint32_t p_rank) {
    WorkerInfo src("0", 0);
    WorkerInfo dst("1", 0);
    make_qwen3_next_flashinfer_pgtd_workers(src, dst);
    src.worker_tp_rank = p_rank;
    dst.worker_tp_rank = p_rank / 2;
    src.indexer_blk_ntpb = 4;
    dst.indexer_blk_ntpb = 8;
    src.hybrid_indexer_token_size = 32;
    dst.hybrid_indexer_token_size = 32;

    auto valid_ranks = compute_valid_ranks_pd(
        src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
    uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
    uint32_t eff_tp_rank = static_cast<uint32_t>(
        (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

    auto req = std::make_shared<RequestInfo>(
        "1", 0, "req0",
        BlockIds{{100}, {30, 31}, {0}},
        BlockIds{{200}, {40}, {10}});
    ReqSendTask task(req, 1, 5, /*reach_last_token=*/false);

    std::vector<std::vector<IpcBlock>> send_blocks;
    vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
        &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
    return std::make_tuple(src, dst, send_blocks);
  };

  auto [rep_src, rep_dst, rep_send_blocks] = run_for_rank(2);
  ASSERT_EQ(rep_send_blocks.size(), 1u);
  const auto& rep_blocks = rep_send_blocks.at(0);
  ASSERT_GE(rep_blocks.size(), 2u);

  const size_t p_block_size = rep_src.block_sizes[0];
  const size_t d_block_size = rep_dst.block_sizes[0];
  EXPECT_EQ(rep_blocks[0].src_offset, 30u * p_block_size + 1u * 32u);
  EXPECT_EQ(rep_blocks[0].dst_offset, 40u * d_block_size + 1u * 32u);
  EXPECT_EQ(rep_blocks[0].length, 3u * 32u);
  EXPECT_EQ(rep_blocks[1].src_offset, 31u * p_block_size);
  EXPECT_EQ(rep_blocks[1].dst_offset, 40u * d_block_size + 4u * 32u);
  EXPECT_EQ(rep_blocks[1].length, 2u * 32u);

  auto [nonrep_src, nonrep_dst, nonrep_send_blocks] = run_for_rank(1);
  ASSERT_EQ(nonrep_send_blocks.size(), 1u);
  const auto& nonrep_blocks = nonrep_send_blocks.at(0);
  EXPECT_EQ(nonrep_blocks.size() + 2u, rep_blocks.size());
  const size_t nonrep_p_block_size = nonrep_src.block_sizes[0];
  const size_t nonrep_d_block_size = nonrep_dst.block_sizes[0];
  for (const auto& block : nonrep_blocks) {
    EXPECT_FALSE(block.src_offset == 30u * nonrep_p_block_size + 1u * 32u &&
                 block.dst_offset == 40u * nonrep_d_block_size + 1u * 32u &&
                 block.length == 3u * 32u);
  }
}

TEST(SendStubTest, Qwen3NextFlashinferPGtD_FillLastBlock) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flashinfer_pgtd_workers(src, dst);

  const size_t p_block_size = src.block_sizes[0];
  const size_t d_block_size = dst.block_sizes[0];
  const size_t p_token_size = src.token_sizes[0];
  const size_t d_token_size = dst.token_sizes[0];
  const uint32_t ntpb = static_cast<uint32_t>(p_block_size / p_token_size);  // 16
  const int src_heads_per_rank = src.num_kv_heads / static_cast<int>(src.engine_tp_size);
  const size_t head_dim_size = p_token_size / 2 / static_cast<size_t>(src_heads_per_rank);
  const size_t d_kv_section = (ntpb * d_token_size) / 2;

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  // BlockIds: [gdn_layer_0 (1 block), attn (2 blocks)].
  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {0, 1}},
      BlockIds{{200}, {10, 11}});
  // total = 0 + 20 = 20; 20 % 16 = 4 filled, 12 to fill in last dst block (11).
  ReqSendTask task(req, 0, 20, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  // GDN: 1 block * (3*2 conv + 1 ssm) = 7 records.
  // Attn (HND): chunk[0..15] + chunk[16..19] -> 2 chunks * per-rank heads * 2.
  // Fill: single chunk * per-rank heads * 2.
  const uint32_t total = 20;
  const uint32_t dst_ntpb = static_cast<uint32_t>(d_block_size / d_token_size);  // 16
  const uint32_t filled = total % dst_ntpb;     // 4
  const uint32_t to_fill = dst_ntpb - filled;   // 12
  const size_t fill_records = static_cast<size_t>(src_heads_per_rank) * 2u;  // single chunk
  ASSERT_EQ(blocks.size(),
            7u + static_cast<size_t>(src_heads_per_rank) * 2u * 2u + fill_records);

  // First fill record: head 0, writing zero data from block 0 into the last
  // dst block (id 11) at token slot `filled`, length = to_fill * head_dim_size.
  // Block 0 is all zeros, so both K and V source from the same zero region
  // (block 0 start, offset 0).
  const size_t fill_start = blocks.size() - fill_records;
  const auto& fk = blocks[fill_start];      // head0 K
  const auto& fv = blocks[fill_start + 1];  // head0 V
  const size_t last_d_blk_off = 11u * d_block_size;
  const size_t length = static_cast<size_t>(to_fill) * head_dim_size;
  EXPECT_EQ(fk.src_offset, 0u);
  EXPECT_EQ(fk.dst_offset, last_d_blk_off + filled * head_dim_size);
  EXPECT_EQ(fk.length, length);
  EXPECT_EQ(fv.src_offset, 0u);
  EXPECT_EQ(fv.dst_offset, last_d_blk_off + filled * head_dim_size + d_kv_section);
  EXPECT_EQ(fv.length, length);
}

TEST(SendStubTest, Qwen3NextFlashinferPGtD_NoFillWhenAligned) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flashinfer_pgtd_workers(src, dst);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {0, 1}},
      BlockIds{{200}, {10, 11}});
  // total = 0 + 16 = 16, exactly one full dst block -> no fill.
  ReqSendTask task(req, 0, 16, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const int src_heads_per_rank = src.num_kv_heads / static_cast<int>(src.engine_tp_size);
  // GDN 7 + attn (1 chunk * per-rank heads * 2); no fill.
  EXPECT_EQ(send_blocks.at(0).size(),
            7u + static_cast<size_t>(src_heads_per_rank) * 2u);
}

TEST(SendStubTest, Qwen3NextFlashinferPGtD_NoFillWhenNotLastToken) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flashinfer_pgtd_workers(src, dst);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {0, 1}},
      BlockIds{{200}, {10, 11}});
  // Partial last block but reach_last_token=false -> no GDN and no fill.
  ReqSendTask task(req, 0, 20, /*reach_last_token=*/false);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_gt_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const int src_heads_per_rank = src.num_kv_heads / static_cast<int>(src.engine_tp_size);
  // Attn only: 2 chunks * per-rank heads * 2 records.
  EXPECT_EQ(send_blocks.at(0).size(),
            static_cast<size_t>(src_heads_per_rank) * 2u * 2u);
}

TEST(SendStubTest, Qwen3NextFlashinferPEqDChunkedNoFlush) {
  // Verify chunked-prefill mid-block does NOT emit GDN (reach_last=false).
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  const int kv_heads = 4;
  const int head_dim = 64;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);
  const size_t token_size =
      static_cast<size_t>(2 * kv_heads * head_dim * dtype_size);
  const size_t block_size = token_size * ntpb;

  WorkerInfo src("0", 0);
  src.engine_tp_size = 2;
  src.worker_tp_rank = 0;
  src.num_kv_heads = kv_heads;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.token_sizes = {token_size};
  src.block_sizes = {block_size};
  src.layer_num_blocks = 16;

  WorkerInfo dst = src;
  dst.inst_id = "1";

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{6}, {0, 1}},
      BlockIds{{10}, {4, 5}});
  ReqSendTask task(req, 0, ntpb, /*reach_last_token=*/false);

  auto valid_ranks = compute_valid_ranks_pd(src.engine_tp_size, dst.engine_tp_size, kv_heads);
  uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
  uint32_t eff_tp_rank = static_cast<uint32_t>(
      (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_eq_d(
      &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank, &task, send_blocks);

  ASSERT_EQ(send_blocks.size(), 1u);
  // Only attn (4 heads * 2 = 8) records, no GDN.
  EXPECT_EQ(send_blocks.at(0).size(), 8u);
}

// =============================================================================
// Qwen3-next P<D tests (replicated / mixed p_tp < heads < d_tp regimes).
// =============================================================================

namespace {
// Build qwen3_next FLASH P<D src/dst WorkerInfo.
// p_tp=2, d_tp=8. kv_heads selects the regime:
//   kv_heads=2  -> replicated (heads <= p_tp), P/D rows identical;
//   kv_heads=4  -> mixed (p_tp < heads < d_tp), P row = 2x D row.
void make_qwen3_next_flash_pltd_workers(WorkerInfo& src, WorkerInfo& dst,
                                        int kv_heads) {
  const int head_dim = 128;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);

  src.engine_tp_size = 2;
  src.worker_tp_rank = 0;
  src.num_gdn_layers = 1;
  src.indexer_blk_ntpb = 0;
  src.attn_kernel_blk_ntpb = ntpb;
  src.hybrid_indexer_token_size = 0;
  src.num_kv_heads = kv_heads;
  src.conv_state_shape = {1, 3, 8};
  src.ssm_state_shape = {1, 1, 2, 4};
  src.gdn_conv_channel_dims = {4, 4};
  src.gdn_conv_elem_size = 1;
  src.gdn_ssm_elem_size = 1;
  const int p_heads_per_rank =
      std::max(1, kv_heads / static_cast<int>(src.engine_tp_size));
  src.token_sizes = {static_cast<size_t>(
      2 * p_heads_per_rank * head_dim * dtype_size)};
  src.block_sizes = {src.token_sizes[0] * ntpb};
  // P<D fixtures use block 100 for GDN and block 1 for attention.
  src.layer_num_blocks = 256;

  dst.engine_tp_size = 8;
  dst.worker_tp_rank = 0;
  dst.num_gdn_layers = 1;
  dst.indexer_blk_ntpb = 0;
  dst.attn_kernel_blk_ntpb = ntpb;
  dst.hybrid_indexer_token_size = 0;
  dst.num_kv_heads = kv_heads;
  dst.conv_state_shape = src.conv_state_shape;
  dst.ssm_state_shape = src.ssm_state_shape;
  dst.gdn_conv_channel_dims = src.gdn_conv_channel_dims;
  dst.gdn_conv_elem_size = src.gdn_conv_elem_size;
  dst.gdn_ssm_elem_size = src.gdn_ssm_elem_size;
  const int d_heads_per_rank =
      std::max(1, kv_heads / static_cast<int>(dst.engine_tp_size));
  dst.token_sizes = {static_cast<size_t>(
      2 * d_heads_per_rank * head_dim * dtype_size)};
  dst.block_sizes = {dst.token_sizes[0] * ntpb};
  // Destination fixtures use block 200 for GDN and block 10 for attention.
  dst.layer_num_blocks = 256;
}

// Same TP config for the FLASHINFER shape (head_dim=64).
void make_qwen3_next_flashinfer_pltd_workers(WorkerInfo& src, WorkerInfo& dst,
                                             int kv_heads) {
  make_qwen3_next_flash_pltd_workers(src, dst, kv_heads);
  const int head_dim = 64;
  const int ntpb = 16;
  const size_t dtype_size = sizeof(uint16_t);
  const int p_heads_per_rank =
      std::max(1, kv_heads / static_cast<int>(src.engine_tp_size));
  const int d_heads_per_rank =
      std::max(1, kv_heads / static_cast<int>(dst.engine_tp_size));
  src.token_sizes = {static_cast<size_t>(
      2 * p_heads_per_rank * head_dim * dtype_size)};
  src.block_sizes = {src.token_sizes[0] * ntpb};
  dst.token_sizes = {static_cast<size_t>(
      2 * d_heads_per_rank * head_dim * dtype_size)};
  dst.block_sizes = {dst.token_sizes[0] * ntpb};
}
}  // namespace

TEST(SendStubTest, Qwen3NextFlashPLtD_ReplicatedFullBlockCopy) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  // heads=2 <= p_tp=2: replicated regime, identical rows on both sides.
  make_qwen3_next_flash_pltd_workers(src, dst, /*kv_heads=*/2);
  dst.worker_tp_rank = 1;  // fan-out target 1 of src rank 0

  auto valid_ranks = compute_valid_ranks_pd(
      src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
  EXPECT_EQ(valid_ranks.count(), 2u);

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {1}},
      BlockIds{{200}, {10}});
  // Exactly one full block: aligned copy emits one full-block record.
  ReqSendTask task(req, 0, 16, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_hybrid_block_send_p_lt_d(
      &src, &dst, valid_ranks, 2, 0, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  const size_t p_block_size = src.block_sizes[0];
  // Attn: 1 full-block record. GDN: 1 block * (3*2 conv + 1 ssm) = 7 records.
  ASSERT_EQ(blocks.size(), 1u + 7u);
  EXPECT_EQ(blocks[0].src_offset, 1u * p_block_size);
  EXPECT_EQ(blocks[0].dst_offset, 10u * p_block_size);
  EXPECT_EQ(blocks[0].length, p_block_size);
}

TEST(SendStubTest, Qwen3NextFlashPLtD_MixedHeadSlice) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  // p_tp=2, heads=4, d_tp=8: each P rank holds 2 heads, each head is
  // replicated on 2 D ranks. D ranks 0/1 get head 0 of P rank 0; D ranks 2/3
  // get head 1.
  auto run_for_dst_rank = [](uint32_t d_rank) {
    WorkerInfo src("0", 0);
    WorkerInfo dst("1", 0);
    make_qwen3_next_flash_pltd_workers(src, dst, /*kv_heads=*/4);
    dst.worker_tp_rank = d_rank;

    auto valid_ranks = compute_valid_ranks_pd(
        src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
    EXPECT_EQ(valid_ranks.count(), 2u);

    auto req = std::make_shared<RequestInfo>(
        "1", 0, "req0",
        BlockIds{{100}, {1}},
        BlockIds{{200}, {10}});
    ReqSendTask task(req, 0, 5, /*reach_last_token=*/false);

    std::vector<std::vector<IpcBlock>> send_blocks;
    vllm_parse_hybrid_block_send_p_lt_d(
        &src, &dst, valid_ranks, 2, 0, &task, send_blocks);
    return std::make_tuple(src, dst, send_blocks);
  };

  auto [src, dst, send_blocks] = run_for_dst_rank(2);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  const size_t p_block_size = src.block_sizes[0];   // 16 * 1024
  const size_t d_block_size = dst.block_sizes[0];   // 16 * 512
  const size_t p_token_step = src.token_sizes[0] / 2;  // 512
  const size_t d_token_step = dst.token_sizes[0] / 2;  // 256
  const size_t p_kernel_half = p_block_size / 2;
  const size_t d_kernel_half = d_block_size / 2;

  // Attn only (not last token): 5 tokens * 2 (K,V) = 10 records.
  ASSERT_EQ(blocks.size(), 10u);
  // D rank 2 -> attn_group_off = 1 (head 1 of P rank 0). Per token t:
  //   K: src = blk1 * p_bs + t * p_step + 1 * d_step, dst = blk10 * d_bs + t * d_step
  //   V: src += p_kernel/2, dst += d_kernel/2
  for (uint32_t t = 0; t < 5; ++t) {
    const auto& k = blocks[t * 2];
    const auto& v = blocks[t * 2 + 1];
    EXPECT_EQ(k.src_offset,
              1u * p_block_size + t * p_token_step + 1u * d_token_step);
    EXPECT_EQ(k.dst_offset, 10u * d_block_size + t * d_token_step);
    EXPECT_EQ(k.length, d_token_step);
    EXPECT_EQ(v.src_offset, k.src_offset + p_kernel_half);
    EXPECT_EQ(v.dst_offset, k.dst_offset + d_kernel_half);
    EXPECT_EQ(v.length, d_token_step);
  }

  // D rank 1 shares head 0 with D rank 0: attn_group_off = 0.
  auto [src0, dst0, send_blocks0] = run_for_dst_rank(1);
  const auto& blocks0 = send_blocks0.at(0);
  ASSERT_EQ(blocks0.size(), 10u);
  EXPECT_EQ(blocks0[0].src_offset, 1u * src0.block_sizes[0]);
  EXPECT_EQ(blocks0[0].dst_offset, 10u * dst0.block_sizes[0]);
}

TEST(SendStubTest, Qwen3NextFlashPLtD_MixedFillTail) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flash_pltd_workers(src, dst, /*kv_heads=*/4);
  dst.worker_tp_rank = 2;

  auto valid_ranks = compute_valid_ranks_pd(
      src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {1}},
      BlockIds{{200}, {10}});
  // total = 5, dst ntpb = 16: fill 11 tail tokens of dst block 10.
  ReqSendTask task(req, 0, 5, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_hybrid_block_send_p_lt_d(
      &src, &dst, valid_ranks, 2, 0, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  const size_t p_block_size = src.block_sizes[0];
  const size_t d_block_size = dst.block_sizes[0];
  const size_t d_token_step = dst.token_sizes[0] / 2;
  const size_t d_kernel_half = d_block_size / 2;

  // Attn 5 tokens * 2 + fill (K seg + V seg) + GDN 7 records.
  ASSERT_EQ(blocks.size(), 10u + 2u + 7u);
  const auto& fk = blocks[10];
  const auto& fv = blocks[11];
  const size_t fill_len = 11u * d_token_step;
  EXPECT_EQ(fk.src_offset, 1u * p_block_size);  // src attn block 0 (zeros)
  EXPECT_EQ(fk.dst_offset, 10u * d_block_size + 5u * d_token_step);
  EXPECT_EQ(fk.length, fill_len);
  EXPECT_EQ(fv.src_offset, 1u * p_block_size);
  EXPECT_EQ(fv.dst_offset, 10u * d_block_size + 5u * d_token_step + d_kernel_half);
  EXPECT_EQ(fv.length, fill_len);

  // GDN slice: gdn_group_n=4, gdn_group_off=2 for D rank 2.
  const auto& gdn_first = blocks[12];
  const size_t d_conv_step = 4 / 4;  // per-channel slice: channel dim / gdn_group_n
  EXPECT_EQ(gdn_first.src_offset, 100u * p_block_size + 2u * d_conv_step);
  EXPECT_EQ(gdn_first.dst_offset, 200u * d_block_size);
  EXPECT_EQ(gdn_first.length, d_conv_step);
}

TEST(SendStubTest, Qwen3NextFlashinferPLtD_MixedHeadSlice) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  auto run_for_dst_rank = [](uint32_t d_rank) {
    WorkerInfo src("0", 0);
    WorkerInfo dst("1", 0);
    make_qwen3_next_flashinfer_pltd_workers(src, dst, /*kv_heads=*/4);
    dst.worker_tp_rank = d_rank;

    auto valid_ranks = compute_valid_ranks_pd(
        src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
    EXPECT_EQ(valid_ranks.count(), 2u);

    auto req = std::make_shared<RequestInfo>(
        "1", 0, "req0",
        BlockIds{{100}, {1}},
        BlockIds{{200}, {10}});
    ReqSendTask task(req, 0, 5, /*reach_last_token=*/false);

    std::vector<std::vector<IpcBlock>> send_blocks;
    vllm_parse_qwen3_next_flashinfer_block_send_p_lt_d(
        &src, &dst, valid_ranks, 2, 0, &task, send_blocks);
    return std::make_tuple(src, dst, send_blocks);
  };

  auto [src, dst, send_blocks] = run_for_dst_rank(2);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  const size_t p_block_size = src.block_sizes[0];   // 16 * 512
  const size_t d_block_size = dst.block_sizes[0];   // 16 * 256
  const size_t head_dim_size = dst.token_sizes[0] / 2;  // 128 (1 head)
  const size_t p_head_section = 16u * head_dim_size;    // 2048
  const size_t p_kv_section = p_block_size / 2;
  const size_t d_kv_section = d_block_size / 2;

  // HND: one chunk (5 tokens), 1 dst head -> K + V = 2 records.
  ASSERT_EQ(blocks.size(), 2u);
  // D rank 2 -> attn_group_off = 1: src reads head section 1 of P rank 0.
  const auto& k = blocks[0];
  const auto& v = blocks[1];
  EXPECT_EQ(k.src_offset, 1u * p_block_size + 1u * p_head_section);
  EXPECT_EQ(k.dst_offset, 10u * d_block_size);
  EXPECT_EQ(k.length, 5u * head_dim_size);
  EXPECT_EQ(v.src_offset, k.src_offset + p_kv_section);
  EXPECT_EQ(v.dst_offset, k.dst_offset + d_kv_section);
  EXPECT_EQ(v.length, 5u * head_dim_size);

  // D rank 1 shares head 0: attn_group_off = 0.
  auto [src0, dst0, send_blocks0] = run_for_dst_rank(1);
  const auto& blocks0 = send_blocks0.at(0);
  ASSERT_EQ(blocks0.size(), 2u);
  EXPECT_EQ(blocks0[0].src_offset, 1u * src0.block_sizes[0]);
}

TEST(SendStubTest, Qwen3NextFlashinferPLtD_MixedFillTail) {
  if (env_cache_shape() != QWEN3_NEXT_FLASHINFER_CACHE_SHAPE) return;

  WorkerInfo src("0", 0);
  WorkerInfo dst("1", 0);
  make_qwen3_next_flashinfer_pltd_workers(src, dst, /*kv_heads=*/4);
  dst.worker_tp_rank = 2;

  auto valid_ranks = compute_valid_ranks_pd(
      src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);

  auto req = std::make_shared<RequestInfo>(
      "1", 0, "req0",
      BlockIds{{100}, {1}},
      BlockIds{{200}, {10}});
  // total = 5, dst ntpb = 16: fill 11 tail tokens of dst block 10.
  ReqSendTask task(req, 0, 5, /*reach_last_token=*/true);

  std::vector<std::vector<IpcBlock>> send_blocks;
  vllm_parse_qwen3_next_flashinfer_block_send_p_lt_d(
      &src, &dst, valid_ranks, 2, 0, &task, send_blocks);
  ASSERT_EQ(send_blocks.size(), 1u);
  const auto& blocks = send_blocks.at(0);

  const size_t p_block_size = src.block_sizes[0];
  const size_t d_block_size = dst.block_sizes[0];
  const size_t head_dim_size = dst.token_sizes[0] / 2;
  const size_t d_kv_section = d_block_size / 2;

  // Attn (K+V) + fill (K seg + V seg) + GDN 7 records.
  ASSERT_EQ(blocks.size(), 2u + 2u + 7u);
  const auto& fk = blocks[2];
  const auto& fv = blocks[3];
  const size_t fill_len = 11u * head_dim_size;
  EXPECT_EQ(fk.src_offset, 1u * p_block_size);  // src attn block 0 (zeros)
  EXPECT_EQ(fk.dst_offset, 10u * d_block_size + 5u * head_dim_size);
  EXPECT_EQ(fk.length, fill_len);
  EXPECT_EQ(fv.src_offset, 1u * p_block_size);
  EXPECT_EQ(fv.dst_offset, 10u * d_block_size + 5u * head_dim_size + d_kv_section);
  EXPECT_EQ(fv.length, fill_len);
}

TEST(SendStubTest, Qwen4PleFlashPGtDReplicatedState) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  auto count_ple_records = [](uint32_t p_rank) {
    WorkerInfo src("0", 0);
    WorkerInfo dst("1", 0);
    make_qwen3_next_flash_pgtd_workers(src, dst);
    src.worker_tp_rank = p_rank;
    dst.worker_tp_rank = p_rank / 2;
    src.num_ple_layers = 1;
    src.ple_block_group = 2;
    src.ple_conv_state_shape = {256, 3, 4};
    src.ple_conv_elem_size = 2;

    auto valid_ranks = compute_valid_ranks_pd(
        src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
    const uint32_t eff_tp_size = static_cast<uint32_t>(valid_ranks.count());
    const uint32_t eff_tp_rank = static_cast<uint32_t>(
        (valid_ranks << (valid_ranks.size() - src.worker_tp_rank)).count());

    auto req = std::make_shared<RequestInfo>(
        "1", 0, "req0",
        BlockIds{{100}, {1}, {150}},
        BlockIds{{200}, {10}, {220}});
    ReqSendTask task(req, 0, 1, /*reach_last_token=*/true);

    std::vector<std::vector<IpcBlock>> send_blocks;
    vllm_parse_hybrid_block_send_p_gt_d(
        &src, &dst, valid_ranks, eff_tp_size, eff_tp_rank,
        &task, send_blocks);

    const size_t expected_src = 150u * src.block_sizes[0];
    const size_t expected_dst = 220u * dst.block_sizes[0];
    const size_t expected_length = 3u * 4u * 2u;
    size_t count = 0;
    for (const auto& block : send_blocks.at(0)) {
      if (block.src_offset == expected_src &&
          block.dst_offset == expected_dst &&
          block.length == expected_length) {
        ++count;
      }
    }
    return count;
  };

  // P ranks 0/1 both map to D rank 0. Rank 0 is the representative and sends
  // the complete replicated PLE state; rank 1 must not write it again.
  EXPECT_EQ(count_ple_records(/*p_rank=*/0), 1u);
  EXPECT_EQ(count_ple_records(/*p_rank=*/1), 0u);
}

TEST(SendStubTest, Qwen4PleFlashPLtDReplicatedState) {
  if (env_cache_shape() != QWEN3_NEXT_FLASH_CACHE_SHAPE) return;

  for (uint32_t d_rank : {0u, 3u}) {
    WorkerInfo src("0", 0);
    WorkerInfo dst("1", 0);
    make_qwen3_next_flash_pltd_workers(src, dst, /*kv_heads=*/4);
    dst.worker_tp_rank = d_rank;
    src.num_ple_layers = 1;
    src.ple_block_group = 2;
    src.ple_conv_state_shape = {256, 3, 4};
    src.ple_conv_elem_size = 2;

    auto valid_ranks = compute_valid_ranks_pd(
        src.engine_tp_size, dst.engine_tp_size, src.num_kv_heads);
    auto req = std::make_shared<RequestInfo>(
        "1", 0, "req0",
        BlockIds{{100}, {1}, {150}},
        BlockIds{{200}, {10}, {220}});
    ReqSendTask task(req, 0, 1, /*reach_last_token=*/true);

    std::vector<std::vector<IpcBlock>> send_blocks;
    vllm_parse_hybrid_block_send_p_lt_d(
        &src, &dst, valid_ranks,
        /*kvt_tp_size=*/2, /*kvt_tp_rank=*/0,
        &task, send_blocks);

    const size_t expected_src = 150u * src.block_sizes[0];
    const size_t expected_dst = 220u * dst.block_sizes[0];
    const size_t expected_length = 3u * 4u * 2u;
    size_t count = 0;
    for (const auto& block : send_blocks.at(0)) {
      if (block.src_offset == expected_src &&
          block.dst_offset == expected_dst &&
          block.length == expected_length) {
        ++count;
      }
    }
    EXPECT_EQ(count, 1u) << "d_rank=" << d_rank;
  }
}

TEST(ComputeValidRanksTest, MixedRegimePLtD) {
  // p_tp=2 < heads=4 < d_tp=8: all P ranks valid (each holds distinct heads).
  auto vr = compute_valid_ranks_pd(2, 8, 4);
  EXPECT_EQ(vr.count(), 2u);
  EXPECT_TRUE(vr[0]);
  EXPECT_TRUE(vr[1]);
}

TEST(ComputeValidRanksTest, AllRanksValidWhenKvHeadsEqPtp) {
  // When d_tp <= num_kv_heads, use original logic
  auto vr = compute_valid_ranks_pd(8, 4, 8);  // p_tp=8, d_tp=4, num_kv_heads=8
  EXPECT_EQ(vr.count(), 8u);
  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(vr[i]);
  }
}

TEST(ComputeValidRanksTest, AllRanksValidWhenNoKvHeadsSet) {
  // When num_kv_heads <= 0, all ranks are valid
  auto vr = compute_valid_ranks_pd(8, 4, -1);  // p_tp=8, d_tp=4, num_kv_heads=-1
  EXPECT_EQ(vr.count(), 8u);
}

TEST(ComputeValidRanksTest, StridedValidRanks) {
  // When d_tp <= num_kv_heads, use original logic
  auto vr = compute_valid_ranks_pd(8, 2, 2);  // p_tp=8, d_tp=2, num_kv_heads=2
  EXPECT_EQ(vr.count(), 2u);
  EXPECT_TRUE(vr[0]);
  EXPECT_FALSE(vr[1]);
  EXPECT_FALSE(vr[2]);
  EXPECT_FALSE(vr[3]);
  EXPECT_TRUE(vr[4]);
  EXPECT_FALSE(vr[5]);
  EXPECT_FALSE(vr[6]);
  EXPECT_FALSE(vr[7]);
}

TEST(ComputeValidRanksTest, FourHeadsOnEightRanks) {
  // When d_tp <= num_kv_heads, use original logic
  auto vr = compute_valid_ranks_pd(8, 2, 4);  // p_tp=8, d_tp=2, num_kv_heads=4
  EXPECT_EQ(vr.count(), 4u);
  EXPECT_TRUE(vr[0]);
  EXPECT_FALSE(vr[1]);
  EXPECT_TRUE(vr[2]);
  EXPECT_FALSE(vr[3]);
  EXPECT_TRUE(vr[4]);
  EXPECT_FALSE(vr[5]);
  EXPECT_TRUE(vr[6]);
  EXPECT_FALSE(vr[7]);
}

TEST(ComputeValidRanksTest, DTpGtKvHeads) {
  // When d_tp > num_kv_heads, select one representative P rank per D subgroup.
  auto vr = compute_valid_ranks_pd(8, 4, 2);
  EXPECT_EQ(vr.count(), 4u);  // 4 valid ranks
  EXPECT_TRUE(vr[0]);   // rank 0 valid
  EXPECT_FALSE(vr[1]);  // rank 1 invalid
  EXPECT_TRUE(vr[2]);   // rank 2 valid
  EXPECT_FALSE(vr[3]);  // rank 3 invalid
  EXPECT_TRUE(vr[4]);   // rank 4 valid
  EXPECT_FALSE(vr[5]);  // rank 5 invalid
  EXPECT_TRUE(vr[6]);   // rank 6 valid
  EXPECT_FALSE(vr[7]);  // rank 7 invalid
}

}  // namespace blade_llm
