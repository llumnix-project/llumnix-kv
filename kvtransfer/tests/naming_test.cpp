#include <gtest/gtest.h>
#include <thread>
#include <gmock/gmock.h>
#include <filesystem>
#include "naming.h"
#include "naming/shm_naming.h"
#include "naming/eas_naming.h"
#include "naming/filesys_naming.h"
#include "context.h"
#include "thrid_party/logging.h"
#include "protocol/rdma_protocol.h"


using namespace blade_llm;

TEST(NamingTest, TestShmNaming) {
  const std::string PATH = "shm_naming_test_x";
  ShmNamingServer n_server(PATH);
  n_server.start();

  ShmNamingClient n_client;
  n_client.connect(PATH);
  auto info = n_client.get_worker_info("1", 1);
  EXPECT_FALSE(info.has_value());
  Context ctx("1", 1);
  ctx.set_tp(2, 0);
  ctx.set_block_params({512}, {128}, 8);
  SupportTransferProtocols protocols;
  protocols.set_support(TransferProtocol::Kind::RDMA_DIRECT);
  auto worker_info = ctx.worker_info();
  worker_info.transfer_protocols = protocols.value();
  n_client.register_worker(worker_info);

  ShmNamingClient n_client2;
  n_client2.connect(PATH);
  auto info_opt = n_client2.get_worker_info("1", 1);
  EXPECT_TRUE(info_opt.has_value());
  EXPECT_EQ(worker_info.inst_id, info_opt->inst_id);
  EXPECT_EQ(worker_info.worker_id, info_opt->worker_id);
  EXPECT_EQ(worker_info.tp_size, info_opt->tp_size);
  EXPECT_EQ(worker_info.worker_tp_rank, info_opt->worker_tp_rank);
  EXPECT_EQ(worker_info.block_sizes, info_opt->block_sizes);
  EXPECT_EQ(worker_info.token_sizes, info_opt->token_sizes);
  EXPECT_EQ(worker_info.transfer_protocols, info_opt->transfer_protocols);
  SupportTransferProtocols check_protocol(info_opt->transfer_protocols);
  EXPECT_TRUE(check_protocol.is_support(TransferProtocol::Kind::RDMA_DIRECT));
}




using namespace std;
using ::testing::Eq;
using ::testing::Assign;

class MOCKHTTPClient : public IHttpClient {
 public:
  MOCKHTTPClient() = default;
  MOCK_METHOD(void, post, (const string &, const string &), (override));
  MOCK_METHOD(optional<string>, get, (const string &), (override));
};

class ProxyHTTPClient : public IHttpClient {
 public:
  explicit ProxyHTTPClient(IHttpClient *hc) : http_client_(hc) {}
  optional<string> get(const string &url) override {
    //LOG(INFO) << "get data from :" << url;
    return http_client_->get(url);
  }
  void post(const string &url, const string &data) override {
    //LOG(INFO) << "post data to : " << url;
    return http_client_->post(url, data);
  }

 private:
  IHttpClient *http_client_;
};

TEST(NamingTest, TestEASNaming) {
  string http_url = "http://127.0.0.1/mock";

  string instance_list_v0 = R"({"podname1":{"__timestamp__":"1730864908","__version__":"1"}})";
  string instance_list_v1 =
      R"({"podname1":{"__timestamp__":"1730864908","__version__":"1"},"podname2":{"__timestamp__":"1730864908","__version__":"1"}})";
  string instance_list_v2 =
      R"({"podname1":{"__timestamp__":"1730864908","__version__":"2"},"podname2":{"__timestamp__":"1730864908","__version__":"1"}})";
  string post_kv = R"({"k3":"v3"})";

  string instances_str_v0 = R"([{"__instance__":"podname1","__timestamp__":"1730858075","__version__":"1","key1":"value1","key2":"value2"}])";
  string instances_str_v1 =
      R"([{"__instance__":"podname1","__timestamp__":"1730858075","__version__":"1","key1":"value1","key2":"value2"},{"__instance__":"podname2","__timestamp__":"1730858075","__version__":"1","key1":"value11","key2":"value22"}])";
  string instances_str_v2 =
      R"([{"__instance__":"podname1","__timestamp__":"1730858075","__version__":"2","key1":"value1","key2":"value2", "key3":"value3"},{"__instance__":"podname2","__timestamp__":"1730858075","__version__":"1","key1":"value11","key2":"value22"}])";
  string instance_list_v3 =
      R"({"podname2":{"__timestamp__":"1730864908","__version__":"1"}})";
  string instances_str_v3 =
      R"([{"__instance__":"podname2","__timestamp__":"1730858075","__version__":"1","key1":"value11","key2":"value22"}])";

  string pod1_v1 =
      R"({"__instance__":"podname1","__timestamp__":"1730858075","__version__":"1","key1":"value1","key2":"value2"})";
  string pod1_v2 =
      R"({"__instance__":"podname1","__timestamp__":"1730858075","__version__":"2","key1":"value1","key2":"value2", "key3":"value3"})";
  string pod2_v1 = R"({"__instance__":"podname2","__timestamp__":"1730858075","__version__":"1","key1":"value11","key2":"value22"})";
  std::pair<string *, string *> post_kv_pair{&instance_list_v0, &instances_str_v0};

  MOCKHTTPClient mock_client;
  string get_api = http_url + EAS_GET_API;
  string list_api = http_url + EAS_LIST_API;
  string get_pod1_api = get_api + "/podname1";
  string get_pod2_api = get_api + "/podname2";
  EXPECT_CALL(mock_client, get(Eq(get_api)))
      .WillRepeatedly([&post_kv_pair]() { return *post_kv_pair.second; });
  EXPECT_CALL(mock_client, get(Eq(list_api)))
      .WillRepeatedly([&post_kv_pair]() { return *post_kv_pair.first; });
  EXPECT_CALL(mock_client, post(Eq(EAS_POST_PATH), Eq(post_kv))).Times(1);
  EXPECT_CALL(mock_client, get(Eq(get_pod1_api)))
       .WillOnce(::testing::Return(pod1_v1))
       .WillOnce(::testing::Return(pod1_v2));
  EXPECT_CALL(mock_client, get(Eq(get_pod2_api)))
      .WillOnce(::testing::Return(pod2_v1));

  EASNamingClient eas_naming_client("podname1", std::make_unique<ProxyHTTPClient>(&mock_client));
  eas_naming_client.connect(EAS_NAMING_SCHEMA, http_url);

  const auto &instance_v0 = eas_naming_client.list();
  EXPECT_EQ(instance_v0.size(), 1);
  EXPECT_EQ(instance_v0[0], "podname1");

  post_kv_pair = {&instance_list_v1, &instances_str_v1};
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  const auto &instances = eas_naming_client.list();
  EXPECT_EQ(instances.size(), 2);

  auto value1 = eas_naming_client.get("podname1", "key1");
  EXPECT_TRUE(value1.has_value());
  EXPECT_EQ(value1.value(), "value1");
  auto value2 = eas_naming_client.get("podname2", "key1");
  EXPECT_TRUE(value2.has_value());
  EXPECT_EQ(value2.value(), "value11");

  eas_naming_client.store("k3", "v3");
  post_kv_pair = {&instance_list_v2, &instances_str_v2};
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  auto value3 = eas_naming_client.get("podname1", "key3");
  EXPECT_TRUE(value3.has_value());
  EXPECT_EQ(value3.value(), "value3");
  auto values = eas_naming_client.search("podname1", "key");
  std::vector<std::string> expected{"value1", "value2", "value3"};
  EXPECT_TRUE(std::is_permutation(values.begin(), values.end(), expected.begin(), expected.end()));
  post_kv_pair = {&instance_list_v3, &instances_str_v3};
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  const auto &instance_v3 = eas_naming_client.list();
  vector<string> expect_inst_v3{"podname2"};
  EXPECT_EQ(instance_v3, expect_inst_v3);
  auto value = eas_naming_client.get("podname1", "key3");
  EXPECT_FALSE(value.has_value());
}

class FakeHttpClient : public IHttpClient {
 public:
  const static std::string FAKE_URL;
  std::string get_pod_api = EAS_GET_API + "/podname1";
  void post(const std::string &url, const std::string &data) override {
    auto pos = data.find(':');
    EXPECT_TRUE(pos != std::string::npos);
    auto key = data.substr(2, pos - 3);
    auto value = data.substr(pos + 2, data.size() - key.size() - 7);
    //LOG(INFO) << "key=" << key << ", value = " << value;
    kv_[key] = value;
    version++;
  }

  std::optional<std::string> get(const std::string &url) override {
    auto api = url.substr(FAKE_URL.size());
    if (api == EAS_GET_API) {
      std::stringstream ss;
      ss << R"([{"__instance__":"podname1","__timestamp__":"1730858075","__version__":")" << version << R"(")";
      for (const auto &[k, v] : kv_) {
        ss << R"(,")" << k << R"(":")" << v << R"(")";
      }
      ss << R"(}])";
      return ss.str();
    } else if (api == EAS_LIST_API) {
      std::stringstream ss;
      ss << R"({"podname1":{"__timestamp__":"1730864908","__version__":")" << version << R"("}})";
      return ss.str();
    } else if (api == get_pod_api) {
      std::stringstream ss;
      ss << R"({"__instance__":"podname1","__timestamp__":"1730858075","__version__":")" << version << R"(")";
      for (const auto &[k, v] : kv_) {
        ss << R"(,")" << k << R"(":")" << v << R"(")";
      }
      ss << R"(})";
      return ss.str();
    }
    EXPECT_TRUE(false);
    return std::nullopt;
  }
 private:
  size_t version{0};
  std::unordered_map<std::string, std::string> kv_;
};

const std::string FakeHttpClient::FAKE_URL = "http://fake";

void worker_naming_test(INamingClient* client) {
  WorkerNamingClient worker_client(client);
  WorkerInfo wi(client->inst_id, 1);
  wi.tp_size = 4;
  wi.worker_tp_rank = 1;
  wi.block_sizes = {1024 * 64};
  wi.token_sizes = {512};
  wi.layer_num_blocks = 10;
  wi.num_layers = 80;
  wi.transfer_protocols = 1;
  wi.addr = "127.0.0.1:8800";

  std::vector<void *> ptrs(80);
  std::vector<uint32_t> rkeys(80);

  for (auto i = 0; i < 80; ++i) {
    ptrs[i] = &wi;
    rkeys[i] = 512;
  }

  auto ptr_size = sizeof(void *) * ptrs.size();
  auto other_info_size = sizeof(uint32_t) * rkeys.size() + ptr_size;
  wi.other_info.resize(other_info_size);
  auto other_info_ptr = wi.other_info.data();
  memcpy(other_info_ptr, ptrs.data(), ptr_size);
  other_info_ptr += ptr_size;
  memcpy(other_info_ptr, rkeys.data(), sizeof(uint32_t) * rkeys.size());
  worker_client.register_worker(wi);

  auto worker_opt = worker_client.get_worker_info(client->inst_id, wi.worker_id);

  EXPECT_TRUE(worker_opt.has_value());
  auto wii = worker_opt.value();
  EXPECT_EQ(wi.inst_id, wii.inst_id);
  EXPECT_EQ(wi.worker_id, wii.worker_id);
  EXPECT_EQ(wi.tp_size, wii.tp_size);
  EXPECT_EQ(wi.worker_tp_rank, wii.worker_tp_rank);
  EXPECT_EQ(wi.block_sizes, wii.block_sizes);
  EXPECT_EQ(wi.token_sizes, wii.token_sizes);
  EXPECT_EQ(wi.layer_num_blocks, wii.layer_num_blocks);
  EXPECT_EQ(wi.num_layers, wii.num_layers);
  EXPECT_EQ(wi.transfer_protocols, wii.transfer_protocols);
  EXPECT_EQ(wi.addr, wii.addr);
  EXPECT_EQ(wi.other_info, wii.other_info);
}

TEST(NamingTest, TestWorkerEASNamingClient) {
  FakeHttpClient fake_http_client;
  EASNamingClient eas_naming_client("podname1", std::make_unique<ProxyHTTPClient>(&fake_http_client));
  eas_naming_client.connect(EAS_NAMING_SCHEMA, FakeHttpClient::FAKE_URL);
  worker_naming_test(&eas_naming_client);
}

TEST(NamingTest, TestFileSysNaming) {
  std::string naming_path = "./naming_test";
  std::filesystem::remove_all(naming_path);
  FileSysNaming::create_dir(naming_path);
  FileSysNaming fn("podname1");
  fn.connect(FILESYS_NAMING_SCHEMA, naming_path);
  fn.store("key1", "value1");
  auto pods = fn.list();
  EXPECT_EQ(pods.size(), 1);
  EXPECT_EQ(pods[0], "podname1");
  auto value = fn.get("podname1", "key1");
  EXPECT_TRUE(value.has_value());
  EXPECT_EQ(value.value(), "value1");

  FileSysNaming fn2("podname2");
  fn2.connect(FILESYS_NAMING_SCHEMA, naming_path);
  fn2.store("ekey1", "evalue1");
  fn2.store("ekey2", "evalue2");
  fn2.store("wkey1", "wvalue1");
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  pods = fn.list();
  std::vector<std::string> expected_pods = {"podname1", "podname2"};
  EXPECT_TRUE(std::is_permutation(pods.begin(), pods.end(), expected_pods.begin(), expected_pods.end()));
  auto values = fn.search("podname2", "ekey");
  std::vector<std::string> expected_values = {"evalue1", "evalue2"};
  EXPECT_TRUE(std::is_permutation(values.begin(), values.end(), expected_values.begin(), expected_values.end()));
}

TEST(NamingTest, TestWorkerFileNamingClient) {
  std::string naming_path = "./worker_naming_test";
  std::filesystem::remove_all(naming_path);
  FileSysNaming::create_dir(naming_path);
  FileSysNaming fn("podname1");
  fn.connect(FILESYS_NAMING_SCHEMA, naming_path);
  fn.store("endpoint", "127.0.0.1");
  auto v1 = fn.get("podname1", "endpoint");
  EXPECT_TRUE(v1.has_value());
  EXPECT_EQ(v1.value(), "127.0.0.1");
  worker_naming_test(&fn);
  auto v2 = fn.get("podname1", "endpoint");
  EXPECT_TRUE(v2.has_value());
  EXPECT_EQ(v2.value(), "127.0.0.1");
}
