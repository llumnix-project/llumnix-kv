
#include <sys/epoll.h>
#include <future>
#include <sstream>
#include "protocol/rdma_protocol.h"
#include "thrid_party/logging.h"
#include "naming.h"
#include "assert.h"
#include "utils/timer.h"
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>

#ifdef ENABLE_TORCH
#include <c10/core/Storage.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Storage.h>
#endif  // ENABLE_TORCH

#ifdef ENABLE_RDMA
// just in this cpp
using namespace accl::barex;
#endif

namespace blade_llm {

static constexpr uint32_t SEND_MAGIC = 0x53456e64;  /* SEnd */

//
// magic, inst_id_len, worker_id, num_block 都是 4 字节.
// num_block 指定了 block_ids 中 block 个数, 每个 block 4 字节.
// reqid 以 0 结尾的 C 字符串.
// +-------+-------------+-----------+-----------+---------+-----------+-------+
// | magic | inst_id_len | worker_id | num_block | inst_id | block_ids | reqid |
//
size_t get_encode_size(const InstanceId &inst_id,
                       uint32_t worker_id,
                       const std::string &reqid,
                       const std::vector<uint32_t> &block_ids) {
  return inst_id.size() + 4 * sizeof(uint32_t) + block_ids.size() * sizeof(uint32_t) + reqid.size() + 1;
}

void encode_notification(char *ptr,
                         const InstanceId &inst_id,
                         uint32_t worker_id,
                         const std::string &reqid,
                         const std::vector<uint32_t> &block_ids) {
  const uint32_t num_block = block_ids.size();
  const uint32_t magic = SEND_MAGIC;

  memcpy(ptr, &magic, sizeof(magic));
  ptr += sizeof(magic);
  uint32_t inst_id_len = inst_id.size();
  memcpy(ptr, &inst_id_len, sizeof(inst_id_len));
  ptr += sizeof(inst_id_len);
  memcpy(ptr, &worker_id, sizeof(worker_id));
  ptr += sizeof(worker_id);
  memcpy(ptr, &num_block, sizeof(num_block));
  ptr += sizeof(num_block);
  memcpy(ptr, inst_id.data(), inst_id_len);
  ptr += inst_id_len;

  memcpy(ptr, block_ids.data(), num_block * sizeof(uint32_t));
  ptr += num_block * sizeof(uint32_t);

  // *(s.begin() + s.size()) has value CharT() (a null terminator)
  memcpy(ptr, reqid.data(), reqid.size() + 1);
  return;
}

bool decode_notification(const char *in_buf,
                         size_t len,
                         InstanceId &inst_id,
                         uint32_t &worker_id,
                         std::string &req_id,
                         std::vector<uint32_t> &block_ids) {
  if (len < sizeof(uint32_t) * 4) {
    // bad data, ignore.
    return false;
  }
  const char *const end_buf = in_buf + len;
  uint32_t inst_id_len;
  uint32_t magic, num_block;
  memcpy(&magic, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  memcpy(&inst_id_len, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  memcpy(&worker_id, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  memcpy(&num_block, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  if (magic != SEND_MAGIC) {
    // bad data, ignore
    LOG(ERROR) << "KVT RDMA: unrecognized messages;";
    return false;
  }

  inst_id.resize(inst_id_len);
  memcpy(inst_id.data(), in_buf, inst_id_len);
  in_buf += inst_id_len;

  size_t expected_len = sizeof(uint32_t) * (4 + num_block) + 1 + inst_id_len;
  if (len < expected_len) {
    // +1 for reqid null terminator
    // bad data, ignore.
    LOG(ERROR) << "KVT RDMA: unexpected message, expect size: " << expected_len << " actual: " << len;
    return false;
  }

  block_ids.resize(num_block);
  memcpy(block_ids.data(), in_buf, num_block * sizeof(uint32_t));
  in_buf += num_block * sizeof(uint32_t);

  const char *reqid = in_buf;
  size_t reqid_len = end_buf - in_buf;
  assert(reqid_len >= 1);
  reqid_len -= 1;
  if (reqid[reqid_len] != '\0') {
    // bad data, ignore
    LOG(ERROR) << "KVT RDMA: unexpected eof of request id;";
    return false;
  }
  req_id = std::string(reqid, reqid_len);
  return true;
}

#ifdef ENABLE_RDMA

static int env2posint(const char* env, int def) {
  assert(def > 0);
  const char *valstr = getenv(env);
  if (valstr == nullptr) {
    return def;
  }
  int ret = atoi(valstr);
  if (ret <= 0) {
    return def;
  }
  return ret;
}

static int env_send_parallel() {
  static constexpr int DEFVAL = 1;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_RDMA_SP", DEFVAL);
  });
  return val;
}

static int env_ctx_tpsize() {
  static constexpr int DEFVAL = 4;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_CTX_TPSIZE", DEFVAL);
  });
  return val;
}


static int env_conn_tpsize() {
  static constexpr int DEFVAL = 2;
  static int val = DEFVAL;
  static std::once_flag flag;
  std::call_once(flag, [] () {
    val = env2posint("BLLM_KVTRANS_CONN_TPSIZE", DEFVAL);
  });
  return val;
}


void XContextDeleter::operator()(accl::barex::XContext *ctx) {
  ctx->Shutdown();
  ctx->WaitStop();
  delete ctx;
}

void XMempoolDeleter::operator()(accl::barex::XSimpleMempool *mempool) {
  mempool->Shutdown();
  mempool->WaitStop();
  delete mempool;
}

void XThreadpoolDeleter::operator()(accl::barex::XThreadpool *tp) {
  tp->Shutdown();
  tp->WaitStop();
  delete tp;
}

void XListenerDeleter::operator()(accl::barex::XListener *tp) {
  tp->Shutdown();
  tp->WaitStop();
  delete tp;
}

void XConnectorDeleter::operator()(accl::barex::XConnector *tp) {
  tp->Shutdown();
  tp->WaitStop();
  delete tp;
}

BarexMRGuard::~BarexMRGuard() {
  auto &self = *this;
  if (self.mp_ == nullptr) {
    return;  // moved
  }
  BarexResult result;
  if (release_) {
    result = self.mp_->ReleaseAndDeregBuffer(self.mr_.buf, self.mr_.d_type);
  } else {
    result = self.mp_->DeregUserMr(self.mr_.buf, self.mr_.d_type);
  }
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
}

// GPU 0 对应 NIC: RET[0]
// filename 格式: vsolar_1,vsolar_1,vsolar_1,vsolar_1,vsolar_0,vsolar_0,vsolar_0,vsolar_0
static std::vector<std::string> load_nic_affinity(const char* filename) {
  auto file = std::ifstream(filename);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("load_nic_affinity failed. filename=") + filename);
  }

  auto result = std::vector<std::string>();
  std::string line;
  std::getline(file, line);
  size_t start = 0;
  size_t end;
  while ((end = line.find(',', start)) != std::string::npos) {
    result.push_back(line.substr(start, end - start));
    start = end + 1;
  }
  if (start < line.length()) {
    result.push_back(line.substr(start));
  }

  return result;
}

static const std::vector<std::string>& get_nic_affinity() noexcept {
  static std::vector<std::string> result;
  static std::once_flag load_nic_aff_once;
  std::call_once(load_nic_aff_once, [] () {
    result = load_nic_affinity("/tmp/pai_blade_llm_kvtransfer_rdma_nic_affinity.txt");
  });
  return result;
}

// CUDA_VISIBLE_DEVICES=4,5,6,7
// 返回空若 CUDA_VISIBLE_DEVICES 未设置.
static std::vector<int> parse_cuda_visible_devs() {
  const char* env_data = getenv("CUDA_VISIBLE_DEVICES");
  if (env_data == nullptr) {
    return {};
  }
  std::string line(env_data);
  if (line.empty()) {
    return {};
  }

  std::vector<int> result;
  size_t start = 0;
  size_t end;
  while ((end = line.find(',', start)) != std::string::npos) {
    result.push_back(std::stoi(line.substr(start, end - start)));
    start = end + 1;
  }
  if (start < line.length()) {
    result.push_back(std::stoi(line.substr(start)));
  }
  return result;
}

static const std::vector<int>& get_cuda_visible_devs() noexcept {
  static std::vector<int> result;
  static std::once_flag once;
  std::call_once(once, [] () {
    result = parse_cuda_visible_devs();
  });
  return result;
}

using XDevice = accl::barex::XDevice;
using accl::barex::XSimpleMempool;

static XDevice* choose_nic(const std::vector<XDevice *> &nic_devs, int gpu_dev) noexcept {
  assert(gpu_dev >= 0);
  assert(!nic_devs.empty());

  int real_gpu_rank = gpu_dev;
  const auto& cuda_visible_env = get_cuda_visible_devs();
  if (!cuda_visible_env.empty()) {
    real_gpu_rank = cuda_visible_env.at(gpu_dev);
  }
  const auto& lovely_nic = get_nic_affinity().at(real_gpu_rank);

  for (auto* nic_dev : nic_devs) {
    if (nic_dev->GetName() == lovely_nic) {
      LOG(INFO) << "choose_nic. gpu_dev=" << gpu_dev
                << ", real_gpu_rank=" << real_gpu_rank
                << ", lovely_nic=" << lovely_nic;
      return nic_dev;
    }
  }

  // https://project.aone.alibaba-inc.com/v2/project/664220/req/61087840
  // 如 aone 所示, XPU 要求必须亲和网卡, 这里 kvtransfer 可能无法发送数据.
  auto* dev = uint32_t(gpu_dev) >= nic_devs.size() ? nic_devs[0] : nic_devs[gpu_dev];
  LOG(WARNING) << "choose_nic fallback, may not work on XPU. gpu_dev=" << gpu_dev
               << ", dev=" << dev->GetName();
  return dev;
}

class MpManager {
public:
  struct GPUCtx {
    XDevice* lovely_nic = nullptr;
    std::unique_ptr<accl::barex::XSimpleMempool, XMempoolDeleter> mp;
  };
public:
  MpManager() = default;

  MpManager(const MpManager&) = delete;
  MpManager(MpManager&&) = delete;

  std::pair<XDevice*, XSimpleMempool*> get_gpu_ctx(int gpu_id) const;
private:
  std::mutex m_;
  std::unordered_map<int, GPUCtx> map_;
};

std::pair<XDevice*, XSimpleMempool*> MpManager::get_gpu_ctx(int gpu_id) const {
  auto& self = *const_cast<MpManager*>(this);   // SAFETY: we have a mutex!
  auto guard = std::lock_guard<std::mutex>(self.m_);

  auto iter = self.map_.find(gpu_id);
  if (iter != self.map_.end()) {
    return {iter->second.lovely_nic, iter->second.mp.get()};
  }

  GPUCtx ctx;
  XDeviceManager *manager = nullptr;
  auto result = XDeviceManager::Singleton(manager);
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  auto all_nic_devs = manager->AllDevices();
  RTCHECK(!all_nic_devs.empty());
  ctx.lovely_nic = choose_nic(all_nic_devs, gpu_id);

  XSimpleMempool *mempool = nullptr;
  std::string mpname = "mp-" + std::to_string(gpu_id);
  result = XSimpleMempool::NewInstance(mempool, std::move(mpname), {ctx.lovely_nic});
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  ctx.mp.reset(mempool);

  auto [iter2, ok] = self.map_.emplace(gpu_id, std::move(ctx));
  assert(ok);
  assert(iter2->second.mp.get() == mempool);
  return {iter2->second.lovely_nic, iter2->second.mp.get()};
}

static MpManager g_mp_manager;

static void BarexCtxMain(XContext *ctx, std::atomic<bool> *stop_flag) {
  int evfd = ctx->GetEventFd();
  constexpr int EVENT_MAX = 8;
  struct epoll_event events[EVENT_MAX];
  while (!stop_flag->load()) {
    int ret = epoll_wait(evfd, events, EVENT_MAX, 100 /* ms */);
    if (ret == -1 && errno == EINTR) {
      continue;
    }
    RTCHECK(ret >= 0);

    while (ctx->ProgressEvents() > 0) {
      // pass
    }
  }
  ctx->Shutdown();
  ctx->WaitStop();
  return;
}

BarexCtx::BarexCtx(std::string mp_name,
                   std::string tp_name,
                   int tpcnt,
                   Context *ctx,
                   std::unique_ptr<accl::barex::XChannelCallback> ctxcb) {
  auto &self = *this;
  auto [nic_dev, mp] = g_mp_manager.get_gpu_ctx(ctx->device_id());
  self.mp_ = mp;

  const auto &layer_ptr = ctx->layer_data_address();
  // accl.barex 注册 MR 个数最大为 65536, 即要求 layer_ptr.size <= 65536.
  // 考虑到 LAYER_NUM_MAX 远小于 65536, 所以只需要 LAYER_NUM_MAX 限制即可.
  RTCHECK(layer_ptr.size() <= LAYER_NUM_MAX);
  // 每个 mr 默认大小限制为 1GB.
  size_t max_mr_size = 1L * 1024 * 1024 * 1024;
  const char *max_mr_gb_str = getenv("ACCL_MAX_USER_MR_GB");
  if (max_mr_gb_str != nullptr) {
    auto tmp_max_mr_gb = atoi(max_mr_gb_str);
    if (tmp_max_mr_gb > 0) {
      max_mr_size = max_mr_size * tmp_max_mr_gb;
    }
  }
  auto layer_blk_size = ctx->block_size() * ctx->layer_num_blocks();
  LOG(INFO) << "layer size(layer_blk_size) = " << layer_blk_size << ", max_mr_size = " << max_mr_size;
  // 如果这里跪了, 需要配置环境变量 ACCL_MAX_USER_MR_GB
  RTCHECK(layer_blk_size <= max_mr_size);

  self.layer_mr_.reserve(layer_ptr.size());
  for (auto layer_p : layer_ptr) {
    memp_t out;
    auto layer_blk_p = reinterpret_cast<void *>(layer_p);
    // 虽然注释上提到 RegUserMr 要求对齐. 但钉钉确认了, 只要是 cudaMalloc 返回的地址都可以.
    auto result = self.mp_->RegUserMr(out, layer_blk_p, layer_blk_size, GPU, ctx->device_id());
    RTCHECK(result == accl::barex::BAREX_SUCCESS);
    RTCHECK(out.d_type == GPU);
    RTCHECK(out.device_id == ctx->device_id());
    LOG(INFO) << "RegUserMr. layer_blk_p=" << layer_blk_p << ", layer_blk_size=" << layer_blk_size
              << ", gpuid=" << ctx->device_id();
    self.layer_mr_.emplace_back(BarexMRGuard::DeregGuard(std::move(out), self.mp()));
  }

  XThreadpool *threadpool = nullptr;
  auto result = XThreadpool::NewInstance(threadpool, tpcnt, std::move(tp_name));
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  self.tp_.reset(threadpool);

  XContext *context = nullptr;
  ContextConfig config = XConfigUtil::DefaultContextConfig();
  result = XContext::NewInstance(context, config, ctxcb.get(), nic_dev, self.mp(), threadpool);
  ctxcb.release();
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  self.xctx_.reset(context);

  context->Start();
}

BarexCtx::~BarexCtx() {
  this->xctx_->Shutdown();
  this->xctx_->WaitStop();
}

CliBarexCtx::CliBarexCtx(std::string mp_name,
                         std::string tp_name,
                         int tpcnt,
                         Context *ctx,
                         std::unique_ptr<accl::barex::XChannelCallback> ctxcb) :
    BarexCtx(std::move(mp_name), std::move(tp_name), tpcnt, ctx, std::move(ctxcb)),
    layer_blk_size(ctx->layer_num_blocks() * ctx->block_size()) {
  XConnector *connector = nullptr;
  auto result = XConnector::NewInstance(connector, env_conn_tpsize(), TIMER_3S, {this->xctx()});
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  this->connector_.reset(connector);
  return;
}

void RDMAServer::CtxCallback::OnRecvCall(XChannel *_ch, char *in_buf, size_t len, x_msg_header _header) {
  InstanceId inst_id;
  uint32_t worker_id;
  std::string req_id;
  std::vector<uint32_t> block_ids;

  if (decode_notification(in_buf, len, inst_id, worker_id, req_id, block_ids)) {
    // OnRecvCall 在 barex 线程池中调用. 注意 data race.
    this->ser_->on_recv(inst_id, worker_id, req_id, std::move(block_ids));
  }
  return;
}

class FdGuard {
  int const fd_;
 public:
  FdGuard(int fd) noexcept: fd_(fd) {}
  ~FdGuard() {
    ::close(this->fd_);
  }
};

static int get_port() {
  int sockfd, portno = 0;
  struct sockaddr_in serv_addr;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    return -1;
  }
  auto guard = FdGuard(sockfd);

  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(portno);
  if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    return -1;
  }

  socklen_t len = sizeof(serv_addr);
  if (getsockname(sockfd, (struct sockaddr *) &serv_addr, &len) == -1) {
    return -1;
  }
  assert(serv_addr.sin_family == AF_INET);
  portno = ntohs(serv_addr.sin_port);
  assert(portno > 0);

  return portno;
}

// barex listen 实现是 bind INADDR_ANY, 因此我们随便返回一个对外可用的 ip 地址均可.
static void get_ip(RDMAInfo *out) {
  // SUSv2 guarantees that "Host names are limited to 255 bytes".
  char hostname[256];
  int status = gethostname(hostname, sizeof(hostname));
  RTCHECK(status == 0);

  struct addrinfo *res = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  status = getaddrinfo(hostname, NULL, &hints, &res);
  RTCHECK(status == 0);
  auto guard = std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)>(res, freeaddrinfo);

  assert(res->ai_family == AF_INET);
  assert(res->ai_addr->sa_family == AF_INET);
  auto *ipaddr = (struct sockaddr_in *) res->ai_addr;
  auto *ret = inet_ntop(ipaddr->sin_family, &ipaddr->sin_addr, out->ip, sizeof(out->ip));
  RTCHECK(ret != nullptr);
  return;
}

void RDMAServer::start_server(ITransferService *service, Context *ctx) {
  auto &self = *this;
  if (service == nullptr) {
    throw std::runtime_error("KvTransferService should not be null;");
  }
  auto rdma_ctx = RDMAProtoContext::server_context("KVTServer", std::make_unique<CtxCallback>(service));
  if (!rdma_ctx->check_support()) {
    throw std::runtime_error("can't start RDMA transfer server as RDMA protocol not support;");
  }
  ctx->register_protocol(std::move(rdma_ctx));

  WorkerInfo *winfo = ctx->worker_info_mutable();
  auto layer_num_blocks = ctx->layer_num_blocks();
  auto layer_ptr = ctx->layer_data_address();
  auto proto = TransferProtocol::rdma_direct();
  auto proto_ctx = ctx->get_protocol_ctx<RDMAProtoContext>(proto);
  if (proto_ctx == nullptr) {
    throw std::runtime_error("KVT server: rdma context not register.");
  }
  auto barex_ctx = proto_ctx->barex_ctx();
  std::vector<void *> ptrs(layer_ptr.size());
  std::vector<uint32_t> rkeys(layer_ptr.size());

  for (size_t idx = 0; idx < layer_ptr.size(); ++idx) {
    auto &out = barex_ctx->layer_mr()[idx].mr();
    auto layer_blk_p = reinterpret_cast<void *>(layer_ptr[idx]);
    ptrs[idx] = layer_blk_p;
    rkeys[idx] = out.mr->rkey;
  }
  auto ptr_size = sizeof(void *) * layer_ptr.size();
  auto other_info_size = sizeof(uint32_t) * layer_ptr.size() + ptr_size;
  winfo->other_info.resize(other_info_size);
  auto other_info_ptr = winfo->other_info.data();
  memcpy(other_info_ptr, ptrs.data(), ptr_size);
  other_info_ptr += ptr_size;
  memcpy(other_info_ptr, rkeys.data(), sizeof(uint32_t) * layer_ptr.size());

  RDMAInfo info;
  get_ip(&info);
  info.port = get_port();
  RTCHECK(info.port > 0);

  std::stringstream ss;
  ss << info.ip << ":" << info.port;
  winfo->addr = ss.str();

  XListener *listener = nullptr;
  auto xctx = barex_ctx->xctx();
  if (xctx == nullptr) {
    LOG(ERROR) << "xctx is nullptr";
  }
  auto result = XListener::NewInstance(listener, env_conn_tpsize(), info.port, TIMER_3S, {xctx});
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  self.listener_.reset(listener);
  result = self.listener_->Listen();
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  LOG(INFO) << "RDMAServer.start_server: ip=" << info.ip << " port=" << info.port << " layer_num_blocks="
            << layer_num_blocks;
}

void RDMAChannel::connect(const WorkerInfo &dst_info) {
  auto &self = *this;
  std::string addr(dst_info.addr);
  auto pos = addr.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid rdma address: " + addr);
  }

  self.ip_ = addr.substr(0, pos);
  auto port_str = addr.substr(pos + 1, addr.size());
  self.port_ = std::stoi(port_str);
  dst_layer_blk_size_ = dst_info.layer_num_blocks * dst_info.block_size;
  dst_layer_num_ = dst_info.num_layers;
  auto handle_size = (sizeof(void *) + sizeof(uint32_t)) * dst_layer_num_;
  assert(handle_size == dst_info.other_info.size());
  dst_ptrs_.resize(dst_layer_num_);
  dst_rkeys_.resize(dst_layer_num_);
  auto bytes = dst_info.other_info.data();
  auto ptr_size = sizeof(void *) * dst_layer_num_;
  memcpy(dst_ptrs_.data(), dst_info.other_info.data(), ptr_size);
  bytes += ptr_size;
  memcpy(dst_rkeys_.data(), bytes, sizeof(uint32_t) * dst_layer_num_);
}

template<typename T, typename E>
static std::future<T> make_exp_future(E ex) {
  std::promise<T> pr;
  pr.set_exception(std::make_exception_ptr(std::move(ex)));
  return pr.get_future();
}

[[nodiscard]] static std::future<XChannel *> Connect(XConnector &self, std::string server_addr, int port) {
  // std::promise<XChannel*> pr;
  auto pr = std::make_shared<std::promise<XChannel *>>();
  auto fut = pr->get_future();

  auto result = self.Connect(std::move(server_addr), port,
                             [pr = std::move(pr)](XChannel *res, Status s) mutable {
                               if (!s.IsOk()) {
                                 auto ex = std::make_exception_ptr(std::runtime_error("Connect ERR: " + s.ErrMsg()));
                                 pr->set_exception(std::move(ex));
                                 return;
                               }
                               pr->set_value(res);
                             }
  );

  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Connect Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<XChannel *>(std::move(ex));
  }
  return fut;
}

void RDMAChannel::do_init() {
  auto &self = *this;
  if (!self.chs_.empty()) {
    return;
  }
  const int sp = env_send_parallel();
  assert(sp > 0);

  self.chs_.reserve(sp);
  auto futs = std::vector<std::future<XChannel *>>();
  futs.reserve(sp);
  auto conn = self.ctx_->connector();
  assert(conn != nullptr);
  LOG(INFO) << "KVT: rdma connect to : " << self.ip_ << ":" << self.port_;
  for (int i = 0; i < sp; ++i) {
    auto fut = Connect(*conn, self.ip_, self.port_);
    futs.emplace_back(std::move(fut));
  }

  for (auto &fut : futs) {
    self.chs_.emplace_back(fut.get());
  }
  assert(!self.chs_.empty());
  return;
}

[[nodiscard]] static std::future<void> WriteSingle(XChannel *ch, memp_t sdata, uint64_t raddr, uint32_t rkey) {
  // std::promise<void> pr;
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();

  // LOG(INFO) << "zydebug WriteSingle: raddr=" << raddr << " rkey=" << rkey;
  auto result = ch->WriteSingle(std::move(sdata), raddr, rkey, false, 0,
                                [pr = std::move(pr)](Status s) mutable {
                                  if (!s.IsOk()) {
                                    auto ex = std::make_exception_ptr(std::runtime_error("Write ERR: " + s.ErrMsg()));
                                    pr->set_exception(std::move(ex));
                                    return;
                                  }
                                  pr->set_value();
                                }
  );
  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<void>(std::move(ex));
  }

  return fut;
}

// return send_us
[[nodiscard]] static std::future<uint64_t> WriteBatch(XChannel *ch, std::shared_ptr<std::vector<rw_memp_t>> datasp) {
  // std::promise<void> pr;
  auto pr = std::make_shared<std::promise<uint64_t>>();
  auto fut = pr->get_future();
  auto &datas = *datasp;

  const auto write_start_ts = SteadyClock::now();
  auto result = ch->WriteBatch(datas,
                               [pr = std::move(pr), d = std::move(datasp), write_start_ts](Status s) mutable {
                                 // WriteBatch 要求 datasp 直至 callback 中才能回收.
                                 if (!s.IsOk()) {
                                   auto ex = std::make_exception_ptr(std::runtime_error("Write ERR: " + s.ErrMsg()));
                                   pr->set_exception(std::move(ex));
                                   return;
                                 }
                                 auto const send_us = elapse_us(write_start_ts, SteadyClock::now());
                                 pr->set_value(send_us);
                               }
  );

  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<uint64_t>(std::move(ex));
  }

  return fut;
}

accl::barex::XChannel *RDMAChannel::ch() noexcept {
  auto &self = *this;
  int n = self.chs_.size();
  int idx = (++self.prev_ch_idx_) % n;
  return self.chs_[idx];
}


// group 之后, input 呈现出:
// <src_off_1, dst_off_0, len1>
// <src_off_2, dst_off_0, len2>
// 这里意味着 src_off_1, len1; src_off_2, len2 的内容要写入 dst_off_0
// <src_off_3, dst_off_1, len3>
// <src_off_4, dst_off_1, len4>
// <src_off_5, dst_off_1, len5>
// return min_size, max_size, total_size, cnt
static std::tuple<size_t, size_t, size_t, size_t> group_by_dst(std::vector<IpcBlock>& input) {
  assert(!input.empty());

  std::sort(input.begin(), input.end(),
    [](const IpcBlock& x, const IpcBlock& y) { return x.dst_offset < y.dst_offset; });

  size_t min_size = UINT64_MAX;
  size_t max_size = 0;
  size_t total_size = 0;
  size_t cnt = 0;

  size_t prev_idx = 0;
  size_t prev_end = input[0].length + input[0].dst_offset;
  for (size_t idx = 1; idx < input.size(); ++idx) {
    auto& blk = input[idx];
    if (blk.dst_offset > prev_end) {
      cnt += 1;
      size_t blksize = prev_end - input[prev_idx].dst_offset;
      min_size = std::min(blksize, min_size);
      max_size = std::max(blksize, max_size);
      total_size += blksize;

      prev_idx = idx;
      prev_end = input[idx].length + input[idx].dst_offset;
      continue;
    }

    if (blk.dst_offset == prev_end) {
      input[idx].dst_offset = input[prev_idx].dst_offset;
      prev_end += input[idx].length;
      continue;
    }

    abort();
  }

  cnt += 1;
  size_t blksize = prev_end - input[prev_idx].dst_offset;
  min_size = std::min(blksize, min_size);
  max_size = std::max(blksize, max_size);
  total_size += blksize;
  return {min_size, max_size, total_size, cnt};
}

void RDMAChannel::register_data(std::vector<IpcBlock>& data, TPKind kind) {
  auto& self = *this;
  assert(!data.empty());

  assert(self.data_ == nullptr);
  self.data_ = &data;
  self.kind_ = kind;

#ifndef NDEBUG
  size_t total_len_debug = std::accumulate(data.begin(), data.end(), 0, [] (size_t acc, const IpcBlock& item) {
    return acc + item.length;
  });
#endif

  self.origin_sb_num_ = data.size();
  if (kind == TPKind::PGTD) {
    self.merged_sb_num_ = data.size();
    self.sb_size_max_ = data[0].length;
    self.sb_size_min_ = data[0].length;
    self.sb_size_total_ = data[0].length * data.size();
    return ;
  }

  if (kind == TPKind::PEQD) {
    auto const [min, max, total, cnt] = merge_interval(data);
    self.merged_sb_num_ = cnt;
    self.sb_size_min_ = min;
    self.sb_size_max_ = max;
    self.sb_size_total_ = total;
    assert(total_len_debug == self.sb_size_total_);
    assert(self.merged_sb_num_ <= self.origin_sb_num_);

    auto new_end = std::remove_if(data.begin(), data.end(), [] (const IpcBlock& item) {
      return item.length == 0;
    });
    data.erase(new_end, data.end());
    assert(data.size() == cnt);
    assert(!data.empty());
    return ;
  }

  assert(kind == TPKind::PLTD);
  auto const [min, max, total, cnt] = group_by_dst(data);
  self.merged_sb_num_ = cnt;
  self.sb_size_min_ = min;
  self.sb_size_max_ = max;
  self.sb_size_total_ = total;
  assert(self.merged_sb_num_ > 0);
  assert(self.merged_sb_num_ <= self.origin_sb_num_);
  assert(total_len_debug == self.sb_size_total_);

  return ;
}

void RDMAChannel::send_data(size_t layer_idx) {
  auto &self = *this;
  assert(layer_idx < self.dst_layer_num_);
  assert(self.dst_layer_num_ == self.ctx_->layer_mr().size());
  assert(self.dst_layer_num_ == self.dst_ptrs_.size());
  assert(self.dst_layer_num_ == self.dst_rkeys_.size());
  self.do_init();

  if (self.kind_ == TPKind::PLTD) {
    return self.send_data_pltd(layer_idx);
  }

  auto datasp = std::make_shared<std::vector<rw_memp_t>>();
  const auto& data = *self.data_;
  assert(!data.empty());
  for (const auto &[src_offset, dst_offset, len] : data) {
    assert(len > 0);
    assert(src_offset < self.ctx_->layer_blk_size);
    assert(len < self.ctx_->layer_blk_size);
    assert(src_offset + len <= self.ctx_->layer_blk_size);
    assert(dst_offset < self.dst_layer_blk_size_);
    assert(dst_offset + len <= self.dst_layer_blk_size_);

    auto src_mr = self.ctx_->layer_mr()[layer_idx].mr();
    src_mr.buf += src_offset;
    src_mr.buf_len = len;

    auto rkey = self.dst_rkeys_[layer_idx];
    auto *rladdr = reinterpret_cast<char *>(self.dst_ptrs_[layer_idx]);
    assert(rladdr);
    uint64_t raddr = reinterpret_cast<uint64_t>(rladdr + dst_offset);
    datasp->emplace_back(rw_memp_t{std::move(src_mr), raddr, rkey});
  }
  assert(!datasp->empty());

  auto fut = WriteBatch(self.ch(), std::move(datasp));
  self.write_futs_.emplace_back(std::move(fut));
  return;
}


// return send_us
[[nodiscard]] static std::future<uint64_t> WriteBySgList(XChannel* ch, uint64_t remote_addr, uint32_t rkey, std::shared_ptr<std::vector<memp_t>> prefills) {
  auto pr = std::make_shared<std::promise<uint64_t>>();
  auto fut = pr->get_future();
  auto& datas = *prefills;

  const auto start_ts = SteadyClock::now();
  auto result = ch->WriteBySgList(datas,
    remote_addr, rkey,
    /* signal_peer */ false,
    /* imm_data */ 0,
    [prefills=std::move(prefills), pr=std::move(pr), start_ts] (Status s) {
      // WriteBySgList 要求 prefills 直至 callback 中才能回收.
      if (!s.IsOk()) {
        auto ex = std::make_exception_ptr(std::runtime_error("Write ERR: " + s.ErrMsg()));
        pr->set_exception(std::move(ex));
        return;
      }
      auto send_us = elapse_us(start_ts, SteadyClock::now());
      pr->set_value(send_us);
    });

  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<uint64_t>(std::move(ex));
  }

  return fut;
}

void RDMAChannel::send_data_pltd(size_t layer_idx) {
  auto& self = *this;
  assert(self.kind_ == TPKind::PLTD);
  assert(!self.chs_.empty());
  assert(!self.data_->empty());
  const auto& input = *self.data_;
  uint64_t const rladdr = reinterpret_cast<uint64_t>(self.dst_ptrs_[layer_idx]);
  auto const rkey = self.dst_rkeys_[layer_idx];

#ifndef NDEBUG
  uint64_t dst_len_debug = 0;
#endif
  uint64_t dst_offset = input[0].dst_offset;
  auto prefills = std::make_shared<std::vector<memp_t>>();
  assert(dst_offset < self.dst_layer_blk_size_);
  {
    const auto& blk = input[0];
    auto src_mr = self.ctx_->layer_mr()[layer_idx].mr();
    assert(blk.length > 0);
    assert(blk.length < self.ctx_->layer_blk_size);
    assert(blk.src_offset < self.ctx_->layer_blk_size);
    assert(blk.src_offset + blk.length <= self.ctx_->layer_blk_size);
    src_mr.buf += blk.src_offset;
    src_mr.buf_len = blk.length;
    prefills->emplace_back(std::move(src_mr));

#ifndef NDEBUG
    dst_len_debug += blk.length;
#endif
  }

  for (size_t idx = 1; idx < input.size(); ++idx) {
    const auto& blk = input[idx];
    if (blk.dst_offset == dst_offset) {
      auto src_mr = self.ctx_->layer_mr()[layer_idx].mr();
      assert(blk.length > 0);
      assert(blk.length < self.ctx_->layer_blk_size);
      assert(blk.src_offset < self.ctx_->layer_blk_size);
      assert(blk.src_offset + blk.length <= self.ctx_->layer_blk_size);
      src_mr.buf += blk.src_offset;
      src_mr.buf_len = blk.length;
      prefills->emplace_back(std::move(src_mr));

  #ifndef NDEBUG
      dst_len_debug += blk.length;
  #endif
      continue;
    }
    assert(!prefills->empty());
    assert(dst_len_debug > 0);
    assert(dst_len_debug <= self.dst_layer_blk_size_);
    assert(dst_offset + dst_len_debug <= self.dst_layer_blk_size_);
    self.write_futs_.emplace_back(WriteBySgList(self.ch(), dst_offset + rladdr, rkey, std::move(prefills)));

  #ifndef NDEBUG
    dst_len_debug = 0;
  #endif
    dst_offset = blk.dst_offset;
    prefills = std::make_shared<std::vector<memp_t>>();
    assert(dst_offset < self.dst_layer_blk_size_);
    {
      auto src_mr = self.ctx_->layer_mr()[layer_idx].mr();
      assert(blk.length > 0);
      assert(blk.length < self.ctx_->layer_blk_size);
      assert(blk.src_offset < self.ctx_->layer_blk_size);
      assert(blk.src_offset + blk.length <= self.ctx_->layer_blk_size);
      src_mr.buf += blk.src_offset;
      src_mr.buf_len = blk.length;
      prefills->emplace_back(std::move(src_mr));

  #ifndef NDEBUG
      dst_len_debug += blk.length;
  #endif
    }
  }

  assert(!prefills->empty());
  assert(dst_len_debug > 0);
  assert(dst_len_debug <= self.dst_layer_blk_size_);
  assert(dst_offset + dst_len_debug <= self.dst_layer_blk_size_);
  self.write_futs_.emplace_back(WriteBySgList(self.ch(), dst_offset + rladdr, rkey, std::move(prefills)));
  return ;
}

void RDMAChannel::do_write(uint32_t layer_idx,
                           size_t src_offset,
                           size_t dst_offset,
                           size_t len) {
  auto &self = *this;
  self.do_init();

  assert(src_offset < self.ctx_->layer_blk_size);
  assert(len < self.ctx_->layer_blk_size);
  assert(src_offset + len <= self.ctx_->layer_blk_size);
  assert(dst_offset < self.dst_layer_blk_size_);
  assert(dst_offset + len <= self.dst_layer_blk_size_);
  assert(layer_idx < self.dst_layer_num_);

  auto src_mr = self.ctx_->layer_mr()[layer_idx].mr();
  src_mr.buf += src_offset;
  src_mr.buf_len = len;

  auto rkey = self.dst_rkeys_[layer_idx];
  auto *rladdr = reinterpret_cast<char *>(self.dst_ptrs_[layer_idx]);
  assert(rladdr);
  uint64_t raddr = reinterpret_cast<uint64_t>(rladdr + dst_offset);

  WriteSingle(self.ch(), std::move(src_mr), raddr, rkey).get();
  return;
}

static void when_all_succeed(std::vector<std::future<void>> &futs) {
  for (auto &fut : futs) {
    fut.get();
  }
  return;
}

void RDMAChannel::flush(std::string& outstr) {
  auto &self = *this;
  self.data_ = nullptr;

  const auto inflyn = self.write_futs_.size();
  auto out = std::ostringstream();
  out << std::fixed << std::setprecision(3)
      << "OriginSbNum=" << self.origin_sb_num_
      << ",MergedSbNum=" << self.merged_sb_num_
      << ",SbSizeMin=" << self.sb_size_min_
      << ",SbSizeMax=" << self.sb_size_max_
      << ",SbSizeTotal=" << self.sb_size_total_
      << ",InflyWrite=" << inflyn;

  uint64_t send_us_min = UINT64_MAX;
  uint64_t send_us_max = 0;
  uint64_t send_us_total = 0;
  for (auto& fut : self.write_futs_) {
    uint64_t send_us = 0;
    try {
      send_us = fut.get();
    } catch (...) {
      outstr = std::move(out).str();
      throw;
    }

    send_us_min = std::min(send_us_min, send_us);
    send_us_max = std::max(send_us_max, send_us);
    send_us_total += send_us;
  }
  self.write_futs_.clear();

  out << ",SendUsMin=" << send_us_min
      << ",SendUsMax=" << send_us_max
      << ",SendUsAvg=" << send_us_total / float(inflyn);
  outstr = std::move(out).str();

  return;
}

// Send will release sdata
[[nodiscard]] static std::future<void> Send(XChannel *ch, memp_t sdata) {
  // barex Send 要求 callback copyable...
  // std::promise<void> pr;
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();
  auto bufptr = sdata.buf;
  auto bufdev = sdata.d_type;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  struct x_msg_header header = {0};
#pragma GCC diagnostic pop

  auto result = ch->Send(std::move(sdata), /* auto_release */ true, header,
                         [pr = std::move(pr), bufptr, bufdev, ch](Status s) mutable {
                           if (!s.IsOk()) {
                             // send 失败时, auto_release 不会 release.
                             ch->ReleaseBuffer(bufptr, bufdev);
                             auto ex = std::make_exception_ptr(std::runtime_error("Send ERR: " + s.ErrMsg()));
                             pr->set_exception(std::move(ex));
                             return;
                           }
                           pr->set_value();
                         }
  );
  if (result != BAREX_SUCCESS) {
    ch->ReleaseBuffer(bufptr, bufdev);
    auto ex = std::runtime_error("Send Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<void>(std::move(ex));
  }

  return fut;
}

void RDMAChannel::send_notification(const std::vector<const ReqSendTask*>& reqs) {
  auto &self = *this;
  self.do_init();

  assert(self.send_futs_.empty());
  self.send_futs_.reserve(reqs.size());

  for (const auto* r : reqs) {
    const auto &reqid = r->req_id();
    const auto &block_ids = r->dst_blocks();
    // 编码规则见 RDMAServer::CtxCallback::OnRecvCall
    memp_t bufmr;
    auto const msglen = get_encode_size(src_inst_id_, src_worker_id_, reqid, block_ids);
    auto result = self.ctx_->mp()->AllocBuffer(bufmr, msglen, CPU);
    RTCHECK(result == accl::barex::BAREX_SUCCESS);
    encode_notification(bufmr.buf, self.src_inst_id_, self.src_worker_id_, reqid, block_ids);

    auto *use_ch = self.ch();
    assert(use_ch->GetMempool() == self.ctx_->mp());
    auto fut = Send(use_ch, std::move(bufmr));
    self.send_futs_.emplace_back(std::move(fut));
  }

  when_all_succeed(self.send_futs_);
  self.send_futs_.clear();
  return;
}

bool RDMAProtoContext::check_support() {
  try {
    XDeviceManager *manager = nullptr;
    auto result = XDeviceManager::Singleton(manager);
    if (result != accl::barex::BAREX_SUCCESS) {
      return false;
    }
    auto all_nic_devs = manager->AllDevices();
    if (all_nic_devs.empty()) {
      return false;
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "RDMAProtoContext::check_support: " << e.what() << " return false;";
    return false;
  }
  return true;
}
void RDMAProtoContext::init(Context *ctx) {
  if (is_server) {
    barex_ctx_ = std::make_unique<BarexCtx>(
      name_prefix + "mp",
      name_prefix + "tp",
      env_ctx_tpsize(),
      ctx,
      std::move(callback_));
  } else {
    cli_barex_ctx_ = std::make_unique<CliBarexCtx>(
      name_prefix + "mp",
      name_prefix + "tp",
      env_ctx_tpsize(),
      ctx,
      std::move(callback_));
  }
}
std::unique_ptr<RDMAProtoContext> RDMAProtoContext::server_context(std::string &&name,
                                                                   std::unique_ptr<accl::barex::XChannelCallback> &&bk) {
  return std::make_unique<RDMAProtoContext>(std::move(name), true, std::move(bk));
}
std::unique_ptr<RDMAProtoContext> RDMAProtoContext::client_context(std::string &&name) {
  return std::make_unique<RDMAProtoContext>(std::move(name), false, std::make_unique<CliCtxCallback>());
}

#ifdef ENABLE_TORCH

struct DataPtrCtx {
  void* const ptr = nullptr;
  XAllocator* const allocator = nullptr;
  int const gpu_id = 0;  // for debug
  size_t const size = 0;
public:
  DataPtrCtx(void* p, XAllocator* a, int g, size_t s) noexcept:
    ptr(p), allocator(a), gpu_id(g), size(s) {
    // DataPtrCtx 构造场景也非常稀疏, 打印点日志没关系的.
    auto& self = *this;
    LOG(INFO) << "DataPtrCtx. ptr=" << self.ptr
              << ", allocator=" << self.allocator
              << ", gpu_id=" << self.gpu_id
              << ", size=" << self.size;
  }

  DataPtrCtx(const DataPtrCtx&) = delete;
  DataPtrCtx(DataPtrCtx&&) = delete;

  ~DataPtrCtx() {
    auto& self = *this;
    self.allocator->Release(self.ptr);
    // DataPtrCtx 析构场景非常稀疏. 打印点日志没关系的.
    LOG(INFO) << "~DataPtrCtx. ptr=" << self.ptr
              << ", allocator=" << self.allocator
              << ", gpu_id=" << self.gpu_id
              << ", size=" << self.size;
  }
};

static void DataPtrCtxDeleter(void* rctx) noexcept {
  auto* ctx = reinterpret_cast<DataPtrCtx*>(rctx);
  delete ctx;
}

// def alloc_phy_cont_mem(size, device: torch.device) -> torch.UntypedStorage
PyObject* alloc_phy_cont_mem(size_t size, PyObject* device) {
  RTCHECK(THPDevice_Check(device));
  auto* dev = reinterpret_cast<THPDevice*>(device);
  RTCHECK(dev->device.type() == c10::DeviceType::CUDA);
  RTCHECK(dev->device.has_index());
  int gpu_id = dev->device.index();

  XAllocator* gpu_allocator = nullptr;
  auto [_, mp] = g_mp_manager.get_gpu_ctx(gpu_id);
  auto result = mp->GetXAllocator(gpu_allocator, GPU);
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  // cudaMalloc 至少 256 对齐. align 在 PPU 上不生效, 即 PPU 上 kvcache 不保证对齐.
  void* const buf = gpu_allocator->Alloc(size, gpu_id, nullptr /* attr */, 512 /* align */);

  auto* dpctx = new DataPtrCtx(buf, gpu_allocator, gpu_id, size);
  auto data_ptr = c10::DataPtr(buf, dpctx, DataPtrCtxDeleter, dev->device);
  auto storage = c10::Storage(c10::Storage::use_byte_size_t{},
    size,
    std::move(data_ptr),
    nullptr,  // allocator, non-resizable!
    false /* resizable */);
  return THPStorage_Wrap(std::move(storage));
}

#endif   // ENABLE_TORCH

#endif // ENABLE_RDMA


}  // namespace blade_llm