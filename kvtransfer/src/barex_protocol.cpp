
#include <sys/epoll.h>
#include <future>
#include <sstream>
#include "protocol/barex_protocol.h"
#include "thrid_party/logging.h"
#include "naming.h"
#include "assert.h"
#include "envcfg.h"
#include "utils/timer.h"
#include "utils/id_generator.h"
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <numeric>
#include <functional>
#include <zlib.h>
#include "fault_inject.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

#ifdef ENABLE_RDMA

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

[[nodiscard]] static std::future<Status> CloseChannel(XConnector &self, XChannel* ch) {
  auto pr = std::make_shared<std::promise<Status>>();
  auto fut = pr->get_future();

  auto result = self.CloseChannel(ch, [pr] (Status s) {
    pr->set_value(std::move(s));
  });

  if (result != BAREX_SUCCESS) {
    pr->set_value(Status(result));
  }

  return fut;
}

BarexChannel::BarexChannel(accl::barex::XConnector* conn, std::shared_ptr<accl::barex::XChannel> ch) noexcept:
    connector_(conn), channel_(std::move(ch)) {
  LOG(INFO) << "BarexChannel new: connector=" << conn << ";channel=" << this->channel_.get();
}

void BarexChannel::destroy() {
  auto& self = *this;
  auto* ch = self.ch();
  auto close_ret = CloseChannel(*self.connector_, ch).get();
  auto destroy_ret = ch->Destroy();
  auto delete_ret = self.connector_->CloseAndDeleteChannel(ch);
  LOG(INFO) << "BarexChannel close:connector=" << self.connector_ << ";ch=" << ch
            << ";close_ret=" << close_ret.ErrMsg() << ";destroy_ret=" << destroy_ret
            << ";delete_ret=" << delete_ret;
  return;
}

BarexChannel::~BarexChannel() noexcept {
  if (this->channel_ == nullptr) {
    return;
  }
  try {
    this->destroy();
  } catch (const std::exception& ex) {
    LOG(ERROR) << "BarexChannel close:connector=" << this->connector_
               << ";ch=" << this->channel_.get() << ";ex=" << ex.what();
  }
}

// Send will release sdata
// cb is invoked after sdata is freed.
void Send(std::shared_ptr<XChannel>& ch, memp_t sdata, DoneCallback cb) {
  auto bufptr = sdata.buf;
  auto bufdev = sdata.d_type;
  auto wrapped_cb = [bufptr, bufdev, ch, cb=std::move(cb)] (Status s) {
    ch->ReleaseBuffer(bufptr, bufdev);
    cb(std::move(s));
  };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  struct x_msg_header header = {0};
#pragma GCC diagnostic pop

  try {
    auto result = ch->Send(std::move(sdata), /* auto_release */ false, header, wrapped_cb);
    RTCHECK_EQ(result, BAREX_SUCCESS);
    fault_inject_throw();
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Send error: ch=" << ch->ToString() << " ex=" << ex.what();
    wrapped_cb(Status(BAREX_ERR_INTERNAL));
  }

  return ;
}

// Send will release sdata
[[nodiscard]] std::future<void> Send(std::shared_ptr<XChannel>& ch, memp_t sdata) {
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();

  Send(ch, std::move(sdata), [pr=std::move(pr)] (Status s) {
    if (s.IsOk()) {
      pr->set_value();
    } else {
      auto ex = std::make_exception_ptr(std::runtime_error("Send ERR: " + s.ErrMsg()));
      pr->set_exception(std::move(ex));
    }
  });

  return fut;
}

memp_t AllocCPUBuffer(std::shared_ptr<XChannel>& ch, uint64_t size) {
  memp_t bufmr;
  auto result = ch->AllocBuffer(bufmr, size, CPU);
  RTCHECK_EQ(result, accl::barex::BAREX_SUCCESS);
  return bufmr;
}

memp_t AllocCudaHostBuffer(XChannel* ch, uint64_t size) {
  memp_t bufmr;
  auto result = ch->AllocBuffer(bufmr, size, CUDA_HOST);
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  return bufmr;
}

BarexMRGuard::~BarexMRGuard() {
  auto &self = *this;
  if (self.mp_ == nullptr) {
    return;  // moved
  }
  BarexResult result;
  if (release_ && dereg_) {
    result = self.mp_->ReleaseAndDeregBuffer(self.mr_.buf, self.mr_.d_type);
  } else if (!release_ && dereg_) {
    result = self.mp_->DeregUserMr(self.mr_.buf, self.mr_.d_type);
  } else if (release_ && !dereg_) {
    result = self.mp_->ReleaseBuffer(self.mr_.buf, self.mr_.d_type);
  } else {
    RTASSERT(false && "Unsupported BarexMRGuard type");
    return;  // unreachable, but satisfy compiler
  }
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
}

// GPU 0 corresponds to NIC RET[0].
// filename format:
// vsolar_1,vsolar_1,vsolar_1,vsolar_1,vsolar_0,vsolar_0,vsolar_0,vsolar_0
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
// Return an empty value when CUDA_VISIBLE_DEVICES is not set.
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

  const char* nic_override = blade_llm::env_nic_name();
  if (nic_override != nullptr) {
    std::string target_nic(nic_override);
    for (auto* nic_dev : nic_devs) {
      if (nic_dev->GetName().find(target_nic) != std::string::npos) {
        LOG(INFO) << "choose_nic override. gpu_dev=" << gpu_dev
                  << ", target_nic=" << target_nic
                  << ", matched=" << nic_dev->GetName();
        return nic_dev;
      }
    }
    LOG(WARNING) << "choose_nic override nic not found: " << target_nic
                 << ", falling back to affinity";
  }

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

  // XPU transfers require explicit NIC affinity.
  // XPU requires NIC affinity; without it, KV transfer may be unable to send.
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

  std::pair<XDevice*, XSimpleMempool*> get_gpu_ctx(int gpu_id, TransferProtocol::Kind kind) const;
private:
  std::mutex m_;
  std::unordered_map<int, GPUCtx> map_;
};

std::pair<XDevice*, XSimpleMempool*> MpManager::get_gpu_ctx(int gpu_id, TransferProtocol::Kind kind) const {
  auto& self = *const_cast<MpManager*>(this);  // SAFETY: we have a mutex!
  auto guard = std::lock_guard<std::mutex>(self.m_);

  auto iter = self.map_.find(gpu_id);
  if (iter != self.map_.end()) {
    return {iter->second.lovely_nic, iter->second.mp.get()};
  }

  GPUCtx ctx;
  XDeviceManager *manager = nullptr;
  if (kind == TransferProtocol::Kind::RDMA_DIRECT) {
    auto result = XDeviceManager::Singleton(manager, XDT_RDMA);
    RTASSERT(result == accl::barex::BAREX_SUCCESS);
    auto all_nic_devs = manager->AllDevices();
    RTASSERT(!all_nic_devs.empty());
    ctx.lovely_nic = choose_nic(all_nic_devs, gpu_id);
    XSimpleMempool *mempool = nullptr;
    std::string mpname = "mp-" + std::to_string(gpu_id);
    result = XSimpleMempool::NewInstance(mempool, std::move(mpname), {ctx.lovely_nic});
    RTASSERT(result == accl::barex::BAREX_SUCCESS);
    ctx.mp.reset(mempool);
    auto [iter2, ok] = self.map_.emplace(gpu_id, std::move(ctx));
    assert(ok);
    assert(iter2->second.mp.get() == mempool);
    return {iter2->second.lovely_nic, iter2->second.mp.get()};
  } else if (kind == TransferProtocol::Kind::TCP) {
    auto result = XDeviceManager::Singleton(manager, XDT_TCP);
    RTASSERT(result == accl::barex::BAREX_SUCCESS);
    auto all_nic_devs = manager->AllDevices();
    RTASSERT(!all_nic_devs.empty());
    ctx.lovely_nic = all_nic_devs[0];
    XSimpleMempool *mempool = nullptr;
    std::string mpname = "mp-" + std::to_string(gpu_id);
    result = XSimpleMempool::NewInstance(mempool, std::move(mpname), {ctx.lovely_nic});
    RTASSERT(result == accl::barex::BAREX_SUCCESS);
    ctx.mp.reset(mempool);
    auto [iter2, ok] = self.map_.emplace(gpu_id, std::move(ctx));
    assert(ok);
    assert(iter2->second.mp.get() == mempool);
    return {iter2->second.lovely_nic, iter2->second.mp.get()};
  }
  throw std::runtime_error("Unknown Transfer Protocol:" + std::to_string(kind));
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
    RTASSERT(ret >= 0);

    while (ctx->ProgressEvents() > 0) {
      // pass
    }
  }
  ctx->Shutdown();
  ctx->WaitStop();
  return;
}

static void mp_reserve(XSimpleMempool* mp) {
  std::map<uint64_t, int> reserve_map;
  const auto* reserve_vec = env_reserve();
  for (auto [size, cnt] : *reserve_vec) {
    reserve_map[size] = cnt;
    LOG(INFO) << "mp reserve: size=" << size << ";cnt=" << cnt;
  }
  if (reserve_map.empty()) {
    return;
  }
  auto ret = mp->Reserve(CPU, reserve_map);
  if (ret != accl::barex::BAREX_SUCCESS) {
    LOG(ERROR) << "mp reserve. ret=" << ret;
  }
  return;
}

BarexCtx::BarexCtx(std::string mp_name,
                   std::string tp_name,
                   int tpcnt,
                   Context *ctx,
                   std::unique_ptr<accl::barex::XChannelCallback> ctxcb,
                   TransferProtocol::Kind kind,
                   bool is_server) {
  auto &self = *this;
  // Set device_id from Context
  self.device_id_ = ctx->device_id();
  self.is_server_ = is_server;
  LOG(INFO) << "BarexCtx: device_id=" << self.device_id_;
  auto [nic_dev, mp] = g_mp_manager.get_gpu_ctx(ctx->device_id(), kind);
  self.mp_ = mp;
  mp_reserve(mp);

  const auto &layer_infos = ctx->all_layer_infos();
  // accl.barex supports at most 65536 registered MRs, so the number of
  // registered cache tensors must not exceed 65536. Since LAYER_NUM_MAX is
  // much smaller, enforcing LAYER_NUM_MAX is sufficient.
  RTASSERT(layer_infos.size() * layer_infos[0].size() <= LAYER_NUM_MAX);
  // Each MR has a default size limit of 1 GiB.
  size_t max_mr_size = 1L * 1024 * 1024 * 1024;
  const char *max_mr_gb_str = getenv("ACCL_MAX_USER_MR_GB");
  if (max_mr_gb_str != nullptr) {
    auto tmp_max_mr_gb = atoi(max_mr_gb_str);
    if (tmp_max_mr_gb > 0) {
      max_mr_size = max_mr_size * tmp_max_mr_gb;
    }
  }
  // check
  const auto &block_sizes = ctx->block_sizes();
  for (size_t i = 0; i < block_sizes.size(); i++) {
    auto layer_blk_size = block_sizes[i] * ctx->layer_num_blocks();
    LOG(INFO) << "layer size(layer_blk_size) = " << layer_blk_size
                          << ", max_mr_size = " << max_mr_size;
    // If this fails, configure ACCL_MAX_USER_MR_GB.
    RTASSERT(layer_blk_size <= max_mr_size);
  }

  self.layer_mrs_.reserve(layer_infos.size());
  self.layer_gdrcpy_mem_.reserve(layer_infos.size());
  for (const auto& layer_info : layer_infos) {
    std::vector<BarexMRGuard> layer_mrs;
    std::vector<std::unique_ptr<GdrMemDesc>> layer_gdrcpy_mem;
    layer_gdrcpy_mem.reserve(layer_info.size());
    for (const auto& info : layer_info) {
      memp_t out;
      auto layer_blk_p = reinterpret_cast<void *>(info.layer_addr);
      auto layer_blk_size = info.block_size * ctx->layer_num_blocks(); // size_t * uint32_t = size_t
      if (kind == TransferProtocol::Kind::RDMA_DIRECT){
        // RegUserMr documentation mentions alignment, but any address returned
        // by cudaMalloc is accepted. layer_blk_size of each tensor in one layer may
        // not be the same so we need to register each tensor as a separate mr
        auto result = self.mp_->RegUserMr(out, layer_blk_p, layer_blk_size, GPU,
                                          ctx->device_id());
        RTASSERT(result == accl::barex::BAREX_SUCCESS);
        RTASSERT(out.d_type == GPU);
        RTASSERT(out.device_id == ctx->device_id());
        RTASSERT(out.buf == layer_blk_p);
        RTASSERT(out.buf_len == layer_blk_size);
        LOG(INFO) << "RegUserMr. layer_blk_p=" << layer_blk_p
                  << ", layer_blk_size=" << layer_blk_size
                  << ", gpuid=" << ctx->device_id();
        if (env_crc()) { // TODO: Simplify TCP CRC by comparing the allocated CPU buffer directly?
          auto desc = gdrcpy_mmap(layer_blk_p, layer_blk_size);
          layer_gdrcpy_mem.emplace_back(std::move(desc));
        } else {
          layer_gdrcpy_mem.emplace_back(nullptr);
        }
        layer_mrs.emplace_back(BarexMRGuard::DeregGuard(std::move(out), self.mp()));
      } else if (kind == TransferProtocol::Kind::TCP) {
        out.d_type = GPU;
        out.base = (char*)layer_blk_p;
        out.buf = (char*)layer_blk_p;
        out.buf_len = layer_blk_size;
        layer_mrs.emplace_back(BarexMRGuard::ReleaseGuard(std::move(out), self.mp()));
      } else {
        throw std::runtime_error("Unsupported Transfer Protocol");
      }
    }
    self.layer_mrs_.emplace_back(std::move(layer_mrs));
    self.layer_gdrcpy_mem_.emplace_back(std::move(layer_gdrcpy_mem));
  }

  XThreadpool *threadpool = nullptr;
  auto result = XThreadpool::NewInstance(threadpool, tpcnt, std::move(tp_name));
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  self.tp_.reset(threadpool);

  XContext *context = nullptr;
  ContextConfig config = XConfigUtil::DefaultContextConfig();
  result = XContext::NewInstance(context, config, ctxcb.get(), nic_dev, self.mp(), threadpool);
  ctxcb.release();
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
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
  TransferProtocol::Kind kind) :
BarexCtx(std::move(mp_name), std::move(tp_name), tpcnt, ctx, this->get_ctx_cb(), kind, false),
  layer_blk_sizes([ctx]() {
    const auto layer_blocks = static_cast<uint64_t>(ctx->layer_num_blocks());
    auto block_sizes = ctx->block_sizes();
    std::vector<uint64_t> sizes;
    sizes.reserve(block_sizes.size());
    for (auto block_size : block_sizes) {
      sizes.emplace_back(layer_blocks * static_cast<uint64_t>(block_size));
    }
    return sizes;
  }()) {
  XConnector *connector = nullptr;
  auto result = XConnector::NewInstance(connector, env_conn_tpsize(), TIMER_3S, {this->xctx()});
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  this->connector_.reset(connector);
  return;
}

void CliBarexCtx::RpcCtxCb::OnRecvCall(accl::barex::XChannel *_ch,
  char *buf, size_t len,
  accl::barex::x_msg_header header) {
  if (len < RPC_HEADER) {
    LOG(ERROR) << "RpcCtxCb.OnRecvCall. BadMsg. len=" << len;
    return;
  }
  auto& self = *this;
  auto [magic, reqid] = deser_rpc_header(buf);
  auto on_resp = self.cli_ctx_->pop(reqid);
  if (!on_resp) {
    LOG(ERROR) << "RpcCtxCb.OnRecvCall. UnknownReqId. reqid=" << reqid << ",magic=" << magic;
    return ;
  }
  return on_resp(Status::OK(), buf, len);
}

void CliBarexCtx::push(uint64_t reqid, OnRespF on_resp) const {
  // SAFTEY: lock
  auto& self = *const_cast<CliBarexCtx*>(this);
  auto lock = std::lock_guard<std::mutex>(self.mtx_);
  auto [_, ok] = self.rpc_.emplace(reqid, std::move(on_resp));
  assert(ok);
  return ;
}

auto CliBarexCtx::pop(uint64_t reqid) const -> OnRespF {
  // SAFTEY: lock
  auto& self = *const_cast<CliBarexCtx*>(this);
  auto lock = std::lock_guard<std::mutex>(self.mtx_);
  auto iter = self.rpc_.find(reqid);
  if (iter == self.rpc_.end()) {
    // empty func
    return {};
  }
  auto handle = std::move(iter->second);
  self.rpc_.erase(iter);
  return handle;
}

void CliBarexCtx::on_send_error(accl::barex::Status s, uint64_t reqid) const {
  auto &self = *this;
  auto handle = self.pop(reqid);
  if (!handle) {
    // empty func
    LOG(ERROR) << "on_send_error. unknown reqid=" << reqid
               << ",err=" << s.ErrMsg();
    return;
  }
  return handle(std::move(s), nullptr, 0);
}

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

// Barex listen binds INADDR_ANY, so any externally reachable IP is valid.
// Follow the same selection rule as vLLM get_ip.
void get_ip(char *info_ip, size_t bufcap) {
  const auto* vllm_host_ip = getenv("VLLM_HOST_IP");
  if (vllm_host_ip != nullptr && vllm_host_ip[0] != '\0') {
    auto last_idx = bufcap - 1;
    strncpy(info_ip, vllm_host_ip, bufcap);
    info_ip[last_idx] = '\0';
    return;
  }

  struct sockaddr_in remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(80);
  int status = ::inet_pton(AF_INET, "8.8.8.8", &remote_addr.sin_addr);
  RTASSERT(status == 1);

  struct sockaddr_in local_addr;
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  RTASSERT(fd >= 0);
  status = ::connect(
      fd,
      reinterpret_cast<const struct sockaddr*>(&remote_addr),
      sizeof(remote_addr));
  RTASSERT(status == 0);
  socklen_t local_addr_len = sizeof(local_addr);
  memset(&local_addr, 0, sizeof(local_addr));
  status = ::getsockname(
      fd,
      reinterpret_cast<struct sockaddr*>(&local_addr),
      &local_addr_len);
  ::close(fd);
  RTASSERT(status == 0);
  RTASSERT(local_addr.sin_family == AF_INET);

  const char* ret = ::inet_ntop(
      AF_INET, &local_addr.sin_addr, info_ip, bufcap);
  RTASSERT(ret != nullptr);
}

[[nodiscard]] std::future<BarexChannel>
Connect(XConnector &self, std::string server_addr, int port) {
  // std::promise<XChannel*> pr;
  auto pr = std::make_shared<std::promise<BarexChannel>>();
  auto fut = pr->get_future();

  auto result = self.Connect(std::move(server_addr), port,
    [pr = std::move(pr), conn=&self](XChannel *res, Status s) mutable {
      if (!s.IsOk()) {
        auto ex = std::make_exception_ptr(std::runtime_error("Connect ERR: " + s.ErrMsg()));
        pr->set_exception(std::move(ex));
        return;
      }
      std::shared_ptr<XChannel> shared_ch;
      auto* ctx = res->GetContext();
      bool ret = ctx->GetSharedChannel(shared_ch, res);
      if (!ret) {
        std::stringstream ss;
        ss << "Connect GetSharedError; ctx=" << ctx << ";ch=" << res
          << ";conn=" << conn;
        auto ex = std::make_exception_ptr(std::runtime_error(std::move(ss).str()));
        pr->set_exception(std::move(ex));
        return;
      }
      pr->set_value(BarexChannel(conn, std::move(shared_ch)));
    }
  );

  if (result != BAREX_SUCCESS) {
    auto ex =
        std::runtime_error("Connect Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<BarexChannel>(std::move(ex));
  }
  return fut;
}

void delete_channels(CliBarexCtx* ctx, std::vector<BarexChannel> chs) {
  auto& tp = ctx->close_tp();
  tp.spawn([chs=std::move(chs)] () {
    // Destroy chs on a background thread.
  });
  assert(chs.empty());
  return;
}

bool valid_channels(std::vector<BarexChannel>& chs) {
  if (chs.empty()) {
    return false;
  }
  for (auto& ch : chs) {
    if (!ch.ch()->IsActive()) {
      return false;
    }
  }
  return true;
}

bool BarexProtoContext::check_support() {
  try {
    XDeviceManager *manager = nullptr;
    if (kind == TransferProtocol::Kind::RDMA_DIRECT) {
      auto result = XDeviceManager::Singleton(manager, XDT_RDMA);
      if (result != accl::barex::BAREX_SUCCESS) {
        return false;
      }
    }
    else if (kind == TransferProtocol::Kind::TCP) {
      auto result = XDeviceManager::Singleton(manager, XDT_TCP);
      if (result != accl::barex::BAREX_SUCCESS) {
        return false;
      }
    } else {
      return false;
    }
    auto all_nic_devs = manager->AllDevices();
    if (all_nic_devs.empty()) {
      return false;
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "BarexProtoContext::check_support: " << e.what()
               << " return false;";
    return false;
  }
  return true;
}
void BarexProtoContext::init(Context *ctx) {
  if (is_server) {
    barex_ctx_ = std::make_unique<BarexCtx>(
      name_prefix + "mp",
      name_prefix + "tp",
      env_ctx_tpsize(),
      ctx,
      std::move(callback_),
      kind,
      true
    );
  } else {
    cli_barex_ctx_ = std::make_unique<CliBarexCtx>(
      name_prefix + "mp",
      name_prefix + "tp",
      env_ctx_tpsize(),
      ctx,
      kind
    );
  }
}
std::unique_ptr<BarexProtoContext> BarexProtoContext::server_context(
  std::string &&name,
  std::unique_ptr<accl::barex::XChannelCallback> &&bk,
  TransferProtocol::Kind kind) {
  return std::make_unique<BarexProtoContext>(std::move(name), true, std::move(bk), kind);
}
std::unique_ptr<BarexProtoContext> BarexProtoContext::client_context(std::string &&name, TransferProtocol::Kind kind) {
  return std::make_unique<BarexProtoContext>(std::move(name), false, nullptr, kind);
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
    // DataPtrCtx construction is rare, so logging here is acceptable.
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
    // DataPtrCtx destruction is rare, so logging here is acceptable.
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
  RTASSERT(THPDevice_Check(device));
  auto* dev = reinterpret_cast<THPDevice*>(device);
  RTASSERT(dev->device.type() == c10::DeviceType::CUDA);
  RTASSERT(dev->device.has_index());
  int gpu_id = dev->device.index();
  XAllocator* gpu_allocator = nullptr;
  // Use RDMA_DIRECT as default for memory allocation
  auto [_, mp] = g_mp_manager.get_gpu_ctx(gpu_id, TransferProtocol::Kind::RDMA_DIRECT);
  auto result = mp->GetXAllocator(gpu_allocator, GPU);
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  // cudaMalloc is at least 256-byte aligned. align is ineffective on PPU,
  // where KV cache alignment is not guaranteed.
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

} // namespace blade_llm
