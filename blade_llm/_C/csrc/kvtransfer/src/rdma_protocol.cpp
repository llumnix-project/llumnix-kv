
#ifdef ENABLE_RDMA
#include "protocol/rdma_protocol.h"
#include "thrid_party/logging.h"
#include "naming.h"
#include "assert.h"
#include <stdlib.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <future>
#include <sstream>

// just in this cpp
using namespace accl::barex;

namespace blade_llm {

#define RTCHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "Runtime error: Assertion failed in %s on line %d: %s\n", __FILE__, __LINE__, #expr); \
        abort(); \
    } \
} while (0)

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

auto BarexCtx::choose_nic(const std::vector<XDevice *> &nic_devs, int gpu_dev) -> XDevice * {
  // TODO(zhanyi.ww): 选择离 gpu_dev 最近的网卡.
  assert(gpu_dev >= 0);
  assert(!nic_devs.empty());
  return gpu_dev >= nic_devs.size() ? nic_devs[0] : nic_devs[gpu_dev];
}

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
  XDeviceManager *manager = nullptr;
  auto result = XDeviceManager::Singleton(manager);
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  auto all_nic_devs = manager->AllDevices();
  RTCHECK(!all_nic_devs.empty());
  auto *nic_dev = choose_nic(all_nic_devs, ctx->device_id());

  XSimpleMempool *mempool = nullptr;
  result = XSimpleMempool::NewInstance(mempool, std::move(mp_name), {nic_dev});
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  self.mp_.reset(mempool);

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
      max_mr_size = max_mr_size * tmp_max_mr_gb ;
    }
  } else {
    setenv("ACCL_MAX_USER_MR_GB", "2", 1);
    max_mr_size = max_mr_size * 2;
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
    result = self.mp_->RegUserMr(out, layer_blk_p, layer_blk_size, GPU);
    RTCHECK(result == accl::barex::BAREX_SUCCESS);
    self.layer_mr_.emplace_back(BarexMRGuard::DeregGuard(std::move(out), self.mp_.get()));
  }

  XThreadpool *threadpool = nullptr;
  result = XThreadpool::NewInstance(threadpool, tpcnt, std::move(tp_name));
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  self.tp_.reset(threadpool);

  XContext *context = nullptr;
  ContextConfig config = XConfigUtil::DefaultContextConfig();
  result = XContext::NewInstance(context, config, ctxcb.get(), nic_dev, mempool, threadpool);
  ctxcb.release();
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  self.xctx_.reset(context);

  self.ctx_loop_thd_.emplace([context, this]() {
    BarexCtxMain(context, &this->stopped_);
  });
}

BarexCtx::~BarexCtx() {
  this->stopped_.store(true);
  this->ctx_loop_thd_->join();
}

CliBarexCtx::CliBarexCtx(std::string mp_name,
                         std::string tp_name,
                         int tpcnt,
                         Context *ctx,
                         std::unique_ptr<accl::barex::XChannelCallback> ctxcb) :
    BarexCtx(std::move(mp_name), std::move(tp_name), tpcnt, ctx, std::move(ctxcb)),
    layer_blk_size(ctx->layer_num_blocks() * ctx->block_size()) {
  XConnector *connector = nullptr;
  // 处理建联/断链的线程个数, 2 来自 barex write-client example~
  constexpr int CONN_THD_CNT = 2;
  auto result = XConnector::NewInstance(connector, CONN_THD_CNT, TIMER_3S, {this->xctx()});
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  this->connector_.reset(connector);
  return;
}

static constexpr uint32_t SEND_MAGIC = 0x53456e64;  /* SEnd */
//
// magic, inst_id, worker_id, num_block 都是 4 字节.
// num_block 指定了 block_ids 中 block 个数, 每个 block 4 字节.
// reqid 以 0 结尾的 C 字符串. reqid 好像固定 32 字节但不确认..
// +-------+---------+-----------+-----------+-----------+-------+
// | magic | inst_id | worker_id | num_block | block_ids | reqid |
//
void RDMAServer::CtxCallback::OnRecvCall(XChannel *_ch, char *in_buf, size_t len, x_msg_header _header) {
  if (len < sizeof(uint32_t) * 4) {
    // bad data, ignore.
    return;
  }
  const char *const end_buf = in_buf + len;
  InstanceId inst_id;
  uint32_t magic, worker_id, num_block;
  memcpy(&magic, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  memcpy(&inst_id, in_buf, sizeof(InstanceId));
  in_buf += sizeof(InstanceId);
  memcpy(&worker_id, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  memcpy(&num_block, in_buf, sizeof(uint32_t));
  in_buf += sizeof(uint32_t);
  if (magic != SEND_MAGIC) {
    // bad data, ignore
    return;
  }
  if (len < sizeof(uint32_t) * (4 + num_block) + 1) {
    // +1 for reqid null terminator
    // bad data, ignore.
    return;
  }

  std::vector<uint32_t> block_ids(num_block);
  memcpy(block_ids.data(), in_buf, num_block * sizeof(uint32_t));
  in_buf += num_block * sizeof(uint32_t);

  const char *reqid = in_buf;
  size_t reqid_len = end_buf - in_buf;
  assert(reqid_len >= 1);
  reqid_len -= 1;
  if (reqid[reqid_len] != '\0') {
    // bad data, ignore
    return;
  }
  std::string reqidstr(reqid, reqid_len);

  // OnRecvCall 在 barex 线程池中调用. 注意 data race.
  this->ser_->on_recv(inst_id, worker_id, reqidstr, std::move(block_ids));
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
  auto rdma_ctx = RDMAProtoContext::server_context("KVTServer", 4, std::make_unique<CtxCallback>(service));
  if (!rdma_ctx->check_support()) {
    throw std::runtime_error("can't start RDMA transfer server as RDMA protocol not support;");
  }
  ctx->register_protocol(std::move(rdma_ctx));

  WorkerInfo *winfo = ctx->worker_info_mutable();
  auto layer_num_blocks = ctx->layer_num_blocks();
  const uint64_t layer_blk_size = layer_num_blocks * winfo->block_size;
  auto layer_ptr = ctx->layer_data_address();
  auto proto = TransferProtocol::rdma_direct();
  auto proto_ctx = ctx->get_protocol_ctx<RDMAProtoContext>(proto);
  if (proto_ctx == nullptr) {
    throw std::runtime_error("KVT server: rdma context not register.");
  }
  auto barex_ctx = proto_ctx->barex_ctx();
  std::vector<void*> ptrs(layer_ptr.size());
  std::vector<uint32_t> rkeys(layer_ptr.size());

  for (int idx = 0; idx < layer_ptr.size(); ++idx) {
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

  // 处理建联/断链的线程个数, 2 来自 barex write-client example~
  constexpr int LISTEN_THD_CNT = 2;
  XListener *listener = nullptr;
  auto xctx = barex_ctx->xctx();
  if (xctx == nullptr) {
    LOG(ERROR) << "xctx is nullptr";
  }
  auto result = XListener::NewInstance(listener, LISTEN_THD_CNT, info.port, TIMER_3S, {xctx});
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

static int get_send_parallel() {
  const char *valstr = getenv("BLLM_KVTRANS_RDMA_SP");
  if (valstr == nullptr) {
    return 1;
  }
  int ret = atoi(valstr);
  if (ret == 0) {
    return 1;
  }
  return ret;
}

void RDMAChannel::do_init() {
  auto &self = *this;
  if (!self.chs_.empty()) {
    return;
  }
  const int sp = get_send_parallel();
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

[[nodiscard]] static std::future<void> WriteBatch(XChannel *ch, std::shared_ptr<std::vector<rw_memp_t>> datasp) {
  // std::promise<void> pr;
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();
  auto &datas = *datasp;

  auto result = ch->WriteBatch(datas,
                               [pr = std::move(pr), d = std::move(datasp)](Status s) mutable {
                                 // WriteBatch 要求 datasp 直至 callback 中才能回收.
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

accl::barex::XChannel *RDMAChannel::ch() noexcept {
  auto &self = *this;
  int n = self.chs_.size();
  int idx = (++self.prev_ch_idx_) % n;
  return self.chs_[idx];
}

void RDMAChannel::send_data(size_t layer_idx, const std::vector<IpcBlock> &data) {
  auto &self = *this;
  self.do_init();
  auto datasp = std::make_shared<std::vector<rw_memp_t>>();
  for (const auto &[src_offset, dst_offset, len] : data) {
    if (len <= 0) {
      continue;
    }
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
    datasp->emplace_back(rw_memp_t{std::move(src_mr), raddr, rkey});
  }
  if (datasp->empty()) {
    return;
  }

  auto fut = WriteBatch(self.ch(), std::move(datasp));
  self.write_futs_.emplace_back(std::move(fut));
  return;
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

void RDMAChannel::flush() {
  auto &self = *this;
  std::vector<std::future<void>> futs;
  std::swap(futs, self.write_futs_);
  when_all_succeed(futs);
  return;
}

static void Encode(char *ptr, InstanceId inst_id, uint32_t worker_id,
                   const std::string &reqid, const std::vector<uint32_t> &block_ids) {
  const uint32_t num_block = block_ids.size();
  const uint32_t magic = SEND_MAGIC;

  memcpy(ptr, &magic, sizeof(magic));
  ptr += sizeof(magic);
  memcpy(ptr, &inst_id, sizeof(inst_id));
  ptr += sizeof(inst_id);
  memcpy(ptr, &worker_id, sizeof(worker_id));
  ptr += sizeof(worker_id);
  memcpy(ptr, &num_block, sizeof(num_block));
  ptr += sizeof(num_block);

  memcpy(ptr, block_ids.data(), num_block * sizeof(uint32_t));
  ptr += num_block * sizeof(uint32_t);

  // *(s.begin() + s.size()) has value CharT() (a null terminator)
  memcpy(ptr, reqid.data(), reqid.size() + 1);
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

  auto result = ch->Send(std::move(sdata), /* auto_release */ true, {0},
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

void RDMAChannel::send_notification(IIterator<const RequestInfo *> *reqs) {
  auto &self = *this;
  self.do_init();
  // self.send_futs_.reserve(data.size());
  assert(self.send_futs_.empty());

  auto opt = reqs->next();
  while (opt.has_value()) {
    auto r = opt.value();
    const auto &reqid = r->req_id;
    const auto &block_ids = r->dst_blocks();
    LOG(INFO) << "KVT: send notification of request " << reqid;
    // 编码规则见 RDMAServer::CtxCallback::OnRecvCall
    memp_t bufmr;
    size_t const msglen = sizeof(InstanceId) + 3 * sizeof(uint32_t) + block_ids.size() * sizeof(uint32_t) + reqid.size() + 1;
    auto result = self.ctx_->mp()->AllocBuffer(bufmr, msglen, CPU);
    RTCHECK(result == accl::barex::BAREX_SUCCESS);
    Encode(bufmr.buf, self.src_inst_id_, self.src_worker_id_, reqid, block_ids);

    auto *use_ch = self.ch();
    assert(use_ch->GetMempool() == self.ctx_->mp());
    auto fut = Send(use_ch, std::move(bufmr));
    self.send_futs_.emplace_back(std::move(fut));
    opt = reqs->next();
  }

  when_all_succeed(self.send_futs_);
  self.send_futs_.clear();
  return;
}

void RDMAChannel::do_notify_send_done(const std::string &reqid,
                                      const std::vector<uint32_t> &block_ids) {
  auto &self = *this;
  self.do_init();

  // 编码规则见 RDMAServer::CtxCallback::OnRecvCall
  memp_t bufmr;
  size_t const msglen = 4 * sizeof(uint32_t) + block_ids.size() * sizeof(uint32_t) + reqid.size() + 1;
  auto result = self.ctx_->mp()->AllocBuffer(bufmr, msglen, CPU);
  RTCHECK(result == accl::barex::BAREX_SUCCESS);
  auto mrguard = BarexMRGuard::RelDeregGuard(std::move(bufmr), self.ctx_->mp());

  Encode(mrguard.mr().buf, self.src_inst_id_, self.src_worker_id_, reqid, block_ids);

  // LOG(INFO) << "zydebug notify_send_done reqid=" << reqid <<  " block_ids.size=" << block_ids.size();
  Send(self.ch(), mrguard.mr()).get();
  // LOG(INFO) << "zydebug notify_send_done done reqid=" << reqid <<  " block_ids.size=" << block_ids.size();
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
    barex_ctx_ = std::make_unique<BarexCtx>(name_prefix + "mp", name_prefix + "tp", 4, ctx, std::move(callback_));
  } else {
    cli_barex_ctx_ =
        std::make_unique<CliBarexCtx>(name_prefix + "mp", name_prefix + "tp", 4, ctx, std::move(callback_));
  }
}
std::unique_ptr<RDMAProtoContext> RDMAProtoContext::server_context(std::string &&name,
                                                                   int num_threads,
                                                                   std::unique_ptr<accl::barex::XChannelCallback> &&bk) {
  return std::make_unique<RDMAProtoContext>(std::move(name), num_threads, true, std::move(bk));
}
std::unique_ptr<RDMAProtoContext> RDMAProtoContext::client_context(std::string &&name, int num_threads) {
  return std::make_unique<RDMAProtoContext>(std::move(name), num_threads, false, std::make_unique<CliCtxCallback>());
}

}  // namespace blade_llm {
#endif // ENABLE_RDMA