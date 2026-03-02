
#include <sys/epoll.h>
#include <future>
#include <sstream>
#include "protocol/rdma_protocol.h"
#include "protocol/tcp_channel.h"
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
#include "copy_kernels.h"
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

// Thread-local context for kernel copy operations
struct CopyKernelCtx : public noncopyable {
  // GPU buffer for kernel copy metadata (src_offsets, dst_offsets, lengths)
  // Size: 3 * sizeof(int64_t) * env_kernel_copy_max_block_num()
  char* device_blk_buffer = nullptr; // on device
  char* host_blk_buffer = nullptr; // on host
  int device_id = -1;
  bool initialized = false;
  
  cudaStream_t h2d_stream = nullptr;
  cudaStream_t d2h_stream = nullptr;

  CopyKernelCtx() = default;

  CopyKernelCtx(const CopyKernelCtx&) = delete;
  CopyKernelCtx& operator=(const CopyKernelCtx&) = delete;
  CopyKernelCtx(CopyKernelCtx&&) = delete;
  CopyKernelCtx& operator=(CopyKernelCtx&&) = delete;

  // Initialize buffer for the given device
  cudaError_t init(int dev_id) {
    if (initialized && this->device_id == dev_id) {
      return cudaSuccess;
    }

    if (device_blk_buffer != nullptr) {
      cudaFree(device_blk_buffer);
      device_blk_buffer = nullptr;
    }
    if (host_blk_buffer != nullptr) {
      cudaFreeHost(host_blk_buffer);
      host_blk_buffer = nullptr;
    }

    auto buffer_num = 3;
    size_t buffer_size = buffer_num * sizeof(int64_t) * env_kernel_copy_max_block_num();

    cudaError_t err = cudaSetDevice(dev_id);
    RTASSERT(err == cudaSuccess);

    err = cudaMalloc(&device_blk_buffer, buffer_size);
    RTASSERT(err == cudaSuccess);
    err = cudaMallocHost(&host_blk_buffer, buffer_size);
    RTASSERT(err == cudaSuccess);

    LOG(INFO) << "CopyKernelCtx: initialized for device " << dev_id
              << ", block num=" << env_kernel_copy_max_block_num()
              << ", bf162fp8_conversion=" << (env_bf162fp8_conversion() ? "true" : "false")
              << ", buffer_num=" << buffer_num
              << ", buffer_size=" << buffer_size;

    this->device_id = dev_id;
    initialized = true;
    return cudaSuccess;
  }

  std::pair<char*, char*> get_buffers() const {
    return std::make_pair(device_blk_buffer, host_blk_buffer);
  }

  cudaStream_t get_h2d_stream() {
    if (h2d_stream == nullptr) {
      auto cuda_rt = cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking);
      RTCHECK(cuda_rt == cudaSuccess);
    }
    return h2d_stream;
  }

  void destroy_h2d_stream() {
    if (h2d_stream != nullptr) {
      cudaError_t err = cudaStreamDestroy(h2d_stream);
      if (err != cudaSuccess) {
        LOG(ERROR) << "CopyKernelCtx: Failed to destroy h2d_stream: " << cudaGetErrorString(err);
      }
      h2d_stream = nullptr;
    }
  }

  cudaStream_t get_d2h_stream() {
    if (d2h_stream == nullptr) {
      auto cuda_rt = cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking);
      RTCHECK(cuda_rt == cudaSuccess);
    }
    return d2h_stream;
  }

  void destroy_d2h_stream() {
    if (d2h_stream != nullptr) {
      cudaError_t err = cudaStreamDestroy(d2h_stream);
      if (err != cudaSuccess) {
        LOG(ERROR) << "CopyKernelCtx: Failed to destroy d2h_stream: " << cudaGetErrorString(err);
      }
      d2h_stream = nullptr;
    }
  }

  ~CopyKernelCtx() {
    if (device_blk_buffer != nullptr) {
      cudaFree(device_blk_buffer);
      device_blk_buffer = nullptr;
    }
    if (host_blk_buffer != nullptr) {
      cudaFreeHost(host_blk_buffer);
      host_blk_buffer = nullptr;
    }
    destroy_h2d_stream();
    destroy_d2h_stream();
  }
};

// Thread-local context instance
// each thread from barex thread pool will have its own context
thread_local CopyKernelCtx g_copy_kernel_ctx;

std::pair<char*, char*> get_kernel_copy_buffer(int device_id) {
  cudaError_t err = g_copy_kernel_ctx.init(device_id);
  if (err != cudaSuccess) {
    LOG(ERROR) << "get_kernel_copy_buffer: failed to initialize buffer for device " << device_id
                << ", error=" << cudaGetErrorString(err);
    return std::make_pair(nullptr, nullptr);
  }
  return g_copy_kernel_ctx.get_buffers();
}

cudaStream_t TCPServer::get_h2d_stream() {
  return g_copy_kernel_ctx.get_h2d_stream();
}

// rpc req/resp
// +-------+-------+------------+
// | magic | reqid |  rpc body  |
// rpc header: magic: 4 bytes. reqid: 8 bytes.
// rpc body: req/resp body is decoded according to magic.
static constexpr size_t RPC_HEADER = sizeof(uint32_t) + sizeof(uint64_t);
static constexpr uint32_t MEM_HANDLES_REQ_MAGIC = 0x20181218;

static std::pair<uint32_t, uint64_t> deser_rpc_header(const char* buf) noexcept {
  uint32_t magic;
  memcpy(&magic, buf, sizeof(magic));
  uint64_t reqid;
  memcpy(&reqid, buf + sizeof(magic), sizeof(reqid));
  return std::make_pair(magic, reqid);
}

static void ser_rpc_header(char* buf, uint32_t magic, uint64_t reqid) noexcept {
  memcpy(buf, &magic, sizeof(magic));
  memcpy(buf + sizeof(magic), &reqid, sizeof(reqid));
  return;
}

// req:
// +--------+---------+--------+--------+--------+--------+
//    crc       cnt       off1     len1    off2     len2
// crc: uint32_t.
// cnt: uint32_t specifying the number of subsequent off/len pairs.
// off, len; uint64_t.
// resp:
// crc: uint32_t
static constexpr uint32_t REMOTE_CRC_REQ_MAGIC = 0x20250924;

static constexpr uint32_t SEND_MAGIC = 0x53456e64; /* SEnd */

static constexpr uint32_t KV_CACHE_DATA_MAGIC = 0x4B564361;  /* KVCa (KV Cache data) */

// magic, inst_id_len, worker_id, num_block are all 4 bytes.
// num_block specifies the number of blocks in block_ids, each block is 4 bytes.
// reqid is a null-terminated C string.
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

template<typename T, typename E>
static std::future<T> make_exp_future(E ex) {
  std::promise<T> pr;
  pr.set_exception(std::make_exception_ptr(std::move(ex)));
  return pr.get_future();
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
static void Send(std::shared_ptr<XChannel>& ch, memp_t sdata, DoneCallback cb) {
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
[[nodiscard]] static std::future<void> Send(std::shared_ptr<XChannel>& ch, memp_t sdata) {
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

static memp_t AllocCPUBuffer(std::shared_ptr<XChannel>& ch, uint64_t size) {
  memp_t bufmr;
  auto result = ch->AllocBuffer(bufmr, size, CPU);
  RTCHECK_EQ(result, accl::barex::BAREX_SUCCESS);
  return bufmr;
}

static memp_t AllocCudaHostBuffer(XChannel* ch, uint64_t size) {
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

// GPU 0 maps to NIC: RET[0]
// filename format: vsolar_1,vsolar_1,vsolar_1,vsolar_1,vsolar_0,vsolar_0,vsolar_0,vsolar_0
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
// Returns empty if CUDA_VISIBLE_DEVICES is not set.
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
  // accl.barex allows registering at most 65536 MRs, i.e. cache tensor count <= 65536.
  // Since LAYER_NUM_MAX is much smaller than 65536, the LAYER_NUM_MAX limit suffices.
  RTASSERT(layer_infos.size() * layer_infos[0].size() <= LAYER_NUM_MAX);
  // Default max MR size is 1GB.
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
    // If this fails, set the ACCL_MAX_USER_MR_GB environment variable
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
        // Although the docs say RegUserMr requires alignment, it was confirmed that
        // any address returned by cudaMalloc is acceptable. llx: layer_blk_size of each tensor in one layer may
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
        if (env_crc()) { // TODO: CRC for TCP could be simplified? Compare directly using allocated cpu buffer?
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

  // Initialize thread-local kernel copy buffer for all threads in the thread pool
  // Only for Decode node using TCP protocol
  // prefill node will init at target_thdpool_
  if (kind == TransferProtocol::Kind::TCP && is_server) {
    int thread_pool_size = tpcnt;
    // Submit one initialization task per thread to ensure each thread initializes its buffer
    // Each task uses a different thread_hint (0 to thread_pool_size-1) to distribute across threads
    for (int i = 0; i < thread_pool_size; ++i) {
      int device_id_val = self.device_id_;
      threadpool->Submit([device_id_val]() {
        get_kernel_copy_buffer(device_id_val);
      }, i);
    }
    LOG(INFO) << "BarexCtx: submitted kernel copy buffer initialization tasks for " 
              << thread_pool_size << " thread pool threads (TCP protocol)";
  }
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

struct GetMemHandles {
private:
  using Promise = std::promise<std::vector<RDMAMemHandle>>;
private:
  std::shared_ptr<XChannel> const ch_ = nullptr;
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;

public:
  GetMemHandles(std::shared_ptr<XChannel> ch, uint64_t r) noexcept:
    ch_(std::move(ch)),
    reqid_(r),
    pr_(std::make_shared<Promise>()) {}

  auto future() {
    return this->pr_->get_future();
  }

  void operator()(Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      std::stringstream ss;
      ss << "GetMemHandles error: ex=" << s.ErrMsg() << " reqid=" << self.reqid_ << " ch=" << self.ch_->ToString();
      auto ex = std::make_exception_ptr(std::runtime_error(std::move(ss).str()));
      self.pr_->set_exception(std::move(ex));
      return ;
    }
    RTASSERT(len >= RPC_HEADER);
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == MEM_HANDLES_REQ_MAGIC);
    RTASSERT(reqid == self.reqid_);

    buf += RPC_HEADER;
    len -= RPC_HEADER;
    RTASSERT(len % sizeof(RDMAMemHandle) == 0);
    size_t handle_n = len / sizeof(RDMAMemHandle);
    auto vec = std::vector<RDMAMemHandle>(handle_n);
    memcpy(vec.data(), buf, handle_n * sizeof(RDMAMemHandle));
    self.pr_->set_value(std::move(vec));
    return ;
  }
};

struct SendKVCacheData {
  using Promise = std::promise<TCPTimePoints>;
private:
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;
  std::chrono::system_clock::time_point send_data_start_ts_;
  std::chrono::system_clock::time_point d2h_start_ts_;
  std::chrono::system_clock::time_point d2h_end_ts_;
  uint64_t recv_start_;
  uint32_t recv_time_;
  uint32_t onrecv_queue_us_;
public:
  SendKVCacheData(
    uint64_t r, std::shared_ptr<Promise> pr, 
    std::chrono::system_clock::time_point send_data_start_ts, 
    std::chrono::system_clock::time_point d2h_start_ts, 
    std::chrono::system_clock::time_point d2h_end_ts) noexcept:
    reqid_(r),
    pr_(std::move(pr)),
    send_data_start_ts_(send_data_start_ts),
    d2h_start_ts_(d2h_start_ts),
    d2h_end_ts_(d2h_end_ts) {}

  void operator()(Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      auto ex = std::make_exception_ptr(std::runtime_error("SendKVCacheData ERR: " + s.ErrMsg()));
      self.pr_->set_exception(std::move(ex));
      return ;
    }
    RTASSERT(len >= RPC_HEADER);
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == KV_CACHE_DATA_MAGIC);
    RTASSERT(reqid == self.reqid_);
    // Server sends h2d_start_ts and h2d_end_ts in the response
    // Response format: [RPC_HEADER] [h2d_start_ts (time_point)] [h2d_end_ts (time_point)]
    std::chrono::system_clock::time_point h2d_start_ts;
    std::chrono::system_clock::time_point h2d_end_ts;
    uint64_t recv_start;
    uint32_t recv_time;
    uint32_t onrecv_queue_us;
    if (len >= RPC_HEADER + sizeof(h2d_start_ts) + sizeof(h2d_end_ts)) {
      auto buf_ptr = buf + RPC_HEADER;
      memcpy(&recv_start, buf_ptr, sizeof(recv_start));
      buf_ptr += sizeof(recv_start);
      memcpy(&recv_time, buf_ptr, sizeof(recv_time));
      buf_ptr += sizeof(recv_time);
      memcpy(&onrecv_queue_us, buf_ptr, sizeof(onrecv_queue_us));
      buf_ptr += sizeof(onrecv_queue_us);
      memcpy(&h2d_start_ts, buf_ptr, sizeof(h2d_start_ts));
      buf_ptr += sizeof(h2d_start_ts);
      memcpy(&h2d_end_ts, buf_ptr, sizeof(h2d_end_ts));
    } else {
      auto ex = std::make_exception_ptr(std::runtime_error("SendKVCacheData ERR: No h2d_start_ts or h2d_end_ts in response"));
      self.pr_->set_exception(std::move(ex));
      return ;
    }
    TCPTimePoints time_points;
    time_points.send_data_start_ts_ = self.send_data_start_ts_;
    time_points.d2h_start_ts_ = self.d2h_start_ts_;
    time_points.d2h_end_ts_ = self.d2h_end_ts_;
    time_points.h2d_start_ts = h2d_start_ts;
    time_points.h2d_end_ts = h2d_end_ts;
    time_points.recv_start_ = recv_start;
    time_points.recv_time_ = recv_time;
    time_points.onrecv_queue_us_ = onrecv_queue_us;
  
    self.pr_->set_value(std::move(time_points));
    return ;
  }
};

std::vector<RDMAMemHandle> CliBarexCtx::get_mem_handles(std::shared_ptr<XChannel>& dst) const {
  auto& self = *this;
  uint32_t magic = MEM_HANDLES_REQ_MAGIC;
  uint64_t reqid = new_id();
  memp_t bufmr = AllocCPUBuffer(dst, sizeof(magic) + sizeof(reqid));
  ser_rpc_header(bufmr.buf, magic, reqid);

  auto memhandles = GetMemHandles(dst, reqid);
  auto fut = memhandles.future();
  self.push(reqid, std::move(memhandles));

  Send(dst, std::move(bufmr), [this, reqid](Status s) {
    // this owner: KV_CLIENT.ctx_
    if (s.IsOk()) {
      return;
    }
    this->on_send_error(std::move(s), reqid);
  });

  auto futstate = fut.wait_for(std::chrono::seconds(env_rpc_timeout_s()));
  if (futstate != std::future_status::ready) {
    this->on_send_error(Status(BAREX_ERR_TIMEOUT), reqid);
  }
  return fut.get();
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


struct RemoteCrc {
private:
  using Promise = std::promise<uint32_t>;
private:
  std::shared_ptr<XChannel> const ch_ = nullptr;
  uint64_t const reqid_;
  std::shared_ptr<Promise> pr_;

public:
  RemoteCrc(std::shared_ptr<XChannel> ch, uint64_t r) noexcept:
    ch_(std::move(ch)),
    reqid_(r),
    pr_(std::make_shared<Promise>()) {}

  auto future() {
    return this->pr_->get_future();
  }

  void operator()(Status s, char* buf, size_t len) {
    auto& self = *this;
    if (!s.IsOk()) {
      std::stringstream ss;
      ss << "RemoteCrc error: ex=" << s.ErrMsg() << " reqid=" << self.reqid_ << " ch=" << self.ch_->ToString();
      auto ex = std::make_exception_ptr(std::runtime_error(std::move(ss).str()));
      self.pr_->set_exception(std::move(ex));
      return ;
    }
    RTASSERT(len >= RPC_HEADER);
    auto [magic, reqid] = deser_rpc_header(buf);
    RTASSERT(magic == REMOTE_CRC_REQ_MAGIC);
    RTASSERT(reqid == self.reqid_);

    buf += RPC_HEADER;
    len -= RPC_HEADER;
    RTASSERT(len >= sizeof(uint32_t));
    uint32_t crc = 0;
    memcpy(&crc, buf, sizeof(uint32_t));
    self.pr_->set_value(crc);
    return;
  }
};

uint32_t CliBarexCtx::get_remote_crc(std::shared_ptr<XChannel>& dst, const std::vector<std::vector<IpcBlock>>* data, uint32_t lcrc) {
  assert(!data->empty());
  auto& self = *this;
  uint64_t const reqid = new_id();

  // calculate total size: tensor count + block count per tensor + all offset/length pairs
  uint32_t tensor_cnt = static_cast<uint32_t>(data->size());
  uint32_t total_blocks = 0;
  for (const auto& per_tensor_data : *data) {
    total_blocks += static_cast<uint32_t>(per_tensor_data.size());
  }

  // protocol format: 
  // (magic + reqid) + lcrc + tensor_cnt 
  // + [tensor0_block_cnt + off1+len1 + off2+len2 + ...] + [tensor1_block_cnt + ...] + ...
  const auto bodysize = sizeof(uint32_t) + // tensor_cnt
                        tensor_cnt * sizeof(uint32_t) + // block count per tensor
                        total_blocks * (sizeof(uint64_t) + sizeof(uint64_t)); // all offset/length pairs
  memp_t bufmr = AllocCPUBuffer(dst, RPC_HEADER + sizeof(uint32_t) + bodysize);
  ser_rpc_header(bufmr.buf, REMOTE_CRC_REQ_MAGIC, reqid);
  char* bufstart = bufmr.buf + RPC_HEADER;
  const char* const bufend = bufmr.buf + bufmr.buf_len;
  memcpy(bufstart, &lcrc, sizeof(uint32_t));
  bufstart += sizeof(uint32_t);
  memcpy(bufstart, &tensor_cnt, sizeof(uint32_t));
  bufstart += sizeof(uint32_t);

  for (const auto& per_tensor_data : *data) {
    uint32_t block_cnt = static_cast<uint32_t>(per_tensor_data.size());
    memcpy(bufstart, &block_cnt, sizeof(uint32_t));
    bufstart += sizeof(uint32_t);

    for (const auto& sb : per_tensor_data) {
      RTASSERT((bufend - bufstart) >= 16 /* sizeof(uint64_t) * 2 */);
      memcpy(bufstart, &sb.dst_offset, sizeof(uint64_t));
      bufstart += sizeof(uint64_t);
      memcpy(bufstart, &sb.length, sizeof(uint64_t));
      bufstart += sizeof(uint64_t);
    }
  }

  auto req = RemoteCrc(dst, reqid);
  auto fut = req.future();
  self.push(reqid, std::move(req));

  Send(dst, std::move(bufmr), [this, reqid](Status s) {
    // this owner: KV_CLIENT.ctx_
    if (s.IsOk()) {
      return;
    }
    this->on_send_error(std::move(s), reqid);
  });

  auto futstate = fut.wait_for(std::chrono::seconds(env_rpc_timeout_s()));
  if (futstate != std::future_status::ready) {
    this->on_send_error(Status(BAREX_ERR_TIMEOUT), reqid);
  }
  return fut.get();
}

void RDMAServer::CtxCallback::resp_remote_crc(std::shared_ptr<XChannel>& channel, uint64_t reqid, char *inbuf, size_t inlen) {
  const auto tp1 = SteadyClock::now();
  const auto& layer_descs = this->server_->ctx_->layer_gdrcpy_mem();
  const auto& layer_mrs = this->server_->ctx_->layer_mrs();
  const char* const inbuf_bak = inbuf;
  const char* const inbuf_end = inbuf + inlen;
  RTASSERT(inlen > RPC_HEADER + sizeof(uint32_t) + sizeof(uint32_t));
  inbuf += RPC_HEADER;
  uint32_t local_crc, tensor_cnt;
  memcpy(&local_crc, inbuf, sizeof(uint32_t));
  inbuf += sizeof(uint32_t);
  memcpy(&tensor_cnt, inbuf, sizeof(uint32_t));
  inbuf += sizeof(uint32_t);

  size_t num_tensors_per_layer = layer_mrs.empty() ? 0 : layer_mrs[0].size();
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> tensor_offlens(tensor_cnt);

  bool crc_enabled = true;
  if (tensor_cnt != num_tensors_per_layer) {
    crc_enabled = false;
    LOG(WARNING) << "tensor_cnt != num_tensors_per_layer. tensor_cnt=" << tensor_cnt
    << " num_tensors_per_layer=" << num_tensors_per_layer << " reqid=" << reqid;
  } else {
    for (uint32_t tensor_idx = 0; tensor_idx < tensor_cnt; ++tensor_idx) {
      uint32_t block_cnt;
      RTASSERT(static_cast<size_t>(inbuf_end - inbuf) >= sizeof(uint32_t));
      memcpy(&block_cnt, inbuf, sizeof(uint32_t));
      inbuf += sizeof(uint32_t);

      tensor_offlens[tensor_idx].reserve(block_cnt);
      for (uint32_t i = 0; i < block_cnt; ++i) {
        RTASSERT((inbuf_end - inbuf) >= 16 /* sizeof(uint64_t) * 2 */);
        uint64_t off, len;
        memcpy(&off, inbuf, sizeof(uint64_t));
        inbuf += sizeof(uint64_t);
        memcpy(&len, inbuf, sizeof(uint64_t));
        inbuf += sizeof(uint64_t);
        tensor_offlens[tensor_idx].emplace_back(off, len);
      }
    }
  }
  const auto tp2 = SteadyClock::now();
  // rebuild crc based on layer_descs and tensor_offlens
  crc_enabled = !layer_descs.empty();
  uint32_t remote_crc = crc32_z(0L, Z_NULL, 0);
  for (size_t layer_idx = 0; layer_idx < layer_mrs.size(); ++layer_idx) {
    if (layer_idx >= layer_descs.size()) {
      crc_enabled = false;
      break;
    }
    const auto& layer_desc = layer_descs[layer_idx];
    if (tensor_cnt > layer_desc.size()) {
      crc_enabled = false;
      break;
    }
    for (uint32_t tensor_idx = 0; tensor_idx < tensor_cnt; ++tensor_idx) {
      const auto& tensor_desc = layer_desc[tensor_idx];
      if (!tensor_desc) {
        crc_enabled = false;
        break;
      }
      const Bytef *const tensor_cpu_ptr = (Bytef *)tensor_desc->cpu_ptr();

      for (const auto &[off, len] : tensor_offlens[tensor_idx]) {
        remote_crc = crc32_z(remote_crc, tensor_cpu_ptr + off, len);
      }
    }
  }
  const auto tp3 = SteadyClock::now();

  if (!crc_enabled) {
    LOG(WARNING) << "crc not enabled. use BLLM_KVTRANS_CRC=1";
    remote_crc = local_crc;
  }

  auto bufmr = AllocCPUBuffer(channel, RPC_HEADER + sizeof(uint32_t));
  memcpy(bufmr.buf, inbuf_bak, RPC_HEADER);
  memcpy(bufmr.buf + RPC_HEADER, &remote_crc, sizeof(uint32_t));
  Send(channel, std::move(bufmr),
       [channel, reqid, remote_crc, local_crc](Status s) {
         RTASSERT_EQ(local_crc, remote_crc);
         if (s.IsOk()) {
           return;
         }
         LOG(ERROR) << "resp_remote_crc err=" << s.ErrMsg()
                    << " reqid=" << reqid << " ch=" << channel->ToString();
       });
  const auto tp4 = SteadyClock::now();

  LOG(INFO) << "remote crc: reqid=" << reqid << " localcrc=" << local_crc
            << " remotecrc=" << remote_crc << " inlen=" << inlen
            << " parsedur_us=" << elapse_us(tp1, tp2)
            << " crcdur_us=" << elapse_us(tp2, tp3)
            << " senddur_us=" << elapse_us(tp3, tp4);
  return;
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

void RDMAServer::CtxCallback::resp_mem_handles(std::shared_ptr<XChannel>& channel, uint64_t reqid, char *inbuf, size_t inlen) {
  assert(inlen == RPC_HEADER);
  auto& self = *this;
  auto& handles = self.server_->info_.handles;
  const size_t msglen = handles.size() * sizeof(RDMAMemHandle);
  memp_t bufmr = AllocCPUBuffer(channel, RPC_HEADER + msglen);
  memcpy(bufmr.buf, inbuf, RPC_HEADER);
  memcpy(bufmr.buf + RPC_HEADER, handles.data(), msglen);

  fault_inject_throw();

  return Send(channel, std::move(bufmr), [channel, reqid](Status s) {
    if (s.IsOk()) {
      return;
    }
    LOG(ERROR) << "resp_mem_handles err=" << s.ErrMsg() << " reqid=" << reqid
               << " ch=" << channel->ToString();
  });
}

void RDMAServer::CtxCallback::OnRecvCall(std::shared_ptr<XChannel> ch, char *in_buf, size_t len, x_msg_header _header) noexcept {
  if (len >= RPC_HEADER) {
    const auto [magic, reqid] = deser_rpc_header(in_buf);
    if (magic == MEM_HANDLES_REQ_MAGIC) {
      try {
        this->resp_mem_handles(ch, reqid, in_buf, len);
      } catch (const std::exception &ex) {
        LOG(ERROR) << "resp mem handle: error=" << ex.what()
                   << ";ch=" << ch->ToString() << ";reqid=" << reqid;
      }
      return;
    }
    if (magic == REMOTE_CRC_REQ_MAGIC) {
      try {
        this->resp_remote_crc(ch, reqid, in_buf, len);
      } catch (const std::exception &ex) {
        LOG(ERROR) << "resp remote crc: error=" << ex.what()
                   << ";ch=" << ch->ToString() << ";reqid=" << reqid;
      }
      return;
    }
  }

  InstanceId inst_id;
  uint32_t worker_id;
  std::string req_id;
  std::vector<uint32_t> block_ids;

  if (decode_notification(in_buf, len, inst_id, worker_id, req_id, block_ids)) {
    // OnRecvCall is invoked in the barex thread pool. Beware of data races.
    this->ser_->on_recv(inst_id, worker_id, req_id, std::move(block_ids));
  }
  return;
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

// barex listen binds to INADDR_ANY, so any externally reachable IP address will do.
static void get_ip(char *info_ip, size_t bufcap) {
  const auto* vllm_host_ip = getenv("VLLM_HOST_IP");
  if (vllm_host_ip != nullptr && vllm_host_ip[0] != '\0') {
    auto last_idx = bufcap - 1;
    strncpy(info_ip, vllm_host_ip, bufcap);
    info_ip[last_idx] = '\0';
    return;
  }
  // SUSv2 guarantees that "Host names are limited to 255 bytes".
  char hostname[256];
  int status = gethostname(hostname, sizeof(hostname));
  RTASSERT(status == 0);

  struct addrinfo *res = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  status = getaddrinfo(hostname, NULL, &hints, &res);
  RTASSERT(status == 0);
  auto guard = std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)>(res, freeaddrinfo);
  assert(res->ai_family == AF_INET);
  assert(res->ai_addr->sa_family == AF_INET);
  auto *ipaddr = (struct sockaddr_in *) res->ai_addr;
  auto *ret = inet_ntop(ipaddr->sin_family, &ipaddr->sin_addr, info_ip, bufcap);
  RTASSERT(ret != nullptr);
  return;
}

void RDMAServer::start_server(ITransferService *service, Context *ctx) {
  auto &self = *this;
  RDMAInfo& info = self.info_;
  if (service == nullptr) {
    throw std::runtime_error("KvTransferService should not be null;");
  }
  auto rdma_ctx = BarexProtoContext::server_context(
    "KVTServer", 
    std::make_unique<CtxCallback>(service, this),
    TransferProtocol::Kind::RDMA_DIRECT
  );
  if (!rdma_ctx->check_support()) {
    throw std::runtime_error("can't start RDMA transfer server as RDMA protocol not support;");
  }
  ctx->register_protocol(std::move(rdma_ctx));

  WorkerInfo *winfo = ctx->worker_info_mutable();
  auto layer_num_blocks = ctx->layer_num_blocks();
  auto layer_ptr = ctx->layer_data_address();
  auto proto = TransferProtocol::rdma_direct();
  auto proto_ctx = ctx->get_protocol_ctx<BarexProtoContext>(proto);
  if (proto_ctx == nullptr) {
    throw std::runtime_error("KVT server: rdma context not register.");
  }
  auto barex_ctx = proto_ctx->barex_ctx();
  assert(self.ctx_ == nullptr);
  self.ctx_ = barex_ctx;

  info.handles.reserve(layer_ptr.size());
  for (size_t layer_idx = 0; layer_idx < layer_ptr.size(); ++layer_idx) {
    auto &layer_mrs = barex_ctx->layer_mrs()[layer_idx];
    auto &handle = info.handles.emplace_back();

    RTASSERT(layer_ptr[layer_idx].size() <= MAX_CACHE_NUM_PER_LAYER);
    RTASSERT_EQ(layer_ptr[layer_idx].size(), layer_mrs.size());
    // Now only support each layer have same number of cache
    for (size_t cache_idx = 0; cache_idx < layer_ptr[layer_idx].size();
         cache_idx++) {
      auto &out = layer_mrs[cache_idx].mr();
      auto layer_blk_p = reinterpret_cast<void *>(layer_ptr[layer_idx][cache_idx]);
      handle.ptrs[cache_idx] = layer_blk_p;
      handle.rkeys[cache_idx] = out.mr->rkey;
    }
  }
  RTASSERT_EQ(info.handles.size(), layer_ptr.size());

  get_ip(info.ip, INET_ADDRSTRLEN);
  info.port = env_port_base() + winfo->worker_id;
  RTASSERT(info.port > 0);

  {
    std::stringstream ss;
    ss << info.ip << ":" << info.port;
    winfo->addr = std::move(ss).str();
  }
  XListener *listener = nullptr;
  auto xctx = barex_ctx->xctx();
  if (xctx == nullptr) {
    LOG(ERROR) << "xctx is nullptr";
  }
  LOG(INFO) << "RDMAServer.start_server: ip=" << info.ip
            << " port=" << info.port
            << " layer_num_blocks=" << layer_num_blocks;
  auto result = XListener::NewInstance(listener, env_conn_tpsize(), info.port,
                                       TIMER_3S, {xctx});
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  self.listener_.reset(listener);
  result = self.listener_->Listen();
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
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
  for (auto &block_size : dst_info.block_sizes) {
    dst_layer_blk_sizes_.emplace_back(
      dst_info.layer_num_blocks * block_size // uint32_t * size_t = size_t
    );
  }
  dst_layer_num_ = dst_info.num_layers;
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());
  assert(dst_layer_blk_sizes_.size() == self.ctx_->layer_mrs().at(0).size());
}

[[nodiscard]] static std::future<BarexChannel>
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

static void delete_channels(CliBarexCtx* ctx, std::vector<BarexChannel> chs) {
  auto& tp = ctx->close_tp();
  tp.spawn([chs=std::move(chs)] () {
    // Destruct chs in the background thread.
  });
  assert(chs.empty());
  return;
}


RDMAChannel::~RDMAChannel() {
  delete_channels(this->ctx_, std::move(this->chs_));
  assert(this->chs_.empty());
  return;
}

static bool valid_channels(std::vector<BarexChannel>& chs) {
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

void RDMAChannel::do_init() {
  auto &self = *this;
  if (valid_channels(self.chs_)) {
    assert(self.dst_handles_.size() == self.dst_layer_num_);
    return;
  }
  std::vector<BarexChannel> tmp_chs;
  tmp_chs.swap(self.chs_);
  delete_channels(self.ctx_, std::move(tmp_chs));

  auto init_time = TimeWatch();
  const int sp = env_send_parallel();
  assert(sp > 0);

  auto& chs = self.chs_;
  chs.reserve(sp);
  auto futs = std::vector<std::future<BarexChannel>>();
  futs.reserve(sp);
  auto conn = self.ctx_->connector();
  assert(conn != nullptr);
  for (int i = 0; i < sp; ++i) {
    auto fut = Connect(*conn, self.ip_, self.port_);
    fault_inject_throw();
    futs.emplace_back(std::move(fut));
  }

  for (auto &fut : futs) {
    chs.emplace_back(fut.get());
  }
  assert(!chs.empty());

#ifndef NDEBUG
  auto delay_ms = env_debug_tx_delay_ms();
  LOG(INFO) << "RDMAChannel connect: done: delayms=" << delay_ms;
  usleep(delay_ms * 1000);
#endif

  self.dst_handles_ = self.ctx_->get_mem_handles(chs[0].sch());
  LOG(INFO) << "RDMAChannel connect. dstip=" << self.ip_
            << ",init_us=" << init_time.get_elapse_us()
            << ",dstport=" << self.port_
            << ",dsthandles_n=" << self.dst_handles_.size();
  assert(valid_channels(self.chs_));
  assert(self.dst_handles_.size() == self.dst_layer_num_);
  return;
}

[[nodiscard]] static std::future<void>
WriteSingle(XChannel *ch, memp_t sdata, uint64_t raddr, uint32_t rkey) {
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
  auto datas = datasp;
  const auto write_start_ts = SteadyClock::now();
  auto result = ch->WriteBatch(std::move(datas),
                               [pr = std::move(pr), d = std::move(datasp), write_start_ts](Status s) mutable {
                                 // WriteBatch requires datasp to stay alive until this callback.
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
  return this->sch().get();
}

std::shared_ptr<accl::barex::XChannel>& RDMAChannel::sch() noexcept {
  auto &self = *this;
  int n = self.chs_.size();
  int idx = (++self.prev_ch_idx_) % n;
  return self.chs_[idx].sch();
}

// After grouping, the input looks like:
// <src_off_1, dst_off_0, len1>
// <src_off_2, dst_off_0, len2>
// This means the contents at src_off_1,len1 and src_off_2,len2 are written to dst_off_0
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

static size_t cdiv(size_t a, size_t b) { return (a + (b - 1)) / b; }

bool RDMAChannel::is_active() {
  auto& self = *this;
  if (self.chs_.empty()) {
    return true;
  }
  return valid_channels(self.chs_);
}

void RDMAChannel::register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) {
  auto& self = *this;
  assert(!data.empty());

  assert(self.data_ == nullptr);
  self.data_ = &data;
  self.kind_ = kind;
  self.do_init();

  self.enable_crc_ = env_crc();
  if (self.enable_crc_) {
    self.crc_ = crc32_z(0L, Z_NULL, 0);
  }

#ifndef NDEBUG
  size_t total_len_debug = 0;
  for (const auto &tensor_data : data) {
    for (const auto &item : tensor_data) {
      total_len_debug += item.length;
    }
  }
#endif

  self.origin_sb_num_ = 0;
  self.merged_sb_num_ = 0;
  self.sb_size_min_ = std::numeric_limits<size_t>::max();
  self.sb_size_max_ = 0;
  self.sb_size_total_ = 0;

  if (kind == TPKind::PEQD) {
    for (auto& tensor_data : data) {
      assert(!tensor_data.empty());
      self.origin_sb_num_ += tensor_data.size();
      auto const [min, max, total, cnt] = merge_interval(tensor_data);
      assert(cnt > 0);
      self.merged_sb_num_ += cnt;
      self.sb_size_min_ = std::min(self.sb_size_min_, min);
      self.sb_size_max_ = std::max(self.sb_size_max_, max);
      self.sb_size_total_ += total;

      auto new_end =
          std::remove_if(tensor_data.begin(), tensor_data.end(),
                         [](const IpcBlock &item) { return item.length == 0; });
      tensor_data.erase(new_end, tensor_data.end());
      assert(tensor_data.size() == cnt);
    }
  } else {
    assert(data.size() == 1);
    const auto& tensor_data = data[0];
    assert(!tensor_data.empty());
    self.origin_sb_num_ = tensor_data.size();
    self.merged_sb_num_ = tensor_data.size();
    self.sb_size_min_ = tensor_data[0].length;
    self.sb_size_max_ = tensor_data[0].length;
    self.sb_size_total_ = self.sb_size_min_ * self.origin_sb_num_;
  }
  assert(self.merged_sb_num_ > 0);
  assert(self.merged_sb_num_ <= self.origin_sb_num_);
  assert(total_len_debug == self.sb_size_total_);

  self.dataperch_ = cdiv(self.merged_sb_num_, self.chs_.size());
  return;
}


// may be nullptr
static const Bytef* get_layer_tensor_cpu_ptr(CliBarexCtx* ctx, size_t layer_idx, size_t tensor_idx) {
  const auto& layer_descs = ctx->layer_gdrcpy_mem();
  const auto& layer_mrs = ctx->layer_mrs();
  if (layer_idx >= layer_mrs.size()) {
    return nullptr;
  }
  if (tensor_idx >= layer_descs[layer_idx].size()) {
    return nullptr;
  }
  const auto& tensor_desc = layer_descs[layer_idx][tensor_idx];
  return tensor_desc ? (Bytef*)tensor_desc->cpu_ptr() : (Bytef*)nullptr;
}

void RDMAChannel::send_data(size_t layer_idx) {
  auto &self = *this;
  assert(layer_idx < self.dst_layer_num_);
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());

  auto &dst_layer_handle = self.dst_handles_[layer_idx];
  const auto &data = *self.data_;
  assert(!data.empty());

  auto datasp = std::make_shared<std::vector<rw_memp_t>>();
  datasp->reserve(self.dataperch_);
  const auto &temp_src_mrs = self.ctx_->layer_mrs()[layer_idx];
  assert(data.size() == temp_src_mrs.size());

  uint32_t tensor_cnt = 0;
  for (const auto &per_tensor_data : data) {
    assert(tensor_cnt < self.ctx_->layer_blk_sizes.size());
    assert(tensor_cnt < self.dst_layer_blk_sizes_.size());
    assert(tensor_cnt < dst_layer_handle.ptrs.size());
    assert(tensor_cnt < temp_src_mrs.size());

    // Get the CPU pointer of the current tensor for CRC calculation
    const Bytef *const tensor_cpu_ptr = get_layer_tensor_cpu_ptr(self.ctx_, layer_idx, tensor_cnt);
    if (tensor_cpu_ptr == nullptr && self.enable_crc_) {
      LOG(WARNING) << "disable crc check. layer_idx=" << layer_idx << ", tensor_cnt=" << tensor_cnt;
      self.enable_crc_ = false;
    }

    for (const auto &[src_offset, dst_offset, len] : per_tensor_data) {
      const uint64_t layer_blk_size = self.ctx_->layer_blk_sizes[tensor_cnt];
      const uint64_t dst_layer_blk_size = self.dst_layer_blk_sizes_[tensor_cnt];
      assert(len > 0);
      assert(src_offset < layer_blk_size);
      assert(len < layer_blk_size);
      assert(src_offset + len <= layer_blk_size);
      assert(dst_offset < dst_layer_blk_size);
      assert(dst_offset + len <= dst_layer_blk_size);

      auto *rladdr = reinterpret_cast<char *>(dst_layer_handle.ptrs[tensor_cnt]);
      const auto &src_mr_guard = temp_src_mrs[tensor_cnt];
      const auto &src_mr = src_mr_guard.mr();

      auto rwmemp = rw_memp_t();
      rwmemp.r_addr = reinterpret_cast<uint64_t>(rladdr + dst_offset);
      rwmemp.r_key = dst_layer_handle.rkeys[tensor_cnt];
      rwmemp.sg.addr = uint64_t(src_mr.buf) + src_offset;
      rwmemp.sg.length = len;
      rwmemp.sg.lkey = src_mr.mr->lkey;

#if 0
      LOG(INFO) << "Send data: layer_idx=" << layer_idx << ", tensor_cnt=" << tensor_cnt
      << ", src_offset=" << src_offset << ", dst_offset=" << dst_offset << ", len=" << len
      << ", layer_blk_size=" << layer_blk_size << ", dst_layer_blk_size=" << dst_layer_blk_size
      << ", r_addr=" << reinterpret_cast<uint64_t>(rladdr) << ", sg.addr=" << uint64_t(src_mr.buf);
#endif
      datasp->emplace_back(std::move(rwmemp));
      if (datasp->size() >= self.dataperch_) {
        auto fut = WriteBatch(self.ch(), std::move(datasp));
        self.write_futs_.emplace_back(std::move(fut));
        datasp = std::make_shared<std::vector<rw_memp_t>>();
        datasp->reserve(self.dataperch_);
      }
      if (self.enable_crc_) {
        assert(tensor_cpu_ptr);
        auto *src_addr = tensor_cpu_ptr + src_offset;
        self.crc_ = crc32_z(self.crc_, src_addr, len);
      }
    }
    tensor_cnt++;
  }
  if (!datasp->empty()) {
    auto fut = WriteBatch(self.ch(), std::move(datasp));
    self.write_futs_.emplace_back(std::move(fut));
  }
  return;
}

// return send_us
[[nodiscard]] static std::future<uint64_t> WriteBySgList(
  XChannel *ch,
  uint64_t remote_addr,
  uint32_t rkey,
  std::shared_ptr<std::vector<memp_t>> prefills
) {
  auto pr = std::make_shared<std::promise<uint64_t>>();
  auto fut = pr->get_future();
  auto &datas = *prefills;

  const auto start_ts = SteadyClock::now();
  auto result = ch->WriteBySgList(
      datas, remote_addr, rkey,
      /* signal_peer */ false,
      /* imm_data */ 0,
      [prefills = std::move(prefills), pr = std::move(pr), start_ts](Status s) {
        // WriteBySgList requires prefills to stay alive until this callback.
        if (!s.IsOk()) {
          auto ex = std::make_exception_ptr(
              std::runtime_error("Write ERR: " + s.ErrMsg())
            );
          pr->set_exception(std::move(ex));
          return;
        }
        auto send_us = elapse_us(start_ts, SteadyClock::now());
        pr->set_value(send_us);
      });

  if (result != BAREX_SUCCESS) {
    auto ex =
        std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<uint64_t>(std::move(ex));
  }

  return fut;
}

static void when_all_succeed(std::vector<std::future<void>> &futs) {
  for (auto &fut : futs) {
    fut.get();
  }
  return;
}

void RDMAChannel::flush(std::string &outstr) {
  auto &self = *this;

  const auto inflyn = self.write_futs_.size();
  auto out = std::ostringstream();
  out << std::fixed << std::setprecision(3)
      << "OriginSbNum=" << self.origin_sb_num_
      << ",MergedSbNum=" << self.merged_sb_num_
      << ",SbSizeMin=" << self.sb_size_min_
      << ",SbSizeMax=" << self.sb_size_max_
      << ",SbSizeTotal=" << self.sb_size_total_ << ",CRC32=" << self.crc_
      << ",InflyWrite=" << inflyn;

  uint64_t send_us_min = UINT64_MAX;
  uint64_t send_us_max = 0;
  uint64_t send_us_total = 0;
  for (auto &fut : self.write_futs_) {
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

  out << ",SendUsMin=" << send_us_min << ",SendUsMax=" << send_us_max
      << ",SendUsAvg=" << send_us_total / float(inflyn);

  if (self.enable_crc_) {
    auto crcrt = TimeWatch();
    assert(!self.chs_.empty());
    auto rcrc = self.ctx_->get_remote_crc(self.chs_[0].sch(), self.data_, self.crc_);
    auto crc_dus_us = crcrt.get_elapse_us();
    out << ",CrcDurUs=" << crc_dus_us;
    RTASSERT_EQ(rcrc, self.crc_);
  }

  self.data_ = nullptr;
  outstr = std::move(out).str();
  return;
}

void RDMAChannel::send_notification(
    const std::vector<const ReqSendTask *> &reqs) {
  auto &self = *this;

  assert(self.send_futs_.empty());
  self.send_futs_.reserve(reqs.size());

  for (const auto *r : reqs) {
    assert(r->state() != ReqState::INPROCESS);
    if (r->state() != ReqState::OK) {
      continue;
    }

    const auto &reqid = r->req_id();
    const auto &block_ids = r->dst_blocks();
    // See RDMAServer::CtxCallback::OnRecvCall for encoding rules
    auto const msglen = get_encode_size(src_inst_id_, src_worker_id_, reqid, block_ids);
    auto& use_ch = self.sch();
    assert(use_ch->GetMempool() == self.ctx_->mp());
    auto bufmr = AllocCPUBuffer(use_ch, msglen);
    encode_notification(bufmr.buf, self.src_inst_id_, self.src_worker_id_,
                        reqid, block_ids);

    auto fut = Send(use_ch, std::move(bufmr));
    self.send_futs_.emplace_back(std::move(fut));
  }

  when_all_succeed(self.send_futs_);
  self.send_futs_.clear();
  return;
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
    // DataPtrCtx construction is very infrequent, so logging here is fine.
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
    // DataPtrCtx destruction is very infrequent. Logging here is fine.
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
  // cudaMalloc guarantees at least 256-byte alignment. align has no effect on PPU, i.e. kvcache alignment is not guaranteed on PPU.
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

// -----------------------------------------------------------------------------
// TCP Channel
// -----------------------------------------------------------------------------
TCPChannel::~TCPChannel() {
  delete_channels(this->ctx_, std::move(this->chs_));
  assert(this->chs_.empty());
  return;
}

cudaStream_t TCPChannel::get_d2h_stream() {
  return g_copy_kernel_ctx.get_d2h_stream();
}

void TCPServer::start_server(ITransferService *service, Context *ctx) {
  auto &self = *this;
  TCPInfo& info = self.info_;
  if (service == nullptr) {
    throw std::runtime_error("KvTransferService should not be null;");
  }
  auto tcp_ctx = BarexProtoContext::server_context(
    "KVTServer", 
    std::make_unique<CtxCallback>(service, this),
    TransferProtocol::Kind::TCP
  );
  if (!tcp_ctx->check_support()) {
    throw std::runtime_error("can't start TCP transfer server as TCP protocol not support;");
  }
  ctx->register_protocol(std::move(tcp_ctx));

  WorkerInfo *winfo = ctx->worker_info_mutable();
  auto layer_num_blocks = ctx->layer_num_blocks();
  auto layer_ptr = ctx->layer_data_address();

  auto proto = TransferProtocol::tcp();
  auto proto_ctx = ctx->get_protocol_ctx<BarexProtoContext>(proto);
  if (proto_ctx == nullptr) {
    throw std::runtime_error("KVT server: tcp context not register.");
  }
  auto barex_ctx = proto_ctx->barex_ctx();
  assert(barex_ctx != nullptr);
  self.ctx_ = barex_ctx;
  self.num_layers_ = winfo->num_layers;


  info.ptrs.reserve(layer_ptr.size());
  // Decode currently initializes as 1 layer with n tensors
  for (size_t layer_idx = 0; layer_idx < layer_ptr.size(); ++layer_idx) {
    auto &layer_mrs = barex_ctx->layer_mrs()[layer_idx];
    RTASSERT(layer_ptr[layer_idx].size() <= MAX_CACHE_NUM_PER_LAYER);
    RTASSERT_EQ(layer_ptr[layer_idx].size(), layer_mrs.size());
    for (size_t cache_idx = 0; cache_idx < layer_ptr[layer_idx].size();
        cache_idx++) {
      auto &out = layer_mrs[cache_idx].mr();
      auto layer_blk_p = reinterpret_cast<void *>(layer_ptr[layer_idx][cache_idx]);
      info.ptrs.emplace_back(layer_blk_p);
    }
  }
  RTASSERT_EQ(info.ptrs.size(), layer_ptr.size());

  get_ip(info.ip, INET_ADDRSTRLEN);
  info.port = env_port_base() + winfo->worker_id;
  RTCHECK(info.port > 0);

  {
    std::stringstream ss;
    ss << info.ip << ":" << info.port;
    winfo->addr = std::move(ss).str();
  }

  XListener *listener = nullptr;
  auto xctx = barex_ctx->xctx();
  if (xctx == nullptr) {
    LOG(ERROR) << "xctx is nullptr";
  }
  LOG(INFO) << "TCPServer.start_server: ip=" << info.ip
            << " port=" << info.port
            << " layer_num_blocks=" << layer_num_blocks;
  auto result = XListener::NewInstance(listener, env_conn_tpsize(), info.port, 
                                       TIMER_3S, {xctx});
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  self.listener_.reset(listener);
  result = self.listener_->Listen();
  RTASSERT(result == accl::barex::BAREX_SUCCESS);
  // Initialize thread pool for async CUDA stream synchronization
  // size_t sync_thread_pool_size = static_cast<size_t>(env_h2d_sync_tpsize());
  // result = accl::barex::XThreadpool::NewInstance(self.sync_thread_pool_, sync_thread_pool_size, "tcpsync");
  // RTASSERT(result == accl::barex::BAREX_SUCCESS);
  // LOG(INFO) << "TCPServer: initialized sync thread pool with " << sync_thread_pool_size << " threads";
}

void TCPServer::CtxCallback::OnRecvCall(std::shared_ptr<XChannel> ch, char *in_buf, size_t len, x_msg_header _header) noexcept {
  auto recv_start = _header.c_start;
  auto recv_time = _header.s_on_recv;
  auto onrecv_queue_us = _header.s_on_recv_call - recv_time;

  // Check if this is KV cache data
  if (len < sizeof(uint32_t)) {
    LOG(ERROR) << "TCP OnRecvCall: message too short, len=" << len;
    return;
  }
  uint32_t magic;
  memcpy(&magic, in_buf, sizeof(uint32_t));
  
  if (magic == KV_CACHE_DATA_MAGIC) {
    return this->handle_kv_cache_data(ch, in_buf, len, recv_start, recv_time, onrecv_queue_us);
  }
  return;
}

void TCPServer::CtxCallback::handle_kv_cache_data(
  std::shared_ptr<XChannel>& channel, char *in_buf, size_t len,
  uint64_t recv_start,
  uint32_t recv_time,
  uint32_t onrecv_queue_us) {
  const auto h2d_start_ts = std::chrono::system_clock::now(); // t4
  
  auto& self = *this;
  // Parse header: [magic (uint32_t)] [reqid (uint64_t)] [layer_idx (size_t)] 
  //               [metadata_size (size_t)] [IpcBlock array] [tensor data]
  if (len < RPC_HEADER + sizeof(size_t) + sizeof(size_t)) {
    LOG(ERROR) << "TCP handle_kv_cache_data: message too short, len=" << len;
    return;
  }
  char* buf_ptr = in_buf;
  // Parse RPC header: magic + reqid
  auto [magic, reqid] = deser_rpc_header(buf_ptr);
  buf_ptr += RPC_HEADER;
  assert(magic == KV_CACHE_DATA_MAGIC);
  
  size_t layer_idx;
  memcpy(&layer_idx, buf_ptr, sizeof(size_t));
  buf_ptr += sizeof(size_t);
  
  size_t metadata_size;
  memcpy(&metadata_size, buf_ptr, sizeof(size_t));
  buf_ptr += sizeof(size_t);
  
  size_t metadata_bytes = metadata_size * sizeof(IpcBlock);
  size_t expected_len = RPC_HEADER + sizeof(size_t) + sizeof(size_t) + metadata_bytes;
  if (len < expected_len) {
    LOG(ERROR) << "TCP handle_kv_cache_data: incomplete metadata, expected=" << expected_len << ", actual=" << len;
    return;
  }
  std::vector<IpcBlock> blocks;
  blocks.reserve(metadata_size);
  const IpcBlock* raw_blocks = reinterpret_cast<const IpcBlock*>(buf_ptr);
  for (size_t i = 0; i < metadata_size; ++i) {
    blocks.emplace_back(raw_blocks[i].src_offset, raw_blocks[i].dst_offset, raw_blocks[i].length);
  }
  buf_ptr += metadata_bytes;
  size_t tensor_data_size = len - expected_len;

  auto& ptrs = self.server_->info_.ptrs;
  if (layer_idx >= ptrs.size()) {
    LOG(ERROR) << "TCP handle_kv_cache_data: invalid layer_idx=" << layer_idx << " max=" << ptrs.size();
    return;
  }
  void* layer_gpu_ptr = ptrs[layer_idx];
  assert(layer_gpu_ptr != nullptr);
  
  // Get thread-local preallocated buffer for kernel metadata
  int device_id = self.server_->ctx_->device_id();
  auto [device_blk_buffer, host_blk_buffer] = get_kernel_copy_buffer(device_id);
  int64_t* device_blk_buffer_ptr = reinterpret_cast<int64_t*>(device_blk_buffer);
  int64_t* host_blk_buffer_ptr = reinterpret_cast<int64_t*>(host_blk_buffer);

  // Use thread-local stream to allow concurrent H2D copies from different threads
  cudaStream_t h2d_stream = TCPServer::get_h2d_stream();

  cudaError_t cuda_rt = copy_handle_data_with_kernel(
    buf_ptr, layer_gpu_ptr, 
    blocks, 
    tensor_data_size, 
    CopyDirection::H2D, 
    device_id, 
    h2d_stream,
    device_blk_buffer_ptr,
    host_blk_buffer_ptr
  );
  RTASSERT(cuda_rt == cudaSuccess);

  // TODO: add resp code to check success
  // Send response after copy operation completes
  auto buffer_size = RPC_HEADER 
                    + sizeof(h2d_start_ts) * 2 // h2d_start_ts and h2d_end_ts
                    + sizeof(recv_start) + sizeof(recv_time) 
                    + sizeof(onrecv_queue_us);

  memp_t resp_buf = AllocCPUBuffer(channel, buffer_size);
  auto resp_buf_ptr = resp_buf.buf;
  ser_rpc_header(resp_buf_ptr, magic, reqid);
  resp_buf_ptr += RPC_HEADER;
  memcpy(resp_buf_ptr, &recv_start, sizeof(recv_start));
  resp_buf_ptr += sizeof(recv_start);
  memcpy(resp_buf_ptr, &recv_time, sizeof(recv_time));
  resp_buf_ptr += sizeof(recv_time);
  memcpy(resp_buf_ptr, &onrecv_queue_us, sizeof(onrecv_queue_us));
  resp_buf_ptr += sizeof(onrecv_queue_us);
  memcpy(resp_buf_ptr, &h2d_start_ts, sizeof(h2d_start_ts));
  resp_buf_ptr += sizeof(h2d_start_ts);

  // Log before synchronization for debugging
  // Check stream status before synchronization
  cudaError_t stream_query_err = cudaStreamQuery(h2d_stream);
  if (stream_query_err != cudaSuccess && stream_query_err != cudaErrorNotReady) {
    LOG(ERROR) << "TCP handle_kv_cache_data: cudaStreamQuery failed before sync, "
               << "reqid=" << reqid
               << ", error=" << cudaGetErrorString(stream_query_err)
               << " (" << stream_query_err << ")";
  }
  
  // Check for any pending CUDA errors before synchronization
  cudaError_t pending_err = cudaGetLastError();
  if (pending_err != cudaSuccess) {
    LOG(ERROR) << "TCP handle_kv_cache_data: pending CUDA error before sync, "
               << "reqid=" << reqid
               << ", error=" << cudaGetErrorString(pending_err)
               << " (" << pending_err << ")";
  }

  auto cuda_rt_sync = cudaStreamSynchronize(h2d_stream);
  if (cuda_rt_sync != cudaSuccess) {
    LOG(ERROR) << "TCP handle_kv_cache_data: cudaStreamSynchronize failed, "
               << "reqid=" << reqid
               << ", layer_idx=" << layer_idx
               << ", device_id=" << device_id
               << ", h2d_stream=" << reinterpret_cast<void*>(h2d_stream)
               << ", tensor_data_size=" << tensor_data_size
               << ", blocks_count=" << blocks.size()
               << ", error=" << cudaGetErrorString(cuda_rt_sync)
               << " (" << cuda_rt_sync << ")";
  }
  RTASSERT(cuda_rt_sync == cudaSuccess);
  const auto h2d_end_ts = std::chrono::system_clock::now(); // t5
  memcpy(resp_buf_ptr, &h2d_end_ts, sizeof(h2d_end_ts));

  Send(channel, std::move(resp_buf), [] (Status s) {
    if (s.IsOk()) {
      return;
    }
    LOG(ERROR) << "TCP handle_kv_cache_data: send response err=" << s.ErrMsg();
  });
  // });
  return;
}

void TCPChannel::register_data(std::vector<std::vector<IpcBlock>>& data, TPKind kind) {
  auto& self = *this;

  assert(!data.empty());
  // TCP Currently not support dpsk v32
  assert(data.size() == 1);

  assert(self.data_ == nullptr);
  self.data_ = &data;
  self.kind_ = kind;

  self.do_init();

#ifndef NDEBUG
  size_t total_len_debug = 0;
  for (const auto &tensor_data : data) {
    for (const auto &item : tensor_data) {
      total_len_debug += item.length;
    }
  }
#endif
  // TODO: TCP CRC check
  self.origin_sb_num_ = 0;
  self.merged_sb_num_ = 0;
  self.sb_size_min_ = std::numeric_limits<size_t>::max();
  self.sb_size_max_ = 0;
  self.sb_size_total_ = 0;

  std::vector<size_t> tensor_sizes;

  if (kind == TPKind::PEQD) {
    for (auto& tensor_data : data) {
      assert(!tensor_data.empty());
      self.origin_sb_num_ += tensor_data.size();
      auto const [min, max, total, cnt] = merge_interval(tensor_data);
      assert(cnt > 0);
      self.merged_sb_num_ += cnt;
      self.sb_size_min_ = std::min(self.sb_size_min_, min);
      self.sb_size_max_ = std::max(self.sb_size_max_, max);
      self.sb_size_total_ += total;
      tensor_sizes.emplace_back(total);
      auto new_end =
          std::remove_if(tensor_data.begin(), tensor_data.end(),
                         [](const IpcBlock &item) { return item.length == 0; });
      tensor_data.erase(new_end, tensor_data.end());
      assert(tensor_data.size() == cnt);
    }
  } else {
    assert(data.size() == 1);
    const auto& tensor_data = data[0];
    assert(!tensor_data.empty());
    self.origin_sb_num_ = tensor_data.size();
    self.merged_sb_num_ = tensor_data.size();
    self.sb_size_min_ = tensor_data[0].length;
    self.sb_size_max_ = tensor_data[0].length;
    self.sb_size_total_ = self.sb_size_min_ * self.origin_sb_num_;
    tensor_sizes.emplace_back(self.sb_size_total_);
  }
  assert(self.merged_sb_num_ > 0);
  assert(self.merged_sb_num_ <= self.origin_sb_num_);
  // both fp8 length
  assert(total_len_debug == self.sb_size_total_);

  if (self.cast2fp8_) {
    // block's length generated by parse_block is fp8 dtype length
    // multiply by 2 to get the actual length
    self.sb_size_total_ *= 2;
  }

  const auto& tensor_data = data[0];

  // TODO: TCP adopt multi-tensor per layer
  // Allocate CPU buffer for each thread, and use as thread-local buffer
  if (self.host_buffers_.size() == 0 && self.sb_size_total_ > 0) {
    self.host_buffers_.reserve(self.dst_layer_num_); // prepare for each layer previously
    size_t metadata_size = tensor_data.size();
    size_t header_bytes = RPC_HEADER + sizeof(size_t);  // magic + reqid + layer_idx
    size_t metadata_bytes = sizeof(size_t) + metadata_size * sizeof(IpcBlock);
    const size_t tensor_bytes = self.cast2fp8_ ? (self.sb_size_total_ / 2) : self.sb_size_total_;

    size_t total_bytes = header_bytes + metadata_bytes + tensor_bytes;
    
    uint64_t alloc_us_min = UINT64_MAX, alloc_us_max = 0, alloc_us_total = 0;

    for (size_t i = 0; i < self.dst_layer_num_; ++i) {
      auto alloc_start = std::chrono::system_clock::now();
      memp_t buffer_mr = AllocCudaHostBuffer(self.ch(), total_bytes);
      auto alloc_end = std::chrono::system_clock::now();
      auto alloc_elapsed_us = elapse_us_system(alloc_start, alloc_end);
      RTCHECK(buffer_mr.buf != nullptr);
      RTCHECK(buffer_mr.buf_len >= total_bytes);
      self.host_buffers_.emplace_back(std::move(buffer_mr));
      alloc_us_min = std::min(alloc_us_min, alloc_elapsed_us);
      alloc_us_max = std::max(alloc_us_max, alloc_elapsed_us);
      alloc_us_total += alloc_elapsed_us;
    }
    LOG(INFO) << "TCPChannel::do_init: alloc cuda host buffer total_bytes=" << total_bytes
              << " time min=" << alloc_us_min 
              << " max=" << alloc_us_max 
              << " total=" << alloc_us_total
              << " avg=" << alloc_us_total / self.dst_layer_num_ << " us"
              << " dst_layer_num=" << self.dst_layer_num_
              << " metadata_size=" << metadata_size
              << " header_bytes=" << header_bytes
              << " metadata_bytes=" << metadata_bytes
              << " tensor_bytes=" << tensor_bytes;
  }
  assert(self.host_buffers_.size() == self.dst_layer_num_);
  return ;
}

// Current TCP only support single tensor per layer
void TCPChannel::send_data(size_t layer_idx) {
  auto &self = *this;
  assert(layer_idx < self.dst_layer_num_);
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());

  // TCP Currently not support dpsk v32
  assert(self.data_ != nullptr);
  assert(self.data_->size() > 0);
  const auto& data = (*self.data_)[0];
  assert(!data.empty());

  const auto send_data_start_ts = std::chrono::system_clock::now();  // t1

  // Prepare data to send: magic + reqid + layer_idx + metadata (IpcBlock array) + tensor data
  // Format: [magic (uint32_t)] [reqid (uint64_t)] [layer_idx (size_t)] [metadata_size (size_t)] [IpcBlock array] [tensor data]
  const uint32_t magic = KV_CACHE_DATA_MAGIC;
  const uint64_t reqid = new_id();
  const size_t metadata_size = data.size();
  const size_t header_bytes = RPC_HEADER + sizeof(size_t);  // magic + reqid + layer_idx
  const size_t metadata_bytes = sizeof(size_t) + metadata_size * sizeof(IpcBlock);  // metadata_size + IpcBlock array
  
  const bool cast2fp8 = self.cast2fp8_;
  const size_t tensor_send_size = cast2fp8 ? (self.sb_size_total_ / 2) : self.sb_size_total_;
  
  const size_t total_send_size = header_bytes + metadata_bytes + tensor_send_size;
  assert(self.host_buffers_.size() == self.dst_layer_num_ - layer_idx);
  assert(self.host_buffers_.back().buf_len >= total_send_size);

  // Copy all GPU data to host_buffer in continuous memory
  char* buf_ptr = self.host_buffers_.back().buf;
  char* meta_buf_ptr = buf_ptr + header_bytes;  // Skip header for metadata
  char* tensor_buf_ptr = buf_ptr + header_bytes + metadata_bytes;  // Skip header and metadata for tensor data

  const auto& src_mrs = self.ctx_->layer_mrs()[layer_idx];
  const auto& src_mr_guard = src_mrs[0];
  const auto& src_mr_base = src_mr_guard.mr();
  
  // Copy GPU data to host buffer and prepare metadata
  const auto d2h_start_ts = std::chrono::system_clock::now();   // t2
  
  std::vector<IpcBlock> kernel_blocks;
  kernel_blocks.reserve(metadata_size);
  for (const auto& block : data) {
    kernel_blocks.emplace_back(0, block.src_offset, block.length * (cast2fp8 ? 2 : 1));
  }
  
  int device_id = self.ctx_->device_id();

  void* gpu_src_ptr = reinterpret_cast<void*>(const_cast<char*>(src_mr_base.buf));

  // Get thread-local preallocated buffer for kernel metadata
  auto [device_blk_buffer, host_blk_buffer] = get_kernel_copy_buffer(device_id);
  int64_t* device_blk_buffer_ptr = reinterpret_cast<int64_t*>(device_blk_buffer);
  int64_t* host_blk_buffer_ptr = reinterpret_cast<int64_t*>(host_blk_buffer);

  cudaStream_t d2h_stream = TCPChannel::get_d2h_stream();

  cudaError_t cuda_rt;
  if (cast2fp8) {
    cuda_rt = copy_d2h_bf16_to_fp8(
      tensor_buf_ptr,           // CPU destination buffer (FP8 E4M3)
      gpu_src_ptr,               // GPU source base pointer (BF16)
      kernel_blocks,             // Blocks: src_offset=CPU offset, dst_offset=GPU src_offset
      self.sb_size_total_,       // Total BF16 tensor data size (in bytes)
      device_id,                 // Target CUDA device ID
      d2h_stream,                // CUDA stream
      device_blk_buffer_ptr,     // Preallocated GPU buffer for metadata
      host_blk_buffer_ptr        // Preallocated host pinned buffer for metadata
    );
  } else {
    cuda_rt = copy_handle_data_with_kernel(
      tensor_buf_ptr,           // CPU destination buffer (contiguous)
      gpu_src_ptr,              // GPU source base pointer
      kernel_blocks,            // Blocks: src_offset=CPU offset, dst_offset=GPU src_offset
      self.sb_size_total_,      // Total tensor data size
      CopyDirection::D2H,       // D2H direction
      device_id,                // Target CUDA device ID
      d2h_stream,               // CUDA stream
      device_blk_buffer_ptr,    // Preallocated GPU buffer for metadata
      host_blk_buffer_ptr       // Preallocated host pinned buffer for metadata
    );
  }
  RTCHECK(cuda_rt == cudaSuccess);

  // Write RPC header: magic + reqid
  ser_rpc_header(buf_ptr, magic, reqid);
  buf_ptr += RPC_HEADER;
  // Write layer_idx
  memcpy(buf_ptr, &layer_idx, sizeof(size_t));
  buf_ptr += sizeof(size_t);
  
  // Write metadata: metadata_size + IpcBlock array
  memcpy(meta_buf_ptr, &metadata_size, sizeof(size_t));
  meta_buf_ptr += sizeof(size_t);
  memcpy(meta_buf_ptr, data.data(), metadata_size * sizeof(IpcBlock));
  
  auto cuda_rt_sync = cudaStreamSynchronize(d2h_stream);
  RTCHECK(cuda_rt_sync == cudaSuccess);
  const auto d2h_end_ts = std::chrono::system_clock::now(); // t3

  self.host_buffers_.back().buf_len = total_send_size;

  auto pr = std::make_shared<SendKVCacheData::Promise>();
  auto time_fut = pr->get_future();

  self.ctx_->push(reqid, SendKVCacheData(
    reqid, pr, send_data_start_ts, d2h_start_ts, d2h_end_ts
  ));

  memp_t send_buf = std::move(self.host_buffers_.back());
  self.host_buffers_.pop_back();

  // After send, send_buf(host_buffer_) will be released.
  Send(self.sch(), std::move(send_buf), [&self, reqid](Status s) mutable {
    if (!s.IsOk()) {
      self.ctx_->on_send_error(std::move(s), reqid);
      return;
    }
  });
  self.write_futs_.emplace_back(std::move(time_fut));
  return;
}

accl::barex::XChannel *TCPChannel::ch() noexcept {
  return this->sch().get();
}

std::shared_ptr<accl::barex::XChannel>& TCPChannel::sch() noexcept {
  auto &self = *this;
  int n = self.chs_.size();
  int idx = (++self.prev_ch_idx_) % n;
  return self.chs_[idx].sch();
}

void TCPChannel::connect(const WorkerInfo &dst_info) {
  auto &self = *this;
  std::string addr(dst_info.addr);
  auto pos = addr.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid tcp address: " + addr);
  }

  self.ip_ = addr.substr(0, pos);
  auto port_str = addr.substr(pos + 1, addr.size());
  self.port_ = std::stoi(port_str);
  for (auto &block_size : dst_info.block_sizes) {
    dst_layer_blk_sizes_.emplace_back(
      dst_info.layer_num_blocks * block_size // uint32_t * size_t = size_t
    );
  }
  dst_layer_num_ = dst_info.num_layers;
  assert(self.dst_layer_num_ == self.ctx_->layer_mrs().size());
  assert(dst_layer_blk_sizes_.size() == self.ctx_->layer_mrs().at(0).size());
}

bool TCPChannel::is_active() {
  auto& self = *this;
  if (self.chs_.empty()) {
    return true;
  }
  return valid_channels(self.chs_);
}

void TCPChannel::do_init() {
  auto &self = *this;
  if (valid_channels(self.chs_)) {
    assert(self.ctx_->layer_mrs().size() == self.dst_layer_num_);
    return;
  }
  std::vector<BarexChannel> tmp_chs;
  tmp_chs.swap(self.chs_);
  delete_channels(self.ctx_, std::move(tmp_chs));
  
  auto init_time = TimeWatch();
  const int sp = env_send_parallel();
  assert(sp > 0);

  auto& chs = self.chs_;
  chs.reserve(sp);
  auto futs = std::vector<std::future<BarexChannel>>();
  futs.reserve(sp);
  auto conn = self.ctx_->connector();
  assert(conn != nullptr);
  for (int i = 0; i < sp; ++i) {
    auto fut = Connect(*conn, self.ip_, self.port_);
    fault_inject_throw();
    futs.emplace_back(std::move(fut));
  }

  for (auto &fut : futs) {
    chs.emplace_back(fut.get());
  }
  assert(!chs.empty());

#ifndef NDEBUG
  auto delay_ms = env_debug_tx_delay_ms();
  LOG(INFO) << "TCPChannel connect: done: delayms=" << delay_ms;
  usleep(delay_ms * 1000);
#endif

  LOG(INFO) << "TCPChannel connect. dstip=" << self.ip_
            << ",init_us=" << init_time.get_elapse_us()
            << ",dstport=" << self.port_;
  assert(valid_channels(self.chs_));
  return;
}

void TCPChannel::flush(std::string& outstr) {
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
  uint64_t send_us_min = UINT64_MAX, d2h_us_min = UINT64_MAX, h2d_us_min = UINT64_MAX, trans_us_min = UINT64_MAX;
  uint64_t send_us_max = 0, d2h_us_max = 0, h2d_us_max = 0, trans_us_max = 0;
  uint64_t send_us_total = 0, d2h_us_total = 0, h2d_us_total = 0, trans_us_total = 0;
  uint64_t link_tx_us_min = UINT64_MAX, link_tx_us_max = 0, link_tx_us_total = 0;
  uint64_t recv_us_min = UINT64_MAX, recv_us_max = 0, recv_us_total = 0;
  uint64_t onrecv_queue_us_min = UINT64_MAX, onrecv_queue_us_max = 0, onrecv_queue_us_total = 0;
  for (auto& fut : self.write_futs_) {
    TCPTimePoints time_points = {};
    try {
      time_points = fut.get();
    } catch (...) {
      outstr = std::move(out).str();
      throw;
    }
    auto send_us = elapse_us_system(time_points.send_data_start_ts_, time_points.h2d_end_ts);
    auto d2h_us = elapse_us_system(time_points.d2h_start_ts_, time_points.d2h_end_ts_);
    auto h2d_us = elapse_us_system(time_points.h2d_start_ts, time_points.h2d_end_ts);
    auto trans_us = elapse_us_system(time_points.d2h_end_ts_, time_points.h2d_start_ts);
    auto link_tx_us = time_points.recv_start_ - time_point_to_microseconds(time_points.d2h_end_ts_);
    uint64_t recv_us = static_cast<uint64_t>(time_points.recv_time_);
    uint64_t onrecv_queue_us = static_cast<uint64_t>(time_points.onrecv_queue_us_);

    send_us_min = std::min(send_us_min, send_us);
    send_us_max = std::max(send_us_max, send_us);
    send_us_total += send_us;
    d2h_us_min = std::min(d2h_us_min, d2h_us);
    d2h_us_max = std::max(d2h_us_max, d2h_us);
    d2h_us_total += d2h_us;
    h2d_us_min = std::min(h2d_us_min, h2d_us);
    h2d_us_max = std::max(h2d_us_max, h2d_us);
    h2d_us_total += h2d_us;
    trans_us_min = std::min(trans_us_min, trans_us);
    trans_us_max = std::max(trans_us_max, trans_us);
    trans_us_total += trans_us;
    link_tx_us_min = std::min(link_tx_us_min, link_tx_us);
    link_tx_us_max = std::max(link_tx_us_max, link_tx_us);
    link_tx_us_total += link_tx_us;
    recv_us_min = std::min(recv_us_min, recv_us);
    recv_us_max = std::max(recv_us_max, recv_us);
    recv_us_total += recv_us;
    onrecv_queue_us_min = std::min(onrecv_queue_us_min, onrecv_queue_us);
    onrecv_queue_us_max = std::max(onrecv_queue_us_max, onrecv_queue_us);
    onrecv_queue_us_total += onrecv_queue_us;
  }
  self.write_futs_.clear();
  // After send done, host_buffer_ mempt's buffer should be released by Send callback.
  // And host_buffers_ should be empty.
  assert(self.host_buffers_.size() == 0);
  out << ",SendUsMin=" << send_us_min
      << ",SendUsMax=" << send_us_max
      << ",SendUsAvg=" << send_us_total / float(inflyn)
      << ",D2HUsMin=" << d2h_us_min
      << ",D2HUsMax=" << d2h_us_max
      << ",D2HUsAvg=" << d2h_us_total / float(inflyn)
      << ",H2DUsMin=" << h2d_us_min
      << ",H2DUsMax=" << h2d_us_max
      << ",H2DUsAvg=" << h2d_us_total / float(inflyn)
      << ",TransUsMin=" << trans_us_min
      << ",TransUsMax=" << trans_us_max
      << ",TransUsAvg=" << trans_us_total / float(inflyn)
      << ",LinkTxUsMin=" << link_tx_us_min
      << ",LinkTxUsMax=" << link_tx_us_max
      << ",LinkTxUsAvg=" << link_tx_us_total / float(inflyn)
      << ",RecvUsMin=" << recv_us_min
      << ",RecvUsMax=" << recv_us_max
      << ",RecvUsAvg=" << recv_us_total / float(inflyn)
      << ",OnRecvQueueUsMin=" << onrecv_queue_us_min
      << ",OnRecvQueueUsMax=" << onrecv_queue_us_max
      << ",OnRecvQueueUsAvg=" << onrecv_queue_us_total / float(inflyn);
  outstr = std::move(out).str();
  return;
}

void TCPChannel::send_notification(const std::vector<const ReqSendTask*>& reqs) {
  return;
}

} // namespace blade_llm
