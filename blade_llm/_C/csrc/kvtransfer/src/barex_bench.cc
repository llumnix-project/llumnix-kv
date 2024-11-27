#if 0

#include <accl/barex/barex.h>
#include <accl/barex/barex_types.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>
#include <accl/barex/xtimer.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <future>
#include <chrono>
#include <assert.h>
#include <cuda_runtime.h>


using namespace accl::barex;


#define RTCHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "Runtime error: Assertion failed in %s on line %d: %s\n", __FILE__, __LINE__, #expr); \
        abort(); \
    } \
} while (0)

struct BarexCtx {
  XDeviceManager *manager = nullptr;
  XDevice* nic_dev = nullptr;
  XSimpleMempool* mp = nullptr;
  XThreadpool* tp = nullptr;
  XContext* xctx = nullptr;
  memp_t mr;
public:
  static BarexCtx setup();
};


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


static size_t env_token_size() {
  const char* envval = getenv("ZY_TOKEN_SIZE");
  RTCHECK(envval != nullptr);
  int ret = atoi(envval);
  RTCHECK(ret > 0);
  return ret;
}


static size_t env_block_size() {
  const char* envval = getenv("ZY_BLOCK_SIZE");
  RTCHECK(envval != nullptr);
  auto ret = atol(envval);
  RTCHECK(ret > 0);
  return ret;
}


static size_t env_srv_token_size() {
  const char* envval = getenv("ZY_SRV_TOKEN_SIZE");
  RTCHECK(envval != nullptr);
  int ret = atoi(envval);
  RTCHECK(ret > 0);
  return ret;
}


static size_t env_srv_block_size() {
  const char* envval = getenv("ZY_SRV_BLOCK_SIZE");
  RTCHECK(envval != nullptr);
  auto ret = atol(envval);
  RTCHECK(ret > 0);
  return ret;
}


static size_t env_block_number() {
  const char* envval = getenv("ZY_BLOCK_NUM");
  RTCHECK(envval != nullptr);
  auto ret = atol(envval);
  RTCHECK(ret > 0);
  return ret;
}


static int env_gpu_id() {
  const char* envval = getenv("ZY_GPU_ID");
  RTCHECK(envval != nullptr);
  int ret = atoi(envval);
  RTCHECK(ret >= 0);
  return ret;
}


static int env_server_port() {
  const char* envval = getenv("ZY_SERVER_PORT");
  RTCHECK(envval != nullptr);
  int ret = atoi(envval);
  RTCHECK(ret > 0);
  return ret;
}


static const char* env_server_ip() {
  const char* envval = getenv("ZY_SERVER_IP");
  RTCHECK(envval != nullptr);
  return envval;
}

static int env_is_server() {
  const char* envval = getenv("ZY_IS_SERVER");
  RTCHECK(envval != nullptr);
  int ret = atoi(envval);
  return ret;
}


static int env_rounds() {
  const char* envval = getenv("ZY_ROUNDS");
  RTCHECK(envval != nullptr);
  int ret = atoi(envval);
  RTCHECK(ret > 0);
  return ret;
}


static const char* env_method() {
  const char* envval = getenv("ZY_METHOD");
  RTCHECK(envval != nullptr);
  return envval;
}


static long env_server_addr() {
  const char* envval = getenv("ZY_SERVER_ADDR");
  RTCHECK(envval != nullptr);
  auto ret = strtol(envval, (char **)NULL, 0);
  RTCHECK(ret > 0);
  return ret;
}


static long env_server_rkey() {
  const char* envval = getenv("ZY_SERVER_RKEY");
  RTCHECK(envval != nullptr);
  auto ret = strtol(envval, (char **)NULL, 0);
  RTCHECK(ret >= 0);
  return ret;
}


static std::vector<int> env_src_block_ids() {
  const char* envval = getenv("ZY_BLOCK_IDS");
  RTCHECK(envval != nullptr);
  std::string line(envval);

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


static std::vector<int> env_dst_block_ids() {
  // 好像没有必要搞两个 block id?
  return env_src_block_ids();
}



template<typename T, typename E>
static std::future<T> make_exp_future(E ex) {
  std::promise<T> pr;
  pr.set_exception(std::make_exception_ptr(std::move(ex)));
  return pr.get_future();
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


static void WriteBatch(const std::vector<XChannel*>& chs, std::shared_ptr<std::vector<rw_memp_t>> datasp) {
  assert(!chs.empty());
  if (chs.size() == 1) {
    return WriteBatch(chs.front(), std::move(datasp)).get();
  }

  // 这里假设 datasp 中每个 block 大小相近.
  std::vector<std::future<void>> futs;
  futs.reserve(chs.size());
  auto const part_size = (datasp->size() + chs.size() - 1) / chs.size();
  assert(part_size >= 1);
  for (size_t ch_idx = 0; ch_idx < chs.size(); ++ch_idx) {
    size_t data_start = ch_idx * part_size;
    if (data_start >= datasp->size()) {
      break;
    }
    size_t data_end = std::min(data_start + part_size, datasp->size());
    assert(data_start < data_end);

    auto part_sp = std::make_shared<std::vector<rw_memp_t>>();
    part_sp->reserve(part_size);
    for (size_t data_idx = data_start; data_idx < data_end; ++data_idx) {
      part_sp->emplace_back(std::move((*datasp)[data_idx]));
    }
    assert(!part_sp->empty());
    futs.emplace_back(WriteBatch(chs[ch_idx], std::move(part_sp)));
  }
  for (auto& fut : futs) {
    fut.get();
  }

  return ;
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


class FackCb : public XChannelCallback {
 public:
  void OnRecvCall(XChannel *channel,
                  char *in_buf,
                  size_t len,
                  x_msg_header header) override {}
};


BarexCtx BarexCtx::setup() {
  BarexCtx ret;
  auto result = XDeviceManager::Singleton(ret.manager);
  RTCHECK(result == BAREX_SUCCESS);

  const auto& all_nic_devs = ret.manager->AllDevices();
  RTCHECK(!all_nic_devs.empty());
  const char* nicdev_name = getenv("ZY_NIC_DEV");
  RTCHECK(nicdev_name != nullptr);
  for (const auto& dev : all_nic_devs) {
    if (dev->GetName() == nicdev_name) {
      ret.nic_dev = dev;
      break;
    }
  }
  RTCHECK(ret.nic_dev != nullptr);

  result = XSimpleMempool::NewInstance(ret.mp, "mp", {ret.nic_dev});
  RTCHECK(result == BAREX_SUCCESS);

  int gpuid = env_gpu_id();
  auto cuda_rt = cudaSetDevice(gpuid);
  RTCHECK(cuda_rt == cudaSuccess);

  XAllocator* gpu_allocator = nullptr;
  result = ret.mp->GetXAllocator(gpu_allocator, GPU);
  RTCHECK(result == BAREX_SUCCESS);
  size_t kvcache_size = env_block_size() * env_block_number();
  void* const buf = gpu_allocator->Alloc(kvcache_size, gpuid, nullptr /* attr */, 512 /* align */);
  ret.mp->RegUserMr(ret.mr, buf, kvcache_size, GPU, gpuid);
  printf("RegUserMr. buf=%p bufsize=%lu gpuid=%d rkey=%u\n", buf, kvcache_size, gpuid, ret.mr.mr->rkey);

  result = XThreadpool::NewInstance(ret.tp, 4, "tp");
  RTCHECK(result == BAREX_SUCCESS);

  ContextConfig config = XConfigUtil::DefaultContextConfig();
  result = XContext::NewInstance(ret.xctx, config, new FackCb(), ret.nic_dev, ret.mp, ret.tp);
  RTCHECK(result == BAREX_SUCCESS);
  ret.xctx->Start();

  return ret;
}


struct ServerCtx {
  BarexCtx ctx;
  XListener* listener = nullptr;
public:
  static ServerCtx setup();
};


ServerCtx ServerCtx::setup() {
  ServerCtx ret;
  ret.ctx = BarexCtx::setup();
  int p = env_server_port();
  auto result = XListener::NewInstance(ret.listener, 2, p, TIMER_3S, {ret.ctx.xctx});
  RTCHECK(result == BAREX_SUCCESS);
  result = ret.listener->Listen();
  RTCHECK(result == BAREX_SUCCESS);
  printf("ServerCtx.setup. p=%d\n", p);
  return ret;
}


struct ClientCtx {
  BarexCtx ctx;
  XConnector* connector = nullptr;
  memp_t cpu_mr;
  memp_t gpu_mr;
  cudaStream_t cpy_stream;
private:
  std::vector<accl::barex::XChannel *> chs_;
  size_t prev_ch_idx_ = 0;
public:
  XChannel* ch() noexcept {
    auto &self = *this;
    int n = self.chs_.size();
    int idx = (++self.prev_ch_idx_) % n;
    return self.chs_[idx];
  }

  const auto& chs() const noexcept {
    return this->chs_;
  }
public:
  static ClientCtx setup();
};


ClientCtx ClientCtx::setup() {
  ClientCtx ret;
  ret.ctx = BarexCtx::setup();
  int gpuid = env_gpu_id();

  auto cuda_rt = cudaStreamCreateWithFlags(&ret.cpy_stream, cudaStreamNonBlocking);
  RTCHECK(cuda_rt == cudaSuccess);

  XAllocator* cpu_alloc = nullptr;
  auto result = ret.ctx.mp->GetXAllocator(cpu_alloc, CPU);
  RTCHECK(result == BAREX_SUCCESS);
  size_t cpu_size = env_srv_block_size() * env_block_number();
  auto* cpu_buf = cpu_alloc->Alloc(cpu_size);
  result = ret.ctx.mp->RegUserMr(ret.cpu_mr, cpu_buf, cpu_size, CPU);
  RTCHECK(result == BAREX_SUCCESS);

  XAllocator* gpu_allocator = nullptr;
  result = ret.ctx.mp->GetXAllocator(gpu_allocator, GPU);
  RTCHECK(result == BAREX_SUCCESS);
  void* const buf = gpu_allocator->Alloc(cpu_size, gpuid, nullptr /* attr */, 512 /* align */);
  result = ret.ctx.mp->RegUserMr(ret.gpu_mr, buf, cpu_size, GPU, gpuid);
  RTCHECK(result == BAREX_SUCCESS);

  result = XConnector::NewInstance(ret.connector, 2, TIMER_3S, {ret.ctx.xctx});
  RTCHECK(result == BAREX_SUCCESS);

  const int sp = get_send_parallel();
  assert(sp > 0);
  const char* sip = env_server_ip();
  const auto sport = env_server_port();
  ret.chs_.reserve(sp);
  auto futs = std::vector<std::future<XChannel *>>();
  futs.reserve(sp);
  for (int i = 0; i < sp; ++i) {
    auto fut = Connect(*ret.connector, sip, sport);
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    ret.chs_.emplace_back(fut.get());
  }
  assert(!ret.chs_.empty());

  printf("ClientCtx.setup(). env_server_ip=%s env_server_port=%d\n", sip, sport);
  return ret;
}


void server_main() {
  ServerCtx ctx = ServerCtx::setup();
  pause();
  return ;
}


struct IpcBlock {
  size_t src_offset;
  size_t dst_offset;
  size_t length;
public:
  IpcBlock(size_t s, size_t d, size_t l) : src_offset(s), dst_offset(d), length(l) {}
};


static void do_parse_block_send(
  const size_t p_token_size,
  const size_t p_block_size,
  const std::vector<int>& p_blocks,
  const size_t d_token_size,
  const size_t d_block_size,
  const std::vector<int>& d_blocks,
  std::vector<IpcBlock> &send_blocks) {
  const uint32_t group_n = d_token_size / p_token_size;
  const uint32_t group_off = 1;
  // cache shape [num_gpu_blocks, block_size, 2, num_kv_heads, head_dim]
  const size_t p_k_size = p_token_size / 2;
  const size_t d_k_size = d_token_size / 2;
  // ntpb: number tokens per block
  const uint32_t ntpb = p_block_size / p_token_size;
  uint32_t left_tokens = ntpb * p_blocks.size();
  uint32_t wrote_tokens = 0;

  while (left_tokens > 0) {
    const uint32_t block_idx = wrote_tokens / ntpb;
    const uint32_t token_idx_base = wrote_tokens % ntpb;
    const size_t p_blk_off = p_blocks[block_idx] * p_block_size;
    const size_t d_blk_off = d_blocks[block_idx] * d_block_size;
    const uint32_t tokens = std::min(ntpb - token_idx_base, left_tokens);

    for (uint32_t idx = 0; idx < tokens; ++idx) {
      const uint32_t token_idx = token_idx_base + idx;
      const size_t pk_token_off = p_blk_off + token_idx * p_token_size;
      const size_t pv_token_off = pk_token_off + p_k_size;
      const size_t d_token_off = d_blk_off + token_idx * d_token_size;
      const size_t dk_token_off = d_token_off + group_off * p_k_size;
      const size_t dv_token_off = dk_token_off + d_k_size;
      send_blocks.emplace_back(pk_token_off, dk_token_off, p_k_size);
      send_blocks.emplace_back(pv_token_off, dv_token_off, p_k_size);
    }

    wrote_tokens += tokens;
    left_tokens -= tokens;
  }

  return;
}


static std::vector<IpcBlock> parse_block_send_p_lt_d() {
  std::vector<IpcBlock> send_blocks;
  auto d_token_size = env_srv_token_size();
  auto d_block_size = env_srv_block_size();
  const auto& d_blocks = env_dst_block_ids();
  auto p_token_size = env_token_size();
  auto p_block_size = env_block_size();
  const auto& p_blocks = env_src_block_ids();
  do_parse_block_send(d_token_size, d_block_size, d_blocks, p_token_size, p_block_size, p_blocks, send_blocks);
  for (auto& sb : send_blocks) {
    std::swap(sb.src_offset, sb.dst_offset);
  }
  return send_blocks;
}


std::tuple<uint64_t, uint64_t, uint64_t> method_writebatch(ClientCtx& ctx, const int rounds, std::vector<IpcBlock>& sb) {
  auto rkey = (uint32_t)env_server_rkey();
  auto rladdr = uint64_t(env_server_addr());

  uint64_t max = 0;
  uint64_t min = UINT64_MAX;
  uint64_t sum = 0;
  for (int i = 0; i < rounds; ++i) {
    auto now = std::chrono::steady_clock::now();
    auto datasp = std::make_shared<std::vector<rw_memp_t>>();
    for (const auto &[src_offset, dst_offset, len] : sb) {
      if (len <= 0) {
        continue;
      }

      auto src_mr = ctx.ctx.mr;
      src_mr.buf += src_offset;
      src_mr.buf_len = len;

      uint64_t raddr = rladdr + dst_offset;
      datasp->emplace_back(rw_memp_t{std::move(src_mr), raddr, rkey});
    }

    WriteBatch(ctx.chs(), std::move(datasp));
    uint64_t dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;
  }

  return {min, max, sum};
}


struct SgBlock {
  uint64_t decode_off;

};


// group 之后, input 呈现出:
// <src_off_1, dst_off_0, len1>
// <src_off_2, dst_off_0, len2>
// 这里意味着 src_off_1, len1; src_off_2, len2 的内容要写入 dst_off_0
// <src_off_3, dst_off_1, len3>
// <src_off_4, dst_off_1, len4>
// <src_off_5, dst_off_1, len5>
static void group_by_dst(std::vector<IpcBlock>& input) {
  assert(!input.empty());

  // input may be empty.
  std::sort(input.begin(), input.end(),
    [](const IpcBlock& x, const IpcBlock& y) { return x.dst_offset < y.dst_offset; });

  size_t prev_idx = 0;
  size_t prev_end = input[0].length + input[0].dst_offset;
  for (size_t idx = 1; idx < input.size(); ++idx) {
    auto& blk = input[idx];
    if (blk.dst_offset > prev_end) {
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

  return ;
}


[[nodiscard]] static std::future<void> WriteBySgList(XChannel* ch, uint64_t remote_addr, uint32_t rkey, std::shared_ptr<std::vector<memp_t>> prefills) {
  auto pr = std::make_shared<std::promise<void>>();
  auto fut = pr->get_future();
  auto& datas = *prefills;

  auto result = ch->WriteBySgList(datas,
    remote_addr, rkey,
    /* signal_peer */ false,
    /* imm_data */ 0,
    [prefills=std::move(prefills), pr=std::move(pr)] (Status s) {
      // WriteBySgList 要求 prefills 直至 callback 中才能回收.
      if (!s.IsOk()) {
        auto ex = std::make_exception_ptr(std::runtime_error("Write ERR: " + s.ErrMsg()));
        pr->set_exception(std::move(ex));
        return;
      }
      pr->set_value();
    });
  if (result != BAREX_SUCCESS) {
    auto ex = std::runtime_error("Write Submit Err: " + Status(result).ErrMsg());
    return make_exp_future<void>(std::move(ex));
  }

  return fut;
}


std::tuple<uint64_t, uint64_t, uint64_t> method_writebysglist(ClientCtx& ctx, const int rounds, std::vector<IpcBlock>& input) {
  auto rkey = (uint32_t)env_server_rkey();
  auto rladdr = uint64_t(env_server_addr());
  uint64_t max = 0;
  uint64_t min = UINT64_MAX;
  uint64_t sum = 0;
  if (input.empty()) {
    return {min, max, sum};
  }
  group_by_dst(input);

  std::vector<std::future<void>> futs;
  for (int i = 0; i < rounds; ++i) {
    assert(futs.empty());
    auto now = std::chrono::steady_clock::now();

    uint64_t dst_offset = input[0].dst_offset;
    auto prefills = std::make_shared<std::vector<memp_t>>();
    {
      auto src_mr = ctx.ctx.mr;
      src_mr.buf += input[0].src_offset;
      src_mr.buf_len = input[0].length;
      prefills->emplace_back(std::move(src_mr));
    }
    for (size_t idx = 1; idx < input.size(); ++idx) {
      const auto& blk = input[idx];
      if (blk.dst_offset == dst_offset) {
        auto src_mr = ctx.ctx.mr;
        src_mr.buf += blk.src_offset;
        src_mr.buf_len = blk.length;
        prefills->emplace_back(std::move(src_mr));
        continue;
      }
      assert(!prefills->empty());
      futs.emplace_back(WriteBySgList(ctx.ch(), dst_offset + rladdr, rkey, std::move(prefills)));

      dst_offset = blk.dst_offset;
      prefills = std::make_shared<std::vector<memp_t>>();
      {
        auto src_mr = ctx.ctx.mr;
        src_mr.buf += blk.src_offset;
        src_mr.buf_len = blk.length;
        prefills->emplace_back(std::move(src_mr));
      }
    }
    assert(!prefills->empty());
    futs.emplace_back(WriteBySgList(ctx.ch(), dst_offset + rladdr, rkey, std::move(prefills)));

    for (auto& fut : futs) {
      fut.get();
    }
    futs.clear();

    uint64_t dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;
  }

  return {min, max, sum};
}


std::tuple<uint64_t, uint64_t, uint64_t> method_copywrite(ClientCtx& ctx, const int rounds, std::vector<IpcBlock>& input) {
  auto rkey = (uint32_t)env_server_rkey();
  auto rladdr = uint64_t(env_server_addr());
  uint64_t max = 0;
  uint64_t min = UINT64_MAX;
  uint64_t sum = 0;
  if (input.empty()) {
    return {min, max, sum};
  }
  group_by_dst(input);

  for (int i = 0; i < rounds; ++i) {
    auto now = std::chrono::steady_clock::now();

    auto datasp = std::make_shared<std::vector<rw_memp_t>>();
    uint64_t dst_offset = input[0].dst_offset;
    char* src_start = ctx.cpu_mr.buf;
    char* src_end = ctx.cpu_mr.buf;
    for (size_t idx = 0; idx < input.size(); ++idx) {
      const auto& blk = input[idx];

      if (blk.dst_offset == dst_offset) {
        auto cuda_rt = cudaMemcpyAsync(src_end,
          ctx.ctx.mr.buf + blk.src_offset,
          blk.length,
          cudaMemcpyDeviceToHost,
          ctx.cpy_stream);
        RTCHECK(cuda_rt == cudaSuccess);
        src_end += blk.length;
        continue;
      }
      auto src_mr = ctx.cpu_mr;
      src_mr.buf = src_start;
      src_mr.buf_len = src_end - src_start;
      assert(src_mr.buf_len > 0);
      datasp->emplace_back(rw_memp_t{
        std::move(src_mr),
        dst_offset + rladdr,
        rkey
      });

      dst_offset = blk.dst_offset;
      src_start = src_end;
      --idx;
    }
    auto src_mr = ctx.cpu_mr;
    src_mr.buf = src_start;
    src_mr.buf_len = src_end - src_start;
    assert(src_mr.buf_len > 0);
    datasp->emplace_back(rw_memp_t{
      std::move(src_mr),
      dst_offset + rladdr,
      rkey
    });

    auto cuda_rt = cudaStreamSynchronize(ctx.cpy_stream);
    RTCHECK(cuda_rt == cudaSuccess);

    WriteBatch(ctx.chs(), std::move(datasp));

    uint64_t dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;
  }

  return {min, max, sum};
}


std::tuple<uint64_t, uint64_t, uint64_t> method_copygpuwrite(ClientCtx& ctx, const int rounds, std::vector<IpcBlock>& input) {
  auto rkey = (uint32_t)env_server_rkey();
  auto rladdr = uint64_t(env_server_addr());
  uint64_t max = 0;
  uint64_t min = UINT64_MAX;
  uint64_t sum = 0;
  if (input.empty()) {
    return {min, max, sum};
  }
  group_by_dst(input);

  for (int i = 0; i < rounds; ++i) {
    auto now = std::chrono::steady_clock::now();

    auto datasp = std::make_shared<std::vector<rw_memp_t>>();
    uint64_t dst_offset = input[0].dst_offset;
    char* src_start = ctx.gpu_mr.buf;
    char* src_end = ctx.gpu_mr.buf;
    for (size_t idx = 0; idx < input.size(); ++idx) {
      const auto& blk = input[idx];

      if (blk.dst_offset == dst_offset) {
        auto cuda_rt = cudaMemcpyAsync(src_end,
          ctx.ctx.mr.buf + blk.src_offset,
          blk.length,
          cudaMemcpyDeviceToDevice,
          ctx.cpy_stream);
        RTCHECK(cuda_rt == cudaSuccess);
        src_end += blk.length;
        continue;
      }
      auto src_mr = ctx.gpu_mr;
      src_mr.buf = src_start;
      src_mr.buf_len = src_end - src_start;
      assert(src_mr.buf_len > 0);
      datasp->emplace_back(rw_memp_t{
        std::move(src_mr),
        dst_offset + rladdr,
        rkey
      });

      dst_offset = blk.dst_offset;
      src_start = src_end;
      --idx;
    }
    auto src_mr = ctx.gpu_mr;
    src_mr.buf = src_start;
    src_mr.buf_len = src_end - src_start;
    assert(src_mr.buf_len > 0);
    datasp->emplace_back(rw_memp_t{
      std::move(src_mr),
      dst_offset + rladdr,
      rkey
    });

    auto cuda_rt = cudaStreamSynchronize(ctx.cpy_stream);
    RTCHECK(cuda_rt == cudaSuccess);

    WriteBatch(ctx.chs(), std::move(datasp));

    uint64_t dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;
  }

  return {min, max, sum};
}


void client_main() {
  auto ctx = ClientCtx::setup();
  int const rounds = env_rounds();
  auto sb = parse_block_send_p_lt_d();

  uint64_t max = 0;
  uint64_t min = 0;
  uint64_t sum = 0;
  const auto* method = env_method();
  if (strcasecmp(method, "writebatch") == 0) {
    auto ret = method_writebatch(ctx, rounds, sb);
    min = std::get<0>(ret);
    max = std::get<1>(ret);
    sum = std::get<2>(ret);
  } else if (strcasecmp(method, "writebysglist") == 0) {
    auto ret = method_writebysglist(ctx, rounds, sb);
    min = std::get<0>(ret);
    max = std::get<1>(ret);
    sum = std::get<2>(ret);
  } else if (strcasecmp(method, "copywrite") == 0) {
    auto ret = method_copywrite(ctx, rounds, sb);
    min = std::get<0>(ret);
    max = std::get<1>(ret);
    sum = std::get<2>(ret);
  } else if (strcasecmp(method, "copygpuwrite") == 0) {
    auto ret = method_copygpuwrite(ctx, rounds, sb);
    min = std::get<0>(ret);
    max = std::get<1>(ret);
    sum = std::get<2>(ret);
  }

  int round = env_rounds();
  printf("method=%s min_us=%f max_us=%f sum_us=%f, avg_us=%f\n",
    method,
    double(min) / 1000,
    double(max) / 1000,
    double(sum) / 1000,
    double(sum) / 1000 / round);
  _exit(0);  // 不要那么温和.
  return ;
}


int main(int argc, char** argv) {
  if (env_is_server()) {
    server_main();
  } else {
    client_main();
  }
  return 0;
}

#if 0

g++ -DCMAKE_INCLUDE barex_bench.cc -O2 -ggdb -DNDEBUG -laccl_barex  -std=gnu++17 -I/usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so -o kvbench

ZY_IS_SERVER=1 ZY_TOKEN_SIZE=1024 ZY_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=2 ZY_SERVER_PORT=33333 ZY_NIC_DEV=vsolar_1 ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=7 ZY_SERVER_PORT=33333 ZY_SERVER_IP=11.224.33.193 ZY_ROUNDS=80 ZY_METHOD=writebatch ZY_SERVER_ADDR=0xd3000000 ZY_SERVER_RKEY=3074 ZY_BLOCK_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51 ZY_NIC_DEV=vsolar_0  ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=7 ZY_SERVER_PORT=33333 ZY_SERVER_IP=11.224.33.193 ZY_ROUNDS=30 ZY_METHOD=writebatch ZY_SERVER_ADDR=0xd3000000 ZY_SERVER_RKEY=2048 ZY_BLOCK_IDS=0 ZY_NIC_DEV=vsolar_0  ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=7 ZY_SERVER_PORT=33333 ZY_SERVER_IP=11.224.33.193 ZY_ROUNDS=80 ZY_METHOD=writebysglist ZY_SERVER_ADDR=0xd3000000 ZY_SERVER_RKEY=3074 ZY_BLOCK_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51 ZY_NIC_DEV=vsolar_0  ./kvbench

===
ZY_IS_SERVER=1 ZY_TOKEN_SIZE=1024 ZY_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_SERVER_PORT=33333 ZY_GPU_ID=4  ZY_NIC_DEV=mlx5_4 ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_ROUNDS=80  ZY_METHOD=copywrite ZY_SERVER_PORT=33333 ZY_SERVER_IP=22.10.109.211 ZY_GPU_ID=7 ZY_NIC_DEV=mlx5_7 ZY_SERVER_ADDR=0x7fdb44800000 ZY_SERVER_RKEY=331331 ZY_BLOCK_IDS=0   ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_ROUNDS=80 ZY_BLOCK_IDS=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102 ZY_SERVER_PORT=33333 ZY_SERVER_IP=22.10.109.211 ZY_METHOD=copygpuwrite ZY_GPU_ID=7 ZY_NIC_DEV=mlx5_7 ZY_SERVER_ADDR=0x7f7a84800000 ZY_SERVER_RKEY=331374 ./kvbench

===== ppu

g++ -DCMAKE_INCLUDE barex_bench.cc -O2 -ggdb -DNDEBUG -laccl_barex  -std=gnu++17 -I/usr/local/cuda/targets/x86_64-linux/include -lcudart -o kvbench

ZY_IS_SERVER=1 ZY_TOKEN_SIZE=1024 ZY_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_SERVER_PORT=33333 ZY_GPU_ID=0  ZY_NIC_DEV=vsolar_1 ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_ROUNDS=80 ZY_BLOCK_IDS=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102 ZY_SERVER_PORT=33333 ZY_SERVER_IP=`hostname -i` ZY_METHOD=copywrite ZY_GPU_ID=7 ZY_NIC_DEV=vsolar_0 ZY_SERVER_ADDR=0xd3800000 ZY_SERVER_RKEY=0 ./kvbench

#endif

#endif