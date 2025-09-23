#if ENABLE_BAREX_BENCH

#include <accl/barex/barex.h>
#include <accl/barex/barex_types.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>
#include <accl/barex/xtimer.h>
#ifdef ENABLE_IBVERBS_BENCH
#include <barex/accl_verbs.h>
#include <barex/env.h>
#include <barex/common.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <future>
#include <chrono>
#include <assert.h>
#include <cuda_runtime.h>

static uint64_t cdiv(uint64_t a, uint64_t b) {
  return (a + (b - 1)) / b;
}

#define RTASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "Runtime error: Assertion failed in %s on line %d: %s\n", __FILE__, __LINE__, #expr); \
        abort(); \
    } \
} while (0)

using namespace accl::barex;

#ifdef ENABLE_IBVERBS_BENCH
class ZYXContext;

class ZYChannel : public XChannel {
  int local_nic_id_ = -1;
  int peer_nic_id_ = -1;
  std::string local_addr_;
  int local_port_;
  std::string peer_addr_;
  int peer_port_;
  TcpProtoType tcp_type_;
  std::atomic<uint64_t> lastest_alive_ts_;
  int traffic_class = 0;

  // not own
  ZYXContext *ctx_ = nullptr;
  // not own
  IbvDevice *ibv_device_ = nullptr;
  // not own
  Verbs *verbs_ = nullptr;
  // not own
  struct ibv_comp_channel *ibv_cc_ = nullptr;
  // not own
  struct ibv_cq *ibv_cq_ = nullptr;
  ChannelConfig config_;
  // own
  struct ibv_qp *ibv_qp_ = nullptr;
public:
  auto* qp() const noexcept {
    return ibv_qp_;
  }

  const auto& config() const noexcept {
    return config_;
  }
  auto* verbs() const noexcept {
    return verbs_;
  }

  ZYChannel(ZYXContext *ctx,
            IbvDevice *ibv_device,
            Verbs *verbs,
            ibv_comp_channel *ibv_cc,
            ibv_cq *ibv_cq,
            ChannelConfig config):
    ctx_(ctx)
    , ibv_device_(ibv_device)
    , verbs_(verbs)
    , ibv_cc_(ibv_cc)
    , ibv_cq_(ibv_cq)
    , config_(config) {}

  ~ZYChannel() {
    abort();
  }

  uint64_t GetMetric(const std::string &metric) override {
    abort();
    return 1;
  }

  std::shared_ptr<Stats> GetStats() override {
    abort();
    return nullptr;
  }

  BarexResult Send(
    memp_t data,
    bool auto_release = true,
    x_msg_header header = {0},
    DoneCallback done = [](Status s) {},
    bool done_inline = true) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult WriteSingle(
    memp_t data,
    uint64_t remote_addr,
    uint32_t rkey,
    bool signal_peer = true,
    uint32_t imm_data = 0,
    DoneCallback done = [](Status s) {},
    bool done_inline = true,
    uint64_t r_ttl_ms = UINT64_MAX) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult WriteBatch(
    std::vector<rw_memp_t> &datas, DoneCallback done = [](Status s) {}, bool done_inline = true) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult WriteBySgList(
    std::vector<memp_t> &datas,
    uint64_t remote_addr,
    uint32_t rkey,
    bool signal_peer = true,
    uint32_t imm_data = 0,
    DoneCallback done = [](Status s) {},
    bool done_inline = true,
    uint64_t r_ttl_ms = UINT64_MAX) override {
    abort();
    return BAREX_SUCCESS;
  }

  /** 使用RDMA write 探测对方是否活着 */
  BarexResult WriteHeartbeat(
    DoneCallback done = [](Status s) {}, bool done_inline = true) override {
    done(Status::OK());
    return BAREX_SUCCESS;
  }

  BarexResult ReadSingle(
    memp_t data,
    uint64_t remote_addr,
    uint32_t rkey,
    DoneCallback done = [](Status s) {},
    bool done_inline = true,
    uint64_t r_ttl_ms = UINT64_MAX) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult ReadBatch(
    std::vector<rw_memp_t> &datas, DoneCallback done = [](Status s) {}, bool done_inline = true) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult ReadBySgList(
    std::vector<memp_t> &datas,
    uint64_t remote_addr,
    uint32_t rkey,
    DoneCallback done = [](Status s) {},
    bool done_inline = true,
    uint64_t r_ttl_ms = UINT64_MAX) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult Incubate() override {
    ibv_pd *pd = ibv_device_->GetIbvPd();
    local_nic_id_ = ibv_device_->GetId();
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    // send/recv use the same cq
    qp_init_attr.send_cq = ibv_cq_;
    qp_init_attr.recv_cq = ibv_cq_;
    qp_init_attr.cap.max_send_wr = config_.soft_tx_depth;
    qp_init_attr.cap.max_recv_wr = config_.rx_depth;
    qp_init_attr.cap.max_send_sge = config_.max_sge;
    qp_init_attr.sq_sig_all = 0;
    if (Env::Instance().GetUseErdma()) {
        qp_init_attr.cap.max_recv_sge = ERDMA_MAX_RECV_SGE;
        qp_init_attr.cap.max_inline_data = ERDMA_MAX_INLINE_SIZE;
    } else {
        qp_init_attr.cap.max_recv_sge = config_.max_sge;
        qp_init_attr.cap.max_inline_data = BAREX_MAX_INLINE_DATA;
    }
    qp_init_attr.qp_type = IBV_QPT_RC;
    struct ibv_qp *qp = verbs_->IbvCreateQp(pd, &qp_init_attr);
    ibv_qp_ = qp;
    printf(">>>>>>> tx_depth=%u soft_tx_depth=%u qp=%p cp=%p\n", config_.tx_depth, config_.soft_tx_depth, qp, ibv_cq_);
    return BAREX_SUCCESS;
  }

  BarexResult GetInitData(ChannelInitMeta &out) override {
    SimpleQpInfo qp_info;
    qp_info.qp_num = ibv_qp_->qp_num;
    qp_info.psn = 0;
    qp_info.lid = ibv_device_->GetIbvPortAttr().lid;
    ibv_gid gid = ibv_device_->GetIbvGid(tcp_type_);

    memcpy(qp_info.gid, &gid, BAREX_GID_SIZE);
    qp_info.heartbeat_addr = 0;
    qp_info.heartbeat_key = 0;
    qp_info.nic_id = ibv_device_->GetId();

    out.context = (uint64_t)ctx_;
    // 用父类的指针
    XChannel *ch = this;
    out.channel = (uint64_t)ch;
    out.qp_infos.push_back(qp_info);

    return BAREX_SUCCESS;
  }

  BarexResult Init(ChannelInitMeta peer_init_meta) override {
    traffic_class = Env::Instance().GetTrafficClass();
    if (traffic_class) {
        LogInfo("XChannelImpl::Init, set traffic_class = %d, %s", traffic_class, ToString().c_str());
    }
    SimpleQpInfo &qp_info = peer_init_meta.qp_infos[0];
    peer_nic_id_ = qp_info.nic_id;
    uint32_t target_qp_num = qp_info.qp_num;
    uint16_t target_lid = qp_info.lid;
    uint8_t *dgid = qp_info.gid;

    int ret = 0;
    /* change QP state to INIT */
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = config_.ib_port;
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    ret = verbs_->IbvModifyQp(ibv_qp_, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    RTASSERT(ret == 0);
    /* Change QP state to RTR */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = config_.mtu;
    qp_attr.dest_qp_num = target_qp_num;
    qp_attr.rq_psn = qp_info.psn;
    qp_attr.max_dest_rd_atomic = config_.max_dest_rd_atomic;
    qp_attr.min_rnr_timer = config_.min_rnr_timer;
    if (target_lid == 0) { // RoCE
        qp_attr.ah_attr.is_global = 1;
    } else { // IB
        qp_attr.ah_attr.is_global = 0;
    }
    qp_attr.ah_attr.dlid = target_lid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = config_.ib_port;
    memcpy(&qp_attr.ah_attr.grh.dgid, dgid, 16);
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.hop_limit = 8;
    qp_attr.ah_attr.grh.sgid_index = ibv_device_->GetIbvGidIndex(tcp_type_);
    qp_attr.ah_attr.grh.traffic_class = traffic_class;

    ret = verbs_->IbvModifyQp(ibv_qp_,
                              &qp_attr,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    RTASSERT(ret == 0);

    /* Change QP state to RTS */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = config_.retransmit_timeout;
    qp_attr.retry_cnt = config_.retry_cnt;
    qp_attr.rnr_retry = config_.rnr_retry;
    qp_attr.sq_psn = 0;
    qp_attr.max_rd_atomic = config_.max_rd_atomic;
    ret = verbs_->IbvModifyQp(ibv_qp_,
                              &qp_attr,
                              IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                  IBV_QP_MAX_QP_RD_ATOMIC);
    RTASSERT(ret == 0);
    return BAREX_SUCCESS;
  }

  BarexResult GetCloseData(ChannelCloseMeta &out) override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult Close() override {
    abort();
    return BAREX_SUCCESS;
  }

  BarexResult Destroy() override {
    abort();
    return BAREX_SUCCESS;
  }

  bool IsActive() override { return true; }

  XChannel::ChannelState CurrentState() {
    abort();
    return XChannel::INIT_SUCCESS;
  }

  bool IsServerChannel() override {
    return config_.is_server_channel;
  }

  std::string ToString() override {
    return "zyxchannel";
  }

  XContext *GetContext() override {
    abort();
    return nullptr;
  }

  XSimpleMempool *GetMempool() override {
    abort();
    return nullptr;
  }

  BarexResult
  AllocBuffer(memp_t &out, uint64_t size, device_type d_type, int device_id = 0, void *attr = nullptr) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult
  AllocLimitBuffer(memp_t &out, uint64_t size, device_type d_type, int device_id = 0, void *attr = nullptr) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult
  AllocBuffersAsync(std::vector<mem_config> &mem_configs, std::function<void(std::vector<memp_t>, BarexResult)> done, int device_id = 0, bool limited = true, void *attr = nullptr) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult ReleaseBuffer(void *buf, device_type d_type) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult ReleaseAndDeregBuffer(void *to_free, device_type d_type) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult FindBufferMr(memp_t &out, void *buf, device_type d_type, int nic_id = -1, bool adopt = false) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult RegUserMr(void *buf, uint64_t size, device_type d_type, int device_id = 0) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult RegUserMr(memp_t &out, void *buf, uint64_t size, device_type d_type, int device_id = 0) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult DeregUserMr(void *buf, device_type d_type) override {
    abort();
    return BAREX_SUCCESS;
  }
  void* GetMemptMrAddr(const memp_t& mem) override {
    abort();
    return nullptr;
  }
  uint32_t GetMemptMrRkey(const memp_t& mem) override {
    abort();
    return 0;
  }
  bool SetUserData(std::string key, std::shared_ptr<XChannel::UserData> data) override {
    abort();
    return false;
  }
  std::shared_ptr<XChannel::UserData> GetUserData(std::string key) override {
    abort();
    return nullptr;
  }
  std::shared_ptr<XChannel::UserData> RemoveUserData(std::string key) override {
    abort();
    return nullptr;
  }

  bool NoInflightWorkRequest() override {
    abort();
    return false;
  }

  void SetLocalAddrAndPort(const std::string &addr, int port) override {
    local_addr_ = addr;
    local_port_ = port;
  }

  void SetPeerAddrAndPort(const std::string &addr, int port) override {
    peer_addr_ = addr;
    peer_port_ = port;
  }

  void SetTcpType(TcpProtoType tcp_type) override {
    tcp_type_ = tcp_type;
  }

  TcpProtoType GetTcpType() override {
    return tcp_type_;
  }

  std::pair<std::string, int> GetPeerAddrAndPort() override { return std::make_pair<>(peer_addr_, peer_port_); }

  std::pair<std::string, int> GetLocalAddrAndPort() override { return std::make_pair<>(local_addr_, local_port_); }

  void UpdateLastestAliveTime() override { lastest_alive_ts_.store(current_microseconds()); }

  int GetLocalNicId() override { return local_nic_id_; }

  int GetPeerNicId() override { return peer_nic_id_; };

  ConnectedHook GetConnectedHook() override {
    return nullptr;
  }
  bool CheckConfig(ChannelConfig &config) override {
    abort();
    return true;
  }
};

class ZYXContext : public XContext {
  ContextConfig config_;
  // not own
  IbvDevice *ibv_device_ = nullptr;
  Verbs *const verbs_ = new RealVerbs();
  struct ibv_cq *ibv_cq_ = nullptr;
private:

  void MergeEnvConfig() {
    Env &env = Env::Instance();
    uint32_t env_tx_depth = env.GetTxDepth();
    uint32_t env_soft_tx_depth = env.GetSoftTxDepth();
    uint32_t env_rx_depth = env.GetRxDepth();
    uint32_t env_max_sge = env.GetMaxSge();
    uint32_t env_min_rnr_timer = env.GetMinRnrTimer();
    uint32_t env_rnr_retry = env.GetRnrRetry();
    uint32_t env_retransmit_timeout = env.GetRetransmitTimeout();
    uint32_t env_retry_cnt = env.GetRetryCnt();
    uint32_t env_small_msg_size = env.GetSmallMsgSize();
    uint32_t env_ib_port = env.GetIbPort();
    uint32_t env_cq_entries = env.GetContextCqe();
    uint32_t env_busy_poll_count = env.GetBusyPollCnt();
    uint32_t env_post_recv_batch_size = env.GetPostRecvBatchSize();
    TimerTick env_heartbeat_interval = env.GetHeartbeatInTick();
    ibv_mtu env_ibv_mtu = env.GetIbvMtu();
    uint32_t env_max_dest_rd_atomic = env.GetMaxDestRdAtomic();
    uint32_t env_max_rd_atomic = env.GetMaxRdAtomic();

    if (env.IsTxDepthSet() && env_tx_depth != config_.tx_depth) {
        LogInfo("replace tx_depth(%u) with env value: %u, %s", config_.tx_depth, env_tx_depth, ToString().c_str());
        config_.tx_depth = env_tx_depth;
    }
    if (env.IsSoftTxDepthSet() && env_soft_tx_depth != config_.soft_tx_depth) {
        LogInfo("replace soft_tx_depth(%u) with env value: %u, %s", config_.soft_tx_depth, env_soft_tx_depth, ToString().c_str());
        config_.soft_tx_depth = env_soft_tx_depth;
    }
    if (env.IsRxDepthSet() && env_rx_depth != config_.rx_depth) {
        LogInfo("replace rx_depth(%u) with env value: %u, %s", config_.rx_depth, env_rx_depth, ToString().c_str());
        config_.rx_depth = env_rx_depth;
    }
    if (env.IsMaxSgeSet() && env_max_sge != config_.max_sge) {
        LogInfo("replace max_sge(%u) with env value: %u, %s", config_.max_sge, env_max_sge, ToString().c_str());
        config_.max_sge = env_max_sge;
    }
    if (env.IsMinRnrTimerSet() && env_min_rnr_timer != config_.min_rnr_timer) {
        LogInfo("replace min_rnr_timer(%u) with env value: %u, %s",
                config_.min_rnr_timer,
                env_min_rnr_timer,
                ToString().c_str());
        config_.min_rnr_timer = env_min_rnr_timer;
    }
    if (env.IsRnrRetrySet() && env_rnr_retry != config_.rnr_retry) {
        LogInfo("replace rnr_retry(%u) with env value: %u, %s", config_.rnr_retry, env_rnr_retry, ToString().c_str());
        config_.rnr_retry = env_rnr_retry;
    }
    if (env.IsRetransmitTimeoutSet() && env_retransmit_timeout != config_.retransmit_timeout) {
        LogInfo("replace retransmit_timeout(%u) with env value: %u, %s",
                config_.retransmit_timeout,
                env_retransmit_timeout,
                ToString().c_str());
        config_.retransmit_timeout = env_retransmit_timeout;
    }
    if (env.IsRetryCntSet() && env_retry_cnt != config_.retry_cnt) {
        LogInfo("replace retry_cnt(%u) with env value: %u, %s", config_.retry_cnt, env_retry_cnt, ToString().c_str());
        config_.retry_cnt = env_retry_cnt;
    }
    if (env.IsSmallMsgSizeSet() && env_small_msg_size != config_.small_msg_size) {
        LogInfo("replace small_msg_size(%u) with env value: %u, %s",
                config_.small_msg_size,
                env_small_msg_size,
                ToString().c_str());
        config_.small_msg_size = env_small_msg_size;
    }
    if (env.IsIbPortSet() && env_ib_port != config_.ib_port) {
        LogInfo("replace ib_port(%u) with env value: %u, %s", config_.ib_port, env_ib_port, ToString().c_str());
        config_.ib_port = env_ib_port;
    }
    if (env.IsContextCqeSet() && env_cq_entries != config_.cq_entries) {
        LogInfo(
            "replace cq_entries(%u) with env value: %u, %s", config_.cq_entries, env_cq_entries, ToString().c_str());
        config_.cq_entries = env_cq_entries;
    }
    if (env.IsBusyPollCntSet() && env_busy_poll_count != config_.busy_poll_count) {
        LogInfo("replace busy_poll_count(%u) with env value: %u, %s",
                config_.busy_poll_count,
                env_busy_poll_count,
                ToString().c_str());
        config_.busy_poll_count = env_busy_poll_count;
    }
    if (env.IsPostRecvBatchSizeSet() && env_post_recv_batch_size != config_.post_recv_batch_size) {
        LogInfo("replace post_recv_batch_size(%u) with env value: %u, %s",
                config_.post_recv_batch_size,
                env_post_recv_batch_size,
                ToString().c_str());
        config_.post_recv_batch_size = env_post_recv_batch_size;
    }
    if (env.IsHeartbeatTickSet() && env_heartbeat_interval != config_.heartbeat_interval) {
        LogInfo("replace heartbeat_interval(%u) with env value: %u, %s",
                config_.heartbeat_interval,
                env_heartbeat_interval,
                ToString().c_str());
        config_.heartbeat_interval = env_heartbeat_interval;
    }
    if (env.IsIbvMtuSet() && env_ibv_mtu != config_.mtu) {
        LogInfo("replace ibv_mtu(%u) with env value: %u, %s",
                config_.mtu,
                env_ibv_mtu,
                ToString().c_str());
        config_.mtu = env_ibv_mtu;
    }
    if (env.IsMaxDestRdAtomicSet() && env_max_dest_rd_atomic != config_.max_dest_rd_atomic) {
        LogInfo("replace max_dest_rd_atomic(%u) with env value: %u, %s", config_.max_dest_rd_atomic, env_max_dest_rd_atomic, ToString().c_str());
        config_.max_dest_rd_atomic = env_max_dest_rd_atomic;
    }
    if (env.IsMaxRdAtomicSet() && env_max_rd_atomic != config_.max_rd_atomic) {
        LogInfo("replace max_rd_atomic(%u) with env value: %u, %s", config_.max_rd_atomic, env_max_rd_atomic, ToString().c_str());
        config_.max_rd_atomic = env_max_rd_atomic;
    }
  }

public:
  ZYXContext(ContextConfig config, IbvDevice* dev): config_(config), ibv_device_(dev) {
  }
  BarexResult Init() override {
    MergeEnvConfig();
    auto* ibv_ctx = ibv_device_->GetIbvContext();
    ibv_cq_ = verbs_->IbvCreateCq(ibv_ctx, config_.cq_entries, nullptr, nullptr, 0);
    return BAREX_SUCCESS;
  }
  BarexResult Start() override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult Shutdown() override {
    abort();
    return BAREX_SUCCESS;
  }
  bool IsStopped() override {
    abort();
    return false;
  }
  BarexResult WaitStop() override {
    pause();
    return BAREX_SUCCESS;
  }
  uint64_t GetMetric(const std::string &metric) override {
    abort();
    return 0;
  }
  BarexResult SpawnChannel(XChannel *&out, bool is_server) override {
    ChannelConfig ch_config;
    ch_config.tx_depth = config_.tx_depth;
    ch_config.soft_tx_depth = config_.soft_tx_depth;
    ch_config.rx_depth = config_.rx_depth;
    ch_config.post_recv_batch_size = config_.post_recv_batch_size;
    ch_config.max_sge = config_.max_sge;
    ch_config.min_rnr_timer = config_.min_rnr_timer;
    ch_config.retransmit_timeout = config_.retransmit_timeout;
    ch_config.retry_cnt = config_.retry_cnt;
    ch_config.rnr_retry = config_.rnr_retry;
    ch_config.small_msg_size = config_.small_msg_size;
    ch_config.ib_port = config_.ib_port;
    ch_config.max_dest_rd_atomic = config_.max_dest_rd_atomic;
    ch_config.max_rd_atomic = config_.max_rd_atomic;
    ch_config.auto_release_recv_buf = config_.auto_release_recv_buf;
    ch_config.heartbeat_interval = config_.heartbeat_interval;
    ch_config.mtu = config_.mtu;
    ch_config.is_server_channel = is_server;
    auto* channel = new ZYChannel(this, ibv_device_, verbs_, nullptr, ibv_cq_, ch_config);
    out = channel;
    return BAREX_SUCCESS;
  }
  BarexResult DestroyChannel(XChannel *channel) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult DeleteVirginChannel(XChannel *channel) override {
    abort();
    return BAREX_SUCCESS;
  }
  bool IsValidChannel(XChannel *channel) override {
    abort();
    return false;
  }
  int ActiveChannelsCount() override {
    abort();
    return 0;
  }
  int DestroyedChannelsCount() override {
    abort();
    return 0;
  }
  int GetEventFd() override {
    abort();
    return -1;
  }
  int ProgressEvents() override {
    abort();
    return -1;
  }
  XDevice *GetXDevice() override {
    abort();
    return nullptr;
  }
  XThreadpool *SetThreadpool(XThreadpool *threadpool) override {
    abort();
    return nullptr;
  }
  XSimpleMempool *GetMempool() override {
    abort();
    return nullptr;
  }
  BarexResult SetChannelConnectedHook(ConnectedHook hook) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult SetChannelClosedHook(ClosedHook hook) override {
    abort();
    return BAREX_SUCCESS;
  }
  std::string ToString() override {
    return "zyxcontext";
  }
  BarexResult AddTimer(uint64_t &timer_id, TimerTick time, TimerCall call) override {
    abort();
    return BAREX_SUCCESS;
  }
  BarexResult AddRepeatTimer(uint64_t &timer_id, TimerTick time, RepeatTimerCall call) override {
    abort();
    return BAREX_SUCCESS;
  }
  bool CancelOneTimer(uint64_t timer_id) override {
    abort();
    return true;
  }
  bool CheckConfig(ContextConfig &config) override {
    abort();
    return false;
  }
  bool IsActive() override {
    return true;
  }
  auto* cq() const noexcept {
    return ibv_cq_;
  }
  auto* verbs() const noexcept {
    return verbs_;
  }
};
#endif

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
  RTASSERT(envval != nullptr);
  int ret = atoi(envval);
  RTASSERT(ret > 0);
  return ret;
}


static size_t env_block_size() {
  const char* envval = getenv("ZY_BLOCK_SIZE");
  RTASSERT(envval != nullptr);
  auto ret = atol(envval);
  RTASSERT(ret > 0);
  return ret;
}


static size_t env_srv_token_size() {
  const char* envval = getenv("ZY_SRV_TOKEN_SIZE");
  RTASSERT(envval != nullptr);
  int ret = atoi(envval);
  RTASSERT(ret > 0);
  return ret;
}


static size_t env_srv_block_size() {
  const char* envval = getenv("ZY_SRV_BLOCK_SIZE");
  RTASSERT(envval != nullptr);
  auto ret = atol(envval);
  RTASSERT(ret > 0);
  return ret;
}


static size_t env_block_number() {
  const char* envval = getenv("ZY_BLOCK_NUM");
  RTASSERT(envval != nullptr);
  auto ret = atol(envval);
  RTASSERT(ret > 0);
  return ret;
}


static int env_gpu_id() {
  const char* envval = getenv("ZY_GPU_ID");
  RTASSERT(envval != nullptr);
  int ret = atoi(envval);
  RTASSERT(ret >= 0);
  return ret;
}


static int env_server_port() {
  const char* envval = getenv("ZY_SERVER_PORT");
  RTASSERT(envval != nullptr);
  int ret = atoi(envval);
  RTASSERT(ret > 0);
  return ret;
}


static const char* env_server_ip() {
  const char* envval = getenv("ZY_SERVER_IP");
  RTASSERT(envval != nullptr);
  return envval;
}

static int env_is_server() {
  const char* envval = getenv("ZY_IS_SERVER");
  RTASSERT(envval != nullptr);
  int ret = atoi(envval);
  return ret;
}


static int env_rounds() {
  const char* envval = getenv("ZY_ROUNDS");
  RTASSERT(envval != nullptr);
  int ret = atoi(envval);
  RTASSERT(ret > 0);
  return ret;
}


static const char* env_method() {
  const char* envval = getenv("ZY_METHOD");
  RTASSERT(envval != nullptr);
  return envval;
}


static long env_server_addr() {
  const char* envval = getenv("ZY_SERVER_ADDR");
  RTASSERT(envval != nullptr);
  auto ret = strtol(envval, (char **)NULL, 0);
  RTASSERT(ret > 0);
  return ret;
}


static long env_server_rkey() {
  const char* envval = getenv("ZY_SERVER_RKEY");
  RTASSERT(envval != nullptr);
  auto ret = strtol(envval, (char **)NULL, 0);
  RTASSERT(ret >= 0);
  return ret;
}


static std::vector<int> env_src_block_ids() {
  const char* envval = getenv("ZY_BLOCK_IDS");
  RTASSERT(envval != nullptr);
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
  RTASSERT(result.size() > 0);
  if (result.size() == 1) {
    auto n = result[0];
    result.clear();
    for (int i = 0; i < n; ++i) {
      result.emplace_back(i);
    }
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
  auto datas = datasp;

  auto result = ch->WriteBatch(std::move(datas),
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
  RTASSERT(result == BAREX_SUCCESS);

  const auto& all_nic_devs = ret.manager->AllDevices();
  RTASSERT(!all_nic_devs.empty());
  const char* nicdev_name = getenv("ZY_NIC_DEV");
  RTASSERT(nicdev_name != nullptr);
  for (const auto& dev : all_nic_devs) {
    if (dev->GetName() == nicdev_name) {
      ret.nic_dev = dev;
      break;
    }
  }
  RTASSERT(ret.nic_dev != nullptr);

  result = XSimpleMempool::NewInstance(ret.mp, "mp", {ret.nic_dev});
  RTASSERT(result == BAREX_SUCCESS);

  int gpuid = env_gpu_id();
  auto cuda_rt = cudaSetDevice(gpuid);
  RTASSERT(cuda_rt == cudaSuccess);

  XAllocator* gpu_allocator = nullptr;
  result = ret.mp->GetXAllocator(gpu_allocator, GPU);
  RTASSERT(result == BAREX_SUCCESS);
  size_t kvcache_size = env_block_size() * env_block_number();
  void* const buf = gpu_allocator->Alloc(kvcache_size, gpuid, nullptr /* attr */, 512 /* align */);
  ret.mp->RegUserMr(ret.mr, buf, kvcache_size, GPU, gpuid);
  printf(">>>>>>> RegUserMr. buf=%p bufsize=%lu gpuid=%d rkey=%u\n", buf, kvcache_size, gpuid, ret.mr.mr->rkey);

  result = XThreadpool::NewInstance(ret.tp, 4, "tp");
  RTASSERT(result == BAREX_SUCCESS);

  ContextConfig config = XConfigUtil::DefaultContextConfig();
#ifdef ENABLE_IBVERBS_BENCH
  ret.xctx = new ZYXContext(config, dynamic_cast<IbvDevice *>(ret.nic_dev));
  ret.xctx->Init();
#else
  result = XContext::NewInstance(ret.xctx, config, new FackCb(), ret.nic_dev, ret.mp, ret.tp);
  RTASSERT(result == BAREX_SUCCESS);
  ret.xctx->Start();
#endif

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
  RTASSERT(result == BAREX_SUCCESS);
  result = ret.listener->Listen();
  RTASSERT(result == BAREX_SUCCESS);
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
  RTASSERT(cuda_rt == cudaSuccess);

  XAllocator* cpu_alloc = nullptr;
  auto result = ret.ctx.mp->GetXAllocator(cpu_alloc, CPU);
  RTASSERT(result == BAREX_SUCCESS);
  size_t cpu_size = env_srv_block_size() * env_block_number();
  auto* cpu_buf = cpu_alloc->Alloc(cpu_size);
  result = ret.ctx.mp->RegUserMr(ret.cpu_mr, cpu_buf, cpu_size, CPU);
  RTASSERT(result == BAREX_SUCCESS);

  XAllocator* gpu_allocator = nullptr;
  result = ret.ctx.mp->GetXAllocator(gpu_allocator, GPU);
  RTASSERT(result == BAREX_SUCCESS);
  void* const buf = gpu_allocator->Alloc(cpu_size, gpuid, nullptr /* attr */, 512 /* align */);
  result = ret.ctx.mp->RegUserMr(ret.gpu_mr, buf, cpu_size, GPU, gpuid);
  RTASSERT(result == BAREX_SUCCESS);

  result = XConnector::NewInstance(ret.connector, 2, TIMER_3S, {ret.ctx.xctx});
  RTASSERT(result == BAREX_SUCCESS);

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

#ifdef ENABLE_IBVERBS_BENCH
struct RWInfo {
  ibv_sge local;
  uint64_t raddr;
  uint32_t rkey;
};

class QPState {
  const std::vector<RWInfo>& rwinfo_;
  const uint64_t start_;
  const uint64_t end_;
  ZYChannel* const ch_;
  const uint64_t qpid_;
  uint64_t sent_;  // < sent_ 为已经 post send 的.
  uint64_t acked_;  // < acked_ 为 post send ack 的.
  std::vector<ibv_sge> sges_;
  std::vector<ibv_send_wr> wrs_;
public:

  QPState(uint64_t qpid, ZYChannel* ch, const std::vector<RWInfo>& rwinfo, uint64_t start, uint64_t end):
    rwinfo_(rwinfo), start_(start), end_(end), ch_(ch), qpid_(qpid << 48) {
    size_t wrsize = ch_->config().soft_tx_depth;
    sges_.resize(wrsize);
    wrs_.resize(wrsize);
    assert(qpid < 32);
    assert(!sges_.empty());
    memset(wrs_.data(), 0, wrsize * sizeof(ibv_send_wr));
    memset(sges_.data(), 0, wrsize * sizeof(ibv_sge));
    size_t i = 0;
    for (; i + 1 < wrsize; ++i) {
      wrs_[i].next = &wrs_[i + 1];
      wrs_[i].sg_list = &sges_[i];
      wrs_[i].num_sge = 1;
      wrs_[i].opcode = IBV_WR_RDMA_WRITE;
    }
    assert(i == wrsize - 1);
    wrs_[i].next = nullptr;
    wrs_[i].sg_list = &sges_[i];
    wrs_[i].num_sge = 1;
    wrs_[i].opcode = IBV_WR_RDMA_WRITE;
    rewind();
  }

  bool finished() const noexcept {
    return acked_ >= end_;
  }

  void rewind() noexcept {
    sent_ = start_;
    acked_ = start_;
  }

  void ack(uint64_t acked) noexcept {
    assert(acked > start_);
    assert(acked <= end_);
    acked_ = acked;
  }

  void send(uint64_t cap) noexcept {
    auto& self = *this;
    const auto& cfg = self.ch_->config();
    assert(cfg.soft_tx_depth % cfg.tx_depth == 0);
    assert(cap == cfg.soft_tx_depth || cap == cfg.tx_depth);
    if (self.sent_ >= self.end_) {
      return;
    }

    uint64_t const n = std::min(cap, self.end_ - self.sent_);
    uint64_t const sent_end = n + self.sent_;
    assert(sent_end <= self.end_);
    uint32_t signed_cnt = cdiv(self.end_ - sent_end, cfg.tx_depth);
    uint32_t signed_wrcnt = 0;
    size_t start_idx = self.wrs_.size() - n;
    auto* const start_wr = &self.wrs_[start_idx];

    for (; self.sent_ < sent_end; ++self.sent_, ++start_idx) {
      auto& wr = self.wrs_[start_idx];
      const auto& wrinfo = self.rwinfo_[self.sent_];
      *wr.sg_list = wrinfo.local;
      wr.wr.rdma.remote_addr = wrinfo.raddr;
      wr.wr.rdma.rkey = wrinfo.rkey;
      ++signed_wrcnt;
      if (signed_wrcnt == cfg.tx_depth && signed_cnt > 0) {
        --signed_cnt;
        signed_wrcnt = 0;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr_id = self.qpid_ | self.sent_;
      } else {
        wr.send_flags = 0;
      }
    }
    assert(start_idx == self.wrs_.size());
    if (self.sent_ >= self.end_) {
      auto& wr = self.wrs_.back();
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr_id = self.qpid_ | self.sent_;
    }

    struct ibv_send_wr* bad_wr = nullptr;
    int ret = self.ch_->verbs()->IbvPostSend(self.ch_->qp(), start_wr, &bad_wr);
    RTASSERT(ret == 0);
    return;
  }

  const auto& config() const noexcept {
    return ch_->config();
  }
};

static bool finished(const std::vector<QPState>& qps) noexcept {
  for (const auto& qp : qps) {
    if (!qp.finished()) {
      return false;
    }
  }
  return true;
}

static void do_writebatch(ZYXContext& ctx, std::vector<QPState>& qps) {
  const auto& cfg = qps.front().config();
  for (auto& qp : qps) {
    qp.send(cfg.soft_tx_depth);
  }
  uint64_t finished_qp = 0;
  static constexpr int wccap = 256;
  struct ibv_wc wcs[wccap];
  while (true) {
    int ret = ctx.verbs()->IbvPollCq(ctx.cq(), wccap, wcs);
    RTASSERT(ret >= 0);
    for (int wcidx = 0; wcidx < ret; ++wcidx) {
      const auto& wc = wcs[wcidx];
      RTASSERT(wc.opcode == IBV_WC_RDMA_WRITE);
      RTASSERT(wc.status == IBV_WC_SUCCESS);
      uint64_t qpidx = wc.wr_id >> 48;
      uint64_t acked = wc.wr_id & ((1ul << 48) - 1);
      auto& qp = qps[qpidx];
      qp.ack(acked);
      if (qp.finished()) {
        ++finished_qp;
        if (finished_qp == qps.size()) {
          return;
        }
        continue;
      }
      qp.send(cfg.tx_depth);
    }
    if (ret < wccap) {
      sched_yield();
    }
  }
}
#endif


std::tuple<uint64_t, uint64_t, uint64_t> method_writebatch(ClientCtx& ctx, const int rounds, std::vector<IpcBlock>& sb) {
  auto rkey = (uint32_t)env_server_rkey();
  auto rladdr = uint64_t(env_server_addr());

#ifdef ENABLE_IBVERBS_BENCH
  ZYXContext* zyctx = dynamic_cast<ZYXContext*>(ctx.ctx.xctx);

  std::vector<RWInfo> rwinfos;
  rwinfos.reserve(sb.size());
  size_t sb_min = UINT64_MAX, sb_max = 0, sb_total = 0;
  auto prepare_now = std::chrono::steady_clock::now();
  for (const auto& sb1 : sb) {
    sb_min = std::min(sb_min, sb1.length);
    sb_max = std::max(sb_max, sb1.length);
    sb_total += sb1.length;

    auto& rwinfo = rwinfos.emplace_back();
    rwinfo.local.addr = uint64_t(ctx.ctx.mr.buf + sb1.src_offset);
    rwinfo.local.length = sb1.length;
    rwinfo.local.lkey = ctx.ctx.mr.mr->lkey;
    rwinfo.raddr = rladdr + sb1.dst_offset;
    rwinfo.rkey = rkey;
  }
  auto prepare_end = std::chrono::steady_clock::now();
  uint64_t dur_ns = std::chrono::nanoseconds(prepare_end - prepare_now).count();
  printf(">>>>>> sb_min=%lu sb_max=%lu sb_total=%lu sb_cnt=%lu dur_ns=%lu\n", sb_min, sb_max, sb_total, sb.size(), dur_ns);

  std::vector<QPState> qps;
  qps.reserve(ctx.chs().size());
  uint64_t const qpsize = cdiv(rwinfos.size(), ctx.chs().size());
  for (uint64_t qpidx = 0; qpidx < ctx.chs().size(); ++qpidx) {
    uint64_t const rwstart = qpidx * qpsize;
    assert(rwstart < rwinfos.size());
    uint64_t const rwcnt = std::min(qpsize, rwinfos.size() - rwstart);
    auto* zyc = dynamic_cast<ZYChannel*>(ctx.chs()[qpidx]);
    qps.emplace_back(qpidx, zyc, rwinfos, rwstart, rwstart + rwcnt);
  }
  uint64_t max = 0;
  uint64_t min = UINT64_MAX;
  uint64_t sum = 0;
  for (int i = 0; i < rounds; ++i) {
    auto now = std::chrono::steady_clock::now();
    do_writebatch(*zyctx, qps);
    uint64_t dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;

    for (auto& qp : qps) {
      qp.rewind();
    }
  }
#else
  size_t sb_min = UINT64_MAX, sb_max = 0, sb_total = 0;
  for (const auto& sb1 : sb) {
    sb_min = std::min(sb_min, sb1.length);
    sb_max = std::max(sb_max, sb1.length);
    sb_total += sb1.length;
  }
  printf(">>>>>> sb_min=%lu sb_max=%lu sb_total=%lu sb_cnt=%lu\n", sb_min, sb_max, sb_total, sb.size());

  uint64_t max = 0;
  uint64_t min = UINT64_MAX;
  uint64_t sum = 0;
  uint64_t prepare_max = 0;
  uint64_t prepare_min = UINT64_MAX;
  uint64_t prepare_sum = 0;
  size_t const sbperch = cdiv(sb.size(), ctx.chs().size());
  std::vector<std::future<void>> futs;
  futs.reserve(ctx.chs().size());
  for (int i = 0; i < rounds; ++i) {
    auto now = std::chrono::steady_clock::now();

    auto datasp = std::make_shared<std::vector<rw_memp_t>>();
    datasp->reserve(sbperch);
    for (const auto &[src_offset, dst_offset, len] : sb) {
      if (len <= 0) {
        continue;
      }

      auto rwmemp = rw_memp_t();
      rwmemp.r_addr = rladdr + dst_offset;
      rwmemp.r_key = rkey;
      rwmemp.sg.addr = uintptr_t(ctx.ctx.mr.buf) + src_offset;
      rwmemp.sg.length = len;
      rwmemp.sg.lkey = ctx.ctx.mr.mr->lkey;
      datasp->emplace_back(std::move(rwmemp));
      if (datasp->size() >= sbperch) {
        auto fut = WriteBatch(ctx.ch(), std::move(datasp));
        futs.emplace_back(std::move(fut));
        datasp = std::make_shared<std::vector<rw_memp_t>>();
        datasp->reserve(sbperch);
      }
    }
    if (!datasp->empty()) {
      auto fut = WriteBatch(ctx.ch(), std::move(datasp));
      futs.emplace_back(std::move(fut));
    }
    auto copy_end = std::chrono::steady_clock::now();

    for (auto& fut : futs) {
      fut.get();
    }
    auto send_end = std::chrono::steady_clock::now();
    uint64_t dur_ns = std::chrono::nanoseconds(send_end - copy_end).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;
    uint64_t prepare_ns = std::chrono::nanoseconds(copy_end - now).count();
    prepare_max = std::max(prepare_max, prepare_ns);
    prepare_min = std::min(prepare_min, prepare_ns);
    prepare_sum += prepare_ns;
    futs.clear();
  }
  printf(">>>>>> prepare_min_us=%f prepare_max_us=%f prepare_avg_us=%f\n",
        prepare_min / 1000.0, prepare_max / 1000.0, prepare_sum / 1000.0 / rounds);
#endif

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
        RTASSERT(cuda_rt == cudaSuccess);
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
    RTASSERT(cuda_rt == cudaSuccess);

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

  uint64_t sync_max = 0;
  uint64_t sync_min = UINT64_MAX;
  uint64_t sync_sum = 0;
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
        RTASSERT(cuda_rt == cudaSuccess);
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
    RTASSERT(cuda_rt == cudaSuccess);
    uint64_t sync_dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    sync_max = std::max(sync_max, sync_dur_ns);
    sync_min = std::min(sync_min, sync_dur_ns);
    sync_sum += sync_dur_ns;

    WriteBatch(ctx.chs(), std::move(datasp));

    uint64_t dur_ns = std::chrono::nanoseconds(std::chrono::steady_clock::now() - now).count();
    max = std::max(max, dur_ns);
    min = std::min(min, dur_ns);
    sum += dur_ns;
  }

  printf(">>>>>> sync_min_us=%f sync_max_us=%f sync_sum_us=%f, sync_avg_us=%f\n",
    double(sync_min) / 1000,
    double(sync_max) / 1000,
    double(sync_sum) / 1000,
    double(sync_sum) / 1000 / rounds);
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
  printf(">>>>>>> method=%s min_us=%f max_us=%f sum_us=%f, avg_us=%f\n",
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

g++ -DENABLE_BAREX_BENCH -DCMAKE_INCLUDE barex_bench.cc -O2 -ggdb -DNDEBUG -laccl_barex  -std=gnu++17 -I/usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so -o kvbench

ZY_IS_SERVER=1 ZY_TOKEN_SIZE=1024 ZY_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=2 ZY_SERVER_PORT=33333 ZY_NIC_DEV=vsolar_1 ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=7 ZY_SERVER_PORT=33333 ZY_SERVER_IP=11.224.33.193 ZY_ROUNDS=80 ZY_METHOD=writebatch ZY_SERVER_ADDR=0xd3000000 ZY_SERVER_RKEY=3074 ZY_BLOCK_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51 ZY_NIC_DEV=vsolar_0  ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=7 ZY_SERVER_PORT=33333 ZY_SERVER_IP=11.224.33.193 ZY_ROUNDS=30 ZY_METHOD=writebatch ZY_SERVER_ADDR=0xd3000000 ZY_SERVER_RKEY=2048 ZY_BLOCK_IDS=0 ZY_NIC_DEV=vsolar_0  ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_GPU_ID=7 ZY_SERVER_PORT=33333 ZY_SERVER_IP=11.224.33.193 ZY_ROUNDS=80 ZY_METHOD=writebysglist ZY_SERVER_ADDR=0xd3000000 ZY_SERVER_RKEY=3074 ZY_BLOCK_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51 ZY_NIC_DEV=vsolar_0  ./kvbench

===
ZY_IS_SERVER=1 ZY_TOKEN_SIZE=1024 ZY_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_SERVER_PORT=33333 ZY_GPU_ID=4  ZY_NIC_DEV=mlx5_4 ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_ROUNDS=80  ZY_METHOD=copywrite ZY_SERVER_PORT=33333 ZY_SERVER_IP=22.10.109.211 ZY_GPU_ID=7 ZY_NIC_DEV=mlx5_7 ZY_SERVER_ADDR=0x7fdb44800000 ZY_SERVER_RKEY=331331 ZY_BLOCK_IDS=0   ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_ROUNDS=80 ZY_BLOCK_IDS=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102 ZY_SERVER_PORT=33333 ZY_SERVER_IP=22.10.109.211 ZY_METHOD=copygpuwrite ZY_GPU_ID=7 ZY_NIC_DEV=mlx5_7 ZY_SERVER_ADDR=0x7f42c4800000 ZY_SERVER_RKEY=331346 ./kvbench

===== ppu

g++ -DCMAKE_INCLUDE barex_bench.cc -O2 -ggdb -DNDEBUG -laccl_barex  -std=gnu++17 -I/usr/local/cuda/targets/x86_64-linux/include -lcudart -o kvbench

ZY_IS_SERVER=1 ZY_TOKEN_SIZE=1024 ZY_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_SERVER_PORT=33333 ZY_GPU_ID=0  ZY_NIC_DEV=vsolar_1 ./kvbench

ZY_IS_SERVER=0 ZY_TOKEN_SIZE=2048 ZY_BLOCK_SIZE=131072 ZY_SRV_TOKEN_SIZE=1024 ZY_SRV_BLOCK_SIZE=65536 ZY_BLOCK_NUM=128 ZY_ROUNDS=80 ZY_BLOCK_IDS=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102 ZY_SERVER_PORT=33333 ZY_SERVER_IP=`hostname -i` ZY_METHOD=copywrite ZY_GPU_ID=7 ZY_NIC_DEV=vsolar_0 ZY_SERVER_ADDR=0xd3800000 ZY_SERVER_RKEY=0 ./kvbench

#endif

#endif
