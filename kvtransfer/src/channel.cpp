#include <stdexcept>
#include "channel.h"
#include "protocol/rdma_protocol.h"
#include "protocol/tcp_channel.h"

namespace blade_llm {

// return true if merged
static bool can_merge(IpcBlock& pre_block, IpcBlock& cur_block) {
  if (cur_block.src_offset != pre_block.src_offset + pre_block.length) {
    return false;
  }
  if (cur_block.dst_offset != pre_block.dst_offset + pre_block.length) {
    return false;
  }
  return true;
}

std::tuple<size_t, size_t, size_t, size_t> merge_interval(std::vector<IpcBlock> &input) {
  size_t min_size = std::numeric_limits<size_t>::max();
  size_t max_size = 0;
  size_t total_size = 0;
  size_t cnt = 0;
  size_t prev_idx = 0;
  std::sort(input.begin(), input.end(),
            [](IpcBlock x, IpcBlock y) { return x.src_offset < y.src_offset; });
  for (size_t idx = 1; idx < input.size(); ++idx) {
    auto &pre_block = input[prev_idx];
    assert(&pre_block.length == &(input[prev_idx].length));
    auto &cur_block = input[idx];
    assert(&cur_block.length == &(input[idx].length));

    if (can_merge(pre_block, cur_block)) {
      pre_block.length += cur_block.length;
      cur_block.length = 0;
    } else {
      min_size = std::min(min_size, pre_block.length);
      max_size = std::max(max_size, pre_block.length);
      total_size += pre_block.length;
      cnt += 1;
      prev_idx = idx;
    }
  }
  if (prev_idx < input.size()) {
    auto prev_len = input[prev_idx].length;
    min_size = std::min(min_size, prev_len);
    max_size = std::max(max_size, prev_len);
    total_size += prev_len;
    cnt += 1;
  }
  // 暂没必要..
  // std::remove_if(input, [] len == 0)
  return {min_size, max_size, total_size, cnt};
}


bool IpcBlock::operator==(const IpcBlock &other) const {
  return src_offset == other.src_offset &&
      dst_offset == other.dst_offset &&
      length == other.length;
}

std::unique_ptr<IChannel> create_channel(Context *ctx, const TransferProtocol &proto) {
  if (!ctx->check_transfer_support(proto)) {
    throw std::runtime_error("unsupported transfer protocol: " + proto.to_string());
  }
  switch (proto.type) {
    case TransferProtocol::Kind::TCP: {
      //llx: RDMAProtoContext改个名字?
      auto proto_ctx = ctx->get_protocol_ctx<BarexProtoContext>(proto);
      if (proto_ctx == nullptr) {
        throw std::runtime_error("tcp channel context not registered;");
      }
      return std::make_unique<TCPChannel>(ctx->inst_name, ctx->worker_id, proto_ctx->cli_barex_ctx());
    }
    case TransferProtocol::Kind::RDMA_DIRECT: {
#ifdef ENABLE_RDMA
      auto proto_ctx = ctx->get_protocol_ctx<BarexProtoContext>(proto);
      if (proto_ctx == nullptr) {
        throw std::runtime_error("RDMA channel context not registered;");
      }
      return std::make_unique<RDMAChannel>(ctx->inst_name, ctx->worker_id, proto_ctx->cli_barex_ctx());

#else
      throw std::runtime_error("RDMA Direct transport not support yet;");
#endif
    }
    default:throw std::runtime_error("Unknown transport protocol;");
  }
}

Channel ChannelFactory::create(const WorkerInfo& info) {
  SupportTransferProtocols target_supports(info.transfer_protocols);
  const auto& proto_opt = this->proto_;
  if (proto_opt.has_value()) {
    auto &proto = proto_opt.value();
    if (!target_supports.is_support(proto)) {
      throw std::runtime_error("target worker not support protocol: " + proto.to_string());
    }
    auto ch = create_channel(this->ctx_, proto);
    ch->connect(info);
    return ch;
  }

  auto protos = target_supports.as_vector();
  // TODO : set priority for each protocol;
  for (const auto &p : protos) {
    try {
      auto ch = create_channel(ctx_, p);
      ch->connect(info);
      LOG(INFO) << "KVT: connect target worker. dst=" << info.inst_id
                << ",dst_id=" << info.worker_id
                << ",use_proto=" << p.to_string();
      return ch;
    } catch (std::exception &e) {
      LOG(WARNING) << "KVT: connect target worker failed. dst=" << info.inst_id
                   << ",dst_id=" << info.worker_id
                   << ",use_proto=" << p.to_string()
                   << ",ex=" << e.what();
    }
  }
  throw std::runtime_error("target worker not support protocol, target_worker=" + info.to_string());
}
}
