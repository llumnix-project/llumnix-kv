#include "parse_block_turboquant.h"
#include "envcfg.h"
#include <algorithm>
#include <cassert>

namespace blade_llm {

// TURBOQUANT_CACHE_SHAPE: P == D
void turboquant_parse_block_send_p_eq_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // block-wise parse
  vllm_parse_hybrid_block_send_p_eq_d_block_aligned(
    src_worker_info, dst_worker_info, valid_ranks,
    kvt_tp_size, kvt_tp_rank, task, send_blocks);
}

// TURBOQUANT_CACHE_SHAPE: P > D
void turboquant_parse_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
) {
  // Origin method already parse block by k/v, since p tp size > d tp size,
  // head num on p node would be continous and can be parsed as one IpcBlock
  vllm_parse_hybrid_block_send_p_gt_d(
    src_worker_info, dst_worker_info, valid_ranks,
    kvt_tp_size, kvt_tp_rank, task, send_blocks);
}

}  // namespace blade_llm
