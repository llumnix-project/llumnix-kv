#ifndef KVTRANSFER_INCLUDE_PARSE_BLOCK_TURBOQUANT_H_
#define KVTRANSFER_INCLUDE_PARSE_BLOCK_TURBOQUANT_H_

#pragma once

#include "parse_block_common.h"

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
);

// TURBOQUANT_CACHE_SHAPE: P > D
void turboquant_parse_block_send_p_gt_d(
  const WorkerInfo *src_worker_info,
  const WorkerInfo *dst_worker_info,
  const std::bitset<MAX_TP_SIZE>& valid_ranks,
  uint32_t kvt_tp_size,
  uint32_t kvt_tp_rank,
  const ReqSendTask *task,
  std::vector<std::vector<IpcBlock>> &send_blocks
);

}  // namespace blade_llm

#endif  // KVTRANSFER_INCLUDE_PARSE_BLOCK_TURBOQUANT_H_
