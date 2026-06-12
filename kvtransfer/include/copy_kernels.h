#ifndef KVTRANSFER_INCLUDE_COPY_KERNELS_H_
#define KVTRANSFER_INCLUDE_COPY_KERNELS_H_

#include <cuda_runtime.h>
#include <vector>
#include "channel.h"

namespace blade_llm {

enum class CopyDirection {
    H2D,  // Host to Device
    D2H   // Device to Host
};

/**
 * @brief Initialize the startup copy strategy profile table for a CUDA device.
 *
 * Controlled by BLLM_KVTRANS_COPY_PROFILE_ENABLE and related
 * BLLM_KVTRANS_COPY_PROFILE_* environment variables. Safe to call multiple
 * times; profiling is done once per process.
 */
void initialize_copy_method_profile(int target_device);

/**
 * @brief Copy data directly between CPU pinned memory and scattered GPU memory (single batch).
 *
 * Low-level helper: builds src/dst/length arrays in host_blk_buffer_ptr, copies them to
 * device via cudaMemcpyAsync, then launches a kernel that scatters/gathers using
 * device-accessible pinned memory. No intermediate GPU tensor buffer is used.
 *
 * @param tensor_buf_ptr       CPU pinned memory (contiguous). H2D: source; D2H: destination.
 * @param layer_gpu_ptr       GPU base pointer. H2D: destination; D2H: source.
 * @param blocks               Copy layout: dst_offset = GPU offset, length = block size;
 *                             CPU side is contiguous (src offsets derived from blocks order).
 * @param tensor_data_size    Total tensor size in bytes (sum of block lengths).
 * @param direction            H2D or D2H.
 * @param target_device        CUDA device ID.
 * @param stream               CUDA stream.
 * @param src_offsets_dev      Preallocated device buffer for source offsets (num_blocks int64_t).
 * @param dst_offsets_dev      Preallocated device buffer for destination offsets (num_blocks int64_t).
 * @param lengths_dev         Preallocated device buffer for lengths (num_blocks int64_t).
 * @param host_blk_buffer_ptr  Preallocated host pinned buffer for building metadata;
 *                             layout: [src_offsets][dst_offsets][lengths], each num_blocks int64_t.
 *                             Size >= 3 * num_blocks * sizeof(int64_t). Contents are overwritten.
 */
void copy_handle_data_kernel_direct(
    char* tensor_buf_ptr,
    void* layer_gpu_ptr,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size,
    CopyDirection direction,
    int target_device,
    cudaStream_t stream,
    int64_t* src_offsets_dev,
    int64_t* dst_offsets_dev,
    int64_t* lengths_dev,
    int64_t* host_blk_buffer_ptr,
    bool use_explicit_host_offsets = false
);

/**
 * @brief Copy data using CUDA kernel; supports batching when block count exceeds limit.
 *
 * If num_blocks <= env_kernel_copy_max_block_num(), a single kernel is launched.
 * Otherwise blocks are processed in batches of at most env_kernel_copy_max_block_num();
 * the stream is synchronized after each batch so host_blk_buffer_ptr can be safely reused.
 * Max block num: BLLM_KVTRANS_KERNEL_COPY_MAX_BLOCK_NUM (default 8192).
 *
 * @param tensor_buf_ptr       CPU pinned memory (contiguous). H2D: source; D2H: destination.
 * @param layer_gpu_ptr       GPU base pointer. H2D: destination; D2H: source.
 * @param blocks               Copy layout: dst_offset = GPU offset, length = block size;
 *                             CPU side is contiguous.
 * @param tensor_data_size    Total tensor size in bytes.
 * @param direction            H2D or D2H.
 * @param target_device        CUDA device ID.
 * @param stream               CUDA stream.
 * @param preallocated_buffer  GPU buffer for metadata (src_offsets, dst_offsets, lengths).
 *                             Required when num_blocks > 0. Size >= 3 * sizeof(int64_t) * env_kernel_copy_max_block_num().
 * @param host_blk_buffer_ptr  Host pinned buffer for building metadata (reused per batch).
 *                             Required when num_blocks > 0. Size >= 3 * sizeof(int64_t) * env_kernel_copy_max_block_num().
 * @return cudaError_t         cudaSuccess on success.
 */
cudaError_t copy_handle_data_with_kernel(
    char* tensor_buf_ptr,
    void* layer_gpu_ptr,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size,
    CopyDirection direction,
    int target_device,
    cudaStream_t stream,
    int64_t* preallocated_buffer = nullptr,
    int64_t* host_blk_buffer_ptr = nullptr
);

/**
 * @brief Copy scattered BF16 from GPU to contiguous FP8 (E4M3) in host pinned memory (D2H).
 *
 * Performs BF16->FP8 conversion in the copy kernel. Host buffer size is half the BF16
 * tensor size (one byte per element in FP8). When num_blocks > env_kernel_copy_max_block_num(),
 * blocks are processed in batches; the stream is synchronized after each batch so
 * host_blk_buffer_ptr can be reused. Max block num: BLLM_KVTRANS_KERNEL_COPY_MAX_BLOCK_NUM (default 8192).
 *
 * @param tensor_buf_ptr       Host pinned destination (contiguous FP8 E4M3).
 * @param layer_gpu_ptr       GPU source base (BF16).
 * @param blocks               Layout: dst_offset = GPU offset (bytes), length = block size in bytes (BF16).
 *                             Host output is contiguous FP8 (element count = length/2 per block).
 * @param tensor_data_size    Total BF16 tensor size in bytes.
 * @param target_device        CUDA device ID.
 * @param stream               CUDA stream.
 * @param preallocated_buffer  GPU buffer for metadata. Size >= 3 * sizeof(int64_t) * env_kernel_copy_max_block_num().
 * @param host_blk_buffer_ptr  Host pinned buffer for metadata (reused per batch). Same size as above.
 * @return cudaError_t         cudaSuccess on success.
 */
cudaError_t copy_d2h_bf16_to_fp8(
    char* tensor_buf_ptr,
    void* layer_gpu_ptr,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size,
    int target_device,
    cudaStream_t stream,
    int64_t* preallocated_buffer,
    int64_t* host_blk_buffer_ptr
);


}  // namespace blade_llm

#endif  // KVTRANSFER_INCLUDE_COPY_KERNELS_H_
