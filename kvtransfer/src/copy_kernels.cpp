#include "copy_kernels.h"
#include "common.h"
#include "thrid_party/logging.h"
#include "envcfg.h"
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <functional>

#ifdef ENABLE_TORCH
#include <torch/extension.h>
#endif

// KERNEL_COPY_MAX_BLOCK_NUM is now obtained from environment variable via env_kernel_copy_max_block_num()
// This is the maximum number of blocks that can be processed in a single kernel launch

#define CUDA_CHECK(call)                                                 \
  {                                                                      \
    cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;             \
      exit(1);                                                           \
    }                                                                    \
  }

namespace blade_llm {
// Forward declaration of CUDA kernel (defined in copy_kernels.cu)
extern void copy_h2d_direct_kernel(
    const char* __restrict__ host_src,
    char* __restrict__ gpu_dst_base,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ lengths,
    int64_t num_blocks
);

extern void copy_d2h_direct_kernel(
    const char* __restrict__ gpu_src_base,
    char* __restrict__ host_dst,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ lengths,
    int64_t num_blocks
);

extern "C" void launch_copy_h2d_direct_kernel(
    const char* host_src,
    char* gpu_dst_base,
    const int64_t* src_offsets,
    const int64_t* dst_offsets,
    const int64_t* lengths,
    int64_t num_blocks,
    int num_cuda_blocks,
    int threads_per_block,
    cudaStream_t stream
);

extern "C" void launch_copy_d2h_direct_kernel(
    const char* gpu_src_base,
    char* host_dst,
    const int64_t* src_offsets,
    const int64_t* dst_offsets,
    const int64_t* lengths,
    int64_t num_blocks,
    int num_cuda_blocks,
    int threads_per_block,
    cudaStream_t stream
);

extern "C" void launch_copy_d2h_bf16_to_fp8_kernel(
    const char* gpu_src_base,
    void* host_dst,
    const int64_t* src_offsets,
    const int64_t* dst_offsets,
    const int64_t* lengths,
    int64_t num_blocks,
    int num_cuda_blocks,
    int threads_per_block,
    cudaStream_t stream
);
extern "C" void launch_copy_h2d_fp8_scaled_to_fp8_kernel(
    const void* host_src,
    const void* host_scales,
    void* gpu_dst_base,
    const int64_t* src_offsets,
    const int64_t* scale_offsets,
    const int64_t* dst_offsets,
    const int64_t* lengths,
    int64_t num_blocks,
    int num_cuda_blocks,
    int threads_per_block,
    cudaStream_t stream
);

extern "C" void launch_copy_int64_array_h2d_kernel(
    const int64_t* host_src,
    int64_t* device_dst,
    int64_t num_elements,
    int num_cuda_blocks,
    int threads_per_block,
    cudaStream_t stream
);

// Global configuration for copy kernel launch
namespace {
    struct CopyKernelConfig {
        int threads_per_block = 512;
        int max_blocks = 1;

        void initialize(CopyDirection direction) {
            int max_threads_per_block = 0;
            int max_threads_per_sm = 0;
            int total_gpu_sm_count = 0;
            int max_blocks_per_sm = 0;
            
            CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block,
                                              cudaDevAttrMaxThreadsPerBlock, 0));
            CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_sm,
                                              cudaDevAttrMaxThreadsPerMultiProcessor, 0));
            CUDA_CHECK(cudaDeviceGetAttribute(&total_gpu_sm_count,
                                              cudaDevAttrMultiProcessorCount, 0));

            for (int i = 1; i <= max_threads_per_sm; i++) {
                int tmp_num_threads = max_threads_per_sm / i;
                if (tmp_num_threads <= max_threads_per_block) {
                    threads_per_block = tmp_num_threads;
                    break;
                }
            }
            if (direction == CopyDirection::H2D) {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_blocks_per_sm, copy_h2d_direct_kernel,
                    threads_per_block, 0));
                max_blocks = max_blocks_per_sm * total_gpu_sm_count * env_kernel_copy_sm_usage();
            } else {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_blocks_per_sm, copy_d2h_direct_kernel,
                    threads_per_block, 0));
                max_blocks = 1;
            }

            std::cerr << "[CopyKernelConfig] Initialized: threads_per_block=" << threads_per_block
                      << ", max_blocks=" << max_blocks << ", sm_count=" << total_gpu_sm_count
                      << ", max_blocks_per_sm=" << max_blocks_per_sm
                      << ", sm_usage=" << env_kernel_copy_sm_usage()
                      << ", direction=" << (direction == CopyDirection::H2D ? "H2D" : "D2H")
                      << std::endl;
        }
    };
    CopyKernelConfig g_copy_kernel_config;
    std::once_flag g_init_flag;
}

inline void get_optimal_launch_config(
    int& num_cuda_blocks,
    int& threads_per_block,
    CopyDirection direction,
    size_t num_blocks = 0
) {
    std::call_once(g_init_flag, [direction]() {
        g_copy_kernel_config.initialize(direction);
    });
    threads_per_block = g_copy_kernel_config.threads_per_block;
    num_cuda_blocks = g_copy_kernel_config.max_blocks;
}

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
    int64_t* host_blk_buffer_ptr
) {
    const size_t num_blocks = blocks.size();
    if (num_blocks == 0) {
        return;
    }
    
    // Prepare offset/length arrays directly in preallocated pinned memory buffer
    // GPU memory layout: [src_offsets] [dst_offsets] [lengths] (continuous)
    size_t tensor_offset = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        const auto& block = blocks[i];
        assert(block.length > 0);
        assert(tensor_offset + block.length <= tensor_data_size);
        
        // src_offset is cumulative (contiguous CPU buffer)
        host_blk_buffer_ptr[i] = static_cast<int64_t>(tensor_offset);
        host_blk_buffer_ptr[num_blocks + i] = static_cast<int64_t>(block.dst_offset);
        host_blk_buffer_ptr[2 * num_blocks + i] = static_cast<int64_t>(block.length);
        
        tensor_offset += block.length;
    }
    assert(tensor_offset == tensor_data_size);
    
    size_t array_size = num_blocks * sizeof(int64_t) * 3;
    CUDA_CHECK(cudaMemcpyAsync(src_offsets_dev, host_blk_buffer_ptr, 
                        array_size, cudaMemcpyHostToDevice, stream));

    char* host_ptr_device = tensor_buf_ptr;
    
    int num_cuda_blocks, threads_per_block;
    get_optimal_launch_config(num_cuda_blocks, threads_per_block, direction, num_blocks);
    
    if (direction == CopyDirection::H2D) {
        // H2D: CPU pinned memory -> GPU scattered memory (direct kernel access)
        launch_copy_h2d_direct_kernel(
            host_ptr_device,  // Pinned host memory source (device-accessible)
            reinterpret_cast<char*>(layer_gpu_ptr),  // GPU destination base
            src_offsets_dev,  // Source offsets in host buffer
            dst_offsets_dev,  // Destination offsets in GPU memory
            lengths_dev,
            static_cast<int64_t>(num_blocks),
            num_cuda_blocks,
            threads_per_block,
            stream
        );
    } else {
        // D2H: GPU scattered memory -> CPU pinned memory (direct kernel access)
        launch_copy_d2h_direct_kernel(
            reinterpret_cast<const char*>(layer_gpu_ptr),  // GPU source base
            host_ptr_device,  // Pinned host memory destination (device-accessible)
            dst_offsets_dev,  // Source offsets in GPU scattered memory
            src_offsets_dev,  // Destination offsets in host buffer
            lengths_dev,
            static_cast<int64_t>(num_blocks),
            num_cuda_blocks,
            threads_per_block,
            stream
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

cudaError_t copy_handle_data_with_kernel(
    char* tensor_buf_ptr,
    void* layer_gpu_ptr,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size,
    CopyDirection direction,
    int target_device,
    cudaStream_t stream,
    int64_t* preallocated_buffer,
    int64_t* host_blk_buffer_ptr
) {
    size_t num_blocks = blocks.size();
    if (num_blocks == 0) {
        return cudaSuccess;
    }
    
    char* buffer = reinterpret_cast<char*>(preallocated_buffer);
    size_t max_blocks_per_batch = env_kernel_copy_max_block_num();
    size_t array_size_per_batch = max_blocks_per_batch * sizeof(int64_t);
    
    // Process blocks in batches if num_blocks exceeds max_blocks_per_batch
    if (num_blocks <= max_blocks_per_batch) {
        size_t array_size = num_blocks * sizeof(int64_t);
        int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
        int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + array_size);
        int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + array_size * 2);

        copy_handle_data_kernel_direct(
            tensor_buf_ptr, layer_gpu_ptr, blocks, 
            tensor_data_size, direction, target_device, stream,
            src_offsets_dev, dst_offsets_dev, lengths_dev, 
            host_blk_buffer_ptr
        );
    } else {
        size_t tensor_offset = 0;
        size_t processed_blocks = 0;
        
        while (processed_blocks < num_blocks) {
            size_t batch_size = std::min(max_blocks_per_batch, num_blocks - processed_blocks);
            
            std::vector<IpcBlock> batch_blocks;
            batch_blocks.reserve(batch_size);
            size_t batch_tensor_size = 0;
            size_t batch_tensor_offset = 0;  // Offset within this batch
            
            for (size_t i = 0; i < batch_size; ++i) {
                const auto& block = blocks[processed_blocks + i];
                // src_offset is relative to batch_tensor_buf_ptr (starts from 0 for each batch)
                batch_blocks.emplace_back(batch_tensor_offset, block.dst_offset, block.length);
                batch_tensor_size += block.length;
                batch_tensor_offset += block.length;
                tensor_offset += block.length;
            }

            int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
            int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + array_size_per_batch);
            int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + array_size_per_batch * 2);

            char* batch_tensor_buf_ptr = tensor_buf_ptr + (tensor_offset - batch_tensor_size);

            copy_handle_data_kernel_direct(
                batch_tensor_buf_ptr, layer_gpu_ptr, 
                batch_blocks, batch_tensor_size, 
                direction, target_device, stream,
                src_offsets_dev, dst_offsets_dev, lengths_dev, 
                host_blk_buffer_ptr
            );
            processed_blocks += batch_size;
            // Sync before next iteration: host_blk_buffer_ptr is reused and will be
            // overwritten. Must ensure cudaMemcpyAsync has finished reading it.
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    
    cudaError_t err = cudaGetLastError();
    return err;
}

//===============================================
// BF16 to FP8 Conversion Related Functions
//===============================================

cudaError_t copy_d2h_bf16_to_fp8(
    char* tensor_buf_ptr,
    void* layer_gpu_ptr,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size,
    int target_device,
    cudaStream_t stream,
    int64_t* preallocated_buffer,
    int64_t* host_blk_buffer_ptr
) {
    size_t num_blocks = blocks.size();
    if (num_blocks == 0) {
        return cudaSuccess;
    }
    
    char* buffer = reinterpret_cast<char*>(preallocated_buffer);
    size_t max_blocks_per_batch = env_kernel_copy_max_block_num();
    size_t array_size_per_batch = max_blocks_per_batch * sizeof(int64_t);
    
    // Process blocks in batches if num_blocks exceeds max_blocks_per_batch
    if (num_blocks <= max_blocks_per_batch) {
        // Single batch processing
        size_t array_size = num_blocks * sizeof(int64_t);
        int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
        int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + array_size);
        int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + array_size * 2);
        
        // Prepare offset/length arrays directly in preallocated pinned memory buffer
        // src_offsets: GPU source offsets (bytes, BF16)
        // dst_offsets: Host dest offsets (bytes, FP8 - half the size)
        // lengths: number of elements (not bytes)
        size_t dst_byte_offset = 0;  // FP8 destination offset (in bytes)
        
        for (size_t i = 0; i < num_blocks; ++i) {
            const auto& block = blocks[i];
            size_t num_elements = block.length / sizeof(uint16_t);  // BF16 is 2 bytes
            host_blk_buffer_ptr[i] = static_cast<int64_t>(block.dst_offset);  // GPU src offset (bytes)
            host_blk_buffer_ptr[num_blocks + i] = static_cast<int64_t>(dst_byte_offset);  // Host dst offset (bytes)
            host_blk_buffer_ptr[2 * num_blocks + i] = static_cast<int64_t>(num_elements);  // Element count
            dst_byte_offset += num_elements;  // FP8 is 1 byte per element
        }
        
        size_t total_metadata_size = num_blocks * sizeof(int64_t) * 3;
        CUDA_CHECK(cudaMemcpyAsync(src_offsets_dev, host_blk_buffer_ptr, 
                            total_metadata_size, cudaMemcpyHostToDevice, stream));
        
        int num_cuda_blocks, threads_per_block;
        get_optimal_launch_config(num_cuda_blocks, threads_per_block, CopyDirection::D2H, num_blocks);
        launch_copy_d2h_bf16_to_fp8_kernel(
            reinterpret_cast<const char*>(layer_gpu_ptr),
            reinterpret_cast<uint8_t*>(tensor_buf_ptr),
            src_offsets_dev,
            dst_offsets_dev,
            lengths_dev,
            static_cast<int64_t>(num_blocks),
            num_cuda_blocks,
            threads_per_block,
            stream
        );
    } else {
        // Batch processing
        size_t dst_byte_offset_total = 0;  // Cumulative FP8 destination offset (in bytes)
        size_t processed_blocks = 0;
        
        while (processed_blocks < num_blocks) {
            size_t batch_size = std::min(max_blocks_per_batch, num_blocks - processed_blocks);
            
            std::vector<IpcBlock> batch_blocks;
            batch_blocks.reserve(batch_size);
            size_t batch_dst_byte_offset = dst_byte_offset_total;  // Start from cumulative offset
            
            // Prepare batch blocks
            for (size_t i = 0; i < batch_size; ++i) {
                const auto& block = blocks[processed_blocks + i];
                batch_blocks.push_back(block);
            }
            
            // Prepare metadata arrays for this batch
            int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
            int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + array_size_per_batch);
            int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + array_size_per_batch * 2);
            
            // Offsets relative to batch buffer start (will be 0 for first batch, but kept for consistency)
            size_t batch_dst_offset = 0;  // Relative to batch_tensor_buf_ptr
            
            for (size_t i = 0; i < batch_size; ++i) {
                const auto& block = batch_blocks[i];
                size_t num_elements = block.length / sizeof(uint16_t);  // BF16 is 2 bytes
                host_blk_buffer_ptr[i] = static_cast<int64_t>(block.dst_offset);  // GPU src offset (bytes)
                host_blk_buffer_ptr[batch_size + i] = static_cast<int64_t>(batch_dst_offset);  // Host dst offset (bytes, relative to batch buffer)
                host_blk_buffer_ptr[2 * batch_size + i] = static_cast<int64_t>(num_elements);  // Element count
                batch_dst_offset += num_elements;  // FP8 is 1 byte per element
            }
            
            // Update cumulative offsets for next batch
            dst_byte_offset_total += batch_dst_offset;
            
            size_t array_size = batch_size * sizeof(int64_t) * 3;
            CUDA_CHECK(cudaMemcpyAsync(src_offsets_dev, host_blk_buffer_ptr, 
                                array_size, cudaMemcpyHostToDevice, stream));

            // Calculate batch tensor buffer pointer
            // batch_dst_byte_offset is the start offset for this batch
            uint8_t* batch_tensor_buf_ptr = reinterpret_cast<uint8_t*>(tensor_buf_ptr) + batch_dst_byte_offset;
            
            int num_cuda_blocks, threads_per_block;
            get_optimal_launch_config(num_cuda_blocks, threads_per_block, CopyDirection::D2H, batch_size);
            launch_copy_d2h_bf16_to_fp8_kernel(
                reinterpret_cast<const char*>(layer_gpu_ptr),
                batch_tensor_buf_ptr,
                src_offsets_dev,
                dst_offsets_dev,
                lengths_dev,
                static_cast<int64_t>(batch_size),
                num_cuda_blocks,
                threads_per_block,
                stream
            );
            processed_blocks += batch_size;
            // Sync before next iteration: host_blk_buffer_ptr is reused and will be
            // overwritten. Must ensure cudaMemcpyAsync has finished reading it.
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    
    return cudaGetLastError();
}

}  // namespace blade_llm
