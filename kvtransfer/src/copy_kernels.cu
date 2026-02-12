#include "copy_kernels.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

namespace blade_llm {

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)


template<typename T>
__device__ __forceinline__ void vectorized_copy(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t count,
    int thread_id,
    int num_threads
) {
    // Unroll loop for better performance and instruction-level parallelism
    const int unroll_factor = 4;
    const int64_t stride = num_threads * unroll_factor;
    int64_t i = thread_id;
    
    if (count >= stride) {
        const int64_t aligned_count = count - (count % stride);
        for (; i < aligned_count; i += stride) {
            dst[i]                     = src[i];
            dst[i + num_threads]       = src[i + num_threads];
            dst[i + 2 * num_threads]   = src[i + 2 * num_threads];
            dst[i + 3 * num_threads]   = src[i + 3 * num_threads];
        }
    }
    
    for (; i < count; i += num_threads) {
        dst[i] = src[i];
    }
}

/**
 * Kernel to copy int64_t array from host pinned memory to device memory.
 * This kernel replaces cudaMemcpyAsync for copying vector data to device.
 * 
 * @param host_src    Host pinned memory source (device-accessible, contiguous)
 * @param device_dst  Device memory destination (contiguous)
 * @param num_elements Number of int64_t elements to copy
 */
__global__ void copy_int64_array_h2d_kernel(
    const int64_t* __restrict__ host_src,
    int64_t* __restrict__ device_dst,
    int64_t num_elements
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int global_idx = blockIdx.x * block_size + tid;
    const int total_threads = gridDim.x * block_size;
    
    // Check alignment for vectorized path
    const uintptr_t src_addr = reinterpret_cast<uintptr_t>(host_src);
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(device_dst);
    const bool src_aligned_16 = (src_addr & 0xF) == 0;
    const bool dst_aligned_16 = (dst_addr & 0xF) == 0;
    
    if (src_aligned_16 && dst_aligned_16 && num_elements >= 2) {
        const int64_t num_vec = num_elements / 2;  // 2 int64_t per int4
        const int4* src_vec = reinterpret_cast<const int4*>(host_src);
        int4* dst_vec = reinterpret_cast<int4*>(device_dst);
        
        // Use vectorized_copy for the main vectorized portion
        vectorized_copy<int4>(src_vec, dst_vec, num_vec, global_idx, total_threads);
        
        // Handle remaining scalar elements (odd number of int64_t)
        // Since num_vec = num_elements / 2, remaining_count is at most 1
        // Use the first thread to handle the single remaining element
        const int64_t remaining_start = num_vec * 2;
        if (remaining_start < num_elements && global_idx == 0) {
            device_dst[remaining_start] = host_src[remaining_start];
        }
        return;
    }
    vectorized_copy<int64_t>(host_src, device_dst, num_elements, global_idx, total_threads);
}

extern "C" void launch_copy_int64_array_h2d_kernel(
    const int64_t* host_src,
    int64_t* device_dst,
    int64_t num_elements,
    int num_cuda_blocks,
    int threads_per_block,
    cudaStream_t stream
) {
    copy_int64_array_h2d_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(
        host_src,
        device_dst,
        num_elements
    );
}

/**
 * @param host_src        Pinned host memory source (contiguous, device-accessible)
 * @param gpu_dst_base    GPU destination base pointer (scattered)
 * @param src_offsets     Array of source offsets in host buffer (cumulative)
 * @param dst_offsets     Array of destination offsets in GPU memory
 * @param lengths         Array of copy lengths for each block
 * @param num_blocks      Number of blocks to copy
 */
__global__ void copy_h2d_direct_kernel(
    const char* __restrict__ host_src,
    char* __restrict__ gpu_dst_base,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ lengths,
    int64_t num_blocks
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = block_size / 32;
    
    // Grid-stride over blocks: each CUDA block processes multiple data blocks in parallel
    // For large threads_per_block (e.g., 1024), we can process multiple blocks simultaneously
    // Each warp handles one data block, allowing better thread utilization
    // Total blocks processed per iteration = gridDim.x * warps_per_block
    for (int64_t base_block_idx = blockIdx.x * warps_per_block; base_block_idx < num_blocks; base_block_idx += gridDim.x * warps_per_block) {
        // Process multiple blocks in parallel within this CUDA block
        // Each warp handles one data block
        int64_t block_idx = base_block_idx + warp_id;
        if (block_idx >= num_blocks) continue;
        
        int64_t length = lengths[block_idx];
        if (length == 0) continue;

        const char* src = host_src + src_offsets[block_idx];
        char* dst = gpu_dst_base + dst_offsets[block_idx];

        // Fast path: 256-byte aligned block (common in KV cache)
        // Use 32 threads (1 warp) per block for better memory coalescing
        if (length == 256) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0) {
                // Inline the 256-byte copy: 16 int4 vectors, use 32 threads (1 warp)
                const int4* src_vec = reinterpret_cast<const int4*>(src);
                int4* dst_vec = reinterpret_cast<int4*>(dst);
                // 256 bytes = 16 x int4, use 32 threads for better coalescing
                for (int i = lane_id; i < 16; i += 32) {
                    dst_vec[i] = src_vec[i];
                }
                continue;
            }
        }

        if (length >= 16 && (length & 0xF) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0) {
                const int64_t vec_count = length >> 4; // /16
                const int4* src_vec = reinterpret_cast<const int4*>(src);
                int4* dst_vec = reinterpret_cast<int4*>(dst);
                vectorized_copy<int4>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        if (length >= 8 && (length & 0x7) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0x7) == 0 && (dst_addr & 0x7) == 0) {
                const int64_t vec_count = length >> 3; // /8
                const int64_t* src_vec = reinterpret_cast<const int64_t*>(src);
                int64_t* dst_vec = reinterpret_cast<int64_t*>(dst);
                vectorized_copy<int64_t>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        if (length >= 4 && (length & 0x3) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0x3) == 0 && (dst_addr & 0x3) == 0) {
                const int64_t vec_count = length >> 2; // /4
                const int32_t* src_vec = reinterpret_cast<const int32_t*>(src);
                int32_t* dst_vec = reinterpret_cast<int32_t*>(dst);
                vectorized_copy<int32_t>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        if (length >= 2 && (length & 0x1) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0x1) == 0 && (dst_addr & 0x1) == 0) {
                const int64_t vec_count = length >> 1; // /2
                const int16_t* src_vec = reinterpret_cast<const int16_t*>(src);
                int16_t* dst_vec = reinterpret_cast<int16_t*>(dst);
                vectorized_copy<int16_t>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        for (int64_t i = lane_id; i < length; i += 32) {
            dst[i] = src[i];
        }
    }
}

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
) {
    copy_h2d_direct_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(
        host_src,
        gpu_dst_base,
        src_offsets,
        dst_offsets,
        lengths,
        num_blocks
    );
}

/**
 * @param gpu_src_base    GPU source base pointer (scattered)
 * @param host_dst        Pinned host memory destination (contiguous, device-accessible)
 * @param src_offsets     Array of source offsets in GPU scattered memory
 * @param dst_offsets     Array of destination offsets in host buffer (cumulative)
 * @param lengths         Array of copy lengths for each block
 * @param num_blocks      Number of blocks to copy
 */
__global__ void copy_d2h_direct_kernel(
    const char* __restrict__ gpu_src_base,
    char* __restrict__ host_dst,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ lengths,
    int64_t num_blocks
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = block_size / 32;
    
    // Total blocks processed per iteration = gridDim.x * warps_per_block
    for (int64_t base_block_idx = blockIdx.x * warps_per_block; base_block_idx < num_blocks; base_block_idx += gridDim.x * warps_per_block) {
        int64_t block_idx = base_block_idx + warp_id;
        if (block_idx >= num_blocks) continue;
        
        int64_t length = lengths[block_idx];
        if (length == 0) continue;

        const char* src = gpu_src_base + src_offsets[block_idx];
        char* dst = host_dst + dst_offsets[block_idx];

        if (length == 256) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0) {
                const int4* src_vec = reinterpret_cast<const int4*>(src);
                int4* dst_vec = reinterpret_cast<int4*>(dst);
                for (int i = lane_id; i < 16; i += 32) {
                    dst_vec[i] = src_vec[i];
                }
                continue;
            }
        }

        if (length >= 16 && (length & 0xF) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0) {
                const int64_t vec_count = length >> 4; // /16
                const int4* src_vec = reinterpret_cast<const int4*>(src);
                int4* dst_vec = reinterpret_cast<int4*>(dst);
                vectorized_copy<int4>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        if (length >= 8 && (length & 0x7) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0x7) == 0 && (dst_addr & 0x7) == 0) {
                const int64_t vec_count = length >> 3; // /8
                const int64_t* src_vec = reinterpret_cast<const int64_t*>(src);
                int64_t* dst_vec = reinterpret_cast<int64_t*>(dst);
                vectorized_copy<int64_t>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        if (length >= 4 && (length & 0x3) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0x3) == 0 && (dst_addr & 0x3) == 0) {
                const int64_t vec_count = length >> 2; // /4
                const int32_t* src_vec = reinterpret_cast<const int32_t*>(src);
                int32_t* dst_vec = reinterpret_cast<int32_t*>(dst);
                vectorized_copy<int32_t>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        if (length >= 2 && (length & 0x1) == 0) {
            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
            if ((src_addr & 0x1) == 0 && (dst_addr & 0x1) == 0) {
                const int64_t vec_count = length >> 1; // /2
                const int16_t* src_vec = reinterpret_cast<const int16_t*>(src);
                int16_t* dst_vec = reinterpret_cast<int16_t*>(dst);
                vectorized_copy<int16_t>(src_vec, dst_vec, vec_count, lane_id, 32);
                continue;
            }
        }

        // Fallback: byte-wise copy
        for (int64_t i = lane_id; i < length; i += 32) {
            dst[i] = src[i];
        }
    }
}

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
) {
    copy_d2h_direct_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(
        gpu_src_base,
        host_dst,
        src_offsets,
        dst_offsets,
        lengths,
        num_blocks
    );
}

//===============================================
// BF16 to FP8 Conversion Related Functions
//===============================================

/**
 * Convert BF16 to FP8 E4M3 format using software method.
 * FP8 E4M3: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
 * BF16: 1 sign bit, 8 exponent bits (bias=127), 7 mantissa bits
 * Uses "round to nearest, ties to even" to match hardware behavior.
 */
 __device__ __forceinline__ uint8_t bf16_to_fp8_e4m3(__nv_bfloat16 val) {
    uint16_t bits = *reinterpret_cast<uint16_t*>(&val);
    uint16_t sign = (bits >> 15) & 0x1;
    int16_t exp = ((bits >> 7) & 0xFF) - 127;  // Unbias BF16 exponent
    uint16_t mant = bits & 0x7F;  // 7-bit mantissa
 
    // Handle NaN/Inf - clamp to max finite value (E4M3 has no inf)
    if (exp == 128) {
        return (sign << 7) | 0x7E;  // Max E4M3: ±448
    }
 
    // Handle zero
    if (exp == -127 && mant == 0) {
        return sign << 7;
    }
 
    // Rebias for E4M3 (bias = 7)
    int16_t new_exp = exp + 7;
 
    // Handle subnormal output (new_exp <= 0)
    if (new_exp <= 0) {
        // Add implicit 1 bit to mantissa: 1.mant -> 0x80 | mant
        uint16_t full_mant = 0x80 | mant;  // 8 bits: 1.xxxxxxx
 
        // Shift right to create subnormal
        int shift = 1 - new_exp + 4;  // +4 because we need to go from 8 bits to 3 bits
        if (shift >= 12) {
            // Too small, flush to zero
            return sign << 7;
        }
 
        uint16_t shifted = full_mant >> (shift - 1);  // Keep one extra bit for rounding
        uint8_t new_mant = shifted >> 1;
        uint8_t round_bit = shifted & 1;
        uint16_t sticky_mask = (1 << (shift - 1)) - 1;
        uint8_t sticky = (full_mant & sticky_mask) ? 1 : 0;
 
        if (round_bit && (sticky || (new_mant & 1))) {
            new_mant++;
        }
 
        if (new_mant == 0) return sign << 7;  // Rounded to zero
        if (new_mant >= 8) return (sign << 7) | 0x08;  // Smallest normal
 
        return (sign << 7) | new_mant;  // Subnormal: exp=0
    }
 
    if (new_exp >= 15) {
        return (sign << 7) | 0x7E;
    }
    uint16_t dropped_bits = mant & 0xF;  // Lower 4 bits being dropped
    uint8_t new_mant = mant >> 4;        // Truncate first
 
    if (dropped_bits > 8) {
        // round up
        new_mant++;
    } else if (dropped_bits == 8) {
        // round to even
        if (new_mant & 1) {
            new_mant++;
        }
    }
    if (new_mant > 7) {
        new_mant = 0;
        new_exp++;
        if (new_exp >= 15) return (sign << 7) | 0x7E;
    }
    return (sign << 7) | (new_exp << 3) | new_mant;
}

/**
 * D2H copy kernel with BF16 to FP8 E4M3 conversion.
 * Reads BF16 data from scattered GPU memory, converts to FP8, writes to contiguous host buffer.
 *
 * @param gpu_src_base    GPU source base pointer (scattered, BF16)
 * @param host_dst        Pinned host memory destination (contiguous, FP8)
 * @param src_offsets     Array of source offsets in GPU memory (bytes)
 * @param dst_offsets     Array of destination offsets in host buffer (bytes, FP8)
 * @param lengths         Array of element counts (number of BF16 elements per block)
 * @param num_blocks      Number of blocks to process
 */
 __global__ void copy_d2h_bf16_to_fp8_kernel(
    const char* __restrict__ gpu_src_base,
    uint8_t* __restrict__ host_dst,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ lengths,
    int64_t num_blocks
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = block_size / 32;
 
    for (int64_t base_block_idx = blockIdx.x * warps_per_block; base_block_idx < num_blocks; base_block_idx += gridDim.x * warps_per_block) {
        int64_t block_idx = base_block_idx + warp_id;
        if (block_idx >= num_blocks) continue;
 
        int64_t num_elements = lengths[block_idx];
        if (num_elements == 0) continue;
 
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(
            gpu_src_base + src_offsets[block_idx]);
        uint8_t* dst = host_dst + dst_offsets[block_idx];
 
        // Vectorized path: read 8 BF16 (int4), convert, write 8 FP8 (uint64_t)
        const int64_t num_vec = num_elements / 8;
        const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
        const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
 
        if (num_vec > 0 && (src_addr & 0xF) == 0 && (dst_addr & 0x7) == 0) {
            // Aligned: use vectorized read/write
            for (int64_t vec_idx = lane_id; vec_idx < num_vec; vec_idx += 32) {
                int4 src_vec = reinterpret_cast<const int4*>(src)[vec_idx];
                const __nv_bfloat16* bf16_vals = reinterpret_cast<const __nv_bfloat16*>(&src_vec);
 
                uint64_t packed =
                    static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[0])) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[1])) << 8) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[2])) << 16) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[3])) << 24) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[4])) << 32) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[5])) << 40) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[6])) << 48) |
                    (static_cast<uint64_t>(bf16_to_fp8_e4m3(bf16_vals[7])) << 56);
 
                reinterpret_cast<uint64_t*>(dst)[vec_idx] = packed;
            }
 
            // Handle remaining elements after vectorized portion
            const int64_t remaining_start = num_vec * 8;
            for (int64_t i = remaining_start + lane_id; i < num_elements; i += 32) {
                dst[i] = bf16_to_fp8_e4m3(src[i]);
            }
        } else {
            // Unaligned fallback: element-wise with unrolling
            const int unroll_factor = 4;
            const int64_t stride = 32 * unroll_factor;
            int64_t i = lane_id;
 
            if (num_elements >= stride) {
                const int64_t aligned_count = num_elements - (num_elements % stride);
                for (; i < aligned_count; i += stride) {
                    dst[i] = bf16_to_fp8_e4m3(src[i]);
                    dst[i + 32] = bf16_to_fp8_e4m3(src[i + 32]);
                    dst[i + 64] = bf16_to_fp8_e4m3(src[i + 64]);
                    dst[i + 96] = bf16_to_fp8_e4m3(src[i + 96]);
                }
            }
 
            for (; i < num_elements; i += 32) {
                dst[i] = bf16_to_fp8_e4m3(src[i]);
            }
        }
    }
}
 
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
) {
    copy_d2h_bf16_to_fp8_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(
        gpu_src_base,
        reinterpret_cast<uint8_t*>(host_dst),
        src_offsets,
        dst_offsets,
        lengths,
        num_blocks
    );
}

}  // namespace blade_llm
