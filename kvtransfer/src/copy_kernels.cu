#include "copy_kernels.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define BLADE_HAS_CUDA_FP8 1
#endif

namespace blade_llm {

__device__ __forceinline__ void fast_copy_int4(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int64_t num_bytes,
    int thread_id,
    int num_threads
) {
    const int64_t num_vec = num_bytes >> 4;
    const int4* src_vec = reinterpret_cast<const int4*>(src);
    int4* dst_vec = reinterpret_cast<int4*>(dst);

    const int unroll = 4;
    const int64_t stride = (int64_t)num_threads * unroll;
    int64_t i = thread_id;

    if (num_vec >= stride) {
        const int64_t aligned_end = num_vec - (num_vec % stride);
        for (; i < aligned_end; i += stride) {
            dst_vec[i] = src_vec[i];
            dst_vec[i + num_threads] = src_vec[i + num_threads];
            dst_vec[i + 2 * num_threads] = src_vec[i + 2 * num_threads];
            dst_vec[i + 3 * num_threads] = src_vec[i + 3 * num_threads];
        }
    }
    for (; i < num_vec; i += num_threads) {
        dst_vec[i] = src_vec[i];
    }

    const int64_t vec_bytes = num_vec << 4;
    for (int64_t j = vec_bytes + thread_id; j < num_bytes; j += num_threads) {
        dst[j] = src[j];
    }
}

__global__ void copy_int64_array_h2d_kernel(
    const int64_t* __restrict__ host_src,
    int64_t* __restrict__ device_dst,
    int64_t num_elements
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const uintptr_t src_addr = reinterpret_cast<uintptr_t>(host_src);
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(device_dst);

    if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0 && num_elements >= 2) {
        const int64_t num_vec = num_elements / 2;
        const int4* src_vec = reinterpret_cast<const int4*>(host_src);
        int4* dst_vec = reinterpret_cast<int4*>(device_dst);

        const int unroll = 4;
        const int64_t stride = (int64_t)total_threads * unroll;
        int64_t i = global_idx;
        if (num_vec >= stride) {
            const int64_t aligned_end = num_vec - (num_vec % stride);
            for (; i < aligned_end; i += stride) {
                dst_vec[i] = src_vec[i];
                dst_vec[i + total_threads] = src_vec[i + total_threads];
                dst_vec[i + 2 * total_threads] = src_vec[i + 2 * total_threads];
                dst_vec[i + 3 * total_threads] = src_vec[i + 3 * total_threads];
            }
        }
        for (; i < num_vec; i += total_threads) {
            dst_vec[i] = src_vec[i];
        }

        const int64_t remaining_start = num_vec * 2;
        if (remaining_start < num_elements && global_idx == 0) {
            device_dst[remaining_start] = host_src[remaining_start];
        }
        return;
    }

    for (int64_t i = global_idx; i < num_elements; i += total_threads) {
        device_dst[i] = host_src[i];
    }
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
        host_src, device_dst, num_elements);
}

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

    if (num_blocks <= gridDim.x) {
        for (int64_t block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            int64_t length = lengths[block_idx];
            if (length == 0) continue;
            const char* src = host_src + src_offsets[block_idx];
            char* dst = gpu_dst_base + dst_offsets[block_idx];

            const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
            const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
            if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
                fast_copy_int4(src, dst, length, tid, block_size);
            } else {
                for (int64_t i = tid; i < length; i += block_size) {
                    dst[i] = src[i];
                }
            }
        }
    } else {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int warps_per_block = block_size / 32;

        for (int64_t base_idx = (int64_t)blockIdx.x * warps_per_block;
             base_idx < num_blocks;
             base_idx += (int64_t)gridDim.x * warps_per_block) {
            int64_t block_idx = base_idx + warp_id;
            if (block_idx >= num_blocks) continue;

            int64_t length = lengths[block_idx];
            if (length == 0) continue;

            const char* src = host_src + src_offsets[block_idx];
            char* dst = gpu_dst_base + dst_offsets[block_idx];

            if (length == 256) {
                const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
                const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
                if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
                    const int4* sv = reinterpret_cast<const int4*>(src);
                    int4* dv = reinterpret_cast<int4*>(dst);
                    if (lane_id < 16) {
                        dv[lane_id] = sv[lane_id];
                    }
                    continue;
                }
            }

            const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
            const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
            if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
                fast_copy_int4(src, dst, length, lane_id, 32);
            } else if ((sa & 0x7) == 0 && (da & 0x7) == 0 && (length & 0x7) == 0) {
                const int64_t count = length >> 3;
                const int64_t* s64 = reinterpret_cast<const int64_t*>(src);
                int64_t* d64 = reinterpret_cast<int64_t*>(dst);
                for (int64_t i = lane_id; i < count; i += 32) d64[i] = s64[i];
            } else {
                for (int64_t i = lane_id; i < length; i += 32) dst[i] = src[i];
            }
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
        host_src, gpu_dst_base, src_offsets, dst_offsets, lengths, num_blocks);
}

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

    if (num_blocks <= gridDim.x) {
        for (int64_t block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            int64_t length = lengths[block_idx];
            if (length == 0) continue;
            const char* src = gpu_src_base + src_offsets[block_idx];
            char* dst = host_dst + dst_offsets[block_idx];

            const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
            const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
            if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
                fast_copy_int4(src, dst, length, tid, block_size);
            } else {
                for (int64_t i = tid; i < length; i += block_size) {
                    dst[i] = src[i];
                }
            }
        }
    } else {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int warps_per_block = block_size / 32;

        for (int64_t base_idx = (int64_t)blockIdx.x * warps_per_block;
             base_idx < num_blocks;
             base_idx += (int64_t)gridDim.x * warps_per_block) {
            int64_t block_idx = base_idx + warp_id;
            if (block_idx >= num_blocks) continue;

            int64_t length = lengths[block_idx];
            if (length == 0) continue;

            const char* src = gpu_src_base + src_offsets[block_idx];
            char* dst = host_dst + dst_offsets[block_idx];

            if (length == 256) {
                const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
                const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
                if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
                    const int4* sv = reinterpret_cast<const int4*>(src);
                    int4* dv = reinterpret_cast<int4*>(dst);
                    if (lane_id < 16) {
                        dv[lane_id] = sv[lane_id];
                    }
                    continue;
                }
            }

            const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
            const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
            if ((sa & 0xF) == 0 && (da & 0xF) == 0) {
                fast_copy_int4(src, dst, length, lane_id, 32);
            } else if ((sa & 0x7) == 0 && (da & 0x7) == 0 && (length & 0x7) == 0) {
                const int64_t count = length >> 3;
                const int64_t* s64 = reinterpret_cast<const int64_t*>(src);
                int64_t* d64 = reinterpret_cast<int64_t*>(dst);
                for (int64_t i = lane_id; i < count; i += 32) d64[i] = s64[i];
            } else {
                for (int64_t i = lane_id; i < length; i += 32) dst[i] = src[i];
            }
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
        gpu_src_base, host_dst, src_offsets, dst_offsets, lengths, num_blocks);
}

//===============================================
// BF16 to FP8 Conversion
//===============================================

__device__ __forceinline__ uint8_t bf16_to_fp8_e4m3(__nv_bfloat16 val) {
#ifdef BLADE_HAS_CUDA_FP8
    __nv_bfloat16_raw raw;
    raw.x = *reinterpret_cast<const uint16_t*>(&val);
    return static_cast<uint8_t>(
        __nv_cvt_bfloat16raw_to_fp8(raw, __NV_SATFINITE, __NV_E4M3));
#else
    uint16_t bits = *reinterpret_cast<uint16_t*>(&val);
    uint16_t sign = (bits >> 15) & 0x1;
    int16_t exp = ((bits >> 7) & 0xFF) - 127;
    uint16_t mant = bits & 0x7F;

    if (exp == 128) return (sign << 7) | 0x7E;
    if (exp == -127 && mant == 0) return sign << 7;

    int16_t new_exp = exp + 7;

    if (new_exp <= 0) {
        uint16_t full_mant = 0x80 | mant;
        int shift = 1 - new_exp + 4;
        if (shift >= 12) return sign << 7;
        uint16_t shifted = full_mant >> (shift - 1);
        uint8_t new_mant = shifted >> 1;
        uint8_t round_bit = shifted & 1;
        uint16_t sticky_mask = (1 << (shift - 1)) - 1;
        uint8_t sticky = (full_mant & sticky_mask) ? 1 : 0;
        if (round_bit && (sticky || (new_mant & 1))) new_mant++;
        if (new_mant == 0) return sign << 7;
        if (new_mant >= 8) return (sign << 7) | 0x08;
        return (sign << 7) | new_mant;
    }

    if (new_exp >= 15) return (sign << 7) | 0x7E;
    uint16_t dropped_bits = mant & 0xF;
    uint8_t new_mant = mant >> 4;
    if (dropped_bits > 8) {
        new_mant++;
    } else if (dropped_bits == 8) {
        if (new_mant & 1) new_mant++;
    }
    if (new_mant > 7) {
        new_mant = 0;
        new_exp++;
        if (new_exp >= 15) return (sign << 7) | 0x7E;
    }
    return (sign << 7) | (new_exp << 3) | new_mant;
#endif
}

__device__ __forceinline__ int4 convert_16_bf16_to_fp8(const int4& src_lo, const int4& src_hi) {
    const __nv_bfloat16* bf16_lo = reinterpret_cast<const __nv_bfloat16*>(&src_lo);
    const __nv_bfloat16* bf16_hi = reinterpret_cast<const __nv_bfloat16*>(&src_hi);

    uint32_t p0 =
        static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[0])) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[1])) << 8) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[2])) << 16) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[3])) << 24);
    uint32_t p1 =
        static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[4])) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[5])) << 8) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[6])) << 16) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_lo[7])) << 24);
    uint32_t p2 =
        static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[0])) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[1])) << 8) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[2])) << 16) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[3])) << 24);
    uint32_t p3 =
        static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[4])) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[5])) << 8) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[6])) << 16) |
        (static_cast<uint32_t>(bf16_to_fp8_e4m3(bf16_hi[7])) << 24);

    int4 out;
    out.x = *reinterpret_cast<int*>(&p0);
    out.y = *reinterpret_cast<int*>(&p1);
    out.z = *reinterpret_cast<int*>(&p2);
    out.w = *reinterpret_cast<int*>(&p3);
    return out;
}

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

    if (num_blocks <= gridDim.x) {
        for (int64_t block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            int64_t num_elements = lengths[block_idx];
            if (num_elements == 0) continue;

            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(
                gpu_src_base + src_offsets[block_idx]);
            uint8_t* dst = host_dst + dst_offsets[block_idx];

            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

            if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0) {
                const int64_t num_vec16 = num_elements / 16;
                const int64_t stride2 = (int64_t)block_size * 2;
                int64_t vec_idx = tid;
                if (num_vec16 >= stride2) {
                    const int64_t aligned_end = num_vec16 - (num_vec16 % stride2);
                    for (; vec_idx < aligned_end; vec_idx += stride2) {
                        int4 s0 = reinterpret_cast<const int4*>(src)[vec_idx * 2];
                        int4 s1 = reinterpret_cast<const int4*>(src)[vec_idx * 2 + 1];
                        reinterpret_cast<int4*>(dst)[vec_idx] = convert_16_bf16_to_fp8(s0, s1);

                        int64_t vi2 = vec_idx + block_size;
                        int4 s2 = reinterpret_cast<const int4*>(src)[vi2 * 2];
                        int4 s3 = reinterpret_cast<const int4*>(src)[vi2 * 2 + 1];
                        reinterpret_cast<int4*>(dst)[vi2] = convert_16_bf16_to_fp8(s2, s3);
                    }
                }
                for (; vec_idx < num_vec16; vec_idx += block_size) {
                    int4 s0 = reinterpret_cast<const int4*>(src)[vec_idx * 2];
                    int4 s1 = reinterpret_cast<const int4*>(src)[vec_idx * 2 + 1];
                    reinterpret_cast<int4*>(dst)[vec_idx] = convert_16_bf16_to_fp8(s0, s1);
                }

                const int64_t remaining_start = num_vec16 * 16;
                for (int64_t i = remaining_start + tid; i < num_elements; i += block_size) {
                    dst[i] = bf16_to_fp8_e4m3(src[i]);
                }
            } else {
                for (int64_t i = tid; i < num_elements; i += block_size) {
                    dst[i] = bf16_to_fp8_e4m3(src[i]);
                }
            }
        }
    } else {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int warps_per_block = block_size / 32;

        for (int64_t base_block_idx = (int64_t)blockIdx.x * warps_per_block;
             base_block_idx < num_blocks;
             base_block_idx += (int64_t)gridDim.x * warps_per_block) {
            int64_t block_idx = base_block_idx + warp_id;
            if (block_idx >= num_blocks) continue;

            int64_t num_elements = lengths[block_idx];
            if (num_elements == 0) continue;

            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(
                gpu_src_base + src_offsets[block_idx]);
            uint8_t* dst = host_dst + dst_offsets[block_idx];

            const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
            const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

            if ((src_addr & 0xF) == 0 && (dst_addr & 0xF) == 0) {
                const int64_t num_vec16 = num_elements / 16;
                for (int64_t vec_idx = lane_id; vec_idx < num_vec16; vec_idx += 32) {
                    int4 s0 = reinterpret_cast<const int4*>(src)[vec_idx * 2];
                    int4 s1 = reinterpret_cast<const int4*>(src)[vec_idx * 2 + 1];
                    reinterpret_cast<int4*>(dst)[vec_idx] = convert_16_bf16_to_fp8(s0, s1);
                }

                const int64_t remaining_start = num_vec16 * 16;
                for (int64_t i = remaining_start + lane_id; i < num_elements; i += 32) {
                    dst[i] = bf16_to_fp8_e4m3(src[i]);
                }
            } else {
                for (int64_t i = lane_id; i < num_elements; i += 32) {
                    dst[i] = bf16_to_fp8_e4m3(src[i]);
                }
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
        gpu_src_base, reinterpret_cast<uint8_t*>(host_dst),
        src_offsets, dst_offsets, lengths, num_blocks);
}

}  // namespace blade_llm
