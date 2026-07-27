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
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>

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
    enum class CopyMethod {
        Kernel,
        MemcpyLoop
    };

    struct CopyProfileKey {
        CopyDirection direction;
        size_t block_count_bucket;
        size_t block_size_bucket;

        bool operator==(const CopyProfileKey& other) const {
            return direction == other.direction &&
                   block_count_bucket == other.block_count_bucket &&
                   block_size_bucket == other.block_size_bucket;
        }
    };

    struct CopyProfileKeyHash {
        size_t operator()(const CopyProfileKey& key) const {
            size_t h = std::hash<int>{}(static_cast<int>(key.direction));
            h ^= std::hash<size_t>{}(key.block_count_bucket + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
            h ^= std::hash<size_t>{}(key.block_size_bucket + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
            return h;
        }
    };

    struct CopyProfileEntry {
        CopyMethod method = CopyMethod::Kernel;
        float kernel_us = 0.0f;
        float memcpy_us = 0.0f;
    };

    struct CopyLayoutSummary {
        size_t num_blocks = 0;
        size_t total_bytes = 0;
        size_t avg_block_size = 0;
        size_t max_block_size = 0;
        bool all_same_size = true;
    };

    std::once_flag g_copy_profile_init_flag;
    std::unordered_map<CopyProfileKey, CopyProfileEntry, CopyProfileKeyHash> g_copy_profile_table;
    std::vector<size_t> g_copy_profile_block_counts;
    std::vector<size_t> g_copy_profile_block_sizes;
    bool g_copy_profile_enabled = false;
    size_t g_memcpy_fallback_block_threshold = 256 * 1024;
    size_t g_memcpy_fallback_max_blocks = 16;

    const char* copy_method_name(CopyMethod method) {
        return method == CopyMethod::MemcpyLoop ? "MemcpyLoop" : "Kernel";
    }

    std::string direction_name(CopyDirection direction) {
        return direction == CopyDirection::H2D ? "H2D" : "D2H";
    }

    bool env_bool(const char* name, bool default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr) {
            return default_value;
        }
        return std::atoi(value) != 0;
    }

    size_t env_size(const char* name, size_t default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr) {
            return default_value;
        }
        long long parsed = std::atoll(value);
        return parsed > 0 ? static_cast<size_t>(parsed) : default_value;
    }

    std::vector<size_t> parse_size_list(const char* env_name) {
        std::vector<size_t> values;
        const char* raw = std::getenv(env_name);
        if (raw == nullptr || raw[0] == '\0') {
            return values;
        }

        std::stringstream ss(raw);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) {
                continue;
            }
            long long parsed = std::atoll(token.c_str());
            if (parsed > 0) {
                values.push_back(static_cast<size_t>(parsed));
            }
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    bool profile_direction_enabled(CopyDirection direction) {
        const char* raw = std::getenv("BLLM_KVTRANS_COPY_PROFILE_DIRECTIONS");
        if (raw == nullptr || raw[0] == '\0') {
            return true;
        }
        std::string directions(raw);
        return directions.find(direction_name(direction)) != std::string::npos;
    }

    size_t upper_bucket(const std::vector<size_t>& buckets, size_t value) {
        if (buckets.empty()) {
            return 0;
        }
        auto it = std::lower_bound(buckets.begin(), buckets.end(), value);
        if (it != buckets.end()) {
            return *it;
        }
        return buckets.back();
    }

    CopyLayoutSummary summarize_layout(const std::vector<IpcBlock>& blocks) {
        CopyLayoutSummary summary;
        summary.num_blocks = blocks.size();
        if (blocks.empty()) {
            return summary;
        }

        size_t first_length = blocks[0].length;
        for (const auto& block : blocks) {
            summary.total_bytes += block.length;
            summary.max_block_size = std::max(summary.max_block_size, block.length);
            summary.all_same_size = summary.all_same_size && (block.length == first_length);
        }
        summary.avg_block_size = summary.total_bytes / summary.num_blocks;
        return summary;
    }

    bool generate_profile_blocks(
        size_t num_blocks,
        size_t block_size,
        size_t total_gpu_size,
        std::vector<IpcBlock>& blocks
    ) {
        if (num_blocks == 0 || block_size == 0 || total_gpu_size < block_size) {
            return false;
        }
        size_t stride = ((block_size + 15) / 16) * 16;
        if (num_blocks > (total_gpu_size - block_size) / stride + 1) {
            return false;
        }

        blocks.clear();
        blocks.reserve(num_blocks);
        for (size_t i = 0; i < num_blocks; ++i) {
            blocks.emplace_back(0, i * stride, block_size);
        }
        return true;
    }

    cudaError_t enqueue_memcpy_loop(
        char* tensor_buf_ptr,
        void* layer_gpu_ptr,
        const std::vector<IpcBlock>& blocks,
        size_t tensor_data_size,
        CopyDirection direction,
        cudaStream_t stream
    ) {
        size_t tensor_offset = 0;
        for (const auto& block : blocks) {
            assert(block.length > 0);
            assert(tensor_offset + block.length <= tensor_data_size);
            char* host_ptr = tensor_buf_ptr + tensor_offset;
            char* gpu_ptr = reinterpret_cast<char*>(layer_gpu_ptr) + block.dst_offset;
            cudaError_t err = cudaSuccess;
            if (direction == CopyDirection::H2D) {
                err = cudaMemcpyAsync(gpu_ptr, host_ptr, block.length, cudaMemcpyHostToDevice, stream);
            } else {
                err = cudaMemcpyAsync(host_ptr, gpu_ptr, block.length, cudaMemcpyDeviceToHost, stream);
            }
            if (err != cudaSuccess) {
                return err;
            }
            tensor_offset += block.length;
        }
        assert(tensor_offset == tensor_data_size);
        return cudaSuccess;
    }

    float elapsed_us(cudaEvent_t start, cudaEvent_t stop) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms * 1000.0f;
    }

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
                max_blocks = max_blocks_per_sm * total_gpu_sm_count * env_kernel_copy_sm_usage();
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
    int64_t* host_blk_buffer_ptr,
    bool use_explicit_host_offsets
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
        
        size_t host_offset = use_explicit_host_offsets ? block.src_offset : tensor_offset;
        assert(host_offset + block.length <= tensor_data_size);
        host_blk_buffer_ptr[i] = static_cast<int64_t>(host_offset);
        host_blk_buffer_ptr[num_blocks + i] = static_cast<int64_t>(block.dst_offset);
        host_blk_buffer_ptr[2 * num_blocks + i] = static_cast<int64_t>(block.length);
        
        tensor_offset += block.length;
    }
    if (!use_explicit_host_offsets) {
        assert(tensor_offset == tensor_data_size);
    }
    
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

namespace {
    float profile_method_once(
        CopyMethod method,
        char* host_buffer,
        void* gpu_buffer,
        const std::vector<IpcBlock>& blocks,
        size_t tensor_data_size,
        CopyDirection direction,
        cudaStream_t stream,
        int64_t* metadata_device,
        int64_t* metadata_host,
        int warmup_iters,
        int profile_iters
    ) {
        const size_t num_blocks = blocks.size();
        size_t array_size = num_blocks * sizeof(int64_t);
        int64_t* src_offsets_dev = metadata_device;
        int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(metadata_device) + array_size);
        int64_t* lengths_dev = reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(metadata_device) + array_size * 2);

        auto enqueue = [&]() {
            if (method == CopyMethod::MemcpyLoop) {
                CUDA_CHECK(enqueue_memcpy_loop(
                    host_buffer, gpu_buffer, blocks, tensor_data_size, direction, stream));
            } else {
                copy_handle_data_kernel_direct(
                    host_buffer, gpu_buffer, blocks, tensor_data_size, direction, 0, stream,
                    src_offsets_dev, dst_offsets_dev, lengths_dev, metadata_host, false);
            }
        };

        for (int i = 0; i < warmup_iters; ++i) {
            enqueue();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < profile_iters; ++i) {
            enqueue();
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float avg_us = elapsed_us(start, stop) / static_cast<float>(profile_iters);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return avg_us;
    }

    void initialize_copy_method_profile_impl(int target_device) {
        g_memcpy_fallback_block_threshold = env_size(
            "BLLM_KVTRANS_COPY_MEMCPY_BLOCK_THRESHOLD_BYTES", 256 * 1024);
        g_memcpy_fallback_max_blocks = env_size(
            "BLLM_KVTRANS_COPY_MEMCPY_MAX_BLOCKS", 16);

        g_copy_profile_enabled = env_bool(
            "BLLM_KVTRANS_COPY_PROFILE_ENABLE", true);
        if (!g_copy_profile_enabled) {
            LOG(INFO) << "Copy profile disabled. fallback memcpy threshold bytes="
                      << g_memcpy_fallback_block_threshold
                      << ", max_blocks=" << g_memcpy_fallback_max_blocks;
            return;
        }

        g_copy_profile_block_counts = parse_size_list("BLLM_KVTRANS_COPY_PROFILE_BLOCK_COUNTS");
        g_copy_profile_block_sizes = parse_size_list("BLLM_KVTRANS_COPY_PROFILE_BLOCK_SIZES");
        if (g_copy_profile_block_counts.empty()) {
            g_copy_profile_block_counts = {1, 2, 4, 16, 256, 8192};
        }
        if (g_copy_profile_block_sizes.empty()) {
            g_copy_profile_block_sizes = {256, 4096, 65536, 262144, 1048576, 4194304, 8388608};
        }

        int warmup_iters = static_cast<int>(env_size("BLLM_KVTRANS_COPY_PROFILE_WARMUP", 2));
        int profile_iters = static_cast<int>(env_size("BLLM_KVTRANS_COPY_PROFILE_ITERS", 5));
        size_t max_profile_bytes = env_size("BLLM_KVTRANS_COPY_PROFILE_MAX_BYTES", 67108864);

        LOG(INFO) << "Copy profile config: enabled=" << g_copy_profile_enabled
                  << ", block_counts=" << g_copy_profile_block_counts.size()
                  << ", block_sizes=" << g_copy_profile_block_sizes.size()
                  << ", warmup=" << warmup_iters
                  << ", iters=" << profile_iters
                  << ", max_profile_bytes=" << max_profile_bytes
                  << ", memcpy_threshold_bytes=" << g_memcpy_fallback_block_threshold
                  << ", memcpy_max_blocks=" << g_memcpy_fallback_max_blocks;

        CUDA_CHECK(cudaSetDevice(target_device));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        for (CopyDirection direction : {CopyDirection::H2D, CopyDirection::D2H}) {
            if (!profile_direction_enabled(direction)) {
                continue;
            }
            for (size_t num_blocks : g_copy_profile_block_counts) {
                for (size_t block_size : g_copy_profile_block_sizes) {
                    size_t tensor_data_size = num_blocks * block_size;
                    if (tensor_data_size == 0 || tensor_data_size > max_profile_bytes) {
                        continue;
                    }

                    size_t stride = ((block_size + 15) / 16) * 16;
                    size_t total_gpu_size = (num_blocks - 1) * stride + block_size;
                    std::vector<IpcBlock> blocks;
                    if (!generate_profile_blocks(num_blocks, block_size, total_gpu_size, blocks)) {
                        continue;
                    }

                    char* host_buffer = nullptr;
                    void* gpu_buffer = nullptr;
                    int64_t* metadata_device = nullptr;
                    int64_t* metadata_host = nullptr;
                    CUDA_CHECK(cudaMallocHost(&host_buffer, tensor_data_size));
                    CUDA_CHECK(cudaMalloc(&gpu_buffer, total_gpu_size));
                    CUDA_CHECK(cudaMalloc(&metadata_device, num_blocks * sizeof(int64_t) * 3));
                    CUDA_CHECK(cudaMallocHost(&metadata_host, num_blocks * sizeof(int64_t) * 3));
                    CUDA_CHECK(cudaMemset(gpu_buffer, 0, total_gpu_size));

                    float memcpy_us = profile_method_once(
                        CopyMethod::MemcpyLoop, host_buffer, gpu_buffer, blocks, tensor_data_size,
                        direction, stream, metadata_device, metadata_host, warmup_iters, profile_iters);
                    float kernel_us = profile_method_once(
                        CopyMethod::Kernel, host_buffer, gpu_buffer, blocks, tensor_data_size,
                        direction, stream, metadata_device, metadata_host, warmup_iters, profile_iters);

                    CopyMethod best = memcpy_us < kernel_us ? CopyMethod::MemcpyLoop : CopyMethod::Kernel;
                    CopyProfileKey key{direction, num_blocks, block_size};
                    g_copy_profile_table[key] = CopyProfileEntry{best, kernel_us, memcpy_us};
                    LOG(INFO) << "Copy profile result: direction=" << direction_name(direction)
                              << ", blocks=" << num_blocks
                              << ", block_size=" << block_size
                              << ", method=" << copy_method_name(best)
                              << ", kernel_us=" << kernel_us
                              << ", memcpy_us=" << memcpy_us;

                    CUDA_CHECK(cudaFreeHost(metadata_host));
                    CUDA_CHECK(cudaFree(metadata_device));
                    CUDA_CHECK(cudaFree(gpu_buffer));
                    CUDA_CHECK(cudaFreeHost(host_buffer));
                }
            }
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        LOG(INFO) << "Copy profile done: entries=" << g_copy_profile_table.size();
    }

    CopyMethod select_copy_method(const std::vector<IpcBlock>& blocks, CopyDirection direction) {
        CopyLayoutSummary summary = summarize_layout(blocks);
        if (summary.num_blocks == 0) {
            return CopyMethod::Kernel;
        }

        if (g_copy_profile_enabled && !g_copy_profile_table.empty()) {
            size_t count_bucket = upper_bucket(g_copy_profile_block_counts, summary.num_blocks);
            size_t size_bucket = upper_bucket(
                g_copy_profile_block_sizes,
                summary.all_same_size ? summary.max_block_size : summary.avg_block_size);
            auto it = g_copy_profile_table.find(CopyProfileKey{direction, count_bucket, size_bucket});
            if (it != g_copy_profile_table.end()) {
                return it->second.method;
            }
        }

        if (summary.num_blocks <= g_memcpy_fallback_max_blocks &&
            summary.avg_block_size >= 64 * 1024) {
            return CopyMethod::MemcpyLoop;
        }
        if (summary.max_block_size >= g_memcpy_fallback_block_threshold &&
            summary.num_blocks <= g_memcpy_fallback_max_blocks) {
            return CopyMethod::MemcpyLoop;
        }
        return CopyMethod::Kernel;
    }

    CopyMethod select_single_block_method(size_t block_size, CopyDirection direction) {
        if (g_copy_profile_enabled && !g_copy_profile_table.empty()) {
            size_t size_bucket = upper_bucket(g_copy_profile_block_sizes, block_size);
            auto it = g_copy_profile_table.find(CopyProfileKey{direction, 1, size_bucket});
            if (it != g_copy_profile_table.end()) {
                return it->second.method;
            }
        }
        return block_size >= g_memcpy_fallback_block_threshold ?
            CopyMethod::MemcpyLoop : CopyMethod::Kernel;
    }

    cudaError_t enqueue_kernel_explicit_offsets(
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
        size_t processed_blocks = 0;
        while (processed_blocks < num_blocks) {
            size_t batch_size = std::min(max_blocks_per_batch, num_blocks - processed_blocks);
            std::vector<IpcBlock> batch_blocks;
            batch_blocks.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                batch_blocks.push_back(blocks[processed_blocks + i]);
            }

            size_t batch_array_size = batch_size * sizeof(int64_t);
            int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
            int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + batch_array_size);
            int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + batch_array_size * 2);
            copy_handle_data_kernel_direct(
                tensor_buf_ptr, layer_gpu_ptr, batch_blocks, tensor_data_size,
                direction, target_device, stream,
                src_offsets_dev, dst_offsets_dev, lengths_dev,
                host_blk_buffer_ptr, true);
            processed_blocks += batch_size;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        return cudaGetLastError();
    }

    cudaError_t enqueue_hybrid_copy(
        char* tensor_buf_ptr,
        void* layer_gpu_ptr,
        const std::vector<IpcBlock>& blocks,
        size_t tensor_data_size,
        CopyDirection direction,
        int target_device,
        cudaStream_t stream,
        int64_t* preallocated_buffer,
        int64_t* host_blk_buffer_ptr,
        bool& handled
    ) {
        handled = false;
        std::vector<IpcBlock> kernel_blocks;
        kernel_blocks.reserve(blocks.size());
        size_t tensor_offset = 0;
        size_t memcpy_blocks = 0;

        for (const auto& block : blocks) {
            CopyMethod method = select_single_block_method(block.length, direction);
            if (method == CopyMethod::MemcpyLoop) {
                char* host_ptr = tensor_buf_ptr + tensor_offset;
                char* gpu_ptr = reinterpret_cast<char*>(layer_gpu_ptr) + block.dst_offset;
                cudaError_t err = cudaSuccess;
                if (direction == CopyDirection::H2D) {
                    err = cudaMemcpyAsync(gpu_ptr, host_ptr, block.length, cudaMemcpyHostToDevice, stream);
                } else {
                    err = cudaMemcpyAsync(host_ptr, gpu_ptr, block.length, cudaMemcpyDeviceToHost, stream);
                }
                if (err != cudaSuccess) {
                    return err;
                }
                ++memcpy_blocks;
            } else {
                kernel_blocks.emplace_back(tensor_offset, block.dst_offset, block.length);
            }
            tensor_offset += block.length;
        }

        if (memcpy_blocks == 0) {
            return cudaSuccess;
        }
        handled = true;
        if (!kernel_blocks.empty()) {
            return enqueue_kernel_explicit_offsets(
                tensor_buf_ptr, layer_gpu_ptr, kernel_blocks, tensor_data_size,
                direction, target_device, stream, preallocated_buffer, host_blk_buffer_ptr);
        }
        return cudaSuccess;
    }
}

void initialize_copy_method_profile(int target_device) {
    std::call_once(g_copy_profile_init_flag, [target_device]() {
        initialize_copy_method_profile_impl(target_device);
    });
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

    initialize_copy_method_profile(target_device);
    const CopyLayoutSummary summary = summarize_layout(blocks);
    const CopyMethod selected_method = select_copy_method(blocks, direction);
    if (selected_method == CopyMethod::MemcpyLoop) {
        return enqueue_memcpy_loop(
            tensor_buf_ptr, layer_gpu_ptr, blocks, tensor_data_size, direction, stream);
    }
    
    char* buffer = reinterpret_cast<char*>(preallocated_buffer);
    size_t max_blocks_per_batch = env_kernel_copy_max_block_num();

    if (!summary.all_same_size) {
        bool handled = false;
        cudaError_t hybrid_err = enqueue_hybrid_copy(
            tensor_buf_ptr, layer_gpu_ptr, blocks, tensor_data_size, direction, target_device,
            stream, preallocated_buffer, host_blk_buffer_ptr, handled);
        if (handled || hybrid_err != cudaSuccess) {
            return hybrid_err;
        }
    }
    
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
            host_blk_buffer_ptr, false
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

            size_t batch_array_size = batch_size * sizeof(int64_t);
            int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
            int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + batch_array_size);
            int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + batch_array_size * 2);

            char* batch_tensor_buf_ptr = tensor_buf_ptr + (tensor_offset - batch_tensor_size);

            copy_handle_data_kernel_direct(
                batch_tensor_buf_ptr, layer_gpu_ptr, 
                batch_blocks, batch_tensor_size, 
                direction, target_device, stream,
                src_offsets_dev, dst_offsets_dev, lengths_dev, 
                host_blk_buffer_ptr, false
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
            size_t batch_array_size = batch_size * sizeof(int64_t);
            int64_t* src_offsets_dev = reinterpret_cast<int64_t*>(buffer);
            int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(buffer + batch_array_size);
            int64_t* lengths_dev = reinterpret_cast<int64_t*>(buffer + batch_array_size * 2);
            
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
