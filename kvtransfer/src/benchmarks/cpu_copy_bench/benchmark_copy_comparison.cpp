// Benchmark comparison between CUDA runtime API-based copy functions and kernel-based copy functions
// This benchmark compares:
// 1. copy_handle_data (CUDA runtime API - multiple cudaMemcpyAsync calls)
// 2. copy_handle_data_batch (CUDA runtime API - cudaMemcpyBatchAsync)
// 3. copy_handle_data_with_kernel (current - CUDA kernel-based)

#include "copy_kernels.h"
#include "common.h"
#include "thrid_party/logging.h"
#include "cuda_runtime_api_functions.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(call)                                                 \
  {                                                                      \
    cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;             \
      exit(1);                                                           \
    }                                                                    \
  }

using namespace blade_llm;

// Forward declaration of kernel launcher
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

// Forward declaration of test function
int test_host_get_device_pointer();

// Forward declaration of copy_d2h_bf16_to_fp8 function
namespace blade_llm {
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
}

// Forward declaration of benchmark functions
int run_benchmark_for_direction(
    CopyDirection direction,
    int device_id,
    int num_blocks,
    size_t block_size,
    size_t total_gpu_size,
    int iterations,
    int warmup_iterations
);

int run_fp8_benchmark(
    int device_id,
    int num_blocks,
    size_t block_size_bf16,
    size_t total_gpu_size,
    int iterations,
    int warmup_iterations
);

// Helper function to allocate pinned memory
void* allocate_pinned_memory(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return ptr;
}

// Helper function to free pinned memory
void free_pinned_memory(void* ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

// Helper function to allocate GPU memory
void* allocate_gpu_memory(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return ptr;
}

// Helper function to free GPU memory
void free_gpu_memory(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

// Initialize test data
void initialize_test_data(char* cpu_buf, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (size_t i = 0; i < size; ++i) {
        cpu_buf[i] = static_cast<char>(dis(gen));
    }
}

size_t round_up_to_alignment(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

bool generate_non_overlapping_offsets(
    size_t num_blocks,
    size_t block_size,
    size_t total_gpu_size,
    size_t alignment,
    std::mt19937& gen,
    std::vector<size_t>& offsets
) {
    if (block_size == 0 || total_gpu_size < block_size) {
        return false;
    }

    const size_t stride = round_up_to_alignment(block_size, alignment);
    const size_t max_non_overlapping_blocks = (total_gpu_size - block_size) / stride + 1;
    if (num_blocks > max_non_overlapping_blocks) {
        std::cerr << "Not enough GPU space for non-overlapping blocks: num_blocks="
                  << num_blocks << ", block_size=" << block_size
                  << ", total_gpu_size=" << total_gpu_size
                  << ", alignment=" << alignment
                  << ", max_non_overlapping_blocks=" << max_non_overlapping_blocks
                  << std::endl;
        return false;
    }

    offsets.clear();
    offsets.reserve(max_non_overlapping_blocks);
    for (size_t i = 0; i < max_non_overlapping_blocks; ++i) {
        offsets.push_back(i * stride);
    }
    std::shuffle(offsets.begin(), offsets.end(), gen);
    offsets.resize(num_blocks);
    return true;
}

// Verify the H2D copy result
bool verify_copy_result(
    const char* cpu_src,
    void* gpu_dst_base,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size
) {
    size_t cpu_offset = 0;
    bool all_match = true;
    
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto& block = blocks[i];
        char* cpu_block = new char[block.length];
        char* gpu_block = static_cast<char*>(gpu_dst_base) + block.dst_offset;
        
        cudaError_t err = cudaMemcpy(cpu_block, gpu_block, block.length, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy block " << i << " from GPU: " << cudaGetErrorString(err) << std::endl;
            delete[] cpu_block;
            all_match = false;
            continue;
        }
        
        const char* src_block = cpu_src + cpu_offset;
        if (memcmp(src_block, cpu_block, block.length) != 0) {
            // std::cerr << "Block " << i << " mismatch: dst_offset=" << block.dst_offset 
            //           << ", length=" << block.length << std::endl;
            all_match = false;
        }
        
        delete[] cpu_block;
        cpu_offset += block.length;
    }
    
    return all_match;
}

// Verify the D2H copy result
bool verify_d2h_copy_result(
    const char* gpu_src_base,
    char* cpu_dst,
    const std::vector<IpcBlock>& blocks,
    size_t tensor_data_size
) {
    size_t cpu_offset = 0;
    bool all_match = true;
    
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto& block = blocks[i];
        char* gpu_block = new char[block.length];
        const char* gpu_src = static_cast<const char*>(gpu_src_base) + block.dst_offset;
        
        cudaError_t err = cudaMemcpy(gpu_block, gpu_src, block.length, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy block " << i << " from GPU: " << cudaGetErrorString(err) << std::endl;
            delete[] gpu_block;
            all_match = false;
            continue;
        }
        
        const char* dst_block = cpu_dst + cpu_offset;
        if (memcmp(gpu_block, dst_block, block.length) != 0) {
            // std::cerr << "Block " << i << " mismatch: src_offset=" << block.src_offset 
            //           << ", length=" << block.length << std::endl;
            all_match = false;
        }
        
        delete[] gpu_block;
        cpu_offset += block.length;
    }
    
    return all_match;
}

// Benchmark a copy function
template<typename Func>
double benchmark_copy(
    Func func,
    const std::string& name,
    int iterations,
    cudaStream_t stream
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
        cudaStreamSynchronize(stream);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(duration) / iterations;
}

// Run benchmark for a specific direction
int run_benchmark_for_direction(
    CopyDirection direction,
    int device_id,
    int num_blocks,
    size_t block_size,
    size_t total_gpu_size,
    int iterations,
    int warmup_iterations
) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Copy Function Performance Comparison" << std::endl;
    std::cout << "Direction: " << (direction == CopyDirection::H2D ? "H2D (Host to Device)" : "D2H (Device to Host)") << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  device_id: " << device_id << std::endl;
    std::cout << "  num_blocks: " << num_blocks << std::endl;
    std::cout << "  block_size: " << block_size << " bytes" << std::endl;
    std::cout << "  total_gpu_size: " << total_gpu_size << " bytes" << std::endl;
    std::cout << "  iterations: " << iterations << std::endl;
    std::cout << "  warmup_iterations: " << warmup_iterations << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Create CUDA stream
    cudaStream_t stream = nullptr;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Keep startup profiling out of measured copy timings. Production code
    // triggers this during copy buffer pre-warm; benchmark does it explicitly.
    initialize_copy_method_profile(device_id);
    
    // Calculate tensor data size
    size_t tensor_data_size = num_blocks * block_size;
    
    // Allocate CPU pinned memory
    char* cpu_buf = static_cast<char*>(allocate_pinned_memory(tensor_data_size));
    if (cpu_buf == nullptr) {
        std::cerr << "Failed to allocate CPU pinned memory" << std::endl;
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // Allocate GPU memory
    void* gpu_mem = allocate_gpu_memory(total_gpu_size);
    if (gpu_mem == nullptr) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // Generate IpcBlock vector with scattered offsets
    std::vector<IpcBlock> blocks;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> scattered_offsets;
    if (!generate_non_overlapping_offsets(
            static_cast<size_t>(num_blocks), block_size, total_gpu_size, 16, gen, scattered_offsets)) {
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    if (direction == CopyDirection::H2D) {
        // H2D: Initialize CPU data, GPU memory starts at zero
        initialize_test_data(cpu_buf, tensor_data_size);
        err = cudaMemset(gpu_mem, 0, total_gpu_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize GPU memory: " << cudaGetErrorString(err) << std::endl;
            free_gpu_memory(gpu_mem);
            free_pinned_memory(cpu_buf);
            cudaStreamDestroy(stream);
            return 1;
        }
        
        // For H2D: src_offset is cumulative (contiguous CPU buffer), dst_offset is scattered in GPU
        for (int i = 0; i < num_blocks; ++i) {
            blocks.emplace_back(0, scattered_offsets[i], block_size);
        }
    } else {
        // D2H: Initialize GPU data, CPU buffer starts at zero
        // Initialize GPU memory with test data
        char* gpu_test_data = new char[tensor_data_size];
        initialize_test_data(gpu_test_data, tensor_data_size);
        
        // Initialize CPU buffer to zero
        memset(cpu_buf, 0, tensor_data_size);
        
        // copy_handle_data_with_kernel uses dst_offset as the GPU source offset for D2H.
        size_t cpu_offset = 0;
        for (int i = 0; i < num_blocks; ++i) {
            size_t src_offset = scattered_offsets[i];
            
            // Copy test data to GPU at scattered locations
            err = cudaMemcpy(static_cast<char*>(gpu_mem) + src_offset, 
                           gpu_test_data + cpu_offset, block_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Failed to initialize GPU block " << i << ": " << cudaGetErrorString(err) << std::endl;
                delete[] gpu_test_data;
                free_gpu_memory(gpu_mem);
                free_pinned_memory(cpu_buf);
                cudaStreamDestroy(stream);
                return 1;
            }
            
            blocks.emplace_back(0, src_offset, block_size);
            cpu_offset += block_size;
        }
        delete[] gpu_test_data;
    }
    
    // Allocate preallocated buffers for kernel copy
    constexpr size_t max_blocks_per_batch = 8192;
    size_t array_size_per_batch = max_blocks_per_batch * sizeof(int64_t);
    size_t device_buffer_size = array_size_per_batch * 3;  // src_offsets, dst_offsets, lengths
    int64_t* preallocated_buffer = nullptr;
    err = cudaMalloc(&preallocated_buffer, device_buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate preallocated device buffer: " << cudaGetErrorString(err) << std::endl;
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // Allocate host pinned buffer for metadata (same size as device buffer)
    int64_t* host_blk_buffer_ptr = nullptr;
    err = cudaMallocHost(&host_blk_buffer_ptr, device_buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate preallocated host pinned buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(preallocated_buffer);
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    std::cout << "\nWarming up..." << std::endl;
    
    // Warmup runs
    for (int i = 0; i < warmup_iterations; ++i) {
        if (direction == CopyDirection::H2D) {
            err = cudaMemset(gpu_mem, 0, total_gpu_size);
        } else {
            memset(cpu_buf, 0, tensor_data_size);
        }
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize memory: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(host_blk_buffer_ptr);
            cudaFree(preallocated_buffer);
            free_gpu_memory(gpu_mem);
            free_pinned_memory(cpu_buf);
            cudaStreamDestroy(stream);
            return 1;
        }
        
        // Warmup kernel copy
        err = copy_handle_data_with_kernel(
            cpu_buf, gpu_mem, blocks, tensor_data_size,
            direction, device_id, stream, preallocated_buffer, host_blk_buffer_ptr
        );
        if (err != cudaSuccess) {
            std::cerr << "Failed in warmup: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(host_blk_buffer_ptr);
            cudaFree(preallocated_buffer);
            free_gpu_memory(gpu_mem);
            free_pinned_memory(cpu_buf);
            cudaStreamDestroy(stream);
            return 1;
        }
        cudaStreamSynchronize(stream);
    }
    
    std::cout << "\nRunning benchmarks..." << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Test 1: copy_handle_data (CUDA runtime API) - only for H2D
    double time_deprecated = 0.0;
    bool verified_deprecated = false;
    if (direction == CopyDirection::H2D) {
        std::cout << "\n1. Testing copy_handle_data (CUDA runtime API - multiple cudaMemcpyAsync)..." << std::endl;
        err = cudaMemset(gpu_mem, 0, total_gpu_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to memset GPU memory: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(host_blk_buffer_ptr);
            cudaFree(preallocated_buffer);
            free_gpu_memory(gpu_mem);
            free_pinned_memory(cpu_buf);
            cudaStreamDestroy(stream);
            return 1;
        }
        cudaStreamSynchronize(stream);
        
        auto copy_handle_data_func = [&]() {
            copy_handle_data(cpu_buf, gpu_mem, blocks, tensor_data_size, stream);
        };
        
        time_deprecated = benchmark_copy(copy_handle_data_func, "copy_handle_data", iterations, stream);
        
        // Verify result
        verified_deprecated = verify_copy_result(cpu_buf, gpu_mem, blocks, tensor_data_size);
        std::cout << "  Average time: " << time_deprecated << " us" << std::endl;
        std::cout << "  Verified: " << (verified_deprecated ? "✓ PASSED" : "✗ FAILED") << std::endl;
    } else {
        std::cout << "\n1. Skipping copy_handle_data (not supported for D2H)..." << std::endl;
    }
    
    // Test 2: copy_handle_data_batch (CUDA runtime API, if enabled) - only for H2D
    double time_batch = 0.0;
    bool verified_batch = false;
#ifdef ENABLE_BATCH_COPY
    if (direction == CopyDirection::H2D) {
        std::cout << "\n2. Testing copy_handle_data_batch (CUDA runtime API - cudaMemcpyBatchAsync)..." << std::endl;
        err = cudaMemset(gpu_mem, 0, total_gpu_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to memset GPU memory: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(host_blk_buffer_ptr);
            cudaFree(preallocated_buffer);
            free_gpu_memory(gpu_mem);
            free_pinned_memory(cpu_buf);
            cudaStreamDestroy(stream);
            return 1;
        }
        cudaStreamSynchronize(stream);
        
        auto copy_handle_data_batch_func = [&]() {
            copy_handle_data_batch(cpu_buf, gpu_mem, blocks, tensor_data_size, stream);
        };
        
        time_batch = benchmark_copy(copy_handle_data_batch_func, "copy_handle_data_batch", iterations, stream);
        
        // Verify result
        verified_batch = verify_copy_result(cpu_buf, gpu_mem, blocks, tensor_data_size);
        std::cout << "  Average time: " << time_batch << " us" << std::endl;
        std::cout << "  Verified: " << (verified_batch ? "✓ PASSED" : "✗ FAILED") << std::endl;
    } else {
        std::cout << "\n2. Skipping copy_handle_data_batch (not supported for D2H)..." << std::endl;
    }
#endif
    
    // Test 3: copy_handle_data_with_kernel (current)
    std::cout << "\n" << (direction == CopyDirection::H2D ? "3" : "1") 
              << ". Testing copy_handle_data_with_kernel (current - CUDA kernel)..." << std::endl;
    if (direction == CopyDirection::H2D) {
        err = cudaMemset(gpu_mem, 0, total_gpu_size);
    } else {
        memset(cpu_buf, 0, tensor_data_size);
    }
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize memory: " << cudaGetErrorString(err) << std::endl;
        cudaFree(preallocated_buffer);
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    cudaStreamSynchronize(stream);
    
    auto copy_handle_data_kernel_func = [&]() {
        cudaError_t kernel_err = copy_handle_data_with_kernel(
            cpu_buf, gpu_mem, blocks, tensor_data_size,
            direction, device_id, stream, preallocated_buffer, host_blk_buffer_ptr
        );
        if (kernel_err != cudaSuccess) {
            std::cerr << "copy_handle_data_with_kernel failed: " << cudaGetErrorString(kernel_err) << std::endl;
            exit(1);
        }
    };
    
    double time_kernel = benchmark_copy(copy_handle_data_kernel_func, "copy_handle_data_with_kernel", iterations, stream);
    
    // Verify result
    bool verified_kernel;
    if (direction == CopyDirection::H2D) {
        verified_kernel = verify_copy_result(cpu_buf, gpu_mem, blocks, tensor_data_size);
    } else {
        verified_kernel = verify_d2h_copy_result(static_cast<const char*>(gpu_mem), cpu_buf, blocks, tensor_data_size);
    }
    std::cout << "  Average time: " << time_kernel << " us" << std::endl;
    std::cout << "  Verified: " << (verified_kernel ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    // Test 4: copy_handle_data_with_kernel with different SM usage values
    std::cout << "\n" << (direction == CopyDirection::H2D ? "4" : "2") 
              << ". Testing copy_handle_data_with_kernel with different SM usage..." << std::endl;
    std::vector<double> sm_usage_values = {
        0.02, 0.05, 0.1, 0.2, 0.3, 
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    };
    std::vector<double> times_per_sm_usage;
    std::vector<bool> verified_per_sm_usage;
    
    // Get GPU attributes to calculate max_blocks for different SM usage
    int max_threads_per_block = 0;
    int max_threads_per_sm = 0;
    int total_gpu_sm_count = 0;
    int max_blocks_per_sm = 0;
    
    err = cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id);
    err = cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
    err = cudaDeviceGetAttribute(&total_gpu_sm_count, cudaDevAttrMultiProcessorCount, device_id);
    
    // Calculate optimal threads_per_block (same logic as CopyKernelConfig)
    int threads_per_block_optimal = 512;
    
    // Calculate max_blocks_per_sm using estimated value
    // Note: We cannot directly access copy_h2d_direct_kernel from .cpp file,
    // so we use the same estimation logic as CopyKernelConfig fallback
    max_blocks_per_sm = 1; //max_threads_per_sm / threads_per_block_optimal;
    
    std::cout << "  GPU Info: SM_count=" << total_gpu_sm_count 
              << ", max_threads_per_sm=" << max_threads_per_sm
              << ", threads_per_block=" << threads_per_block_optimal
              << ", max_blocks_per_sm (estimated)=" << max_blocks_per_sm << std::endl;
    
    // Prepare offset/length arrays once (reused for all SM usage tests).
    // This sweep launches kernels directly with all num_blocks at once, so it
    // needs metadata storage sized by num_blocks rather than max batch size.
    size_t array_size = num_blocks * sizeof(int64_t);
    int64_t* sm_usage_metadata_buffer = nullptr;
    err = cudaMalloc(&sm_usage_metadata_buffer, array_size * 3);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate SM usage metadata buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(host_blk_buffer_ptr);
        cudaFree(preallocated_buffer);
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    int64_t* src_offsets_dev = sm_usage_metadata_buffer;
    int64_t* dst_offsets_dev = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(sm_usage_metadata_buffer) + array_size);
    int64_t* lengths_dev = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(sm_usage_metadata_buffer) + array_size * 2);
    
    std::vector<int64_t> src_offsets_host(num_blocks);
    std::vector<int64_t> dst_offsets_host(num_blocks);
    std::vector<int64_t> lengths_host(num_blocks);
    
    size_t tensor_offset = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto& block = blocks[i];
        if (direction == CopyDirection::H2D) {
            // H2D: src_offset is cumulative (contiguous CPU buffer), dst_offset is scattered in GPU
            src_offsets_host[i] = static_cast<int64_t>(tensor_offset);
            dst_offsets_host[i] = static_cast<int64_t>(block.dst_offset);
        } else {
            // D2H: dst_offset stores the scattered GPU source offset in copy_handle_data_with_kernel.
            src_offsets_host[i] = static_cast<int64_t>(block.dst_offset);
            dst_offsets_host[i] = static_cast<int64_t>(tensor_offset);
        }
        lengths_host[i] = static_cast<int64_t>(block.length);
        tensor_offset += block.length;
    }
    
    // Copy arrays to GPU once
    err = cudaMemcpyAsync(src_offsets_dev, src_offsets_host.data(), array_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy src_offsets to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(sm_usage_metadata_buffer);
        cudaFreeHost(host_blk_buffer_ptr);
        cudaFree(preallocated_buffer);
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    err = cudaMemcpyAsync(dst_offsets_dev, dst_offsets_host.data(), array_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy dst_offsets to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(sm_usage_metadata_buffer);
        cudaFreeHost(host_blk_buffer_ptr);
        cudaFree(preallocated_buffer);
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    err = cudaMemcpyAsync(lengths_dev, lengths_host.data(), array_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy lengths to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(sm_usage_metadata_buffer);
        cudaFreeHost(host_blk_buffer_ptr);
        cudaFree(preallocated_buffer);
        free_gpu_memory(gpu_mem);
        free_pinned_memory(cpu_buf);
        cudaStreamDestroy(stream);
        return 1;
    }
    cudaStreamSynchronize(stream);
    
    // Get device-accessible pointer for pinned memory (once)
    char* host_ptr_device = cpu_buf;
    // err = cudaHostGetDevicePointer(&host_ptr_device, cpu_buf, 0);
    
    // Helper function to test with specific SM usage
    auto test_with_sm_usage = [&](double sm_usage) -> std::pair<double, bool> {
        if (direction == CopyDirection::H2D) {
            err = cudaMemset(gpu_mem, 0, total_gpu_size);
        } else {
            memset(cpu_buf, 0, tensor_data_size);
        }
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize memory: " << cudaGetErrorString(err) << std::endl;
            return {0.0, false};
        }
        cudaStreamSynchronize(stream);
        
        // Use the same logic as CopyKernelConfig::initialize()
        // Calculate num_cuda_blocks based on SM usage: max_blocks = sm_usage * total_gpu_sm_count * max_blocks_per_sm
        int num_cuda_blocks = static_cast<int>(sm_usage * total_gpu_sm_count * max_blocks_per_sm);
        int threads_per_block = threads_per_block_optimal;
        
        // Launch kernel directly
        auto kernel_func = [&, num_cuda_blocks, threads_per_block]() {
            if (direction == CopyDirection::H2D) {
                launch_copy_h2d_direct_kernel(
                    host_ptr_device,
                    reinterpret_cast<char*>(gpu_mem),
                    src_offsets_dev,
                    dst_offsets_dev,
                    lengths_dev,
                    static_cast<int64_t>(num_blocks),
                    num_cuda_blocks,
                    threads_per_block,
                    stream
                );
            } else {
                launch_copy_d2h_direct_kernel(
                    reinterpret_cast<const char*>(gpu_mem),
                    host_ptr_device,
                    src_offsets_dev,
                    dst_offsets_dev,
                    lengths_dev,
                    static_cast<int64_t>(num_blocks),
                    num_cuda_blocks,
                    threads_per_block,
                    stream
                );
            }
        };
        
        double time = benchmark_copy(kernel_func, "", iterations, stream);
        bool verified;
        if (direction == CopyDirection::H2D) {
            verified = verify_copy_result(cpu_buf, gpu_mem, blocks, tensor_data_size);
        } else {
            verified = verify_d2h_copy_result(static_cast<const char*>(gpu_mem), cpu_buf, blocks, tensor_data_size);
        }
        
        return {time, verified};
    };
    
    for (double sm_usage : sm_usage_values) {
        int max_blocks_for_sm_usage = static_cast<int>(sm_usage * total_gpu_sm_count * max_blocks_per_sm);
        // std::cout << "\n  Testing with SM usage = " << sm_usage 
        //           << " (max_blocks = " << max_blocks_for_sm_usage << ")..." << std::endl;
        auto result = test_with_sm_usage(sm_usage);
        times_per_sm_usage.push_back(result.first);
        verified_per_sm_usage.push_back(result.second);
        // std::cout << "    Average time: " << result.first << " us" << std::endl;
        // std::cout << "    Verified: " << (result.second ? "✓ PASSED" : "✗ FAILED") << std::endl;
    }
    
    // Print comparison
    const int col1_width = 50;  // Function name column
    const int col2_width = 15;  // Time column
    const int col3_width = 15;   // Speedup column
    const int total_width = col1_width + col2_width + col3_width;
    
    std::cout << "\n" << std::string(total_width, '=') << std::endl;
    std::cout << "Performance Comparison Summary" << std::endl;
    std::cout << std::string(total_width, '=') << std::endl;
    std::cout << std::left << std::setw(col1_width) << "Function" 
              << std::right << std::setw(col2_width) << "Time (us)" 
              << std::setw(col3_width) << "Speedup" << std::endl;
    std::cout << std::string(total_width, '-') << std::endl;
    
    double baseline_time = time_kernel;
    if (direction == CopyDirection::H2D && time_deprecated > 0) {
        std::cout << std::left << std::setw(col1_width) << "copy_handle_data (CUDA runtime API)"
                  << std::right << std::fixed << std::setprecision(2) << std::setw(col2_width) << time_deprecated
                  << std::setw(col3_width) << "1.00x" << std::endl;
        baseline_time = time_deprecated;
        
#ifdef ENABLE_BATCH_COPY
        if (time_batch > 0) {
            std::cout << std::left << std::setw(col1_width) << "copy_handle_data_batch (CUDA runtime API)"
                      << std::right << std::fixed << std::setprecision(2) << std::setw(col2_width) << time_batch
                      << std::fixed << std::setprecision(2) << std::setw(col3_width) << (baseline_time / time_batch) << "x" << std::endl;
        }
#endif
    }
    
    std::cout << std::left << std::setw(col1_width) << "copy_handle_data_with_kernel (CUDA kernel)"
              << std::right << std::fixed << std::setprecision(2) << std::setw(col2_width) << time_kernel
              << std::fixed << std::setprecision(2) << std::setw(col3_width) 
              << (baseline_time > 0 ? (baseline_time / time_kernel) : 1.0) << "x" << std::endl;
    
    std::cout << std::string(total_width, '=') << std::endl;
    
    // Print SM usage comparison
    if (!times_per_sm_usage.empty()) {
        std::cout << "\n" << std::string(total_width, '=') << std::endl;
        std::cout << "SM Usage Performance Comparison" << std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << std::left << std::setw(col1_width) << "SM Usage" 
                  << std::right << std::setw(col2_width) << "Time (us)" 
                  << std::setw(col3_width) << "Speedup vs baseline" << std::endl;
        std::cout << std::string(total_width, '-') << std::endl;
        
        double sm_baseline_time = times_per_sm_usage.empty() ? time_kernel : times_per_sm_usage[0];
        for (size_t i = 0; i < sm_usage_values.size(); ++i) {
            int max_blocks_for_sm_usage = static_cast<int>(sm_usage_values[i] * total_gpu_sm_count * max_blocks_per_sm);
            std::ostringstream label;
            label << "SM usage = " << std::fixed << std::setprecision(1) << sm_usage_values[i] 
                  << " (max_blocks = " << max_blocks_for_sm_usage << ")";
            std::cout << std::left << std::setw(col1_width) << label.str()
                      << std::right << std::fixed << std::setprecision(2) << std::setw(col2_width) << times_per_sm_usage[i]
                      << std::fixed << std::setprecision(2) << std::setw(col3_width) 
                      << (sm_baseline_time / times_per_sm_usage[i]) << "x" << std::endl;
        }
        std::cout << std::string(total_width, '=') << std::endl;
    }
    
    // Calculate bandwidth
    double total_data_gb = static_cast<double>(tensor_data_size) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "\nBandwidth (GB/s):" << std::endl;
    if (direction == CopyDirection::H2D && time_deprecated > 0) {
        std::cout << "  copy_handle_data: " << std::fixed << std::setprecision(2) 
                  << (total_data_gb / (time_deprecated / 1e6)) << " GB/s" << std::endl;
#ifdef ENABLE_BATCH_COPY
        if (time_batch > 0) {
            std::cout << "  copy_handle_data_batch: " << std::fixed << std::setprecision(2) 
                      << (total_data_gb / (time_batch / 1e6)) << " GB/s" << std::endl;
        }
#endif
    }
    std::cout << "  copy_handle_data_with_kernel: " << std::fixed << std::setprecision(2) 
              << (total_data_gb / (time_kernel / 1e6)) << " GB/s" << std::endl;
    
    // Calculate bandwidth for different SM usage values
    if (!times_per_sm_usage.empty()) {
        std::cout << "\nBandwidth by SM Usage (GB/s):" << std::endl;
        for (size_t i = 0; i < sm_usage_values.size(); ++i) {
            int max_blocks_for_sm_usage = static_cast<int>(sm_usage_values[i] * total_gpu_sm_count * max_blocks_per_sm);
            double bandwidth = total_data_gb / (times_per_sm_usage[i] / 1e6);
            std::cout << "  SM usage = " << std::fixed << std::setprecision(2) << sm_usage_values[i] 
                      << " (max_blocks = " << max_blocks_for_sm_usage << "): " 
                      << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
        }
    }
    
    // Cleanup
    cudaFree(sm_usage_metadata_buffer);
    cudaFreeHost(host_blk_buffer_ptr);
    cudaFree(preallocated_buffer);
    free_gpu_memory(gpu_mem);
    free_pinned_memory(cpu_buf);
    cudaStreamDestroy(stream);
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Check if running the host pointer test
    if (argc > 1 && std::string(argv[1]) == "--test-host-pointer") {
        return test_host_get_device_pointer();
    }
    
    // Check if running FP8 benchmark
    bool is_fp8_benchmark = false;
    if (argc > 1 && std::string(argv[1]) == "--fp8") {
        is_fp8_benchmark = true;
    }
    
    // Check direction: H2D, D2H, or both (default) - only for regular benchmark
    CopyDirection direction = CopyDirection::H2D;
    bool run_both = false;
    int arg_offset = is_fp8_benchmark ? 1 : 0;
    
    if (!is_fp8_benchmark) {
        // Regular benchmark
        if (argc > 1 && std::string(argv[1]) == "--d2h") {
            direction = CopyDirection::D2H;
            arg_offset = 1;
        } else if (argc > 1 && std::string(argv[1]) == "--h2d") {
            direction = CopyDirection::H2D;
            arg_offset = 1;
        } else {
            // No direction specified, run both
            run_both = true;
        }
    }
    
    // Default parameters
    int device_id = 0;
    int num_blocks = 1000;
    size_t block_size = 256;  // bytes
    size_t total_gpu_size = 1024 * 1024 * 10;  // 10MB
    int iterations = 100;
    int warmup_iterations = 10;
    
    // Parse command line arguments (accounting for direction flag)
    if (argc > arg_offset + 1) {
        device_id = std::stoi(argv[arg_offset + 1]);
    }
    if (argc > arg_offset + 2) {
        num_blocks = std::stoi(argv[arg_offset + 2]);
    }
    if (argc > arg_offset + 3) {
        block_size = std::stoul(argv[arg_offset + 3]);
    }
    if (argc > arg_offset + 4) {
        total_gpu_size = std::stoul(argv[arg_offset + 4]);
    }
    if (argc > arg_offset + 5) {
        iterations = std::stoi(argv[arg_offset + 5]);
    }
    if (argc > arg_offset + 6) {
        warmup_iterations = std::stoi(argv[arg_offset + 6]);
    }
    
    // Run benchmark(s)
    if (is_fp8_benchmark) {
        // Run FP8 benchmark (D2H only: GPU BF16 -> Host FP8)
        return run_fp8_benchmark(
            device_id, num_blocks, block_size, 
            total_gpu_size, iterations, warmup_iterations
        );
    } else {
        // Run regular benchmark
        if (run_both) {
            // Run both H2D and D2H
            int result_h2d = run_benchmark_for_direction(
                CopyDirection::H2D, device_id, num_blocks, block_size, 
                total_gpu_size, iterations, warmup_iterations
            );
            if (result_h2d != 0) {
                return result_h2d;
            }
            
            int result_d2h = run_benchmark_for_direction(
                CopyDirection::D2H, device_id, num_blocks, block_size, 
                total_gpu_size, iterations, warmup_iterations
            );
            return result_d2h;
        } else {
            // Run specified direction only
            return run_benchmark_for_direction(
                direction, device_id, num_blocks, block_size, 
                total_gpu_size, iterations, warmup_iterations
            );
        }
    }
}

// Test function for cudaHostGetDevicePointer and cudaHostRegister
int test_host_get_device_pointer() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing cudaHostGetDevicePointer" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int device_id = 0;
    size_t tensor_data_size = 1024 * 1024;  // 1MB
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Allocate pinned memory using cudaMallocHost
    char* cpu_buf = static_cast<char*>(allocate_pinned_memory(tensor_data_size));
    if (cpu_buf == nullptr) {
        std::cerr << "Failed to allocate pinned memory" << std::endl;
        return 1;
    }
    
    std::cout << "Allocated pinned memory: " << cpu_buf << " (size: " << tensor_data_size << " bytes)" << std::endl;
    
    // Initialize test data
    initialize_test_data(cpu_buf, tensor_data_size);
    
    // Try to get device-accessible pointer
    char* host_ptr_device = nullptr;
    bool memory_registered = false;
    
    std::cout << "\nStep 1: cudaHostGetDevicePointer" << std::endl;
    err = cudaHostGetDevicePointer(&host_ptr_device, cpu_buf, 0);
    if (err == cudaSuccess) {
        std::cout << "  Success! Device pointer: " << host_ptr_device << std::endl;
        std::cout << "  Host pointer: " << static_cast<void*>(cpu_buf) << std::endl;
        std::cout << "  Pointers match: " << (host_ptr_device == cpu_buf ? "Yes" : "No") << std::endl;
    } else {
        std::cout << "  Failed: " << cudaGetErrorString(err) << std::endl;
        std::cout << "  Error code: " << err << std::endl;
        
        // Try to register the memory
        std::cout << "\nStep 2: cudaHostRegister" << std::endl;
        err = cudaHostRegister(cpu_buf, tensor_data_size, cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            std::cout << "  Success! Memory registered with cudaHostRegisterMapped" << std::endl;
            memory_registered = true;
            
            // Try again to get device pointer
            std::cout << "\nStep 3: cudaHostGetDevicePointer (after register)" << std::endl;
            err = cudaHostGetDevicePointer(&host_ptr_device, cpu_buf, 0);
            if (err == cudaSuccess) {
                std::cout << "  Success! Device pointer: " << host_ptr_device << std::endl;
                std::cout << "  Host pointer: " << static_cast<void*>(cpu_buf) << std::endl;
                std::cout << "  Pointers match: " << (host_ptr_device == cpu_buf ? "Yes" : "No") << std::endl;
            } else {
                std::cout << "  Failed: " << cudaGetErrorString(err) << std::endl;
                std::cout << "  Error code: " << err << std::endl;
                std::cout << "  Fallback: Using host pointer directly" << std::endl;
                host_ptr_device = cpu_buf;
            }
        } else {
            std::cout << "  Failed: " << cudaGetErrorString(err) << std::endl;
            std::cout << "  Error code: " << err << std::endl;
            std::cout << "  Fallback: Using host pointer directly" << std::endl;
            host_ptr_device = cpu_buf;
        }
    }
    
    // Verify the pointer is usable (simple test)
    std::cout << "\nStep 4: Verification" << std::endl;
    if (host_ptr_device != nullptr) {
        std::cout << "  Device pointer is valid: " << host_ptr_device << std::endl;
        std::cout << "  First byte value: " << static_cast<int>(host_ptr_device[0]) << std::endl;
        std::cout << "  Last byte value: " << static_cast<int>(host_ptr_device[tensor_data_size - 1]) << std::endl;
    } else {
        std::cerr << "  Error: Device pointer is null!" << std::endl;
        free_pinned_memory(cpu_buf);
        return 1;
    }
    
    // Cleanup
    if (memory_registered) {
        // If we registered the memory, unregister it
        err = cudaHostUnregister(cpu_buf);
        if (err == cudaSuccess) {
            std::cout << "\nUnregistered memory successfully" << std::endl;
        } else {
            std::cerr << "\nWarning: Failed to unregister memory: " << cudaGetErrorString(err) << std::endl;
        }
    }
    free_pinned_memory(cpu_buf);
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}

// Run benchmark for FP8 conversion kernels
// Run FP8 conversion benchmark for D2H direction only
// D2H: GPU (BF16 format) -> Host (FP8 format)
int run_fp8_benchmark(
    int device_id,
    int num_blocks,
    size_t block_size_bf16,  // BF16 block size in bytes (source format on GPU)
    size_t total_gpu_size,
    int iterations,
    int warmup_iterations
) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FP8 Conversion Kernel Performance Benchmark" << std::endl;
    std::cout << "Direction: D2H (Device to Host)" << std::endl;
    std::cout << "  Source: GPU (BF16 format)" << std::endl;
    std::cout << "  Destination: Host (FP8 format)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  device_id: " << device_id << std::endl;
    std::cout << "  num_blocks: " << num_blocks << std::endl;
    std::cout << "  block_size: " << block_size_bf16 << " bytes (BF16, source format)" << std::endl;
    std::cout << "  total_gpu_size: " << total_gpu_size << " bytes" << std::endl;
    std::cout << "  iterations: " << iterations << std::endl;
    std::cout << "  warmup_iterations: " << warmup_iterations << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Create CUDA stream
    cudaStream_t stream = nullptr;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Calculate sizes for D2H copy: BF16 (GPU) -> FP8 (Host)
    size_t bf16_data_size = num_blocks * block_size_bf16;  // BF16 data size (source on GPU)
    size_t fp8_data_size = bf16_data_size / 2;  // FP8 is half the size of BF16 (destination on Host)
    
    // Allocate buffers for D2H copy: BF16 GPU -> FP8 Host
    char* bf16_buf = static_cast<char*>(allocate_pinned_memory(bf16_data_size));
    char* fp8_buf = static_cast<char*>(allocate_pinned_memory(fp8_data_size));
    void* gpu_mem = allocate_gpu_memory(total_gpu_size);
    
    if (bf16_buf == nullptr || fp8_buf == nullptr || gpu_mem == nullptr) {
        std::cerr << "Failed to allocate memory" << std::endl;
        free_pinned_memory(bf16_buf);
        free_pinned_memory(fp8_buf);
        free_gpu_memory(gpu_mem);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // Initialize BF16 data buffer (will be copied to GPU in scattered locations for D2H test)
    initialize_test_data(bf16_buf, bf16_data_size);
    
    // Generate IpcBlock vector and initialize scattered GPU data for D2H copy
    std::vector<IpcBlock> blocks;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> scattered_offsets;
    if (!generate_non_overlapping_offsets(
            static_cast<size_t>(num_blocks), block_size_bf16, total_gpu_size, 16, gen, scattered_offsets)) {
        free_pinned_memory(bf16_buf);
        free_pinned_memory(fp8_buf);
        free_gpu_memory(gpu_mem);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // D2H copy pattern: 
    //   - src_offset: scattered locations in GPU memory (BF16 format)
    //   - dst_offset: cumulative/contiguous in Host memory (FP8 format)
    // Initialize GPU memory with scattered BF16 data
    err = cudaMemset(gpu_mem, 0, total_gpu_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize GPU memory: " << cudaGetErrorString(err) << std::endl;
        free_pinned_memory(bf16_buf);
        free_pinned_memory(fp8_buf);
        free_gpu_memory(gpu_mem);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    size_t cpu_offset = 0;
    for (int i = 0; i < num_blocks; ++i) {
        size_t src_offset = scattered_offsets[i];
        
        // Copy BF16 data to GPU at scattered location (preparing D2H test data)
        err = cudaMemcpy(static_cast<char*>(gpu_mem) + src_offset, 
                       bf16_buf + cpu_offset, block_size_bf16, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize GPU block " << i << ": " << cudaGetErrorString(err) << std::endl;
            free_pinned_memory(bf16_buf);
            free_pinned_memory(fp8_buf);
            free_gpu_memory(gpu_mem);
            cudaStreamDestroy(stream);
            return 1;
        }
        
        blocks.emplace_back(0, src_offset, block_size_bf16);
        cpu_offset += block_size_bf16;
    }
    
    // Allocate preallocated buffers for D2H kernel copy
    // Buffer contains 3 arrays: src_offsets (GPU), dst_offsets (Host), lengths
    constexpr size_t max_blocks_per_batch = 8192;
    size_t array_size_per_batch = max_blocks_per_batch * sizeof(int64_t);
    size_t device_buffer_size = array_size_per_batch * 3;  // src_offsets, dst_offsets, lengths
    int64_t* preallocated_buffer = nullptr;
    err = cudaMalloc(&preallocated_buffer, device_buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate preallocated device buffer: " << cudaGetErrorString(err) << std::endl;
        free_pinned_memory(bf16_buf);
        free_pinned_memory(fp8_buf);
        free_gpu_memory(gpu_mem);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    int64_t* host_blk_buffer_ptr = nullptr;
    err = cudaMallocHost(&host_blk_buffer_ptr, device_buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate preallocated host pinned buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(preallocated_buffer);
        free_pinned_memory(bf16_buf);
        free_pinned_memory(fp8_buf);
        free_gpu_memory(gpu_mem);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    std::cout << "\nWarming up..." << std::endl;
    
    // Warmup runs
    for (int i = 0; i < warmup_iterations; ++i) {
        memset(fp8_buf, 0, fp8_data_size);
        
        err = copy_d2h_bf16_to_fp8(
            fp8_buf, gpu_mem, blocks, bf16_data_size,
            device_id, stream, preallocated_buffer, host_blk_buffer_ptr
        );
        
        if (err != cudaSuccess) {
            std::cerr << "Failed in warmup: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(host_blk_buffer_ptr);
            cudaFree(preallocated_buffer);
            free_pinned_memory(bf16_buf);
            free_pinned_memory(fp8_buf);
            free_gpu_memory(gpu_mem);
            cudaStreamDestroy(stream);
            return 1;
        }
        cudaStreamSynchronize(stream);
    }
    
    std::cout << "\nRunning benchmarks..." << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Benchmark D2H FP8 conversion: GPU (BF16) -> Host (FP8)
    std::cout << "\n1. Testing D2H FP8 conversion kernel (GPU BF16 -> Host FP8)..." << std::endl;
    memset(fp8_buf, 0, fp8_data_size);
    cudaStreamSynchronize(stream);
    
    auto fp8_kernel_func = [&]() {
        cudaError_t kernel_err = copy_d2h_bf16_to_fp8(
            fp8_buf, gpu_mem, blocks, bf16_data_size,
            device_id, stream, preallocated_buffer, host_blk_buffer_ptr
        );
        if (kernel_err != cudaSuccess) {
            std::cerr << "D2H FP8 conversion kernel failed: " << cudaGetErrorString(kernel_err) << std::endl;
            exit(1);
        }
    };
    
    double time_fp8 = benchmark_copy(fp8_kernel_func, "D2H FP8 conversion kernel", iterations, stream);
    
    std::cout << "  Average time: " << time_fp8 << " us" << std::endl;
    std::cout << "  Note: Verification skipped (FP8 conversion is lossy)" << std::endl;
    
    // Calculate bandwidth for D2H copy (based on source BF16 data size)
    double total_data_gb = static_cast<double>(bf16_data_size) / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "\nBandwidth (GB/s):" << std::endl;
    std::cout << "  D2H FP8 conversion kernel (GPU BF16 -> Host FP8): " << std::fixed << std::setprecision(2) 
              << (total_data_gb / (time_fp8 / 1e6)) << " GB/s" << std::endl;
    
    // Cleanup
    cudaFreeHost(host_blk_buffer_ptr);
    cudaFree(preallocated_buffer);
    free_pinned_memory(bf16_buf);
    free_pinned_memory(fp8_buf);
    free_gpu_memory(gpu_mem);
    cudaStreamDestroy(stream);
    
    return 0;
}

