#include <thread>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "step.h"
#include "thrid_party/logging.h"

using namespace blade_llm;

__global__ void testKernel(char *ptr, int sz, char val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("thread%d: blockIdx.x=%d,blockDim.x=%d, gridDim.x=%d;\n", idx, blockIdx.x, blockDim.x, gridDim.x);
  int stride = gridDim.x * blockDim.x;
  for (; idx < sz; idx += stride) {
    ptr[idx] = val;
  }
}

TEST(StepTest, TestCudaKernelSync) {
  size_t num_layers = 2;
  cudaSetDevice(0);
  std::vector<cudaEvent_t> events(num_layers);
  std::vector<uint64_t> event_addrs;

  for (size_t i = 0; i < num_layers; ++i) {
    cudaEventCreate(&events[i]);
    auto addr = reinterpret_cast<uintptr_t>(events[i]);
    event_addrs.push_back(addr);
  }

  auto cu_barrier = std::make_unique<CudaEventBarrier>(event_addrs);
  int blocks = 1;
  int threads = 128;

  size_t data_size = blocks * threads * (1 << 14);
  void *layer_0, *layer_1;
  cudaMalloc(&layer_0, data_size);
  cudaMalloc(&layer_1, data_size);
  cudaMemset(layer_0, 0, data_size);
  cudaMemset(layer_1, 0, data_size);
  void *host_buf = malloc(data_size);

  char steps = 4;
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  for (char i = 0; i < steps; ++i) {
    memset(host_buf, 0, data_size);
    auto step_i = std::make_shared<Step>(i);
    StepGuard guard(2, cu_barrier.get(), step_i);
    StepGuard *guard_ptr = &guard;
    std::thread t0([guard_ptr]() {
      guard_ptr->wait_layers();
    });

    std::thread t1([step_i, host_buf, data_size, layer_0, layer_1]() {
      char * host_val = (char *)host_buf;
      auto idx = step_i->step_idx;
      step_i->wait_layer_ready(0);
      cudaMemcpy(host_buf, layer_0, data_size, cudaMemcpyDeviceToHost);
      size_t cnt = 0;
      for(auto j =0;j< data_size;j++) {
        size_t val = host_val[j];
        if (j < 128) {
          EXPECT_EQ(val, idx + 1);
        }
        cnt += val;
      }
      EXPECT_EQ(cnt, (idx + 1) * data_size);
      step_i->wait_layer_ready(1);
      cnt = 0;
      cudaMemcpy(host_buf, layer_1, data_size, cudaMemcpyDeviceToHost);
      for(auto j =0;j< data_size;j++) {
        size_t val = host_val[j];
        if (j < 128) {
          EXPECT_EQ(val, idx + 2);
        }
        cnt += val;
      }
      EXPECT_EQ(cnt, (idx + 2) * data_size);
      LOG(INFO)  << "check step: " << idx << " result done, cnt = " << cnt << ",data_size=" << data_size;
    });

    testKernel<<<blocks, threads, 0, stream>>>((char *)(layer_0), data_size, i + 1);
    cudaEventRecord(events[0], stream);
    guard.after_record_one();
    testKernel<<<blocks, threads, 0, stream>>>((char *)(layer_1), data_size, i + 2);
    cudaEventRecord(events[1], stream);
    guard.after_record_one();
    cudaStreamSynchronize(stream);

    t0.join();
    t1.join();
  }

  free(host_buf);
  cudaFree(layer_0);
  cudaFree(layer_1);
}
