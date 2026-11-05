/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/profiler/gpu/profile_with_cuda_kernels.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <vector>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

__global__ void VecAdd(const double* a, const double* b, double* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void VecSub(const double* a, const double* b, double* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}

__global__ void EmptyKernel() {}

}  // namespace

std::vector<double> SimpleAddSubWithProfiler(int num_elements) {
  std::vector<double> vec_a;
  std::vector<double> vec_b;
  std::vector<double> vec_c;
  {
    // Allocates input/output vectors in host memory.
    vec_a.resize(num_elements, 10.);
    vec_b.resize(num_elements, 20.);
    vec_c.resize(num_elements, -1.);
  }

  double* d_a = nullptr;
  double* d_b = nullptr;
  double* d_c = nullptr;
  cudaStream_t stream = nullptr;
  const size_t num_bytes = num_elements * sizeof(double);

  {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // Allocates vectors in device memory.
    cudaMalloc((void**)&d_a, num_bytes);
    cudaMalloc((void**)&d_b, num_bytes);
    cudaMalloc((void**)&d_c, num_bytes);
  }

  {
    {
      // Copies vectors from host to device memory.
      cudaMemcpyAsync(d_a, vec_a.data(), num_bytes, cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(d_b, vec_b.data(), num_bytes, cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(d_c, vec_c.data(), num_bytes, cudaMemcpyHostToDevice,
                      stream);
    }

    {
      constexpr int kThreadsPerBlock = 256;
      const int blocks_per_grid =
          (num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;

      // b1[i] = a[i] + b[i]
      VecAdd<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_a, d_b, d_b,
                                                               num_elements);
      // c1[i] = a[i] - b1[i] = a[i] - (a[i] + b[i]) = -b[i]
      VecSub<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_a, d_b, d_c,
                                                               num_elements);
      // c2[i] = c1[i] + b1[i]  ==> -b[i] + (a[i] + b[i]) = a[i]
      VecAdd<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_c, d_b, d_c,
                                                               num_elements);
      // c3[i] = c2[i] - a[i] = a[i] - a[i] = 0
      VecSub<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_c, d_a, d_c,
                                                               num_elements);
    }

    {
      // Copies vectors from device to host memory.
      cudaMemcpyAsync(vec_c.data(), d_c, num_bytes, cudaMemcpyDeviceToHost,
                      stream);
    }
  }

  {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }

  return vec_c;
}

double AverageEmptyKernelTimeUs(int num_kernels, bool use_cuda_graph) {
  constexpr int kKernelsPerGraph = 100;
  cudaStream_t stream = nullptr;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaGraphExec_t graph_exec = nullptr;
  if (use_cuda_graph) {
    cudaGraph_t graph = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < kKernelsPerGraph; ++i) {
      EmptyKernel<<<1, 1, 0, stream>>>();
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, 0);
    cudaGraphDestroy(graph);
  }
  auto run = [&](int n) {
    if (use_cuda_graph) {
      for (int i = 0; i < n / kKernelsPerGraph; ++i) {
        cudaGraphLaunch(graph_exec, stream);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        EmptyKernel<<<1, 1, 0, stream>>>();
      }
    }
    cudaStreamSynchronize(stream);
  };
  run(kKernelsPerGraph);  // Warm up.
  double best_us = 0;
  for (int attempt = 0; attempt < 5; ++attempt) {
    const auto start = std::chrono::steady_clock::now();
    run(num_kernels);
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - start;
    const double us = elapsed.count() / num_kernels;
    if (attempt == 0 || us < best_us) best_us = us;
  }
  if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
  cudaStreamDestroy(stream);
  return best_us;
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
