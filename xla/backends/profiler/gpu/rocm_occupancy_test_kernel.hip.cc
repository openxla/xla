/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

// This file must be compiled by hipcc (not plain clang) so that __global__,
// blockIdx, blockDim, threadIdx and <<<...>>> are available.  The BUILD rule
// uses a .hip.cc extension which routes the file through hipcc_wrapper.

#include "xla/backends/profiler/gpu/rocm_occupancy_test_kernel.h"

#include "rocm/include/hip/hip_runtime.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

__global__ void ScaleKernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f;
}

}  // namespace

int RunOccupancyTestKernel(int n, int block_size) {
  void* d_in = nullptr;
  void* d_out = nullptr;

  hipError_t err = hipMalloc(&d_in, static_cast<size_t>(n) * sizeof(float));
  if (err != hipSuccess) return static_cast<int>(err);

  err = hipMalloc(&d_out, static_cast<size_t>(n) * sizeof(float));
  if (err != hipSuccess) {
    hipFree(d_in);
    return static_cast<int>(err);
  }

  int grid = (n + block_size - 1) / block_size;
  ScaleKernel<<<grid, block_size>>>(static_cast<const float*>(d_in),
                                    static_cast<float*>(d_out), n);

  err = hipDeviceSynchronize();
  hipFree(d_in);
  hipFree(d_out);
  return static_cast<int>(err);
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
