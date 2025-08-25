/* Copyright 2025 The OpenXLA Authors.

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

#include <array>

#include "xla/service/gpu/kernels/prefetch_kernel_common.h"

namespace xla::gpu {

namespace {
__global__ void l2_prefetch(
    std::array<const char* __restrict__, kMaxNumPrefetchBuffers> pointers,
    std::array<const int, kMaxNumPrefetchBuffers> buffer_sizes) {
  constexpr int kTransactionSizeBytes = 128;
  const int size_per_block_per_iteration = kTransactionSizeBytes * blockDim.x;
  const int total_size_per_iteration = size_per_block_per_iteration * gridDim.x;
  for (int buffer_index = 0; buffer_index < kMaxNumPrefetchBuffers;
       ++buffer_index) {
    const int num_iterations =
        (buffer_sizes[buffer_index] + total_size_per_iteration - 1) /
        total_size_per_iteration;
    const int total_size_per_thread = kTransactionSizeBytes * num_iterations;
    const int total_size_per_block =
        size_per_block_per_iteration * num_iterations;

    const char* ptr = pointers[buffer_index] +
                      blockIdx.x * total_size_per_block +
                      threadIdx.x * total_size_per_thread;

    for (int i = 0; i < num_iterations; ++i) {
      asm("prefetch.global.L2 [%0];" ::"l"(ptr));
      ptr += kTransactionSizeBytes;
    }
  }
}
}  // namespace

void* GetL2PrefetchKernel() { return reinterpret_cast<void*>(&l2_prefetch); }

}  // namespace xla::gpu
