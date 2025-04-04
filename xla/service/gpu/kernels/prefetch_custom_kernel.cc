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

#include "xla/service/gpu/kernels/prefetch_custom_kernel.h"

#include <algorithm>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/service/gpu/kernels/prefetch_kernel_common.h"
#include "xla/tsl/platform/logging.h"

namespace xla::gpu::kernel::prefetch {

namespace {
se::KernelLoaderSpec::KernelArgsPacking CreatePrefetchArgsPacking(
    std::array<int, kMaxNumPrefetchBuffers> buffer_sizes) {
  return [=](const se::Kernel& kernel, const se::KernelArgs& args) {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);
    const int num_buffers = mem_args->number_of_arguments() / 2;
    CHECK_LE(num_buffers, kMaxNumPrefetchBuffers);
    std::array<const void*, kMaxNumPrefetchBuffers> pointers;
    for (int i = 0; i < num_buffers; ++i) {
      pointers[i] = mem_args->device_memory_ptr(i);
    }
    CHECK_GT(buffer_sizes[0], 0);
    return se::PackKernelArgs(/*shmem_bytes=*/0, pointers, buffer_sizes);
  };
}
}  // namespace

absl::StatusOr<CustomKernel> GetL2PrefetchCustomKernel(
    std::array<int, kMaxNumPrefetchBuffers> buffer_sizes,
    const int num_blocks) {
  se::KernelLoaderSpec spec = se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      GetL2PrefetchKernel(), "l2_prefetch", /*arity=*/2,
      CreatePrefetchArgsPacking(buffer_sizes));
  return CustomKernel("l2_prefetch", std::move(spec), se::BlockDim(num_blocks),
                      se::ThreadDim(1024),
                      /*shared_memory_bytes=*/0);
}

}  // namespace xla::gpu::kernel::prefetch
