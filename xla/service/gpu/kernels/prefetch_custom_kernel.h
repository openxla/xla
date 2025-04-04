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

#ifndef XLA_SERVICE_GPU_KERNELS_PREFETCH_CUSTOM_KERNEL_H_
#define XLA_SERVICE_GPU_KERNELS_PREFETCH_CUSTOM_KERNEL_H_

#include "absl/status/statusor.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/prefetch_kernel_common.h"

namespace xla::gpu::kernel::prefetch {

absl::StatusOr<CustomKernel> GetL2PrefetchCustomKernel(
    std::array<int, kMaxNumPrefetchBuffers> buffer_sizes, int num_blocks);

}  // namespace xla::gpu::kernel::prefetch

#endif  // XLA_SERVICE_GPU_KERNELS_PREFETCH_CUSTOM_KERNEL_H_
