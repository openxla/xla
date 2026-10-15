/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_COST_MODEL_GPU_BANDWIDTH_KERNELS_H_
#define XLA_TOOLS_COST_MODEL_GPU_BANDWIDTH_KERNELS_H_

#include <cstdint>

#include "absl/status/statusor.h"

namespace xla::gpu {

// Measures cold-L2 bulk write bandwidth in bytes per second for
// `dma_size_bytes` (a positive multiple of 16) on CUDA device `ordinal` with
// Hopper TMA bulk copies (`cp.async.bulk`): the fastest of a fixed number of
// launches, each preceded by an L2 cache flush. Leaves `ordinal` as the calling
// thread's current CUDA device. Returns `kUnimplemented` for devices older than
// compute capability 9.0 and in builds without CUDA.
absl::StatusOr<double> MeasureWriteUblkcpBandwidthBytesPerSec(
    int ordinal, int64_t dma_size_bytes);

}  // namespace xla::gpu

#endif  // XLA_TOOLS_COST_MODEL_GPU_BANDWIDTH_KERNELS_H_
