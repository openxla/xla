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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_SCALE_UTILS_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_SCALE_UTILS_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Whether `scaled_dot` can use hipBLASLt's swizzled 32x8 scale layout.
bool CanUseHipblasLtScale32x8(
    const HloInstruction& scaled_dot,
    const stream_executor::GpuComputeCapability& gpu_version);

// Returns `scale` permuted into hipBLASLt's swizzled 32x8 tile order.
absl::StatusOr<HloInstruction*> SwizzleHipblasLtScale32x8(
    HloInstruction* scale);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_SCALE_UTILS_H_
