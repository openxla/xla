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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_TRITON_SCALED_DOT_SUPPORT_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_TRITON_SCALED_DOT_SUPPORT_H_

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace xla {
namespace gpu {

// Returns true for a scaled-dot instruction with F4E2M1FN data operands when
// its data operand types, layouts, scale types, or target CC are unsupported by
// Triton. Scaled-dot instructions without F4E2M1FN data operands return false.
bool ShouldRejectTritonF4ScaledDot(const HloScaledDotInstruction& dot,
                                   const se::CudaComputeCapability& cc);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_TRITON_SCALED_DOT_SUPPORT_H_
