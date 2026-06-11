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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_COPY_FUSION_ANALYSIS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_COPY_FUSION_ANALYSIS_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla::gpu {

// Describes a fusion that can lower to DynamicSliceFusionV2Thunk with an
// embedded DeviceToDeviceCopyThunk.
struct DynamicSliceCopyFusionAnalysis {
  // DS or DUS instruction controlling the runtime offset for original
  // non-custom fusions. Null when `existing_copy_hero` is set.
  const HloInstruction* slicing = nullptr;

  // Operand that should be copied by the embedded copy thunk.
  const HloInstruction* copy_operand = nullptr;

  // Existing copy hero for already-materialized dynamic_slice_fusion custom
  // fusions. Null for original non-custom DS/DUS-root fusions.
  const HloInstruction* existing_copy_hero = nullptr;
};

// Returns analysis when `fusion` is a memcpy-like dynamic slice/update fusion
// that can be lowered to a DynamicSliceFusionV2Thunk containing a D2D copy.
absl::StatusOr<std::optional<DynamicSliceCopyFusionAnalysis>>
AnalyzeDynamicSliceCopyFusion(const HloFusionInstruction& fusion);

bool IsDynamicSliceCopyFusion(const HloInstruction* instr);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_COPY_FUSION_ANALYSIS_H_
