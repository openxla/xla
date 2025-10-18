/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_KERNELS_CK_GEMM_FUSION_H_
#define XLA_SERVICE_GPU_KERNELS_CK_GEMM_FUSION_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Pattern matches simple row-major gemms to CK kernels.
class CkGemmPattern : public CustomKernelFusionPattern {
 public:
  std::optional<Match> TryMatch(const se::DeviceDescription& device,
                                HloInstruction* instr) const override;
};

// Fusion implementation that bridges CK GEMM patterns to actual kernel creation.
class CkGemmFusion : public CustomKernelFusion {
 public:
  absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
      const se::DeviceDescription& device,
      const HloComputation* computation) const final;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNELS_CK_GEMM_FUSION_H_
