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

#include "xla/service/gpu/kernels/ck_gemm_fusion.h"

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/kernels/gemm_fusion_helpers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Returns OK if dot instruction is compatible with CK GEMM kernels.
absl::Status MatchCkGemm(HloDotInstruction* dot) {
  // Use shared helper for basic GEMM validation
  TF_RETURN_IF_ERROR(MatchSimpleGemm(dot, {PrimitiveType::F16, PrimitiveType::BF16}));
  
  return absl::OkStatus();
}

}  // namespace

//===----------------------------------------------------------------------===//
// CkGemmPattern Implementation
//===----------------------------------------------------------------------===//

std::optional<CustomKernelFusionPattern::Match> CkGemmPattern::TryMatch(
    const se::DeviceDescription& device, HloInstruction* instr) const {

  // Check if instruction is a dot operation
  auto* dot = DynCast<HloDotInstruction>(instr);
  if (!dot) {
    return std::nullopt;
  }

  // Validate that this dot operation is compatible with CK GEMM
  auto status = MatchCkGemm(dot);
  if (!status.ok()) {
    return std::nullopt;
  }

  // Create fusion configuration
  CustomFusionConfig config;
  config.set_name("ck_gemm");

  // Return successful match with the dot instruction
  return Match{config, {instr}};
}

//===----------------------------------------------------------------------===//
// CkGemmFusion Implementation
//===----------------------------------------------------------------------===//

absl::StatusOr<std::vector<CustomKernel>> CkGemmFusion::LoadKernels(
    const se::DeviceDescription& device,
    const HloComputation* computation) const {

  // Find the root instruction which should be a dot operation
  HloInstruction* root = computation->root_instruction();
  auto* dot = DynCast<HloDotInstruction>(root);

  if (dot == nullptr) {
    return absl::InternalError(
        "ck_gemm requires ROOT operation to be a dot");
  }

  // Validate that this is a supported GEMM operation
  TF_RETURN_IF_ERROR(MatchSimpleGemm(dot, {PrimitiveType::F16, PrimitiveType::BF16}));

  // Extract data types
  PrimitiveType dot_type = dot->shape().element_type();
  PrimitiveType lhs_type = dot->operand(0)->shape().element_type();
  PrimitiveType rhs_type = dot->operand(1)->shape().element_type();

  // Extract matrix dimensions
  auto lhs_shape = dot->operand(0)->shape();
  auto rhs_shape = dot->operand(1)->shape();

  int32_t m = lhs_shape.dimensions(0);  // LHS rows
  int32_t k = lhs_shape.dimensions(1);  // LHS cols = RHS rows (contracting dim)
  int32_t n = rhs_shape.dimensions(1);  // RHS cols

  // Get parameter instructions for argument mapping
  auto* lhs = Cast<HloParameterInstruction>(dot->operand(0));
  auto* rhs = Cast<HloParameterInstruction>(dot->operand(1));

  // Map fusion arguments to GEMM kernel arguments
  kernel::gemm_universal::ArgsIndices indices = {
      lhs->parameter_number(),    // lhs
      rhs->parameter_number(),    // rhs
      computation->num_parameters() // out (fusion result)
  };

  // Call the CK GEMM kernel creation function
  return GetCkGemmKernels("ck_gemm", dot_type, lhs_type, rhs_type,
                          m, n, k, indices, device);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

XLA_REGISTER_CUSTOM_FUSION_PATTERN(::xla::gpu::CkGemmPattern);
XLA_REGISTER_CUSTOM_FUSION("ck_gemm", ::xla::gpu::CkGemmFusion);

}  // namespace xla::gpu
