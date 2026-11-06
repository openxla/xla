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

#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

bool IsSupportedScaleShape(const Shape& shape) {
  if (shape.element_type() != F8E8M0FNU) {
    return false;
  }
  // [M or N, K/32], or the same with a leading 1.
  int64_t rank = shape.dimensions().size();
  if (rank != 2 && rank != 3) {
    return false;
  }
  if (rank == 3 && shape.dimensions(0) != 1) {
    return false;
  }
  // M or N in whole tiles of 32 rows.
  if (shape.dimensions(rank - 2) % 32 != 0) {
    return false;
  }
  // K/32 in whole tiles of 8 blocks.
  if (shape.dimensions(rank - 1) % 8 != 0) {
    return false;
  }
  return true;
}

}  // namespace

bool CanUseHipblasLtScale32x8(
    const HloInstruction& scaled_dot,
    const stream_executor::GpuComputeCapability& gpu_version) {
  const auto* rocm = gpu_version.rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    return false;
  }
  PrimitiveType input_type = scaled_dot.operand(0)->shape().element_type();
  if ((input_type != F8E4M3FN && input_type != F4E2M1FN) ||
      scaled_dot.operand(1)->shape().element_type() != input_type) {
    return false;
  }
  const DotDimensionNumbers& dims = scaled_dot.dot_dimension_numbers();
  for (int side = 0; side < 2; ++side) {
    int64_t rank = scaled_dot.operand(side)->shape().dimensions().size();
    const auto& contracting = side == 0 ? dims.lhs_contracting_dimensions()
                                        : dims.rhs_contracting_dimensions();
    // Scales must be [M or N, K/32]: K is the last dimension.
    if (contracting.size() != 1 || contracting[0] != rank - 1 ||
        !IsSupportedScaleShape(scaled_dot.operand(side + 2)->shape())) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<HloInstruction*> SwizzleHipblasLtScale32x8(
    HloInstruction* scale) {
  const Shape& shape = scale->shape();
  TF_RET_CHECK(IsSupportedScaleShape(shape));
  int64_t rank = shape.dimensions().size();
  int64_t rows = shape.dimensions(rank - 2);
  int64_t cols = shape.dimensions(rank - 1);

  HloComputation::Builder builder("hipblaslt_mx_scale_32x8");
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "scale"));
  HloInstruction* split = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F8E8M0FNU, {rows / 32, 2, 16, cols / 8, 2, 4}),
      input));
  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F8E8M0FNU, {rows / 32, cols / 8, 4, 16, 2, 2}),
          split, {0, 3, 5, 2, 4, 1}));
  HloInstruction* result =
      builder.AddInstruction(HloInstruction::CreateReshape(shape, transpose));
  HloComputation* computation =
      scale->GetModule()->AddEmbeddedComputation(builder.Build(result));
  return scale->parent()->AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kLoop,
                                   {scale}, computation),
      "hipblaslt_mx_scale_32x8");
}

}  // namespace xla::gpu
