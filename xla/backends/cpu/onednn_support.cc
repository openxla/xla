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

#include "xla/backends/cpu/onednn_support.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "dnnl.hpp"  // NOLINT: for DNNL_MAX_NDIMS
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

bool IsOneDnnSupportedDType(PrimitiveType dtype,
                            const TargetMachineFeatures* cpu_features) {
  if (dtype == F32) {
    return true;
  }
  if (cpu_features == nullptr) {
    return IsSupportedType(dtype);
  }
  if (dtype == BF16) {
    return cpu_features->has_avx512bf16();
  }
  if (dtype == F16) {
    return cpu_features->has_avx512fp16();
  }
  return false;
}

absl::StatusOr<bool> IsOneDnnDotSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape,
    const TargetMachineFeatures* cpu_features) {
  if (lhs_shape.element_type() != rhs_shape.element_type() ||
      lhs_shape.element_type() != out_shape.element_type()) {
    return false;
  }
  if (!IsOneDnnSupportedDType(out_shape.element_type(), cpu_features)) {
    return false;
  }

  if (ShapeUtil::IsZeroElementArray(lhs_shape) ||
      ShapeUtil::IsZeroElementArray(rhs_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    return false;
  }

  // NOLINTNEXTLINE: Use dnnl.hpp for DNNL_MAX_NDIMS for now.
  if (lhs_shape.dimensions_size() > DNNL_MAX_NDIMS ||
      rhs_shape.dimensions_size() > DNNL_MAX_NDIMS) {
    return false;
  }

  auto dot_shape_result =
      GetDotShape(dot_dimensions, lhs_shape, rhs_shape, out_shape);
  if (!dot_shape_result.ok()) {
    VLOG(2) << "GetDotShape Error: " << dot_shape_result.status();
    return false;
  }
  DotShape dot_shape = dot_shape_result.value();

  auto dot_canonical_result = GetDotCanonicalDims(dot_dimensions, dot_shape);
  if (!dot_canonical_result.ok()) {
    VLOG(2) << "GetDotCanonicalDims Error: " << dot_canonical_result.status();
    return false;
  }
  DotCanonicalDims dot_canonical_dims = dot_canonical_result.value();

  // Restrict support to row-major layouts.
  return !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

// TODO(intel-tf): Refactor this once oneDNN thunks are implemented
absl::StatusOr<HloInstruction*> ReconfigureDotDimensions(
    HloInstruction* dot_instr) {
  HloInstruction* lhs = dot_instr->mutable_operand(0);
  HloInstruction* rhs = dot_instr->mutable_operand(1);
  DotDimensionNumbers dim_numbers = dot_instr->dot_dimension_numbers();

  auto lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
  auto lhs_contraction_dims = dim_numbers.lhs_contracting_dimensions();
  bool is_lhs_vector = lhs->shape().dimensions_size() ==
                       (lhs_batch_dims.size() + lhs_contraction_dims.size());

  auto rhs_batch_dims = dim_numbers.rhs_batch_dimensions();
  auto rhs_contraction_dims = dim_numbers.rhs_contracting_dimensions();
  bool is_rhs_vector = rhs->shape().dimensions_size() ==
                       (rhs_batch_dims.size() + rhs_contraction_dims.size());

  if (!is_lhs_vector && !is_rhs_vector) return dot_instr;

  std::vector<int64_t> adjusted_lhs_dims(lhs->shape().dimensions().begin(),
                                         lhs->shape().dimensions().end());
  std::vector<int64_t> adjusted_rhs_dims(rhs->shape().dimensions().begin(),
                                         rhs->shape().dimensions().end());
  std::vector<int64_t> adjusted_dot_dims(
      dot_instr->shape().dimensions().begin(),
      dot_instr->shape().dimensions().end());

  if (is_lhs_vector) {
    auto lhs_it = adjusted_lhs_dims.begin() + lhs_batch_dims.size();
    adjusted_lhs_dims.insert(lhs_it, 1, 1);
    auto result_it = adjusted_dot_dims.begin() + lhs_batch_dims.size();
    adjusted_dot_dims.insert(result_it, 1, 1);
    auto lhs_contraction_dim =
        dot_instr->dot_dimension_numbers().lhs_contracting_dimensions(0);
    dim_numbers.set_lhs_contracting_dimensions(0, lhs_contraction_dim + 1);
    lhs = lhs->AddInstruction(HloInstruction::CreateBitcast(
        ShapeUtil::MakeShape(lhs->shape().element_type(), adjusted_lhs_dims),
        lhs));
  }

  if (is_rhs_vector) {
    auto it = adjusted_rhs_dims.end();
    adjusted_rhs_dims.insert(it, 1, 1);
    auto result_it = adjusted_dot_dims.end();
    adjusted_dot_dims.insert(result_it, 1, 1);
    rhs = rhs->AddInstruction(HloInstruction::CreateBitcast(
        ShapeUtil::MakeShape(rhs->shape().element_type(), adjusted_rhs_dims),
        rhs));
  }

  HloInstruction* adjusted_dot =
      dot_instr->AddInstruction(HloInstruction::CreateDot(
          ShapeUtil::MakeShape(dot_instr->shape().element_type(),
                               adjusted_dot_dims),
          lhs, rhs, dim_numbers, dot_instr->precision_config()));

  HloInstruction* replacement_instr = adjusted_dot->AddInstruction(
      HloInstruction::CreateBitcast(dot_instr->shape(), adjusted_dot));

  TF_RETURN_IF_ERROR(
      dot_instr->parent()->ReplaceInstruction(dot_instr, replacement_instr));
  return adjusted_dot;
}

}  // namespace xla::cpu
