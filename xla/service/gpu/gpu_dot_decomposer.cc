/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gpu_dot_decomposer.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/sparse_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

Status GpuDotDecomposer::CanonicalizeDot(HloInstruction* original_dot) {
  auto computation = original_dot->parent();
  const auto& original_dnums = original_dot->dot_dimension_numbers();
  const int64_t num_batch_dims = original_dnums.lhs_batch_dimensions_size();
  const int64_t num_contracting_dims =
      original_dnums.lhs_contracting_dimensions_size();

  HloInstruction* lhs_operand = original_dot->mutable_operand(0);
  const auto& lhs_shape = lhs_operand->shape();
  const int64_t lhs_rank = lhs_shape.rank();
  const int64_t num_lhs_non_contracting_dims =
      lhs_rank - num_batch_dims - num_contracting_dims;

  std::vector<int64_t> lhs_non_contracting_dims;
  lhs_non_contracting_dims.reserve(num_lhs_non_contracting_dims);
  for (int64_t i = 0; i < lhs_rank; ++i) {
    if (!absl::c_linear_search(original_dnums.lhs_contracting_dimensions(),
                               i) &&
        !absl::c_linear_search(original_dnums.lhs_batch_dimensions(), i)) {
      lhs_non_contracting_dims.push_back(i);
    }
  }
  TF_ASSIGN_OR_RETURN(
      Shape lhs_reshape_shape,
      GetBatchRowColumnShape(lhs_shape, original_dnums.lhs_batch_dimensions(),
                             original_dnums.lhs_contracting_dimensions(),
                             lhs_non_contracting_dims));
  if (lhs_reshape_shape.dimensions(0) == 1) {
    lhs_reshape_shape.DeleteDimension(0);
  }
  // Reshape the batch, contracting and non-contracting dimensions together.
  HloInstruction* reshaped_lhs = computation->AddInstruction(
      HloInstruction::CreateBitcast(lhs_reshape_shape, lhs_operand),
      &lhs_operand->metadata());

  HloInstruction* rhs_operand = original_dot->mutable_operand(1);
  const auto& rhs_shape = rhs_operand->shape();
  const int64_t rhs_rank = rhs_shape.rank();
  const int64_t num_rhs_non_contracting_dims =
      rhs_rank - num_batch_dims - num_contracting_dims;
  std::vector<int64_t> rhs_non_contracting_dims;
  rhs_non_contracting_dims.reserve(num_rhs_non_contracting_dims);
  for (int64_t i = 0; i < rhs_rank; ++i) {
    if (!absl::c_linear_search(original_dnums.rhs_contracting_dimensions(),
                               i) &&
        !absl::c_linear_search(original_dnums.rhs_batch_dimensions(), i)) {
      rhs_non_contracting_dims.push_back(i);
    }
  }
  TF_ASSIGN_OR_RETURN(
      Shape rhs_reshape_shape,
      GetBatchRowColumnShape(rhs_shape, original_dnums.rhs_batch_dimensions(),
                             original_dnums.rhs_contracting_dimensions(),
                             rhs_non_contracting_dims));
  if (rhs_reshape_shape.dimensions(0) == 1) {
    rhs_reshape_shape.DeleteDimension(0);
  }
  // Reshape the batch, contracting and non-contracting dimensions together.
  HloInstruction* reshaped_rhs = computation->AddInstruction(
      HloInstruction::CreateBitcast(rhs_reshape_shape, rhs_operand),
      &rhs_operand->metadata());

  std::vector<int64_t> dot_dims;
  DotDimensionNumbers dot_dnums;
  int64_t batch_offset = 1;
  if (lhs_reshape_shape.rank() == 3) {
    dot_dnums.add_lhs_batch_dimensions(0);
    dot_dnums.add_rhs_batch_dimensions(0);
    dot_dims.push_back(lhs_reshape_shape.dimensions(0));
    dot_dims.push_back(lhs_reshape_shape.dimensions(2));
    dot_dims.push_back(rhs_reshape_shape.dimensions(2));
  } else {
    dot_dims.push_back(lhs_reshape_shape.dimensions(1));
    dot_dims.push_back(rhs_reshape_shape.dimensions(1));
    batch_offset = 0;
  }
  dot_dnums.add_lhs_contracting_dimensions(batch_offset);
  dot_dnums.add_rhs_contracting_dimensions(batch_offset);

  std::vector<int64_t> dims(original_dot->shape().rank());
  absl::c_iota(dims, 0);

  auto batch_dims = absl::Span<const int64_t>(dims).first(num_batch_dims);
  auto row_dims = absl::Span<const int64_t>(dims).subspan(
      num_batch_dims, num_lhs_non_contracting_dims);
  auto col_dims =
      absl::Span<const int64_t>(dims).last(num_rhs_non_contracting_dims);

  TF_ASSIGN_OR_RETURN(Shape new_dot_shape,
                      GetBatchRowColumnShape(original_dot->shape(), batch_dims,
                                             row_dims, col_dims));
  if (dot_dims.size() == 2) {
    new_dot_shape.DeleteDimension(0);
  }
  HloInstruction* dot = computation->AddInstruction(
      HloInstruction::CreateDot(new_dot_shape, reshaped_lhs, reshaped_rhs,
                                dot_dnums, original_dot->precision_config()));
  original_dot->SetupDerivedInstruction(dot);

  std::unique_ptr<HloInstruction> replacement =
      HloInstruction::CreateBitcast(original_dot->shape(), dot);
  VLOG(3) << "Canonicalizing dot:\n"
          << "\t old: " << original_dot->ToString() << "\n"
          << "\t new: " << dot->ToString() << "\n"
          << "\t   -> " << replacement->ToString();
  return computation->ReplaceWithNewInstruction(original_dot,
                                                std::move(replacement));
}

bool GpuDotDecomposer::DotIsCanonical(HloInstruction* dot) {
  // Skips sparse instruction as DotDecomposer does not know how to handle
  // sparse input yet.
  if (SparseUtil::HasSparseInOut(dot)) {
    return true;
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  // A dot it not canonical if there is more than one contracting dimension.
  if (dnums.lhs_contracting_dimensions_size() != 1) {
    return false;
  }
  // A dot is not canonical if it has more than one non-contracting
  // dimension.
  if (dnums.lhs_batch_dimensions_size() + 2 < dot->operand(0)->shape().rank() ||
      dnums.rhs_batch_dimensions_size() + 2 < dot->operand(1)->shape().rank()) {
    return false;
  }
  if (dnums.lhs_batch_dimensions().empty() &&
      dnums.lhs_contracting_dimensions().empty()) {
    return false;
  }
  return true;
}

}  // namespace gpu
}  // namespace xla
