/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/overflow_util.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace {

bool IsExplicitBatchGather(const GatherDimensionNumbers& d) {
  return !d.operand_batching_dims().empty() ||
         !d.start_indices_batching_dims().empty();
}

void CopyAttrs(const Shape& src, Shape* dst) {
  if (!LayoutUtil::HasLayout(src) || !LayoutUtil::HasLayout(*dst)) return;
  auto* l = dst->mutable_layout();
  l->set_element_size_in_bits(src.layout().element_size_in_bits());
  l->set_memory_space(src.layout().memory_space());
}

absl::StatusOr<HloInstruction*> Finish(HloInstruction* src, HloInstruction* in,
                                       HloInstruction* root) {
  for (HloInstruction* pred : src->control_predecessors()) {
    if (in != root) ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(in));
  }
  ABSL_RETURN_IF_ERROR(root->CopyAllControlDepsFrom(src));
  ABSL_RETURN_IF_ERROR(src->DropAllControlDeps());
  return root;
}

bool CanSimplifyGather(const HloGatherInstruction* g) {
  const auto& d = g->gather_dimension_numbers();
  const Shape& op = g->operand(0)->shape();
  const Shape& idx = g->operand(1)->shape();
  if (IsExplicitBatchGather(d) || op.is_unbounded_dynamic() ||
      idx.is_unbounded_dynamic() || g->shape().is_unbounded_dynamic()) {
    return false;
  }
  if (absl::c_linear_search(g->gather_slice_sizes(), 0) ||
      op.dimensions().empty()) {
    return g->shape().is_static();
  }
  for (int64_t c : d.collapsed_slice_dims()) {
    if (op.is_dynamic_dimension(c)) return false;
  }
  int64_t prod = 1, dyn = 0;
  for (int64_t i = 0, e = idx.dimensions().size(); i < e; ++i) {
    if (i == d.index_vector_dim()) {
      if (idx.is_dynamic_dimension(i)) return false;
      continue;
    }
    auto [next, bad] = OverflowSafeMultiply(prod, idx.dimensions(i));
    if (bad || (dyn += idx.is_dynamic_dimension(i) ? 1 : 0) > 1) return false;
    prod = next;
  }
  return true;
}

}  // namespace

absl::StatusOr<HloInstruction*> GatherSimplifier::ExpandInstruction(
    HloInstruction* inst) {
  auto* gather = DynCast<HloGatherInstruction>(inst);
  if (gather == nullptr || !CanSimplifyGather(gather)) {
    return InvalidArgument("Gather cannot be simplified: %s", inst->ToString());
  }
  if (absl::c_linear_search(gather->gather_slice_sizes(), 0)) {
    auto* zero = gather->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(gather->shape().element_type())));
    return Finish(gather, zero,
                  gather->AddInstruction(HloInstruction::CreateBroadcast(
                      gather->shape(), zero, {})));
  }

  const auto& dims = gather->gather_dimension_numbers();
  auto* operand = gather->operands()[0];
  auto* start_indices = gather->operands()[1];
  const int64_t operand_rank = operand->shape().dimensions().size();
  if (operand_rank == 0) {
    auto* bcast = gather->AddInstruction(
        HloInstruction::CreateBroadcast(gather->shape(), operand, {}));
    return Finish(gather, bcast, bcast);
  }

  // Make the start_indices a two-dimensional tensor.
  ABSL_ASSIGN_OR_RETURN(
      start_indices,
      TransformStartIndices(start_indices, dims.index_vector_dim()));

  // Permute the slice sizes according to start_index_map and compute the new
  // output shape for the Gather op.
  const auto slice_sizes = gather->gather_slice_sizes();
  std::vector<int64_t> output_dims = {start_indices->shape().dimensions(0)};
  absl::c_copy(slice_sizes, std::back_inserter(output_dims));
  std::vector<bool> output_dyn = {
      start_indices->shape().is_dynamic_dimension(0)};
  for (int64_t d = 0; d < operand_rank; ++d) {
    output_dyn.push_back(operand->shape().is_dynamic_dimension(d) &&
                         slice_sizes[d] == operand->shape().dimensions(d));
  }
  Shape output_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                            output_dims, output_dyn);
  CopyAttrs(gather->shape(), &output_shape);

  std::vector<int64_t> offset_dims(operand_rank);
  absl::c_iota(offset_dims, 1);

  auto* inner_gather = gather->AddInstruction(HloInstruction::CreateGather(
      output_shape, operand, start_indices,
      HloGatherInstruction::MakeGatherDimNumbers(offset_dims,
                                                 /*collapsed_slice_dims=*/{},
                                                 dims.start_index_map(),
                                                 /*index_vector_dim=*/1),
      slice_sizes, gather->indices_are_sorted()));
  HloInstruction* result = inner_gather;

  // Collapse the requested slice dimensions.
  if (!dims.collapsed_slice_dims().empty()) {
    std::vector<int64_t> collapsed_slice_dims(
        dims.collapsed_slice_dims().size());
    absl::c_transform(dims.collapsed_slice_dims(), collapsed_slice_dims.begin(),
                      [](int64_t dim) { return dim + 1; });
    ABSL_ASSIGN_OR_RETURN(result,
                          ElideDegenerateDims(result, collapsed_slice_dims));
    CopyAttrs(gather->shape(), result->mutable_shape());
  }

  // Expand the start index dimensions.
  const Shape& orig_idx = gather->operand(1)->shape();
  std::vector<int64_t> start_indices_dims;
  std::vector<bool> start_indices_dyn;
  for (int64_t i = 0; i < orig_idx.dimensions().size(); ++i) {
    if (i != dims.index_vector_dim()) {
      start_indices_dims.push_back(orig_idx.dimensions(i));
      start_indices_dyn.push_back(orig_idx.is_dynamic_dimension(i));
    }
  }
  if (start_indices_dims.size() > 1) {
    std::vector<int64_t> exp_dims = start_indices_dims;
    std::vector<bool> exp_dyn = start_indices_dyn;
    absl::c_copy(result->shape().dimensions().subspan(1),
                 std::back_inserter(exp_dims));
    absl::c_copy(result->shape().dynamic_dimensions().subspan(1),
                 std::back_inserter(exp_dyn));
    Shape exp =
        ShapeUtil::MakeShape(result->shape().element_type(), exp_dims, exp_dyn);
    CopyAttrs(gather->shape(), &exp);
    ABSL_ASSIGN_OR_RETURN(result, MakeReshapeHlo(exp, result));
  } else if (start_indices_dims.empty()) {
    ABSL_ASSIGN_OR_RETURN(result, ElideDegenerateDims(result, {0}));
    CopyAttrs(gather->shape(), result->mutable_shape());
  }

  // Move the offset dims to the final locations.
  std::vector<int64_t> output_perm;
  auto output_rank = static_cast<int64_t>(start_indices_dims.size() +
                                          dims.offset_dims().size());
  output_perm.reserve(output_rank);
  auto offset_dim_index = static_cast<int64_t>(start_indices_dims.size());
  int64_t start_index_dim_index = 0;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (absl::c_linear_search(dims.offset_dims(), i)) {
      output_perm.push_back(offset_dim_index++);
    } else {
      output_perm.push_back(start_index_dim_index++);
    }
  }
  ABSL_ASSIGN_OR_RETURN(result, MaybeTranspose(result, output_perm));
  return Finish(gather, inner_gather, result);
}

bool GatherSimplifier::IsSimplifiedGather(const HloGatherInstruction* gather) {
  const auto& dims = gather->gather_dimension_numbers();
  const int64_t operand_rank = gather->operand(0)->shape().dimensions().size();
  if (IsExplicitBatchGather(dims) || operand_rank == 0 ||
      absl::c_linear_search(gather->gather_slice_sizes(), 0) ||
      gather->operand(1)->shape().dimensions().size() != 2 ||
      dims.index_vector_dim() != 1 || !dims.collapsed_slice_dims().empty() ||
      dims.offset_dims().size() != operand_rank) {
    return false;
  }
  for (int64_t i = 0; i < operand_rank; ++i) {
    if (dims.offset_dims(i) != i + 1) return false;
  }
  return true;
}

bool GatherSimplifier::InstructionMatchesPattern(HloInstruction* inst) {
  auto* gather = DynCast<HloGatherInstruction>(inst);
  return gather && !IsSimplifiedGather(gather) && CanSimplifyGather(gather);
}

}  // namespace xla
