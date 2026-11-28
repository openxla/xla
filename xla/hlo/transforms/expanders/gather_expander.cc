/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/gather_expander.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/layout_util.h"
#include "xla/overflow_util.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/while_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {

namespace {

void CopyAttrs(const Shape& src, Shape* dst) {
  if (!LayoutUtil::HasLayout(src) || !LayoutUtil::HasLayout(*dst)) return;
  auto* l = dst->mutable_layout();
  l->set_element_size_in_bits(src.layout().element_size_in_bits());
  l->set_memory_space(src.layout().memory_space());
}
void CopyAttrs(const Shape& src, HloInstruction* dst) {
  CopyAttrs(src, dst->mutable_shape());
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

bool IsIota(absl::Span<const int64_t> range) {
  for (int64_t i = 0; i < range.size(); ++i) {
    if (range[i] != i) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* start_indices, int64_t index_vector_dim) {
  const Shape& start_indices_shape = start_indices->shape();

  if (start_indices_shape.dimensions().size() == index_vector_dim) {
    return start_indices;
  }

  if (index_vector_dim == (start_indices_shape.dimensions().size() - 1)) {
    return start_indices;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(start_indices_shape.dimensions().size());
  for (int64_t i = 0, e = start_indices_shape.dimensions().size(); i < e; i++) {
    if (i != index_vector_dim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(index_vector_dim);
  return IsIota(permutation) ? start_indices
                             : MakeTransposeHlo(start_indices, permutation);
}

// Canonicalizes the start_indices tensors so that we only have deal with some
// specific cases in the while loop that does the heavy lifting.
//
// See the "High Level Algorithm" section for a broader picture.
absl::StatusOr<HloInstruction*> CanonicalizeGatherIndices(
    HloInstruction* start_indices, int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * transposed_start_indices,
      TransposeIndexVectorDimToLast(start_indices, index_vector_dim));
  bool indices_are_scalar =
      index_vector_dim == start_indices->shape().dimensions().size();

  // The number of dimensions in start_indices that are index dimensions.
  const int64_t index_dims_in_start_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. start_indices has rank 1 and this gather
  // is really just a dynamic slice) add a leading degenerate dimension for
  // uniformity.  Otherwise create a "collapsed" leading dimension that subsumes
  // all of the non-index-vector dimensions.
  const Shape& shape = transposed_start_indices->shape();
  CopyAttrs(start_indices->shape(), transposed_start_indices);
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * canonical,
      shape.dimensions().size() == index_dims_in_start_indices
          ? PrependDegenerateDims(transposed_start_indices, 1)
          : CollapseFirstNDims(
                transposed_start_indices,
                shape.dimensions().size() - index_dims_in_start_indices));
  CopyAttrs(start_indices->shape(), canonical);
  return canonical;
}

// Expands out or contracts away the gather dimensions in the accumulator
// produced by the while loop.
absl::StatusOr<HloInstruction*> AdjustBatchDimsInAccumulator(
    const Shape& start_indices_shape, HloInstruction* accumulator,
    int64_t index_vector_dim) {
  std::vector<int64_t> batch_dim_bounds;
  batch_dim_bounds.reserve(start_indices_shape.dimensions().size());
  for (int64_t i = 0, e = start_indices_shape.dimensions().size(); i < e; i++) {
    if (i != index_vector_dim) {
      batch_dim_bounds.push_back(start_indices_shape.dimensions(i));
    }
  }

  if (batch_dim_bounds.empty()) {
    // If batch_dim_bounds is empty we must be lowering a (effectively)
    // dynamic-slice.  In that case, there is a leading degenerate gather
    // dimension that we added to make this special case play well with the
    // general while loop which we need to remove now.
    return ElideDegenerateDims(accumulator, {0});
  }

  return ExpandFirstDimIntoNDims(accumulator, batch_dim_bounds);
}

// Returns the dimensions in a slice that are either collapsed or corresponding
// to an operand batching dimension.
std::vector<int64_t> GetDegeneratedSliceDims(
    const GatherDimensionNumbers& dim_numbers) {
  absl::Span<const int64_t> collapsed_slice_dims =
      dim_numbers.collapsed_slice_dims();
  absl::Span<const int64_t> batching_dims = dim_numbers.operand_batching_dims();
  std::vector<int64_t> removed_dims;
  removed_dims.reserve(collapsed_slice_dims.size() + batching_dims.size());
  absl::c_copy(collapsed_slice_dims, std::back_inserter(removed_dims));
  absl::c_copy(batching_dims, std::back_inserter(removed_dims));
  return removed_dims;
}

// This generates the body of the while that implements the main data movement
// behavior of gather using dynamic-slice and dynamic-update-slice.
absl::StatusOr<std::vector<HloInstruction*>> GatherLoopBody(
    const HloInstruction& gather, HloInstruction* induction_var,
    const std::vector<HloInstruction*>& incoming_loop_state) {
  const GatherDimensionNumbers& dim_numbers = gather.gather_dimension_numbers();
  CHECK_EQ(incoming_loop_state.size(), 3);
  HloInstruction* const operand = incoming_loop_state[0];
  HloInstruction* const start_indices = incoming_loop_state[1];
  HloInstruction* const output_accumulator = incoming_loop_state[2];
  const Shape& orig_start_indices_shape = gather.operand(1)->shape();

  bool has_scalar_indices = start_indices->shape().dimensions().size() == 1;
  CHECK_EQ(has_scalar_indices,
           dim_numbers.index_vector_dim() ==
               orig_start_indices_shape.dimensions().size());

  HloInstruction* induction_var_as_vector =
      MakeBroadcastHlo(induction_var, /*broadcast_dimensions=*/{},
                       /*result_shape_bounds=*/{1});

  HloInstruction* index_vector;

  if (has_scalar_indices) {
    // In this case start_indices has rank 1 and induction_var_as_vector (of
    // shape {1}) is an index into this rank 1 tensor.
    ABSL_ASSIGN_OR_RETURN(
        index_vector,
        MakeDynamicSliceHlo(start_indices, induction_var_as_vector, {1}));
  } else {
    // In this case start_indices has rank 2 and induction_var_as_vector (of
    // shape {1}) is an index into just the first dimension of this rank 2
    // tensor.
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * index_into_start_indices,
        PadVectorWithZeros(induction_var_as_vector,
                           /*zeros_to_prepend=*/0, /*zeros_to_append=*/1));

    int64_t index_vector_size = start_indices->shape().dimensions(1);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * index_vector_2d,
        MakeDynamicSliceHlo(start_indices, index_into_start_indices,
                            {1, index_vector_size}));
    CopyAttrs(start_indices->shape(), index_vector_2d);

    ABSL_ASSIGN_OR_RETURN(index_vector,
                          ElideDegenerateDims(index_vector_2d, {0}));
  }
  CopyAttrs(start_indices->shape(), index_vector);

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * gathered_slice_start,
      ExpandIndexVectorIntoOperandSpace(
          orig_start_indices_shape, operand->shape().dimensions().size(),
          dim_numbers.index_vector_dim(), dim_numbers.start_index_map(),
          dim_numbers.start_indices_batching_dims(),
          dim_numbers.operand_batching_dims(), index_vector, induction_var));

  ABSL_ASSIGN_OR_RETURN(HloInstruction * gathered_slice,
                        MakeDynamicSliceHlo(operand, gathered_slice_start,
                                            gather.gather_slice_sizes()));
  CopyAttrs(operand->shape(), gathered_slice);

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction* const gathered_slice_with_dims_collapsed,
      ElideDegenerateDims(gathered_slice,
                          GetDegeneratedSliceDims(dim_numbers)));
  CopyAttrs(output_accumulator->shape(), gathered_slice_with_dims_collapsed);

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction* const gathered_slice_for_update,
      PrependDegenerateDims(gathered_slice_with_dims_collapsed, 1));
  CopyAttrs(output_accumulator->shape(), gathered_slice_for_update);

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction* const index_vector_into_accumulator,
      PadVectorWithZeros(
          induction_var_as_vector, /*zeros_to_prepend=*/0,
          /*zeros_to_append=*/
          gathered_slice_with_dims_collapsed->shape().dimensions().size()));

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction* const updated_accumulator,
      MakeDynamicUpdateSliceHlo(output_accumulator, gathered_slice_for_update,
                                index_vector_into_accumulator));
  CopyAttrs(output_accumulator->shape(), updated_accumulator);

  // New loop state -- only the accumulator has changed.  The
  // WhileUtil::MakeCountedLoop functions takes care of the induction variable
  // and the while loop exit condition.
  return absl::StatusOr<std::vector<HloInstruction*>>{
      {operand, start_indices, updated_accumulator}};
}

HloInstruction* CreateGatherLoopAccumulatorInitValue(
    HloComputation* computation, const Shape& output_shape,
    absl::Span<const int64_t> slice_sizes, int64_t gather_loop_trip_count,
    const GatherDimensionNumbers& dim_numbers) {
  std::vector<int64_t> accumulator_state_shape_dims;
  accumulator_state_shape_dims.reserve(1 + slice_sizes.size());
  accumulator_state_shape_dims.push_back(gather_loop_trip_count);
  for (int64_t i = 0; i < slice_sizes.size(); i++) {
    if (!IsCollapsedOrBatchingDim(dim_numbers.collapsed_slice_dims(),
                                  dim_numbers.operand_batching_dims(), i)) {
      accumulator_state_shape_dims.push_back(slice_sizes[i]);
    }
  }
  Shape acc_shape = ShapeUtil::MakeShape(output_shape.element_type(),
                                         accumulator_state_shape_dims);
  CopyAttrs(output_shape, &acc_shape);
  return BroadcastZeros(computation, acc_shape);
}

// `accumulator` is almost the tensor the gather operation would have produced,
// except that it has the dimensions in the wrong order -- the batch dimensions
// are the major dimensions and the offset dimensions are the minor dimensions.
// Fix this up with a transpose.
absl::StatusOr<HloInstruction*> PermuteBatchAndOffsetDims(
    HloInstruction* accumulator, absl::Span<const int64_t> offset_dims,
    int64_t output_rank) {
  std::vector<int64_t> permutation;
  permutation.reserve(output_rank);

  int64_t batch_idx_counter = 0;
  int64_t offset_idx_counter = output_rank - offset_dims.size();
  for (int64_t i = 0; i < output_rank; i++) {
    bool is_offset_dim = absl::c_binary_search(offset_dims, i);
    if (is_offset_dim) {
      permutation.push_back(offset_idx_counter++);
    } else {
      permutation.push_back(batch_idx_counter++);
    }
  }

  return IsIota(permutation) ? accumulator
                             : MakeTransposeHlo(accumulator, permutation);
}

std::optional<int64_t> GatherLoopTripCount(const HloInstruction* g) {
  const Shape& idx = g->operand(1)->shape();
  int64_t iv_dim = g->gather_dimension_numbers().index_vector_dim();
  int64_t trip_count = 1;
  for (int64_t i = 0, e = idx.dimensions().size(); i < e; ++i) {
    if (i == iv_dim) continue;
    auto [prod, bad] = OverflowSafeMultiply(trip_count, idx.dimensions(i));
    if (bad) return std::nullopt;
    trip_count = prod;
  }
  return trip_count;
}

bool HasAnyDynamicShape(const HloInstruction* inst) {
  return !inst->shape().is_static() || !inst->operand(0)->shape().is_static() ||
         !inst->operand(1)->shape().is_static();
}

bool GatherIsBroadcast(const HloInstruction* gather_instr) {
  return absl::c_equal(gather_instr->gather_slice_sizes(),
                       gather_instr->operand(0)->shape().dimensions());
}

bool CanExpandGatherAsBroadcast(const HloInstruction* g) {
  const Shape& op = g->operand(0)->shape();
  const Shape& idx = g->operand(1)->shape();
  if (!GatherIsBroadcast(g) || op.is_unbounded_dynamic() ||
      idx.is_unbounded_dynamic() || g->shape().is_unbounded_dynamic()) {
    return false;
  }
  if (ShapeUtil::IsZeroElementArray(op)) return g->shape().is_static();
  const auto& d = g->gather_dimension_numbers();
  for (int64_t i = 0, e = idx.dimensions().size(); i < e; ++i) {
    if (i != d.index_vector_dim() && idx.is_dynamic_dimension(i)) return false;
  }
  return absl::c_none_of(GetDegeneratedSliceDims(d),
                         [&](int64_t c) { return op.is_dynamic_dimension(c); });
}
}  // namespace
// High Level Algorithm
//
// We follow the following steps in sequence:
//
//  1. We canonicalize the start_indices tensor such that it has rank
//     2 (i.e. is a matrix) where each row is an index vector into the
//     operand.
//  2. We iterate over the set of indices in the canonicalized
//     start_indices tensor using a while loop, accumulating slices
//     of the operand tensor into an accumulator using
//     DynamicUpdateSlice.
//  3. The accumulator result from the while loop from (2) is then
//     reshaped to split out all the individual gather dimensions and
//     then transposed to give the final result.
//
// As an example, if we started with the following operation:
//
//   HloModule TensorFlowGatherMultipleBatchDims
//
//   ENTRY main {
//     operand = s32[3,3] parameter(0)
//     indices = s32[2,2] parameter(1)
//     ROOT gather = s32[2,3,2] gather(operand, indices),
//         offset_dims={1},
//         collapsed_slice_dims={1},
//         start_index_map={1},
//         index_vector_dim=2,
//         slice_sizes={3, 1}
//   }
//
// We'd first reshape indices to s32[4,1], where each row is an index
// into operand.  We'd then run a loop to slice out 4 tensors of shape
// [3,1] out of operand into an accumulator of shape [4,3,1].  We then
// reshape this result to [2,2,3] and finally transpose it to [2,3,2].

absl::StatusOr<HloInstruction*> GatherExpander::ExpandInstruction(
    HloInstruction* gather_instr) {
  CHECK(!ShapeUtil::IsZeroElementArray(gather_instr->shape()));

  if (GatherIsBroadcast(gather_instr)) {
    if (!CanExpandGatherAsBroadcast(gather_instr)) {
      return Unimplemented("Dynamic gather broadcast is not supported: %s.",
                           gather_instr->ToString());
    }
    if (ShapeUtil::IsZeroElementArray(gather_instr->operand(0)->shape())) {
      HloInstruction* bcast =
          BroadcastZeros(gather_instr->parent(), gather_instr->shape());
      gather_instr->SetupDerivedInstruction(bcast);
      return Finish(gather_instr, bcast->mutable_operand(0), bcast);
    }
    Shape broadcast_operand_shape = ShapeUtil::DeleteDimensions(
        GetDegeneratedSliceDims(gather_instr->gather_dimension_numbers()),
        gather_instr->operand(0)->shape());
    CopyAttrs(gather_instr->operand(0)->shape(), &broadcast_operand_shape);
    ABSL_ASSIGN_OR_RETURN(HloInstruction * broadcast_operand,
                          MakeReshapeHlo(broadcast_operand_shape,
                                         gather_instr->mutable_operand(0)));
    gather_instr->SetupDerivedInstruction(broadcast_operand);
    HloInstruction* broadcast =
        MakeBroadcastHlo(broadcast_operand,
                         gather_instr->gather_dimension_numbers().offset_dims(),
                         gather_instr->shape());
    gather_instr->SetupDerivedInstruction(broadcast);
    return Finish(gather_instr, broadcast_operand, broadcast);
  }
  if (HasAnyDynamicShape(gather_instr)) {
    return Unimplemented("Dynamic gather expansion is not supported: %s.",
                         gather_instr->ToString());
  }

  HloComputation* computation = gather_instr->parent();
  HloInstruction* operand = gather_instr->mutable_operand(0);
  HloInstruction* start_indices = gather_instr->mutable_operand(1);
  const Shape& output_shape = gather_instr->shape();
  int64_t output_rank = output_shape.dimensions().size();

  const GatherDimensionNumbers& dim_numbers =
      gather_instr->gather_dimension_numbers();

  std::optional<int64_t> trip_count = GatherLoopTripCount(gather_instr);
  if (!trip_count.has_value() || !IsInt32(*trip_count)) {
    return Unimplemented(
        "Gather operations with more than 2147483647 gather indices are not "
        "supported. This error occurred for %s.",
        gather_instr->ToString());
  }
  int64_t gather_loop_trip_count = *trip_count;

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * canonical_start_indices,
      CanonicalizeGatherIndices(start_indices, dim_numbers.index_vector_dim()));

  CHECK_EQ(gather_loop_trip_count,
           canonical_start_indices->shape().dimensions(0));

  HloInstruction* accumulator_init = CreateGatherLoopAccumulatorInitValue(
      computation, output_shape, gather_instr->gather_slice_sizes(),
      gather_loop_trip_count, gather_instr->gather_dimension_numbers());

  absl::StatusOr<std::vector<HloInstruction*>> gather_loop_result_or_error =
      WhileUtil::MakeCountedLoop(
          computation, gather_loop_trip_count,
          {operand, canonical_start_indices, accumulator_init},
          [&](HloInstruction* indvar,
              const std::vector<HloInstruction*>& loop_state) {
            return GatherLoopBody(*gather_instr, indvar, loop_state);
          },
          gather_instr->metadata());

  ABSL_ASSIGN_OR_RETURN(std::vector<HloInstruction*> gather_loop_result,
                        gather_loop_result_or_error);

  HloInstruction* accumulator_result = gather_loop_result.back();

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction* const accumulator_with_batch_dims_decanonicalized,
      AdjustBatchDimsInAccumulator(start_indices->shape(), accumulator_result,
                                   dim_numbers.index_vector_dim()));
  CopyAttrs(output_shape, accumulator_with_batch_dims_decanonicalized);

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * result,
      PermuteBatchAndOffsetDims(accumulator_with_batch_dims_decanonicalized,
                                dim_numbers.offset_dims(), output_rank));
  CopyAttrs(output_shape, result);
  return Finish(gather_instr, accumulator_result->mutable_operand(0), result);
}

bool GatherExpander::InstructionMatchesPattern(HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kGather ||
      ShapeUtil::IsZeroElementArray(inst->shape())) {
    return false;
  }
  if (GatherIsBroadcast(inst)) {
    return CanExpandGatherAsBroadcast(inst);
  }
  return !HasAnyDynamicShape(inst) &&
         (mode_ == kEliminateAllGathers ||
          GatherLoopTripCount(inst) == std::optional<int64_t>{1});
}

namespace {

class GatherExpanderFactory : public HloPassFactory {
 public:
  absl::string_view name() const override { return "gather_expander"; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext&, const HloPassOptions& options) const override {
    static const absl::NoDestructor<
        absl::flat_hash_map<absl::string_view, GatherExpander::Mode>>
        kModes({
            {"eliminate_all_gathers", GatherExpander::kEliminateAllGathers},
            {"eliminate_simple_gathers",
             GatherExpander::kEliminateSimpleGathers},
        });
    ABSL_ASSIGN_OR_RETURN(GatherExpander::Mode mode,
                          options.GetEnum("mode", *kModes));
    return std::make_unique<GatherExpander>(mode);
  }
};

const HloPassFactory* const kGatherExpanderRegistration =
    RegisterHloPassFactory<GatherExpanderFactory>();

}  // namespace

}  // namespace xla
