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

#include "xla/hlo/transforms/expanders/permutation_sort_expander.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IotaDimensionFitsInType(int64_t dim_size, PrimitiveType ty) {
  if (!primitive_util::IsIntegralType(ty) || dim_size < 0) return false;
  if (dim_size <= 1) return true;
  const int bits = primitive_util::BitWidth(ty);
  const uint64_t max_idx =
      primitive_util::IsSignedIntegralType(ty)
          ? (uint64_t{1} << (bits - 1)) - 1
          : (bits == 64 ? std::numeric_limits<uint64_t>::max()
                        : (uint64_t{1} << bits) - 1);
  return static_cast<uint64_t>(dim_size - 1) <= max_idx;
}

void SetShardingIfValid(HloInstruction* inst, const HloSharding& sharding,
                        bool allow_unreduced = true) {
  if ((allow_unreduced || !sharding.IsUnreduced()) &&
      sharding.Validate(inst->shape()).ok()) {
    inst->set_sharding(sharding);
  }
}

bool IsSimpleLessThanComparator(HloComputation* compare) {
  HloInstruction* root = compare->root_instruction();
  return !compare->HasSideEffect() && compare->num_parameters() >= 2 &&
         HloPredicateIsOp<HloOpcode::kCompare>(root) &&
         root->comparison_direction() == ComparisonDirection::kLt &&
         root->operand(0) == compare->parameter_instruction(0) &&
         root->operand(1) == compare->parameter_instruction(1);
}

}  // namespace

bool PermutationSortExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloPredicateIsNotOp<HloOpcode::kSort>(instruction) ||
      instruction->operand_count() != 2 ||
      instruction->dimensions().size() != 1 ||
      !instruction->shape().IsTuple() ||
      instruction->shape().tuple_shapes().size() != 2 ||
      instruction->shape().is_dynamic() ||
      !IsSimpleLessThanComparator(instruction->to_apply())) {
    return false;
  }
  const HloInstruction* sort_key = instruction->operand(0);
  const HloInstruction* values = instruction->operand(1);
  const int64_t sort_dim = instruction->dimensions(0);
  const PrimitiveType key_ty = sort_key->shape().element_type();
  if (!HloPredicateIsOp<HloOpcode::kGetTupleElement>(sort_key) ||
      sort_key->shape().is_dynamic() || values->shape().is_dynamic() ||
      sort_dim < 0 || sort_dim >= sort_key->shape().dimensions().size() ||
      instruction->shape().tuple_shapes(0).element_type() != key_ty ||
      !ShapeUtil::SameDimensions(sort_key->shape(), values->shape())) {
    return false;
  }
  for (const HloInstruction* user : instruction->users()) {
    if (!instruction->control_successors().empty() &&
        HloPredicateIsOp<HloOpcode::kGetTupleElement>(user) &&
        user->tuple_index() == 1 && !user->control_predecessors().empty()) {
      return false;
    }
  }
  const HloInstruction* other_sort = sort_key->operand(0);
  const int64_t idx = sort_key->tuple_index();
  if (!HloPredicateIsOp<HloOpcode::kSort>(other_sort) ||
      other_sort->operand_count() < 2 || other_sort->dimensions().size() != 1 ||
      other_sort->dimensions(0) != sort_dim || idx < 0 ||
      idx >= other_sort->operand_count() ||
      !HloPredicateIsOp<HloOpcode::kIota>(other_sort->operand(idx))) {
    return false;
  }
  const auto* iota = Cast<HloIotaInstruction>(other_sort->operand(idx));
  return !iota->shape().is_dynamic() && iota->iota_dimension() == sort_dim &&
         iota->shape().element_type() == key_ty &&
         ShapeUtil::SameDimensions(iota->shape(), sort_key->shape()) &&
         IotaDimensionFitsInType(iota->shape().dimensions(sort_dim), key_ty);
}

absl::StatusOr<HloInstruction*> PermutationSortExpander::ExpandInstruction(
    HloInstruction* instruction) {
  // We use the following construction method for a Scatter that applies
  // the permutation from `keys` to the `values` parameter.
  // - Set updates of the scatter to be a reshaped `values` parameter of sort
  //   (adding `rank` many 1 dimensions at the end).
  // - Take the `keys` parameter of the second sort and reshape it to have
  //   another `1` dimension at the end. This is the permutation we want to
  //   apply in the `dimension_to_sort` dimension.
  // - We want to create scatter indices such that the most minor dimension
  //   represents the indices in order from most major to most minor where the
  //   scatter updates should be stored. This can be done with a concatenate of
  //   iotas with different iota dimensions, except the index that corresponds
  //   to the dimension to sort. For that we will use the reshaped `keys`
  //   parameter. E.g. with rank 3 and dimension_to_sort = 1, we would have
  //   concatenate of (iota with iota_dimension=0, keys, iota with
  //   iota_dimension = 2).
  int64_t dimension_to_sort = instruction->dimensions(0);
  HloInstruction* sort_key = instruction->mutable_operand(0);
  const Shape& key_shape = sort_key->shape();
  int rank = static_cast<int>(key_shape.dimensions().size());
  const PrimitiveType key_type = key_shape.element_type();
  int64_t max_batch = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dimension_to_sort)
      max_batch = std::max(max_batch, key_shape.dimensions(i));
  }
  const int64_t max_dim =
      std::max(max_batch, key_shape.dimensions(dimension_to_sort));
  const PrimitiveType idx_ty =
      IotaDimensionFitsInType(max_batch, key_type)
          ? key_type
          : (IotaDimensionFitsInType(max_dim, S32) ? S32 : S64);
  Shape index_base = idx_ty == key_type
                         ? key_shape
                         : ShapeUtil::ChangeElementType(key_shape, idx_ty);
  if (index_base.has_layout()) {
    index_base.mutable_layout()->clear_tiles();
    index_base.mutable_layout()->set_tail_padding_alignment_in_elements(1);
  }
  auto add = [&](auto op) {
    return instruction->AddInstruction(std::move(op));
  };
  HloInstruction* index_key =
      idx_ty == key_type
          ? sort_key
          : add(HloInstruction::CreateConvert(index_base, sort_key));
  Shape extended_shape = index_base;
  ShapeUtil::AppendMinorDimension(1, &extended_shape);
  HloInstruction* reshaped_permutation =
      add(HloInstruction::CreateReshape(extended_shape, index_key));
  std::vector<HloInstruction*> concat_operands;
  concat_operands.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    concat_operands.push_back(
        i == dimension_to_sort
            ? reshaped_permutation
            : add(HloInstruction::CreateIota(extended_shape, i)));
  }
  Shape concat_shape = index_base;
  ShapeUtil::AppendMinorDimension(rank, &concat_shape);
  HloInstruction* scatter_indices =
      rank > 1 ? add(HloInstruction::CreateConcatenate(concat_shape,
                                                       concat_operands, rank))
               : reshaped_permutation;

  // We don't care about the operand, it will be completely overridden by
  // the updates. Initialize it to zero.
  const Shape& update_shape = instruction->shape().tuple_shapes(1);
  PrimitiveType elem_ty = update_shape.element_type();
  const int64_t eb = update_shape.has_layout()
                         ? update_shape.layout().element_size_in_bits()
                         : 0;
  auto set_eb = [&](Shape* s) {
    if (eb != 0 && s->has_layout()) {
      s->mutable_layout()->set_element_size_in_bits(eb);
    }
  };
  HloInstruction* values = instruction->mutable_operand(1);
  if (values->shape().element_type() != elem_ty) {
    Shape conv = ShapeUtil::ChangeElementType(values->shape(), elem_ty);
    set_eb(&conv);
    values = add(HloInstruction::CreateConvert(conv, values));
  }
  HloInstruction* zero =
      add(HloInstruction::CreateConstant(LiteralUtil::Zero(elem_ty)));
  set_eb(zero->mutable_shape());
  HloInstruction* scatter_operand =
      add(HloInstruction::CreateBroadcast(update_shape, zero, {}));

  // Build replacement reducer using the output element type.
  HloComputation::Builder b("update_replace_computation");
  Shape scalar_shape = ShapeUtil::MakeShape(elem_ty, {});
  set_eb(&scalar_shape);
  b.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "scalar_lhs"));
  HloInstruction* scalar_rhs = b.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "scalar_rhs"));
  HloComputation* update_replace_computation =
      instruction->GetModule()->AddEmbeddedComputation(b.Build(scalar_rhs));
  update_replace_computation->SetExecutionThread(
      instruction->parent()->execution_thread());

  ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(rank);
  for (int64_t i = 0; i < rank; ++i) {
    dim_numbers.add_inserted_window_dims(i);
    dim_numbers.add_scatter_dims_to_operand_dims(i);
  }
  HloInstruction* scatter = add(HloInstruction::CreateScatter(
      update_shape, scatter_operand, scatter_indices, values,
      update_replace_computation, dim_numbers, /*indices_are_sorted=*/false,
      /*unique_indices=*/true));
  HloInstruction* output_iota = add(HloInstruction::CreateIota(
      instruction->shape().tuple_shapes(0), dimension_to_sort));
  if (instruction->has_sharding() &&
      (!instruction->sharding().IsTuple() ||
       instruction->sharding().tuple_elements().size() == 2)) {
    const HloSharding& s = instruction->sharding();
    const HloSharding ks =
        s.IsTuple() ? s.GetSubSharding(instruction->shape(), {0}) : s;
    const HloSharding vs =
        s.IsTuple() ? s.GetSubSharding(instruction->shape(), {1}) : s;
    if (index_key != sort_key) SetShardingIfValid(index_key, ks, false);
    for (HloInstruction* o : concat_operands) SetShardingIfValid(o, ks, false);
    SetShardingIfValid(scatter_indices, ks, false);
    SetShardingIfValid(output_iota, ks, false);
    const bool keep_vs = vs.IsReplicatedOrSingleDevice() || vs.IsManual();
    SetShardingIfValid(zero, keep_vs ? vs : HloSharding::Replicate(), false);
    if (values != instruction->operand(1)) SetShardingIfValid(values, vs);
    SetShardingIfValid(scatter_operand, vs);
    SetShardingIfValid(scatter, vs);
  }
  for (HloInstruction* pred : instruction->control_predecessors()) {
    ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(output_iota));
  }
  ABSL_RETURN_IF_ERROR(instruction->CopyAllControlDepsTo(scatter, scatter));
  ABSL_RETURN_IF_ERROR(instruction->DropAllControlDeps());
  return add(HloInstruction::CreateTuple({output_iota, scatter}));
}

}  // namespace xla
