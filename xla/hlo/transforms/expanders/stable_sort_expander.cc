/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/stable_sort_expander.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
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

// Returns the index of a non-wrapping 32-bit or 64-bit integral iota operand
// along the sort dimension (preferring S32, then U32, then 64-bit), or -1.
int64_t IotaOperandIndexForStableSort(const HloSortInstruction& sort) {
  const int64_t sort_dim = sort.sort_dimension();
  for (PrimitiveType ty : {S32, U32, S64, U64}) {
    for (int64_t i = 0; i < sort.operand_count(); ++i) {
      const HloInstruction* op = sort.operand(i);
      if (op->opcode() == HloOpcode::kIota &&
          Cast<HloIotaInstruction>(op)->iota_dimension() == sort_dim &&
          op->shape().element_type() == ty && sort_dim >= 0 &&
          sort_dim < op->shape().dimensions().size() &&
          IotaDimensionFitsInType(op->shape().dimensions(sort_dim), ty)) {
        return i;
      }
    }
  }
  return -1;
}

struct AddIotaResult {
  HloComputation* comparator = nullptr;
  HloInstruction* new_root = nullptr;
};

absl::StatusOr<AddIotaResult> AddIotaAndExpand(HloSortInstruction* sort) {
  HloComputation* computation = sort->parent();
  if (sort->shape().is_unbounded_dynamic() ||
      sort->operand(0)->shape().is_unbounded_dynamic()) {
    return Unimplemented("Unbounded dynamic stable sort is not implemented");
  }
  std::vector<Shape> new_shapes = sort->operand_count() == 1
                                      ? std::vector<Shape>{sort->shape()}
                                      : sort->shape().tuple_shapes();
  const int64_t sort_dim = sort->sort_dimension();
  const PrimitiveType iota_type =
      IotaDimensionFitsInType(new_shapes[0].dimensions(sort_dim), S32) ? S32
                                                                       : S64;
  Shape iota_shape = ShapeUtil::ChangeElementType(new_shapes[0], iota_type);
  if (iota_shape.has_layout()) {
    iota_shape.mutable_layout()->clear_tiles();
    iota_shape.mutable_layout()->set_tail_padding_alignment_in_elements(1);
  }
  std::vector<HloSharding> leaf_sh;
  if (sort->has_sharding() &&
      (!sort->sharding().IsTuple() ||
       sort->sharding().tuple_elements().size() == sort->operand_count())) {
    leaf_sh.reserve(sort->operand_count() + 1);
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      leaf_sh.push_back(
          sort->sharding().IsTuple()
              ? sort->sharding().GetSubSharding(sort->shape(), {i})
              : sort->sharding());
    }
  }
  auto finish = [&](std::unique_ptr<HloInstruction> inst, const HloSharding* sh,
                    bool allow_unreduced = true) {
    HloInstruction* added = computation->AddInstruction(std::move(inst));
    added->set_metadata(sort->metadata());
    if (sh != nullptr && (allow_unreduced || !sh->IsUnreduced()) &&
        sh->Validate(added->shape()).ok()) {
      added->set_sharding(*sh);
    }
    return added;
  };
  const HloSharding* key_sh = leaf_sh.empty() ? nullptr : &leaf_sh[0];
  HloInstruction* iota =
      finish(HloInstruction::CreateIota(ShapeUtil::MakeStaticShape(iota_shape),
                                        sort_dim),
             key_sh, false);
  HloInstruction* key = sort->mutable_operand(0);
  for (int64_t d = 0;
       iota_shape.is_dynamic() && d < iota_shape.dimensions().size(); ++d) {
    if (!iota_shape.is_dynamic_dimension(d) ||
        key->shape().dimensions(d) > std::numeric_limits<int32_t>::max()) {
      continue;
    }
    const HloSharding ds =
        (key_sh != nullptr &&
         (key_sh->IsReplicatedOrSingleDevice() || key_sh->IsManual()))
            ? *key_sh
            : HloSharding::Replicate();
    HloInstruction* dim_size =
        finish(HloInstruction::CreateGetDimensionSize(
                   ShapeUtil::MakeShape(S32, {}), key, d),
               key_sh != nullptr ? &ds : nullptr, false);
    Shape dyn_shape = iota->shape();
    dyn_shape.set_dynamic_dimension(d, true);
    iota = finish(
        HloInstruction::CreateSetDimensionSize(dyn_shape, iota, dim_size, d),
        key_sh, false);
  }

  // Create a new comparator with tie-breaker iota parameters.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  Shape scalar_shape = ShapeUtil::MakeShape(iota_type, {});
  const int64_t n = sort->operand_count();
  auto p_lhs = HloInstruction::CreateParameter(2 * n, scalar_shape,
                                               absl::StrCat("p.", n, ".lhs"));
  auto p_rhs = HloInstruction::CreateParameter(2 * n + 1, scalar_shape,
                                               absl::StrCat("p.", n, ".rhs"));
  auto new_compare = sort->GetModule()->AddEmbeddedComputation(
      sort->to_apply()->CloneWithReplacements(&replacements,
                                              {p_lhs.get(), p_rhs.get()}));

  // Replace the original sort op.
  std::vector<HloInstruction*> new_operands(sort->operands().begin(),
                                            sort->operands().end());
  new_operands.push_back(iota);
  new_shapes.push_back(iota->shape());
  Shape new_sort_shape = ShapeUtil::MakeTupleShape(new_shapes);

  auto* new_sort = Cast<HloSortInstruction>(computation->AddInstruction(
      sort->CloneWithNewOperands(new_sort_shape, new_operands)));
  new_sort->set_is_stable(false);
  new_sort->set_to_apply(new_compare);
  if (!leaf_sh.empty()) {
    leaf_sh.push_back(leaf_sh[0].IsUnreduced() ? HloSharding::Replicate()
                                               : leaf_sh[0]);
    HloSharding new_sh = HloSharding::Tuple(new_sort_shape, leaf_sh);
    if (new_sh.Validate(new_sort_shape).ok()) new_sort->set_sharding(new_sh);
  }
  ABSL_RETURN_IF_ERROR(sort->CopyAllControlDepsTo(new_sort, new_sort));
  ABSL_RETURN_IF_ERROR(sort->DropAllControlDeps());

  // Wrap new_sort to preserve the original single-array or tuple output shape.
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(sort->operand_count());
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    tuple_elements.push_back(finish(
        HloInstruction::CreateGetTupleElement(new_shapes[i], new_sort, i),
        leaf_sh.empty() ? nullptr : &leaf_sh[i]));
  }
  HloInstruction* new_root = tuple_elements[0];
  if (tuple_elements.size() > 1) {
    new_root = finish(HloInstruction::CreateTuple(tuple_elements),
                      sort->has_sharding() ? &sort->sharding() : nullptr);
  }

  return AddIotaResult{new_compare, new_root};
}

}  // namespace

// Looks for a iota operand that can be used as tie breaker in the computation.
// If no matching iota operand is found, a iota operand is added to Sort. The
// comparison computation is adjusted to break ties using the values from the
// iota operand.
absl::StatusOr<HloInstruction*> StableSortExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* sort = Cast<HloSortInstruction>(instruction);
  if (sort->to_apply()->HasSideEffect()) {
    return FailedPrecondition(
        "Cannot expand stable sort with side-effecting comparator: %s",
        sort->name());
  }
  if (sort->operand_count() == 0 || sort->sort_dimension() < 0 ||
      sort->sort_dimension() >= sort->operand(0)->shape().dimensions().size() ||
      sort->to_apply()->num_parameters() != 2 * sort->operand_count() ||
      (sort->operand_count() > 1 &&
       (!sort->shape().IsTuple() ||
        sort->shape().tuple_shapes().size() != sort->operand_count()))) {
    return InvalidArgument("Invalid stable sort instruction: %s", sort->name());
  }

  int64_t iota_index = IotaOperandIndexForStableSort(*sort);
  HloComputation* comparator = nullptr;
  HloInstruction* new_root = nullptr;

  // If there is currently no iota operand which we could use for making the
  // sort stable, we will have to add a new such operand.
  if (iota_index == -1) {
    ABSL_ASSIGN_OR_RETURN(auto result, AddIotaAndExpand(sort));
    comparator = result.comparator;
    new_root = result.new_root;
    iota_index = sort->operand_count();
  } else {
    HloComputation* compare = sort->to_apply();
    if (compare->caller_instructions().size() > 1 ||
        compare == sort->GetModule()->entry_computation()) {
      compare = sort->GetModule()->AddEmbeddedComputation(compare->Clone());
      sort->set_to_apply(compare);
    }
    sort->set_is_stable(false);
    comparator = compare;
  }

  // Modify the computation to break ties using the iota operand.
  std::vector<HloInstruction*> instructions_postorder =
      comparator->MakeInstructionPostOrder();
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;
  // Look up instr in the replacements map, and return either the replacement,
  // or instr, if the replacement isn't present.
  auto replace = [&](HloInstruction* instr) {
    auto it = replacements.find(instr);
    if (it == replacements.end()) {
      return instr;
    }
    return it->second;
  };
  HloInstruction* old_root = comparator->root_instruction();
  // The comparison computation gets 2 * n parameters (n being the number of
  // operands of Sort), where parameters 2 * i and 2 * i + 1 correspond to two
  // different scalars of operand i of Sort which are to be compared. The
  // comparison computation should induce a strict weak order, so if
  // to_apply(p1.lhs, p1.rhs, ..., pn.lhs, pn.rhs) is equal to
  // to_apply(p1.rhs, p1.lhs, ..., pn.rhs, pn.lhs), we can conclude that the
  // values to be compared are equivalent, and perform a tie-breaker comparison.
  //
  // We clone each instruction with at least one operand, but use as new
  // operands of the instruction the replacements of the original operands.
  // Parameter 2 * i is replaced by parameter 2 * i + 1 and vice versa. This
  // should make sure that the cloned root instruction gives the result of the
  // comparison computation when being called with each scalar pair reversed.
  // parameters corresponding to the iota operand.
  for (int64_t i = 0; i < comparator->num_parameters(); ++i) {
    replacements[comparator->parameter_instruction(i)] =
        comparator->parameter_instruction(i ^ 1);
  }
  for (HloInstruction* inst : instructions_postorder) {
    if (inst->operand_count() == 0) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(inst->operand_count());
    for (HloInstruction* operand : inst->operands()) {
      new_operands.push_back(replace(operand));
    }
    HloInstruction* cloned = comparator->AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), new_operands));
    replacements[inst] = cloned;
    for (HloInstruction* pred : inst->control_predecessors()) {
      ABSL_RETURN_IF_ERROR(replace(pred)->AddControlDependencyTo(cloned));
    }
  }
  HloInstruction* cloned_root = replace(old_root);
  CHECK_NE(cloned_root, nullptr);
  const Shape& pred_shape = old_root->shape();
  auto add_cmp_op = [&](std::unique_ptr<HloInstruction> op) {
    HloInstruction* added = comparator->AddInstruction(std::move(op));
    added->set_metadata(sort->metadata());
    return added;
  };
  HloInstruction* same = add_cmp_op(HloInstruction::CreateCompare(
      pred_shape, old_root, cloned_root, ComparisonDirection::kEq));
  HloInstruction* tie_breaker = add_cmp_op(HloInstruction::CreateCompare(
      pred_shape, comparator->parameter_instruction(2 * iota_index),
      comparator->parameter_instruction(2 * iota_index + 1),
      ComparisonDirection::kLt));
  comparator->set_root_instruction(add_cmp_op(HloInstruction::CreateTernary(
      pred_shape, HloOpcode::kSelect, same, tie_breaker, old_root)));

  return new_root;
}

bool StableSortExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kSort &&
         Cast<HloSortInstruction>(instruction)->is_stable();
}
}  // namespace xla
