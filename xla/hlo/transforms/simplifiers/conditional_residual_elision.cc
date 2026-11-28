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

#include "xla/hlo/transforms/simplifiers/conditional_residual_elision.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Custom call target of a buffer that is allocated but not written.
constexpr absl::string_view kAllocateBuffer = "AllocateBuffer";

// Whether `a` and `b` are distinct conditionals that take the same branch:
// one instruction selects it for both, among as many branches.
bool TakeSameBranch(const HloInstruction* a, const HloInstruction* b) {
  return a != b && a->opcode() == HloOpcode::kConditional &&
         b->opcode() == HloOpcode::kConditional &&
         a->operand(0) == b->operand(0) &&
         a->branch_count() == b->branch_count();
}

// Returns `value` without the bitcasts on top of it, which are appended to
// `bitcasts`, outermost first.
HloInstruction* StripBitcasts(HloInstruction* value,
                              std::vector<HloInstruction*>* bitcasts) {
  while (value->opcode() == HloOpcode::kBitcast) {
    bitcasts->push_back(value);
    value = value->mutable_operand(0);
  }
  return value;
}

// If `value` is an element of the parameter of its computation, possibly
// bitcast or passed through optimization barriers, returns the read of that
// element and appends the bitcasts to `bitcasts`, outermost first. Otherwise
// returns nullptr.
HloInstruction* SavedOperand(HloInstruction* value,
                             std::vector<HloInstruction*>* bitcasts) {
  while (true) {
    value = StripBitcasts(value, bitcasts);
    if (value->opcode() != HloOpcode::kGetTupleElement) {
      return nullptr;
    }
    HloInstruction* tuple = value->mutable_operand(0);
    if (tuple->opcode() == HloOpcode::kParameter) {
      return value;
    }
    // An optimization barrier returns the elements of its operand unchanged.
    if (tuple->opcode() != HloOpcode::kOptimizationBarrier ||
        tuple->operand(0)->opcode() != HloOpcode::kTuple) {
      return nullptr;
    }
    value = tuple->mutable_operand(0)->mutable_operand(value->tuple_index());
  }
}

// Removes `instruction`, which has no control dependencies, if it has no
// users. Returns whether it did.
absl::StatusOr<bool> RemoveIfDead(HloInstruction* instruction) {
  if (!instruction->IsDead()) {
    return false;
  }
  ABSL_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(instruction));
  return true;
}

// Makes `consumer` read each residual that a conditional taking the same
// branch returns unchanged from its own operand straight from that operand.
absl::StatusOr<bool> ForwardPassThroughs(HloInstruction* consumer) {
  HloComputation* computation = consumer->parent();
  bool changed = false;
  for (int64_t branch = 0; branch < consumer->branch_count(); ++branch) {
    HloInstruction* tuple = consumer->mutable_operand(branch + 1);
    if (tuple->opcode() != HloOpcode::kTuple) {
      continue;
    }
    for (int64_t i = 0; i < tuple->operand_count(); ++i) {
      HloInstruction* element = tuple->mutable_operand(i);
      // The bitcasts between the operand of the producer and `tuple`.
      std::vector<HloInstruction*> bitcasts;
      HloInstruction* residual = StripBitcasts(element, &bitcasts);
      if (residual->opcode() != HloOpcode::kGetTupleElement ||
          !TakeSameBranch(residual->operand(0), consumer)) {
        continue;
      }
      // `consumer` stops reading these instructions, which would drop the
      // order that their control dependencies ask for.
      if (tuple->HasControlDependencies() ||
          residual->HasControlDependencies() ||
          absl::c_any_of(bitcasts, [](const HloInstruction* bitcast) {
            return bitcast->HasControlDependencies();
          })) {
        continue;
      }
      HloInstruction* producer = residual->mutable_operand(0);
      HloInstruction* root =
          producer->branch_computation(branch)->root_instruction();
      if (root->opcode() != HloOpcode::kTuple) {
        continue;
      }
      HloInstruction* saved = SavedOperand(
          root->mutable_operand(residual->tuple_index()), &bitcasts);
      if (saved == nullptr) {
        continue;
      }
      HloInstruction* operand = producer->mutable_operand(branch + 1);
      const int64_t saved_index = saved->tuple_index();
      // A bitcast only keeps its meaning on an operand of the same layout, and
      // the tuple only keeps its shape with an element of the same layout.
      const Shape* shape = &operand->shape().tuple_shapes(saved_index);
      bool same_shapes = true;
      for (auto it = bitcasts.rbegin(); it != bitcasts.rend(); ++it) {
        same_shapes &= ShapeUtil::Equal(*shape, (*it)->operand(0)->shape());
        shape = &(*it)->shape();
      }
      if (!same_shapes || !ShapeUtil::Equal(*shape, element->shape())) {
        continue;
      }
      HloInstruction* value = operand->opcode() == HloOpcode::kTuple
                                  ? operand->mutable_operand(saved_index)
                                  : computation->AddInstruction(
                                        HloInstruction::CreateGetTupleElement(
                                            operand, saved_index));
      for (auto it = bitcasts.rbegin(); it != bitcasts.rend(); ++it) {
        value = computation->AddInstruction(
            (*it)->CloneWithNewOperands((*it)->shape(), {value}));
      }
      // `consumer` gets a tuple of its own: other readers of `tuple` keep the
      // residual.
      std::vector<HloInstruction*> elements(tuple->operands().begin(),
                                            tuple->operands().end());
      elements[i] = value;
      HloInstruction* forwarded = computation->AddInstruction(
          tuple->CloneWithNewOperands(tuple->shape(), elements));
      ABSL_RETURN_IF_ERROR(consumer->ReplaceOperandWith(branch + 1, forwarded));
      changed = true;
      // Remove the tuple and the reads of the residual that this leaves
      // without users, so that the output of `producer` counts as unread.
      ABSL_ASSIGN_OR_RETURN(bool removed, RemoveIfDead(tuple));
      while (removed && element != producer) {
        HloInstruction* read = element;
        element = element->mutable_operand(0);
        ABSL_ASSIGN_OR_RETURN(removed, RemoveIfDead(read));
      }
      tuple = forwarded;
    }
  }
  return changed;
}

// Whether `branch` may read element `index` of its operand.
bool MayReadElement(const HloComputation* branch, int64_t index) {
  const HloInstruction* parameter = branch->parameter_instruction(0);
  return parameter->IsRoot() ||
         absl::c_any_of(parameter->users(), [&](const HloInstruction* user) {
           return user->opcode() != HloOpcode::kGetTupleElement ||
                  user->tuple_index() == index;
         });
}

// Sets in `branches` the branches in which a conditional taking the same
// branch as `producer` may read `output`, an output of `producer`. Returns
// false if anything else may read it.
bool FindReadingBranches(const HloInstruction* producer,
                         const HloInstruction* output,
                         std::vector<bool>* branches) {
  std::vector<const HloInstruction*> worklist = {output};
  while (!worklist.empty()) {
    const HloInstruction* value = worklist.back();
    worklist.pop_back();
    if (value->IsRoot()) {
      return false;
    }
    for (const HloInstruction* user : value->users()) {
      if (user->opcode() == HloOpcode::kBitcast) {
        worklist.push_back(user);
        continue;
      }
      if (user->opcode() != HloOpcode::kTuple || user->IsRoot()) {
        return false;
      }
      for (const HloInstruction* reader : user->users()) {
        if (!TakeSameBranch(reader, producer)) {
          return false;
        }
        for (int64_t branch = 0; branch < reader->branch_count(); ++branch) {
          if (reader->operand(branch + 1) != user) {
            continue;
          }
          for (int64_t i = 0; i < user->operand_count(); ++i) {
            if (user->operand(i) == value &&
                MayReadElement(reader->branch_computation(branch), i)) {
              (*branches)[branch] = true;
            }
          }
        }
      }
    }
  }
  return true;
}

// Whether `branch` of `conditional` is the only caller of its computation.
bool IsOnlyCaller(const HloInstruction* conditional, int64_t branch) {
  const HloComputation* computation = conditional->branch_computation(branch);
  return computation->caller_instructions().size() == 1 &&
         absl::c_count(conditional->branch_computations(), computation) == 1;
}

// Removes the outputs of `producer` that have no readers. A branch of
// `producer` returns an uninitialized buffer for each other output that is
// only read when `producer` takes another branch. Returns whether it changed
// anything.
bool ElideUnreadOutputs(HloInstruction* producer) {
  if (producer->IsRoot() || producer->has_sharding()) {
    return false;
  }
  const int64_t branch_count = producer->branch_count();
  for (int64_t branch = 0; branch < branch_count; ++branch) {
    // With tuples as results of its branches, `producer` returns a tuple.
    if (!IsOnlyCaller(producer, branch) ||
        producer->branch_computation(branch)->root_instruction()->opcode() !=
            HloOpcode::kTuple) {
      return false;
    }
  }
  const int64_t output_count = producer->shape().tuple_shapes().size();
  // Whether an output has a reader, and in which branches it may be read.
  std::vector<bool> has_reader(output_count, false);
  std::vector<std::vector<bool>> read_in_branch(
      output_count, std::vector<bool>(branch_count, false));
  for (const HloInstruction* user : producer->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    const int64_t index = user->tuple_index();
    has_reader[index] = true;
    if (!FindReadingBranches(producer, user, &read_in_branch[index])) {
      read_in_branch[index].assign(branch_count, true);
    }
  }
  const bool remove_outputs = absl::c_linear_search(has_reader, false);
  bool changed = false;
  for (int64_t branch = 0; branch < branch_count; ++branch) {
    HloComputation* computation = producer->branch_computation(branch);
    HloInstruction* root = computation->root_instruction();
    std::vector<HloInstruction*> elements;
    bool new_root = remove_outputs;
    for (int64_t index = 0; index < output_count; ++index) {
      if (!has_reader[index]) {
        continue;
      }
      HloInstruction* element = root->mutable_operand(index);
      const Shape& shape = element->shape();
      if (!read_in_branch[index][branch] && shape.IsArrayExcludingBuffer() &&
          shape.is_static() && !element->IsCustomCall(kAllocateBuffer)) {
        element = computation->AddInstruction(HloInstruction::CreateCustomCall(
            shape, /*operands=*/{}, kAllocateBuffer));
        new_root = true;
      }
      elements.push_back(element);
    }
    if (new_root) {
      // A new tuple: the old root may have other users in the branch.
      computation->set_root_instruction(
          computation->AddInstruction(HloInstruction::CreateTuple(elements)),
          /*accept_different_shape=*/true);
      changed = true;
    }
  }
  if (remove_outputs) {
    std::vector<int64_t> new_index(output_count, -1);
    std::vector<Shape> shapes;
    for (int64_t index = 0; index < output_count; ++index) {
      if (has_reader[index]) {
        new_index[index] = shapes.size();
        shapes.push_back(producer->shape().tuple_shapes(index));
      }
    }
    *producer->mutable_shape() = ShapeUtil::MakeTupleShape(shapes);
    // The origin of a synthetic call has no elements to renumber.
    if (producer->original_value() != nullptr &&
        !producer->original_value()->is_synthetic_call()) {
      CopyOriginalValue(producer, producer, new_index);
    }
    for (HloInstruction* user : producer->users()) {
      user->set_tuple_index(new_index[user->tuple_index()]);
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ConditionalResidualElision::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> conditionals;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kConditional &&
          absl::c_any_of(instruction->operand(0)->users(),
                         [&](const HloInstruction* user) {
                           return TakeSameBranch(user, instruction);
                         })) {
        conditionals.push_back(instruction);
      }
    }
    for (HloInstruction* consumer : conditionals) {
      ABSL_ASSIGN_OR_RETURN(bool forwarded, ForwardPassThroughs(consumer));
      changed |= forwarded;
    }
    for (HloInstruction* producer : conditionals) {
      changed |= ElideUnreadOutputs(producer);
    }
  }
  return changed;
}

}  // namespace xla
