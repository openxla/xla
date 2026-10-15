/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

// Relays control dependencies safely when an instruction is removed and
// replaced by an earlier producer (replacement):
// - Control predecessors of `instruction` must complete before any of
//   `instruction`'s data users execute. In a scheduled module, predecessors
//   were scheduled before `instruction`, and `instruction` was scheduled before
//   its users. Adding control dependencies from predecessors to users preserves
//   topological order and the existing schedule. If `instruction` has no users,
//   transfer its control predecessors to its control successors.
// - Control successors of `instruction` must execute after `replacement`
//   (which was computed before `instruction`). Adding control dependencies
//   from `replacement` to successors preserves topological order and schedule.
// Control predecessors of `instruction` must NOT be copied to `replacement`,
// because `replacement` was already scheduled before `instruction`, and could
// be scheduled before `instruction`'s control predecessors, which would invert
// the schedule order or create self-cycles.
absl::Status RelayControlDependenciesSafely(HloInstruction* instruction,
                                            HloInstruction* replacement) {
  for (HloInstruction* pred : instruction->control_predecessors()) {
    if (!instruction->users().empty()) {
      for (HloInstruction* user : instruction->users()) {
        if (pred != user) {
          ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(user));
        }
      }
    } else {
      for (HloInstruction* succ : instruction->control_successors()) {
        if (pred != succ) {
          ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(succ));
        }
      }
    }
  }
  for (HloInstruction* succ : instruction->control_successors()) {
    if (replacement != succ) {
      ABSL_RETURN_IF_ERROR(replacement->AddControlDependencyTo(succ));
    }
  }
  return instruction->DropAllControlDeps();
}

}  // namespace

TupleSimplifier::TupleSimplifier(bool exclude_entry_computation)
    : exclude_entry_computation_(exclude_entry_computation) {}

absl::StatusOr<HloInstruction*> TupleSimplifier::RemoveWholeTuple(
    HloInstruction* tuple) {
  HloInstruction* top_tuple = nullptr;
  for (int64_t operand_number = 0; operand_number < tuple->operand_count();
       ++operand_number) {
    HloInstruction* operand = tuple->mutable_operand(operand_number);
    if (operand->opcode() != HloOpcode::kGetTupleElement ||
        operand->tuple_index() != operand_number) {
      return nullptr;
    }
    if (top_tuple == nullptr) {
      top_tuple = operand->mutable_operand(0);
      if (!ShapeUtil::Compatible(top_tuple->shape(), tuple->shape())) {
        return nullptr;
      }
    } else if (top_tuple != operand->operand(0)) {
      return nullptr;
    }
  }
  if (top_tuple == nullptr) {
    return nullptr;
  }
  ABSL_RETURN_IF_ERROR(RelayControlDependenciesSafely(tuple, top_tuple));
  ABSL_ASSIGN_OR_RETURN(bool changed,
                        tuple->parent()->ReplaceInstruction(
                            tuple, top_tuple, /*preserve_sharding=*/true,
                            /*relay_control_dependency=*/false));
  if (changed) {
    return top_tuple;
  }
  return nullptr;
}

absl::StatusOr<bool> TupleSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Initially add all GTE and Tuple instructions to the worklist.
  bool changed = false;
  for (auto* computation : module->computations(execution_threads)) {
    if (exclude_entry_computation_ &&
        computation == module->entry_computation()) {
      continue;
    }
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kTuple) {
        ABSL_ASSIGN_OR_RETURN(HloInstruction * instr,
                              RemoveWholeTuple(instruction));
        if (instr != nullptr) {
          changed = true;
        }
      } else {
        auto [ancestor, index] = instruction->LatestNonGteAncestorAndIndex();
        if (ancestor == instruction) {
          continue;
        }
        // If possible replace a chain of GTE with the operation which produces
        // the element. For example, replace uses of GTE with below with just
        // 'Op' (assuming 'Op' is at the index of the GTE instruction):
        //
        //     ...  Op ...
        //       \  |   /
        //        Tuple
        //          |
        //         GTE
        //         ...
        //          |
        //         GTE
        //          |
        //         GTE
        //
        // Note that this deletes the Tuple instruction altogether. In addition,
        // if only a subset of tuple's elements are used, this transform
        // optimizes them one at a time, and after the last use is optimized,
        // the Tuple will also be deleted.
        HloInstruction* replacement = ancestor;
        for (int i = 0; i < index.size(); ++i) {
          if (replacement->opcode() != HloOpcode::kTuple) {
            replacement = nullptr;
            break;
          }
          replacement = replacement->mutable_operand(index[i]);
        }

        if (replacement) {
          ABSL_RETURN_IF_ERROR(
              RelayControlDependenciesSafely(instruction, replacement));
          ABSL_ASSIGN_OR_RETURN(bool replaced,
                                computation->ReplaceInstruction(
                                    instruction, replacement,
                                    /*preserve_sharding=*/true,
                                    /*relay_control_dependency=*/false));
          changed |= replaced;
        }
      }
    }
  }

  if (changed && module->has_schedule()) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update(execution_threads));
  }

  return changed;
}

}  // namespace xla
