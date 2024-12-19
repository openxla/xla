/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/all_gather_code_motion.h"

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {

bool HasMatchingReplicaGroups(const HloInstruction* instruction,
                              const std::vector<ReplicaGroup> replica_groups) {
  bool match = true;
  if (instruction->replica_groups().size() == replica_groups.size() &&
      instruction->replica_groups()[0].replica_ids().size() ==
          replica_groups[0].replica_ids().size()) {
    for (int i = 0; i < instruction->replica_groups().size(); i++) {
      for (int j = 0; j < instruction->replica_groups()[0].replica_ids().size();
           j++) {
        if (instruction->replica_groups()[i].replica_ids()[j] !=
            replica_groups[i].replica_ids()[j]) {
          match = false;
          break;
        }
      }
      if (!match) break;
    }
  } else {
    match = false;
  }
  return match;
}

AllGatherCodeMotion::AllGatherCodeMotion(
    std::vector<ReplicaGroup>* moveable_group)
    : moveable_group_(moveable_group) {}

absl::Status AllGatherCodeMotion::RewriteGetTupleElement(
    HloInstruction* while_op) {
  // Algorithm:
  // 1. For every get tuple element user of the while op
  // 2. If destination dtype is different than source dtype
  // 3. Create a new get tuple element with the original dtype
  // 4. Replace uses of original get tuple element with the new GTE
  // 5. Find users of the original GTE
  // 6. If the only user is a tuple - change the tuple shape to now have the
  // correct shape at that index

  for (auto* user : while_op->users()) {
    CHECK(user->opcode() == HloOpcode::kGetTupleElement)
        << "User is not a get-tuple-element. User " << user->ToString();

    auto* tuple_element = user;
    int64_t index = tuple_element->tuple_index();

    // Get the source and destination types
    const Shape& source_shape = while_op->shape().tuple_shapes(index);
    const Shape& dest_shape = tuple_element->shape();

    if (source_shape.element_type() != dest_shape.element_type()) {
      VLOG(2) << "Bad GTE " << tuple_element->ToString();
      // Create a new get tuple element with the original dtype
      HloInstruction* new_tuple_element = while_op->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(source_shape, while_op, index));
      VLOG(2) << "New GTE " << new_tuple_element->ToString();

      // Replace the uses of the original get tuple element with the convert
      // instruction
      TF_CHECK_OK(
          tuple_element->ReplaceAllUsesWithDifferentShape(new_tuple_element));

      // Change Tuple mutable shape at the corresponding index to be correct
      // with the new GTE
      for (auto* tuple_user : new_tuple_element->users()) {
        CHECK(tuple_user->opcode() == HloOpcode::kTuple)
            << "User is not a tuple. User: " << tuple_user->ToString();
        for (int64_t i = 0; i < tuple_user->operand_count(); ++i) {
          if (tuple_user->operand(i) == new_tuple_element) {
            *tuple_user->mutable_shape()->mutable_tuple_shapes(i) =
                source_shape;
            break;
          }
        }
      }
    }
  }
  return absl::OkStatus();
}
int64_t AllGatherCodeMotion::CountWhileLoops(HloModule* module) {
  int64_t num_while_loops = 0;
  // Count number of while loops to determine constraint.
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        num_while_loops++;
      }
    }
  }
  return num_while_loops;
}

absl::Status AllGatherCodeMotion::RewriteGetTupleElementUsers(
    HloInstruction* inst, HloInstruction* new_tuple) {
  const Shape& new_shape = new_tuple->operand(inst->tuple_index())->shape();
  if (!ShapeUtil::Equal(inst->shape(), new_shape)) {
    // Update the shape of the get-tuple-element instruction
    *inst->mutable_shape() = new_shape;

    // Update the tuple shape using ShapeUtil::UpdateTupleShape
    for (auto* user : inst->users()) {
      if (user->opcode() == HloOpcode::kTuple) {
        // Find the index of the get-tuple-element instruction in
        // the user tuple
        int64_t user_tuple_index = user->operand_index(inst);
        if (user_tuple_index >= 0) {
          Shape* tuple_shape = user->mutable_shape();
          ShapeUtil::UpdateTupleShape(new_shape, user_tuple_index, tuple_shape);
        }
      }
    }
    // Update the optimization barrier shape to match the updated
    // tuple shape
    for (auto* user : inst->users()) {
      for (auto* barrier_user : user->users()) {
        if (barrier_user->opcode() == HloOpcode::kOptimizationBarrier) {
          *barrier_user->mutable_shape() = user->shape();
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status AllGatherCodeMotion::RewriteTuple(
    HloInstruction* original_input_tuple, HloComputation* computation,
    HloInstruction* while_loop) {
  // Make a new tuple
  // Create a new tuple with the updated shapes
  std::vector<HloInstruction*> new_tuple_operands;
  for (int64_t i = 0; i < original_input_tuple->operand_count(); ++i) {
    new_tuple_operands.push_back(original_input_tuple->mutable_operand(i));
  }
  auto* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_operands));
  TF_RETURN_IF_ERROR(
      original_input_tuple->ReplaceAllUsesWithDifferentShape(new_tuple));
  // Replace the old tuple with the new tuple
  TF_RETURN_IF_ERROR(
      while_loop->ReplaceOperandWithDifferentShape(0, new_tuple));
  // Find the parameter tuple in the while loop body
  auto* while_body = while_loop->while_body();
  auto* param_tuple = while_body->parameter_instruction(0);
  if (param_tuple != nullptr) {
    // Modify the shape of the existing parameter tuple in place
    *param_tuple->mutable_shape() = new_tuple->shape();

    // Loop through the instructions in the while loop body
    for (auto* inst : while_body->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kGetTupleElement &&
          inst->operand(0) == param_tuple) {
        CHECK(RewriteGetTupleElementUsers(inst, new_tuple).ok());
      }
    }

    // Update the shape of the output tuple of the while loop body
    HloInstruction* root_tuple = while_body->root_instruction();
    if (root_tuple->opcode() == HloOpcode::kTuple) {
      std::vector<Shape> new_output_shapes;
      for (int64_t i = 0; i < root_tuple->operand_count(); ++i) {
        new_output_shapes.push_back(new_tuple->operand(i)->shape());
      }
      Shape new_output_tuple_shape =
          ShapeUtil::MakeTupleShape(new_output_shapes);
      *root_tuple->mutable_shape() = new_output_tuple_shape;
    }
  }
  // Update the parameter tuple shape in the while loop condition
  auto* while_condition = while_loop->while_condition();
  param_tuple = nullptr;
  param_tuple = while_condition->parameter_instruction(0);

  if (param_tuple != nullptr) {
    *param_tuple->mutable_shape() = new_tuple->shape();
  }
  // Update the output tuple shape of the while loop
  *while_loop->mutable_shape() = new_tuple->shape();
  return absl::OkStatus();
}

absl::Status AllGatherCodeMotion::RewriteConvert(
    HloInstruction* convert_element, HloInstruction* tuple_element_in_body) {
  const Shape& new_shape = tuple_element_in_body->shape();
  // Update the tuple shape using ShapeUtil::UpdateTupleShape
  for (auto* user : convert_element->users()) {
    if (user->opcode() == HloOpcode::kTuple) {
      // Find the index of the get-tuple-element instruction in
      // the user tuple
      int64_t user_tuple_index = user->operand_index(convert_element);
      if (user_tuple_index >= 0) {
        Shape* tuple_shape = user->mutable_shape();
        ShapeUtil::UpdateTupleShape(new_shape, user_tuple_index, tuple_shape);
      }
    }
  }
  // Update the optimization barrier shape to match the updated
  // tuple shape
  for (auto* user : convert_element->users()) {
    for (auto* barrier_user : user->users()) {
      if (barrier_user->opcode() == HloOpcode::kOptimizationBarrier) {
        *barrier_user->mutable_shape() = user->shape();
      }
    }
  }
  TF_RETURN_IF_ERROR(
      convert_element->ReplaceAllUsesWithDifferentShape(tuple_element_in_body));
  return absl::OkStatus();
}

bool AllGatherCodeMotion::MaybeSkipCodeMotion(int64_t num_while_loops,
                                              HloInstruction* convert_element) {
  if (num_while_loops != 1) {
    // Repeated Transformer logic
    // TODO: This moves all-gather out of inner while loop.
    // If you want to move out of outer while loop.
    // Change logic to run else logic below for outer while loop.
    for (auto* user : convert_element->users()) {
      if (user->opcode() != HloOpcode::kTuple) {
        VLOG(2) << convert_element->ToString();
        VLOG(2) << "Not removing this convert: user: " << user->opcode();

        return true;
      }
      for (auto* tuple_user : user->users()) {
        if (tuple_user->opcode() != HloOpcode::kOptimizationBarrier) {
          VLOG(2) << "Not removing this tuple: user: "
                  << tuple_user->ToString();
          return true;
        }
      }
    }
  } else {
    // Single while loop logic
    for (auto* user : convert_element->users()) {
      // In single while loop flow, opt-barrier(tuple(convert()))
      // is allowed
      if (user->opcode() == HloOpcode::kTuple) {
        for (auto* tuple_user : user->users()) {
          if (tuple_user->opcode() != HloOpcode::kOptimizationBarrier) {
            VLOG(2) << "Not removing this convert: user: "
                    << tuple_user->ToString();
            return true;
          }
        }
      } else if (user->opcode() != HloOpcode::kAllGather) {
        VLOG(2) << "Not removing this convert: user: " << user->ToString();
        return true;
      }
    }
  }
  return false;
}
absl::StatusOr<HloInstruction*> AllGatherCodeMotion::CreateAndReplaceAllGather(
    HloComputation* computation, HloComputation* while_body,
    HloInstruction* original_input_tuple, HloInstruction* while_loop,
    HloInstruction* operand, HloInstruction* all_gather, int64_t tuple_index) {
  // Perform the convert (if applicable) and all-gather on the tuple
  // element
  auto* original_tuple_element =
      original_input_tuple->mutable_operand(tuple_index);
  HloInstruction* new_operand = nullptr;
  if (operand->opcode() == HloOpcode::kConvert) {
    new_operand = computation->AddInstruction(operand->CloneWithNewOperands(
        operand->shape(), {original_tuple_element}));
  } else {
    new_operand = original_tuple_element;
  }
  auto* new_all_gather = computation->AddInstruction(
      all_gather->CloneWithNewOperands(all_gather->shape(), {new_operand}));

  VLOG(2) << "Moved all-gather (and convert if applicable) out of "
             "while loop: "
          << new_all_gather->name();

  // Replace the tuple element with the new all-gathered operand
  TF_RETURN_IF_ERROR(original_input_tuple->ReplaceOperandWithDifferentShape(
      tuple_index, new_all_gather));

  VLOG(2) << "Replaced Operand with Different Shape";

  // Replace uses of the all-gather with the tuple element in the
  // while loop body
  auto* parameter = while_body->parameter_instruction(0);
  auto* tuple_element_in_body =
      while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_all_gather->shape(), parameter, tuple_index));

  TF_RETURN_IF_ERROR(
      all_gather->ReplaceAllUsesWith(tuple_element_in_body));  // Same shape

  VLOG(2) << "Replaced all uses with different shape";
  return tuple_element_in_body;
}

absl::StatusOr<bool> AllGatherCodeMotion::TransformWhileLoop(
    HloInstruction* instruction, int64_t num_while_loops, bool rewrite_tuple,
    bool rewrite_convert) {
  bool changed = false;
  HloComputation* computation = instruction->parent();
  auto* while_loop = instruction;
  auto* while_body = while_loop->while_body();

  VLOG(2) << "Found while loop: " << while_loop->name();

  HloInstruction* param_inst = while_body->parameter_instruction(0);

  std::vector<HloInstruction*> all_gathers;
  for (auto* inst : while_body->instructions()) {
    if (inst->opcode() == HloOpcode::kAllGather) {
      // Strictly follow moveable group if given
      if (moveable_group_ != nullptr &&
          HasMatchingReplicaGroups(inst, *moveable_group_)) {
        continue;
      } else {
        VLOG(2) << "Found moveable AG " << inst->ToString();
      }
      all_gathers.push_back(inst);
    }
  }

  VLOG(2) << "Number of all-gather operations found: " << all_gathers.size();

  HloInstruction* original_input_tuple = while_loop->mutable_operand(0);

  for (auto* all_gather : all_gathers) {
    HloInstruction* operand = all_gather->mutable_operand(0);
    HloInstruction* tuple_element = nullptr;
    HloInstruction* convert_element = nullptr;
    int64_t tuple_index = -1;

    if (operand->opcode() == HloOpcode::kGetTupleElement &&
        operand->operand(0)->opcode() == HloOpcode::kParameter) {
      tuple_element = operand;
      tuple_index = tuple_element->tuple_index();
      VLOG(2) << "All-gather on tuple element: " << tuple_index;

    } else if (operand->opcode() == HloOpcode::kConvert &&
               operand->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
               operand->operand(0)->operand(0)->opcode() ==
                   HloOpcode::kParameter) {
      tuple_element = operand->mutable_operand(0);
      tuple_index = tuple_element->tuple_index();
      convert_element = operand;
      rewrite_convert = true;
      VLOG(2) << "All-gather on convert(tuple element): " << tuple_index;
    }

    if (tuple_element != nullptr && tuple_index >= 0) {
      bool modify = false;
      for (auto* user : tuple_element->users()) {
        if (user->opcode() != HloOpcode::kAllGather &&
            user->opcode() != HloOpcode::kTuple &&
            user->opcode() != HloOpcode::kConvert) {
          VLOG(2) << all_gather->ToString();
          VLOG(2) << "Not removing this all gather: tuple user: "
                  << user->opcode();

          modify = true;
          break;
        }
      }
      if (convert_element != nullptr) {
        if (MaybeSkipCodeMotion(num_while_loops, convert_element)) {
          // We only support users of get tuple element as AllGather,
          // convert and output root tuple
          continue;
        }
      }
      absl::StatusOr<HloInstruction*> status_or_tuple_element_in_body =
          CreateAndReplaceAllGather(computation, while_body,
                                    original_input_tuple, while_loop, operand,
                                    all_gather, tuple_index);
      CHECK(status_or_tuple_element_in_body.ok());
      HloInstruction* tuple_element_in_body = *status_or_tuple_element_in_body;

      if (rewrite_convert == true) {
        CHECK(RewriteConvert(convert_element, tuple_element_in_body).ok());
      }

      changed = true;
      rewrite_tuple = true;
    }
  }

  if (rewrite_tuple == true) {
    CHECK(RewriteTuple(original_input_tuple, computation, while_loop).ok());
  }
  return changed;
}

absl::StatusOr<bool> AllGatherCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  const int64_t num_while_loops = CountWhileLoops(module);

  for (auto* computation : module->computations(execution_threads)) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        absl::StatusOr<bool> result =
            TransformWhileLoop(instruction, num_while_loops, false, false);
        CHECK(result.ok());
        changed |= *result;
      }
    }
  }
  // Clean up any get tuple elements.
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        CHECK(RewriteGetTupleElement(instruction).ok());
      }
    }
  }
  return changed;
}

}  // namespace xla