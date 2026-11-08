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

#include "xla/service/host_offload_utils.h"

#include <array>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/call_graph.h"
#include "xla/service/computation_layout.h"
#include "xla/service/memory_annotations.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"

namespace xla {
namespace host_offload_utils {

namespace {

using ::xla::memory_annotations::kMoveToDeviceCustomCallTarget;
using ::xla::memory_annotations::kMoveToHostCustomCallTarget;

std::optional<ShapeIndex> GetAliasedOutputShapeIndex(
    const HloInstruction* custom_call, int64_t operand_index,
    const ShapeIndex& operand_shape_index = {}) {
  if (custom_call->custom_call_target() == kMoveToDeviceCustomCallTarget ||
      custom_call->custom_call_target() == kMoveToHostCustomCallTarget) {
    // Does not define a new buffer.
    return ShapeIndex{};
  }
  // Check the custom call's output_to_operand_aliasing.
  for (const auto& alias : custom_call->output_operand_aliasing()) {
    if (alias.second.first == operand_index &&
        alias.second.second == operand_shape_index) {
      return alias.first;
    }
  }
  // By default, assume custom calls define new buffers.
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::vector<InstructionAndShapeIndex>> GetSuccessors(
    const InstructionAndShapeIndex& instruction_and_shape_index,
    const CallGraph& call_graph) {
  std::vector<InstructionAndShapeIndex> result;
  HloInstruction* instruction = instruction_and_shape_index.instruction;
  if (instruction->IsRoot()) {
    // Successor of the root is the call instruction(s).
    const std::vector<HloInstruction*> callers =
        call_graph.GetComputationCallers(instruction->parent());
    for (HloInstruction* caller : callers) {
      result.push_back({caller, instruction_and_shape_index.shape_index});
    }
  }
  for (HloInstruction* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kTuple ||
        (user->opcode() == HloOpcode::kSort && user->shape().IsTuple())) {
      auto operand_indices = user->OperandIndices(instruction);
      for (const auto i : operand_indices) {
        auto tmp_shape_index = instruction_and_shape_index.shape_index;
        tmp_shape_index.push_front(i);
        result.push_back({user, std::move(tmp_shape_index)});
      }
    } else if (user->opcode() == HloOpcode::kGetTupleElement) {
      ShapeIndex tmp_shape_index = instruction_and_shape_index.shape_index;
      if (tmp_shape_index.empty()) {
        // The instruction itself produces the tuple (e.g. a variadic reduce),
        // so the whole tuple is on host and every element read from it is a
        // successor.
        result.push_back({user, std::move(tmp_shape_index)});
      } else if (tmp_shape_index.front() == user->tuple_index()) {
        // This GTE is for the buffer we're tracking.
        tmp_shape_index.pop_front();
        result.push_back({user, std::move(tmp_shape_index)});
      }
    } else if (user->opcode() == HloOpcode::kCall) {
      auto operand_indices = user->OperandIndices(instruction);
      CHECK(user->called_computations().size() == 1)
          << "Expect call to only have one called computation.";
      for (const auto i : operand_indices) {
        HloComputation* called_computation =
            user->called_computations().front();
        HloInstruction* parameter_instruction =
            called_computation->parameter_instruction(i);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kWhile) {
      auto operand_indices = user->OperandIndices(instruction);
      HloComputation* while_body_computation = user->while_body();
      HloComputation* while_condition_computation = user->while_condition();
      for (const auto i : operand_indices) {
        HloInstruction* parameter_instruction =
            while_body_computation->parameter_instruction(i);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});

        HloInstruction* condition_instruction =
            while_condition_computation->parameter_instruction(i);
        result.push_back(
            {condition_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kConditional) {
      auto operand_indices = user->OperandIndices(instruction);
      for (const int64_t index : operand_indices) {
        CHECK(user->branch_computations().size() >= index)
            << "Not enough branch computations for conditional";
        HloComputation* branch_computation =
            user->branch_computation(index - 1);
        HloInstruction* parameter_instruction =
            branch_computation->parameter_instruction(0);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kAsyncStart) {
      auto operand_indices = user->OperandIndices(instruction);
      CHECK(user->called_computations().size() == 1)
          << "Expect async-start to only have one called computation.";
      for (const auto i : operand_indices) {
        HloComputation* called_computation =
            user->called_computations().front();
        HloInstruction* parameter_instruction =
            called_computation->parameter_instruction(i);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kCustomCall) {
      const auto operand_indices = user->OperandIndices(instruction);
      absl::flat_hash_set<ShapeIndex> seen_output_indices;
      for (const auto i : operand_indices) {
        if (std::optional<ShapeIndex> output_shape_index =
                GetAliasedOutputShapeIndex(
                    user, i, instruction_and_shape_index.shape_index)) {
          if (!seen_output_indices.insert(*output_shape_index).second) {
            return absl::InternalError(
                "Found multiple operands of a custom call that reuse the same "
                "output buffer.");
          }
          result.push_back({user, *std::move(output_shape_index)});
        }
      }
    } else {
      result.push_back({user, instruction_and_shape_index.shape_index});
    }
  }
  return result;
}

std::vector<InstructionAndShapeIndex> GetPredecessors(
    const InstructionAndShapeIndex& instruction_and_shape_index,
    const CallGraph& call_graph, std::optional<int64_t> operand_index) {
  std::vector<InstructionAndShapeIndex> result;
  HloInstruction* instruction = instruction_and_shape_index.instruction;
  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    const int64_t index = instruction->tuple_index();
    auto tmp_shape_index = instruction_and_shape_index.shape_index;
    tmp_shape_index.push_front(index);
    result.push_back({instruction->mutable_operand(0), tmp_shape_index});
  } else if (instruction->opcode() == HloOpcode::kTuple ||
             (instruction->opcode() == HloOpcode::kSort &&
              instruction->shape().IsTuple())) {
    CHECK(!instruction_and_shape_index.shape_index.empty())
        << "Did not store an index before encountering a tuple.";
    auto tmp_shape_index = instruction_and_shape_index.shape_index;
    const int64_t index = tmp_shape_index.front();
    tmp_shape_index.pop_front();
    result.push_back({instruction->mutable_operand(index), tmp_shape_index});
  } else if (instruction->opcode() == HloOpcode::kCall) {
    // Predecessor of a call is its computation's root instruction.
    CHECK(instruction->called_computations().size() == 1)
        << "Expect call to only have one called computation.";
    HloComputation* called_computation =
        instruction->called_computations().front();
    result.push_back({called_computation->root_instruction(),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kParameter) {
    const std::vector<HloInstruction*> callers =
        call_graph.GetComputationCallers(instruction->parent());
    absl::flat_hash_set<HloInstruction*> unique_callers(callers.begin(),
                                                        callers.end());
    for (HloInstruction* caller : unique_callers) {
      if (caller->opcode() == HloOpcode::kConditional) {
        bool found_computation = false;
        for (int64_t i = 0; i < caller->branch_computations().size(); ++i) {
          if (caller->branch_computation(i) == instruction->parent()) {
            found_computation = true;
            // Operand 0 is the predicate/index, so the input to the i-th branch
            // computation is operand (i + 1).
            result.push_back({caller->mutable_operand(i + 1),
                              instruction_and_shape_index.shape_index});
          }
        }
        CHECK(found_computation) << "Computation not found in conditional";
      } else {
        result.push_back(
            {caller->mutable_operand(instruction->parameter_number()),
             instruction_and_shape_index.shape_index});
      }
    }
  } else if (instruction->opcode() == HloOpcode::kDynamicSlice) {
    result.push_back({instruction->mutable_operand(operand_index.value_or(0)),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
    // Ensure that we are only following the base data operand (index 0).
    CHECK(!operand_index.has_value() || *operand_index == 0);
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body_computation = instruction->while_body();
    result.push_back({while_body_computation->root_instruction(),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kPad) {
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kSend) {
    // Explicitly handle Send, which has 2 operands (data, token).
    // We follow the data path (operand 0).
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kConditional) {
    for (HloComputation* computation : instruction->called_computations()) {
      result.push_back({computation->root_instruction(),
                        instruction_and_shape_index.shape_index});
    }
  } else if (instruction->opcode() == HloOpcode::kCustomCall &&
             !operand_index.has_value() &&
             !instruction->output_operand_aliasing().empty()) {
    for (const auto& alias : instruction->output_operand_aliasing()) {
      if (alias.first == instruction_and_shape_index.shape_index) {
        result.push_back({instruction->mutable_operand(alias.second.first),
                          alias.second.second});
      }
    }
  } else {
    if (!operand_index.has_value()) {
      CHECK_EQ(instruction->operands().size(), 1)
          << "Expected instruction to have 1 operand. Found: "
          << instruction->ToString();
    }
    result.push_back({instruction->mutable_operand(operand_index.value_or(0)),
                      instruction_and_shape_index.shape_index});
  }
  return result;
}

bool IsValidDuringPureMemoryOffload(const HloInstruction* instruction) {
  static constexpr std::array allowed_opcodes = {
      HloOpcode::kGetTupleElement,
      HloOpcode::kBitcast,
      HloOpcode::kTuple,
      HloOpcode::kCall,
      HloOpcode::kWhile,
      HloOpcode::kConditional,
      HloOpcode::kParameter,
      HloOpcode::kOptimizationBarrier,
      HloOpcode::kAsyncStart,
      HloOpcode::kAsyncDone,
      HloOpcode::kCustomCall,
      HloOpcode::kSend,
      HloOpcode::kRecv,
      HloOpcode::kSendDone,
      HloOpcode::kRecvDone};
  return absl::c_linear_search(allowed_opcodes, instruction->opcode());
}

bool operator==(const InstructionAndShapeIndex& lhs,
                const InstructionAndShapeIndex& rhs) {
  return lhs.instruction == rhs.instruction &&
         lhs.shape_index == rhs.shape_index;
}

std::string InstructionAndShapeIndex::ToString() const {
  return absl::StrFormat("{Instr: %s, ShapeIndex: %s}", instruction->name(),
                         shape_index.ToString());
}

bool IsHostAsyncStart(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kAsyncStart &&
         instruction->async_execution_thread() == HloInstruction::kHostThread;
}

bool IsSynchronousCopyFromOrToHost(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCopy) {
    return false;
  }
  if (!instruction->shape().has_layout() ||
      !instruction->operand(0)->shape().has_layout()) {
    // Host offloading copies do not exist without layouts.
    return false;
  }
  const int64_t copy_memory_space =
      instruction->shape().layout().memory_space();
  const int64_t operand_memory_space =
      instruction->operand(0)->shape().layout().memory_space();
  return (copy_memory_space == Layout::kHostMemorySpace &&
          operand_memory_space != Layout::kHostMemorySpace) ||
         (copy_memory_space != Layout::kHostMemorySpace &&
          operand_memory_space == Layout::kHostMemorySpace);
}

bool ComputeTypeIsHost(const HloInstruction* hlo_instruction) {
  const auto& frontend_attributes_map =
      hlo_instruction->frontend_attributes().map();
  return (frontend_attributes_map.find(kXlaComputeTypeAttr) !=
              frontend_attributes_map.end() &&
          frontend_attributes_map.find(kXlaComputeTypeAttr)->second ==
              kXlaComputeTypeHost);
}

void SetHostComputeFrontendAttribute(HloInstruction& host_instruction) {
  FrontendAttributes frontend_attributes =
      host_instruction.frontend_attributes();
  frontend_attributes.mutable_map()->insert(
      {kXlaComputeTypeAttr, kXlaComputeTypeHost});
  host_instruction.set_frontend_attributes(frontend_attributes);
}

bool IsMoveToHostWithDynamicUpdateSlice(const HloInstruction* instr) {
  if (!instr->IsCustomCall(kMoveToHostCustomCallTarget)) {
    return false;
  }

  std::vector<const HloInstruction*> to_check = {instr};
  absl::flat_hash_set<const HloInstruction*> visited;

  while (!to_check.empty()) {
    const HloInstruction* current = to_check.back();
    to_check.pop_back();

    auto [_, inserted] = visited.insert(current);
    if (!inserted) {
      continue;
    }
    for (const HloInstruction* user : current->users()) {
      if (user->opcode() == HloOpcode::kDynamicUpdateSlice) {
        return true;
      }
      if (HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kBroadcast,
                           HloOpcode::kTranspose>(user)) {
        to_check.push_back(user);
      }
    }
  }
  return false;
}

bool IsMoveToDeviceWithDynamicSlice(const HloInstruction* instr) {
  if (!instr->IsCustomCall(kMoveToDeviceCustomCallTarget)) {
    return false;
  }

  std::vector<const HloInstruction*> to_check = {instr};
  absl::flat_hash_set<const HloInstruction*> visited;

  while (!to_check.empty()) {
    const HloInstruction* current = to_check.back();
    to_check.pop_back();

    auto [_, inserted] = visited.insert(current);
    if (!inserted) {
      continue;
    }
    for (const HloInstruction* operand : current->operands()) {
      if (operand->opcode() == HloOpcode::kDynamicSlice) {
        return true;
      }
      if (HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kBroadcast,
                           HloOpcode::kTranspose>(operand)) {
        to_check.push_back(operand);
      }
    }
  }
  return false;
}

namespace {

// Recursively finds GTE indices used in DS/DUS index operands.
absl::flat_hash_set<int64_t> FindTupleIndicesInOperand(
    const HloInstruction* operand) {
  absl::flat_hash_set<int64_t> indices;

  if (operand->opcode() == HloOpcode::kGetTupleElement) {
    indices.insert(operand->tuple_index());
  } else if (operand->opcode() == HloOpcode::kCopy &&
             operand->operand_count() == 1) {
    auto copy_indices = FindTupleIndicesInOperand(operand->operand(0));
    indices.insert(copy_indices.begin(), copy_indices.end());
  } else if (operand->opcode() == HloOpcode::kAdd ||
             operand->opcode() == HloOpcode::kSubtract ||
             operand->opcode() == HloOpcode::kMultiply ||
             operand->opcode() == HloOpcode::kDivide) {
    for (int i = 0; i < operand->operand_count(); ++i) {
      auto op_indices = FindTupleIndicesInOperand(operand->operand(i));
      indices.insert(op_indices.begin(), op_indices.end());
    }
  }

  return indices;
}

}  // namespace

absl::flat_hash_set<int64_t> CollectDynamicVariableTupleIndices(
    const HloInstruction* while_loop) {
  absl::flat_hash_set<int64_t> tuple_indices;
  if (while_loop->opcode() != HloOpcode::kWhile || !while_loop->while_body()) {
    return tuple_indices;
  }

  bool has_host_offloading = false;
  for (const HloInstruction* instr : while_loop->while_body()->instructions()) {
    if (IsMoveToHostWithDynamicUpdateSlice(instr) ||
        IsMoveToDeviceWithDynamicSlice(instr)) {
      has_host_offloading = true;
      break;
    }
  }
  if (!has_host_offloading) {
    return tuple_indices;
  }

  for (const HloInstruction* instr : while_loop->while_body()->instructions()) {
    if (instr->opcode() != HloOpcode::kDynamicUpdateSlice &&
        instr->opcode() != HloOpcode::kDynamicSlice) {
      continue;
    }
    int first_index_operand =
        (instr->opcode() == HloOpcode::kDynamicUpdateSlice)
            ? Cast<HloDynamicUpdateSliceInstruction>(instr)
                  ->first_index_operand_number()
            : Cast<HloDynamicSliceInstruction>(instr)
                  ->first_index_operand_number();
    for (int i = first_index_operand; i < instr->operand_count(); ++i) {
      auto op_indices = FindTupleIndicesInOperand(instr->operand(i));
      tuple_indices.insert(op_indices.begin(), op_indices.end());
    }
  }
  return tuple_indices;
}

namespace {

// Returns true if `instruction` passes a buffer through unchanged on the main
// execution thread, so a value that is about to be written into host memory may
// cross it while still in device memory.
bool ForwardsBufferUnchanged(const HloInstruction* instruction,
                             const CallGraph& call_graph) {
  switch (instruction->opcode()) {
    case HloOpcode::kReshape:
      return ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                         instruction->shape());
    case HloOpcode::kParameter: {
      if (instruction->parent()->IsEntryComputation()) {
        return false;
      }
      const std::vector<HloInstruction*> callers =
          call_graph.GetComputationCallers(instruction->parent());
      return !callers.empty() &&
             absl::c_all_of(
                 callers,
                 HloPredicateIsOp<HloOpcode::kCall, HloOpcode::kWhile>);
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kCall:
    case HloOpcode::kCopy:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
      return true;
    default:
      return false;
  }
}

// Returns true if the buffer operand of `dynamic_update_slice` is created by a
// broadcast, an AllocateBuffer or a host memory entry parameter, which is what
// HostOffloader::CreateAllocateBufferForDynamicUpdateSlice can place in host
// memory.
bool DynamicUpdateSliceBufferIsAllocated(HloInstruction* dynamic_update_slice,
                                         const CallGraph& call_graph) {
  std::queue<InstructionAndShapeIndex> queue;
  absl::flat_hash_set<InstructionAndShapeIndex> visited;
  queue.push(InstructionAndShapeIndex(dynamic_update_slice));
  bool found_allocation = false;
  while (!queue.empty()) {
    const InstructionAndShapeIndex current = queue.front();
    queue.pop();
    for (const InstructionAndShapeIndex& predecessor :
         GetPredecessors(current, call_graph)) {
      HloInstruction* instruction = predecessor.instruction;
      if (instruction->opcode() == HloOpcode::kBroadcast ||
          instruction->IsCustomCall("AllocateBuffer")) {
        found_allocation = true;
        continue;
      }
      if (instruction->opcode() == HloOpcode::kParameter &&
          instruction->parent()->IsEntryComputation()) {
        const Shape& parameter_shape = ShapeUtil::GetSubshape(
            instruction->GetModule()
                ->entry_computation_layout()
                .parameter_layout(instruction->parameter_number())
                .shape(),
            predecessor.shape_index);
        if (!parameter_shape.has_layout() ||
            parameter_shape.layout().memory_space() !=
                Layout::kHostMemorySpace) {
          return false;
        }
        found_allocation = true;
        continue;
      }
      if (instruction->opcode() == HloOpcode::kParameter) {
        const std::vector<HloInstruction*> callers =
            call_graph.GetComputationCallers(instruction->parent());
        if (callers.empty() ||
            !absl::c_all_of(callers, [&](const HloInstruction* caller) {
              return caller->opcode() == HloOpcode::kCall ||
                     (caller->opcode() == HloOpcode::kWhile &&
                      caller->while_body() == instruction->parent());
            })) {
          return false;
        }
      } else if (!HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kCall,
                                   HloOpcode::kCopy,
                                   HloOpcode::kDynamicUpdateSlice,
                                   HloOpcode::kGetTupleElement,
                                   HloOpcode::kOptimizationBarrier,
                                   HloOpcode::kTuple, HloOpcode::kWhile>(
                     instruction)) {
        return false;
      }
      if (visited.insert(predecessor).second) {
        queue.push(predecessor);
      }
    }
  }
  return found_allocation;
}

bool InstructionIsAllowedBetweenDsAndMoveToDevice(
    const HloInstruction* instruction, const CallGraph& call_graph) {
  if (instruction->opcode() == HloOpcode::kReduce) {
    // A variadic reduce has a tuple shape; the verifier guarantees all its
    // elements have the same dimensions, so the first one is representative.
    const Shape& output_shape = instruction->shape().IsTuple()
                                    ? instruction->shape().tuple_shapes(0)
                                    : instruction->shape();
    return ShapeUtil::TrueNumDimensions(instruction->operand(0)->shape()) ==
           ShapeUtil::TrueNumDimensions(output_shape);
  }
  return ForwardsBufferUnchanged(instruction, call_graph) &&
         instruction->opcode() != HloOpcode::kCall;
}

}  // namespace

bool HoldsOnlyDataMovedToHost(
    const InstructionAndShapeIndex& value, const CallGraph& call_graph,
    const MovedToHostOperands& moved_to_host_operands) {
  std::queue<InstructionAndShapeIndex> queue;
  absl::flat_hash_set<InstructionAndShapeIndex> visited;
  auto push = [&](const InstructionAndShapeIndex& next) {
    if (visited.insert(next).second) {
      queue.push(next);
    }
  };
  push(value);
  while (!queue.empty()) {
    const InstructionAndShapeIndex current = queue.front();
    queue.pop();
    HloInstruction* instruction = current.instruction;
    if (instruction->IsCustomCall(
            memory_annotations::kMoveToHostCustomCallTarget) ||
        instruction->IsCustomCall("AllocateBuffer") ||
        (instruction->opcode() == HloOpcode::kBroadcast &&
         instruction->operand(0)->opcode() == HloOpcode::kConstant)) {
      continue;
    }
    if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      if (!moved_to_host_operands.contains({instruction, 0})) {
        push(InstructionAndShapeIndex(instruction->mutable_operand(0)));
      }
      if (!moved_to_host_operands.contains({instruction, 1})) {
        push(InstructionAndShapeIndex(instruction->mutable_operand(1)));
      }
      continue;
    }
    if (!ForwardsBufferUnchanged(instruction, call_graph)) {
      return false;
    }
    if (HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kCopy,
                         HloOpcode::kOptimizationBarrier, HloOpcode::kReshape>(
            instruction) &&
        current.shape_index.empty() &&
        moved_to_host_operands.contains({instruction, 0})) {
      continue;
    }
    if (instruction->opcode() == HloOpcode::kTuple &&
        current.shape_index.size() == 1 &&
        moved_to_host_operands.contains(
            {instruction, current.shape_index.front()})) {
      continue;
    }
    if (instruction->opcode() == HloOpcode::kParameter) {
      const int64_t param_number = instruction->parameter_number();
      for (HloInstruction* caller :
           call_graph.GetComputationCallers(instruction->parent())) {
        if (caller->opcode() == HloOpcode::kWhile &&
            caller->while_body() == instruction->parent()) {
          push(InstructionAndShapeIndex(
              instruction->parent()->root_instruction(), current.shape_index));
        }
        if (!current.shape_index.empty() ||
            !moved_to_host_operands.contains({caller, param_number})) {
          push(InstructionAndShapeIndex(caller->mutable_operand(param_number),
                                        current.shape_index));
        }
      }
      continue;
    }
    if (instruction->opcode() == HloOpcode::kWhile &&
        (!current.shape_index.empty() ||
         !moved_to_host_operands.contains({instruction, 0}))) {
      // A loop that runs zero times returns its initial value.
      push(InstructionAndShapeIndex(instruction->mutable_operand(0),
                                    current.shape_index));
    }
    for (const InstructionAndShapeIndex& predecessor :
         GetPredecessors(current, call_graph)) {
      push(predecessor);
    }
  }
  return true;
}

absl::StatusOr<std::optional<std::vector<InstructionAndShapeIndex>>>
FindDynamicUpdateSlicesReceivingUpdate(const InstructionAndShapeIndex& start,
                                       const CallGraph& call_graph,
                                       bool through_in_place_updates) {
  std::vector<InstructionAndShapeIndex> result;
  std::queue<InstructionAndShapeIndex> queue;
  absl::flat_hash_set<InstructionAndShapeIndex> visited;
  queue.push(start);
  visited.insert(start);
  while (!queue.empty()) {
    const InstructionAndShapeIndex current = queue.front();
    queue.pop();
    // GetSuccessors skips custom calls which do not alias the value, but such
    // a custom call still reads it on device.
    if (absl::c_any_of(current.instruction->users(),
                       HloPredicateIsOp<HloOpcode::kCustomCall>)) {
      return std::nullopt;
    }
    if (current.instruction->IsRoot() &&
        current.instruction->parent()->IsEntryComputation()) {
      return std::nullopt;
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<InstructionAndShapeIndex> successors,
                          GetSuccessors(current, call_graph));
    if (current.instruction->IsRoot()) {
      // A loop body root is also read by the next iteration.
      for (HloInstruction* caller :
           call_graph.GetComputationCallers(current.instruction->parent())) {
        if (caller->opcode() == HloOpcode::kWhile &&
            caller->while_body() == current.instruction->parent()) {
          successors.push_back({caller->while_body()->parameter_instruction(0),
                                current.shape_index});
          successors.push_back(
              {caller->while_condition()->parameter_instruction(0),
               current.shape_index});
        }
      }
    }
    for (const InstructionAndShapeIndex& successor : successors) {
      HloInstruction* user = successor.instruction;
      if (user->opcode() == HloOpcode::kDynamicUpdateSlice) {
        const absl::InlinedVector<int64_t, 4> operand_indices =
            user->OperandIndices(current.instruction);
        if (operand_indices.size() != 1) {
          return std::nullopt;
        }
        if (operand_indices.front() == 1) {
          if (!absl::c_linear_search(result, successor)) {
            result.push_back(successor);
          }
          continue;
        }
        if (!through_in_place_updates || operand_indices.front() != 0) {
          return std::nullopt;
        }
      } else if (!ForwardsBufferUnchanged(user, call_graph)) {
        return std::nullopt;
      }
      if (visited.insert(successor).second) {
        queue.push(successor);
      }
    }
  }
  if (result.empty()) {
    return std::nullopt;
  }
  return result;
}

absl::StatusOr<std::vector<InstructionAndShapeIndex>>
GetOutermostStackingDynamicUpdateSlices(
    const InstructionAndShapeIndex& dynamic_update_slice,
    const CallGraph& call_graph,
    const MovedToHostOperands& moved_to_host_operands) {
  std::vector<InstructionAndShapeIndex> result;
  absl::flat_hash_set<HloInstruction*> visited;
  std::queue<InstructionAndShapeIndex> queue;
  queue.push(dynamic_update_slice);
  while (!queue.empty()) {
    const InstructionAndShapeIndex current = queue.front();
    queue.pop();
    if (!visited.insert(current.instruction).second) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<std::vector<InstructionAndShapeIndex>> outer,
        FindDynamicUpdateSlicesReceivingUpdate(
            current, call_graph, /*through_in_place_updates=*/true));
    const Shape& buffer_shape = current.instruction->operand(0)->shape();
    const bool buffer_already_on_host =
        buffer_shape.has_layout() &&
        buffer_shape.layout().memory_space() == Layout::kHostMemorySpace;
    const bool can_stack_on_host =
        outer.has_value() && !buffer_already_on_host &&
        absl::c_all_of(*outer, [&](const InstructionAndShapeIndex& o) {
          return DynamicUpdateSliceBufferIsAllocated(o.instruction,
                                                     call_graph) &&
                 HoldsOnlyDataMovedToHost(
                     InstructionAndShapeIndex(
                         o.instruction->mutable_operand(1)),
                     call_graph, moved_to_host_operands);
        });
    if (!can_stack_on_host) {
      if (!absl::c_linear_search(result, current)) {
        result.push_back(current);
      }
      continue;
    }
    VLOG(2) << absl::StreamFormat(
        "DynamicUpdateSlice \"%s\" only stacks into %d outer "
        "DynamicUpdateSlice(s); offloading there instead.",
        current.instruction->name(), outer->size());
    for (const InstructionAndShapeIndex& o : *outer) {
      queue.push(o);
    }
  }
  return result;
}

absl::StatusOr<bool> SliceLeadsToMoveToDeviceCustomCall(
    HloInstruction* slice, const CallGraph& call_graph) {
  CHECK(slice->opcode() == HloOpcode::kDynamicSlice ||
        slice->opcode() == HloOpcode::kSlice)
      << "This function must only be called with a slice or dynamic slice.";
  std::queue<std::pair<const HloInstruction*, InstructionAndShapeIndex>> queue;
  absl::flat_hash_set<InstructionAndShapeIndex> visited;
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<InstructionAndShapeIndex> successors_of_slice,
      GetSuccessors(InstructionAndShapeIndex(slice), call_graph));
  for (const InstructionAndShapeIndex& successor : successors_of_slice) {
    if (visited.insert(successor).second) {
      queue.push({slice, successor});
    }
  }
  while (!queue.empty()) {
    const auto [prev_instruction, instruction_and_shape] = queue.front();
    queue.pop();
    HloInstruction* current_instruction = instruction_and_shape.instruction;
    if (current_instruction->IsCustomCall(
            memory_annotations::kMoveToDeviceCustomCallTarget)) {
      continue;
    }
    if (HloPredicateIsOp<HloOpcode::kDynamicSlice, HloOpcode::kSlice>(
            current_instruction)) {
      // A nested scan reads its stacked residual back with two chained slices:
      // the outer loop body slices one block, and the inner loop body slices
      // one element of that block. If the inner slice moves its result to
      // device, the outer slice already copies the block to device.
      const absl::InlinedVector<int64_t, 4> operand_indices =
          current_instruction->OperandIndices(prev_instruction);
      if (operand_indices.size() != 1 || operand_indices.front() != 0) {
        return false;
      }
      ABSL_ASSIGN_OR_RETURN(
          const bool nested_slice_leads_to_move_to_device,
          SliceLeadsToMoveToDeviceCustomCall(current_instruction, call_graph));
      if (!nested_slice_leads_to_move_to_device) {
        return false;
      }
      continue;
    }
    if (!InstructionIsAllowedBetweenDsAndMoveToDevice(current_instruction,
                                                      call_graph)) {
      LOG(WARNING) << absl::StreamFormat(
          "Encountered %s on tensor which is in host memory. %s does not move "
          "the tensor back to device. %s will be converted into host compute.",
          HloOpcodeString(slice->opcode()), slice->name(), slice->name());
      return false;
    }
    ABSL_ASSIGN_OR_RETURN(
        const std::vector<InstructionAndShapeIndex> successors,
        GetSuccessors(instruction_and_shape, call_graph));
    for (const InstructionAndShapeIndex& successor : successors) {
      if (visited.insert(successor).second) {
        queue.push({current_instruction, successor});
      }
    }
  }
  return true;
}

}  // namespace host_offload_utils
}  // namespace xla
