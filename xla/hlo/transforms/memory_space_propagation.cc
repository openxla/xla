/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/memory_space_propagation.h"

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

bool MemorySpacePropagation::HasLocalFusionDataflow(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (const HloComputation* computation :
       module.computations(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      // The opcodes whose output values HloDataflowAnalysis takes, in whole or
      // in part, from operands or other computations by rules other than those
      // of DefiningPosition and Positions: the cases of its
      // UpdateInstructionValueSet other than get-tuple-element, tuple,
      // add-dependency, domain, optimization-barrier, copy, bitcast (which
      // defines its values here) and parameter (which defines its values in a
      // fusion computation).
      switch (instruction->opcode()) {
        case HloOpcode::kAllGatherDone:
        case HloOpcode::kAllGatherStart:
        case HloOpcode::kAllReduceDone:
        case HloOpcode::kAsyncDone:
        case HloOpcode::kAsyncStart:
        case HloOpcode::kAsyncUpdate:
        case HloOpcode::kCall:
        case HloOpcode::kCollectivePermuteDone:
        case HloOpcode::kCollectivePermuteStart:
        case HloOpcode::kConditional:
        case HloOpcode::kCopyDone:
        case HloOpcode::kCopyStart:
        case HloOpcode::kRecvDone:
        case HloOpcode::kSend:
        case HloOpcode::kWhile:
          return false;
        default:
          break;
      }
    }
  }
  return true;
}

HloPosition MemorySpacePropagation::DefiningPosition(HloPosition position) {
  while (true) {
    HloInstruction* instruction = position.instruction;
    switch (instruction->opcode()) {
      case HloOpcode::kGetTupleElement:
        position.instruction = instruction->mutable_operand(0);
        position.index.push_front(instruction->tuple_index());
        break;
      case HloOpcode::kTuple:
        if (position.index.empty()) {
          return position;
        }
        position.instruction =
            instruction->mutable_operand(position.index.front());
        position.index.pop_front();
        break;
      case HloOpcode::kCopy:
        if (position.index.empty()) {
          return position;
        }
        [[fallthrough]];
      case HloOpcode::kAddDependency:
      case HloOpcode::kDomain:
      case HloOpcode::kOptimizationBarrier:
        position.instruction = instruction->mutable_operand(0);
        break;
      default:
        return position;
    }
  }
}

absl::InlinedVector<HloPosition, 4> MemorySpacePropagation::Positions(
    const HloPosition& defining) {
  absl::InlinedVector<HloPosition, 4> positions = {defining};
  // Every position has one forwarding predecessor, so the walk over the users
  // of each position reaches each position once.
  for (int64_t i = 0; i < positions.size(); ++i) {
    // A copy: the pushes below may reallocate positions.
    const HloPosition position = positions[i];
    for (HloInstruction* user : position.instruction->users()) {
      switch (user->opcode()) {
        case HloOpcode::kGetTupleElement:
          if (!position.index.empty() &&
              position.index.front() == user->tuple_index()) {
            positions.push_back(HloPosition{user, position.index});
            positions.back().index.pop_front();
          }
          break;
        case HloOpcode::kTuple:
          for (int64_t operand_number :
               user->OperandIndices(position.instruction)) {
            positions.push_back(HloPosition{user, position.index});
            positions.back().index.push_front(operand_number);
          }
          break;
        case HloOpcode::kCopy:
          if (!position.index.empty()) {
            positions.push_back(HloPosition{user, position.index});
          }
          break;
        case HloOpcode::kAddDependency:
        case HloOpcode::kDomain:
        case HloOpcode::kOptimizationBarrier:
          if (user->operand(0) == position.instruction) {
            positions.push_back(HloPosition{user, position.index});
          }
          break;
        default:
          break;
      }
    }
  }
  return positions;
}

absl::InlinedVector<HloUse, 1> MemorySpacePropagation::FusionUses(
    absl::Span<const HloPosition> positions) {
  absl::InlinedVector<HloUse, 1> uses;
  for (const HloPosition& position : positions) {
    for (HloInstruction* user : position.instruction->users()) {
      if (user->opcode() != HloOpcode::kFusion) {
        continue;
      }
      for (int64_t operand_number :
           user->OperandIndices(position.instruction)) {
        uses.push_back(HloUse(user, operand_number, position.index));
      }
    }
  }
  return uses;
}

HloPosition MemorySpacePropagation::DefiningPositionAt(
    HloInstruction* instruction, ShapeIndexView index) const {
  if (dataflow_analysis_ != nullptr) {
    return dataflow_analysis_->GetUniqueValueAt(instruction, ShapeIndex(index))
        .defining_position();
  }
  return DefiningPosition(HloPosition{instruction, ShapeIndex(index)});
}

MemorySpacePropagation::ValueView MemorySpacePropagation::ViewValue(
    const HloPosition& defining) const {
  ValueView view;
  if (dataflow_analysis_ != nullptr) {
    const HloValue& value = dataflow_analysis_->GetValueDefinedAt(
        defining.instruction, defining.index);
    view.positions.assign(value.positions().begin(), value.positions().end());
    for (const HloUse& use : value.GetUses()) {
      if (use.instruction->opcode() == HloOpcode::kFusion) {
        view.fusion_uses.push_back(use);
      }
    }
    return view;
  }
  view.positions = Positions(defining);
  view.fusion_uses = FusionUses(view.positions);
  return view;
}

bool MemorySpacePropagation::RunOnComputation(HloComputation* computation) {
  CHECK(dataflow_analysis_ != nullptr);
  bool modified = false;
  // Propagate the parameter subshapes.
  for (int parameter_idx = 0; parameter_idx < computation->num_parameters();
       ++parameter_idx) {
    ShapeUtil::ForEachLeafShape(
        computation->parameter_instruction(parameter_idx)->shape(),
        [&](const Shape& sub_shape, const ShapeIndex& index) {
          absl::flat_hash_set<HloPosition> visited;
          modified |= Propagate(
              index, computation->parameter_instruction(parameter_idx),
              sub_shape, visited);
        });
  }
  // Propagate output subshapes.
  ShapeUtil::ForEachLeafShape(
      computation->root_instruction()->shape(),
      [&](const Shape& sub_shape, const ShapeIndex& index) {
        absl::flat_hash_set<HloPosition> visited;
        modified |= Propagate(index, computation->root_instruction(), sub_shape,
                              visited);
      });
  return modified;
}

absl::StatusOr<bool> MemorySpacePropagation::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  dataflow_analysis_.reset();
  if (!HasLocalFusionDataflow(*module, execution_threads)) {
    // Configure bitcasts to define values. Otherwise, if there is only a
    // bitcast between a fusion input and output and these two values are in
    // different memory spaces, we can get inconsistent memory spaces between
    // the parameter and fusion operand or root and fusion output.
    ABSL_ASSIGN_OR_RETURN(
        dataflow_analysis_,
        HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                 /*bitcast_defines_value=*/true));
  }

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        // Propagate the operand subshapes.
        for (int operand_idx = 0;
             operand_idx < instruction->fused_parameters().size();
             ++operand_idx) {
          ShapeUtil::ForEachLeafShape(
              instruction->operand(operand_idx)->shape(),
              [&](const Shape& sub_shape, const ShapeIndex& index) {
                absl::flat_hash_set<HloPosition> visited;
                modified |=
                    Propagate(index, instruction->fused_parameter(operand_idx),
                              sub_shape, visited);
              });
        }

        // Propagate output subshapes.
        ShapeUtil::ForEachLeafShape(
            instruction->shape(),
            [&](const Shape& sub_shape, const ShapeIndex& index) {
              absl::flat_hash_set<HloPosition> visited;
              modified |= Propagate(index, instruction->fused_expression_root(),
                                    sub_shape, visited);
            });
      }
    }
  }
  return modified;
}

bool MemorySpacePropagation::Propagate(
    ShapeIndexView index, HloInstruction* callee_instruction,
    const Shape& src_shape, absl::flat_hash_set<HloPosition>& visited) const {
  const HloPosition defining = DefiningPositionAt(callee_instruction, index);
  if (!visited.insert(defining).second) {
    return false;
  }
  bool modified = false;
  const ValueView value = ViewValue(defining);

  for (const HloPosition& position : value.positions) {
    HloInstruction* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    std::optional<SplitConfig> dest_split_config =
        LayoutUtil::GetSplitConfig(*shape);
    std::optional<SplitConfig> src_split_config =
        LayoutUtil::GetSplitConfig(src_shape);

    if (shape->layout().memory_space() != src_shape.layout().memory_space() ||
        dest_split_config != src_split_config) {
      shape->mutable_layout()->set_memory_space(
          src_shape.layout().memory_space());
      shape->mutable_layout()->clear_split_configs();
      if (src_split_config.has_value()) {
        shape->mutable_layout()->add_split_configs(*src_split_config);
      }
      modified = true;
    }

    if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      modified |= Propagate(position.index, instruction->mutable_operand(0),
                            src_shape, visited);
    }

    // For fusion outputs, propagate the memory space to the fusion root.
    if (instruction->opcode() == HloOpcode::kFusion) {
      modified |=
          Propagate(position.index, instruction->fused_expression_root(),
                    src_shape, visited);
    }

    HloInstruction* parent_fusion = instruction->parent()->FusionInstruction();
    // For nested fusion roots, pop one level up and propagate the memory space
    // to the output of the calling fusion instruction.
    if (parent_fusion != nullptr &&
        instruction == instruction->parent()->root_instruction() &&
        parent_fusion->parent()->IsFusionComputation()) {
      modified |= Propagate(position.index, parent_fusion, src_shape, visited);
    }

    // For nested fusion parameters, pop one level up and propagate the memory
    // space to the operand of the calling fusion instruction.
    if (instruction->opcode() == HloOpcode::kParameter &&
        parent_fusion != nullptr &&
        parent_fusion->parent()->IsFusionComputation()) {
      HloInstruction* fusion_operand =
          parent_fusion->mutable_operand(instruction->parameter_number());
      modified |= Propagate(position.index, fusion_operand, src_shape, visited);
    }
  }

  for (const HloUse& use : value.fusion_uses) {
    // For fusion uses, propagate the memory space to the fusion parameter.
    modified |= Propagate(use.operand_index,
                          use.instruction->fused_parameter(use.operand_number),
                          src_shape, visited);
  }
  return modified;
}

}  // namespace xla
