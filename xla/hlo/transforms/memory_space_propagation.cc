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
#include <utility>

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

namespace xla {
namespace {

constexpr HloCallBoundaryOptions kFusionBoundaryOptions(
    /*include_calls_in=*/false, /*include_control_flow_in=*/false,
    /*include_fusions_in=*/true);

// Returns true if every fusion computation of module on the given execution
// threads (all threads when empty) forwards values only through
// get-tuple-element, tuple, add-dependency, domain, optimization-barrier and
// the nested elements of copy, so that DefiningPosition and Positions below
// give the answers of HloDataflowAnalysis with ssa_form=false and
// bitcast_defines_value=true. A call, control flow or an asynchronous
// operation inside a fusion computation needs the analysis.
bool HasLocalFusionDataflow(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (const HloComputation* computation :
       module.computations(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      const HloOpcode opcode = instruction->opcode();
      if (HloDataflowAnalysis::IsAsynchronousOperationStart(opcode) ||
          HloDataflowAnalysis::IsAsynchronousOperationDone(opcode) ||
          opcode == HloOpcode::kAsyncUpdate || opcode == HloOpcode::kCall ||
          opcode == HloOpcode::kConditional || opcode == HloOpcode::kWhile) {
        return false;
      }
    }
  }
  return true;
}

// Returns the position defining the value at position, inside a fusion
// computation with local dataflow (see HasLocalFusionDataflow).
HloPosition DefiningPosition(HloPosition position) {
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

// Returns every position holding the value defined at defining, that position
// first. Every position has one forwarding predecessor, so the walk over the
// users of each position reaches each position once.
absl::InlinedVector<HloPosition, 4> Positions(const HloPosition& defining) {
  absl::InlinedVector<HloPosition, 4> positions = {defining};
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

}  // namespace

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
  // Configure bitcasts to define values. Otherwise, if there is only a bitcast
  // between a fusion input and output and these two values are in different
  // memory spaces, we can get inconsistent memory spaces between the parameter
  // and fusion operand or root and fusion output.
  if (HasLocalFusionDataflow(*module, execution_threads)) {
    dataflow_analysis_ = nullptr;
  } else {
    ABSL_ASSIGN_OR_RETURN(
        auto dataflow_analysis,
        HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                 /*bitcast_defines_value=*/true));
    dataflow_analysis_ = std::move(dataflow_analysis);
  }

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      HloDataflowPropagation::ForEachCallBoundary(
          instruction,
          [&](const HloCallBoundary& boundary) {
            // Propagate the operand subshapes.
            for (int64_t operand_idx = 0;
                 operand_idx < boundary.num_parameters(); ++operand_idx) {
              ShapeUtil::ForEachLeafShape(
                  boundary.caller_operand(operand_idx)->shape(),
                  [&](const Shape& sub_shape, const ShapeIndex& index) {
                    absl::flat_hash_set<HloPosition> visited;
                    modified |=
                        Propagate(index, boundary.callee_parameter(operand_idx),
                                  sub_shape, visited);
                  });
            }

            // Propagate output subshapes.
            ShapeUtil::ForEachLeafShape(
                instruction->shape(),
                [&](const Shape& sub_shape, const ShapeIndex& index) {
                  absl::flat_hash_set<HloPosition> visited;
                  modified |= Propagate(index, boundary.callee_root(),
                                        sub_shape, visited);
                });
          },
          kFusionBoundaryOptions);
    }
  }
  return modified;
}

bool MemorySpacePropagation::Propagate(
    ShapeIndexView index, const HloInstruction* callee_instruction,
    const Shape& src_shape, absl::flat_hash_set<HloPosition>& visited) const {
  bool modified = false;
  // The positions of the value at (callee_instruction, index), from the
  // analysis when RunImpl built one and from the fusion computation otherwise.
  const HloValue* value = nullptr;
  HloPosition defining;
  if (dataflow_analysis_ != nullptr) {
    value = &dataflow_analysis_->GetUniqueValueAt(callee_instruction,
                                                  ShapeIndex(index));
    defining = value->defining_position();
  } else {
    // HloPosition holds the mutable pointer the shape updates below need, as
    // the positions of the analysis do.
    defining = DefiningPosition(HloPosition{
        const_cast<HloInstruction*>(callee_instruction), ShapeIndex(index)});
  }
  if (!visited.insert(defining).second) {
    return false;
  }
  absl::InlinedVector<HloPosition, 4> local_positions;
  if (value == nullptr) {
    local_positions = Positions(defining);
  }
  const absl::Span<const HloPosition> positions =
      value != nullptr ? absl::MakeConstSpan(value->positions())
                       : absl::MakeConstSpan(local_positions);

  for (const HloPosition& position : positions) {
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
      modified |= Propagate(position.index, instruction->operand(0), src_shape,
                            visited);
    }

    // For fusion outputs, propagate the memory space to the fusion root.
    HloDataflowPropagation::ForEachCallBoundary(
        instruction,
        [&](const HloCallBoundary& boundary) {
          modified |= Propagate(position.index, boundary.callee_root(),
                                src_shape, visited);
        },
        kFusionBoundaryOptions);

    // For nested fusion roots and parameters, pop one level up and propagate
    // the memory space to the output or operand of the calling fusion
    // instruction.
    HloDataflowPropagation::ForEachCallerBoundary(
        instruction->parent(),
        [&](const HloCallBoundary& boundary) {
          if (!boundary.callsite->parent()->IsFusionComputation()) {
            return;
          }
          if (instruction == boundary.callee_root()) {
            modified |= Propagate(position.index, boundary.callsite, src_shape,
                                  visited);
          }
          if (instruction->opcode() == HloOpcode::kParameter &&
              instruction->parameter_number() < boundary.num_parameters()) {
            modified |= Propagate(
                position.index,
                boundary.caller_operand(instruction->parameter_number()),
                src_shape, visited);
          }
        },
        kFusionBoundaryOptions);
  }

  // For fusion uses, propagate the memory space to the fusion parameter. A
  // fusion user of a position is a use of the value there
  // (HloValue::ComputeUses), so this visits the uses of the analysis.
  for (const HloPosition& position : positions) {
    for (HloInstruction* user : position.instruction->users()) {
      if (user->opcode() != HloOpcode::kFusion) {
        continue;
      }
      for (int64_t operand_number :
           user->OperandIndices(position.instruction)) {
        modified |=
            Propagate(position.index, user->fused_parameter(operand_number),
                      src_shape, visited);
      }
    }
  }
  return modified;
}

}  // namespace xla
