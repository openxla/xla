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

#include "xla/hlo/analysis/hlo_liveness_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tuple_tree.h"

namespace xla {

HloLivenessAnalysis::HloLivenessAnalysis(const HloModule& module)
    : module_(module), call_graph_(CallGraph::Build(&module)) {}

const HloLivenessAnalysis::InstructionLiveness*
HloLivenessAnalysis::FindLiveness(const HloInstruction* instruction) const {
  const HloComputation* computation = instruction->parent();
  // Removed instructions have no parent. Instructions of other modules or of
  // computations added after the analysis ran were never analyzed.
  if (computation == nullptr || computation->parent() != &module_ ||
      computation->unique_id() < 0 ||
      computation->unique_id() >= static_cast<int64_t>(table_.size())) {
    return nullptr;
  }
  const std::vector<InstructionLiveness>& entries =
      table_[computation->unique_id()].entries;
  const int32_t local_id = instruction->local_id();
  if (local_id >= 0 && local_id < static_cast<int32_t>(entries.size()) &&
      entries[local_id].instruction == instruction) {
    return &entries[local_id];
  }
  // Only an instruction added since the analysis ran may miss its slot, unless
  // the computation renumbered its instructions, after which the table cannot
  // answer for the ones it saw.
  CHECK(absl::c_none_of(entries,
                        [instruction](const InstructionLiveness& entry) {
                          return entry.instruction == instruction;
                        }))
      << instruction->name() << " was renumbered after the liveness analysis "
      << "of " << module_.name() << " ran";
  return nullptr;
}

HloLivenessAnalysis::InstructionLiveness& HloLivenessAnalysis::GetLiveness(
    const HloInstruction* instruction) {
  InstructionLiveness& liveness = table_[instruction->parent()->unique_id()]
                                      .entries[instruction->local_id()];
  DCHECK_EQ(liveness.instruction, instruction);
  return liveness;
}

void HloLivenessAnalysis::AddToWorklist(InstructionLiveness& liveness) {
  if (!liveness.on_worklist) {
    liveness.on_worklist = true;
    worklist_.push_back(&liveness);
    VLOG(3) << "ADD instruction: " << liveness.instruction->name();
  }
}

// Marks 'instruction' output live at 'shape_index'.
// Adds to worklist iff:
// *) 'instruction' is not already on worklist.
// *) 'shape_index' has not yet been visited.
void HloLivenessAnalysis::MarkLiveAtIndex(const HloInstruction* instruction,
                                          const ShapeIndex& shape_index) {
  InstructionLiveness& liveness = GetLiveness(instruction);
  bool* alive;
  if (instruction->shape().IsTuple()) {
    if (liveness.live_subshapes == nullptr) {
      liveness.live_subshapes = std::make_unique<TupleTree<bool>>(
          instruction->shape(), /*init_value=*/false);
    }
    alive = liveness.live_subshapes->mutable_element(shape_index);
  } else {
    CHECK(shape_index.empty())
        << "Index " << shape_index << " into " << instruction->name();
    alive = &liveness.live;
  }
  if (!*alive) {
    AddToWorklist(liveness);
    *alive = true;
    VLOG(3) << "MARK instruction: " << instruction->name()
            << " shape_index: " << shape_index;
  }
}

// Marks 'instruction' live at all shape indices in its output.
void HloLivenessAnalysis::MarkLiveAtAllIndices(
    const HloInstruction* instruction) {
  InstructionLiveness& liveness = GetLiveness(instruction);
  bool add_to_worklist = false;
  if (!instruction->shape().IsTuple()) {
    if (!liveness.live) {
      liveness.live = true;
      add_to_worklist = true;
    }
  } else if (liveness.live_subshapes == nullptr) {
    liveness.live_subshapes = std::make_unique<TupleTree<bool>>(
        instruction->shape(), /*init_value=*/true);
    add_to_worklist = true;
  } else {
    for (auto& entry : *liveness.live_subshapes) {
      if (!entry.second) {
        add_to_worklist = true;
        entry.second = true;
        VLOG(3) << "MARK instruction: " << instruction->name()
                << " shape_index: " << entry.first;
      }
    }
  }
  if (add_to_worklist) {
    AddToWorklist(liveness);
  }
}

// Visits the live shape indices in the pre order of the TupleTree, which is
// the order the propagation below marks operands in.
void HloLivenessAnalysis::ForEachLiveIndex(
    const InstructionLiveness& liveness,
    absl::FunctionRef<void(const ShapeIndex&)> func) {
  if (liveness.live_subshapes != nullptr) {
    for (const auto& entry : *liveness.live_subshapes) {
      if (entry.second) {
        func(entry.first);
      }
    }
  } else if (liveness.live) {
    func(ShapeIndex());
  }
}

// Propagates liveness through Tuple instructions.
// *) For each tuple operand:
//   *) For tuple output shape index associated with operand:
//     *) Propagate live shape indices to tuple operand at the associated
//        shape index in the operands output, and add to worklist.
void HloLivenessAnalysis::PropagateLivenessThroughTuple(
    const InstructionLiveness& liveness) {
  const HloInstruction* instruction = liveness.instruction;
  CHECK_EQ(instruction->opcode(), HloOpcode::kTuple);
  ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
    const size_t size = shape_index.size();
    if (size == 0) {
      return;
    }
    const int64_t operand_index = shape_index[0];
    if (operand_index >= instruction->operand_count()) {
      return;
    }
    // Mark top-level index of operand at 'operand_index'.
    MarkLiveAtIndex(instruction->operand(operand_index), {});
    if (size == 1) {
      return;  // The sub-shape index is the top-level index marked above.
    }
    // Mark sub-shape index of operand at 'operand_index'.
    ShapeIndex operand_shape_index(size - 1);
    for (int i = 1; i < size; ++i) {
      operand_shape_index[i - 1] = shape_index[i];
    }
    MarkLiveAtIndex(instruction->operand(operand_index), operand_shape_index);
  });
}

// Propagates liveness through GetTupleElement instructions.
// *) For each live index in GetTupleElement output, mark output of GTE operand
//    at associated shape index in its output, and add to worklist.
void HloLivenessAnalysis::PropagateLivenessThroughGTE(
    const InstructionLiveness& liveness) {
  const HloInstruction* instruction = liveness.instruction;
  CHECK_EQ(instruction->opcode(), HloOpcode::kGetTupleElement);
  // Mark operand top-level index.
  MarkLiveAtIndex(instruction->operand(0), {});
  // Propagate live shape indices along GTE -> Tuple edge.
  ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
    ShapeIndex operand_shape_index(shape_index);
    operand_shape_index.push_front(instruction->tuple_index());
    MarkLiveAtIndex(instruction->operand(0), operand_shape_index);
  });
}

// Propagates liveness through While instructions.
// *) For each live index in While output, mark shape index of while.body.root
//    and while.operand (adding each to worklist).
// *) Mark while.cond.root and add to worklist.
void HloLivenessAnalysis::PropagateLivenessThroughWhile(
    const InstructionLiveness& liveness) {
  const HloInstruction* instruction = liveness.instruction;
  CHECK_EQ(instruction->opcode(), HloOpcode::kWhile);
  ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
    // Propagate liveness to while body computation root instruction.
    MarkLiveAtIndex(instruction->while_body()->root_instruction(), shape_index);
    // Propagate liveness to tuple-shaped operand.
    MarkLiveAtIndex(instruction->operand(0), shape_index);
  });

  // Propagate liveness to while condition computation root instruction.
  MarkLiveAtIndex(instruction->while_condition()->root_instruction(), {});
}

// Propagates liveness through Call instructions.
// For each live index in Call output, mark shape index of to_apply.root
// and add it to the worklist.
void HloLivenessAnalysis::PropagateLivenessThroughCall(
    const InstructionLiveness& liveness) {
  const HloInstruction* instruction = liveness.instruction;
  CHECK_EQ(instruction->opcode(), HloOpcode::kCall);
  ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
    MarkLiveAtIndex(instruction->to_apply()->root_instruction(), shape_index);
  });
}

// Propagates liveness out of Parameter instructions to callers and aliasing
// positions. This can occur if liveness propagates to a parameter in the
// while.condition computation, requiring liveness to propagate out to caller
// callsite while (and while.body.root).
void HloLivenessAnalysis::PropagateLivenessToParameterCallers(
    const InstructionLiveness& liveness) {
  const HloInstruction* instruction = liveness.instruction;
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(instruction->parent());
  if (call_graph_node.context() == CallContext::kControlFlow) {
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
        auto* xla_while = callsite.instruction();
        ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
          // Propagate liveness to while result{shape_index}
          MarkLiveAtIndex(xla_while, shape_index);
          // Propagate liveness to while body root{shape_index}.
          MarkLiveAtIndex(xla_while->while_body()->root_instruction(),
                          shape_index);
          // Propagate liveness to operand(0){shape_index}.
          MarkLiveAtIndex(xla_while->operand(0), shape_index);
        });
      } else if (callsite.instruction()->opcode() == HloOpcode::kCall) {
        HloInstruction* xla_call = callsite.instruction();
        const HloInstruction* call_operand =
            xla_call->operand(instruction->parameter_number());
        ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
          // Propagate liveness to call operand.
          MarkLiveAtIndex(call_operand, shape_index);
        });
      }
    }
  }
}

// Makes sure that if a live instruction is within a computation used in control
// flow operations, we mark live even other related instructions.
void HloLivenessAnalysis::PropagateLivenessThroughControlFlow(
    const InstructionLiveness& liveness) {
  const HloInstruction* instruction = liveness.instruction;
  // Every mark below depends on the computation alone, except the marks of a
  // parameter's live indices, so a second visit of the computation by another
  // kind of instruction would only repeat marks that are already live.
  ComputationLiveness& computation_liveness =
      table_[instruction->parent()->unique_id()];
  if (computation_liveness.control_flow_propagated &&
      instruction->opcode() != HloOpcode::kParameter) {
    return;
  }
  computation_liveness.control_flow_propagated = true;
  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(instruction->parent());
  if (call_graph_node.context() == CallContext::kControlFlow) {
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      HloInstruction* caller = callsite.instruction();
      if (caller->opcode() == HloOpcode::kWhile) {
        // If a live instruction is within the %while body or condition
        // computation, mark the predicate value returned by the condition
        // computation live as well.
        MarkLiveAtIndex(caller->while_condition()->root_instruction(), {});
      } else if (caller->opcode() == HloOpcode::kConditional) {
        // If a live instruction is within the true or false branches of a
        // conditional, we mark the predicate operand live as well.
        MarkLiveAtIndex(caller->operand(0), {});
        // Mark the caller instruction live.
        MarkLiveAtIndex(caller, {});
        // Propagate liveness to the caller computation.
        const HloComputation* callee_comp = instruction->parent();
        // Initialize 'operand_index' to skip predictate operand.
        int64_t operand_index = 1;
        for (auto* caller_comp : caller->called_computations()) {
          if (callee_comp == caller_comp) {
            MarkLiveAtIndex(caller->operand(operand_index), {});
            if (instruction->opcode() == HloOpcode::kParameter) {
              // If 'instruction' is a parameter, propagate live shape indices
              // to the associated callsite's argument shape indices.
              ForEachLiveIndex(liveness, [&](const ShapeIndex& shape_index) {
                MarkLiveAtIndex(caller->operand(operand_index), shape_index);
              });
            }
            break;
          }
          ++operand_index;
        }
      }
    }
  }
}

// Runs liveness analysis on 'module_'.
// Initializes worklist with entry root instruction (and any instruction with
// side-effects), marking all of their output shape indices live.
// Visits elements on worklist, propagating liveness from an instructions
// live output shape indices to its called computations and operands.
void HloLivenessAnalysis::RunAnalysis() {
  int64_t max_computation_id = -1;
  for (const HloComputation* computation : module_.computations()) {
    CHECK_GE(computation->unique_id(), 0) << computation->name();
    max_computation_id = std::max(max_computation_id, computation->unique_id());
  }
  table_.resize(max_computation_id + 1);
  // One sweep sizes the table and collects the instructions to seed. They are
  // marked after the entry root so that the worklist keeps its order.
  std::vector<const HloInstruction*> seeds;
  // The backend options apply to every instruction of the module alike.
  const bool options_disable_while_loop_dce =
      HasDisableWhileLoopDceOption(module_);
  for (const HloComputation* computation : module_.computations()) {
    std::vector<InstructionLiveness>& entries =
        table_[computation->unique_id()].entries;
    // Local ids index the instruction list, which keeps the slots of deleted
    // instructions until HloComputation::Cleanup compacts it.
    const int32_t local_id_end =
        computation->next_unique_instruction_internal_id();
    entries.resize(local_id_end);
    for (const HloInstruction* instruction : computation->instructions()) {
      const int32_t local_id = instruction->local_id();
      CHECK_GE(local_id, 0) << instruction->name();
      CHECK_LT(local_id, local_id_end) << instruction->name();
      entries[local_id].instruction = instruction;
      if (instruction->HasSideEffectNoRecurse() ||
          options_disable_while_loop_dce ||
          HasDisableWhileLoopDceFrontendAttr(instruction)) {
        seeds.push_back(instruction);
      }
    }
  }

  // Add entry computation root instruction.
  MarkLiveAtAllIndices(module_.entry_computation()->root_instruction());
  for (const HloInstruction* instruction : seeds) {
    // Mark live at all indices if the instruction has side effects or if
    // dead code elimination is explicitly disabled via frontend attribute
    // (e.g., on special while loops).
    MarkLiveAtAllIndices(instruction);
  }

  while (!worklist_.empty()) {
    InstructionLiveness& liveness = *worklist_.front();
    worklist_.pop_front();
    liveness.on_worklist = false;
    VLOG(1) << "VISIT instruction: " << liveness.instruction->name();
    switch (liveness.instruction->opcode()) {
      case HloOpcode::kTuple:
        PropagateLivenessThroughTuple(liveness);
        break;
      case HloOpcode::kGetTupleElement:
        PropagateLivenessThroughGTE(liveness);
        break;
      case HloOpcode::kWhile:
        PropagateLivenessThroughWhile(liveness);
        break;
      case HloOpcode::kCall:
        PropagateLivenessThroughCall(liveness);
        break;
      case HloOpcode::kParameter:
        PropagateLivenessToParameterCallers(liveness);
        break;
      default:
        // Propagate liveness to called computations.
        for (auto* called_computation :
             liveness.instruction->called_computations()) {
          MarkLiveAtAllIndices(called_computation->root_instruction());
        }
        // Propagate liveness to operands.
        for (HloInstruction* operand : liveness.instruction->operands()) {
          MarkLiveAtAllIndices(operand);
        }
        break;
    }
    PropagateLivenessThroughControlFlow(liveness);
  }
}

bool HloLivenessAnalysis::IsLive(const HloInstruction* instruction,
                                 const ShapeIndex& shape_index) const {
  const InstructionLiveness* liveness = FindLiveness(instruction);
  if (liveness == nullptr) {
    return false;
  }
  if (liveness->live_subshapes != nullptr) {
    return liveness->live_subshapes->element(shape_index);
  }
  // A dead instruction answers false before its index is checked, as the
  // former map did for an instruction it had never seen.
  if (!liveness->live) {
    return false;
  }
  CHECK(shape_index.empty())
      << "Index " << shape_index << " into " << instruction->name();
  return true;
}

/* static */
absl::StatusOr<std::unique_ptr<HloLivenessAnalysis>> HloLivenessAnalysis::Run(
    const HloModule& module) {
  VLOG(1) << "HloLivenessAnalysis::Run on module " << module.name();
  XLA_VLOG_LINES(2, module.ToString());

  auto liveness_analysis = absl::WrapUnique(new HloLivenessAnalysis(module));

  liveness_analysis->RunAnalysis();

  return liveness_analysis;
}

}  // namespace xla
