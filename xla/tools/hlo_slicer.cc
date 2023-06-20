/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tools/hlo_slicer.h"

#include <deque>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"

namespace xla {
namespace {

// Intra-Computation forward/backward slicing: Conduct slicing inside the given
// computation. It begins with the relevant instructions in
// `sliced_insts_map`, and it adds all the instructions propagated in-place.
//
// If a frontier instruction is encountered, it will be added to
// `frontier_insts`.
void IntraCompSlicing(
    absl::flat_hash_set<const HloInstruction*>& sliced_insts,
    absl::flat_hash_set<const HloInstruction*>& frontier_insts,
    bool forward_slice, HloSelector hlo_selector,
    bool ignore_control_dependency) {
  std::deque<const HloInstruction*> worklist(sliced_insts.begin(),
                                             sliced_insts.end());

  while (!worklist.empty()) {
    const HloInstruction* inst = worklist.back();

    // Initialize data-dependent instructions
    std::vector<HloInstruction*> insts_to_propagate =
        forward_slice ? std::vector<HloInstruction*>(inst->users().begin(),
                                                     inst->users().end())
                      : std::vector<HloInstruction*>(inst->operands().begin(),
                                                     inst->operands().end());

    // Append control-dependent instructions if necessary
    if (!ignore_control_dependency) {
      if (forward_slice) {
        insts_to_propagate.insert(insts_to_propagate.end(),
                                  inst->control_successors().begin(),
                                  inst->control_successors().end());
      } else {
        insts_to_propagate.insert(insts_to_propagate.end(),
                                  inst->control_predecessors().begin(),
                                  inst->control_predecessors().end());
      }
    }

    for (auto inst : insts_to_propagate) {
      if (!hlo_selector(inst)) {
        frontier_insts.insert(inst);
        sliced_insts.insert(inst);
        continue;
      }

      if (!sliced_insts.contains(inst)) {
        worklist.push_front(inst);
        sliced_insts.insert(inst);
      }
    }
    worklist.pop_back();
  }
}

// The unified implementation of (forward/backward) slicing.
//
// forward_slice: true -> forward slicing; false -> backward slicing
SliceOutputStruct SliceModule(
    const HloModule* hlo_module,
    std::vector<const HloInstruction*>& relevant_instructions,
    HloSelector hlo_selector, bool ignore_control_dependency,
    bool forward_slice) {
  // Initialize `sliced_comp_insts_map`, which keeps track of all the sliced
  // instructions
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_comp_insts_map;
  for (auto inst : relevant_instructions) {
    sliced_comp_insts_map[inst->parent()].insert(inst);
  }

  // Initialize `frontier_comp_insts_map`
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      frontier_comp_insts_map;

  // Build call graph
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module);

  // Traverse computations in the post-order(forward slicing) or
  // pre-order(backward slicing) manner, and conduct intra-computation
  // slicing in that order.
  std::vector<HloComputation*> computations_to_traverse =
      forward_slice ? hlo_module->MakeComputationPostOrder()
                    // TODO: Dummy value for backward_slice for now
                    : std::vector<HloComputation*>();
  for (auto computation : computations_to_traverse) {
    if (sliced_comp_insts_map.contains(computation)) {
      // Do intra-computation slicing
      IntraCompSlicing(sliced_comp_insts_map[computation],
                       frontier_comp_insts_map[computation], forward_slice,
                       hlo_selector, ignore_control_dependency);

      // Forward slicing: Continue propagating to successors of the current
      // computation if the ROOT instruction of the current computation is
      // sliced
      if (forward_slice && sliced_comp_insts_map[computation].contains(
                               computation->root_instruction())) {
        for (auto caller_inst :
             call_graph->GetComputationCallers(computation)) {
          sliced_comp_insts_map[caller_inst->parent()].insert(caller_inst);
        }
      }
      // TODO: Backward slice not implemented yet
      // Backward slicing: propagate to the predecessors of the current
      // computation that the sliced instructions invoke
      if (!forward_slice) {
        QCHECK(false);
      }
    }
  }

  return SliceOutputStruct{sliced_comp_insts_map, frontier_comp_insts_map};
}

}  // namespace

SliceOutputStruct ForwardSliceModule(
    const HloModule* hlo_module,
    std::vector<const HloInstruction*>& relevant_instructions,
    HloSelector hlo_selector, bool ignore_control_dependency) {
  return SliceModule(hlo_module, relevant_instructions, hlo_selector,
                     ignore_control_dependency, /*forward_slice=*/true);
}

}  // namespace xla
