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

#ifndef XLA_TOOLS_HLO_SLICER_H_
#define XLA_TOOLS_HLO_SLICER_H_

#include <functional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Define HloSelector, which is a lambda that, given an HLO
// instruction, returns true if selected, otherwise return false.
using HloSelector = std::function<bool(const HloInstruction*)>;

// The data structure capturing the outputs of forward/backward slicing.
struct SliceOutputStruct {
  // A map that maps from relevant HLO computation to relevant HLO
  // instructions (excluding the parts of the HLO computations/instructions that
  // are irrelevant).
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_insts;

  // Return the total number of the sliced instructions
  int num_sliced_insts() { return _count_map_of_set(sliced_insts); }

  // A map that maps from the computations to the instructions that form the
  // slicing frontier.
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      frontier_insts;

  // Return the total number of the frontier instructions
  int num_frontier_insts() { return _count_map_of_set(frontier_insts); }

  int _count_map_of_set(
      absl::flat_hash_map<const HloComputation*,
                          absl::flat_hash_set<const HloInstruction*>>&
          to_count) {
    int count = 0;
    for (const auto& [key, set] : to_count) {
      count += set.size();
    }
    return count;
  }
};

// Conduct inter-computation forward program slicing (i.e., in the direction
// that from the specified nodes to the ROOT), with the starting points as the
// provided HLO instructions (`relevant_instructions`), and with the ending
// points (frontier) dictated by a lambda function (`hlo_selector`) that users
// specify. If `ignore_control_dependency` is set as true, control dependency
// will be ignored during slicing.
SliceOutputStruct ForwardSliceModule(
    const HloModule* hlo_module,
    std::vector<const HloInstruction*>& relevant_instructions,
    HloSelector hlo_selector, bool ignore_control_dependency = false);

}  // namespace xla

#endif  // XLA_TOOLS_HLO_SLICER_H_
