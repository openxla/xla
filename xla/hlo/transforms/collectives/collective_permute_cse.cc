/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_permute_cse.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

absl::StatusOr<bool> CollectivePermuteCse::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    std::unique_ptr<HloReachabilityMap> reachability;

    std::vector<HloInstruction*> permutes;
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kCollectivePermute) {
        permutes.push_back(inst);
      }
    }

    if (permutes.size() < 2) {
      continue;
    }

    for (size_t i = 0; i < permutes.size(); ++i) {
      HloInstruction* a = permutes[i];
      if (!a || a->user_count() == 0) {
        continue;
      }
      for (size_t j = i + 1; j < permutes.size(); ++j) {
        HloInstruction* b = permutes[j];
        if (!b || b->user_count() == 0) {
          continue;
        }

        auto a_pairs = a->source_target_pairs();
        auto b_pairs = b->source_target_pairs();
        absl::c_sort(a_pairs);
        absl::c_sort(b_pairs);

        if (a_pairs != b_pairs) {
          continue;
        }

        HloInstruction* large = nullptr;
        HloInstruction* small = nullptr;

        if (a->operand(0) == b->operand(0)) {
          large = a;
          small = b;
        } else if (a->operand(0)->opcode() == HloOpcode::kSlice &&
                   a->operand(0)->operand(0) == b->operand(0)) {
          small = a;
          large = b;
        } else if (b->operand(0)->opcode() == HloOpcode::kSlice &&
                   b->operand(0)->operand(0) == a->operand(0)) {
          small = b;
          large = a;
        }

        if (large && small) {
          if (!reachability) {
            reachability = HloReachabilityMap::Build(computation);
          }
          HloReachabilityMap& reach_map = *reachability;

          if (reach_map.IsReachable(small, large)) {
            std::vector<HloInstruction*> preds_to_remove;
            for (HloInstruction* pred : large->control_predecessors()) {
              if (pred == small || reach_map.IsReachable(small, pred)) {
                preds_to_remove.push_back(pred);
              }
            }
            for (HloInstruction* pred : preds_to_remove) {
              pred->RemoveControlDependencyTo(large).IgnoreError();
            }
          }

          HloInstruction* replacement = large;
          if (small->operand(0)->opcode() == HloOpcode::kSlice &&
              large->operand(0) == small->operand(0)->operand(0)) {
            const HloInstruction* slice_op = small->operand(0);
            replacement =
                computation->AddInstruction(HloInstruction::CreateSlice(
                    small->shape(), large, slice_op->slice_starts(),
                    slice_op->slice_limits(), slice_op->slice_strides()));
          }

          // Force large to dominate small's location. By default, creating a
          // slice of large and replacing it means slice executes after large
          // and replaces small. Is it possible small was before large? Yes. If
          // small was before large, slice(large) will be computed where large
          // is, replacing the outputs that used small. We might affect memory
          // size if large is delayed. Replacement handles dependencies.

          small->ReplaceAllUsesWith(replacement).IgnoreError();
          reach_map.Replace(small, replacement);
          if (small->user_count() == 0) {
            computation->RemoveInstruction(small).IgnoreError();
          }
          if (small == a) {
            permutes[i] = nullptr;
          }
          if (small == b) {
            permutes[j] = nullptr;
          }
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace xla
