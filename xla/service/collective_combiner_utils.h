/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_
#define XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_

#include <functional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

// Combines instructions with matching keys together.
//
// Instructions are combined in topological post-order.
//
// `key_fn` should return equal keys for two instructions that might be combined
// together. Instructions will be combined until the threshold for output byte
// size or instruction count is reached.
template <typename K>
StatusOr<bool> CombineInstructionsByKey(
    HloComputation* computation,
    absl::FunctionRef<std::optional<K>(const HloInstruction*)> key_fn,
    absl::FunctionRef<Status(absl::Span<HloInstruction* const>)> combine_fn,
    int64_t combine_threshold_bytes, int64_t combine_threshold_count) {
  // Cache keys for each instruction and build sets of instructions with the
  // same key that might be combined together.
  absl::flat_hash_map<HloInstruction*, K> keys;
  absl::flat_hash_map<K, absl::flat_hash_set<HloInstruction*>> groups;
  for (HloInstruction* instruction : computation->instructions()) {
    std::optional<K> key = key_fn(instruction);
    if (key) {
      keys.insert({instruction, *key});
      groups[*key].insert(instruction);
    }
  }

  HloModule* module = computation->parent();

  // Returns true if the `target` instruction can be trivially combined
  // (and rescheduled) with `to_combine`.
  // What do we consider a trivial rescheduling?
  // - If there is no schedule, return true.
  // - If there is a schedule, a trivial combination allows us to merge
  //   the ops in `to_combine` at target's schedule point, without
  //   having to reschedule any intermediate ops (i.e. non-grouped ops
  //   scheduled between to_combine[0] and target).
  //
  // Note: this helper assumes that we consider augmenting `to_combine`
  // in schedule order. That is, `target` is always scheduled later than
  // the ops in `to_combine`.
  auto can_trivially_reschedule =
      [&module, &computation](
          const std::vector<HloInstruction*>& to_combine,
          HloInstruction* target, HloReachabilityMap* reachability,
          const absl::flat_hash_map<HloInstruction*, unsigned>&
              instruction_pos_map) {
        if (instruction_pos_map.empty() || to_combine.empty()) return true;
        unsigned lo = instruction_pos_map.at(to_combine[0]);
        unsigned hi = instruction_pos_map.at(target);
        DCHECK(lo < hi);

        // There should be no edges in the schedule going from `to_combine` to
        // [lo, hi - 1]; this ensures that we can trivially schedule the
        // combined op at 'hi' (i.e. `target`). Example:
        //
        //  |  Collective.1
        //  |      |          Add
        //  |      v           |
        //  |     Foo          v
        //  v             Collective.2
        // (Schedule order flows downwards.)
        //
        // Collective.1 cannot be trivially combined with Collective.2, because
        // Foo would also have to be rescheduled to occur after Collective.2.
        //
        // If Foo didn't exist or it were originally scheduled after
        // Collective.2, the combination would be trivially rescheduled by
        // combining both collectives where Collective.2 is scheduled.
        unsigned comb_idx = 1;  // we skip lo == to_combine[0].
        for (unsigned i = lo + 1; i < hi; i++) {
          const HloInstruction* curr =
              module->schedule().sequence(computation).instructions()[i];

          // Skip when curr == to_combine[i]. We could check the entire vector
          // or create a hash map; however it is more efficient to rely on the
          // fact that its elements appear in schedule order.
          if (comb_idx < to_combine.size() && to_combine[comb_idx] == curr) {
            comb_idx++;
            continue;
          }

          bool is_reachable = absl::c_any_of(
              to_combine,
              [&reachability, &curr](HloInstruction* to_combine_inst) {
                return reachability->IsReachable(to_combine_inst, curr);
              });
          if (is_reachable) {
            return false;
          }
        }
        return true;
      };

  bool changed = false;

  // Keys are removed after the instruction is combined (or never will be).
  while (!keys.empty()) {
    std::vector<HloInstruction*> to_combine;
    int64_t to_combine_bytes = 0;
    absl::flat_hash_set<HloInstruction*>* group = nullptr;

    // Recompute reachability after every combine group because we can't
    // maintain a cross group topological order to be able to rely on the
    // transitive dependencies to detect cycles.
    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(computation);

    absl::flat_hash_map<HloInstruction*, unsigned> instruction_pos_map;
    if (module->has_schedule()) {
      unsigned i = 0;
      for (HloInstruction* instr :
           module->schedule().sequence(computation).instructions()) {
        instruction_pos_map[instr] = i++;
      }
    }

    // If there is a schedule, visit nodes on the scheduled order. This
    // simplifies the `can_trivially_reschedule` lambda above.
    std::vector<HloInstruction*> instr_sequence =
        module->has_schedule()
            ? module->schedule().sequence(computation).instructions()
            : computation->MakeInstructionPostOrder();
    for (HloInstruction* instruction : instr_sequence) {
      auto it = keys.find(instruction);
      if (it == keys.end()) continue;

      // If this is the first instruction, set the active group.
      if (to_combine.empty()) {
        K key = it->second;
        group = &groups.find(key)->second;
      }

      // Check instruction is in the active group.
      if (group->find(instruction) == group->end()) {
        continue;
      }

      VLOG(1) << "Considering HLO " << instruction->ToString()
              << " with current set size of " << to_combine_bytes
              << " and current operand count of " << to_combine.size();

      // We do not handle ops that have more than one operand since that is
      // simpler and this pass is the only way to generate such ops.
      if (instruction->operands().size() != 1) {
        VLOG(1) << "Skipping due to " << instruction->operands().size()
                << " operands";
        keys.erase(it);
        continue;
      }

      TF_RET_CHECK(instruction->shape().IsArray());
      int64_t instruction_bytes = ShapeUtil::ByteSizeOf(instruction->shape());

      // If the instruction is greater than the threshold, then we can never
      // combine it with anything.
      if (instruction_bytes > combine_threshold_bytes) {
        VLOG(1) << "Size " << instruction_bytes << " above threshold.";
        keys.erase(it);
        continue;
      }

      if (to_combine_bytes + instruction_bytes > combine_threshold_bytes) {
        VLOG(1) << "Combined size threshold exceeded.";
        break;
      }

      // We can't combine dependent instructions.
      bool is_reachable =
          absl::c_any_of(to_combine, [&](HloInstruction* to_combine_inst) {
            return reachability->IsReachable(to_combine_inst, instruction);
          });
      if (is_reachable) {
        VLOG(1) << "Instruction is reachable.";
        break;
      }

      if (!can_trivially_reschedule(to_combine, instruction, reachability.get(),
                                    instruction_pos_map)) {
        VLOG(1)
            << "Instruction is scheduled after instructions dependent "
               "on the set; the resulting set cannot be trivially rescheduled.";
        break;
      }

      VLOG(1) << "Adding instruction to set.";
      to_combine.push_back(instruction);
      to_combine_bytes += instruction_bytes;
      keys.erase(it);

      if (to_combine.size() >= combine_threshold_count) {
        VLOG(1) << "Combined count threshold reached.";
        break;
      }
    }

    if (to_combine.size() > 1) {
      TF_RETURN_IF_ERROR(combine_fn(to_combine));
      changed = true;
    }
  }

  return changed;
}

// Helper function to be called from the CombineInstructionsByKey's combine_fn.
//
// If the module has been scheduled, updates its schedule to reflect the
// transformation described by the `replacements` map (i.e. keys are replaced by
// values) and the introduction of the combining instruction (`combined`).
//
// Correctness requires no dependencies in the schedule between the instructions
// to be combined; we thus schedule the combiner (and subsequent replacements,
// i.e. get-tuple-element's) where the last element of `to_combine` is.
void MaybeUpdateSchedulePostCombining(
    HloComputation* computation, HloInstruction* combined,
    absl::Span<HloInstruction* const> to_combine,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>& replacements);

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_
