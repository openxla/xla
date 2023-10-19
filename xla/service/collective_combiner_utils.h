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
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

enum AsyncCombinerStrategy {
  kTrivial,
  kNear,
};

namespace internal {

StatusOr<int64_t> SizeFromArrayShapedInstruction(
    const HloInstruction* instruction);

}  // namespace internal

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
    absl::FunctionRef<Status(HloModule* module,
                             absl::Span<HloInstruction* const>,
                             absl::Span<HloInstruction* const>)>
        combine_fn,
    int64_t combine_threshold_bytes, int64_t combine_threshold_count,
    bool is_async = false, AsyncCombinerStrategy strategy = kTrivial,
    int64_t near_op_threshold = 5,
    absl::FunctionRef<StatusOr<int64_t>(const HloInstruction*)> size_fn =
        internal::SizeFromArrayShapedInstruction) {
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

  auto scheduled_too_far =
      [&is_async, &near_op_threshold](
          absl::Span<HloInstruction* const> to_combine,
          absl::Span<HloInstruction* const> to_combine_ends,
          HloInstruction* target, HloInstruction* target_end,
          const absl::flat_hash_map<HloInstruction*, unsigned>&
              instruction_pos_map) {
        if (!is_async || instruction_pos_map.empty() || to_combine.empty())
          return false;
        unsigned last_start = instruction_pos_map.at(to_combine.back());
        unsigned new_end = instruction_pos_map.at(target_end);
        DCHECK(last_start < new_end);
        return new_end - last_start > near_op_threshold;
      };

  // Returns true if the `target` instruction can be trivially combined
  // (and rescheduled) with `to_combine`.
  // What do we consider a trivial rescheduling?
  // - If there is no schedule, return true.
  // - If there is a schedule, a trivial combination allows us to merge
  //   `target` onto to_combine[0], without having to reschedule any
  //   intermediate ops (i.e. non-grouped ops scheduled between
  //   to_combine[0] and target).
  //
  // Note: this helper assumes that we consider augmenting `to_combine`
  // in schedule order. That is, `target` is always scheduled later than
  // the ops in `to_combine`.
  auto can_trivially_reschedule =
      [&module, &computation, &is_async](
          absl::Span<HloInstruction* const> to_combine,
          absl::Span<HloInstruction* const> to_combine_ends,
          HloInstruction* target, HloInstruction* target_end,
          HloReachabilityMap* reachability,
          const absl::flat_hash_map<HloInstruction*, unsigned>&
              instruction_pos_map) {
        if (!is_async || instruction_pos_map.empty() || to_combine.empty())
          return true;
        unsigned first_start = instruction_pos_map.at(to_combine.front());
        unsigned last_end = instruction_pos_map.at(to_combine_ends.back());
        unsigned new_start = instruction_pos_map.at(target);
        DCHECK(last_end < new_start);
        unsigned new_end = instruction_pos_map.at(target_end);

        absl::flat_hash_set<HloInstruction*> start_set;
        for (HloInstruction* instruction : to_combine) {
          start_set.insert(instruction);
        }

        // Our goal is to expand to_combine so that the collective starts
        // at to_combine[0] and ends at new_end, with only trivial modifications
        // to the schedule. For this to happen:
        // 1. There shouldn't be any edges from to_combine_ends to
        //    [last_end+1, ..., new_end -1]. This is so that all those
        //    non-combining ops can be rescheduled before the combined end.
        // 2. There shouldn't be any edges from [first_start, new_start-1]
        //    to new_start. This is so that new_start can be rescheduled
        //    at first_start.
        // Example:
        //
        //  |  Collective1.Start      // "first_start" (here == "last_start")
        //  |          |         Foo
        //  |          |          |
        //  |          |          v
        //  |          v         Bar
        //  |  Collective1.End     \     // last_end
        //  |         Baz           ---------
        //  |             Collective2.Start  \-.  // "new_start"
        //  |                     |            |
        //  |                     |            v
        //  |                     |           Qux
        //  |                     |            |
        //  |                     |            v
        //  |                     v           Quux
        //  v              Collective2.End        // "new_end"
        // (Schedule order flows downwards.)
        //
        // Collective2 can be trivially merged onto Collective1 with
        // the resulting schedule being:
        // |  Collective1+2.Start
        // |          |          Foo
        // |          |           |
        // |          |           v
        // |          |          Bar
        // |          |   Baz     |
        // |          |           v
        // |          |          Qux
        // |          |           |
        // |          |           v
        // |          v          Quux
        // v  Collective1+2.End
        // (Schedule order flows downwards.)
        //
        // This trivial rescheduling wouldn't be possible if, for instance:
        // - There was an edge from Collective1.End to Baz (condition 1).
        // - There was an edge from Bar or Baz to Collective2.Start (cond 2).

        // Condition 1: no edges from to_combine_ends to
        // [last_end+1, ..., new_end -1].
        for (HloInstruction* origin : to_combine_ends) {
          for (unsigned j = last_end + 1; j < new_end; j++) {
            const HloInstruction* dest =
                module->schedule().sequence(computation).instructions()[j];
            if (reachability->IsReachable(origin, dest)) {
              VLOG(2) << "condition 1: " << origin->ToShortString() << " -> "
                      << dest->ToShortString();
              return false;
            }
          }
        }
        // Condition 2: no edges from [first_start, new_start - 1] to
        // new_start (i.e. target).
        const HloInstruction* dest = target;
        for (unsigned i = first_start; i < new_start; i++) {
          const HloInstruction* origin =
              module->schedule().sequence(computation).instructions()[i];
          if (reachability->IsReachable(origin, dest)) {
            VLOG(2) << "condition 2: " << origin->ToShortString() << " -> "
                    << dest->ToShortString();
            return false;
          }
        }
        return true;
      };

  bool changed = false;

  // Keys are removed after the instruction is combined (or never will be).
  while (!keys.empty()) {
    std::vector<HloInstruction*> to_combine;
    std::vector<HloInstruction*> to_combine_ends;  // only used in async mode.
    int64_t to_combine_bytes = 0;
    absl::flat_hash_set<HloInstruction*>* group = nullptr;

    // Recompute reachability after every combine group because we can't
    // maintain a cross group topological order to be able to rely on the
    // transitive dependencies to detect cycles.
    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(computation);

    absl::flat_hash_map<HloInstruction*, unsigned> instruction_pos_map;
    bool is_scheduled =
        module->has_schedule() &&
        module->schedule().is_computation_scheduled(computation);
    if (is_scheduled) {
      unsigned i = 0;
      for (HloInstruction* instr :
           module->schedule().sequence(computation).instructions()) {
        instruction_pos_map[instr] = i++;
      }
    }

    // If there is a schedule, visit nodes in the scheduled order.
    std::vector<HloInstruction*> instr_sequence =
        is_scheduled ? module->schedule().sequence(computation).instructions()
                     : computation->MakeInstructionPostOrder();

    absl::flat_hash_map<const HloInstruction*, HloInstruction*>
        start_to_end_map;
    if (is_async) {
      for (HloInstruction* instruction : instr_sequence) {
        switch (instruction->opcode()) {
          case HloOpcode::kAllReduceDone:
          case HloOpcode::kAllGatherDone:
          case HloOpcode::kCollectivePermuteDone:
          case HloOpcode::kAsyncDone:
            start_to_end_map[instruction->operand(0)] = instruction;
            break;
          default:
            break;
        }
      }
    }

    for (HloInstruction* instruction : instr_sequence) {
      auto it = keys.find(instruction);
      if (it == keys.end()) continue;

      // If this is the first instruction, set the active group.
      if (to_combine.empty()) {
        K key = it->second;
        group = &groups.find(key)->second;
      }

      // Check instruction is in the active group.
      if (!group->contains(instruction)) {
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

      TF_ASSIGN_OR_RETURN(int64_t instruction_bytes, size_fn(instruction));

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
      bool is_reachable = absl::c_any_of(
          is_async ? to_combine_ends : to_combine,
          [&](HloInstruction* to_combine_inst) {
            return reachability->IsReachable(to_combine_inst, instruction);
          });
      if (is_reachable) {
        VLOG(1) << "Instruction is reachable.";
        break;
      }

      HloInstruction* instruction_end =
          is_async ? start_to_end_map.at(instruction) : nullptr;

      if (strategy == kNear &&
          scheduled_too_far(to_combine, to_combine_ends, instruction,
                            instruction_end, instruction_pos_map)) {
        VLOG(1)
            << "Instruction is scheduled too far from the last one in the set.";
        break;
      }

      if (strategy == kTrivial &&
          !can_trivially_reschedule(to_combine, to_combine_ends, instruction,
                                    instruction_end, reachability.get(),
                                    instruction_pos_map)) {
        VLOG(1)
            << "Instruction is scheduled after instructions dependent "
               "on the set; the resulting set cannot be trivially rescheduled.";
        break;
      }

      VLOG(1) << "Adding instruction to set.";
      to_combine.push_back(instruction);
      if (is_async) {
        to_combine_ends.push_back(instruction_end);
      }
      to_combine_bytes += instruction_bytes;
      keys.erase(it);

      if (to_combine.size() >= combine_threshold_count) {
        VLOG(1) << "Combined count threshold reached.";
        break;
      }
    }

    if (to_combine.size() > 1) {
      TF_RETURN_IF_ERROR(combine_fn(module, to_combine, to_combine_ends));
      changed = true;
    }
  }

  return changed;
}

Status CombineCollectives(HloComputation* computation, HloInstruction* combined,
                          HloInstruction* combined_end,
                          absl::Span<HloInstruction* const> to_combine,
                          absl::Span<HloInstruction* const> to_combine_ends,
                          bool is_async);

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_
