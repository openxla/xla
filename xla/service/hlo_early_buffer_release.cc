/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/hlo_early_buffer_release.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/decision.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

using Positions = absl::flat_hash_map<const HloInstruction*, int64_t>;

struct MemoryProfile {
  int64_t peak = 0;
  // Byte-instructions, used only to break peak-memory ties. Using floating
  // point avoids overflow for large modules; this is not a timing estimate.
  long double area = 0;
};

// Count physical alias buffers once, including their entire live interval.
// Inputs and outputs coexist at an instruction: release events are applied at
// end + 1. This deliberately does not assume optional operand/output reuse.
// Flattening includes memory inside called computations and async lifetimes.
absl::StatusOr<MemoryProfile> GetMemoryProfile(
    const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
    const HloComputation* computation,
    const BufferValue::SizeFunction& size_function) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> live_range,
      HloLiveRange::Run(schedule, alias_analysis, computation));
  std::vector<int64_t> events(live_range->schedule_end_time() + 2, 0);
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    int64_t start = std::numeric_limits<int64_t>::max();
    int64_t end = -1;
    int64_t bytes = 0;
    for (const HloValue* value : buffer.values()) {
      auto it = live_range->buffer_live_ranges().find(value);
      if (it == live_range->buffer_live_ranges().end()) continue;
      start = std::min(start, it->second.start);
      end = std::max(end, it->second.end);
      bytes = std::max(bytes, size_function(*value));
    }
    if (end < 0 || bytes == 0) continue;
    TF_RET_CHECK(start >= 0 && end + 1 < events.size());
    events[start] += bytes;
    events[end + 1] -= bytes;
  }
  MemoryProfile profile;
  int64_t live = 0;
  for (int64_t delta : events) {
    live += delta;
    profile.peak = std::max(profile.peak, live);
    profile.area += live;
  }
  return profile;
}

Decision ImprovesMemory(const MemoryProfile& candidate,
                        const MemoryProfile& original) {
  if (candidate.peak < original.peak ||
      (candidate.peak == original.peak && candidate.area < original.area)) {
    return Decision::Allow();
  }
  return Decision::Forbid("does not reduce peak memory or lifetime area");
}

struct Candidate {
  const HloBuffer* buffer;
  std::vector<HloInstruction*> users;
  int64_t insertion_position;
  long double priority;
};

// Parameters and escaping/aliased values do not have a lifetime owned by this
// computation. Follow HloValue uses, rather than instruction users, so tuple
// projections and bitcasts cannot hide a remaining reader.
std::vector<Candidate> FindCandidates(
    const HloComputation* computation, const Positions& positions,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function, int64_t min_buffer_bytes) {
  std::vector<Candidate> candidates;
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    if (buffer.values().size() != 1) continue;
    const HloValue* value = buffer.values().front();
    const HloInstruction* definition = value->defining_instruction();
    if (definition->parent() != computation ||
        definition->opcode() == HloOpcode::kParameter ||
        definition->opcode() == HloOpcode::kConstant ||
        !positions.contains(definition) ||
        HloLiveRange::BufferLivesOut(buffer, alias_analysis, computation)) {
      continue;
    }
    int64_t bytes = size_function(*value);
    if (bytes < min_buffer_bytes) continue;
    absl::flat_hash_set<HloInstruction*> seen;
    std::vector<HloInstruction*> users;
    bool local = true;
    for (const HloUse& use : value->GetUses()) {
      if (!positions.contains(use.instruction)) {
        local = false;
        break;
      }
      if (seen.insert(use.instruction).second) users.push_back(use.instruction);
    }
    if (!local || users.empty()) continue;
    std::sort(users.begin(), users.end(),
              [&](const HloInstruction* a, const HloInstruction* b) {
                return positions.at(a) < positions.at(b);
              });
    // Preserve the first consumer's progress on a branch. A sole consumer can
    // instead be moved directly after the definition.
    int64_t insertion = users.size() == 1 ? positions.at(definition) + 1
                                          : positions.at(users.front()) + 1;
    int64_t last_use = positions.at(users.back());
    if (last_use <= insertion) continue;
    candidates.push_back(
        {&buffer, std::move(users), insertion,
         static_cast<long double>(bytes) * (last_use - insertion)});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.priority != b.priority) return a.priority > b.priority;
              return a.buffer->id() < b.buffer->id();
            });
  return candidates;
}

Decision CanMove(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kCustomCall &&
      static_cast<const HloCustomCallInstruction*>(instruction)
              ->custom_call_schedule() != CustomCallSchedule::SCHEDULE_NONE) {
    return Decision::Forbid("custom call has an explicit scheduling policy");
  }
  if (instruction->HasSideEffect() || instruction->IsAsynchronous() ||
      HloDataflowAnalysis::IsAsynchronousOperationStart(
          instruction->opcode()) ||
      HloDataflowAnalysis::IsAsynchronousOperationDone(instruction->opcode()) ||
      hlo_query::IsCollectiveCommunicationOp(instruction->opcode()) ||
      instruction->opcode() == HloOpcode::kWhile ||
      instruction->opcode() == HloOpcode::kConditional ||
      instruction->opcode() == HloOpcode::kCall ||
      instruction->opcode() == HloOpcode::kParameter ||
      instruction == instruction->parent()->root_instruction()) {
    return Decision::Forbid("requires fixed ordering or owns a sub-schedule");
  }
  return Decision::Allow();
}

Decision CollectGroup(const Candidate& candidate, const Positions& positions,
                      int64_t max_group_size,
                      absl::flat_hash_set<HloInstruction*>& group) {
  std::vector<HloInstruction*> pending = candidate.users;
  while (!pending.empty()) {
    HloInstruction* instruction = pending.back();
    pending.pop_back();
    if (positions.at(instruction) < candidate.insertion_position ||
        group.contains(instruction))
      continue;
    if (CanMove(instruction).IsForbidden() || group.size() >= max_group_size) {
      return Decision::Forbid("consumer closure cannot be moved within budget");
    }
    group.insert(instruction);
    pending.insert(pending.end(), instruction->operands().begin(),
                   instruction->operands().end());
    pending.insert(pending.end(), instruction->control_predecessors().begin(),
                   instruction->control_predecessors().end());
  }
  return Decision::Allow();
}

HloInstructionSequence MoveGroup(
    const HloInstructionSequence& sequence,
    const absl::flat_hash_set<HloInstruction*>& group, int64_t insertion) {
  HloInstructionSequence result;
  for (int64_t i = 0; i < sequence.size(); ++i) {
    if (i == insertion) {
      // The original topological order also orders the prerequisite closure.
      for (HloInstruction* instruction : sequence.instructions()) {
        if (group.contains(instruction)) result.push_back(instruction);
      }
    }
    if (!group.contains(sequence.instructions()[i])) {
      result.push_back(sequence.instructions()[i]);
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<bool> HloEarlyBufferRelease::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    return absl::FailedPreconditionError(
        "Early buffer release needs a schedule");
  }
  TF_RET_CHECK(schedule_cost_ != nullptr);
  TF_RET_CHECK(options_.min_buffer_bytes >= 0 && options_.max_group_size > 0 &&
               options_.max_candidates > 0 && options_.max_iterations > 0 &&
               std::isfinite(options_.max_relative_slowdown) &&
               options_.max_relative_slowdown >= 0);
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        HloAliasAnalysis::Run(module, alias_info_));
  HloSchedule& schedule = module->schedule();
  ABSL_RETURN_IF_ERROR(schedule.Verify());
  TF_RET_CHECK(schedule.is_computation_scheduled(module->entry_computation()));
  ABSL_ASSIGN_OR_RETURN(
      MemoryProfile global,
      GetMemoryProfile(schedule, *alias_analysis, module->entry_computation(),
                       size_function_));
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!schedule.is_computation_scheduled(computation) ||
        computation->IsAsyncComputation())
      continue;
    // A schedule repair must not split or insert work into an annotated group.
    // Supporting these groups requires treating each group as an atomic unit.
    bool annotated = false;
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->frontend_attributes().map().contains(
              kXlaSchedulingGroupIdAttr))
        annotated = true;
    }
    if (annotated) continue;
    const double original_cost = schedule_cost_(
        computation, schedule.sequence(computation).instructions());
    if (!std::isfinite(original_cost) || original_cost < 0) continue;
    ABSL_ASSIGN_OR_RETURN(MemoryProfile current,
                          GetMemoryProfile(schedule, *alias_analysis,
                                           computation, size_function_));
    for (int64_t iteration = 0; iteration < options_.max_iterations;
         ++iteration) {
      HloInstructionSequence original = schedule.sequence(computation);
      Positions positions;
      for (int64_t i = 0; i < original.size(); ++i) {
        positions[original.instructions()[i]] = i;
      }
      std::vector<Candidate> candidates =
          FindCandidates(computation, positions, *alias_analysis,
                         size_function_, options_.min_buffer_bytes);
      int64_t evaluated = 0;
      bool accepted = false;
      for (const Candidate& candidate : candidates) {
        Candidate expanded = candidate;
        absl::flat_hash_set<HloInstruction*> group;
        while (evaluated < options_.max_candidates) {
          if (CollectGroup(expanded, positions, options_.max_group_size, group)
                  .IsForbidden() ||
              group.empty())
            break;
          ++evaluated;
          HloInstructionSequence proposed =
              MoveGroup(original, group, candidate.insertion_position);
          double cost = schedule_cost_(computation, proposed.instructions());
          if (!std::isfinite(cost) || cost < 0 ||
              cost > original_cost * (1 + options_.max_relative_slowdown))
            break;
          // Analyze a separate schedule so failed/rejected proposals never
          // leave the module with a tentative order.
          HloSchedule trial = schedule;
          trial.set_sequence(computation, proposed);
          ABSL_ASSIGN_OR_RETURN(MemoryProfile proposed_memory,
                                GetMemoryProfile(trial, *alias_analysis,
                                                 computation, size_function_));
          if (ImprovesMemory(proposed_memory, current).IsAllowed()) {
            ABSL_ASSIGN_OR_RETURN(
                MemoryProfile global_memory,
                GetMemoryProfile(trial, *alias_analysis,
                                 module->entry_computation(), size_function_));
            if (global_memory.peak <= global.peak) {
              VLOG(1) << "Early buffer release in " << computation->name()
                      << ": buffer " << candidate.buffer->id()
                      << ", group size " << group.size() << ", peak "
                      << current.peak << " -> " << proposed_memory.peak
                      << ", lifetime area " << current.area << " -> "
                      << proposed_memory.area << ", estimated time "
                      << original_cost << " -> " << cost;
              schedule.set_sequence(computation, std::move(proposed));
              current = proposed_memory;
              global = global_memory;
              changed = accepted = true;
              break;
            }
          }
          // Look through a temporarily expanding operation (e.g. a conversion)
          // by adding its next consumer, together with any missing
          // prerequisites. Choose in schedule order for deterministic, bounded
          // lookahead.
          HloInstruction* next = nullptr;
          for (HloInstruction* instruction : group) {
            for (HloInstruction* user : instruction->users()) {
              if (!group.contains(user) && CanMove(user).IsAllowed() &&
                  (next == nullptr ||
                   positions.at(user) < positions.at(next))) {
                next = user;
              }
            }
          }
          if (next == nullptr) break;
          expanded.users.push_back(next);
        }
        if (accepted || evaluated >= options_.max_candidates) break;
      }
      if (!accepted) break;
    }
  }
  ABSL_RETURN_IF_ERROR(schedule.Verify());
  return changed;
}

}  // namespace xla
