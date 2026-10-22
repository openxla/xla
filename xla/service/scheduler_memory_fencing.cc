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

#include "xla/service/scheduler_memory_fencing.h"

#include <algorithm>
#include <cstdint>
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
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

struct AsyncOperationWindow {
  HloInstruction* start;
  int64_t start_position;
  int64_t done_position;
};

// Per-computation state derived from the reference schedule.
struct ComputationFencingInfo {
  // Schedule position of every instruction in the computation.
  absl::flat_hash_map<const HloInstruction*, int64_t> position;
  // Async operation windows, ordered by the schedule position of their start
  // operations.
  std::vector<AsyncOperationWindow> windows;
  std::unique_ptr<HloReachabilityMap> reachability;
};

HloInstruction* FindAsyncDone(HloInstruction* async_start) {
  if (async_start->opcode() == HloOpcode::kAsyncStart) {
    return async_start->async_chain_done();
  }
  for (HloInstruction* user : async_start->users()) {
    if (HloDataflowAnalysis::IsAsynchronousOperationDone(user->opcode())) {
      return user;
    }
  }
  return nullptr;
}

ComputationFencingInfo BuildComputationFencingInfo(
    HloComputation* computation, const HloInstructionSequence& sequence) {
  ComputationFencingInfo info;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  info.position.reserve(instructions.size());
  for (int64_t i = 0; i < instructions.size(); ++i) {
    info.position[instructions[i]] = i;
  }
  for (int64_t i = 0; i < instructions.size(); ++i) {
    HloInstruction* instruction = instructions[i];
    if (!HloDataflowAnalysis::IsAsynchronousOperationStart(
            instruction->opcode())) {
      continue;
    }
    // The window closes at the corresponding done operation. If it cannot be
    // located in this sequence, treat the window as closing immediately.
    int64_t done_position = i;
    if (HloInstruction* async_done = FindAsyncDone(instruction)) {
      auto it = info.position.find(async_done);
      if (it != info.position.end()) {
        done_position = it->second;
      }
    }
    info.windows.push_back({instruction, i, done_position});
  }
  info.reachability = HloReachabilityMap::Build(computation);
  return info;
}

// The users of a fenced buffer and the schedule position of its last use in
// the reference schedule.
struct FencedUsers {
  // All distinct users of the buffer, each of which must be fenced: the
  // buffer stays live until every user has executed, so constraining only
  // one of them would not bound the buffer's live range.
  std::vector<HloInstruction*> users;
  int64_t last_use_position = -1;
};

// Returns the users of `buffer`, or an empty user set if the buffer must not
// be fenced: it has no users, is (partially) defined by a parameter, escapes
// its computation, or is not fully contained in one scheduled computation.
FencedUsers FindFencedUsers(const HloBuffer& buffer,
                            const HloComputation* computation,
                            const ComputationFencingInfo& info) {
  FencedUsers result;
  absl::flat_hash_set<HloInstruction*> seen_users;
  for (const HloValue* value : buffer.values()) {
    if (value->live_out_of_module()) {
      return FencedUsers();
    }
    HloInstruction* definition = value->defining_instruction();
    // The lifetime of parameter-backed buffers is not owned by this
    // computation; fencing them cannot release memory.
    if (definition->parent() != computation ||
        definition->opcode() == HloOpcode::kParameter) {
      return FencedUsers();
    }
    for (const HloPosition& hlo_position : value->positions()) {
      // A buffer that appears in the root escapes the computation (e.g. a
      // while-body output); its live range does not end at its last user.
      if (hlo_position.instruction->parent() != computation ||
          hlo_position.instruction == computation->root_instruction()) {
        return FencedUsers();
      }
    }
    for (const HloUse& use : value->GetUses()) {
      if (use.instruction->parent() != computation ||
          use.instruction == computation->root_instruction()) {
        return FencedUsers();
      }
      auto it = info.position.find(use.instruction);
      if (it == info.position.end()) {
        return FencedUsers();
      }
      if (seen_users.insert(use.instruction).second) {
        result.users.push_back(use.instruction);
      }
      result.last_use_position = std::max(result.last_use_position, it->second);
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<SchedulerMemoryFencing::BufferFences>>
SchedulerMemoryFencing::GetFences(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  if (!module->has_schedule()) {
    return absl::FailedPreconditionError(
        "SchedulerMemoryFencing requires a scheduled module.");
  }
  TF_RET_CHECK(slack_windows_ >= 0);
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        HloAliasAnalysis::Run(module, alias_info_));
  const HloSchedule& schedule = module->schedule();

  absl::flat_hash_set<const HloComputation*> fenceable_computations;
  for (const HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (schedule.is_computation_scheduled(computation)) {
      fenceable_computations.insert(computation);
    }
  }

  absl::flat_hash_map<const HloComputation*, ComputationFencingInfo>
      computation_infos;
  std::vector<BufferFences> result;

  for (const HloBuffer& buffer : alias_analysis->buffers()) {
    const int64_t buffer_size = shape_size_bytes_(buffer.values()[0]->shape());
    if (buffer_size < size_threshold_bytes_) {
      continue;
    }
    HloComputation* computation =
        buffer.values()[0]->defining_instruction()->parent();
    if (!fenceable_computations.contains(computation)) {
      continue;
    }

    auto info_it = computation_infos.find(computation);
    if (info_it == computation_infos.end()) {
      info_it = computation_infos
                    .emplace(computation,
                             BuildComputationFencingInfo(
                                 computation, schedule.sequence(computation)))
                    .first;
    }
    const ComputationFencingInfo& info = info_it->second;
    if (info.windows.empty()) {
      continue;
    }

    FencedUsers fenced = FindFencedUsers(buffer, computation, info);
    if (fenced.users.empty()) {
      continue;
    }
    const int64_t last_use_position = fenced.last_use_position;

    // Window index of the last use: the first window (in start order) that is
    // still open at, or opens after, the last use.
    int64_t window_index = 0;
    while (window_index < info.windows.size() &&
           info.windows[window_index].done_position <= last_use_position) {
      ++window_index;
    }

    int64_t target_index = window_index + slack_windows_;
    // Never fence backwards: the target start must come after the last use in
    // the reference schedule, otherwise the edge could contradict existing
    // dependencies. This keeps every edge forward in the schedule order and
    // therefore acyclic.
    while (target_index < info.windows.size() &&
           info.windows[target_index].start_position <= last_use_position) {
      ++target_index;
    }
    if (target_index >= info.windows.size()) {
      continue;
    }
    HloInstruction* target = info.windows[target_index].start;

    BufferFences buffer_fences{buffer_size, {}};
    // All users must finish before the buffer can be released. Only omit edges
    // already implied by the original graph, keeping the plan independent of
    // which thresholds the caller will evaluate.
    for (HloInstruction* user : fenced.users) {
      if (user != target && !info.reachability->IsReachable(user, target)) {
        buffer_fences.edges.emplace_back(user, target);
      }
    }
    if (!buffer_fences.edges.empty()) {
      result.push_back(std::move(buffer_fences));
    }
  }
  return result;
}

absl::StatusOr<bool> SchedulerMemoryFencing::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_ASSIGN_OR_RETURN(std::vector<BufferFences> buffers,
                        GetFences(module, execution_threads));
  absl::flat_hash_set<std::pair<HloInstruction*, HloInstruction*>> added_edges;
  for (const BufferFences& buffer : buffers) {
    for (const auto& [user, target] : buffer.edges) {
      if (added_edges.insert({user, target}).second) {
        ABSL_RETURN_IF_ERROR(user->AddControlDependencyTo(target));
      }
    }
  }
  VLOG(1) << "SchedulerMemoryFencing: added " << added_edges.size()
          << " control edges (threshold_bytes=" << size_threshold_bytes_
          << ", slack_windows=" << slack_windows_ << ").";
  return !added_edges.empty();
}

}  // namespace xla
