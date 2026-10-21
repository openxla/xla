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

absl::StatusOr<bool> SchedulerMemoryFencing::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
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
  absl::flat_hash_set<std::pair<HloInstruction*, HloInstruction*>> added_edges;
  int64_t buffers_considered = 0;
  int64_t edges_added = 0;

  // Adds `from` -> `to` unless it is redundant or already added.
  auto add_edge = [&](HloInstruction* from, HloInstruction* to,
                      const ComputationFencingInfo& info,
                      const HloBuffer& buffer, int64_t buffer_size,
                      absl::string_view kind) -> absl::Status {
    if (from == to || !added_edges.insert({from, to}).second) {
      return absl::OkStatus();
    }
    // Skip edges that are already implied by the dependency graph. The
    // reachability map is not updated for added edges; a stale map can only
    // let a redundant (but still forward and acyclic) edge through.
    if (info.reachability->IsReachable(from, to)) {
      return absl::OkStatus();
    }
    ABSL_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
    ++edges_added;
    VLOG(2) << "Fenced buffer " << buffer.ToString() << " (" << buffer_size
            << " bytes): " << kind << " control edge " << from->name() << " -> "
            << to->name();
    return absl::OkStatus();
  };

  auto fence_users = [&](const HloBuffer& buffer, int64_t buffer_size,
                         const FencedUsers& fenced,
                         const ComputationFencingInfo& info) -> absl::Status {
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
      return absl::OkStatus();
    }
    HloInstruction* target = info.windows[target_index].start;

    // Fence every user: the buffer stays live until all of its users have
    // executed, so a single fenced user would not bound the live range — an
    // unfenced sibling could still be deferred arbitrarily far.
    for (HloInstruction* user : fenced.users) {
      ABSL_RETURN_IF_ERROR(
          add_edge(user, target, info, buffer, buffer_size, "user"));
    }
    return absl::OkStatus();
  };

  // Bounds how much earlier than in the reference schedule LHS may materialize
  // the buffer: every producer depends on the async start that is
  // `producer_slack_windows_` windows before the last window starting before
  // the producer in the reference schedule.
  auto fence_producers =
      [&](const HloBuffer& buffer, int64_t buffer_size,
          const ComputationFencingInfo& info) -> absl::Status {
    for (const HloValue* value : buffer.values()) {
      HloInstruction* producer = value->defining_instruction();
      auto position_it = info.position.find(producer);
      if (position_it == info.position.end()) {
        continue;
      }
      const int64_t producer_position = position_it->second;
      // Index of the last window that starts before the producer.
      int64_t window_index = -1;
      while (window_index + 1 < info.windows.size() &&
             info.windows[window_index + 1].start_position <
                 producer_position) {
        ++window_index;
      }
      const int64_t anchor_index = window_index - producer_slack_windows_;
      if (anchor_index < 0) {
        continue;
      }
      // The anchor starts before the producer in the reference schedule, so
      // the edge points forward and cannot create a cycle.
      ABSL_RETURN_IF_ERROR(add_edge(info.windows[anchor_index].start, producer,
                                    info, buffer, buffer_size, "producer"));
    }
    return absl::OkStatus();
  };

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
    ++buffers_considered;

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
    ABSL_RETURN_IF_ERROR(fence_users(buffer, buffer_size, fenced, info));
    if (producer_slack_windows_ >= 0) {
      ABSL_RETURN_IF_ERROR(fence_producers(buffer, buffer_size, info));
    }
  }

  VLOG(1) << "SchedulerMemoryFencing: considered " << buffers_considered
          << " buffers >= " << size_threshold_bytes_ << " bytes, added "
          << edges_added << " control edges (slack_windows=" << slack_windows_
          << ", producer_slack_windows=" << producer_slack_windows_ << ").";
  return edges_added > 0;
}

}  // namespace xla
