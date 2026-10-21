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
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
  // The user scheduled last in the reference schedule; it releases the buffer.
  HloInstruction* last_user = nullptr;
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
      if (it->second > result.last_use_position) {
        result.last_use_position = it->second;
        result.last_user = use.instruction;
      }
    }
  }
  return result;
}

// A large buffer's allocation and release events in the reference schedule.
struct LargeBuffer {
  const HloBuffer* buffer;
  int64_t size;
  // The instruction defining the buffer's earliest value, i.e. the point at
  // which the buffer becomes live.
  HloInstruction* producer;
  int64_t producer_position;
  // Whether `producer` is an async operation: such buffers are never given a
  // producer fence, so collective overlap stays unconstrained.
  bool async_producer;
  // The last user, i.e. the point at which the buffer is released.
  HloInstruction* last_user;
  int64_t last_use_position;
};

// Returns the producer of `buffer`, or nullptr if any defining instruction is
// not in the reference schedule of `info`.
HloInstruction* FindProducer(const HloBuffer& buffer,
                             const ComputationFencingInfo& info,
                             int64_t* position) {
  HloInstruction* producer = nullptr;
  *position = std::numeric_limits<int64_t>::max();
  for (const HloValue* value : buffer.values()) {
    auto it = info.position.find(value->defining_instruction());
    if (it == info.position.end()) {
      return nullptr;
    }
    if (it->second < *position) {
      *position = it->second;
      producer = value->defining_instruction();
    }
  }
  return producer;
}

bool IsAsyncProducer(const HloInstruction* instruction) {
  return HloDataflowAnalysis::IsAsynchronousOperationStart(
             instruction->opcode()) ||
         HloDataflowAnalysis::IsAsynchronousOperationDone(
             instruction->opcode());
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

  // Bounds how long the users of a buffer may be deferred: every user must
  // run before the async start `slack_windows_` windows after the buffer's
  // last-use window in the reference schedule.
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

  // Bounds how much earlier than in the reference schedule a large buffer may
  // be allocated: its producer must run after the release (last use) of the
  // `producer_slack_ + 1`-th most recent large buffer released before the
  // producer in the reference schedule. This keeps the relative order of
  // large allocation/release events close to the reference schedule, which
  // is what bounds their simultaneous liveness. Async windows are not used as
  // anchors because the scheduler is free to move async starts.
  auto fence_producers =
      [&](const std::vector<LargeBuffer>& buffers,
          const ComputationFencingInfo& info) -> absl::Status {
    std::vector<const LargeBuffer*> releases;
    releases.reserve(buffers.size());
    for (const LargeBuffer& large : buffers) {
      releases.push_back(&large);
    }
    absl::c_sort(releases, [](const LargeBuffer* a, const LargeBuffer* b) {
      return a->last_use_position < b->last_use_position;
    });
    for (const LargeBuffer& large : buffers) {
      if (large.async_producer) {
        continue;
      }
      // Releases strictly before the producer in the reference schedule; the
      // buffer's own release is after its producer and never qualifies.
      auto end =
          absl::c_lower_bound(releases, large.producer_position,
                              [](const LargeBuffer* release, int64_t position) {
                                return release->last_use_position < position;
                              });
      const int64_t anchor_index =
          static_cast<int64_t>(end - releases.begin()) - 1 - producer_slack_;
      if (anchor_index < 0) {
        continue;
      }
      const LargeBuffer* anchor = releases[anchor_index];
      // The anchor's last use precedes the producer in the reference
      // schedule, so the edge points forward and cannot create a cycle.
      ABSL_RETURN_IF_ERROR(add_edge(anchor->last_user, large.producer, info,
                                    *large.buffer, large.size, "producer"));
    }
    return absl::OkStatus();
  };

  absl::flat_hash_map<const HloComputation*, std::vector<LargeBuffer>>
      large_buffers;
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

    FencedUsers fenced = FindFencedUsers(buffer, computation, info);
    if (fenced.users.empty()) {
      continue;
    }
    if (!info.windows.empty()) {
      ABSL_RETURN_IF_ERROR(fence_users(buffer, buffer_size, fenced, info));
    }
    if (producer_slack_ >= 0) {
      int64_t producer_position;
      HloInstruction* producer = FindProducer(buffer, info, &producer_position);
      if (producer != nullptr) {
        large_buffers[computation].push_back(
            {&buffer, buffer_size, producer, producer_position,
             IsAsyncProducer(producer), fenced.last_user,
             fenced.last_use_position});
      }
    }
  }
  for (const auto& [computation, buffers] : large_buffers) {
    ABSL_RETURN_IF_ERROR(
        fence_producers(buffers, computation_infos.at(computation)));
  }

  VLOG(1) << "SchedulerMemoryFencing: considered " << buffers_considered
          << " buffers >= " << size_threshold_bytes_ << " bytes, added "
          << edges_added << " control edges (slack_windows=" << slack_windows_
          << ", producer_slack=" << producer_slack_ << ").";
  return edges_added > 0;
}

}  // namespace xla
