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

#include "xla/service/scheduler_memory_fencing_autotuner.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/scheduler_memory_fencing.h"

namespace xla {
namespace {

using Edge = std::pair<HloInstruction*, HloInstruction*>;
using Edges = std::vector<Edge>;

Edges GetEdges(absl::Span<const SchedulerMemoryFencing::BufferFences> buffers,
               int64_t threshold) {
  Edges edges;
  if (threshold < 0) return edges;
  for (const auto& buffer : buffers) {
    if (buffer.size_bytes >= threshold) {
      edges.insert(edges.end(), buffer.edges.begin(), buffer.edges.end());
    }
  }
  // Stable ordering independent of pointer addresses and alias-analysis order.
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    return std::make_pair(a.first->unique_id(), a.second->unique_id()) <
           std::make_pair(b.first->unique_id(), b.second->unique_id());
  });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

using Score = std::tuple<int64_t, double, int64_t>;
struct Candidate {
  int64_t threshold;
  Score score;
  Edges edges;
};

}  // namespace

absl::Status AutotuneSchedulerMemoryFencing(
    HloModule* module,
    absl::Span<const SchedulerMemoryFencing::BufferFences> buffers,
    uint64_t memory_limit,
    absl::FunctionRef<absl::StatusOr<SchedulerMemoryFencingCost>(HloModule*)>
        schedule) {
  if (!module->has_schedule()) {
    return absl::FailedPreconditionError("Fencing tuning requires a schedule");
  }
  ABSL_RETURN_IF_ERROR(module->schedule().Verify());
  // Module cloning intentionally omits unreachable computations. Do not tune
  // dead buffers or let them influence the candidate size-class distribution.
  auto computations =
      module->entry_computation()->MakeEmbeddedComputationsList();
  computations.push_back(module->entry_computation());
  absl::flat_hash_set<const HloComputation*> reachable(computations.begin(),
                                                       computations.end());
  std::vector<SchedulerMemoryFencing::BufferFences> reachable_buffers;
  for (const auto& buffer : buffers) {
    if (!buffer.edges.empty() &&
        reachable.contains(buffer.edges.front().first->parent())) {
      reachable_buffers.push_back(buffer);
    }
  }
  buffers = reachable_buffers;
  if (buffers.empty() || memory_limit == 0) {
    ABSL_RETURN_IF_ERROR(schedule(module).status());
    return module->schedule().Verify();
  }
  const int64_t limit =
      std::min<uint64_t>(memory_limit, std::numeric_limits<int64_t>::max());
  constexpr int64_t kMiB = 1024 * 1024;
  std::vector<Candidate> evaluated;
  std::set<int64_t> tried_thresholds;
  int best = 0;
  std::optional<int> best_fenced;
  auto evaluate = [&](int64_t threshold) -> absl::Status {
    tried_thresholds.insert(threshold);
    Edges edges = GetEdges(buffers, threshold);
    for (const Candidate& candidate : evaluated) {
      if (edges == candidate.edges) return absl::OkStatus();
    }
    auto [trial, context] = module->CloneWithContext("");
    for (const auto& [from, to] : edges) {
      HloInstruction* cloned_from = context->FindInstruction(from);
      HloInstruction* cloned_to = context->FindInstruction(to);
      // Clone only contains computations reachable from the entry.
      if (cloned_from == nullptr || cloned_to == nullptr) {
        return absl::InvalidArgumentError(
            "Fencing candidate refers to an unreachable computation");
      }
      ABSL_RETURN_IF_ERROR(cloned_from->AddControlDependencyTo(cloned_to));
    }
    ABSL_ASSIGN_OR_RETURN(SchedulerMemoryFencingCost cost,
                          schedule(trial.get()));
    ABSL_RETURN_IF_ERROR(trial->schedule().Verify());
    const int index = evaluated.size();
    evaluated.push_back({threshold,
                         {cost.peak_memory_bytes, cost.estimated_cycles,
                          static_cast<int64_t>(edges.size())},
                         std::move(edges)});
    if (evaluated[index].score < evaluated[best].score) best = index;
    if (!evaluated[index].edges.empty() &&
        (!best_fenced ||
         evaluated[index].score < evaluated[*best_fenced].score)) {
      best_fenced = index;
    }
    VLOG(1) << "SchedulerMemoryFencing candidate: threshold_bytes=" << threshold
            << ", peak_memory_bytes=" << cost.peak_memory_bytes
            << ", estimated_cycles=" << cost.estimated_cycles
            << ", edges=" << evaluated[index].edges.size();
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(evaluate(-1));
  for (int64_t anchor :
       {limit / 100, 1024 * kMiB, 512 * kMiB, 256 * kMiB, 128 * kMiB}) {
    ABSL_RETURN_IF_ERROR(evaluate(std::clamp<int64_t>(anchor, 1, limit)));
  }

  // Fence sets are nested as the threshold falls, even though their resulting
  // memory costs need not be monotonic. An edge activates at the largest size
  // of any buffer that contributes it. Counts therefore identify duplicate
  // sets without constructing and sorting a full edge set for every size class.
  std::map<std::pair<int64_t, int64_t>, int64_t> edge_sizes;
  for (const auto& buffer : buffers) {
    for (const auto& [from, to] : buffer.edges) {
      int64_t& size = edge_sizes[{from->unique_id(), to->unique_id()}];
      size = std::max(size, buffer.size_bytes);
    }
  }
  std::vector<int64_t> activation_sizes;
  for (const auto& [edge, size] : edge_sizes) activation_sizes.push_back(size);
  std::sort(activation_sizes.begin(), activation_sizes.end());

  // Aggregate bytes are a sampling priority, not a prediction of peak savings.
  std::map<int64_t, uint64_t> size_classes;
  for (const auto& buffer : buffers) {
    if (buffer.size_bytes <= 0 || buffer.size_bytes > limit ||
        buffer.edges.empty())
      continue;
    uint64_t& bytes = size_classes[buffer.size_bytes];
    bytes += std::min<uint64_t>(buffer.size_bytes,
                                std::numeric_limits<uint64_t>::max() - bytes);
  }
  // Refine below and above the best fenced anchor. At an endpoint this also
  // explores beyond the anchors. If an interval has no distinct untested fence
  // set, spend its trial on the largest remaining size class anywhere.
  const int64_t center =
      best_fenced ? evaluated[*best_fenced].threshold : limit;
  int64_t lower = 0;
  int64_t upper = limit;
  for (int64_t threshold : tried_thresholds) {
    if (threshold > 0 && threshold < center) lower = std::max(lower, threshold);
    if (threshold > center) upper = std::min(upper, threshold);
  }
  for (bool below : {true, false}) {
    std::optional<int64_t> selected;
    std::tuple<bool, uint64_t, int64_t> selected_priority;
    for (const auto& [size, bytes] : size_classes) {
      if (tried_thresholds.find(size) != tried_thresholds.end()) continue;
      const auto edge_count = activation_sizes.end() -
                              std::lower_bound(activation_sizes.begin(),
                                               activation_sizes.end(), size);
      if (std::any_of(evaluated.begin(), evaluated.end(),
                      [&](const Candidate& c) {
                        return c.edges.size() == edge_count;
                      })) {
        continue;
      }
      bool near_winner = below ? size > lower && size < center
                               : size > center && size <= upper;
      const auto priority = std::make_tuple(near_winner, bytes, size);
      if (!selected || priority > selected_priority) {
        selected = size;
        selected_priority = priority;
      }
    }
    if (selected) ABSL_RETURN_IF_ERROR(evaluate(*selected));
  }

  const Candidate& winner = evaluated[best];
  VLOG(1) << "SchedulerMemoryFencing selected: threshold_bytes="
          << winner.threshold
          << ", peak_memory_bytes=" << std::get<0>(winner.score)
          << ", candidates=" << evaluated.size();
  // Replay on the original graph so original pointers remain valid and target
  // scheduler state/backend-config mutations are retained as well as the order.
  for (const auto& [from, to] : winner.edges) {
    ABSL_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
  }
  ABSL_RETURN_IF_ERROR(schedule(module).status());
  return module->schedule().Verify();
}

}  // namespace xla
