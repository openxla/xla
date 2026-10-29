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

#include "xla/service/post_lhs_memory_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
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
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/service/buffer_value.h"
#include "xla/service/decision.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/shape.h"

namespace xla {
namespace {

using Sequence = std::vector<HloInstruction*>;
using Statistics = LatencyHidingScheduler::SchedulerStatistics;

// These limits bound speculative scheduling work independently of the number
// of heap simulations. The original schedule remains available on exhaustion.
constexpr int64_t kMaxSchedulingWork = 10000000;
constexpr int64_t kMaxHierarchyDepth = 128;

int64_t DeviceBytes(const Shape& shape,
                    const HloCostAnalysis::ShapeSizeFunction& size) {
  if (shape.IsToken() || (shape.has_layout() && shape.layout().memory_space() ==
                                                    Layout::kHostMemorySpace)) {
    return 0;
  }
  return size(shape);
}

struct Region {
  int64_t begin;
  int64_t end;  // Exclusive, in the immutable input sequence.
  bool opaque = false;
  std::vector<int64_t> children;
};

// A heuristic only: each reference is charged once per physical buffer and
// instruction. Full module heap simulation, including callees and storage
// reuse, is the acceptance criterion. Counting an aliased definition as another
// reference can overestimate a live range but cannot admit a worse candidate.
struct BufferModel {
  std::vector<std::vector<int64_t>> references;
  std::vector<int64_t> sizes;
  std::vector<int64_t> counts;
  std::vector<bool> retained;
  std::vector<bool> external;

  BufferModel(const Sequence& sequence, const HloAliasAnalysis& analysis,
              const HloCostAnalysis::ShapeSizeFunction& size) {
    references.resize(sequence.size());
    absl::flat_hash_map<const HloInstruction*, int64_t> position;
    for (int64_t i = 0; i < sequence.size(); ++i) {
      position[sequence[i]] = i;
    }
    const HloComputation* computation = sequence.front()->parent();
    for (const HloBuffer& buffer : analysis.buffers()) {
      absl::flat_hash_set<int64_t> refs;
      bool keep = false;
      bool incoming = false;
      for (const HloValue* value : buffer.values()) {
        auto def = position.find(value->defining_instruction());
        if (def != position.end()) {
          refs.insert(def->second);
        }
        incoming |=
            def == position.end() ||
            value->defining_instruction()->opcode() == HloOpcode::kParameter;
        for (const HloUse& use : value->GetUses()) {
          auto it = position.find(use.instruction);
          if (it != position.end()) {
            refs.insert(it->second);
          } else {
            keep = true;
          }
        }
        for (const HloPosition& p : value->positions()) {
          if (p.instruction == computation->root_instruction()) {
            refs.insert(position.at(p.instruction));
            keep = true;
          }
        }
      }
      if (refs.empty()) {
        continue;
      }
      int64_t id = sizes.size();
      sizes.push_back(DeviceBytes(buffer.values().front()->shape(), size));
      counts.push_back(refs.size());
      retained.push_back(keep);
      external.push_back(incoming);
      for (int64_t i : refs) {
        references[i].push_back(id);
      }
    }
  }
};

struct MemoryState {
  std::vector<int64_t> remaining;
  std::vector<bool> live;
  int64_t bytes = 0;
  int64_t peak = 0;

  explicit MemoryState(const BufferModel& model)
      : remaining(model.counts), live(model.external) {
    for (int64_t i = 0; i < live.size(); ++i) {
      if (live[i]) {
        bytes += model.sizes[i];
      }
    }
    peak = bytes;
  }

  // Evaluate a child without copying the complete computation live set for
  // every ready sibling. Roll back only the buffers touched by that child.
  std::pair<int64_t, int64_t> Evaluate(const std::vector<int64_t>& instructions,
                                       const BufferModel& model) {
    const int64_t saved_bytes = bytes;
    const int64_t saved_peak = peak;
    std::vector<std::tuple<int64_t, int64_t, bool>> undo;
    for (int64_t instruction : instructions) {
      for (int64_t id : model.references[instruction]) {
        undo.emplace_back(id, remaining[id], live[id]);
      }
      Advance(instruction, model);
    }
    std::pair<int64_t, int64_t> result{peak, bytes};
    for (auto it = undo.rbegin(); it != undo.rend(); ++it) {
      auto [id, count, was_live] = *it;
      remaining[id] = count;
      live[id] = was_live;
    }
    bytes = saved_bytes;
    peak = saved_peak;
    return result;
  }

  void Advance(int64_t instruction, const BufferModel& model) {
    for (int64_t id : model.references[instruction]) {
      if (!live[id]) {
        live[id] = true;
        bytes += model.sizes[id];
      }
    }
    peak = std::max(peak, bytes);
    for (int64_t id : model.references[instruction]) {
      if (--remaining[id] == 0 && !model.retained[id]) {
        bytes -= model.sizes[id];
        live[id] = false;
      }
    }
  }
};

class RegionScheduler {
 public:
  RegionScheduler(Sequence sequence,
                  std::shared_ptr<const SchedulingContext> context)
      : original_(std::move(sequence)),
        context_(std::move(context)),
        graph_(&original_, context_) {
    for (int64_t i = 0; i < original_.size(); ++i) {
      position_[original_[i]] = i;
    }
  }

  Decision Initialize() {
    const AsyncTracker& tracker = *context_->GetAsyncTracker();
    for (int64_t i = 0; i < original_.size(); ++i) {
      HloInstruction* instruction = original_[i];
      if (HasSchedulingAnnotation(instruction)) {
        return Decision::Forbid(
            "scheduling annotation requires a joint grouping model");
      }
      if (instruction->opcode() == HloOpcode::kAsyncUpdate) {
        return Decision::Forbid(
            "async-update chains are not supported by the latency check");
      }
      const bool start = HloDataflowAnalysis::IsAsynchronousOperationStart(
          instruction->opcode());
      const bool done = HloDataflowAnalysis::IsAsynchronousOperationDone(
          instruction->opcode());
      if ((start && !tracker.IsSupportedAsyncStart(*instruction)) ||
          (done && !tracker.IsSupportedAsyncDone(*instruction))) {
        return Decision::Forbid(
            "async operation is not supported by the latency model");
      }
      if (done) {
        const HloInstruction* source = FindStart(instruction);
        auto it = position_.find(source);
        if (it == position_.end() || it->second >= i ||
            !tracker.IsSupportedAsyncStart(*source) ||
            ends_.contains(it->second)) {
          return Decision::Forbid("unmatched or cross-computation async chain");
        }
        ends_[it->second] = i;
        starts_[i] = it->second;
      }
      if (start || done) {
        endpoints_.push_back(instruction);
      }
    }
    for (int64_t i = 0; i < original_.size(); ++i) {
      if (tracker.IsSupportedAsyncStart(*original_[i]) && !ends_.contains(i)) {
        return Decision::Forbid("async start has no matching done");
      }
    }
    // Include implicit alias hazards and backend edges supplied by the same
    // scheduling graph used by LHS, not just HLO operands/control predecessors.
    for (int64_t i = 0; i < original_.size(); ++i) {
      for (const HloEdge& edge : graph_.GetNode(original_[i]).GetSuccessors()) {
        int64_t j = position_.at(&edge.Target().GetInstr());
        if (i >= j) {
          return Decision::Forbid("baseline violates a scheduling edge");
        }
        edges_.push_back({i, j});
      }
    }
    // Preserve communication/effect order globally, including synchronous
    // collectives that HLO does not necessarily classify as side effects.
    int64_t previous_ordered = -1;
    for (int64_t i = 0; i < original_.size(); ++i) {
      if (ends_.contains(i) || starts_.contains(i) ||
          IsOrdered(*original_[i])) {
        if (previous_ordered >= 0) {
          edges_.push_back({previous_ordered, i});
        }
        previous_ordered = i;
      }
      if (original_[i]->opcode() == HloOpcode::kCustomCall &&
          static_cast<const HloCustomCallInstruction*>(original_[i])
                  ->custom_call_schedule() !=
              CustomCallSchedule::SCHEDULE_NONE) {
        // A backend early/late custom call is a boundary even if it has no
        // explicit data or control dependence on an adjacent instruction.
        for (int64_t j = 0; j < i; ++j) {
          edges_.push_back({j, i});
        }
        for (int64_t j = i + 1; j < original_.size(); ++j) {
          edges_.push_back({i, j});
        }
      }
      if (edges_.size() > kMaxSchedulingWork) {
        return Decision::Forbid("constraint budget exhausted");
      }
    }
    root_ = Build(0, original_.size(), false, 0);
    if (root_ < 0) {
      return Decision::Forbid("hierarchy depth limit");
    }
    return Decision::Allow();
  }

  Sequence Candidate(const BufferModel& model, bool prefer_release) {
    // Build() visits children first, so this is a bottom-up traversal. A child
    // order is provisional until the full candidate has passed validation.
    int64_t scored = 0;
    for (int64_t r = 0; r < regions_.size(); ++r) {
      Region& region = regions_[r];
      if (region.opaque || region.children.size() < 2) {
        continue;
      }
      if (scored >= kMaxSchedulingWork) {
        break;
      }
      Sequence current = Flatten();
      MemoryState state(model);
      if (r != root_) {
        HloInstruction* first = original_[region.begin];
        for (HloInstruction* instruction : current) {
          if (instruction == first) {
            break;
          }
          state.Advance(position_.at(instruction), model);
        }
        state.Advance(region.begin, model);  // Async start.
      }
      Optimize(r, model, prefer_release, state, scored);
    }
    if (scored >= kMaxSchedulingWork) {
      VLOG(2) << "Post-LHS scheduling work budget exhausted in "
              << original_.front()->parent()->name();
    }
    return Flatten();
  }

  Decision Verify(const Sequence& candidate) const {
    if (candidate.size() != original_.size()) {
      return Decision::Forbid("instruction count changed");
    }
    absl::flat_hash_map<const HloInstruction*, int64_t> positions;
    for (int64_t i = 0; i < candidate.size(); ++i) {
      if (!position_.contains(candidate[i]) ||
          !positions.emplace(candidate[i], i).second) {
        return Decision::Forbid("instruction ownership changed");
      }
    }
    for (const auto& [a, b] : edges_) {
      if (positions.at(original_[a]) >= positions.at(original_[b])) {
        return Decision::Forbid("scheduling dependency changed");
      }
    }
    Sequence events;
    absl::flat_hash_set<const HloInstruction*> endpoint_set(endpoints_.begin(),
                                                            endpoints_.end());
    for (HloInstruction* instruction : candidate) {
      if (endpoint_set.contains(instruction)) {
        events.push_back(instruction);
      }
    }
    if (events != endpoints_) {
      return Decision::Forbid("async event order changed");
    }
    // Contiguity, fixed endpoints, and immutable ownership imply identical
    // enclosing async operations. Also verify the internal order of opaque
    // crossing composites. This costs O(instructions * hierarchy depth),
    // rather than comparing every instruction against every async interval.
    for (const Region& region : regions_) {
      int64_t lo = candidate.size();
      int64_t hi = -1;
      for (int64_t i = region.begin; i < region.end; ++i) {
        lo = std::min(lo, positions.at(original_[i]));
        hi = std::max(hi, positions.at(original_[i]));
      }
      if (hi - lo + 1 != region.end - region.begin) {
        return Decision::Forbid("child block is no longer contiguous");
      }
      if (region.opaque) {
        for (int64_t i = region.begin; i < region.end; ++i) {
          if (candidate[lo + i - region.begin] != original_[i]) {
            return Decision::Forbid("opaque block order changed");
          }
        }
      } else if (&region != &regions_[root_] &&
                 (candidate[lo] != original_[region.begin] ||
                  candidate[hi] != original_[region.end - 1])) {
        return Decision::Forbid("async block endpoints moved");
      }
    }
    return Decision::Allow();
  }

 private:
  bool IsOrdered(const HloInstruction& instruction) {
    if (instruction.HasSideEffectNoRecurse() ||
        hlo_query::IsCollectiveCommunicationOp(instruction.opcode()) ||
        instruction.IsAsynchronous()) {
      return true;
    }
    for (const HloComputation* callee : instruction.called_computations()) {
      auto it = ordered_computations_.find(callee);
      if (it == ordered_computations_.end()) {
        bool ordered = false;
        for (const HloInstruction* child : callee->instructions()) {
          ordered |= IsOrdered(*child);
        }
        it = ordered_computations_.emplace(callee, ordered).first;
      }
      if (it->second) {
        return true;
      }
    }
    return false;
  }

  int64_t Build(int64_t begin, int64_t end, bool async, int64_t depth) {
    if (depth > kMaxHierarchyDepth) {
      return -1;
    }
    Region region{begin, end};
    int64_t limit = async ? end - 1 : end;
    for (int64_t i = async ? begin + 1 : begin; i < limit;) {
      auto it = ends_.find(i);
      if (it == ends_.end()) {
        int64_t id = regions_.size();
        regions_.push_back(Region{i, i + 1, true});
        region.children.push_back(id);
        ++i;
        continue;
      }
      int64_t last = it->second;
      bool crossing = false;
      for (int64_t j = i + 1; j < last; ++j) {
        auto nested = ends_.find(j);
        if (nested != ends_.end() && nested->second > last) {
          last = nested->second;
          crossing = true;
        }
      }
      if (last >= limit) {
        return -1;
      }
      int64_t id;
      if (crossing) {
        id = regions_.size();
        regions_.push_back(Region{i, last + 1, true});
      } else {
        id = Build(i, last + 1, true, depth + 1);
        if (id < 0) {
          return -1;
        }
      }
      region.children.push_back(id);
      i = last + 1;
    }
    int64_t result = regions_.size();
    regions_.push_back(std::move(region));
    return result;
  }

  void Append(int64_t id, Sequence& sequence) const {
    const Region& region = regions_[id];
    if (region.opaque) {
      sequence.insert(sequence.end(), original_.begin() + region.begin,
                      original_.begin() + region.end);
      return;
    }
    if (id != root_) {
      sequence.push_back(original_[region.begin]);
    }
    for (int64_t child : region.children) {
      Append(child, sequence);
    }
    if (id != root_) {
      sequence.push_back(original_[region.end - 1]);
    }
  }

  Sequence Flatten() const {
    Sequence sequence;
    sequence.reserve(original_.size());
    Append(root_, sequence);
    return sequence;
  }

  void Optimize(int64_t id, const BufferModel& model, bool prefer_release,
                MemoryState state, int64_t& scored) {
    Region& region = regions_[id];
    int64_t count = region.children.size();
    std::vector<int64_t> owner(original_.size(), -1);
    std::vector<std::vector<int64_t>> contents(count);
    std::vector<int64_t> work(count, 0);
    for (int64_t c = 0; c < count; ++c) {
      const Region& child = regions_[region.children[c]];
      for (int64_t i = child.begin; i < child.end; ++i) {
        owner[i] = c;
      }
      Sequence instructions;
      Append(region.children[c], instructions);
      for (HloInstruction* instruction : instructions) {
        int64_t index = position_.at(instruction);
        contents[c].push_back(index);
        work[c] += 1 + model.references[index].size();
      }
    }
    std::vector<absl::flat_hash_set<int64_t>> successors(count);
    std::vector<int64_t> indegree(count, 0);
    auto edge = [&](int64_t a, int64_t b) {
      if (a >= 0 && b >= 0 && a != b && successors[a].insert(b).second) {
        ++indegree[b];
      }
    };
    for (const auto& [a, b] : edges_) {
      edge(owner[a], owner[b]);
    }
    std::vector<int64_t> order;
    std::vector<bool> scheduled(count, false);
    while (order.size() < count) {
      int64_t best = -1;
      std::tuple<int64_t, int64_t, int64_t> best_score;
      for (int64_t c = 0; c < count; ++c) {
        if (++scored > kMaxSchedulingWork) {
          return;
        }
        if (scheduled[c] || indegree[c] != 0) {
          continue;
        }
        scored += work[c];
        if (scored > kMaxSchedulingWork) {
          return;
        }
        auto [peak, bytes] = state.Evaluate(contents[c], model);
        auto score = prefer_release
                         ? std::make_tuple(bytes, peak,
                                           regions_[region.children[c]].begin)
                         : std::make_tuple(peak, bytes,
                                           regions_[region.children[c]].begin);
        if (best < 0 || score < best_score) {
          best = c;
          best_score = score;
        }
      }
      if (best < 0) {
        return;  // Preserve the valid original order on a cycle.
      }
      for (int64_t instruction : contents[best]) {
        state.Advance(instruction, model);
      }
      order.push_back(region.children[best]);
      scheduled[best] = true;
      for (int64_t successor : successors[best]) {
        --indegree[successor];
      }
    }
    region.children = std::move(order);
  }

  Sequence original_;
  std::shared_ptr<const SchedulingContext> context_;
  HloScheduleGraph graph_;
  absl::flat_hash_map<const HloInstruction*, int64_t> position_;
  absl::flat_hash_map<int64_t, int64_t> ends_;
  absl::flat_hash_map<int64_t, int64_t> starts_;
  Sequence endpoints_;
  absl::flat_hash_map<const HloComputation*, bool> ordered_computations_;
  std::vector<std::pair<int64_t, int64_t>> edges_;
  std::vector<Region> regions_;
  int64_t root_ = -1;
};

Decision CheckTiming(const Statistics& baseline, const Statistics& candidate) {
  auto nonincreasing = [](double before, double after) {
    return std::isfinite(before) && std::isfinite(after) &&
           after <= before + 1e-9 * std::max(1.0, std::abs(before));
  };
  if (!nonincreasing(baseline.total_cycles, candidate.total_cycles) ||
      !nonincreasing(baseline.GetTotalWastedCycles(),
                     candidate.GetTotalWastedCycles())) {
    return Decision::Forbid(
        "estimated cycles or exposed async latency increased");
  }
  return Decision::Allow();
}

}  // namespace

absl::StatusOr<bool> PostLhsMemoryScheduler::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    return absl::FailedPreconditionError(
        "PostLhsMemoryScheduler requires an LHS schedule");
  }
  if (max_candidate_evaluations_ <= 0) {
    return false;
  }
  ABSL_RETURN_IF_ERROR(module->schedule().Verify());
  // Alias analysis belongs to this pass, not to the preceding LHS invocation.
  auto context = std::make_shared<SchedulingContext>(
      module, scheduling_context_->GetLatencyEstimator(),
      scheduling_context_->GetAsyncTracker(),
      scheduling_context_->GetAliasInfo(),
      scheduling_context_->GetShapeSizeBytes());
  std::shared_ptr<const HloAliasAnalysis> aliases = context->GetAliasAnalysis();
  BufferValue::SizeFunction size = [&](const BufferValue& buffer) {
    return DeviceBytes(buffer.shape(), context->GetShapeSizeBytes());
  };
  HloSchedule best = module->schedule();
  ABSL_ASSIGN_OR_RETURN(int64_t best_peak,
                        HeapSimulator::MinimumMemoryForModule(
                            best, *aliases, context->GetAliasInfo(), &size));
  const int64_t original_peak = best_peak;
  int64_t evaluations = 0;
  bool changed = false;
  // Each computation is optimized once per priority variant. Both recursive
  // child orders and their parent's order are evaluated together.
  for (bool prefer_release : {false, true}) {
    for (HloComputation* computation :
         module->MakeComputationPostOrder(execution_threads)) {
      if (evaluations >= max_candidate_evaluations_) {
        break;
      }
      if (computation->IsFusionComputation() ||
          !best.is_computation_scheduled(computation)) {
        continue;
      }
      const Sequence original = best.sequence(computation).instructions();
      if (original.empty()) {
        continue;
      }
      RegionScheduler scheduler(original, context);
      Decision supported = scheduler.Initialize();
      if (supported.IsForbidden()) {
        VLOG(2) << name() << ": " << computation->name() << ": "
                << supported.Explain();
        continue;
      }
      BufferModel model(original, *aliases, context->GetShapeSizeBytes());
      Sequence candidate = scheduler.Candidate(model, prefer_release);
      if (candidate == original) {
        continue;
      }
      Decision valid = scheduler.Verify(candidate);
      if (valid.IsForbidden()) {
        VLOG(2) << name() << ": " << computation->name() << ": "
                << valid.Explain();
        continue;
      }
      HloSchedule trial = best;
      trial.set_sequence(computation, HloInstructionSequence(candidate));
      ABSL_RETURN_IF_ERROR(trial.Verify());
      ++evaluations;
      ABSL_ASSIGN_OR_RETURN(
          int64_t peak, HeapSimulator::MinimumMemoryForModule(
                            trial, *aliases, context->GetAliasInfo(), &size));
      if (peak >= best_peak) {
        continue;
      }
      Statistics before = LatencyHidingScheduler::LatencyHidingStatistics(
          computation, original, context);
      Statistics after = LatencyHidingScheduler::LatencyHidingStatistics(
          computation, candidate, context);
      Decision timing = CheckTiming(before, after);
      if (timing.IsForbidden()) {
        VLOG(2) << name() << ": " << computation->name() << ": "
                << timing.Explain();
        continue;
      }
      VLOG(1) << name() << ": " << computation->name() << ": peak " << best_peak
              << " -> " << peak;
      best = std::move(trial);
      best_peak = peak;
      changed = true;
    }
  }
  if (changed) {
    ABSL_RETURN_IF_ERROR(module->set_schedule(std::move(best)));
  }
  VLOG(1) << name() << ": " << evaluations << " candidate evaluations; peak "
          << original_peak << " -> " << best_peak;
  return changed;
}

}  // namespace xla
