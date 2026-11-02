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

#ifndef XLA_SERVICE_ASYNC_TRACKER_H_
#define XLA_SERVICE_ASYNC_TRACKER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_estimator.h"
#include "xla/side_effect_util.h"
#include "xla/xla.pb.h"

namespace xla {

inline constexpr int64_t kInvalidAnnotation = -1;

// Returns the matching start instruction for an async-done instruction.
// Supports both canonical async ops (e.g. kAsyncDone, kAllGatherDone) and
// legacy async ops (kCopyDone, kSendDone, kRecvDone).
const HloInstruction* FindStart(const HloInstruction* done);
HloInstruction* FindStart(HloInstruction* done);

// Returns the matching done instruction for an async-start instruction.
// Supports both canonical async ops (e.g. kAsyncStart, kAllGatherStart) and
// legacy async ops (kCopyStart, kSend, kRecv).
const HloInstruction* FindDone(const HloInstruction* start);
HloInstruction* FindDone(HloInstruction* start);

inline bool IsNopInstruction(HloOpcode op, const HloInstruction& hlo) {
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         op == HloOpcode::kBroadcast || op == HloOpcode::kIota ||
         hlo.IsEffectiveBitcast(op) ||
         (op == HloOpcode::kTuple && hlo.user_count() == 1 &&
          hlo.users().front()->opcode() == HloOpcode::kWhile);
}

class HloGraphNode;

enum class ResourceType {
  kNoResource = 0,
  kAllToAll,
  kAllGather,
  kAllReduce,
  kCollectivePermute,
  kCopy,
  kReduceScatter,
  kSendRecv,
  kSendHost,
  kRecvHost,
  kCollectiveBroadcast,
  kNumResources,
  kRaggedAllToAll,
  kTargetDefinedResourceTypeBegin,
};

enum class ResourceUsageType {
  kNoResource,
  kResourceOccupy,
  kResourceRelease,
};

enum class ResourceHazardType {
  kShareable = 0,
  kSerial = 1,
  // The following hazard type represents the resources that are used by the
  // async ops and should be released right after the estimated time cost has
  // past. This hazard type is useful to prevent increasing such ops' overlaps
  // more than necessary.
  kNonextendable = 2,
  // Ops holding this resource can only have their latency/cost covered by
  // ops that are valuable for selective overlap.
  kSelective = 3,
  kUnshareable = 4,
  // An in-order resource (e.g. hardware FIFO command queue) where operations
  // are processed sequentially. Instructions calling a synchronous nested
  // computation using this resource cannot be scheduled while any outer
  // asynchronous operation on this resource is in flight.
  kInOrder = 5,
};

template <typename T, typename = typename std::enable_if_t<std::is_enum_v<T>>>
constexpr int64_t ResourceTypeToIndex(T resource_type) {
  return static_cast<int64_t>(resource_type);
}

constexpr int64_t ResourceUsageTypeToIndex(
    ResourceUsageType resource_usage_type) {
  return static_cast<int64_t>(resource_usage_type);
}

using ResourcePair = std::pair<int64_t, ResourceUsageType>;
using ResourcesVector = absl::InlinedVector<ResourcePair, 1>;

class HloGraphNode;
class HloScheduleGraph;

struct SchedulerConfig {
  int64_t collective_broadcast_overlap_limit = 1;
  int64_t collective_permute_overlap_limit = 1;
  int64_t all_to_all_overlap_limit = 1;
  int64_t ragged_all_to_all_overlap_limit = 1;
  int64_t all_gather_overlap_limit = 1;
  int64_t all_reduce_overlap_limit = 1;
  int64_t reduce_scatter_overlap_limit = 1;
  int64_t send_recv_overlap_limit = 1;
  int64_t send_recv_host_overlap_limit = 1;
  int64_t copy_overlap_limit = 1;
  uint64_t memory_limit = UINT64_MAX;
  int64_t max_hops_to_closest_selective_overlap = 0;
  int64_t rerun = 0;
  int64_t parallel_collective_overlap_limit = 1;
  int64_t parallel_scale_up_collective_overlap_limit = 1;
  int64_t parallel_async_compute_limit = 2;
  bool schedule_send_recvs = false;
  bool deannotate_group_if_blocked = false;
  // Consider send recv as the same resource. Some platforms do not take well
  // overlapping the send/recv ops between themselves.
  bool force_send_recv_to_use_same_resource = false;
  bool use_real_cost_model = false;
  bool aggressive_scheduling_policies = false;
  bool prioritize_async_depth_over_stall = false;
  bool enable_release_start_policy = false;
  bool resource_sharing = false;
  bool resource_serializing = false;
  bool depth_based_memory_pressure_reduction = false;
  bool enable_selective_resources = false;

  // Freely schedule nodes at the start of a scheduling group, allowing the
  // scheduler to delay them to promote better overlap.
  bool flexible_scheduling_annotation_scheduling = false;
  // If the above flag is also set, force the scheduler to provide maximum delay
  // to nodes at the stat of a scheduling group.
  bool aggressive_flexible_annotation_scheduling = false;
  // Prioritize  flexible annotation scheduling over memory pressure; this is
  // useful when the memory pressure is high. Without this, under high memory
  // pressure, aggressive_flexible_annotation_scheduling is not respected.
  bool force_delay_over_memory_pressure = false;
  // If true, estimate the fragmentation size of the module by running the heap
  // simulator.
  bool estimate_fragmentation_size = false;
  // If true, track the resource usage of sync ops in latency hiding scheduler.
  bool track_sync_op_resource_usage = false;
  // If true, use top down scheduling.
  bool top_down_scheduling = false;
  // If true, enable schedule by structure.
  bool enable_schedule_by_structure = false;
  // If true (and enable_schedule_by_structure is true), synchronous
  // collectives are used as schedule anchors by the target-specific
  // schedule-by-structure (critical path depth) heuristic.
  bool enable_cpd_for_sync_collective = true;
  // If set, only log computations that match the given regular expression.
  std::string log_computation_re;
};

// Base class for the core scheduling algorithm.
class SchedulerCore {
 public:
  // Hook function to modify scheduling graph before scheduler runs.
  using GraphProcessingHook = std::function<absl::Status(HloScheduleGraph*)>;

  // Abstract base class for target defined state.
  struct TargetDefinedState {
    virtual ~TargetDefinedState() = default;
    virtual void Reset() {}
  };

  // Abstract base class for scheduling state.
  struct SchedulingState {
    virtual ~SchedulingState() = default;
    virtual void Reset() {
      if (target_defined_state) {
        target_defined_state->Reset();
      }
    }
    GraphProcessingHook graph_processing_hook;
    std::unique_ptr<TargetDefinedState> target_defined_state;
  };

  virtual absl::Status InitializeScheduler(const HloModule* module) = 0;

  virtual absl::Status CaptureScheduleProto() = 0;

  virtual absl::StatusOr<ScheduleProto> GetCapturedScheduleProto() = 0;

  virtual absl::StatusOr<std::shared_ptr<SchedulerCore::SchedulingState>>
  MakeSchedulingState(const HloComputation* computation) {
    return absl::UnimplementedError("Not implemented.");
  }
  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) {
    return absl::UnimplementedError("Not implemented.");
  }
  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation,
      std::shared_ptr<SchedulingState> sched_state) {
    return absl::UnimplementedError("Not implemented.");
  }
  // Note: This function is not thread-safe. It must not be invoked during
  // concurrent BRKGA or simulated evaluation.
  virtual std::shared_ptr<SchedulingState> GetSchedulingState() {
    return nullptr;
  }

  virtual ~SchedulerCore() = default;
  virtual int64_t GetMemoryPeak() = 0;
  virtual void SetMemoryLimit(uint64_t new_limit) = 0;
  virtual uint64_t GetMemoryLimit() = 0;
  virtual int64_t GetRerunTimes() = 0;

  // Set a graph processing hook that will run before scheduling a computation.
  // Heuristics can use this to set scheduling preferences to the scheduling
  // graph nodes.
  virtual absl::Status SetGraphProcessingHook(const GraphProcessingHook& hook) {
    return absl::OkStatus();
  }
  virtual void SetEvaluatingConcurrently(bool val) {
    is_evaluating_concurrently_ = val;
  }
  virtual bool IsEvaluatingConcurrently() const {
    return is_evaluating_concurrently_;
  }

 protected:
  bool is_evaluating_concurrently_ = false;
};

// Helper class to keep track of which instructions are to be supported and
// how many supported instructions per-type are contained in computations
// recursively.
class AsyncTracker {
 public:
  virtual ~AsyncTracker() = default;

  // Returns if this is an Async op done that the scheduler supports.
  virtual bool IsSupportedAsyncDone(const HloInstruction& hlo) const;

  // Returns if this is an Async op start that the scheduler supports.
  virtual bool IsSupportedAsyncStart(const HloInstruction& hlo) const;

  // Returns resources used (i.e., occupied or released) by this instruction
  virtual ResourcesVector GetResourcesFromInstructionImpl(
      const HloInstruction& hlo) const;

  // Gets the resource type associated with the given op.
  static ResourceType GetResourceTypeForOp(HloOpcode op);

  // Returns resources used (i.e., occupied or released) by this instruction
  absl::Span<const ResourcePair> GetResourcesFromInstruction(
      const HloInstruction& hlo) const;

  // Modifies the schedule graph passed as input to add dependencies that are
  // implicit based on the system we are running on.
  virtual void PostProcessScheduleGraph(
      HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator) const {}

  // Returns the number of resources (of type resource_type) that are used by
  // this instruction.
  virtual int64_t GetNumResourcesPerInstruction(
      ResourceType resource_type, const HloInstruction& instr) const;
  virtual int64_t GetNumResourcesPerInstruction(
      int64_t resource_type, const HloInstruction& instr) const;

  // Returns a map of number of resources used per resource type by this
  // instruction.
  virtual absl::flat_hash_map<int64_t, int64_t> GetNumResourcesPerInstruction(
      const HloInstruction& instr) const;

  // Sets the maximum allowed number of instances for each resource
  virtual void SetConcurrentResourceLimits(
      absl::flat_hash_map<int64_t, int64_t>& max_concurrent_resource) const;

  // Returns the name of the given resource
  virtual absl::string_view GetResourceName(int64_t resource_type) const;

  // Returns the name of the given resource usage
  absl::string_view GetResourceUsageName(int64_t resource_usage_type) const;
  absl::string_view GetResourceUsageName(
      ResourceUsageType resource_usage_type) const;

  // Returns the first target defined resource's id, regardless of if it exits
  static int64_t GetTargetDefinedResourceTypeBegin() {
    return ResourceTypeToIndex(ResourceType::kTargetDefinedResourceTypeBegin);
  }

  // Returns the number of target defined resources
  virtual int64_t GetNumTargetDefinedResources() const;

  // Returns how many instructions using the given resource_type we can overlap
  virtual int64_t GetNumAvailableResources(int64_t resource_type) const;

  // Returns the hazard type that describes how to resolve the conflicts when
  // multiple instructions attempt to use the given resource type concurrently.
  // Default resources have a hazard type of kUnshareable.
  virtual ResourceHazardType GetResourceHazardType(int64_t resource_type) const;

  // Returns the list of the released shareable resources filtered from the
  // given resources vector.
  virtual absl::InlinedVector<int64_t, 1>
  GetReleasedShareableResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns the list of the occupied shareable resources filtered from the
  // given resources vector.
  virtual absl::InlinedVector<int64_t, 1>
  GetOccupiedShareableResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns the list of the occupied serial resources filtered from the given
  // resources vector.
  virtual absl::InlinedVector<int64_t, 1> GetOccupiedSerialResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns the list of the released nonextendable resources filtered from the
  // given resources vector.
  virtual absl::InlinedVector<int64_t, 1>
  GetReleasedNonextendableResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns whether the provided node releases a selective resource.
  bool ReleasesSelectiveResource(const HloGraphNode* node) const;

  // Returns whether the provided node occupies a selective resource.
  bool OccupiesSelectiveResource(const HloGraphNode* node) const;

  // Returns whether the resource is an in-order resource (e.g. hardware FIFO
  // command queue) where operations are processed sequentially and cannot be
  // interleaved across nested computation boundaries.
  bool IsInorderResource(int64_t resource_type) const {
    return GetResourceHazardType(resource_type) == ResourceHazardType::kInOrder;
  }

  CanonicalAsyncOp GetCanonicalAsyncOp(const HloInstruction& hlo) const {
    return get_canonical_async_op_(hlo);
  }

  // Updates target defined states after scheduling a node.
  virtual void UpdateTargetDefinedStates(
      const HloInstruction& hlo, const HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator,
      LatencyEstimator::TimeCost current_time,
      SchedulerCore::SchedulingState* sched_state) {}

  // Updates target defined states after scheduling a computation.
  virtual void UpdateTargetDefinedStates(
      HloComputation* computation,
      SchedulerCore::SchedulingState* sched_state) {}

  // Resets target defined states after scheduling a computation.
  virtual void ResetTargetDefinedStates(
      SchedulerCore::SchedulingState* sched_state) {}

  const SchedulerConfig& GetConfig() const { return config_; }

  // Overrides SchedulerConfig::enable_cpd_for_sync_collective. Used by the
  // scheduler to reschedule with a different schedule-by-structure setting.
  void SetEnableCpdForSyncCollective(bool enable) {
    config_.enable_cpd_for_sync_collective = enable;
  }

  // Clears the cache of per-computation resource maps. This is needed when,
  // e.g., we modify the schedule of a computation, which could change the
  // resource usage of the computation.
  void InvalidateCache() {
    absl::MutexLock lock(&async_in_computation_cache_mu_);
    async_in_computation_cache_.clear();
  }

  // Similar to InvalidateCache(), but only invalidates the cache for the given
  // computation.
  void InvalidateCache(const HloComputation* computation) {
    absl::MutexLock lock(&async_in_computation_cache_mu_);
    async_in_computation_cache_.erase(computation);
  }

  // Returns whether the async tracker is using top down scheduling.
  bool IsTopDownScheduling() const { return config_.top_down_scheduling; }

  explicit AsyncTracker(
      const SchedulerConfig& config,
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : get_canonical_async_op_(std::move(func)), config_(config) {}

 private:
  // Returns the number of "occupy" type of resources used by the instructions
  // in the given computation. Uses the scheduling information if available to
  // obtain more accurate resource usage. If an instruction uses multiple
  // instances of the same "occupy" type of resource, that number is respected
  // and returned in the resulting map.
  const absl::flat_hash_map<int64_t, int64_t>& RecursivelyComputeResourceMap(
      const HloComputation* computation) const;
  // Similar as above, but uses scheduling information to obtain more accurate
  // resource usage. Useful for non-fusion computations.
  // REQUIRES: The computation must be scheduled.
  const absl::flat_hash_map<int64_t, int64_t>&
  RecursivelyComputeResourceMapForScheduledComputation(
      const HloComputation* computation) const;

  mutable absl::Mutex async_in_computation_cache_mu_;
  mutable absl::flat_hash_map<
      const HloComputation*,
      std::unique_ptr<absl::flat_hash_map<int64_t, int64_t>>>
      async_in_computation_cache_
          ABSL_GUARDED_BY(async_in_computation_cache_mu_);
  GetCanonicalAsyncOpFunc get_canonical_async_op_;

 protected:
  SchedulerConfig config_;
  mutable absl::Mutex resources_cache_mu_;
  mutable absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<ResourcesVector>>
      resources_cache_ ABSL_GUARDED_BY(resources_cache_mu_);
};

class SchedulingContext {
  // Scheduling context for a single pass.  This object runs HloAliasAnalysis on
  // first use and should not be used across passes as the module may have
  // changed.
 public:
  SchedulingContext(const HloModule* module,
                    std::shared_ptr<const LatencyEstimator> latency_estimator,
                    std::shared_ptr<AsyncTracker> async_tracker,
                    const AliasInfo* alias_info,
                    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes =
                        HloCostAnalysis::DefaultShapeSize)
      : latency_estimator_(std::move(latency_estimator)),
        async_tracker_(std::move(async_tracker)),
        alias_info_(alias_info),
        shape_size_bytes_(shape_size_bytes),
        module_(module) {}

  std::shared_ptr<const HloAliasAnalysis> GetAliasAnalysis() const {
    // Lazy initialization of alias analysis on first use.
    if (alias_analysis_ == nullptr) {
      alias_analysis_ = HloAliasAnalysis::Run(module_, alias_info_).value();
    }
    return alias_analysis_;
  }

  std::shared_ptr<const LatencyEstimator> GetLatencyEstimator() const {
    return latency_estimator_;
  }
  std::shared_ptr<AsyncTracker> GetAsyncTracker() const {
    return async_tracker_;
  }

  const HloCostAnalysis::ShapeSizeFunction& GetShapeSizeBytes() const {
    return shape_size_bytes_;
  }

  const AliasInfo* GetAliasInfo() const { return alias_info_; }

 private:
  mutable std::shared_ptr<const HloAliasAnalysis> alias_analysis_;
  std::shared_ptr<const LatencyEstimator> latency_estimator_;
  std::shared_ptr<AsyncTracker> async_tracker_;
  const AliasInfo* alias_info_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  const HloModule* module_;
};

// Tracks user annotations for scheduling.
class AnnotationTracker {
 public:
  explicit AnnotationTracker(const HloModule* module) : module_(module) {
    for (const HloComputation* comp : module_->MakeNonfusionComputations()) {
      absl::flat_hash_set<int64_t> annotations;
      for (const HloInstruction* instr : comp->instructions()) {
        if (auto annotation = GetAnnotation(instr)) {
          if (annotations.insert(annotation.value()).second) {
            comp_annotation_map_[comp].push_back(annotation.value());
          }
          annotations_[annotation.value()][comp].push_back(instr);
        }
      }
    }
  }
  bool HasAnnotations(const HloComputation* comp) const {
    return ContainsKey(comp_annotation_map_, comp);
  }
  std::vector<int64_t> GetAnnotations(const HloComputation* comp) const {
    return comp_annotation_map_.at(comp);
  }
  std::optional<int64_t> GetAnnotation(const HloInstruction* instr) const {
    const auto& attrs = instr->frontend_attributes().map();
    if (attrs.contains(kXlaSchedulingGroupIdAttr)) {
      return std::stoi(attrs.at(kXlaSchedulingGroupIdAttr));
    }
    return std::nullopt;
  }
  std::vector<const HloInstruction*> GetInstructions(
      const HloComputation* comp, const int64_t annotation) const {
    if (annotation == kInvalidAnnotation) {
      return {};
    }
    return annotations_.at(annotation).at(comp);
  }
  int64_t GetNumInstructions(const HloComputation* comp,
                             const int64_t annotation) {
    return annotations_[annotation][comp].size();
  }
  void PrintAnnotationSets(int64_t level) const {
    for (const auto& [annotation, comp_instr_vector] : annotations_) {
      for (const auto& [comp, instrs] : comp_instr_vector) {
        VLOG(level) << "Annotation " << annotation << " has " << instrs.size()
                    << " instructions in computation " << comp->name();
        for (const HloInstruction* instr : instrs) {
          VLOG(level) << "  " << instr->name();
        }
      }
    }
  }

 private:
  const HloModule* module_;
  absl::flat_hash_map<const HloComputation*, std::vector<int64_t>>
      comp_annotation_map_;
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<const HloComputation*,
                                          std::vector<const HloInstruction*>>>
      annotations_;
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<const HloComputation*,
                                          std::vector<const HloInstruction*>>>
      annotation_successors_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ASYNC_TRACKER_H_
