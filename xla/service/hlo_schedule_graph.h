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

#ifndef XLA_SERVICE_HLO_SCHEDULE_GRAPH_H_
#define XLA_SERVICE_HLO_SCHEDULE_GRAPH_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/async_tracker.h"
#include "xla/service/latency_estimator.h"
#include "xla/status_macros.h"

namespace xla {

// Returns true if `instr` is a custom call or an async op carrying the
// `scheduler_hint=force_delay` frontend attribute.
bool IsCustomCallorAsyncOpWithForceDelayAttribute(const HloInstruction* instr);

// Represents an edge between two nodes in the schedule graph.
class HloEdge {
 public:
  // Constructor used for array resizing
  HloEdge() : latency_(0), original_latency_(0), target_(nullptr) {}

  // Nullptr is not a valid value for 'target'.
  HloEdge(LatencyEstimator::TimeCost latency, HloGraphNode* target)
      : latency_(latency), original_latency_(latency), target_(target) {}
  LatencyEstimator::TimeCost Latency() const { return latency_; }
  LatencyEstimator::TimeCost OriginalLatency() const {
    return original_latency_;
  }
  void SetLatency(LatencyEstimator::TimeCost latency) { latency_ = latency; }
  void SetOriginalLatency(LatencyEstimator::TimeCost original_latency) {
    original_latency_ = original_latency;
  }
  const HloGraphNode& Target() const { return *target_; }
  HloGraphNode& Target() { return *target_; }
  HloGraphNode* TargetPtr() const { return target_; }
  std::string ToString() const;

  // Returns true iff SetSharableResources has been called
  bool SharableResourcesComputed() const {
    return sharable_resources_index_ >= 0;
  }
  // Returns the sharable resources vector for this edge
  // REQUIRES: SetSharableResources has been called
  const std::vector<int64_t>& GetSharableResources(
      const HloScheduleGraph* g) const;
  // Sets the sharable resources vector for this edge to "vals".  Subsequent
  // calls to GetSharableResources for this edge will return a vector
  // whose contents are identical to "vals".
  void SetSharableResources(HloScheduleGraph* g,
                            const std::vector<int64_t>& vals);
  void ResetSharableResources() { sharable_resources_index_ = -1; }

 private:
  // Latency between the two nodes connected by this edge. The other end of the
  // edge is the owner of the HloEdge object. This latency can get updated due
  // to various scheduling optimizations.
  LatencyEstimator::TimeCost latency_;
  // Original latency is the initial latency value (typically computed by a
  // latency estimator).
  LatencyEstimator::TimeCost original_latency_;
  // Target node of this edge.
  HloGraphNode* target_;

  // If -1, not initialized yet.  If >= 0, then this is an index
  // into a vector in the enclosing HloScheduleGraph.  0 is a special
  // value that points to an empty vector shared across all edges that
  // have an empty sharable_resources_ set (most edges).
  // > 0 points to an entry containing an edge-specific vector in the
  // HloScheduleGraph holding the sharable_resources_ for this edge.
  int sharable_resources_index_ = -1;
};

// Node in the schedule graph, plus information used for scheduling.
class HloGraphNode {
 public:
  using TimeCost = LatencyEstimator::TimeCost;

  // Constructor used temporarily for initializing a vector of HloGraphNodes
  HloGraphNode()
      : instr_(nullptr), opcode_(HloOpcode::kAdd), original_position_(0) {
    InitBitFields();
  }

  // Nullptr is not a valid value for 'i'.
  explicit HloGraphNode(const HloInstruction* i, int64_t original_position)
      : instr_(i), opcode_(i->opcode()), original_position_(original_position) {
    InitBitFields();
    is_host_transfer_ =
        (opcode_ == HloOpcode::kSend || opcode_ == HloOpcode::kSendDone ||
         opcode_ == HloOpcode::kRecv || opcode_ == HloOpcode::kRecvDone) &&
        static_cast<const HloSendRecvInstruction*>(i)->is_host_transfer();
  }

  static void UpdateOrAddDependency(HloGraphNode* from, HloGraphNode* to,
                                    LatencyEstimator::TimeCost latency) {
    auto update_latency_if_edge_exists =
        [&](absl::Span<HloEdge> edges, HloGraphNode* to,
            LatencyEstimator::TimeCost latency) {
          auto it = absl::c_find_if(
              edges, [&](HloEdge edge) { return &edge.Target() == to; });
          if (it != edges.end()) {
            it->SetLatency(latency);
            return true;
          }
          return false;
        };
    if (!update_latency_if_edge_exists(from->GetSuccessors(), to, latency)) {
      from->AddSuccessor(HloEdge(latency, to));
      from->outdegree_++;
    }
    if (!update_latency_if_edge_exists(to->GetPredecessors(), from, latency)) {
      to->AddPredecessor(HloEdge(latency, from));
      to->indegree_++;
    }
  }

  static void UpdateOrAddDependency(HloGraphNode* from, HloGraphNode* to,
                                    const LatencyEstimator* latency_estimator) {
    UpdateOrAddDependency(from, to,
                          latency_estimator->GetLatencyBetween(*from, *to));
  }

  static void AddDependency(HloGraphNode* from, HloGraphNode* to,
                            LatencyEstimator::TimeCost latency) {
    to->AddPredecessor(HloEdge(latency, from));
    to->indegree_++;
    from->AddSuccessor(HloEdge(latency, to));
    from->outdegree_++;
  }

  static void AddDependency(HloGraphNode* from, HloGraphNode* to,
                            const LatencyEstimator* latency_estimator) {
    AddDependency(from, to, latency_estimator->GetLatencyBetween(*from, *to));
  }
  // Reset the node to a state where it's ready to be scheduled again.
  void ResetScheduling() {
    predecessors_.RevertToStaticSize();
    successors_.RevertToStaticSize();
    scheduled_ = false;
    indegree_ = predecessors_.size();
    outdegree_ = successors_.size();
    ready_time_ = std::numeric_limits<TimeCost>::max();
    force_delay_after_target_ = false;
    ClearAnnotation();
  }
  size_t GetReadyNodesIfScheduled() const { return ready_nodes_if_scheduled_; }
  void UpdateReadyNodesIfScheduled() {
    ready_nodes_if_scheduled_ = 0;
    if (top_down_scheduling_) {
      for (auto& succ : GetSuccessors()) {
        if (succ.Target().GetIndegree() == 1) {
          ++ready_nodes_if_scheduled_;
        }
      }
    } else {
      for (auto& pred : GetPredecessors()) {
        if (pred.Target().GetOutdegree() == 1) {
          ++ready_nodes_if_scheduled_;
        }
      }
    }
  }
  const HloInstruction& GetInstr() const { return *instr_; }
  HloOpcode GetOpcode() const { return opcode_; }
  bool IsHostTransfer() const { return is_host_transfer_; }
  bool IsScheduled() const { return scheduled_; }
  int32_t GetIndegree() const { return indegree_; }
  int32_t GetOutdegree() const { return outdegree_; }
  TimeCost GetReadyTime() const { return ready_time_; }
  void SetIndegree(int32_t indeg) {
    indegree_ = indeg;
    if (top_down_scheduling_ && indeg == 1) {
      for (HloEdge& pred : GetPredecessors()) {
        pred.Target().UpdateReadyNodesIfScheduled();
      }
    }
  }
  void SetOutdegree(int32_t outdeg) {
    outdegree_ = outdeg;
    if (!top_down_scheduling_ && outdeg == 1) {
      for (HloEdge& succ : GetSuccessors()) {
        succ.Target().UpdateReadyNodesIfScheduled();
      }
    }
  }
  void SetScheduled() { scheduled_ = true; }
  void SetReadyTime(TimeCost ready_time) { ready_time_ = ready_time; }
  TimeCost GetCost() const { return cost_; }
  void SetCost(TimeCost cost) { cost_ = cost; }
  TimeCost GetAsyncDepth() const { return async_depth_; }
  TimeCost GetAsyncHeight() const { return async_height_; }
  TimeCost GetDepth() const { return depth_; }
  TimeCost GetGraphDepth() const { return graph_depth_; }
  void SetAsyncDepth(TimeCost async_depth) { async_depth_ = async_depth; }
  void SetAsyncHeight(TimeCost async_height) { async_height_ = async_height; }
  bool IsSupportedAsyncDone() const { return is_supported_async_done_; }
  bool IsSupportedAsyncStart() const { return is_supported_async_start_; }
  void SetDepth(TimeCost depth) { depth_ = depth; }
  void SetGraphDepth(TimeCost graph_depth) { graph_depth_ = graph_depth; }
  bool GetForceDelay() const { return force_delay_; }
  void SetForceDelay(bool force_delay) { force_delay_ = force_delay; }
  int GetForceDelayPriority() const { return force_delay_priority_; }
  void SetForceDelayPriority(int force_delay_priority) {
    force_delay_priority_ = force_delay_priority;
  }
  bool GetForceEarly() const { return force_early_; }
  void SetForceEarly(bool force_early) { force_early_ = force_early; }
  bool GetForceDelayAfterTarget() const { return force_delay_after_target_; }
  void SetForceDelayAfterTarget(bool force_delay_after_target) {
    force_delay_after_target_ = force_delay_after_target;
  }
  bool GetValuableForSelectiveOverlap() const {
    return valuable_for_selective_overlap_;
  }
  void SetValuableForSelectiveOverlap(bool valuable_for_selective_overlap) {
    valuable_for_selective_overlap_ = valuable_for_selective_overlap;
  }
  bool ReleasesSelectiveResource() const {
    return releases_selective_resource_;
  }
  bool OccupiesSelectiveResource() const {
    return occupies_selective_resource_;
  }
  void SetReleasesSelectiveResource(bool releases_selective_resource) {
    releases_selective_resource_ = releases_selective_resource;
  }
  void SetOccupiesSelectiveResource(bool occupies_selective_resource) {
    occupies_selective_resource_ = occupies_selective_resource;
  }
  int64_t GetNumHopsToClosestSelectiveResourceOccupier() const {
    return num_hops_to_closest_selective_resource_occupier_;
  }
  void SetNumHopsToClosestSelectiveResourceOccupier(
      int64_t num_hops_to_closest_selective_resource_occupier) {
    num_hops_to_closest_selective_resource_occupier_ =
        num_hops_to_closest_selective_resource_occupier;
  }
  void SetPreference(double preference) {
    preference_ = preference;
    has_preference_ = true;
  }
  double GetPreference() const { return preference_; }
  bool HasPreference() const { return has_preference_; }
  const ResourcesVector& GetResources() const { return rare_->resources; }
  bool DoesOccupyAnyResource() const { return does_occupy_any_resource_; }
  bool DoesReleaseAnyResource() const { return does_release_any_resource_; }
  bool DoesOccupyShareableResource(int64_t resource) const {
    if (!has_rare_) {
      return false;
    }
    return absl::c_linear_search(rare_->occupied_shareable_resources, resource);
  }
  bool DoesReleaseResource(int64_t res) const {
    if (!has_rare_) {
      return false;
    }
    return absl::c_any_of(
        rare_->resources, [res](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceRelease &&
                 resource.first == res;
        });
  }
  bool DoesReleaseResource(ResourceType res) const {
    return DoesReleaseResource(ResourceTypeToIndex(res));
  }
  bool DoesOccupyResource(int64_t res) const {
    if (!has_rare_) {
      return false;
    }
    return absl::c_any_of(
        rare_->resources, [res](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceOccupy &&
                 resource.first == res;
        });
  }
  bool DoesOccupyResource(ResourceType res) const {
    return DoesOccupyResource(ResourceTypeToIndex(res));
  }
  // Returns the net resources used by the node. For a while loop, it computes
  // the net resources used by the instructions in the while body. Otherwise, it
  // returns the readily-available resources vector.
  ResourcesVector GetNetResources() const {
    if (GetOpcode() != HloOpcode::kWhile) {
      return rare_->resources;
    }
    ResourcesVector result;
    for (const auto& [resource, usage] : rare_->resources) {
      if (usage == ResourceUsageType::kResourceOccupy &&
          !DoesReleaseResource(resource)) {
        result.push_back(std::make_pair(resource, usage));
      }
      if (usage == ResourceUsageType::kResourceRelease &&
          !DoesOccupyResource(resource)) {
        result.push_back(std::make_pair(resource, usage));
      }
    }
    return result;
  }
  std::optional<ResourceUsageType> UsesResourceType(ResourceType res) const {
    if (has_rare_) {
      int64_t res_type = ResourceTypeToIndex(res);
      for (const auto& [resource_type, usage_type] : rare_->resources) {
        if (resource_type == res_type) {
          return usage_type;
        }
      }
    }
    return std::nullopt;
  }
  std::optional<ResourceUsageType> UsesResourceType(int64_t res) const {
    if (has_rare_) {
      for (const auto& [resource_type, usage_type] : rare_->resources) {
        if (resource_type == res) {
          return usage_type;
        }
      }
    }
    return std::nullopt;
  }
  const std::vector<int64_t>& GetShareableResourcesOnEdge(HloScheduleGraph* g,
                                                          HloEdge& edge) const {
    if (!edge.SharableResourcesComputed()) {
      std::vector<int64_t> resources;
      if (has_rare_) {
        HloGraphNode& to = edge.Target();
        absl::c_for_each(rare_->released_shareable_resources,
                         [this, &to, &resources](const int64_t resource) {
                           if (to.DoesOccupyShareableResource(resource) &&
                               this->DoesReleaseResource(resource)) {
                             resources.push_back(resource);
                           }
                         });
      }
      edge.SetSharableResources(g, resources);
    }
    return edge.GetSharableResources(g);
  }
  bool HasReleasedNonExtendableResources() const {
    return has_rare_ && !rare_->released_non_extendable_resources.empty();
  }
  const absl::InlinedVector<int64_t, 1>& GetReleasedNonExtendableResources()
      const {
    return rare_->released_non_extendable_resources;
  }

  absl::Span<HloEdge> GetPredecessors() { return predecessors_.GetSpan(); }
  absl::Span<const HloEdge> GetPredecessors() const {
    return predecessors_.GetConstSpan();
  }
  void AddPredecessor(const HloEdge& e) { predecessors_.AddEdge(e); }
  absl::Span<HloEdge> GetSuccessors() { return successors_.GetSpan(); }
  absl::Span<const HloEdge> GetSuccessors() const {
    return successors_.GetConstSpan();
  }
  void AddSuccessor(const HloEdge& e) { successors_.AddEdge(e); }
  int64_t GetOriginalPosition() const { return original_position_; }
  int64_t GetAnnotation() const { return annotation_; }
  absl::Status SetAnnotation(int64_t annotation) {
    TF_RET_CHECK(annotation_ == kInvalidAnnotation)
        << "Instruction " << instr_->name()
        << " has an existing annotation: " << annotation_;
    annotation_ = annotation;
    return absl::OkStatus();
  }
  void ClearAnnotation() { annotation_ = -1; }
  std::string ToString(const AsyncTracker* async_tracker = nullptr) const {
    std::string result;
    absl::StrAppend(&result, "Instr: ", instr_->ToShortString(), "\n");
    absl::StrAppend(&result, "ReadyTime: ", ready_time_, "\n");
    absl::StrAppend(&result, "Indegree: ", indegree_, "\n");
    absl::StrAppend(&result, "Outdegree: ", outdegree_, "\n");
    absl::StrAppend(&result, "Cost: ", cost_, "\n");
    absl::StrAppend(&result, "Async Depth: ", async_depth_, "\n");
    absl::StrAppend(&result, "Depth: ", depth_, "\n");
    absl::StrAppend(&result, "Graph Depth: ", graph_depth_, "\n");
    absl::StrAppend(&result, "Force Delay: ", force_delay_, "\n");
    absl::StrAppend(&result, "Force Early: ", force_early_, "\n");
    absl::StrAppend(
        &result, "Force Delay After Target: ", force_delay_after_target_, "\n");
    absl::StrAppend(&result, "Predecessors:\n");
    for (const HloEdge& e : GetPredecessors()) {
      absl::StrAppend(&result, e.ToString());
    }
    absl::StrAppend(&result, "Successors:\n");
    for (const HloEdge& e : GetSuccessors()) {
      absl::StrAppend(&result, e.ToString());
    }
    if (async_tracker != nullptr) {
      absl::StrAppend(&result, "Resources:\n");
      for (const auto& [resource, usage] : rare_->resources) {
        absl::StrAppend(
            &result, "\tResource: ", async_tracker->GetResourceName(resource),
            " usage: ", async_tracker->GetResourceUsageName(usage), "\n");
      }
    }
    return result;
  }
  bool HasRecursiveResources() const { return has_recursive_resources_; }
  const absl::flat_hash_map<int64_t, int64_t>& GetRecursiveResources() const {
    return rare_->recursive_resources;
  }
  bool HasOperandThatIsSupportedAsyncDone() const {
    return has_operand_that_is_supported_async_done_;
  }

  bool HasUserThatIsSupportedAsyncStart() const {
    return has_user_that_is_supported_async_start_;
  }
  void SetTopDownScheduling(bool top_down_scheduling) {
    top_down_scheduling_ = top_down_scheduling;
  }

 private:
  friend class HloScheduleGraph;
  friend class PriorityBasedHloScheduleGraph;

  // Older c++ versions don't allow initializers for bitfields, so we initialize
  // these in this routine, invoked by the constructor
  void InitBitFields() {
    does_occupy_any_resource_ = false;
    does_release_any_resource_ = false;
    is_supported_async_done_ = false;
    is_supported_async_start_ = false;
    has_operand_that_is_supported_async_done_ = false;
    has_user_that_is_supported_async_start_ = false;
    scheduled_ = false;
    valuable_for_selective_overlap_ = true;
    releases_selective_resource_ = false;
    occupies_selective_resource_ = false;
    has_recursive_resources_ = false;
  }

  // Some of the fields in this are rarely non-empty (in one large compilation,
  // 98% of nodes had all the fields empty).  To make each HloGraphNode more
  // compact, we store the state for these fields in a vector of Rare objects in
  // the parent HloScheduleGraph.  All instructions that have all these fields
  // empty point to the same canonical entry with all empty fields.  Other
  // nodes point to their own storage allocated in a vector of Rare objects
  // in the HloScheduleGraph object.
  struct Rare {
    // Non-extendable resources released by this node.
    absl::InlinedVector<int64_t, 1> released_non_extendable_resources;
    // Shareable resources released by this node.
    absl::InlinedVector<int64_t, 1> released_shareable_resources;
    // Shareable resources occupied by this node.
    absl::InlinedVector<int64_t, 1> occupied_shareable_resources;
    // Recursive resources used by the node.
    absl::flat_hash_map<int64_t, int64_t> recursive_resources;
    // AsyncResources used by the node.
    ResourcesVector resources;
  };

  // Instruction this Graph node represents
  const HloInstruction* instr_;
  // Opcode of instr_, copied here for better cache behavior (so we can look at
  // the opcode without having to touch another cache line).
  HloOpcode opcode_;
  // If multiple nodes are there with force_delay_ = true, the one with the
  // lowest delay priority will be scheduled first.
  int force_delay_priority_ = 0;

  // Some of the booleans are looked at very often, so we avoid making them
  // bitfields
  // Force the scheduling of the nodes with attribute set as late as possible.
  bool force_delay_ = false;
  // Whether the instruction is a host transfer (send/recv).
  bool is_host_transfer_ = false;
  // Force the scheduling of the nodes with attribute set as early as possible.
  bool force_early_ = false;
  // If has_rare_ is false, then all the fields in rare can assumed to be
  // empty/default values
  bool has_rare_ = false;
  // Force the scheduling of the nodes with attribute as late as possible,
  // but do it after evaluating the early target scheduling rule.
  bool force_delay_after_target_ = false;
  // If true, schedule the nodes from top to bottom.
  bool top_down_scheduling_ = false;

  // Preference value used for scheduling heuristics,
  // a graph node having a higher preference value means it's scheduled
  // earlier. See ReadySetLt::operator()
  bool has_preference_ = false;
  float preference_ = 0.0;

  // Other boolean fields are less performance sensitive so can be stored in
  // bitfields.  These are initialized to default values in InitBitFields() (due
  // to older c++ versions not supporting initializers for bit fields).

  // Does the node occupy any resource.
  bool does_occupy_any_resource_ : 1;
  // Does the node release any resource.
  bool does_release_any_resource_ : 1;
  // Is the node a supported async done.
  bool is_supported_async_done_ : 1;
  // Is the node a supported async start.
  bool is_supported_async_start_ : 1;
  // Whether the instruction has an operand which is a supported async done.
  bool has_operand_that_is_supported_async_done_ : 1;
  // Whether the instruction has a user which is a supported async start.
  bool has_user_that_is_supported_async_start_ : 1;
  // Whether this node has been scheduled or not yet.
  bool scheduled_ : 1;
  // Whether this node can be overlapped with (can cover the latency/cost of)
  // edges occupying selective resources.
  bool valuable_for_selective_overlap_ : 1;
  // Whether this node releases a selective resource.
  bool releases_selective_resource_ : 1;
  // Whether this node occupies a selective resource.
  bool occupies_selective_resource_ : 1;
  // Whether recursive_resources_.size() > 0
  bool has_recursive_resources_ : 1;
  // The position of this node in the original order.
  int32_t original_position_;
  // Pointer to the HloGraphNode::Rare entry for this node in the parent object
  // (Actual storage is managed by rare_storage_ in parent object)
  Rare* rare_ = nullptr;
  // Estimated time at which this node is gonna be ready to be scheduled.
  // The node should be added to the ready to be scheduled set when ready_time_
  // is less or equal to the current time in the schedule.
  TimeCost ready_time_ = std::numeric_limits<TimeCost>::max();
  // Time cost of the execution of the operation of this nodes represent.
  TimeCost cost_ = 0.0;
  // Accumulated async latency along the path from this node to a node with no
  // predecessors, e.g., parameters. Used in the bottom-up scheduling.
  TimeCost async_depth_ = 0.0;
  // Accumulated async latency along the path from this node to a node with no
  // successors, e.g., root. Used in the top-down scheduling.
  TimeCost async_height_ = 0.0;
  // Depth in latency terms of node based on operation cost on the path to the
  // entry node.
  TimeCost depth_ = 0.0;
  int64_t annotation_ = kInvalidAnnotation;

  // Depth in latency terms of node based on distance to the entry node.
  int64_t graph_depth_ = 0;
  // Nums hops to closest selective resource occupier.
  int32_t num_hops_to_closest_selective_resource_occupier_ =
      std::numeric_limits<int32_t>::max();
  // Number of ready nodes if this node is scheduled.
  int32_t ready_nodes_if_scheduled_ = 0;
  // Number of predecessor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t indegree_ = 0;
  // Number of successor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t outdegree_ = 0;

  // EdgeStorage is used to manage incoming and outgoing edge arrays.
  // The storage is hybrid where most backing store for HloEdge objects
  // is in two vectors allocated in the HloScheduleGraph, and EdgeStorage
  // points into a span within those arrays.  However, sometimes in
  // the graph construction process, it is hard to know the total number
  // of edges needed for certain nodes in rare circumstances.  For handling
  // those cases when the allocated span in the shared arrays is not
  // sufficient, the storage is allocated and managed by EdgeStorage itself.
  class EdgeStorage {
   public:
    enum kOwnership { kNotOwned };
    EdgeStorage() : alloc_(0), owned_(false) {}

    ~EdgeStorage() {
      if (owned_) {
        delete[] edges_;
      }
    }
    // Set the edge storage to point to separately managed space for
    // an array of "alloc" edges pointed to by "ptr".  This memory must
    // remain live for as long as the edges are being used.
    void SetEmptyPointingToSharedSpace(HloEdge* ptr, int alloc) {
      edges_ = ptr;
      size_ = 0;
      alloc_ = alloc;
      owned_ = false;
    }

    int size() const { return size_; }
    absl::Span<HloEdge> GetSpan() { return absl::MakeSpan(edges_, size_); }
    absl::Span<const HloEdge> GetConstSpan() const {
      return absl::MakeConstSpan(edges_, size_);
    }

    void RecordStaticSize() { static_size_ = size_; }
    void RevertToStaticSize() { size_ = static_size_; }

    void AddEdge(const HloEdge& e) {
      if (size_ >= alloc_) {
        // Grow: Make sure we always leave room for at least
        // one new edge since we're adding one right now, and do
        // doubling to avoid too much copying overhead
        int new_size = std::max(1, alloc_ * 2);
        HloEdge* new_edges = new HloEdge[new_size];
        for (int i = 0; i < size_; i++) {
          new_edges[i] = edges_[i];
        }
        if (owned_) {
          delete[] edges_;
        }
        alloc_ = new_size;
        edges_ = new_edges;
        owned_ = true;
      }
      DCHECK_LT(size_, alloc_);
      edges_[size_] = e;
      size_++;
    }

   private:
    // Points to edges.  Usually points into
    // HloScheduleGraph::predecessors_storage_ (for predecessors) or
    // HloScheduleGraph::successors_storage_ (if !owned).  If owned, then this
    // HloGraphNode object is responsible for the memory pointed to by edges
    HloEdge* edges_ = nullptr;
    // Number of valid edges
    int size_ = 0;
    // Allocated space in edges_ array
    int32_t alloc_ : 31;
    bool owned_ : 1;
    int static_size_ = 0;
  };
  EdgeStorage predecessors_;
  EdgeStorage successors_;
};

// HloGraphNode overloads of LatencyEstimator pair predicates. They are defined
// here because HloGraphNode is only complete in this header.
inline bool LatencyEstimator::IsAsyncPair(const HloGraphNode& from,
                                          const HloGraphNode& target) const {
  return IsAsyncPair(from.GetInstr(), target.GetInstr());
}

inline bool LatencyEstimator::IsP2pPair(const HloGraphNode& from,
                                        const HloGraphNode& target) const {
  return IsP2pPair(from.GetInstr(), target.GetInstr());
}

// Schedule graph that can be used to drive scheduling
// of HLO instructions.
class HloScheduleGraph {
 public:
  HloScheduleGraph() = default;

  // Instructions in the list passed to the constructor shouldn't be
  // altered/deleted during the existence of the HloScheduleGraph.
  // Nullptr is not a valid value for 'post_order_instructions' and
  // 'alias_analysis'.
  HloScheduleGraph(
      const std::vector<HloInstruction*>* post_order_instructions,
      std::shared_ptr<const SchedulingContext> scheduling_context,
      std::shared_ptr<const HloReachabilityMap> reachability_map = nullptr);
  virtual ~HloScheduleGraph() = default;
  HloScheduleGraph(HloScheduleGraph&&) = default;
  HloScheduleGraph& operator=(HloScheduleGraph&&) = default;
  void PrintSizes() const {
    LOG(INFO) << "HloScheduleGraph sizes: node_storage: "
              << node_storage_.size()
              << " preds: " << predecessors_storage_.size()
              << " succs: " << successors_storage_.size()
              << " rare: " << rare_storage_.size()
              << " sharable_resources: " << sharable_resources_storage_.size()
              << " nodes: " << nodes_.size()
              << " original_order: " << original_order_.size();
  }

  virtual std::string ToString() const;

  HloGraphNode& GetNode(const HloInstruction* instr) const;
  HloGraphNode* GetNodePtr(const HloInstruction* instr) const;

  std::vector<HloGraphNode*> FindBottomRoots() const;

  std::vector<HloGraphNode*> FindTopRoots() const;

  virtual void InitializeGraphAnalysis();

  void AnnotateGraph(const AnnotationTracker* annotation_tracker);

  const HloReachabilityMap& reachability() const { return *reachability_; }

  // List of instructions in the original scheduled order. (Before scheduling).
  absl::Span<const HloInstruction* const> GetOriginalInstrList() const {
    return absl::MakeConstSpan(original_order_);
  }
  // Returns what was the original instruction position in the original order.
  int64_t OriginalInstructionPosition(const HloInstruction* instr) const {
    auto it = nodes_.find(instr);
    CHECK(it != nodes_.end());
    return it->second;
  }

  // Return node preference values in original order.
  std::vector<double> GetPreferences() {
    std::vector<double> preferences;
    preferences.reserve(original_order_.size());
    for (const HloInstruction* instr : original_order_) {
      preferences.push_back(GetNodePtr(instr)->GetPreference());
    }
    return preferences;
  }

  // Set node preference values in original order.
  void SetPreferences(const std::vector<double>& preferences) {
    CHECK_EQ(preferences.size(), original_order_.size());
    for (int i = 0; i < original_order_.size(); ++i) {
      GetNodePtr(original_order_[i])->SetPreference(preferences[i]);
    }
  }
  void RecordStaticEdges() {
    for (auto& pair : nodes_) {
      node_storage_[pair.second].predecessors_.RecordStaticSize();
      node_storage_[pair.second].successors_.RecordStaticSize();
    }
  }

  void ResetScheduling() {
    for (auto& pair : nodes_) {
      node_storage_[pair.second].ResetScheduling();
    }
    for (auto& pair : nodes_) {
      node_storage_[pair.second].UpdateReadyNodesIfScheduled();
    }
    for (auto& edge : predecessors_storage_) {
      edge.ResetSharableResources();
    }
    for (auto& edge : successors_storage_) {
      edge.ResetSharableResources();
    }
    sharable_resources_storage_.resize(1);
  }

 protected:
  friend class HloEdge;

  // Backing store for the nodes in the graph
  mutable std::vector<HloGraphNode> node_storage_;
  std::vector<HloEdge> predecessors_storage_;
  std::vector<HloEdge> successors_storage_;

  // Rare storage for HloGraphNode objects
  std::vector<std::unique_ptr<HloGraphNode::Rare>> rare_storage_;

  // Storage for HloEdge::sharable_resources_ vectors that are non-empty
  std::vector<std::vector<int64_t>> sharable_resources_storage_;

  // Map from instruction to the index in node_storage_ that holds the node
  absl::flat_hash_map<const HloInstruction*, int> nodes_;
  // List containing the original order (before scheduling) of the
  // instructions).
  std::vector<const HloInstruction*> original_order_;
  // Searches through node's predecessors to see if
  // possible_predecessor can be found.
  bool IsPredecessorTransitively(const HloGraphNode* node,
                                 const HloGraphNode* possible_predecessor);
  // Scheduling context for the graph.
  std::shared_ptr<const SchedulingContext> scheduling_context_;
  std::shared_ptr<const HloReachabilityMap> reachability_;
};

// These HloEdge routines need to be defined after HloScheduleGraph, since
// they peek inside its representation for the shared storage.

inline const std::vector<int64_t>& HloEdge::GetSharableResources(
    const HloScheduleGraph* g) const {
  return g->sharable_resources_storage_[sharable_resources_index_];
}

// Sets the sharable resources vector for this edge to "vals".  Subsequent
// calls to GetSharableResources for this edge will return a vector
// whose contents are identical to "vals".
inline void HloEdge::SetSharableResources(HloScheduleGraph* g,
                                          const std::vector<int64_t>& vals) {
  CHECK_LT(sharable_resources_index_, 0);
  if (vals.empty()) {
    // Share the 0th entry, which is an empty vector
    CHECK(g->sharable_resources_storage_[0].empty());
    sharable_resources_index_ = 0;
  } else {
    // Non-empty: add the vector into g->sharable_resources_storage_ and
    // save the index where it is stored in sharable_resources_index_.
    sharable_resources_index_ = g->sharable_resources_storage_.size();
    g->sharable_resources_storage_.push_back(vals);
  }
}

// AsyncTracker methods that inspect an HloGraphNode. They are defined here
// because HloGraphNode is only complete in this header.
inline bool AsyncTracker::ReleasesSelectiveResource(
    const HloGraphNode* node) const {
  return absl::c_any_of(
      node->GetResources(), [&](const ResourcePair& resource) {
        return resource.second == ResourceUsageType::kResourceRelease &&
               GetResourceHazardType(resource.first) ==
                   ResourceHazardType::kSelective;
      });
}

inline bool AsyncTracker::OccupiesSelectiveResource(
    const HloGraphNode* node) const {
  return absl::c_any_of(
      node->GetResources(), [&](const ResourcePair& resource) {
        return resource.second == ResourceUsageType::kResourceOccupy &&
               GetResourceHazardType(resource.first) ==
                   ResourceHazardType::kSelective;
      });
}

}  // namespace xla

#endif  // XLA_SERVICE_HLO_SCHEDULE_GRAPH_H_
