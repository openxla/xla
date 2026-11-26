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

#include "xla/hlo/transforms/collectives/collective_permute_equivariant_eliminator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr int64_t kInfCap = int64_t{1} << 58;
constexpr int64_t kMaxWeightCap = int64_t{1} << 50;
constexpr int kMaxSubcomputationDepth = 8;

// Exact Dinic max-flow / inclusion-minimal s-side min-cut solver.
class DinicMaxFlow {
 public:
  explicit DinicMaxFlow(int num_nodes)
      : adj_(num_nodes), level_(num_nodes, -1), iter_(num_nodes, 0) {}

  int AddNode() {
    const int id = static_cast<int>(adj_.size());
    adj_.emplace_back();
    level_.push_back(-1);
    iter_.push_back(0);
    return id;
  }

  void AddEdge(int from, int to, int64_t cap) {
    if (cap <= 0 || from == to || from < 0 || to < 0 ||
        adj_.size() <= static_cast<size_t>(from) ||
        adj_.size() <= static_cast<size_t>(to)) {
      return;
    }
    const int rev_from = static_cast<int>(adj_[to].size());
    const int rev_to = static_cast<int>(adj_[from].size());
    adj_[from].push_back(Edge{to, rev_from, cap});
    adj_[to].push_back(Edge{from, rev_to, 0});
  }

  int64_t ComputeMaxFlow(int source, int sink, int64_t cutoff) {
    int64_t flow = 0;
    while (flow < cutoff && Bfs(source, sink)) {
      absl::c_fill(iter_, 0);
      while (flow < cutoff) {
        const int64_t pushed = Dfs(source, sink, cutoff - flow);
        if (pushed == 0) {
          break;
        }
        flow += pushed;
      }
    }
    return flow;
  }

  // Computes the unique inclusion-minimal source-side cut after max flow.
  std::vector<bool> SourceReachable(int source) const {
    std::vector<bool> vis(adj_.size(), false);
    if (source < 0 || vis.size() <= static_cast<size_t>(source)) {
      return vis;
    }
    std::queue<int> q;
    vis[source] = true;
    q.push(source);
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      if (adj_.size() <= static_cast<size_t>(u)) {
        continue;
      }
      for (const Edge& e : adj_[u]) {
        if (e.cap > 0 && e.to >= 0 && vis.size() > static_cast<size_t>(e.to) &&
            !vis[e.to]) {
          vis[e.to] = true;
          q.push(e.to);
        }
      }
    }
    return vis;
  }

 private:
  struct Edge {
    int to;
    int rev;
    int64_t cap;
  };

  bool Bfs(int source, int sink) {
    absl::c_fill(level_, -1);
    if (source < 0 || sink < 0 ||
        level_.size() <= static_cast<size_t>(source) ||
        level_.size() <= static_cast<size_t>(sink)) {
      return false;
    }
    std::queue<int> q;
    level_[source] = 0;
    q.push(source);
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      if (adj_.size() <= static_cast<size_t>(u)) {
        continue;
      }
      for (const Edge& e : adj_[u]) {
        if (e.cap > 0 && e.to >= 0 &&
            level_.size() > static_cast<size_t>(e.to) && level_[e.to] < 0) {
          level_[e.to] = level_[u] + 1;
          q.push(e.to);
        }
      }
    }
    return level_[sink] >= 0;
  }

  int64_t Dfs(int u, int sink, int64_t limit) {
    if (u == sink || limit == 0) {
      return limit;
    }
    if (u < 0 || adj_.size() <= static_cast<size_t>(u) ||
        iter_.size() <= static_cast<size_t>(u) ||
        level_.size() <= static_cast<size_t>(u)) {
      return 0;
    }
    for (int& i = iter_[u]; static_cast<size_t>(i) < adj_[u].size(); ++i) {
      Edge& e = adj_[u][i];
      if (e.cap <= 0 || e.to < 0 ||
          level_.size() <= static_cast<size_t>(e.to) ||
          level_[e.to] != level_[u] + 1) {
        continue;
      }
      const int64_t pushed = Dfs(e.to, sink, std::min(limit, e.cap));
      if (pushed > 0) {
        e.cap -= pushed;
        if (adj_.size() > static_cast<size_t>(e.to) && e.rev >= 0 &&
            adj_[e.to].size() > static_cast<size_t>(e.rev)) {
          adj_[e.to][e.rev].cap += pushed;
        }
        return pushed;
      }
    }
    return 0;
  }

  std::vector<std::vector<Edge>> adj_;
  std::vector<int> level_;
  std::vector<int> iter_;
};

int64_t ArrayByteSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) *
         ShapeUtil::ElementsIn(shape);
}

bool IsPermutableArrayShape(const Shape& shape) {
  if (!shape.IsArray() || !shape.is_static() ||
      ShapeUtil::IsZeroElementArray(shape) ||
      primitive_util::IsSubByteNonPredType(shape.element_type())) {
    return false;
  }
  if (shape.has_layout() &&
      shape.layout().memory_space() != Layout::kDefaultMemorySpace) {
    return false;
  }
  return true;
}

bool HasDisallowedCollectivePermuteAttribute(const HloInstruction* inst) {
  return absl::c_any_of(
      inst->frontend_attributes().map(), [](const auto& entry) {
        return absl::StartsWith(entry.first, "_xla_host_transfer") ||
               absl::StartsWith(entry.first, "_xla_send_recv");
      });
}

bool HasHostMemorySpace(const Shape& shape) {
  bool has_host = false;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& sub, const ShapeIndex& /*idx*/) {
        if (sub.IsArray() && sub.has_layout() &&
            sub.layout().memory_space() == Layout::kHostMemorySpace) {
          has_host = true;
        }
      });
  return has_host;
}

bool IsFlatPermutableTupleOrArray(const Shape& shape) {
  if (shape.IsArray()) {
    return IsPermutableArrayShape(shape);
  }
  if (!shape.IsTuple() || shape.tuple_shapes().empty()) {
    return false;
  }
  for (const Shape& elem : shape.tuple_shapes()) {
    if (!IsPermutableArrayShape(elem)) {
      return false;
    }
  }
  return true;
}

int64_t CollectivePermuteCost(const Shape& shape, int64_t overhead_bytes) {
  const int64_t bytes = std::clamp<int64_t>(ArrayByteSize(shape), 0,
                                            kMaxWeightCap - overhead_bytes);
  return std::min(kMaxWeightCap, bytes + std::max<int64_t>(0, overhead_bytes));
}

std::vector<int64_t> InvertPermutation(absl::Span<const int64_t> perm) {
  std::vector<int64_t> inv(perm.size(), -1);
  for (size_t s = 0; s < perm.size(); ++s) {
    const int64_t t = perm[s];
    if (t >= 0 && inv.size() > static_cast<size_t>(t)) {
      inv[t] = static_cast<int64_t>(s);
    }
  }
  return inv;
}

std::vector<std::pair<int64_t, int64_t>> PermutationToPairs(
    absl::Span<const int64_t> perm) {
  std::vector<std::pair<int64_t, int64_t>> pairs;
  pairs.reserve(perm.size());
  for (size_t s = 0; s < perm.size(); ++s) {
    pairs.emplace_back(static_cast<int64_t>(s), perm[s]);
  }
  return pairs;
}

// Returns `perm` where `perm[src] = tgt` iff `inst` is a non-identity complete
// bijection single-operand `kCollectivePermute` over all `num_devices` ranks.
std::optional<std::pair<bool, std::vector<int64_t>>> ExtractCompleteBijection(
    const HloInstruction* inst, int64_t replica_count, int64_t num_partitions) {
  if (inst == nullptr || inst->opcode() != HloOpcode::kCollectivePermute ||
      inst->operand_count() != 1 || inst->HasControlDependencies() ||
      HasDisallowedCollectivePermuteAttribute(inst) ||
      !IsPermutableArrayShape(inst->shape()) ||
      !IsPermutableArrayShape(inst->operand(0)->shape()) ||
      !ShapeUtil::Compatible(inst->shape(), inst->operand(0)->shape())) {
    return std::nullopt;
  }
  const auto* cp = Cast<HloCollectivePermuteInstruction>(inst);
  const bool is_cross_partition = cp->channel_id().has_value();
  const int64_t n = is_cross_partition ? num_partitions : replica_count;
  if (n < 2 || static_cast<int64_t>(cp->source_target_pairs().size()) != n) {
    return std::nullopt;
  }
  std::vector<int64_t> perm(n, -1);
  std::vector<bool> seen_tgt(n, false);
  bool non_identity = false;
  for (const auto& [src, tgt] : cp->source_target_pairs()) {
    if (src < 0 || src >= n || tgt < 0 || tgt >= n ||
        perm.size() <= static_cast<size_t>(src) ||
        seen_tgt.size() <= static_cast<size_t>(tgt) || perm[src] != -1 ||
        seen_tgt[tgt]) {
      return std::nullopt;
    }
    perm[src] = tgt;
    seen_tgt[tgt] = true;
    if (src != tgt) {
      non_identity = true;
    }
  }
  if (!non_identity) {
    return std::nullopt;
  }
  return std::make_pair(is_cross_partition, std::move(perm));
}

// Returns true iff every device `r` and `perm_inv[r]` hold the exact same data
// slice under `raw_sharding`.
bool IsLeafShardingInvariantUnder(const HloSharding& raw_sharding,
                                  absl::Span<const int64_t> perm_inv) {
  if (raw_sharding.IsTuple() || raw_sharding.IsManual() ||
      raw_sharding.IsUnknown() || raw_sharding.IsUnreduced() ||
      raw_sharding.IsSingleDevice()) {
    return false;
  }
  if (raw_sharding.IsReplicated()) {
    return true;
  }
  const HloSharding sharding = raw_sharding.UseNamedShardingLeaf()
                                   ? HloSharding::V3ToV2Sharding(raw_sharding)
                                   : raw_sharding;
  if (!sharding.IsTiled() || !sharding.HasPartialReplication()) {
    return false;
  }
  for (OpSharding::Type sg : sharding.subgroup_types()) {
    if (sg != OpSharding::REPLICATED) {
      return false;
    }
  }
  const size_t n = perm_inv.size();
  if (sharding.num_devices() != static_cast<int64_t>(n)) {
    return false;
  }
  const int64_t data_rank = sharding.TiledDataRank();
  if (data_rank < 0) {
    return false;
  }
  std::vector<int64_t> data_strides(data_rank, 1);
  for (int64_t d = data_rank - 2; d >= 0; --d) {
    if (data_strides.size() > static_cast<size_t>(d + 1)) {
      data_strides[d] = data_strides[d + 1] * sharding.dimension(d + 1);
    }
  }
  std::vector<int64_t> data_tile_of_device(n, -1);
  bool valid = true;
  sharding.EachTile([&](absl::Span<const int64_t> index, int64_t device) {
    if (!valid || device < 0 ||
        data_tile_of_device.size() <= static_cast<size_t>(device) ||
        static_cast<int64_t>(index.size()) < data_rank ||
        data_tile_of_device[device] != -1) {
      valid = false;
      return;
    }
    int64_t linear = 0;
    for (int64_t d = 0; d < data_rank; ++d) {
      if (data_strides.size() > static_cast<size_t>(d)) {
        linear += index[d] * data_strides[d];
      }
    }
    data_tile_of_device[device] = linear;
  });
  if (!valid) {
    return false;
  }
  for (size_t r = 0; r < n; ++r) {
    const int64_t src = perm_inv[r];
    if (src < 0 || data_tile_of_device.size() <= static_cast<size_t>(src) ||
        data_tile_of_device[r] < 0 ||
        data_tile_of_device[r] != data_tile_of_device[src]) {
      return false;
    }
  }
  return true;
}

HloSharding PermuteLeafSharding(const HloSharding& raw_sharding,
                                absl::Span<const int64_t> perm) {
  if (raw_sharding.IsTuple() || raw_sharding.IsReplicated() ||
      raw_sharding.IsManual() || raw_sharding.IsUnknown() ||
      raw_sharding.IsUnreduced()) {
    return raw_sharding;
  }
  if (raw_sharding.IsSingleDevice()) {
    const int64_t dev = raw_sharding.GetUniqueDevice();
    if (dev >= 0 && perm.size() > static_cast<size_t>(dev)) {
      return HloSharding::SingleDevice(perm[dev], raw_sharding.metadata());
    }
    return raw_sharding;
  }
  const HloSharding sharding = raw_sharding.UseNamedShardingLeaf()
                                   ? HloSharding::V3ToV2Sharding(raw_sharding)
                                   : raw_sharding;
  if (!sharding.IsTiled() ||
      sharding.num_devices() != static_cast<int64_t>(perm.size())) {
    return raw_sharding;
  }
  Array<int64_t> permuted = sharding.tile_assignment().array();
  bool ok = true;
  permuted.Each([&](absl::Span<const int64_t> /*idx*/, int64_t* dev) {
    if (*dev >= 0 && perm.size() > static_cast<size_t>(*dev)) {
      *dev = perm[*dev];
    } else {
      ok = false;
    }
  });
  if (!ok) {
    return raw_sharding;
  }
  TileAssignment ta(
      std::make_shared<const Array<int64_t>>(std::move(permuted)));
  if (sharding.ReplicateOnLastTileDim()) {
    return HloSharding::PartialTile(ta, sharding.metadata());
  }
  if (!sharding.subgroup_types().empty()) {
    return HloSharding::Subgroup(ta, sharding.subgroup_types(),
                                 sharding.metadata());
  }
  return HloSharding::Tile(ta, sharding.metadata());
}

HloSharding PermuteSharding(const HloSharding& sharding, const Shape& shape,
                            absl::Span<const int64_t> perm) {
  if (!sharding.IsTuple()) {
    return PermuteLeafSharding(sharding, perm);
  }
  std::vector<HloSharding> leaves;
  leaves.reserve(sharding.tuple_elements().size());
  for (const HloSharding& elem : sharding.tuple_elements()) {
    leaves.push_back(PermuteLeafSharding(elem, perm));
  }
  return HloSharding::Tuple(shape, leaves);
}

const HloSharding* GetEntryParameterSharding(const HloModule* module,
                                             const HloInstruction* param) {
  if (param->has_sharding()) {
    return &param->sharding();
  }
  if (module->has_spmd_parameters_shardings()) {
    const std::vector<HloSharding>& shardings =
        module->spmd_parameters_shardings();
    const int64_t idx = param->parameter_number();
    if (idx >= 0 && shardings.size() > static_cast<size_t>(idx)) {
      return &shardings[idx];
    }
  }
  return nullptr;
}

bool IsPureLocalDeterministicOp(
    const HloInstruction* inst, int depth,
    const CollectivePermuteEquivariantEliminator::
        CustomCallEquivariancePredicate& custom_call_pred);

bool IsPureLocalComputation(
    const HloComputation* comp, int depth,
    const CollectivePermuteEquivariantEliminator::
        CustomCallEquivariancePredicate& custom_call_pred) {
  if (comp == nullptr || depth > kMaxSubcomputationDepth) {
    return false;
  }
  for (const HloInstruction* inst : comp->instructions()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (!IsPureLocalDeterministicOp(inst, depth + 1, custom_call_pred)) {
      return false;
    }
  }
  return true;
}

bool IsPureLocalDeterministicOp(
    const HloInstruction* inst, int depth,
    const CollectivePermuteEquivariantEliminator::
        CustomCallEquivariancePredicate& custom_call_pred) {
  if (inst == nullptr || depth > kMaxSubcomputationDepth ||
      inst->HasSideEffect() || inst->HasControlDependencies() ||
      HasHostMemorySpace(inst->shape())) {
    return false;
  }
  if (depth == 0 && !IsFlatPermutableTupleOrArray(inst->shape())) {
    return false;
  }
  const HloOpcode op = inst->opcode();
  if (op == HloOpcode::kRng || op == HloOpcode::kRngBitGenerator ||
      op == HloOpcode::kRngGetAndUpdateState ||
      op == HloOpcode::kStochasticConvert || op == HloOpcode::kPartitionId ||
      op == HloOpcode::kReplicaId) {
    return false;
  }
  if (op == HloOpcode::kCustomCall) {
    return depth == 0 && custom_call_pred != nullptr && custom_call_pred(inst);
  }
  if (inst->IsElementwise() && inst->called_computations().empty()) {
    return true;
  }
  switch (op) {
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kConcatenate:
    case HloOpcode::kPad:
    case HloOpcode::kReverse:
    case HloOpcode::kGather:
    case HloOpcode::kDot:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kScaledDot:
    case HloOpcode::kConvolution:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
    case HloOpcode::kFft:
    case HloOpcode::kCopy:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kMap:
    case HloOpcode::kFusion:
      return absl::c_all_of(
          inst->called_computations(), [&](const HloComputation* sub) {
            return IsPureLocalComputation(sub, depth + 1, custom_call_pred);
          });
    case HloOpcode::kScatter: {
      const auto* scatter = Cast<HloScatterInstruction>(inst);
      if (!scatter->unique_indices()) {
        return false;
      }
      return absl::c_all_of(
          inst->called_computations(), [&](const HloComputation* sub) {
            return IsPureLocalComputation(sub, depth + 1, custom_call_pred);
          });
    }
    default:
      return false;
  }
}

// Returns true iff `inst` is a synchronous collective whose ordered replica
// groups are closed under `perm` (G -> P(G) in identical element order).
bool IsOrthogonalCollective(
    const HloInstruction* inst, bool is_cross_partition,
    absl::Span<const int64_t> perm, int64_t replica_count,
    int64_t num_partitions,
    const CollectivePermuteEquivariantEliminator::
        CustomCallEquivariancePredicate& custom_call_pred) {
  if (inst == nullptr || inst->HasControlDependencies() ||
      !IsFlatPermutableTupleOrArray(inst->shape())) {
    return false;
  }
  const HloOpcode op = inst->opcode();
  if (op != HloOpcode::kAllReduce && op != HloOpcode::kReduceScatter &&
      op != HloOpcode::kAllGather && op != HloOpcode::kAllToAll) {
    return false;
  }
  for (const HloComputation* sub : inst->called_computations()) {
    if (!IsPureLocalComputation(sub, /*depth=*/1, custom_call_pred)) {
      return false;
    }
  }
  const auto* coll = Cast<HloCollectiveInstruction>(inst);
  if (coll->replica_groups().empty()) {
    return false;
  }
  std::optional<bool> use_global_device_ids = std::nullopt;
  if (const auto* ar = DynCast<HloAllReduceInstructionBase>(inst)) {
    use_global_device_ids = ar->use_global_device_ids();
  } else if (const auto* ag = DynCast<HloAllGatherInstruction>(inst)) {
    use_global_device_ids = ag->use_global_device_ids();
  }
  const absl::StatusOr<CollectiveOpGroupMode> mode = GetCollectiveOpGroupMode(
      coll->channel_id().has_value(), use_global_device_ids);
  if (!mode.ok()) {
    return false;
  }
  const CollectiveOpGroupMode group_mode = *mode;
  if (is_cross_partition) {
    if (group_mode ==
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA) {
      return true;
    }
    if (group_mode ==
        CollectiveOpGroupMode::
            COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION) {
      return false;
    }
  } else {
    if (group_mode ==
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION) {
      return true;
    }
  }
  auto map_id = [&](int64_t id) -> std::optional<int64_t> {
    if (id < 0) {
      return std::nullopt;
    }
    if (group_mode ==
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID) {
      if (num_partitions <= 0 || replica_count <= 0 ||
          id >= replica_count * num_partitions) {
        return std::nullopt;
      }
      const int64_t rep = id / num_partitions;
      const int64_t part = id % num_partitions;
      if (is_cross_partition) {
        if (perm.size() <= static_cast<size_t>(part)) {
          return std::nullopt;
        }
        return rep * num_partitions + perm[part];
      }
      if (perm.size() <= static_cast<size_t>(rep)) {
        return std::nullopt;
      }
      return perm[rep] * num_partitions + part;
    }
    if (perm.size() <= static_cast<size_t>(id)) {
      return std::nullopt;
    }
    return perm[id];
  };

  absl::flat_hash_set<std::vector<int64_t>> group_set;
  group_set.reserve(coll->replica_groups().size());
  for (const ReplicaGroup& rg : coll->replica_groups()) {
    if (rg.replica_ids().empty()) {
      return false;
    }
    group_set.emplace(rg.replica_ids().begin(), rg.replica_ids().end());
  }
  std::vector<int64_t> mapped;
  for (const ReplicaGroup& rg : coll->replica_groups()) {
    mapped.clear();
    mapped.reserve(rg.replica_ids().size());
    for (int64_t id : rg.replica_ids()) {
      const std::optional<int64_t> next_id = map_id(id);
      if (!next_id.has_value()) {
        return false;
      }
      mapped.push_back(*next_id);
    }
    if (!group_set.contains(mapped)) {
      return false;
    }
  }
  return true;
}

bool IsPureSingleOperandView(const HloInstruction* inst) {
  if (inst == nullptr || inst->operand_count() != 1 || inst->HasSideEffect() ||
      inst->HasControlDependencies() ||
      !IsPermutableArrayShape(inst->shape()) ||
      !IsPermutableArrayShape(inst->operand(0)->shape())) {
    return false;
  }
  switch (inst->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReverse:
      return true;
    default:
      return false;
  }
}

bool IsInvariantPropagatingOp(const HloInstruction* inst) {
  if (inst == nullptr || inst->operand_count() == 0 || inst->HasSideEffect() ||
      inst->HasControlDependencies() ||
      !IsPermutableArrayShape(inst->shape())) {
    return false;
  }
  const HloOpcode op = inst->opcode();
  if (op == HloOpcode::kRng || op == HloOpcode::kRngBitGenerator ||
      op == HloOpcode::kRngGetAndUpdateState ||
      op == HloOpcode::kStochasticConvert) {
    return false;
  }
  if (inst->IsElementwise() && inst->called_computations().empty()) {
    return true;
  }
  switch (op) {
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kSlice:
    case HloOpcode::kConcatenate:
    case HloOpcode::kPad:
    case HloOpcode::kReverse:
    case HloOpcode::kCopy:
    case HloOpcode::kDot:
      return true;
    default:
      return false;
  }
}

// Computes I_P: instructions whose value on device r equals their value on
// device perm_inv[r] for all r in [0, N).
absl::flat_hash_set<const HloInstruction*> ComputeInvariantInstructions(
    const HloComputation* comp, const HloModule* module,
    bool is_cross_partition, absl::Span<const int64_t> perm_inv,
    const absl::flat_hash_set<int64_t>& invariant_param_numbers) {
  absl::flat_hash_set<const HloInstruction*> inv;
  const bool is_entry = (comp == module->entry_computation());
  absl::flat_hash_map<const HloInstruction*, const HloSharding*>
      tuple_param_shardings;

  for (const HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (inst->HasControlDependencies() || inst->HasSideEffect()) {
      continue;
    }
    const HloOpcode op = inst->opcode();
    if (op == HloOpcode::kConstant || op == HloOpcode::kIota) {
      if (IsPermutableArrayShape(inst->shape())) {
        inv.insert(inst);
      }
      continue;
    }
    if (op == HloOpcode::kParameter) {
      if (invariant_param_numbers.contains(inst->parameter_number()) &&
          IsPermutableArrayShape(inst->shape())) {
        inv.insert(inst);
        continue;
      }
      if (is_cross_partition && is_entry) {
        const HloSharding* shd = GetEntryParameterSharding(module, inst);
        if (shd != nullptr) {
          if (shd->IsTuple() && inst->shape().IsTuple()) {
            tuple_param_shardings[inst] = shd;
          } else if (IsPermutableArrayShape(inst->shape()) &&
                     IsLeafShardingInvariantUnder(*shd, perm_inv)) {
            inv.insert(inst);
          }
        }
      }
      continue;
    }
    if (op == HloOpcode::kGetTupleElement &&
        IsPermutableArrayShape(inst->shape())) {
      const auto it = tuple_param_shardings.find(inst->operand(0));
      if (it != tuple_param_shardings.end()) {
        const int64_t idx = inst->tuple_index();
        const Shape& tup_shape = inst->operand(0)->shape();
        if (idx >= 0 &&
            tup_shape.tuple_shapes().size() > static_cast<size_t>(idx)) {
          const HloSharding sub = it->second->GetSubSharding(tup_shape, {idx});
          if (!sub.IsTuple() && IsLeafShardingInvariantUnder(sub, perm_inv)) {
            inv.insert(inst);
          }
        }
      }
      continue;
    }
    if (IsInvariantPropagatingOp(inst) &&
        absl::c_all_of(inst->operands(), [&](const HloInstruction* opnd) {
          return inv.contains(opnd);
        })) {
      inv.insert(inst);
    }
  }
  return inv;
}

HloInstruction* MaybeAdaptLayout(HloComputation* comp, HloInstruction* value,
                                 const Shape& target_shape) {
  if (value->shape().has_layout() && target_shape.has_layout() &&
      value->shape().layout() != target_shape.layout()) {
    return comp->AddInstruction(
        HloInstruction::CreateUnary(target_shape, HloOpcode::kCopy, value));
  }
  return value;
}

// Pre-cancels `CP_P(u) -> u` for `u in I_P` and back-to-back inverse
// `CP_P(view*(CP_{P_inv}(x))) -> view*(x)` before building the min-cut graph.
absl::StatusOr<bool> PreCancelTrivialPermutes(
    HloComputation* comp, const HloModule* module, bool is_cross_partition,
    absl::Span<const int64_t> perm, absl::Span<const int64_t> perm_inv,
    int64_t replica_count, int64_t num_partitions,
    const absl::flat_hash_set<int64_t>& invariant_param_numbers) {
  bool changed = false;
  const absl::flat_hash_set<const HloInstruction*> inv =
      ComputeInvariantInstructions(comp, module, is_cross_partition, perm_inv,
                                   invariant_param_numbers);

  std::vector<HloInstruction*> post_order = comp->MakeInstructionPostOrder();
  for (HloInstruction* inst : post_order) {
    if (inst->parent() == nullptr) {
      continue;
    }
    const std::optional<std::pair<bool, std::vector<int64_t>>> extracted =
        ExtractCompleteBijection(inst, replica_count, num_partitions);
    if (!extracted.has_value() || extracted->first != is_cross_partition ||
        absl::MakeConstSpan(extracted->second) != perm) {
      continue;
    }
    HloInstruction* opnd = inst->mutable_operand(0);
    if (inv.contains(opnd)) {
      HloInstruction* repl = MaybeAdaptLayout(comp, opnd, inst->shape());
      const bool had_sharding = inst->has_sharding();
      const HloSharding old_sharding =
          had_sharding ? inst->sharding() : HloSharding::Replicate();
      ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, repl));
      if (had_sharding && repl != opnd) {
        repl->set_sharding(old_sharding);
      }
      changed = true;
      continue;
    }
    // Walk backward through 0+ single-operand pure views to a matching
    // CP_{P_inv}(x).
    std::vector<HloInstruction*> view_chain;
    HloInstruction* curr = opnd;
    while (IsPureSingleOperandView(curr) && view_chain.size() < 16) {
      view_chain.push_back(curr);
      curr = curr->mutable_operand(0);
    }
    const std::optional<std::pair<bool, std::vector<int64_t>>> prev =
        ExtractCompleteBijection(curr, replica_count, num_partitions);
    if (!prev.has_value() || prev->first != is_cross_partition ||
        absl::MakeConstSpan(prev->second) != perm_inv) {
      continue;
    }
    HloInstruction* base =
        MaybeAdaptLayout(comp, curr->mutable_operand(0), curr->shape());
    if (view_chain.empty()) {
      HloInstruction* repl = MaybeAdaptLayout(comp, base, inst->shape());
      ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, repl));
      if (curr->user_count() == 0 && curr != comp->root_instruction()) {
        ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(curr));
      }
      changed = true;
      continue;
    }
    const bool all_single_use =
        (curr->user_count() == 1 && curr != comp->root_instruction()) &&
        absl::c_all_of(view_chain, [&](const HloInstruction* v) {
          return v->user_count() == 1 && v != comp->root_instruction();
        });
    HloInstruction* rebuilt = base;
    if (all_single_use) {
      ABSL_RETURN_IF_ERROR(view_chain.back()->ReplaceOperandWith(0, base));
      rebuilt = view_chain.front();
    } else {
      for (auto it = view_chain.rbegin(); it != view_chain.rend(); ++it) {
        HloInstruction* v = *it;
        rebuilt = comp->AddInstruction(
            v->CloneWithNewOperands(v->shape(), {rebuilt}));
        if (is_cross_partition && v->has_sharding()) {
          rebuilt->set_sharding(
              PermuteSharding(v->sharding(), v->shape(), perm));
        }
      }
    }
    if (all_single_use && is_cross_partition) {
      for (HloInstruction* v : view_chain) {
        if (v->has_sharding()) {
          v->set_sharding(PermuteSharding(v->sharding(), v->shape(), perm));
        }
      }
    }
    HloInstruction* repl = MaybeAdaptLayout(comp, rebuilt, inst->shape());
    ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, repl));
    if (curr->user_count() == 0 && curr != comp->root_instruction()) {
      ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(curr));
    }
    changed = true;
  }
  return changed;
}

// Partitions involution CPs (perm == perm_inv) into disjoint (entries, exits)
// so no single CP plays both roles in one min-cut solve.
void PartitionInvolutionPermutes(
    const std::vector<HloInstruction*>& post_order,
    const std::vector<HloInstruction*>& candidates,
    const absl::flat_hash_set<const HloInstruction*>& base_pinned,
    std::vector<HloInstruction*>* entries,
    std::vector<HloInstruction*>* exits) {
  absl::flat_hash_set<const HloInstruction*> cand_set(candidates.begin(),
                                                      candidates.end());

  // Forward closure from T_0 through non-expanding single-operand views and
  // GTEs identifies CPs anchored directly on T_0 outputs (natural Entries).
  absl::flat_hash_set<const HloInstruction*> t0_fwd = base_pinned;
  for (const HloInstruction* inst : post_order) {
    if (cand_set.contains(inst) || t0_fwd.contains(inst)) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kGetTupleElement &&
        t0_fwd.contains(inst->operand(0))) {
      t0_fwd.insert(inst);
      continue;
    }
    if (IsPureSingleOperandView(inst) && t0_fwd.contains(inst->operand(0)) &&
        ArrayByteSize(inst->shape()) <=
            ArrayByteSize(inst->operand(0)->shape())) {
      t0_fwd.insert(inst);
      continue;
    }
    if (inst->operand_count() > 0 && inst->shape().IsArray() &&
        absl::c_all_of(inst->operands(), [&](const HloInstruction* opnd) {
          return t0_fwd.contains(opnd);
        })) {
      int64_t in_bytes = 0;
      for (const HloInstruction* opnd : inst->operands()) {
        if (opnd->shape().IsArray()) {
          in_bytes += ArrayByteSize(opnd->shape());
        }
      }
      if (ArrayByteSize(inst->shape()) <= in_bytes) {
        t0_fwd.insert(inst);
      }
    }
  }

  // Backward closure from T_0 through non-shrinking views (and concats/pads)
  // identifies CPs feeding directly into T_0 inputs (natural Exits).
  absl::flat_hash_set<const HloInstruction*> t0_bwd = base_pinned;
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    const HloInstruction* inst = *it;
    if (cand_set.contains(inst) || t0_bwd.contains(inst) ||
        inst->users().empty()) {
      continue;
    }
    const bool is_bwd_view = IsPureSingleOperandView(inst) ||
                             inst->opcode() == HloOpcode::kConcatenate ||
                             inst->opcode() == HloOpcode::kPad ||
                             inst->opcode() == HloOpcode::kTuple;
    if (!is_bwd_view) {
      continue;
    }
    const int64_t in_bytes =
        inst->shape().IsArray() ? ArrayByteSize(inst->shape()) : 0;
    const bool all_users_bwd =
        absl::c_all_of(inst->users(), [&](const HloInstruction* u) {
          if (!t0_bwd.contains(u)) {
            return false;
          }
          if (IsPureSingleOperandView(u) &&
              ArrayByteSize(u->shape()) < in_bytes) {
            return false;
          }
          return true;
        });
    if (all_users_bwd) {
      t0_bwd.insert(inst);
    }
  }

  absl::flat_hash_set<const HloInstruction*> entry_set;
  absl::flat_hash_set<const HloInstruction*> exit_set;
  std::vector<HloInstruction*> undecided;
  for (HloInstruction* cp : candidates) {
    const bool from_t0 = t0_fwd.contains(cp->operand(0));
    const bool to_t0 =
        (cp == cp->parent()->root_instruction()) ||
        (!cp->users().empty() &&
         absl::c_all_of(cp->users(), [&](const HloInstruction* u) {
           return t0_bwd.contains(u);
         }));
    if (from_t0 && !to_t0) {
      entries->push_back(cp);
      entry_set.insert(cp);
    } else if (to_t0 && !from_t0) {
      exits->push_back(cp);
      exit_set.insert(cp);
    } else if (from_t0 && to_t0) {
      // Sandwiched immediately between T_0 on both sides: cannot cancel either
      // way; classify as Entry so it stays at unchanged weight.
      entries->push_back(cp);
      entry_set.insert(cp);
    } else {
      undecided.push_back(cp);
    }
  }

  if (undecided.empty()) {
    return;
  }

  // Classify remaining CPs in topological order: a CP reachable from an active
  // Entry without crossing T_0 or another CP closes that region as an Exit;
  // otherwise it opens a region as an Entry.
  absl::flat_hash_set<const HloInstruction*> undecided_set(undecided.begin(),
                                                           undecided.end());
  absl::flat_hash_set<const HloInstruction*> reach_from_entry;
  for (HloInstruction* inst : post_order) {
    if (entry_set.contains(inst)) {
      reach_from_entry.insert(inst);
      continue;
    }
    if (exit_set.contains(inst) || base_pinned.contains(inst)) {
      continue;
    }
    if (undecided_set.contains(inst)) {
      if (reach_from_entry.contains(inst->operand(0))) {
        exits->push_back(inst);
        exit_set.insert(inst);
      } else {
        entries->push_back(inst);
        entry_set.insert(inst);
        reach_from_entry.insert(inst);
      }
      continue;
    }
    if (absl::c_any_of(inst->operands(), [&](const HloInstruction* opnd) {
          return reach_from_entry.contains(opnd);
        })) {
      reach_from_entry.insert(inst);
    }
  }
}

// Solves the exact global min-cut for one orientation `(perm, perm_inv)` where
// `Entries` have permutation `perm_inv` and `Exits` have permutation `perm`,
// and applies the conjugation rewrite iff it strictly reduces total cost.
absl::StatusOr<bool> OptimizeOrientation(
    HloComputation* comp, const HloModule* module, bool is_cross_partition,
    absl::Span<const int64_t> perm, absl::Span<const int64_t> perm_inv,
    int64_t replica_count, int64_t num_partitions,
    int64_t per_collective_overhead_bytes,
    const absl::flat_hash_set<int64_t>& invariant_param_numbers,
    const CollectivePermuteEquivariantEliminator::
        CustomCallEquivariancePredicate& custom_call_pred,
    int64_t* next_channel_id) {
  const bool is_involution = (perm == perm_inv);
  std::vector<HloInstruction*> post_order = comp->MakeInstructionPostOrder();

  const absl::flat_hash_set<const HloInstruction*> inv_set =
      ComputeInvariantInstructions(comp, module, is_cross_partition, perm_inv,
                                   invariant_param_numbers);

  std::vector<HloInstruction*> entries;
  std::vector<HloInstruction*> exits;
  std::vector<HloInstruction*> involution_cands;
  absl::flat_hash_set<const HloInstruction*> base_pinned;
  HloInstruction* root = comp->root_instruction();
  if (root != nullptr && !root->shape().IsArray()) {
    base_pinned.insert(root);
  }

  for (HloInstruction* inst : post_order) {
    const std::optional<std::pair<bool, std::vector<int64_t>>> bij =
        ExtractCompleteBijection(inst, replica_count, num_partitions);
    if (bij.has_value() && bij->first == is_cross_partition) {
      if (is_involution && absl::MakeConstSpan(bij->second) == perm) {
        involution_cands.push_back(inst);
        continue;
      }
      if (!is_involution && absl::MakeConstSpan(bij->second) == perm_inv) {
        entries.push_back(inst);
        continue;
      }
      if (!is_involution && absl::MakeConstSpan(bij->second) == perm) {
        exits.push_back(inst);
        continue;
      }
    }
    if (inv_set.contains(inst)) {
      base_pinned.insert(inst);
      continue;
    }
    const bool in_ep =
        IsPureLocalDeterministicOp(inst, /*depth=*/0, custom_call_pred) ||
        IsOrthogonalCollective(inst, is_cross_partition, perm, replica_count,
                               num_partitions, custom_call_pred);
    if (!in_ep) {
      base_pinned.insert(inst);
    }
  }

  if (is_involution) {
    if (involution_cands.empty()) {
      return false;
    }
    PartitionInvolutionPermutes(post_order, involution_cands, base_pinned,
                                &entries, &exits);
  }
  if (entries.empty() && exits.empty()) {
    return false;
  }

  absl::flat_hash_set<const HloInstruction*> entry_set(entries.begin(),
                                                       entries.end());
  absl::flat_hash_set<const HloInstruction*> exit_set(exits.begin(),
                                                      exits.end());
  absl::flat_hash_set<const HloInstruction*> pinned = base_pinned;
  for (HloInstruction* y : entries) {
    pinned.insert(y);
    pinned.insert(y->operand(0));
  }
  for (HloInstruction* z : exits) {
    pinned.insert(z);
    for (const HloInstruction* u : z->users()) {
      pinned.insert(u);
    }
  }
  // Any non-permutable array producer (e.g. dynamic, zero-sized, host, token,
  // or sub-byte) cannot have a boundary CP inserted on its outgoing edges, so
  // keep its direct consumers outside S* as well.
  for (HloInstruction* inst : post_order) {
    if (!inst->shape().IsTuple() && !IsPermutableArrayShape(inst->shape())) {
      pinned.insert(inst);
      for (const HloInstruction* u : inst->users()) {
        pinned.insert(u);
      }
    }
  }

  // Assign Dinic node indices to all instructions in `comp`.
  const int source = 0;
  const int sink = 1;
  DinicMaxFlow flow(/*num_nodes=*/2);
  absl::flat_hash_map<const HloInstruction*, int> node_of;
  node_of.reserve(post_order.size());
  for (const HloInstruction* inst : post_order) {
    const int id = flow.AddNode();
    node_of[inst] = id;
    if (pinned.contains(inst)) {
      flow.AddEdge(id, sink, kInfCap);
    }
  }

  // Lock every tuple producer and all of its consumers together so a cut never
  // crosses a tuple-typed edge.
  for (const HloInstruction* inst : post_order) {
    if (!inst->shape().IsTuple()) {
      continue;
    }
    const int u_id = node_of.at(inst);
    for (const HloInstruction* user : inst->users()) {
      const int c_id = node_of.at(user);
      flow.AddEdge(u_id, c_id, kInfCap);
      flow.AddEdge(c_id, u_id, kInfCap);
    }
  }

  int64_t orig_cost = 0;
  // Gadget 2: Entry CPs y = CP_{P_inv}(x).
  for (const HloInstruction* y : entries) {
    const int64_t w =
        CollectivePermuteCost(y->shape(), per_collective_overhead_bytes);
    if (orig_cost > kInfCap / 4 - w) {
      return false;
    }
    orig_cost += w;
    const int b_y = flow.AddNode();
    flow.AddEdge(source, b_y, w);
    if (y == root || y->users().empty()) {
      flow.AddEdge(b_y, sink, kInfCap);
    }
    absl::flat_hash_set<const HloInstruction*> seen_users;
    for (const HloInstruction* c : y->users()) {
      if (seen_users.insert(c).second) {
        flow.AddEdge(b_y, node_of.at(c), kInfCap);
      }
    }
  }

  // Gadget 3: Exit CPs z = CP_P(u).
  for (const HloInstruction* z : exits) {
    const int64_t w =
        CollectivePermuteCost(z->shape(), per_collective_overhead_bytes);
    if (orig_cost > kInfCap / 4 - w) {
      return false;
    }
    orig_cost += w;
    flow.AddEdge(source, node_of.at(z->operand(0)), w);
  }

  if (orig_cost <= 0) {
    return false;
  }

  // Gadget 4: Regular array values v (excluding Entries, Exits, and entry
  // direction on I_P).
  std::vector<const HloInstruction*> reg_users;
  absl::flat_hash_set<const HloInstruction*> seen_reg;
  for (const HloInstruction* v : post_order) {
    if (entry_set.contains(v) || exit_set.contains(v) ||
        !v->shape().IsArray()) {
      continue;
    }
    reg_users.clear();
    seen_reg.clear();
    for (const HloInstruction* c : v->users()) {
      if (!exit_set.contains(c) && seen_reg.insert(c).second) {
        reg_users.push_back(c);
      }
    }
    const bool is_root_array = (v == root);
    if (reg_users.empty() && !is_root_array) {
      continue;
    }
    const int v_id = node_of.at(v);
    const int64_t w =
        CollectivePermuteCost(v->shape(), per_collective_overhead_bytes);
    if (!pinned.contains(v)) {
      const int a_out = flow.AddNode();
      flow.AddEdge(v_id, a_out, w);
      if (is_root_array) {
        flow.AddEdge(a_out, sink, kInfCap);
      }
      for (const HloInstruction* c : reg_users) {
        flow.AddEdge(a_out, node_of.at(c), kInfCap);
      }
    }
    if (!inv_set.contains(v)) {
      bool any_unpinned_user = false;
      for (const HloInstruction* c : reg_users) {
        if (!pinned.contains(c)) {
          any_unpinned_user = true;
          break;
        }
      }
      if (any_unpinned_user) {
        const int a_in = flow.AddNode();
        flow.AddEdge(a_in, pinned.contains(v) ? sink : v_id, w);
        for (const HloInstruction* c : reg_users) {
          if (!pinned.contains(c)) {
            flow.AddEdge(node_of.at(c), a_in, kInfCap);
          }
        }
      }
    }
  }

  const int64_t min_cut = flow.ComputeMaxFlow(source, sink, orig_cost);
  if (min_cut >= orig_cost) {
    return false;
  }
  const std::vector<bool> reachable = flow.SourceReachable(source);
  absl::flat_hash_set<HloInstruction*> in_s;
  for (HloInstruction* inst : post_order) {
    const int id = node_of.at(inst);
    if (id >= 0 && reachable.size() > static_cast<size_t>(id) &&
        reachable[id]) {
      if (pinned.contains(inst)) {
        return false;
      }
      in_s.insert(inst);
    }
  }
  if (in_s.empty()) {
    return false;
  }

  const std::vector<std::pair<int64_t, int64_t>> pairs_p =
      PermutationToPairs(perm);
  const std::vector<std::pair<int64_t, int64_t>> pairs_p_inv =
      PermutationToPairs(perm_inv);
  auto alloc_channel = [&]() -> std::optional<int64_t> {
    if (!is_cross_partition) {
      return std::nullopt;
    }
    return (*next_channel_id)++;
  };

  // Snapshot all operand rewrites against the pre-mutation graph so newly
  // rewired edges from Entry/Exit cancellations are never treated as original
  // Gadget-4 boundary crossings.
  struct SlotGroup {
    HloInstruction* producer;
    std::vector<std::pair<HloInstruction*, int64_t>> consumer_slots;
    bool updates_root = false;
  };
  std::vector<SlotGroup> entry_insertions;
  std::vector<SlotGroup> exit_insertions;
  std::vector<SlotGroup> entry_cancellations;

  for (HloInstruction* v : post_order) {
    if (entry_set.contains(v)) {
      SlotGroup grp{v, {}};
      for (HloInstruction* c : v->users()) {
        if (!in_s.contains(c)) {
          continue;
        }
        for (int64_t k = 0; k < c->operand_count(); ++k) {
          if (c->operand(k) == v) {
            grp.consumer_slots.emplace_back(c, k);
          }
        }
      }
      if (!grp.consumer_slots.empty()) {
        entry_cancellations.push_back(std::move(grp));
      }
      continue;
    }
    if (exit_set.contains(v) || !v->shape().IsArray()) {
      continue;
    }
    if (!in_s.contains(v)) {
      if (inv_set.contains(v)) {
        continue;
      }
      SlotGroup grp{v, {}};
      for (HloInstruction* c : v->users()) {
        if (!in_s.contains(c) || exit_set.contains(c)) {
          continue;
        }
        for (int64_t k = 0; k < c->operand_count(); ++k) {
          if (c->operand(k) == v) {
            grp.consumer_slots.emplace_back(c, k);
          }
        }
      }
      if (!grp.consumer_slots.empty()) {
        entry_insertions.push_back(std::move(grp));
      }
    } else {
      SlotGroup grp{v, {}, /*updates_root=*/(v == root)};
      for (HloInstruction* c : v->users()) {
        if (in_s.contains(c) || exit_set.contains(c)) {
          continue;
        }
        for (int64_t k = 0; k < c->operand_count(); ++k) {
          if (c->operand(k) == v) {
            grp.consumer_slots.emplace_back(c, k);
          }
        }
      }
      if (!grp.consumer_slots.empty() || grp.updates_root) {
        exit_insertions.push_back(std::move(grp));
      }
    }
  }

  int64_t inserted_cps = 0;
  int64_t cancelled_cps = 0;

  // 1. Insert boundary entry CPs `CP_P(v)` for non-invariant `v not in S*`.
  for (const SlotGroup& grp : entry_insertions) {
    HloInstruction* cp =
        comp->AddInstruction(HloInstruction::CreateCollectivePermute(
            grp.producer->shape(), grp.producer, pairs_p, alloc_channel()));
    if (is_cross_partition && grp.producer->has_sharding()) {
      cp->set_sharding(PermuteSharding(grp.producer->sharding(),
                                       grp.producer->shape(), perm));
    }
    for (const auto& [c, k] : grp.consumer_slots) {
      ABSL_RETURN_IF_ERROR(c->ReplaceOperandWith(k, cp));
    }
    ++inserted_cps;
  }

  // 2. Insert boundary exit CPs `CP_{P_inv}(v)` for `v in S*`.
  for (const SlotGroup& grp : exit_insertions) {
    HloInstruction* cp =
        comp->AddInstruction(HloInstruction::CreateCollectivePermute(
            grp.producer->shape(), grp.producer, pairs_p_inv, alloc_channel()));
    if (is_cross_partition && grp.producer->has_sharding()) {
      cp->set_sharding(grp.producer->sharding());
    }
    for (const auto& [c, k] : grp.consumer_slots) {
      ABSL_RETURN_IF_ERROR(c->ReplaceOperandWith(k, cp));
    }
    if (grp.updates_root) {
      comp->set_root_instruction(cp);
    }
    ++inserted_cps;
  }

  // 3. Cancel Entry CPs `y = CP_{P_inv}(x)` on consumers in `S*`.
  for (const SlotGroup& grp : entry_cancellations) {
    HloInstruction* y = grp.producer;
    HloInstruction* repl =
        MaybeAdaptLayout(comp, y->mutable_operand(0), y->shape());
    for (const auto& [c, k] : grp.consumer_slots) {
      ABSL_RETURN_IF_ERROR(c->ReplaceOperandWith(k, repl));
    }
    if (y->user_count() == 0 && y != comp->root_instruction()) {
      ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(y));
      ++cancelled_cps;
    }
  }

  // 4. Cancel Exit CPs `z = CP_P(u)` whose operand `u` entered `S*`.
  for (HloInstruction* z : exits) {
    HloInstruction* u = z->mutable_operand(0);
    if (!in_s.contains(u)) {
      continue;
    }
    HloInstruction* repl = MaybeAdaptLayout(comp, u, z->shape());
    ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(z, repl));
    ++cancelled_cps;
  }

  // 5. Update surviving shardings on conjugated instructions in `S*`.
  if (is_cross_partition) {
    for (HloInstruction* inst : post_order) {
      if (in_s.contains(inst) && inst->has_sharding()) {
        inst->set_sharding(
            PermuteSharding(inst->sharding(), inst->shape(), perm));
      }
    }
  }

  VLOG(1) << "CollectivePermuteEquivariantEliminator conjugated " << in_s.size()
          << " instructions in " << comp->name() << ": cancelled "
          << cancelled_cps << " CPs, inserted " << inserted_cps
          << " CPs, weighted cost " << orig_cost << " -> " << min_cut;
  return true;
}

// Collects parameter indices of non-entry computations called exclusively via
// `kCall` from `entry` whose passed argument at every callsite is in `entry`'s
// `I_P`.
absl::flat_hash_map<const HloComputation*, absl::flat_hash_set<int64_t>>
ComputeCallPropagatedInvariantParams(const HloModule* module,
                                     bool is_cross_partition,
                                     absl::Span<const int64_t> perm_inv) {
  absl::flat_hash_map<const HloComputation*, absl::flat_hash_set<int64_t>> out;
  if (!is_cross_partition || module->entry_computation() == nullptr) {
    return out;
  }
  const HloComputation* entry = module->entry_computation();
  const absl::flat_hash_set<const HloInstruction*> entry_inv =
      ComputeInvariantInstructions(entry, module, /*is_cross_partition=*/true,
                                   perm_inv, /*invariant_param_numbers=*/{});
  if (entry_inv.empty()) {
    return out;
  }
  absl::flat_hash_map<const HloComputation*, std::vector<const HloInstruction*>>
      call_sites;
  std::vector<const HloComputation*> ordered_callees;
  absl::flat_hash_set<const HloComputation*> non_call_callees;
  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      for (const HloComputation* callee : inst->called_computations()) {
        if (comp == entry && inst->opcode() == HloOpcode::kCall) {
          std::vector<const HloInstruction*>& sites = call_sites[callee];
          if (sites.empty()) {
            ordered_callees.push_back(callee);
          }
          sites.push_back(inst);
        } else {
          non_call_callees.insert(callee);
        }
      }
    }
  }
  for (const HloComputation* callee : ordered_callees) {
    const std::vector<const HloInstruction*>& calls = call_sites[callee];
    if (non_call_callees.contains(callee) || calls.empty()) {
      continue;
    }
    const int64_t num_params = callee->num_parameters();
    for (int64_t p = 0; p < num_params; ++p) {
      const bool all_inv =
          absl::c_all_of(calls, [&](const HloInstruction* call) {
            return p < call->operand_count() &&
                   entry_inv.contains(call->operand(p));
          });
      if (all_inv) {
        out[callee].insert(p);
      }
    }
  }
  return out;
}

}  // namespace

absl::StatusOr<bool> CollectivePermuteEquivariantEliminator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const int64_t replica_count = module->config().replica_count();
  const int64_t num_partitions = module->config().num_partitions();
  if (replica_count < 2 && num_partitions < 2) {
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  const absl::flat_hash_set<int64_t> kEmptyParams;
  bool module_changed = false;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    // Collect distinct canonical (is_cross_partition, perm) orientations in
    // deterministic order.
    absl::btree_map<std::pair<bool, std::vector<int64_t>>, std::vector<int64_t>>
        orientations;
    for (const HloInstruction* inst : comp->instructions()) {
      std::optional<std::pair<bool, std::vector<int64_t>>> bij =
          ExtractCompleteBijection(inst, replica_count, num_partitions);
      if (!bij.has_value()) {
        continue;
      }
      std::vector<int64_t> inv = InvertPermutation(bij->second);
      orientations.try_emplace({bij->first, bij->second}, inv);
      if (inv != bij->second) {
        orientations.try_emplace({bij->first, std::move(inv)},
                                 std::move(bij->second));
      }
    }
    if (orientations.empty()) {
      continue;
    }

    for (const auto& [key, perm_inv] : orientations) {
      const bool is_cross_partition = key.first;
      const std::vector<int64_t>& perm = key.second;
      absl::flat_hash_set<int64_t> propagated_params;
      if (is_cross_partition && comp != module->entry_computation()) {
        auto by_comp =
            ComputeCallPropagatedInvariantParams(module, true, perm_inv);
        if (const auto it = by_comp.find(comp); it != by_comp.end()) {
          propagated_params = std::move(it->second);
        }
      }
      const absl::flat_hash_set<int64_t>& inv_params =
          propagated_params.empty() ? kEmptyParams : propagated_params;

      ABSL_ASSIGN_OR_RETURN(
          const bool precancelled,
          PreCancelTrivialPermutes(comp, module, is_cross_partition, perm,
                                   perm_inv, replica_count, num_partitions,
                                   inv_params));
      module_changed |= precancelled;

      ABSL_ASSIGN_OR_RETURN(
          const bool conjugated,
          OptimizeOrientation(comp, module, is_cross_partition, perm, perm_inv,
                              replica_count, num_partitions,
                              per_collective_overhead_bytes_, inv_params,
                              is_equivariant_custom_call_, &next_channel_id));
      module_changed |= conjugated;
    }
  }

  return module_changed;
}

}  // namespace xla
