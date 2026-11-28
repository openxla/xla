/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/transforms/collectives/all_reduce_key.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsFoldableReduction(const HloComputation* comp, PrimitiveType operand_type,
                         bool require_idempotent) {
  if (comp == nullptr || comp->num_parameters() != 2 ||
      comp->instruction_count() != 3) {
    return false;
  }
  const HloInstruction* p0 = comp->parameter_instruction(0);
  const HloInstruction* p1 = comp->parameter_instruction(1);
  const HloInstruction* root = comp->root_instruction();
  if (root->operand_count() != 2 ||
      !ShapeUtil::IsEffectiveScalar(root->shape()) ||
      !ShapeUtil::Equal(p0->shape(), root->shape()) ||
      !ShapeUtil::Equal(p1->shape(), root->shape())) {
    return false;
  }
  if (!((root->operand(0) == p0 && root->operand(1) == p1) ||
        (root->operand(0) == p1 && root->operand(1) == p0))) {
    return false;
  }
  const PrimitiveType type = root->shape().element_type();
  if (type != operand_type &&
      !(primitive_util::IsFloatingPointType(type) &&
        primitive_util::IsFloatingPointType(operand_type))) {
    return false;
  }
  switch (root->opcode()) {
    case HloOpcode::kMinimum:
    case HloOpcode::kMaximum:
      return !primitive_util::IsComplexType(type);
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
      return type == PRED || primitive_util::IsIntegralType(type);
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
      return !require_idempotent;
    case HloOpcode::kXor:
      return !require_idempotent &&
             (type == PRED || primitive_util::IsIntegralType(type));
    default:
      return false;
  }
}

// Folds the given two sets of non-empty replica groups into a single set if
// possible.
std::optional<std::vector<ReplicaGroup>> FoldReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups0,
    absl::Span<const ReplicaGroup> replica_groups1) {
  if (replica_groups0.empty() || replica_groups1.empty()) {
    return std::nullopt;
  }

  // Assign each replica listed in replica_groups0 a dense index and record its
  // group number, rejecting empty subgroups, negative IDs, and duplicates.
  absl::flat_hash_map<int64_t, int64_t> id_to_dense;
  std::vector<int64_t> replica_group_no;
  for (size_t group_no = 0; group_no < replica_groups0.size(); ++group_no) {
    const auto& ids = replica_groups0[group_no].replica_ids();
    if (ids.empty()) {
      return std::nullopt;
    }
    for (int64_t id : ids) {
      if (id < 0) {
        return std::nullopt;
      }
      const int64_t dense = static_cast<int64_t>(replica_group_no.size());
      if (!id_to_dense.emplace(id, dense).second) {
        return std::nullopt;
      }
      replica_group_no.push_back(static_cast<int64_t>(group_no));
    }
  }
  const int64_t num_listed = static_cast<int64_t>(replica_group_no.size());

  // Trace each replica_groups1 subgroup to its sorted set of contributing
  // replica_groups0 indices. Each contributor group may appear at most once.
  absl::flat_hash_map<std::vector<int64_t>, int64_t> contributor_set_id;
  std::vector<std::vector<int64_t>> unique_group0_sets;
  std::vector<int64_t> contributing_set_id(num_listed, 0);
  int64_t covered = 0;

  for (const ReplicaGroup& rg : replica_groups1) {
    const auto& ids = rg.replica_ids();
    if (ids.empty()) {
      return std::nullopt;
    }
    std::vector<int64_t> group0_ids;
    std::vector<int64_t> dense_ids;
    group0_ids.reserve(ids.size());
    dense_ids.reserve(ids.size());
    for (int64_t id : ids) {
      auto it = id_to_dense.find(id);
      if (it == id_to_dense.end() || contributing_set_id[it->second] != 0) {
        return std::nullopt;
      }
      dense_ids.push_back(it->second);
      group0_ids.push_back(replica_group_no[it->second]);
    }
    absl::c_sort(group0_ids);
    if (absl::c_adjacent_find(group0_ids) != group0_ids.end()) {
      return std::nullopt;
    }

    int64_t set_id = 0;
    if (auto it = contributor_set_id.find(group0_ids);
        it != contributor_set_id.end()) {
      set_id = it->second;
    } else {
      set_id = static_cast<int64_t>(unique_group0_sets.size()) + 1;
      contributor_set_id.emplace(group0_ids, set_id);
      unique_group0_sets.push_back(std::move(group0_ids));
    }
    for (int64_t dense : dense_ids) {
      contributing_set_id[dense] = set_id;
      ++covered;
    }
  }
  if (covered != num_listed) {
    return std::nullopt;
  }

  // Verify every contributing replica belongs exclusively to the same folded
  // group and emit deterministically sorted replica IDs and groups.
  std::vector<int64_t> group0_owner(replica_groups0.size(), 0);
  std::vector<ReplicaGroup> new_replica_groups;
  new_replica_groups.reserve(unique_group0_sets.size());
  for (size_t i = 0; i < unique_group0_sets.size(); ++i) {
    const int64_t set_id = static_cast<int64_t>(i) + 1;
    std::vector<int64_t> merged_ids;
    for (int64_t g0 : unique_group0_sets[i]) {
      if (group0_owner[g0] != 0) {
        return std::nullopt;
      }
      group0_owner[g0] = set_id;
      for (int64_t id : replica_groups0[g0].replica_ids()) {
        if (contributing_set_id[id_to_dense.at(id)] != set_id) {
          return std::nullopt;
        }
        merged_ids.push_back(id);
      }
    }
    absl::c_sort(merged_ids);
    ReplicaGroup& group = new_replica_groups.emplace_back();
    for (int64_t id : merged_ids) {
      group.add_replica_ids(id);
    }
  }

  absl::c_sort(new_replica_groups,
               [](const ReplicaGroup& a, const ReplicaGroup& b) {
                 return a.replica_ids(0) < b.replica_ids(0);
               });
  return new_replica_groups;
}

}  // namespace

absl::StatusOr<bool> AllReduceFolder::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip AllReduceFolder because the module contains all-reduce "
               "with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllReduce ||
          inst->operand_count() != 1) {
        continue;
      }
      HloInstruction* producer = inst->mutable_operand(0);
      if (producer->opcode() != HloOpcode::kAllReduce ||
          producer->operand_count() != 1 || producer->user_count() != 1 ||
          producer->IsRoot()) {
        continue;
      }

      auto* ar0 = Cast<HloAllReduceInstruction>(producer);
      auto* ar1 = Cast<HloAllReduceInstruction>(inst);
      if (!ar0->shape().IsArray() ||
          !ShapeUtil::Equal(ar0->shape(), ar1->shape()) ||
          ar0->raw_backend_config_string() !=
              ar1->raw_backend_config_string()) {
        continue;
      }

      // Check if the 2 all-reduce instructions are compatible with the
      // exception of the replica groups.
      std::optional<AllReduceKey> key0 =
          GetAllReduceKey(ar0, /*ignore_replica_groups=*/true);
      std::optional<AllReduceKey> key1 =
          GetAllReduceKey(ar1, /*ignore_replica_groups=*/true);
      if (!key0 || !key1 || *key0 != *key1 || ar0->replica_groups().empty() ||
          ar1->replica_groups().empty()) {
        continue;
      }

      absl::StatusOr<CollectiveOpGroupMode> group_mode =
          GetCollectiveOpGroupMode(ar0->channel_id().has_value(),
                                   ar0->use_global_device_ids());
      if (!group_mode.ok()) {
        continue;
      }
      // In CROSS_REPLICA_AND_PARTITION mode, each replica group includes all
      // partitions, so folding two all-reduces repeats every input once per
      // partition and is only sound for idempotent reductions when
      // num_partitions > 1.
      const bool require_idempotent =
          *group_mode ==
              CollectiveOpGroupMode::
                  COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION &&
          module->config().num_partitions() > 1;
      const PrimitiveType elem_type = ar0->shape().element_type();
      if (!IsFoldableReduction(ar0->to_apply(), elem_type,
                               require_idempotent) ||
          !IsFoldableReduction(ar1->to_apply(), elem_type,
                               require_idempotent)) {
        continue;
      }

      std::optional<std::vector<ReplicaGroup>> new_replica_groups =
          FoldReplicaGroups(ar0->replica_groups(), ar1->replica_groups());
      if (!new_replica_groups) {
        continue;
      }
      std::optional<int64_t> channel_id;
      if (ar0->channel_id()) {
        channel_id = next_channel_id++;
      }

      // Create new all-reduce and delete the 2 existing ones.
      HloInstruction* new_ar =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              ar0->shape(), ar0->operands(), ar0->to_apply(),
              std::make_shared<CollectiveDeviceList>(*new_replica_groups),
              /*constrain_layout=*/false, channel_id,
              ar0->use_global_device_ids()));
      ar1->SetupDerivedInstruction(new_ar);
      if (ar1->metadata().ByteSizeLong() == 0 &&
          ar0->metadata().ByteSizeLong() != 0) {
        new_ar->set_metadata(ar0->metadata());
      }

      // Forward frontend attributes from folded instructions.
      std::vector<HloInstruction*> folded_all_reduces = {ar0, ar1};
      new_ar->set_frontend_attributes(
          MergeFrontendAttributes(folded_all_reduces));

      // Folding a serial chain does not combine independent collective
      // buffers, so group keys do not gate this optimization. Preserve the key
      // when it has one unambiguous representative; otherwise
      // drop it instead of creating a synthetic comma-separated key.
      const bool ar0_is_keyed = HasCollectiveGroupKey(*ar0);
      const bool ar1_is_keyed = HasCollectiveGroupKey(*ar1);
      if (ar0_is_keyed && !ar1_is_keyed) {
        CopyCollectiveGroupKey(*ar0, *new_ar);
      } else if (!ar0_is_keyed && ar1_is_keyed) {
        CopyCollectiveGroupKey(*ar1, *new_ar);
      } else if (ar0_is_keyed &&
                 HaveCompatibleCollectiveGroupKeys(*ar0, *ar1)) {
        CopyCollectiveGroupKey(*ar0, *new_ar);
      } else if (ar0_is_keyed) {
        ClearCollectiveGroupKey(*new_ar);
      }

      ABSL_RETURN_IF_ERROR(ar1->ReplaceAllUsesWith(new_ar));
      if (module->has_schedule() &&
          module->schedule().is_computation_scheduled(computation)) {
        HloInstructionSequence& sequence =
            module->schedule().GetOrCreateSequence(computation);
        if (sequence.contains(ar1)) {
          sequence.replace_instruction(ar1, new_ar);
        }
        sequence.remove_instruction(ar0);
      }
      ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
