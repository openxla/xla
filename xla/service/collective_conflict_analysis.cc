/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/collective_conflict_analysis.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/xla_data.pb.h"

namespace xla {

void AbstractReplicaGroups::merge_groups(int64_t replica_id,
                                         int64_t other_replica_id) {
  if (replica_id < 0 || other_replica_id < 0 ||
      replica_id == other_replica_id) {
    return;
  }
  if (get_index(replica_id) == -1 && get_index(other_replica_id) == -1) {
    set_index(replica_id, groups.size());
    set_index(other_replica_id, groups.size());
    groups.push_back({replica_id, other_replica_id});
    return;
  }
  if (get_index(replica_id) == get_index(other_replica_id)) return;
  if (get_index(replica_id) == -1) {
    std::swap(replica_id, other_replica_id);
  }
  CHECK_NE(get_index(replica_id), -1);
  if (get_index(other_replica_id) == -1) {
    set_index(other_replica_id, get_index(replica_id));
    groups[get_index(replica_id)].insert(other_replica_id);
    return;
  }
  CHECK(get_index(replica_id) != -1 && get_index(other_replica_id) != -1 &&
        get_index(replica_id) != get_index(other_replica_id));
  auto& other_set = groups[get_index(other_replica_id)];
  for (int64_t replica_id_in_other_set : other_set) {
    groups[get_index(replica_id)].insert(replica_id_in_other_set);
    set_index(replica_id_in_other_set, get_index(replica_id));
  }
  other_set.clear();
}

bool IsConflictingAbstractReplicaGroups(AbstractReplicaGroups& lhs,
                                        AbstractReplicaGroups& rhs) {
  std::vector<int64_t> frequency(lhs.groups.size(), 0);
  for (auto& rhs_group : rhs.groups) {
    std::fill(frequency.begin(), frequency.end(), 0);
    for (int64_t rhs_replica_id : rhs_group) {
      if (rhs_replica_id < 0 ||
          rhs_replica_id >= static_cast<int64_t>(lhs.index_map.size())) {
        continue;
      }
      int64_t i = lhs.index_map[rhs_replica_id];
      if (i == -1) continue;
      if (++frequency[i] >= 2) return true;
    }
  }
  return false;
}

namespace {

bool IsDoneOrUpdate(HloOpcode op) {
  return op == HloOpcode::kAllReduceDone || op == HloOpcode::kAllGatherDone ||
         op == HloOpcode::kCollectivePermuteDone ||
         op == HloOpcode::kAsyncUpdate || op == HloOpcode::kAsyncDone;
}

void ExpandGroup(AbstractReplicaGroups& g, absl::Span<const int64_t> ids,
                 CollectiveOpGroupMode mode, int64_t r, int64_t p, bool* wild,
                 bool is_empty = false) {
  const bool cr = mode == COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  const bool cp = mode == COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION;
  const bool crp = mode == COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
  const bool fits = r <= 65536 / p;
  if (is_empty) {
    const int64_t n = cr ? r : (cp ? p : (fits ? r * p : 2));
    if (wild != nullptr && ((r == 1 && p == 1) || (!fits && n > 1))) {
      *wild = true;
    }
    for (int64_t o = 0; fits && o < (p == 1 ? 1 : (cr ? p : (cp ? r : 1)));
         ++o) {
      for (int64_t k = 1, b = cp ? o * p : o; k < n; ++k) {
        g.merge_groups(b, b + k * (cr ? p : 1));
      }
    }
    return;
  }
  for (int64_t i = 1; p == 1 && i < static_cast<int64_t>(ids.size()); ++i) {
    g.merge_groups(ids[0], ids[i]);
  }
  const int64_t lim = (cr || crp) ? r : (cp ? p : (fits ? r * p : -1));
  for (int64_t o = 0; p > 1 && o < (!fits ? 1 : (cr ? p : (cp ? r : 1))); ++o) {
    int64_t lead = -1;
    for (int64_t id : ids) {
      if (id < 0 || (lim >= 0 && id >= lim)) {
        continue;
      }
      if (!fits && wild != nullptr && (crp || (lead >= 0 && lead != id))) {
        *wild = true;
        return;
      }
      for (int64_t q = 0; fits && q < (crp ? p : 1); ++q) {
        int64_t v =
            cr ? id * p + o : (cp ? o * p + id : (crp ? id * p + q : id));
        if (lead >= 0) {
          g.merge_groups(lead, v);
        }
        lead = lead < 0 ? v : lead;
      }
      lead = !fits ? id : lead;
    }
  }
}

void AddInstructionGroups(const HloInstruction* instr,
                          AbstractReplicaGroups& groups, bool* wild) {
  const HloModule* m = instr != nullptr ? instr->GetModule() : nullptr;
  const int64_t r = m ? std::max<int64_t>(m->config().replica_count(), 1) : 1;
  const int64_t p = m ? std::max<int64_t>(m->config().num_partitions(), 1) : 1;
  while (instr != nullptr &&
         (DynCast<HloAsyncInstruction>(instr) != nullptr ||
          (IsDoneOrUpdate(instr->opcode()) && instr->operand_count() > 0))) {
    const auto* a = DynCast<HloAsyncInstruction>(instr);
    instr = a ? a->async_wrapped_instruction() : instr->operand(0);
  }
  if (instr == nullptr) {
    return;
  }
  if (const auto* cp = DynCast<HloCollectivePermuteInstruction>(instr)) {
    const auto mode = cp->channel_id()
                          ? COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION
                          : COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
    for (const auto& [i, j] : cp->source_target_pairs()) {
      ExpandGroup(groups, {i, j}, mode, r, p, wild);
    }
    return;
  }
  if (instr->opcode() == HloOpcode::kSend ||
      instr->opcode() == HloOpcode::kRecv) {
    auto* sr = Cast<HloSendRecvInstruction>(instr);
    CHECK(!sr->is_host_transfer());
    const std::string& source_target_pairs_str =
        sr->frontend_attributes().map().at(kSendRecvSourceTargetPairsAttr);
    absl::StatusOr<std::vector<ReplicaGroup>> source_target_pairs =
        ParseReplicaGroupsOnly(source_target_pairs_str);
    CHECK(source_target_pairs.ok() && "Expect valid source_target_pairs");
    if (auto mode = GetCollectiveOpGroupMode(sr->channel_id().value_or(0) > 0,
                                             std::nullopt);
        mode.ok()) {
      for (const ReplicaGroup& rg : *source_target_pairs) {
        ExpandGroup(groups, rg.replica_ids(), *mode, r, p, wild);
      }
    }
    return;
  }
  if (const auto* coll = DynCast<HloCollectiveInstruction>(instr)) {
    if (auto mode = GetCollectiveOpGroupMode(coll); mode.ok()) {
      const auto& rgs = coll->replica_groups();
      if (rgs.empty()) {
        ExpandGroup(groups, {}, *mode, r, p, wild, /*is_empty=*/true);
      }
      for (const ReplicaGroup& rg : rgs) {
        ExpandGroup(groups, rg.replica_ids(), *mode, r, p, wild);
      }
    }
  }
}

}  // namespace

void GetAbstractReplicaGroups(HloInstruction* instr,
                              AbstractReplicaGroups& groups) {
  AddInstructionGroups(instr, groups, /*wild=*/nullptr);
}

std::vector<HloInstruction*> FindAllConflictingCollectives(
    const HloComputation* computation,
    const std::vector<HloInstruction*>& seed_collectives) {
  absl::flat_hash_set<HloInstruction*> seen;
  AbstractReplicaGroups abstract_replica_groups_supremum;
  bool sup_wild = false;
  for (HloInstruction* instr : seed_collectives) {
    AddInstructionGroups(instr, abstract_replica_groups_supremum, &sup_wild);
    seen.insert(instr);
  }
  if (abstract_replica_groups_supremum.groups.empty() && !sup_wild) {
    return {};
  }

  std::vector<HloInstruction*> conflicing_collectives;
  bool fixpoint_reached;
  do {
    fixpoint_reached = true;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (!IsNonFusionCollective(instr) || IsDoneOrUpdate(instr->opcode()) ||
          seen.contains(instr)) {
        continue;
      }
      AbstractReplicaGroups groups;
      bool wild = false;
      AddInstructionGroups(instr, groups, &wild);
      if (wild || (sup_wild && !groups.groups.empty()) ||
          IsConflictingAbstractReplicaGroups(
              groups, abstract_replica_groups_supremum)) {
        conflicing_collectives.push_back(instr);
        AddInstructionGroups(instr, abstract_replica_groups_supremum,
                             &sup_wild);
        seen.insert(instr);
        fixpoint_reached = false;
      }
    }
  } while (!fixpoint_reached);

  return conflicing_collectives;
}

}  // namespace xla
