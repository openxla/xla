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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_BACKEND_ASSIGNER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_BACKEND_ASSIGNER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

class CollectiveBackendAssigner : public HloModulePass {
 public:
  explicit CollectiveBackendAssigner(int64_t threshold_in_bytes)
      : threshold_in_bytes_(threshold_in_bytes) {}

  absl::string_view name() const override {
    return "collective-backend-assigner";
  }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  static bool HasInternodeCommunication(
      const std::vector<ReplicaGroup>& replica_groups, int64_t num_processes) {
    absl::flat_hash_set<int64_t> nodes;
    for (const auto& group : replica_groups) {
      nodes.clear();
      for (int64_t replica_id : group.replica_ids()) {
        nodes.insert(replica_id / num_processes);
      }
      if (nodes.size() > 1) {
        VLOG(1) << "Found internode communication in replica groups";
        return true;
      }
    }
    VLOG(1) << "Found intranode communication in replica groups";
    return false;
  }

  static bool HasInternodeCommunication(
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      int64_t num_processes) {
    for (const auto& pair : source_target_pairs) {
      int64_t source_node = pair.first / num_processes;
      int64_t target_node = pair.second / num_processes;
      if (source_node != target_node) {
        VLOG(1) << "Found internode communication between source node "
                << source_node << " and target node " << target_node;
        return true;
      }
    }
    VLOG(1) << "Found intranode communication in source-target pairs";
    return false;
  }

  static bool HasInternodeCommunication(const HloInstruction& instr,
                                        int64_t num_processes) {
    if (instr.opcode() == HloOpcode::kAllReduce ||
        instr.opcode() == HloOpcode::kAllReduceStart ||
        instr.opcode() == HloOpcode::kAllReduceDone) {
      return HasInternodeCommunication(instr.replica_groups(), num_processes);
    }
    if (instr.opcode() == HloOpcode::kCollectivePermute ||
        instr.opcode() == HloOpcode::kCollectivePermuteStart ||
        instr.opcode() == HloOpcode::kCollectivePermuteDone) {
      return HasInternodeCommunication(instr.source_target_pairs(),
                                       num_processes);
    }
    return false;
  }

  static bool IsCollectiveOp(const HloInstruction* instr) {
    return HloPredicateIsOp<HloOpcode::kAllReduce>(instr) ||
           HloPredicateIsOp<HloOpcode::kAllReduceStart>(instr) ||
           HloPredicateIsOp<HloOpcode::kAllReduceDone>(instr) ||
           HloPredicateIsOp<HloOpcode::kCollectivePermute>(instr) ||
           HloPredicateIsOp<HloOpcode::kCollectivePermuteStart>(instr) ||
           HloPredicateIsOp<HloOpcode::kCollectivePermuteDone>(instr);
  }

 private:
  static int64_t GetShapeSize(const Shape& shape) {
    int64_t size_in_bytes = 0;
    if (shape.IsTuple()) {
      for (int64_t i = 0; i < shape.tuple_shapes_size(); ++i) {
        size_in_bytes += GetShapeSize(shape.tuple_shapes(i));
      }
      return size_in_bytes;
    }
    return ShapeUtil::ByteSizeOfElements(shape);
  }

  int64_t threshold_in_bytes_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_BACKEND_ASSIGNER_H_
