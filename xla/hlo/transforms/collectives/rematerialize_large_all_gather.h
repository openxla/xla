/* Copyright 2024 The OpenXLA Authors.

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

#ifndef REMATERIALIZE_LARGE_ALLGATHER_H_
#define REMATERIALIZE_LARGE_ALLGATHER_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {

//    A pass that attempts to rematerialize large SPMD partitioner inserted
//    tensor
//  parallel all-gathers. The pass has two options for use:
//  1. The pass can look for the reduce-scatter(dot(a,b)) pattern to find the
//     tensor parallel replica group and remat all matching all-gathers.
// 2. The pass accepts a threshold in bytes in which it will remat all
//  all-gathers greater than or equal.
//
//  The pass also assumes the opt-barrier for activation checkpointing is
//  already
// present by Jax.
//
//  By default, it will use the pattern matcher.

class RematerializeLargeAllGather : public HloModulePass {
 public:
  explicit RematerializeLargeAllGather(int64_t remat_size_in_bytes = 4096 *
                                                                     4096 * 2,
                                       bool disable_pattern_match = false)
      : remat_size_in_bytes_(remat_size_in_bytes),
        disable_pattern_match_(disable_pattern_match) {}

  absl::string_view name() const override {
    return "rematerialize-large-allgather";
  }

  using HloPassInterface::Run;

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  bool IsRemattableAllGather(std::vector<ReplicaGroup> tp_replica_group,
                             HloInstruction* input_inst);

  std::pair<bool, std::vector<ReplicaGroup>> GetTpReplicaGroup(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  std::pair<bool, std::vector<ReplicaGroup>> GetTpReplicaGroup(
      HloComputation* computation);

  absl::StatusOr<bool> RematAllGather(HloInstruction* all_gather,
                                      HloInstruction* gte,
                                      HloInstruction* opt_barrier);

 private:
  const int64_t remat_size_in_bytes_;
  const bool disable_pattern_match_;
};

}  // namespace xla

#endif  // REMATERIALIZE_LARGE_ALLGATHER_H_