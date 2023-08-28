/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_all_gather_combiner.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"

namespace xla {
namespace gpu {

StatusOr<bool> ReduceScatterAllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloModuleConfig& config = module->config();
  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  const int64_t min_rank = 1;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kReduceScatter ||
          !instruction->shape().IsArray()) {
        continue;
      }
      auto* rs = Cast<HloReduceScatterInstruction>(instruction);

      if (!rs->shape().IsArray() || rs->constrain_layout() ||
          (rs->IsCrossModuleAllReduce() &&
           !rs->GetModule()->config().use_spmd_partitioning())) {
        VLOG(2) << "Unsupported reduce-scatter: " << rs->ToString();
        continue;
      }
      if (rs->shape().rank() - absl::c_count(rs->shape().dimensions(), 1) <
          min_rank) {
        VLOG(2) << " Should be at least rank-" << min_rank
                << " excluding trivial dimensions " << rs->ToString();
        continue;
      }
      if (rs->replica_groups().size() > 1) {
        const int64_t size = rs->replica_groups()[0].replica_ids_size();
        absl::Span<const ReplicaGroup> rgs = rs->replica_groups();
        const bool has_uniform_size = absl::c_all_of(
            rgs.subspan(1, size - 1), [size](const ReplicaGroup& group) {
              return group.replica_ids_size() == size;
            });
        if (!has_uniform_size) {
          VLOG(2) << "Unsupported non-uniform replica group size "
                  << rs->ToString();
          continue;
        }
      }
      if (rs->user_count() != 1) {
        // reduce-scatter Should not have more than 1 users
        VLOG(2) << "reduce-scatter user_count > 1 " << rs->ToString();
        continue;
      }

      HloInstruction* user = rs->users()[0];
      if (user->opcode() != HloOpcode::kAllGather) {
        // The next immediate op should be AllGather
        VLOG(2) << "Reduce-Scatter is not followed by all-gather "
                << user->ToString();
        continue;
      }
      HloInstruction* source = rs->mutable_operand(0);

      std::optional<int64_t> channel_id;
      if (rs->channel_id()) {
        // We cannot reuse the channel_id on reduce-scatter
        channel_id = next_channel_id++;
      }

      auto combined = HloInstruction::CreateAllReduce(
          user->shape(), {source}, rs->to_apply(), user->replica_groups(),
          /*constrain_layout=*/false, channel_id, rs->use_global_device_ids());

      TF_RETURN_IF_ERROR(
          computation->ReplaceWithNewInstruction(user, std::move(combined)));

      TF_RETURN_IF_ERROR(computation->RemoveInstruction(rs));

      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
