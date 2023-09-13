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

#include "xla/service/reduce_scatter_all_gather_combiner.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/hlo/utils/hlo_query.h"


namespace xla {

StatusOr<bool> ReduceScatterAllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloModuleConfig& config = module->config();
  bool changed = false;

  if (config.use_spmd_partitioning()) {
    VLOG(2) << "Unsupported module";
    return false;
  }
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kReduceScatter ||
          !instruction->shape().IsArray()) {
        continue;
      }

      auto* rs = Cast<HloReduceScatterInstruction>(instruction);
      if (rs->constrain_layout()) {
        VLOG(2) << " The layout is enforced by the XLA client: " << rs->ToString();
        continue;
      }
      if (rs->user_count() != 1) {
        VLOG(2) << "reduce-scatter user_count > 1 " << rs->ToString();
        continue;
      }

      HloInstruction* user = rs->users()[0];
      if (user->opcode() != HloOpcode::kAllGather) {
        VLOG(2) << "Reduce-Scatter is not followed by all-gather "
                << user->ToString();
        continue;
      }

      auto* ag = Cast<HloAllGatherInstruction>(user);
      // Check whether the Reduce-Scatter and the following
      // all-gather have matching properties.
      if (rs->constrain_layout() != ag->constrain_layout() ||
          rs->dimensions() != ag->dimensions() ||
          rs->use_global_device_ids() != ag->use_global_device_ids() ||
          !ReplicaGroupsEqual(rs->replica_groups(), ag->replica_groups())) {
        VLOG(2) << "The Reduce-Scatter and All-gather ops are not compatible "
                   "to merge. ";
        continue;
      }

      std::optional<int64_t> channel_id;
      if (rs->channel_id()) {
        // We cannot reuse the channel_id on reduce-scatter
        channel_id = next_channel_id++;
      }

      HloInstruction* source = rs->mutable_operand(0);
      auto combined = HloInstruction::CreateAllReduce(
          ag->shape(), {source}, rs->to_apply(), ag->replica_groups(),
          /*constrain_layout=*/false, channel_id, rs->use_global_device_ids());

      TF_RETURN_IF_ERROR(
          computation->ReplaceWithNewInstruction(user, std::move(combined)));

      TF_RETURN_IF_ERROR(computation->RemoveInstruction(rs));

      changed = true;
    }
  }
  return changed;
}
}  // namespace xla
