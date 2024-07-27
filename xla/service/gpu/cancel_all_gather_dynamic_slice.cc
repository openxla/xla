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
#include "xla/service/gpu/cancel_all_gather_dynamic_slice.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/service/collective_opt_utils.h"

namespace xla {
bool CancelAllGatherDynamicSlice::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDynamicSlice) {
    return false;
  }
  // Two cases: dynamic-slice(reshape(all-gather)) or dynamic-slice(all-gather)
  if ((instruction->mutable_operand(0)->opcode() != HloOpcode::kReshape && instruction->mutable_operand(0)->opcode() != HloOpcode::kAllGather) || 
       (instruction->mutable_operand(0)->opcode() == HloOpcode::kReshape && instruction->mutable_operand(0)->mutable_operand(0)->opcode() != HloOpcode::kAllGather)) {
    return false;
  }
  const HloModuleConfig &config = instruction->GetModule()->config();
  HloInstruction* operand = instruction->mutable_operand(0);
  if (instruction->mutable_operand(0)->opcode() == HloOpcode::kReshape){
    operand = instruction->mutable_operand(0)->mutable_operand(0);
  } 
  HloAllGatherInstruction* all_gather =
      Cast<HloAllGatherInstruction>(operand);
  bool match = AllGatherDynamicSliceCancellation(
    all_gather, config.num_partitions(),
    config.replica_count(), /*allow_multiple_split_dims=*/true,
    /*allow_intervening_reshape=*/true, /*min_rank=*/1,
    HloPredicateIsOp<HloOpcode::kPartitionId>,
    HloPredicateIsOp<HloOpcode::kReplicaId>,
    /*allow_intervening_bitcast=*/false,
    /*allow_multiple_users=*/true);

  return match;
}

StatusOr<HloInstruction*> CancelAllGatherDynamicSlice::ExpandInstruction(
    HloInstruction* instruction) {
  if(instruction->mutable_operand(0)->opcode() != HloOpcode::kReshape){
    // dynamic-slice(all-gather) case
    return instruction->mutable_operand(0)->mutable_operand(0);
  }
  // dynamic-slice(reshape(all-gather)) case
  auto* reshape = instruction->parent()->AddInstruction(HloInstruction::CreateReshape(instruction->shape(), 
        instruction->mutable_operand(0)->mutable_operand(0)->mutable_operand(0)));
  return reshape;
}

}  // namespace xla