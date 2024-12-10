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

#include "xla/hlo/transforms/collectives/rematerialize_large_all_gather.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"

namespace xla {

std::pair<bool, std::vector<ReplicaGroup>>
RematerializeLargeAllGather::GetTpReplicaGroup(HloComputation* computation) {
  //   Use pattern matcher to find TP replica group pattern. It finds
  //  reduce-scatter that matches either reduce-scatter(dot) or
  // reduce-scatter(any(dot)) patterns. The replica group is used to determine
  // whether to combine a all-gather or not.

  bool tp_replica_group_found = false;
  std::vector<ReplicaGroup> tp_replica_group;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (Match(instruction,
              match::ReduceScatter().WithOperand(
                  0, match::AnyOf<HloInstruction>(
                         match::Op().WithOpcode(HloOpcode::kDot),
                         match::Op()
                             .WithOpcode(HloOpcode::kReshape)
                             .WithOperand(0, match::Op().WithOpcode(
                                                 HloOpcode::kDot)))))) {
      tp_replica_group = instruction->replica_groups();
      tp_replica_group_found = true;
    }
    if (tp_replica_group_found) break;
  }
  return std::make_pair(tp_replica_group_found, tp_replica_group);
}

std::pair<bool, std::vector<ReplicaGroup>>
RematerializeLargeAllGather::GetTpReplicaGroup(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<ReplicaGroup> tp_replica_group;
  bool tp_replica_group_found = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    auto rval = GetTpReplicaGroup(computation);
    if (rval.first) {
      tp_replica_group = rval.second;
      tp_replica_group_found = true;
      break;
    }
  }
  return std::make_pair(tp_replica_group_found, tp_replica_group);
}

bool RematerializeLargeAllGather::IsRemattableAllGather(
    std::vector<ReplicaGroup> tp_replica_group, HloInstruction* input_inst) {
  if (disable_pattern_match_ && !tp_replica_group.empty()) {
    VLOG(2) << "Matched all-gather with replica group "
            << input_inst->ToString();
    return hlo_query::HasMatchingReplicaGroups(input_inst, tp_replica_group);
  } else {
    int64_t ag_numel = 1;
    for (int64_t dim : input_inst->shape().dimensions()) {
      ag_numel *= dim;
    }
    const int64_t ag_size = ag_numel * ShapeUtil::ByteSizeOfPrimitiveType(
                                           input_inst->shape().element_type());
    VLOG(2) << "Matched all-gather based on size " << ag_size;
    return ag_size >= remat_size_in_bytes_;
  }
}
absl::StatusOr<bool> RematerializeLargeAllGather::RematAllGather(
    HloInstruction* all_gather_inst, HloInstruction* gte,
    HloInstruction* opt_barrier) {
  VLOG(2) << "Will remat all-gather: " << all_gather_inst->ToString();
  int64_t tuple_index = gte->tuple_index();
  HloComputation* computation = all_gather_inst->parent();
  // Replace the operand of the optimization barrier's tuple with the
  // AllGather input
  TF_RETURN_IF_ERROR(
      opt_barrier->mutable_operand(0)->ReplaceOperandWithDifferentShape(
          tuple_index, all_gather_inst->mutable_operand(0)));
  // Update the shape of the optimization barrier's tuple to match the
  // new operand shape
  Shape new_tuple_shape = opt_barrier->operand(0)->shape();
  *new_tuple_shape.mutable_tuple_shapes(tuple_index) =
      all_gather_inst->mutable_operand(0)->shape();
  *opt_barrier->mutable_operand(0)->mutable_shape() = new_tuple_shape;
  // Update the shape of the optimization barrier to match the new
  // tuple shape
  *opt_barrier->mutable_shape() = new_tuple_shape;
  // Replace the GetTupleElement with a new GetTupleElement from the
  // updated OptimizationBarrier
  HloInstruction* new_gte =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          all_gather_inst->mutable_operand(0)->shape(), opt_barrier,
          tuple_index));

  HloAllGatherInstruction* all_gather =
      Cast<HloAllGatherInstruction>(all_gather_inst);
  // Launch the AllGather on the new GetTupleElement
  HloInstruction* new_allgather =
      computation->AddInstruction(HloInstruction::CreateAllGather(
          all_gather->shape(), {new_gte}, all_gather->all_gather_dimension(),
          all_gather->replica_groups(), all_gather->constrain_layout(),
          all_gather->channel_id(), all_gather->use_global_device_ids()));
  // Replace the original GetTupleElement with the new AllGather
  TF_RETURN_IF_ERROR(gte->ReplaceAllUsesWith(new_allgather));
  return true;
}

absl::StatusOr<bool> RematerializeLargeAllGather::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  bool tp_replica_group_found = false;
  std::vector<ReplicaGroup> tp_replica_group;
  if (!disable_pattern_match_) {
    auto rval = GetTpReplicaGroup(module, execution_threads);
    if (rval.first) {
      tp_replica_group_found = true;
      tp_replica_group = rval.second;
    }
  }

  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == HloOpcode::kGetTupleElement) {
        HloInstruction* opt_barrier = inst->mutable_operand(0);
        if (opt_barrier->opcode() == HloOpcode::kOptimizationBarrier) {
          int64_t tuple_index = inst->tuple_index();
          HloInstruction* input_inst =
              opt_barrier->mutable_operand(0)->mutable_operand(tuple_index);
          if (input_inst->opcode() == HloOpcode::kAllGather &&
              IsRemattableAllGather(tp_replica_group, input_inst)) {
            absl::StatusOr<bool> result =
                RematAllGather(input_inst, inst, opt_barrier);
            CHECK(result.ok())
                << "Failed to remat all-gather " << input_inst->ToString();
            changed |= *result;
          }
        }
      }
    }
  }

  return changed;
}

}  // namespace xla