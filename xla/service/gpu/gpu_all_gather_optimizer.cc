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

#include "xla/service/gpu/gpu_all_gather_optimizer.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/reduce_scatter_utils.h"

namespace xla {
namespace gpu {
namespace {
// Structure to keep track of the subgraph to optimize
// "Left" and "Right" are conceptual. The initial
// reduce scatter is always considered "left" and "right"
// points to the other side of subgraph ending in a binary op

struct AllGatherReduceScatterSpec {
  HloInstruction* all_gather_left;
  HloInstruction* all_gather_right;
  HloInstruction* right_source;
  HloInstruction* binary_op;
};

std::optional<AllGatherReduceScatterSpec> MatchReduceScatter(
    const HloReduceScatterInstruction* rs, int64_t min_rank = 1) {
  if (!rs->shape().IsArray() || rs->constrain_layout() ||
      (rs->IsCrossModuleAllReduce() &&
       !rs->GetModule()->config().use_spmd_partitioning())) {
    VLOG(2) << "Unsupported reduce-scatter: " << rs->ToString();
    return std::nullopt;
  }
  if (rs->shape().rank() - absl::c_count(rs->shape().dimensions(), 1) <
      min_rank) {
    VLOG(2) << " Should be at least rank-" << min_rank
            << " excluding trivial dimensions " << rs->ToString();
    return std::nullopt;
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
      return std::nullopt;
    }
  }

  HloInstruction* user = rs->users()[0];
  if (user->opcode() != HloOpcode::kAllGather) {
    // ideally this should cover cases with other ops in between
    VLOG(2) << "Reduce-Scatter is not followed by all-gather " << user->ToString();
    return std::nullopt;
  }
  if (user->user_count() != 1) {
    // all-gather is subject to removal. Should not have more than 1 user
    VLOG(2) << "all-gather user_count > 1 " << user->ToString();
    return std::nullopt;
  }
  HloInstruction* all_gather_left = user;
  HloInstruction* binary_op;
  user = user->users().front();
  // the common node between the left and right branches
  // needs to be a binary op
  if (!HloOpcodeIsBinaryCommutative(user->opcode())) {
    VLOG(2) << "There is no binary op in the pipeline path, "
               "nothing to optimize: "
            << user->ToString();
    return std::nullopt;
  }
  binary_op = user;  // the end of "left" side branch
  HloInstruction* all_gather_right = (user->mutable_operand(0) == all_gather_left)
                                       ? user->mutable_operand(1)
                                       : user->mutable_operand(0);
  if (all_gather_right->opcode() != HloOpcode::kAllGather) {
    VLOG(2) << "Binary op's right operand is not all-gather "
            << all_gather_right->ToString();
    return std::nullopt;
  }
  // right side all-gather is also subject to removal
  // and should not contain more than 1 users
  if (all_gather_right->user_count() != 1) {
    VLOG(2) << "right side all-gather user_count > 1 "
            << all_gather_right->ToString();
    return std::nullopt;
  }
  HloInstruction* right_reduce_scatter = all_gather_right->mutable_operand(0);
  
  // we need to traverse the right branch backwards in search of 
  // a reduce scatter collective
  while (right_reduce_scatter->opcode() != HloOpcode::kReduceScatter &&
             right_reduce_scatter->operand_count() > 0 ) {
        right_reduce_scatter = right_reduce_scatter->mutable_operand(0);
      }

  if (right_reduce_scatter->opcode() != HloOpcode::kReduceScatter) {
    VLOG(2)
        << "Binary op's right operand path doesn not include reduce scatter "
        << right_reduce_scatter->ToString();
    return std::nullopt;
  }
  if (!ReplicaGroupsEqual(rs->replica_groups(),
                          right_reduce_scatter->replica_groups())) {
    VLOG(2)
        << "Reduce-Scatters in two branches don't have similar replica groups"
        << right_reduce_scatter->ToString();
    return std::nullopt;
  }
  AllGatherReduceScatterSpec spec;
  spec.all_gather_left = all_gather_left;
  spec.all_gather_right = all_gather_right;
  spec.binary_op = binary_op;
  spec.right_source = all_gather_right->mutable_operand(0);

  return spec;
}
}
StatusOr<bool> AllGatherOptimizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloModuleConfig& config = module->config();
  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kReduceScatter ||
          !instruction->shape().IsArray()) {
        continue;
      }
      auto* rs = Cast<HloReduceScatterInstruction>(instruction);
      // get the graph structure for which the all-gather ops
      // are combined into a single call after the binary op
      auto rs_spec = MatchReduceScatter(rs);

      if (!rs_spec) {
        VLOG(2) << "Cannot match all-gather combining optimization "
                << rs->ToString();
        continue;
      }
      auto index_in_full_shape =
          computation->AddInstruction(HloInstruction::CreateBinary(
              rs->shape(), rs_spec->binary_op->opcode(), rs,
              rs_spec->right_source));

      int64_t all_gather_dimension =
          Cast<HloAllGatherInstruction>(rs_spec->all_gather_left)
              ->all_gather_dimension();

      auto combined = HloInstruction::CreateAllGather(
          rs_spec->all_gather_left->shape(), {index_in_full_shape},
          all_gather_dimension, rs_spec->all_gather_left->replica_groups(),
          /*constrain_layout=*/false, rs_spec->all_gather_left->channel_id(),
          Cast<HloAllGatherInstruction>(rs_spec->all_gather_left)
              ->use_global_device_ids());

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          rs_spec->binary_op, std::move(combined)));

      TF_RETURN_IF_ERROR(
          computation->RemoveInstruction(rs_spec->all_gather_left));
      TF_RETURN_IF_ERROR(
          computation->RemoveInstruction(rs_spec->all_gather_right));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla