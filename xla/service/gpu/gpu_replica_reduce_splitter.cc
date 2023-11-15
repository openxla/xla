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

#include "xla/service/gpu/gpu_replica_reduce_splitter.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

namespace m = match;

// Returns whether the HLO Computation applied by `op` calculates the largest
// element.
bool IsMaxReduce(HloInstruction *op) {
  if (op->opcode() != HloOpcode::kReduce) {
    return false;
  }
  HloComputation *reduce_comp = op->to_apply();
  HloInstruction *reduce_comp_root = reduce_comp->root_instruction();
  return ShapeUtil::IsScalar(op->shape()) &&
         ShapeUtil::IsScalar(op->operand(1)->shape()) &&
         op->operand(1)->IsConstant() &&
         op->operand(1)->literal().GetAsDouble({}) <= 0. &&
         reduce_comp_root->opcode() == HloOpcode::kMaximum &&
         reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
         reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter;
}

// Recursively find partition-id node.
HloInstruction *FindPartitionIdRecursive(
    HloInstruction *instr, absl::flat_hash_set<int> &visited_instrs) {
  // Avoid visiting the same instruction more than once.
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return nullptr;
  }
  if (instr->opcode() == HloOpcode::kPartitionId) {
    return instr;
  }
  if (instr->operand_count() == 1 || instr->opcode() == HloOpcode::kClamp) {
    int operand_idx = 0;
    if (instr->opcode() == HloOpcode::kClamp) {
      operand_idx = 1;
    }
    return FindPartitionIdRecursive(instr->mutable_operand(operand_idx),
                                    visited_instrs);
  } else if (instr->opcode() == HloOpcode::kMultiply ||
             instr->opcode() == HloOpcode::kDynamicSlice) {
    for (int k = 0; k < 2; ++k) {
      int tmp_k = instr->opcode() == HloOpcode::kDynamicSlice ? k + 1 : k;
      auto binary_subgraph = FindPartitionIdRecursive(
          instr->mutable_operand(tmp_k), visited_instrs);
      if (binary_subgraph) {
        return binary_subgraph;
      }
    }
  }
  return nullptr;
}

StatusOr<bool> ReplicaReduceSplitter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  const HloModuleConfig &config = module->config();

  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  auto is_replicated_parameter = [](const HloInstruction *instr) -> bool {
    return instr->operand_count() == 0 && instr->sharding().IsReplicated();
  };
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloInstruction *sharded_param = nullptr;
    HloInstruction *partition_id = nullptr;
    for (HloInstruction *instr : computation->MakeInstructionPostOrder()) {
      if (IsMaxReduce(instr) &&
          Match(instr->mutable_operand(0),
                m::Abs(m::Op(&sharded_param)
                           .WithPredicate(is_replicated_parameter)))) {
        int dynamic_slice_id = -1;
        for (int i = 0; i < sharded_param->users().size(); ++i) {
          if (sharded_param->users()[i]->opcode() == HloOpcode::kDynamicSlice) {
            dynamic_slice_id = i;
            break;
          }
        }
        if (dynamic_slice_id == -1) {
          VLOG(2) << "Replicated parameter has not been sliced so "
                     "ReplicaReduceSplitter pass not applied.";
          return false;
        }

        auto dynamic_slice_op = sharded_param->users()[dynamic_slice_id];
        absl::flat_hash_set<int> visited_instrs;
        partition_id =
            FindPartitionIdRecursive(dynamic_slice_op, visited_instrs);
        int num_partitions = instr->GetModule()->config().num_partitions();
        const Shape param_shape = sharded_param->shape();

        // Ensure that slicing is performed on the most major dimension
        int most_major_dim =
            param_shape.layout().minor_to_major(param_shape.rank() - 1);
        size_t num_row = param_shape.dimensions(most_major_dim);

        Shape dynamic_slice_shape =
            ShapeUtil::MakeShape(partition_id->shape().element_type(), {1});

        Array<uint32_t> gmap_0({num_partitions});
        gmap_0.FillIota(0);

        HloInstruction *constant_0 =
            instr->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateFromArray(gmap_0)));

        HloInstruction *dynamic_slice_0 =
            instr->AddInstruction(HloInstruction::CreateDynamicSlice(
                dynamic_slice_shape, constant_0, {partition_id}, {1}));
        HloInstruction *convert_0 =
            instr->AddInstruction(HloInstruction::CreateConvert(
                ShapeUtil::MakeShape(S32, {1}), dynamic_slice_0));

        Array<int32_t> gmap_1({num_partitions});
        int32_t start_ind = 0;
        auto loc_row = static_cast<int32_t>(num_row / num_partitions);
        std::generate(gmap_1.begin() + 1, gmap_1.end(),
                      [&] { return start_ind += loc_row; });
        auto constant_1 = instr->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateFromArray(gmap_1)));
        HloInstruction *dynamic_slice_1 =
            instr->AddInstruction(HloInstruction::CreateDynamicSlice(
                ShapeUtil::MakeShape(constant_1->shape().element_type(), {1}),
                constant_1, {convert_0}, {1}));

        HloInstruction *reshape_0 =
            instr->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::MakeShape(dynamic_slice_1->shape().element_type(),
                                     {}),
                dynamic_slice_1));

        HloInstruction *start_from_zero = instr->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
        auto sliced_shape = sharded_param->shape();
        sliced_shape.set_dimensions(most_major_dim, loc_row);
        int param_rank = sharded_param->shape().rank();
        std::vector<int64_t> sliced_dims(param_rank);

        std::vector<HloInstruction *> start_index(param_rank);
        for (size_t i = 0; i < param_rank; ++i) {
          start_index[i] = (i == most_major_dim) ? reshape_0 : start_from_zero;
          sliced_dims[i] = sliced_shape.dimensions(i);
        }

        HloInstruction *dynamic_shape_2 =
            instr->AddInstruction(HloInstruction::CreateDynamicSlice(
                ShapeUtil::MakeShapeWithDenseLayout(
                    sharded_param->shape().element_type(),
                    sliced_shape.dimensions(),
                    sharded_param->shape().layout().minor_to_major()),
                sharded_param, absl::MakeSpan(start_index),
                absl::MakeSpan(sliced_dims)));
        HloInstruction *new_abs =
            instr->AddInstruction(HloInstruction::CreateUnary(
                dynamic_shape_2->shape(), HloOpcode::kAbs, dynamic_shape_2));
        HloInstruction *new_reduce =
            instr->AddInstruction(HloInstruction::CreateReduce(
                instr->shape(), new_abs, instr->mutable_operand(1),
                instr->dimensions(), instr->to_apply()));

        std::vector<ReplicaGroup> groups(1);
        for (int64_t i = 0; i < num_partitions; ++i) {
          groups[0].add_replica_ids(i);
        }

        std::optional<int64_t> channel_id = next_channel_id;
        auto newallreduce = HloInstruction::CreateAllReduce(
            instr->shape(), {new_reduce}, instr->to_apply(), groups, false,
            channel_id, true);
        TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
            instr, std::move(newallreduce)));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
