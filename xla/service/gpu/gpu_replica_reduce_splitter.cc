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

template <typename Pattern>
auto OptionalCopy(HloInstruction **optional_copy, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Copy(optional_copy, pattern),
                                  std::move(pattern));
}

StatusOr<bool> ReplicaReduceSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  const HloModuleConfig &config = module->config();
  if (!config.use_spmd_partitioning()) {
    VLOG(2) << "Unsupported module";
    return false;
  }
  HloInstruction *pid = nullptr;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction *instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kPartitionId) {
        VLOG(2) << "Found partition-id\n";
        pid = instr;
        VLOG(2) << instr->GetModule()->config().num_partitions() << "\n";
        VLOG(2) << instr->GetModule()->config().replica_count() << "\n";
        break;
      }
    }
  }
  if (!pid) {
    VLOG(2) << "No partition-id found!";
    return false;
  }
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction *instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kReduce) {
        // VLOG(2) << "instr:" << instr->ToString() << "\n";
        HloInstruction *absop = nullptr;
        HloInstruction *shard_operand = nullptr;
        HloInstruction *optcpy = nullptr;
        HloInstruction *operand = instr->mutable_operand(0);
        if (Match(
                operand,
                m::Abs(&absop, OptionalCopy(&optcpy, m::Op(&shard_operand))))) {
          if (optcpy) VLOG(2) << "optcpy " << optcpy->ToString() << "\n";
          VLOG(2) << "instr:" << instr->ToString() << "\n";
          VLOG(2) << "shard_operand:" << shard_operand->ToString() << "\n";
          // return true;
          if (shard_operand->opcode() != HloOpcode::kDynamicSlice &&
              shard_operand->sharding().IsReplicated()) {
            VLOG(2) << "Found replicated operand\n";
            int num_partitions = instr->GetModule()->config().num_partitions();
            // HloInstruction *zero = instr->AddInstruction(
            //     HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
            const Shape target = shard_operand->shape();
            size_t num_col = target.dimensions(1);
            size_t num_row = target.dimensions(0);
            VLOG(2) << "tensor shape: " << num_row << " x " << num_col
                    << std::endl;
            Shape ds_shape =
                ShapeUtil::MakeShape(pid->shape().element_type(), {1});
            std::vector<int> all_dev(num_partitions);
            std::iota(std::begin(all_dev), std::end(all_dev), 0);
            Array<uint32_t> gmap({num_partitions});
            gmap.FillIota(0);

            auto dsop0 =
                computation->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateFromArray(gmap)));

            HloInstruction *DS0 =
                instr->AddInstruction(HloInstruction::CreateDynamicSlice(
                    ds_shape, dsop0, {pid}, {1}));
            HloInstruction *CV0 =
                instr->AddInstruction(HloInstruction::CreateConvert(
                    ShapeUtil::MakeShape(S32, {1}), DS0));

            // dynamic-slice.19
            Array<int32_t> gmap2({num_partitions});
            int32_t start_ind = 0;
            auto loc_row = static_cast<int32_t>(num_row / num_partitions);
            std::generate(gmap2.begin() + 1, gmap2.end(),
                          [&] { return start_ind += loc_row; });
            auto dsop1 = instr->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateFromArray(gmap2)));
            HloInstruction *DS1 =
                instr->AddInstruction(HloInstruction::CreateDynamicSlice(
                    ShapeUtil::MakeShape(dsop1->shape().element_type(), {1}),
                    dsop1, {CV0}, {1}));
            // reshape.147
            HloInstruction *RS1 =
                instr->AddInstruction(HloInstruction::CreateReshape(
                    ShapeUtil::MakeShape(DS1->shape().element_type(), {}),
                    DS1));
            // dynamic-slice.20 new
            auto start_col =
                instr->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<int32_t>(0)));
            HloInstruction *possible = optcpy ? optcpy : shard_operand;
            HloInstruction *DS2 =
                instr->AddInstruction(HloInstruction::CreateDynamicSlice(
                    ShapeUtil::MakeShape(possible->shape().element_type(),
                                         {loc_row, num_col}),
                    possible, {RS1, start_col}, {loc_row, num_col}));
            HloInstruction *newabs =
                instr->AddInstruction(HloInstruction::CreateUnary(
                    DS2->shape(), HloOpcode::kAbs, DS2));
            HloInstruction *newreduce =
                instr->AddInstruction(HloInstruction::CreateReduce(
                    instr->shape(), newabs, instr->mutable_operand(1),
                    instr->dimensions(), instr->to_apply()));

            // AllReduce
            std::vector<ReplicaGroup> groups(1);
            for (int64_t i = 0; i < num_partitions; ++i) {
              groups[0].add_replica_ids(i);
            }

            std::optional<int64_t> channel_id = next_channel_id;
            auto newallreduce = HloInstruction::CreateAllReduce(
                instr->shape(), {newreduce}, instr->to_apply(), groups, false,
                channel_id, true);
            TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
                instr, std::move(newallreduce)));
            // return true;
            changed = true;
          }  // if
        }    // match
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
