/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/all_reduce_contiguous.h"

#include <vector>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

void MaybeUpdateSchedule(HloComputation* computation,
                         HloAllReduceInstruction* all_reduce,
                         const std::vector<HloInstruction*>& new_ops) {
  HloModule* module = computation->parent();
  if (!module->has_schedule()) return;

  const auto& sequence = module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size() + new_ops.size() - 1);

  for (HloInstruction* instr : sequence.instructions()) {
    if (instr != all_reduce) {
      new_sequence.push_back(instr);
      continue;
    }
    new_sequence.insert(new_sequence.end(), new_ops.begin(), new_ops.end());
  }
  module->schedule().set_sequence(computation, new_sequence);
}

Status ReplaceWithContiguousAllReduce(HloAllReduceInstruction* all_reduce) {
  TF_RET_CHECK(all_reduce);
  TF_RET_CHECK(!all_reduce->has_sharding());

  HloComputation& computation = *all_reduce->parent();  // never null
  PrimitiveType element_type = all_reduce->operand(0)->shape().element_type();

  std::vector<HloInstruction*> new_ops;
  // For each all_reduce arg, 3 additional ops: bitcast (before all-reduce),
  // slice + bitcast (after).
  // Plus 3 ops: concatenate + the new all-reduce op + tuple.
  new_ops.reserve(all_reduce->operand_count() * 3 + 3);

  // Bitcast operands to 1D so that they may be concatenated together.
  std::vector<HloInstruction*> flat_operands;
  flat_operands.reserve(all_reduce->operand_count());
  int64_t total_size = 0;
  for (HloInstruction* operand : all_reduce->operands()) {
    TF_RET_CHECK(operand->shape().IsArray());
    int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
    Shape flat_shape = ShapeUtil::MakeShape(element_type, {num_elements});
    HloInstruction* bitcast = computation.AddInstruction(
        HloInstruction::CreateBitcast(flat_shape, operand));
    flat_operands.push_back(bitcast);
    new_ops.push_back(bitcast);
    total_size += num_elements;
  }

  Shape concat_shape = ShapeUtil::MakeShape(element_type, {total_size});
  HloInstruction* concatenated =
      computation.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, flat_operands, /*dimension=*/0));
  new_ops.push_back(concatenated);

  HloInstruction* new_all_reduce =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          concat_shape, {concatenated}, all_reduce->to_apply(),
          all_reduce->replica_groups(),
          /*constrain_layout=*/false, all_reduce->channel_id(),
          all_reduce->use_global_device_ids()));
  new_ops.push_back(new_all_reduce);

  // Slice from all-reduce result and bitcast back to the original shapes.
  std::vector<HloInstruction*> outputs;
  outputs.reserve(all_reduce->operand_count());
  int64_t offset = 0;
  for (int64_t i = 0; i < all_reduce->operand_count(); ++i) {
    const Shape& flat_shape = flat_operands[i]->shape();
    int64_t end = offset + flat_shape.dimensions(0);
    HloInstruction* sliced = computation.AddInstruction(
        HloInstruction::CreateSlice(flat_shape, new_all_reduce,
                                    /*start_indices=*/{offset},
                                    /*limit_indices=*/{end},
                                    /*strides=*/{1}));
    new_ops.push_back(sliced);
    HloInstruction* bitcast = computation.AddInstruction(
        HloInstruction::CreateBitcast(all_reduce->operand(i)->shape(), sliced));
    outputs.push_back(bitcast);
    new_ops.push_back(bitcast);
    offset = end;
  }
  // Replace original all-reduce with tuple of slices from new all-reduce.
  std::unique_ptr<HloInstruction> tuple = HloInstruction::CreateTuple(outputs);
  new_ops.push_back(tuple.get());
  TF_RETURN_IF_ERROR(
      computation.ReplaceWithNewInstruction(all_reduce, std::move(tuple)));

  MaybeUpdateSchedule(&computation, all_reduce, new_ops);
  return OkStatus();
}
}  // namespace

StatusOr<bool> AllReduceContiguous::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running AllReduceContiguous";

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceContiguous because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }

  std::vector<HloAllReduceInstruction*> all_reduces;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllReduce &&
          instruction->operand_count() > 1) {
        all_reduces.push_back(Cast<HloAllReduceInstruction>(instruction));
      }
    }
  }

  for (HloAllReduceInstruction* all_reduce : all_reduces) {
    TF_RETURN_IF_ERROR(ReplaceWithContiguousAllReduce(all_reduce));
  }

  return !all_reduces.empty();
}

}  // namespace xla
