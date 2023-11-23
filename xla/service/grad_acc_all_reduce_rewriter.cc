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

#include "xla/service/grad_acc_all_reduce_rewriter.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/spmd/spmd_partitioner_util.h"

namespace xla {
namespace {

HloInstruction* GetAllReduce(HloInstruction* src) {
  auto opcode = src->opcode();
  if (opcode == HloOpcode::kAllReduce) {
    return src;
  } else if (opcode == HloOpcode::kConvert || opcode == HloOpcode::kReshape ||
             opcode == HloOpcode::kCopy || opcode == HloOpcode::kBitcast ||
             opcode == HloOpcode::kTranspose ||
             opcode == HloOpcode::kDynamicSlice) {
    return GetAllReduce(src->mutable_operand(0));
  } else if (opcode == HloOpcode::kMultiply) {
    HloInstruction* lhs = GetAllReduce(src->mutable_operand(0));
    HloInstruction* rhs = GetAllReduce(src->mutable_operand(1));

    if (lhs != nullptr && rhs == nullptr) {
      return lhs;
    } else if (lhs == nullptr && rhs != nullptr) {
      return rhs;
    }
  }

  return nullptr;
}

HloInstruction* GetParameter(HloInstruction* src) {
  auto opcode = src->opcode();
  if (opcode == HloOpcode::kParameter) {
    return src;
  } else if (opcode == HloOpcode::kConvert || opcode == HloOpcode::kReshape ||
             opcode == HloOpcode::kCopy || opcode == HloOpcode::kBitcast ||
             opcode == HloOpcode::kTranspose ||
             opcode == HloOpcode::kDynamicSlice) {
    return GetParameter(src->mutable_operand(0));
  }
  return nullptr;
}

bool IsBackWard(HloInstruction* src) {
  return absl::StrContains(src->metadata().op_name(), "backward");
}

bool IsInBackWard(HloInstruction* src) {
  // TODO: improve matching accuracy.
  auto opcode = src->opcode();
  if (IsBackWard(src)) {
    return true;
  } else if (opcode == HloOpcode::kDot) {
    auto lhs_is_bw = IsInBackWard(src->mutable_operand(0));
    auto rhs_is_bw = IsInBackWard(src->mutable_operand(1));
    if (lhs_is_bw || rhs_is_bw) return true;

    for (auto user : src->mutable_operand(0)->users()) {
      lhs_is_bw |= IsBackWard(user);
    }
    for (auto user : src->mutable_operand(1)->users()) {
      rhs_is_bw |= IsBackWard(user);
    }
    return lhs_is_bw || rhs_is_bw;
  } else if (opcode == HloOpcode::kConvert || opcode == HloOpcode::kReshape ||
             opcode == HloOpcode::kCopy || opcode == HloOpcode::kBitcast ||
             opcode == HloOpcode::kTranspose ||
             opcode == HloOpcode::kBroadcast ||
             opcode == HloOpcode::kScatter || opcode == HloOpcode::kPad) {
    return IsInBackWard(src->mutable_operand(0));
  } else if (opcode == HloOpcode::kSelect) {
    for (auto operand : src->operands()) {
      if (IsInBackWard(operand)) return true;
    }
  }
  return false;
}

bool IsUsedBy(HloInstruction* src, HloInstruction* dst) {
  bool ret = false;
  if (src->IsDead()) return ret;

  for (auto user: src->users()) {
    auto opcode = user->opcode();
    if (user == dst) {
      return true;
    } else if (opcode == HloOpcode::kConvert || opcode == HloOpcode::kReshape ||
               opcode == HloOpcode::kCopy || opcode == HloOpcode::kBitcast ||
               opcode == HloOpcode::kTranspose) {
      ret = IsUsedBy(user, dst);
    }
  }
  return ret;
}

} // namespace

StatusOr<bool> GradAccAllReduceRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  for (HloInstruction* instr : entry->MakeInstructionPostOrder()) {
    HloInstruction* ar_instr = nullptr;

    if (instr->opcode() == HloOpcode::kAdd) {
      // pattern: allreduce(->dynamicslice)(->transpose)(->convert)->add->...
      if (GetParameter(instr->mutable_operand(0)) == nullptr) {
        continue;
      }
      ar_instr = GetAllReduce(instr->mutable_operand(1));
    } else if (IsUsedBy(instr, entry->root_instruction())) {
      // pattern: allreduce(->dynamicslice)(->transpose)(->convert)->root
      ar_instr = GetAllReduce(instr);
    }

    // Only rewrite allreduce in the backward process.
    // The current implementation requires that the metadata op_name
    // for allreduce upstream instructions must include "backward".
    if (ar_instr == nullptr || !IsInBackWard(ar_instr->mutable_operand(0))) {
      continue;
    }

    if (!IsUsedBy(instr, entry->root_instruction())) {
      // In most cases, after updating the model weights, the gradient will be
      // set to zero/none and therefore will not be used as outputs.
      CHECK(instr->opcode() == HloOpcode::kAdd);
      const Shape& new_shape = instr->shape();
      auto old_ar = Cast<HloAllReduceInstruction>(ar_instr);
      // We create new allreduce other than move old one to avoid compatibility
      // issues caused by type conversions such as bf16->fp32.
      auto new_ar =
          entry->AddInstruction(HloInstruction::CreateAllReduce(
              new_shape,
              {instr},
              spmd::MakeBinaryAdd(new_shape.element_type(), entry->parent()),
              old_ar->replica_groups(),
              old_ar->constrain_layout(), old_ar->channel_id(),
              old_ar->use_global_device_ids()));
      new_ar->set_metadata(old_ar->metadata());

      for (auto instr_user : instr->users()) {
        if (instr_user == new_ar) {
          continue;
        }
        for (size_t i = 0; i < instr_user->operand_count(); ++i) {
          if (instr_user->operand(i) == instr) {
            TF_CHECK_OK(instr_user->ReplaceOperandWith(i, new_ar));
          }
        }
      }
      VLOG(1) << "Added instruction: " << new_ar->ToString();
    }

    // remove allreduce
    for (auto ar_user : ar_instr->users()) {
      TF_CHECK_OK(ar_instr->ReplaceUseWith(ar_user,
          ar_instr->mutable_operand(0)));
    }
    ar_instr->DetachFromOperandsAndUsers();
    TF_CHECK_OK(entry->RemoveInstruction(ar_instr));
    VLOG(1) << "Skipped instruction: " << ar_instr->ToString();
  }

  return true;
}

}  // namespace xla
