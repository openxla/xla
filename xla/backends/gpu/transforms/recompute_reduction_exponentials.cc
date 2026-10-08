/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/recompute_reduction_exponentials.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// Match existing expressions without adding operands or cloning the producer's
// input graph. In particular, equal parameter numbers in different fusions do
// not imply equal values: their outer operands must be the same instruction.
class ExistingExpression {
 public:
  ExistingExpression(HloInstruction* producer, HloInstruction* consumer)
      : producer_(producer), consumer_(consumer) {}

  HloInstruction* Find(const HloInstruction* instruction) {
    auto [it, inserted] = matches_.try_emplace(instruction, nullptr);
    if (!inserted) {
      return it->second;
    }
    if (instruction->HasControlDependencies()) {
      return nullptr;
    }
    if (instruction->opcode() == HloOpcode::kParameter) {
      const HloInstruction* operand =
          producer_->operand(instruction->parameter_number());
      for (int64_t i = 0; i < consumer_->operand_count(); ++i) {
        if (consumer_->operand(i) == operand) {
          return matches_[instruction] = consumer_->fused_parameter(i);
        }
      }
      return nullptr;
    }
    if (!instruction->IsElementwise() &&
        instruction->opcode() != HloOpcode::kBitcast &&
        instruction->opcode() != HloOpcode::kBroadcast &&
        instruction->opcode() != HloOpcode::kConstant) {
      return nullptr;
    }
    for (const HloInstruction* operand : instruction->operands()) {
      if (Find(operand) == nullptr) {
        return nullptr;
      }
    }
    for (HloInstruction* candidate :
         consumer_->fused_instructions_computation()->instructions()) {
      if (!candidate->HasControlDependencies() &&
          instruction->Identical(*candidate, [&](const HloInstruction* lhs,
                                                 const HloInstruction* rhs) {
            return matches_.at(lhs) == rhs;
          })) {
        return matches_[instruction] = candidate;
      }
    }
    return nullptr;
  }

 private:
  HloInstruction* producer_;
  HloInstruction* consumer_;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> matches_;
};

// Only allow a single elementwise path to an output. A gather, broadcast, or
// reduction could change the indexing or multiply the recomputation work.
HloInstruction* ElementwiseOutput(HloInstruction* instruction) {
  HloInstruction* root = instruction->parent()->root_instruction();
  while (instruction != root) {
    if (instruction->user_count() != 1 ||
        instruction->HasControlDependencies()) {
      return nullptr;
    }
    HloInstruction* user = instruction->users().front();
    if (user == root && user->opcode() == HloOpcode::kTuple) {
      return instruction;
    }
    if (!user->IsElementwise()) {
      return nullptr;
    }
    instruction = user;
  }
  return instruction;
}

}  // namespace

absl::StatusOr<bool> RecomputeReductionExponentials::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* producer : computation->MakeInstructionPostOrder()) {
      if (producer->opcode() != HloOpcode::kFusion ||
          producer->fusion_kind() != HloInstruction::FusionKind::kInput ||
          !producer->IsMultiOutputFusion() || producer->IsRoot() ||
          producer->HasControlDependencies() || producer->HasSideEffect() ||
          !producer->output_operand_aliasing().empty()) {
        continue;
      }
      HloInstruction* root = producer->fused_expression_root();
      if (root->has_sharding() ||
          !absl::c_any_of(root->operands(),
                          [](const HloInstruction* output) {
                            return output->opcode() == HloOpcode::kReduce;
                          }) ||
          !absl::c_all_of(producer->users(), [](const HloInstruction* user) {
            return user->opcode() == HloOpcode::kGetTupleElement;
          })) {
        continue;
      }
      for (HloInstruction* gte : producer->users()) {
        HloInstruction* exponential = root->mutable_operand(gte->tuple_index());
        if (exponential->opcode() != HloOpcode::kExp ||
            exponential->HasControlDependencies() ||
            exponential->shape().element_type() != F32 ||
            ShapeUtil::ByteSizeOf(exponential->shape()) < min_bytes_ ||
            gte->IsRoot() || gte->HasControlDependencies() ||
            gte->user_count() != 1 ||
            absl::c_count_if(producer->users(),
                             [&](const HloInstruction* user) {
                               return user->tuple_index() == gte->tuple_index();
                             }) != 1) {
          continue;
        }
        HloInstruction* consumer = gte->users().front();
        if (consumer->opcode() != HloOpcode::kFusion ||
            (consumer->fusion_kind() != HloInstruction::FusionKind::kInput &&
             consumer->fusion_kind() != HloInstruction::FusionKind::kLoop) ||
            consumer->HasControlDependencies() || consumer->HasSideEffect() ||
            !consumer->output_operand_aliasing().empty() ||
            absl::c_count(consumer->operands(), gte) != 1) {
          continue;
        }
        HloInstruction* parameter =
            consumer->fused_parameter(consumer->operand_index(gte));
        if (ElementwiseOutput(parameter) == nullptr) {
          continue;
        }
        HloInstruction* input = ExistingExpression(producer, consumer)
                                    .Find(exponential->operand(0));
        if (input == nullptr) {
          continue;
        }

        HloComputation* body = consumer->fused_instructions_computation();
        HloInstruction* recomputed = body->AddInstruction(
            exponential->CloneWithNewOperands(exponential->shape(), {input}));
        ABSL_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(recomputed));
        ABSL_RETURN_IF_ERROR(
            body->RemoveUnusedParametersFromFusedComputation());
        changed = true;
      }
    }
  }
  if (changed) {
    // Remove dead GTEs and the producer's unused tuple output. The exponential
    // stays inside the reduction; its partial-sum graph is unchanged.
    ABSL_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
  }
  return changed;
}

}  // namespace xla::gpu
