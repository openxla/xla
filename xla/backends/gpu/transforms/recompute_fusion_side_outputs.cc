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

#include "xla/backends/gpu/transforms/recompute_fusion_side_outputs.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

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

bool IsNativeFusion(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kFusion &&
         (instruction->fusion_kind() == HloInstruction::FusionKind::kInput ||
          instruction->fusion_kind() == HloInstruction::FusionKind::kLoop) &&
         !instruction->HasSideEffect() &&
         !instruction->HasControlDependencies() &&
         instruction->output_operand_aliasing().empty();
}

std::optional<FusionRecomputeCandidate> MatchCandidate(
    HloInstruction* producer, HloInstruction* extraction, int64_t min_bytes) {
  if (!IsNativeFusion(producer) || !producer->IsMultiOutputFusion() ||
      producer->IsRoot() || producer->fused_expression_root()->has_sharding() ||
      !absl::c_all_of(producer->users(), [](const HloInstruction* user) {
        return user->opcode() == HloOpcode::kGetTupleElement;
      })) {
    return std::nullopt;
  }
  HloInstruction* expression =
      producer->fused_expression_root()->mutable_operand(
          extraction->tuple_index());
  if (!expression->IsElementwise() || expression->operand_count() == 0 ||
      expression->HasSideEffect() || expression->HasControlDependencies() ||
      !expression->shape().IsArray() || expression->shape().is_dynamic() ||
      ShapeUtil::ByteSizeOf(expression->shape()) < min_bytes ||
      extraction->IsRoot() || extraction->HasControlDependencies() ||
      extraction->user_count() != 1 ||
      absl::c_count(producer->fused_expression_root()->operands(),
                    expression) != 1 ||
      absl::c_count_if(producer->users(), [&](const HloInstruction* user) {
        return user->tuple_index() == extraction->tuple_index();
      }) != 1) {
    return std::nullopt;
  }
  HloInstruction* consumer = extraction->users().front();
  if (!IsNativeFusion(consumer) ||
      absl::c_count(consumer->operands(), extraction) != 1 ||
      ElementwiseOutput(consumer->fused_parameter(
          consumer->operand_index(extraction))) == nullptr) {
    return std::nullopt;
  }
  ExistingExpression matcher(producer, consumer);
  FusionRecomputeCandidate candidate{
      producer, consumer, extraction, expression, {}};
  for (const HloInstruction* operand : expression->operands()) {
    HloInstruction* existing = matcher.Find(operand);
    if (existing == nullptr) return std::nullopt;
    candidate.operands.push_back(existing);
  }
  return candidate;
}

absl::Status ApplyCandidate(const FusionRecomputeCandidate& candidate) {
  HloComputation* body = candidate.consumer->fused_instructions_computation();
  HloInstruction* parameter = candidate.consumer->fused_parameter(
      candidate.consumer->operand_index(candidate.extraction));
  HloInstruction* recomputed =
      body->AddInstruction(candidate.expression->CloneWithNewOperands(
          candidate.expression->shape(), candidate.operands));
  ABSL_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(recomputed));
  ABSL_RETURN_IF_ERROR(body->RemoveInstructionAndUnusedOperands(parameter));
  // Preserve emitter/launch choices but invalidate costs for the old bodies.
  for (HloInstruction* fusion : {candidate.producer, candidate.consumer}) {
    ABSL_ASSIGN_OR_RETURN(auto config,
                          fusion->backend_config<GpuBackendConfig>());
    config.clear_reification_cost();
    ABSL_RETURN_IF_ERROR(fusion->set_backend_config(config));
  }
  return HloDCE().Run(candidate.consumer->GetModule()).status();
}

}  // namespace

absl::StatusOr<FusionRecomputeVariants> ExtractFusionRecomputeVariants(
    const FusionRecomputeCandidate& candidate) {
  const std::vector<HloInstruction*> order =
      candidate.producer->parent()->MakeInstructionPostOrder();
  absl::flat_hash_set<HloInstruction*> descendants;
  for (HloInstruction* instruction : order) {
    if (instruction == candidate.producer ||
        absl::c_any_of(instruction->operands(), [&](HloInstruction* operand) {
          return descendants.contains(operand);
        })) {
      descendants.insert(instruction);
    }
  }
  // Include every path from producer to consumer, not just the saved tensor.
  // In softmax this includes the second reduction that produces the row sum.
  absl::flat_hash_set<HloInstruction*> region;
  std::function<void(HloInstruction*)> include =
      [&](HloInstruction* instruction) {
        if (!descendants.contains(instruction) ||
            !region.insert(instruction).second)
          return;
        for (HloInstruction* operand : instruction->operands())
          include(operand);
      };
  include(candidate.consumer);
  // Preserve producer outputs observable by users outside the region.
  for (HloInstruction* extraction : candidate.producer->users()) {
    region.insert(extraction);
  }
  if (region.size() > 32) {
    return absl::UnimplementedError(
        "recomputation region exceeds tuning budget");
  }
  for (HloInstruction* instruction : region) {
    if (instruction->HasControlDependencies() ||
        (!IsNativeFusion(instruction) &&
         instruction->opcode() != HloOpcode::kGetTupleElement &&
         instruction->opcode() != HloOpcode::kBitcast)) {
      return absl::UnimplementedError(
          "unsupported instruction in tuning region");
    }
  }
  HloModuleConfig config;
  config.set_debug_options(
      candidate.producer->GetModule()->config().debug_options());
  auto before = std::make_unique<HloModule>(
      "fusion_recomputation", config,
      std::make_unique<CompilationEnvironments>(
          candidate.producer->GetModule()->comp_envs()));
  HloCloneContext context(before.get());
  HloComputation::Builder builder("region");
  absl::flat_hash_map<HloInstruction*, HloInstruction*> clones;
  int64_t parameter_number = 0;
  std::vector<HloInstruction*> outputs;
  for (HloInstruction* instruction : order) {
    if (!region.contains(instruction)) continue;
    std::vector<HloInstruction*> operands;
    for (HloInstruction* operand : instruction->operands()) {
      if (!clones.contains(operand)) {
        if (!operand->shape().IsArray() || operand->shape().is_dynamic()) {
          return absl::UnimplementedError("non-static-array tuning input");
        }
        if (operand->opcode() == HloOpcode::kConstant) {
          clones[operand] = builder.AddInstruction(operand->Clone());
        } else {
          clones[operand] =
              builder.AddInstruction(HloInstruction::CreateParameter(
                  parameter_number++, operand->shape(), operand->name()));
        }
      }
      operands.push_back(clones.at(operand));
    }
    HloInstruction* clone =
        builder.AddInstruction(instruction->CloneWithNewOperands(
            instruction->shape(), operands, &context));
    clones[instruction] = clone;
    if (instruction == candidate.consumer || instruction->IsRoot() ||
        absl::c_any_of(instruction->users(), [&](HloInstruction* user) {
          return !region.contains(user);
        })) {
      outputs.push_back(clone);
    }
  }
  TF_RET_CHECK(!outputs.empty());
  builder.AddInstruction(HloInstruction::CreateTuple(outputs));
  before->AddEntryComputationWithLayouts(builder.Build())
      ->SetExecutionThread(candidate.producer->parent()->execution_thread());
  auto [after, after_context] = before->CloneWithContext("recomputed");
  auto* producer =
      after_context->FindInstruction(clones.at(candidate.producer));
  auto* extraction =
      after_context->FindInstruction(clones.at(candidate.extraction));
  TF_RET_CHECK(producer != nullptr && extraction != nullptr);
  std::optional<FusionRecomputeCandidate> cloned =
      MatchCandidate(producer, extraction, 0);
  TF_RET_CHECK(cloned.has_value());
  const int64_t bytes = ShapeUtil::ByteSizeOf(candidate.expression->shape());
  ABSL_RETURN_IF_ERROR(ApplyCandidate(*cloned));
  TF_RET_CHECK(ShapeUtil::Equal(
      before->entry_computation()->root_instruction()->shape(),
      after->entry_computation()->root_instruction()->shape()));
  TF_RET_CHECK(before->entry_computation()->num_parameters() ==
               after->entry_computation()->num_parameters());
  for (int64_t i = 0; i < before->entry_computation()->num_parameters(); ++i) {
    TF_RET_CHECK(ShapeUtil::Equal(
        before->entry_computation()->parameter_instruction(i)->shape(),
        after->entry_computation()->parameter_instruction(i)->shape()));
  }
  return FusionRecomputeVariants{std::move(before), std::move(after), bytes};
}

absl::StatusOr<bool> RecomputeFusionSideOutputs::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!evaluator_) return false;
  bool changed = false;
  absl::flat_hash_set<std::string> attempted;
  for (int64_t attempt = 0; attempt < max_candidates_; ++attempt) {
    std::optional<FusionRecomputeCandidate> candidate;
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      for (HloInstruction* instruction :
           computation->MakeInstructionPostOrder()) {
        if (instruction->opcode() != HloOpcode::kGetTupleElement) continue;
        const std::string key =
            absl::StrCat(computation->name(), ":", instruction->name());
        if (attempted.contains(key)) continue;
        candidate = MatchCandidate(instruction->mutable_operand(0), instruction,
                                   min_bytes_);
        if (candidate) {
          attempted.insert(key);
          break;
        }
      }
      if (candidate) break;
    }
    if (!candidate) break;
    const std::string name(candidate->extraction->name());
    absl::StatusOr<FusionRecomputeVariants> variants =
        ExtractFusionRecomputeVariants(*candidate);
    if (!variants.ok()) {
      VLOG(2) << "Recomputation skipped for " << name << ": "
              << variants.status();
      continue;
    }
    absl::StatusOr<Decision> decision = evaluator_(*variants);
    // Compilation/profiling failures must not make a valid program fail.
    if (!decision.ok()) {
      VLOG(1) << "Recomputation evaluation failed for " << name << ": "
              << decision.status();
      continue;
    }
    VLOG(1) << "Recomputation " << name << ": " << decision->Explain();
    if (decision->IsAllowed() && !analyze_only_) {
      ABSL_RETURN_IF_ERROR(ApplyCandidate(*candidate));
      changed = true;
    }
    // Restart discovery after each rewrite. Tuple indices, shared inputs and
    // cost estimates from the previous graph must not survive an accepted edit.
  }
  return changed;
}

}  // namespace xla::gpu
