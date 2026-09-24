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

#include "xla/backends/gpu/transforms/constant_fill_copy_rewriter.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/decision.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

Decision CanReplaceCopyWithFill(const HloInstruction& copy) {
  // Splitting small fills can cost more in launches than it saves in memory.
  constexpr int64_t kMinFillBytes = 1024 * 1024;
  const HloInstruction* fill = copy.operand(0);
  if (!copy.shape().IsArray() || copy.shape() != fill->shape() ||
      copy.HasControlDependencies() || copy.has_sharding()) {
    return Decision::Forbid("Copy changes layout or has ordering or sharding");
  }
  if (fill->opcode() != HloOpcode::kFusion ||
      fill->fusion_kind() != HloInstruction::FusionKind::kLoop ||
      fill->operand_count() != 0 || fill->HasControlDependencies() ||
      fill->has_sharding()) {
    return Decision::Forbid("Not an independent loop fill");
  }
  const HloInstruction* root = fill->fused_expression_root();
  if (fill->fused_instructions_computation()->instruction_count() != 2 ||
      root->opcode() != HloOpcode::kBroadcast ||
      root->operand(0)->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(root->operand(0)->shape()) ||
      root->HasControlDependencies() ||
      root->operand(0)->HasControlDependencies()) {
    return Decision::Forbid("Not a broadcast of a scalar literal");
  }
  if (ShapeUtil::ByteSizeOfElements(copy.shape()) < kMinFillBytes) {
    return Decision::Forbid("Fill is too small to split");
  }
  return Decision::Allow();
}

}  // namespace

absl::StatusOr<bool> ConstantFillCopyRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return false;
  }
  bool changed = false;
  for (HloInstruction* instruction : entry->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kCopy) {
      continue;
    }
    if (Decision decision = CanReplaceCopyWithFill(*instruction);
        decision.IsForbidden()) {
      VLOG(4) << "Not rematerializing " << instruction->name() << ": "
              << decision.Explain();
      continue;
    }
    // A fresh fill preserves the copy's separate writable value without keeping
    // its source alive. Replacement also removes the source if it becomes dead.
    ABSL_RETURN_IF_ERROR(entry->ReplaceWithNewInstruction(
        instruction, instruction->operand(0)->Clone("rematerialized")));
    changed = true;
  }
  if (changed) {
    // Removing the last user of a fill leaves its fused computation orphaned;
    // drop it so dead bodies do not reach dumps, FusionWrapper, or scheduling.
    ABSL_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  }
  return changed;
}

}  // namespace xla::gpu
