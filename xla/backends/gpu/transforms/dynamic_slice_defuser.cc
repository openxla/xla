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

#include "xla/backends/gpu/transforms/dynamic_slice_defuser.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

namespace {

bool IsNoOp(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBitcast;
}

bool IsTrivialDynamicSliceFusion(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kFusion ||
      instr->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return false;
  }

  const HloInstruction* root = instr->fused_expression_root();
  if (root->opcode() != HloOpcode::kDynamicSlice &&
      root->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return false;
  }

  for (const HloInstruction* fused_instr :
       instr->fused_instructions_computation()->instructions()) {
    if (fused_instr->opcode() == HloOpcode::kParameter || fused_instr == root ||
        IsNoOp(fused_instr) || fused_instr->opcode() == HloOpcode::kConstant) {
      continue;
    }
    return false;
  }

  return true;
}

}  // namespace

absl::StatusOr<bool> DynamicSliceDefuser::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> to_defuse;
    for (HloInstruction* instr : computation->instructions()) {
      if (IsTrivialDynamicSliceFusion(instr)) {
        to_defuse.push_back(instr);
      }
    }

    for (HloInstruction* fusion : to_defuse) {
      RETURN_IF_ERROR(fusion->Defuse());
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla::gpu
