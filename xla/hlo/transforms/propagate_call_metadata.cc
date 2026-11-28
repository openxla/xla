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

#include "xla/hlo/transforms/propagate_call_metadata.h"

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/stack_frames.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool UpdateOpNameWithCleanPrefix(OpMetadata& metadata,
                                 absl::string_view clean_prefix) {
  if (clean_prefix.empty()) {
    return false;
  }
  absl::string_view clean_name = absl::StripPrefix(metadata.op_name(), "/");
  clean_name = absl::StripSuffix(clean_name, "/");
  if (absl::StartsWith(clean_name, clean_prefix) ||
      clean_prefix.size() + clean_name.size() >=
          PropagateCallMetadata::kMaxOpNameSize) {
    return false;
  }
  if (clean_name.empty()) {
    metadata.set_op_name(std::string(clean_prefix));
  } else {
    metadata.set_op_name(absl::StrCat(clean_prefix, "/", clean_name));
  }
  return true;
}

bool UpdateStackFrameInPlace(HloInstruction* hlo, StackFrameId parent_frame_id,
                             StackFrames& stack_frames) {
  if (!parent_frame_id.valid()) {
    return false;
  }
  const StackFrameId cur_frame_id{
      hlo->has_metadata() ? hlo->metadata().stack_frame_id() : 0};
  const StackFrameId new_frame_id =
      stack_frames.PrependIfNotPrefix(parent_frame_id, cur_frame_id);
  if (new_frame_id == cur_frame_id) {
    return false;
  }
  hlo->mutable_metadata().set_stack_frame_id(new_frame_id.value);
  return true;
}

bool UpdateInstructionMetadataInPlace(HloInstruction* hlo,
                                      absl::string_view clean_prefix,
                                      StackFrameId parent_frame_id,
                                      StackFrames* stack_frames) {
  bool changed = false;
  if (!clean_prefix.empty()) {
    if (!hlo->has_metadata()) {
      if (clean_prefix.size() < PropagateCallMetadata::kMaxOpNameSize) {
        hlo->mutable_metadata().set_op_name(std::string(clean_prefix));
        changed = true;
      }
    } else if (UpdateOpNameWithCleanPrefix(hlo->mutable_metadata(),
                                           clean_prefix)) {
      changed = true;
    }
  }
  if (stack_frames != nullptr &&
      UpdateStackFrameInPlace(hlo, parent_frame_id, *stack_frames)) {
    changed = true;
  }
  return changed;
}

void PropagateToInstructionImpl(HloInstruction* hlo,
                                absl::string_view clean_prefix,
                                StackFrameId parent_frame_id,
                                StackFrames* stack_frames) {
  if (GetInstructionCallContext(hlo->opcode()) == CallContext::kControlFlow &&
      hlo->opcode() != HloOpcode::kCall) {
    for (HloComputation* computation : hlo->called_computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        PropagateToInstructionImpl(instruction, clean_prefix, parent_frame_id,
                                   stack_frames);
      }
    }
  }
  UpdateInstructionMetadataInPlace(hlo, clean_prefix, parent_frame_id,
                                   stack_frames);
}

bool PropagateIntoComputation(HloComputation* computation,
                              absl::string_view clean_prefix,
                              StackFrameId parent_frame_id,
                              StackFrames* stack_frames) {
  bool changed = false;
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    changed |= UpdateInstructionMetadataInPlace(instr, clean_prefix,
                                                parent_frame_id, stack_frames);
    if (GetInstructionCallContext(instr->opcode()) ==
            CallContext::kControlFlow &&
        instr->opcode() != HloOpcode::kCall) {
      for (HloComputation* sub : instr->called_computations()) {
        changed |= PropagateIntoComputation(sub, clean_prefix, parent_frame_id,
                                            stack_frames);
      }
    }
  }
  return changed;
}

}  // namespace

bool PropagateCallMetadata::UpdateOpName(OpMetadata& metadata,
                                         absl::string_view prefix) {
  return UpdateOpNameWithCleanPrefix(metadata, absl::StripSuffix(prefix, "/"));
}

bool PropagateCallMetadata::UpdateStackFrame(HloInstruction* hlo,
                                             StackFrameId parent_frame_id) {
  if (!parent_frame_id.valid()) {
    return false;
  }
  HloModule* module = hlo->GetModule();
  if (module == nullptr) {
    return false;
  }
  return UpdateStackFrameInPlace(hlo, parent_frame_id,
                                 module->mutable_stack_frames());
}

void PropagateCallMetadata::PropagateMetadataToInstruction(
    HloInstruction* hlo, absl::string_view prefix,
    StackFrameId parent_frame_id) {
  const absl::string_view clean_prefix = absl::StripSuffix(prefix, "/");
  if (clean_prefix.empty() && !parent_frame_id.valid()) {
    return;
  }
  HloModule* module = parent_frame_id.valid() ? hlo->GetModule() : nullptr;
  StackFrames* stack_frames =
      module != nullptr ? &module->mutable_stack_frames() : nullptr;
  PropagateToInstructionImpl(hlo, clean_prefix, parent_frame_id, stack_frames);
}

absl::StatusOr<bool> PropagateCallMetadata::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  auto computations = module->MakeNonfusionComputations(execution_threads);
  absl::c_reverse(computations);

  StackFrames& stack_frames = module->mutable_stack_frames();
  for (HloComputation* computation : computations) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kCall || !instr->has_metadata()) {
        continue;
      }
      const OpMetadata& call_metadata = instr->metadata();
      const absl::string_view clean_prefix =
          absl::StripSuffix(call_metadata.op_name(), "/");
      const StackFrameId parent_frame_id{call_metadata.stack_frame_id()};
      if (clean_prefix.empty() && !parent_frame_id.valid()) {
        continue;
      }
      StackFrames* target_frames =
          parent_frame_id.valid() ? &stack_frames : nullptr;
      for (HloComputation* callee : instr->called_computations()) {
        changed |= PropagateIntoComputation(callee, clean_prefix,
                                            parent_frame_id, target_frames);
      }
    }
  }

  return changed;
}

}  // namespace xla
