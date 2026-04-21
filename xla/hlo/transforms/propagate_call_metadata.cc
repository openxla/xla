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

#include <algorithm>
#include <string>
#include <utility>

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

// Limit on op_name length to prevent unbounded growth from deeply nested calls.
constexpr int kMaxOpNameSize = 1024;

// Sanitize and prepend the prefix to the instruction's op_name.
bool UpdateOpName(OpMetadata& metadata, absl::string_view prefix) {
  if (prefix.empty()) {
    return false;
  }
  // Strip trailing '/' from prefix.
  absl::string_view clean_prefix = absl::StripSuffix(prefix, "/");
  if (clean_prefix.empty()) {
    return false;
  }

  std::string op_name = metadata.op_name();
  // Strip leading/trailing '/' from existing op_name.
  absl::string_view clean_name = absl::StripPrefix(op_name, "/");
  clean_name = absl::StripSuffix(clean_name, "/");

  // Already has the prefix.
  if (absl::StartsWith(clean_name, clean_prefix)) {
    return false;
  }
  // op_name is a substring of prefix — already captured.
  if (!clean_name.empty() && absl::StrContains(clean_prefix, clean_name)) {
    return false;
  }
  std::string result;
  if (clean_name.empty()) {
    result = std::string(clean_prefix);
  } else {
    result = absl::StrCat(clean_prefix, "/", clean_name);
  }
  // Cap at kMaxOpNameSize to avoid unbounded growth from deeply nested calls.
  if (result.size() > kMaxOpNameSize) {
    result.resize(kMaxOpNameSize);
  }
  metadata.set_op_name(std::move(result));
  return true;
}

// Update stack frame: concatenate parent_frame_id as ancestor.
bool UpdateStackFrame(HloInstruction* hlo, StackFrameId parent_frame_id) {
  if (!parent_frame_id.valid()) {
    return false;
  }
  HloModule* module = hlo->GetModule();
  OpMetadata metadata = hlo->metadata();
  if (module->stack_frames().IsPrefix(
          parent_frame_id, StackFrameId{metadata.stack_frame_id()})) {
    return false;
  }
  metadata.set_stack_frame_id(
      module->mutable_stack_frames()
          .Concatenate(parent_frame_id, StackFrameId{metadata.stack_frame_id()})
          .value);
  hlo->set_metadata(metadata);
  return true;
}

// Propagate metadata into all instructions in a computation.
// Recurses into control-flow sub-computations (while, conditional) with the
// same prefix. Does NOT recurse into kCall — nested calls are handled by
// the top-level loop which processes computations in reverse post-order.
bool PropagateIntoComputation(HloComputation* computation,
                              absl::string_view prefix,
                              StackFrameId parent_frame_id) {
  bool changed = false;
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    OpMetadata metadata = instr->metadata();
    if (UpdateOpName(metadata, prefix)) {
      instr->set_metadata(metadata);
      changed = true;
    }
    if (UpdateStackFrame(instr, parent_frame_id)) {
      changed = true;
    }

    // Recurse into while/conditional sub-computations with same prefix.
    if (GetInstructionCallContext(instr->opcode()) ==
            CallContext::kControlFlow &&
        instr->opcode() != HloOpcode::kCall) {
      for (HloComputation* sub : instr->called_computations()) {
        changed |= PropagateIntoComputation(sub, prefix, parent_frame_id);
      }
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> PropagateCallMetadata::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Process in reverse post-order (callers before callees) so that nested
  // call instructions have their metadata updated before we propagate into
  // their callees.
  auto computations = module->MakeNonfusionComputations(execution_threads);
  std::reverse(computations.begin(), computations.end());

  for (HloComputation* computation : computations) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kCall) {
        continue;
      }
      const OpMetadata& call_metadata = instr->metadata();
      absl::string_view prefix = call_metadata.op_name();
      StackFrameId parent_frame_id{call_metadata.stack_frame_id()};
      if (prefix.empty() && !parent_frame_id.valid()) {
        continue;
      }
      for (HloComputation* callee : instr->called_computations()) {
        changed |= PropagateIntoComputation(callee, prefix, parent_frame_id);
      }
    }
  }

  return changed;
}

}  // namespace xla
