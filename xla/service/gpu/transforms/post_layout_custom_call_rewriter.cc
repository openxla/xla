/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/post_layout_custom_call_rewriter.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kAllocatePersistentBuffer =
    "AllocatePersistentBuffer";

absl::StatusOr<bool> RewriteAllocatePersistentBuffer(
    HloCustomCallInstruction* custom_call) {
  custom_call->set_custom_call_target("AllocateBuffer");
  custom_call->mutable_shape()->mutable_layout()->set_memory_space(
      kTempBufferMemorySpaceColor);
  return true;
}
}  // namespace

absl::StatusOr<bool> PostLayoutCustomCallRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomCall(kAllocatePersistentBuffer)) {
        TF_ASSIGN_OR_RETURN(bool rewrited,
                            RewriteAllocatePersistentBuffer(
                                Cast<HloCustomCallInstruction>(instruction)));
        changed |= rewrited;
      }
    }
  }
  return changed;
}
}  // namespace xla::gpu
