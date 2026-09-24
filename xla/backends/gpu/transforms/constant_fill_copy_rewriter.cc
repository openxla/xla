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
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

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
  if (fill->fused_instructions_computation()->instruction_count() != 2 ||
      !hlo_query::IsBroadcastOfScalarConstant(*fill->fused_expression_root())) {
    return Decision::Forbid("Not a broadcast of a scalar literal");
  }
  if (ShapeUtil::ByteSizeOfElements(copy.shape()) < kMinFillBytes) {
    return Decision::Forbid("Fill is too small to split");
  }
  return Decision::Allow();
}

// The rematerialized fill stands in for the copy, so it must carry the copy's
// annotations: stream assignment, scheduling group, provenance. The clone
// starts out with the fill's own annotations; the copy's populated fields win
// on conflict. ReplaceInstruction alone only copies them when the clone has
// none.
absl::Status TransferCopyAnnotations(const HloInstruction& copy,
                                     HloInstruction* fill) {
  FrontendAttributes frontend_attributes = fill->frontend_attributes();
  frontend_attributes.MergeFrom(copy.frontend_attributes());
  fill->set_frontend_attributes(std::move(frontend_attributes));

  OpMetadata metadata = fill->metadata();
  metadata.MergeFrom(copy.metadata());
  fill->set_metadata(metadata);

  if (copy.has_backend_config()) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig fill_config,
                          fill->backend_config<GpuBackendConfig>());
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig copy_config,
                          copy.backend_config<GpuBackendConfig>());
    // A copy never carries a fusion config, so merging only layers the copy's
    // queue assignment and scheduling hints over the fill's config.
    fill_config.MergeFrom(copy_config);
    ABSL_RETURN_IF_ERROR(fill->set_backend_config(fill_config));
  }
  return absl::OkStatus();
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
    std::unique_ptr<HloInstruction> fill =
        instruction->operand(0)->Clone("rematerialized");
    ABSL_RETURN_IF_ERROR(TransferCopyAnnotations(*instruction, fill.get()));
    ABSL_RETURN_IF_ERROR(
        entry->ReplaceWithNewInstruction(instruction, std::move(fill)));
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
