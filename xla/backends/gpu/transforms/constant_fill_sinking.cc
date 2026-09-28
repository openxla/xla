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

#include "xla/backends/gpu/transforms/constant_fill_sinking.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

// Match the minimum size used by ConstantFillCopyRewriter. Smaller fills do
// not justify giving up scheduling freedom across the loop.
constexpr int64_t kMinFillBytes = 1024 * 1024;

absl::StatusOr<Decision> CanSinkFill(const HloInstruction& instruction) {
  if (instruction.opcode() != HloOpcode::kFusion ||
      instruction.fusion_kind() != HloInstruction::FusionKind::kLoop ||
      instruction.operand_count() != 0 || !instruction.shape().IsArray() ||
      instruction.user_count() == 0) {
    return Decision::Forbid("Not an independent array fill");
  }
  if (instruction.fused_instructions_computation()->instruction_count() != 2 ||
      !hlo_query::IsBroadcastOfScalarConstant(
          *instruction.fused_expression_root())) {
    return Decision::Forbid("Not a broadcast of a scalar literal");
  }
  if (ShapeUtil::ByteSizeOfElements(instruction.shape()) < kMinFillBytes) {
    return Decision::Forbid("Fill is too small");
  }
  if (instruction.HasControlDependencies() || instruction.has_sharding() ||
      !instruction.frontend_attributes().map().empty()) {
    return Decision::Forbid("Fill has explicit constraints");
  }
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                        instruction.backend_config<GpuBackendConfig>());
  if (config.force_earliest_schedule() || config.operation_queue_id() != 0 ||
      config.device_type() == DEVICE_TYPE_HOST) {
    return Decision::Forbid("Fill has an explicit schedule or stream");
  }
  return Decision::Allow();
}

}  // namespace

absl::StatusOr<bool> ConstantFillSinking::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    return absl::FailedPreconditionError(
        "ConstantFillSinking requires a scheduled module.");
  }
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads) ||
      !module->schedule().is_computation_scheduled(entry)) {
    return false;
  }
  const std::vector<HloInstruction*>& sequence =
      module->schedule().sequence(entry).instructions();
  absl::flat_hash_map<const HloInstruction*, int64_t> positions;
  positions.reserve(sequence.size());
  // Number of while instructions strictly before each schedule position.
  std::vector<int64_t> whiles_before(sequence.size());
  int64_t while_count = 0;
  for (int64_t i = 0; i < sequence.size(); ++i) {
    positions[sequence[i]] = i;
    whiles_before[i] = while_count;
    while_count += sequence[i]->opcode() == HloOpcode::kWhile;
  }
  if (while_count == 0) {
    return false;
  }

  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      fills_before;
  absl::flat_hash_set<HloInstruction*> moved_fills;
  for (int64_t i = 0; i < sequence.size(); ++i) {
    HloInstruction* fill = sequence[i];
    ABSL_ASSIGN_OR_RETURN(Decision decision, CanSinkFill(*fill));
    if (decision.IsForbidden()) {
      continue;
    }
    int64_t first_user = sequence.size();
    for (HloInstruction* user : fill->users()) {
      first_user = std::min(first_user, positions.at(user));
    }
    if (whiles_before[first_user] == whiles_before[i]) {
      continue;
    }
    // A direct async-start user must see the initialized buffer. Do not look
    // through it to its async-done, or through aliases to a later consumer.
    fills_before[sequence[first_user]].push_back(fill);
    moved_fills.insert(fill);
  }
  if (moved_fills.empty()) {
    return false;
  }

  std::vector<HloInstruction*> result;
  result.reserve(sequence.size());
  for (HloInstruction* instruction : sequence) {
    auto it = fills_before.find(instruction);
    if (it != fills_before.end()) {
      result.insert(result.end(), it->second.begin(), it->second.end());
    }
    if (!moved_fills.contains(instruction)) {
      result.push_back(instruction);
    }
  }
  module->schedule().set_sequence(entry, result);
  ABSL_RETURN_IF_ERROR(module->schedule().Verify());
  return true;
}

}  // namespace xla::gpu
