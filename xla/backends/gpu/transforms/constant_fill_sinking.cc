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
#include <set>
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

// Async windows are the schedule ranges between an async start and its done.
// Work the scheduler placed inside a window runs while the async operation is
// in flight, so it must not be pulled out of the window.
bool IsWindowStart(const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    case HloOpcode::kCopyStart:
    case HloOpcode::kSend:
    case HloOpcode::kRecv: {
      return true;
    }
    default: {
      return instruction.IsAsyncStart();
    }
  }
}

bool IsWindowDone(const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    case HloOpcode::kCopyDone:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone: {
      return true;
    }
    default: {
      return instruction.IsAsyncDone();
    }
  }
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
  const int64_t size = sequence.size();
  absl::flat_hash_map<const HloInstruction*, int64_t> positions;
  positions.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    positions[sequence[i]] = i;
  }

  // window_done[s] is the position of the done that closes the async window
  // opened at position s, or -1 when no window opens there.
  std::vector<int64_t> window_done(size, -1);
  for (int64_t i = 0; i < size; ++i) {
    if (!IsWindowDone(*sequence[i]) ||
        !IsWindowStart(*sequence[i]->operand(0))) {
      continue;
    }
    auto start = positions.find(sequence[i]->operand(0));
    if (start != positions.end()) {
      window_done[start->second] = i;
    }
  }
  // first_close[i] is the position of the earliest done that closes a window
  // open at position i (start < i < done), or `size` when no window is open.
  std::vector<int64_t> first_close(size, size);
  std::multiset<int64_t> open_dones;
  for (int64_t i = 0; i < size; ++i) {
    while (!open_dones.empty() && *open_dones.begin() <= i) {
      open_dones.erase(open_dones.begin());
    }
    if (!open_dones.empty()) {
      first_close[i] = *open_dones.begin();
    }
    if (window_done[i] >= 0) {
      open_dones.insert(window_done[i]);
    }
  }

  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      fills_before;
  absl::flat_hash_set<HloInstruction*> moved_fills;
  for (int64_t i = 0; i < size; ++i) {
    HloInstruction* fill = sequence[i];
    ABSL_ASSIGN_OR_RETURN(Decision decision, CanSinkFill(*fill));
    if (decision.IsForbidden()) {
      continue;
    }
    int64_t first_user = size;
    for (HloInstruction* user : fill->users()) {
      first_user = std::min(first_user, positions.at(user));
    }
    // Every window open at the fill must still be open at its first user.
    // Otherwise the fill hides latency that its first user does not, and
    // moving it would put the fill's kernel back on the critical path.
    if (first_close[i] <= first_user) {
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

  // Fills that already sit directly before their first user are re-inserted
  // in place, so the rebuilt sequence only differs when a fill actually moves.
  std::vector<HloInstruction*> result;
  result.reserve(size);
  for (HloInstruction* instruction : sequence) {
    auto it = fills_before.find(instruction);
    if (it != fills_before.end()) {
      result.insert(result.end(), it->second.begin(), it->second.end());
    }
    if (!moved_fills.contains(instruction)) {
      result.push_back(instruction);
    }
  }
  if (result == sequence) {
    return false;
  }
  module->schedule().set_sequence(entry, result);
  ABSL_RETURN_IF_ERROR(module->schedule().Verify());
  return true;
}

}  // namespace xla::gpu
