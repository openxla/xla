/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/multi_streaming_scheduling.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

// TODO(anlunx): We should use a cost model to determine the instruction to
// reschedule. For now we only reschedule triton gemms.
bool ShouldReschedule(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kFusion) return false;
  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(inst);
  return absl::StrContains(fusion->called_computation()->name(),
                           "triton_gemm_dot");
}

absl::flat_hash_set<HloInstruction*> UsedInstructions(
    const HloInstruction* instr) {
  absl::flat_hash_set<HloInstruction*> used;
  for (HloInstruction* operand : instr->operands()) {
    auto used_by_operand = UsedInstructions(operand);
    used.insert(used_by_operand.begin(), used_by_operand.end());
    used.insert(operand);
  }
  return used;
}

bool Parallelizable(const HloInstruction* first, const HloInstruction* second) {
  return !UsedInstructions(second).contains(first);
}

absl::StatusOr<bool> RescheduleComputation(HloComputation* computation) {
  HloSchedule& schedule = computation->parent()->schedule();
  const std::vector<HloInstruction*> instructions =
      schedule.sequence(computation).instructions();
  const absl::flat_hash_map<HloInstruction*, size_t> instruction_to_index =
      [&]() {
        absl::flat_hash_map<HloInstruction*, size_t> instruction_to_index;
        for (size_t i = 0; i < instructions.size(); i++) {
          instruction_to_index[instructions[i]] = i;
        }
        return instruction_to_index;
      }();

  bool changed = false;
  // The instruction sequence for the updated schedule.
  HloInstructionSequence seq;
  absl::flat_hash_set<HloInstruction*> scheduled;
  ;

  auto add_to_seq = [&](HloInstruction* instr) {
    seq.push_back(instr);
    scheduled.insert(instr);
  };

  // Wrap first inside an async instruction. Reschedule second into the async
  // region. Move instructions used by second before first. Example:
  //
  // first()
  // a()
  // b()
  // second(a, b)
  //
  // will be rescheduled to
  //
  // a()
  // b()
  // first-start()
  // second(a, b)
  // first-done()
  //
  auto reschedule_pair = [&](HloInstruction* first, HloInstruction* second) {
    // Collect instruction used by second and reschedule them before first.
    absl::flat_hash_set<HloInstruction*> used_by_second =
        UsedInstructions(second);
    std::vector<HloInstruction*> used_instructions(used_by_second.begin(),
                                                   used_by_second.end());
    std::sort(used_instructions.begin(), used_instructions.end(),
              [&](HloInstruction* a, HloInstruction* b) {
                return instruction_to_index.at(a) < instruction_to_index.at(b);
              });
    for (HloInstruction* used_instr : used_instructions) {
      if (!scheduled.contains(used_instr)) {
        add_to_seq(used_instr);
      }
    }

    // Set operation queue id of first instruction to 1.
    TF_ASSIGN_OR_RETURN(auto gpu_backend_config,
                        first->backend_config<GpuBackendConfig>());
    gpu_backend_config.set_operation_queue_id(1);
    TF_RETURN_IF_ERROR(first->set_backend_config(gpu_backend_config));

    // Wrap the first instruction into an async instruction.
    TF_ASSIGN_OR_RETURN(HloInstruction * done,
                        computation->CreateAsyncInstructions(first, {}, "",
                                                             /*replace=*/true));
    HloInstruction* start = done->mutable_operand(0);

    // Reschedule first and second.
    seq.push_back(start);
    seq.push_back(second);
    seq.push_back(done);
    scheduled.insert(first);
    scheduled.insert(second);
    changed = true;
    return absl::OkStatus();
  };

  for (auto first_it = instructions.begin(); first_it != instructions.end();
       ++first_it) {
    HloInstruction* first = *first_it;
    if (scheduled.contains(first)) continue;
    if (!ShouldReschedule(first)) {
      add_to_seq(first);
      continue;
    }

    // Find the second instruction to run with the first in parallel.
    HloInstruction* second = nullptr;
    for (auto second_it = first_it + 1; second_it != instructions.end();
         ++second_it) {
      HloInstruction* candidate = *second_it;
      if (scheduled.contains(candidate)) continue;
      if (!ShouldReschedule(candidate)) continue;

      if (Parallelizable(first, candidate)) {
        second = candidate;
        break;
      }
    }

    if (second != nullptr) {
      TF_RETURN_IF_ERROR(reschedule_pair(first, second));
    } else {
      add_to_seq(first);
    }
  }

  if (changed) {
    schedule.set_sequence(computation, seq);
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> MultiStreamingScheduling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RescheduleComputation(comp));
    changed |= result;
  }

  TF_RETURN_IF_ERROR(module->schedule().Update());
  return changed;
}

}  // namespace xla::gpu
