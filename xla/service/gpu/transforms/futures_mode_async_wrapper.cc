/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/futures_mode_async_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

static absl::StatusOr<bool> AsynchronizeInstruction(HloModule* module, HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kCall ||
      !instr->frontend_attributes().map().contains("_async_done")) {
    return false;
  }
  auto start_call_instr = instr->mutable_operand(0);
  TF_RET_CHECK(start_call_instr->frontend_attributes().map().contains("_async_start")) << "No matching start call";

  HloComputation* computation = instr->parent();
  auto original_attributes = instr->frontend_attributes();
  // First, create our async start/done pair and replace the tagged "_async_start" kCall op. 
  TF_ASSIGN_OR_RETURN(
      HloInstruction * done,
      computation->CreateAsyncInstructions(
          start_call_instr, {},
          "futures_mode",
          /*replace=*/true));
  // Next, we fold the async_done instruction into the tagged instruction.
  // This call should always be an identity operator.
  TF_ASSIGN_OR_RETURN(bool check, computation->ReplaceInstruction(instr, done, true, true, true));
  
  // Replace the original attributes after creating the async pair.
  done->set_frontend_attributes(original_attributes);
  auto start = done->mutable_operand(0);
  start->set_frontend_attributes(original_attributes);
  
  // Replace the schedule with 
  if(module->has_schedule()) {
      module->schedule().replace_instruction(computation, start_call_instr, start);
      module->schedule().replace_instruction(computation, instr, done);
  }
  else {
    return absl::InvalidArgumentError("Module must be scheduled");
  }

  return true;
}
}  // namespace

absl::StatusOr<bool> FuturesModeAsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool result, AsynchronizeInstruction(module, instr));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace xla::gpu
