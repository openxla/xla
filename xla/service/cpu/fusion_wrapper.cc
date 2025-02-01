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

#include "xla/service/cpu/fusion_wrapper.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<bool> FusionWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> scatter_instrs;
  VLOG(1) << "FusionWrapper begin";
  XLA_VLOG_LINES(2, module->ToString());
  bool ret = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kScatter) {
        scatter_instrs.push_back(instr);
      }
    }
  }
  for (HloInstruction* scatter : scatter_instrs) {
    HloComputation* computation = scatter->parent();
    HloInstruction* wrapped = computation->CreateFusionInstruction(
        {scatter}, HloInstruction::FusionKind::kLoop);
    VLOG(2) << "Wrapped: " << wrapped->ToString();
    ret = true;
  }
  VLOG(1) << "FusionWrapper end";
  XLA_VLOG_LINES(2, module->ToString());
  return ret;
}

}  // namespace xla
