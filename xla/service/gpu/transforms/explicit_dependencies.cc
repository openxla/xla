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

#include "absl/container/flat_hash_map.h"
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

absl::StatusOr<bool> ExplicitControl::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp :
    module->MakeNonfusionComputations(execution_threads)) {
    absl::flat_hash_map<std::string, HloInstruction*> control_map;    
    for (HloInstruction* instr : comp->instructions()) {
      if(instr->frontend_attributes().map().contains("wait_tag")) {
        control_map["wait_tag"] = instr;
      }
      if(instr->frontend_attributes().map().contains("wait_for")) {
	auto key = instr->frontend_attributes().map()["wait_for"];
        instr->AddControlDependencyTo(control_map[key]);
	changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla::gpu
