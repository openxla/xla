/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/profile_viz_utils.h"

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_render_options.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace profile_viz_utils {

void PopulateStats(const HloModule& module, const std::string& node_name,
                   const HloInstruction* instr,
                   const HloComputation* computation,
                   HloRenderOptions* hlo_render_options) {}

StatusOr<std::unique_ptr<HloModule>> ReadModule(const std::string& profiler_session_id) {
  return tsl::errors::Unimplemented("Not yet implemented");
}

std::string ReadComputationStats(const HloComputation& computation,
                                 const HloRenderOptions& hlo_render_options) {
  return "";
}

std::string ReadInstructionStats(const HloInstruction& instr,
                                 const HloRenderOptions& hlo_render_options) {
  return "";
}

}  // namespace profile_viz_utils

}  // namespace xla
