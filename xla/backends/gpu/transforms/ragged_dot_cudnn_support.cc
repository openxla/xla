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

#include "xla/backends/gpu/transforms/ragged_dot_cudnn_support.h"

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/backends/gpu/transforms/ragged_dot_fusion_rewriter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

Decision IsRaggedDotSupportedByCuDnn(
    const HloInstruction& ragged_dot,
    const stream_executor::DeviceDescription& device_description,
    stream_executor::StreamExecutor* stream_exec) {
  HloModule probe_module("ragged_dot_cudnn_probe",
                         ragged_dot.GetModule()->config());
  HloComputation::Builder builder("probe_entry");
  std::vector<HloInstruction*> operands;
  for (int64_t i = 0; i < ragged_dot.operand_count(); ++i) {
    operands.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, ragged_dot.operand(i)->shape(), "p")));
  }
  builder.AddInstruction(
      ragged_dot.CloneWithNewOperands(ragged_dot.shape(), operands));
  probe_module.AddEntryComputationWithLayouts(builder.Build());

  // No device description is passed: this pass only builds the fusion.
  absl::StatusOr<bool> changed = RaggedDotFusionRewriter().Run(&probe_module);
  if (!changed.ok()) {
    VLOG(1) << "Ragged dot cuDNN probe: fusion rewrite failed: "
            << changed.status();
    return Decision::Forbid("fusion rewrite failed: ") << changed.status().message();
  }
  for (const HloInstruction* instr :
       probe_module.entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kFusion) {
      continue;
    }
    const auto& fusion = *Cast<HloFusionInstruction>(instr);
    if (stream_exec == nullptr) {
      if (!stream_executor::gpu::SupportsDevicelessCudnnCompilation()) {
        VLOG(1) << "Ragged dot cuDNN probe: deviceless cuDNN compilation is "
                   "unavailable and there is no StreamExecutor.";
        return Decision::Forbid(
            "deviceless cuDNN compilation is unavailable and there is no "
            "StreamExecutor");
      }
      if (CuDnnFusionCompiler::SupportsFusionDeviceless(device_description,
                                                        fusion) !=
          CuDnnFusionCompiler::DevicelessFusionSupport::kSupported) {
        return Decision::Forbid("cuDNN does not report deviceless support");
      }
      return Decision::Allow();
    }
    absl::StatusOr<int> plan_count = CuDnnFusionCompiler::GetAvailablePlanCount(
        stream_exec, device_description, fusion);
    if (!plan_count.ok()) {
      VLOG(1) << "Ragged dot cuDNN probe: plan count failed: "
              << plan_count.status();
      return Decision::Forbid("plan count failed: ")
             << plan_count.status().message();
    }
    if (*plan_count <= 0) {
      return Decision::Forbid("cuDNN reports no available plans");
    }
    return Decision::Allow();
  }
  return Decision::Forbid("ragged dot did not produce a cuDNN fusion");
}

}  // namespace gpu
}  // namespace xla
