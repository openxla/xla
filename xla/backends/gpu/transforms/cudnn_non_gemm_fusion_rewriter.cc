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

#include "xla/backends/gpu/transforms/cudnn_non_gemm_fusion_rewriter.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {

class CudnnNonGemmFusionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CudnnNonGemmFusionRewriterVisitor(
      se::StreamExecutor* stream_exec,
      const se::DeviceDescription& gpu_device_info)
      : stream_exec_(stream_exec), gpu_device_info_(gpu_device_info) {}

  absl::Status HandleFusion(HloInstruction* fusion_instr) override;

 private:
  se::StreamExecutor* stream_exec_;
  const se::DeviceDescription& gpu_device_info_;
};

absl::Status CudnnNonGemmFusionRewriterVisitor::HandleFusion(
    HloInstruction* fusion_instr) {
  // Skip fusions that are already marked as custom (e.g. already-assigned
  // cuDNN/Triton/etc. fusions) — we shouldn't re-tag them.
  if (fusion_instr->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return absl::OkStatus();
  }
  // Skip any gemm/conv fusions
  const HloComputation* fused = fusion_instr->fused_instructions_computation();
  if (hlo_query::GetFirstInstructionWithOpcode(
          *fused, {HloOpcode::kDot, HloOpcode::kConvolution,
                   HloOpcode::kRaggedDot, HloOpcode::kScaledDot}) != nullptr) {
    return absl::OkStatus();
  }
  // Skip fusions with more than one concatenate. cuDNN still reports >= 1
  // available execution plan for these, but fails to actually lower them.
  int64_t num_concatenates = 0;
  for (const HloInstruction* instr : fused->instructions()) {
    if (instr->opcode() == HloOpcode::kConcatenate) {
      ++num_concatenates;
    }
  }
  if (num_concatenates > 1) {
    VLOG(3) << "cudnn_non_gemm_fusion_rewriter: not rewriting "
            << fusion_instr->name() << " to cuDNN fusion: fusion contains "
            << num_concatenates << " concatenate instructions";
    return absl::OkStatus();
  }
  // If cuDNN cannot produce any execution plans for this fusion, treat it as
  // unsupported and leave the fusion alone.
  absl::StatusOr<int> plan_count_or =
      CuDnnFusionCompiler::GetAvailablePlanCount(
          stream_exec_, gpu_device_info_,
          *Cast<HloFusionInstruction>(fusion_instr));
  if (!plan_count_or.ok() || *plan_count_or == 0) {
    VLOG(3) << "cudnn_non_gemm_fusion_rewriter: not rewriting "
            << fusion_instr->name() << " to cuDNN fusion: "
            << (!plan_count_or.ok() ? plan_count_or.status().message()
                                    : "no available execution plans");
    return absl::OkStatus();
  } else {
    VLOG(3) << "cudnn_non_gemm_fusion_rewriter: rewriting "
            << fusion_instr->name() << " to cuDNN fusion: " << *plan_count_or
            << " execution plans";
  }
  fusion_instr->set_fusion_kind(HloInstruction::FusionKind::kCustom);

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                        fusion_instr->backend_config<GpuBackendConfig>());
  FusionBackendConfig& fusion_backend_config =
      *gpu_backend_config.mutable_fusion_backend_config();
  fusion_backend_config.set_kind(std::string(kCuDnnFusionKind));
  TF_RETURN_IF_ERROR(fusion_instr->set_backend_config(gpu_backend_config));

  MarkAsChanged();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CudnnNonGemmFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return CudnnNonGemmFusionRewriterVisitor(stream_exec_, gpu_device_info_)
      .RunOnModule(module, execution_threads);
}
}  // namespace gpu
}  // namespace xla
