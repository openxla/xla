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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GPU_PRIORITY_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GPU_PRIORITY_FUSION_H_

#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/gpu_fusion_cost_model.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

class GpuPriorityFusion : public PriorityFusion {
 public:
  GpuPriorityFusion(tsl::thread::ThreadPool* thread_pool,
                    const se::DeviceDescription& device,
                    const AliasInfo* alias_info,
                    GpuHloCostAnalysis::Options cost_analysis_options,
                    mlir::MLIRContext* mlir_context);

  bool IsFusible(const HloInstruction& instruction) override;

 protected:
  std::vector<HloComputation*> GetFusibleComputations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  FusionDecision BackendCanFuse(HloInstruction* producer,
                                HloInstruction* consumer) override;
  HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                        const HloInstruction* consumer,
                                        bool use_multi_output_fusion) override;
  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       bool use_multi_output_fusion) override;
  void PopulateFusionProcessDump(FusionProcessDumpProto* dump) override;

  std::optional<HloInstruction*> GetPreferredMultiOutputConsumer(
      const HloInstruction* producer,
      HloDfsReachability* reachability) override;

 private:
  FusionDecision IsTritonSupported(const HloInstruction& instr);
  FusionDecision CanFuseTriton(const HloInstruction* producer,
                               const HloInstruction* consumer,
                               bool use_multi_output_fusion = false);

  GpuFusionCostModel* gpu_cost_model_;
  const se::DeviceDescription& device_info_;
  FusionInfoCache fusion_info_cache_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GPU_PRIORITY_FUSION_H_
