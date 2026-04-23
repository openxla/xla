/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/fusion_cost_model.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
class HloDfsReachability;

namespace gpu {

class PriorityFusionQueue;

class PriorityFusion : public HloModulePass {
 public:
  PriorityFusion(tsl::thread::ThreadPool* thread_pool,
                 const AliasInfo* alias_info,
                 std::unique_ptr<FusionCostModel> cost_model);

  absl::string_view name() const override { return "priority-fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  virtual std::optional<HloInstruction*> GetPreferredMultiOutputConsumer(
      const HloInstruction* producer, HloDfsReachability* reachability) {
    return std::nullopt;
  }

 protected:
  friend class PriorityFusionQueue;

  // Hooks for backend subclasses.
  virtual std::vector<HloComputation*> GetFusibleComputations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    return module->MakeNonfusionComputations(execution_threads);
  }
  virtual FusionDecision BackendCanFuse(HloInstruction* producer,
                                        HloInstruction* consumer) {
    return FusionDecision::Allow();
  }

  // Default: the GPU allowlist (elementwise, broadcast, reduce, gather,
  // scatter, reduce-window, ...). CPU overrides to drop scatter,
  // reduce-window, gather, and non-scalar constant producers.
  virtual bool IsFusible(const HloInstruction& instruction);

 protected:
  virtual HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                                const HloInstruction* consumer,
                                                bool use_multi_output_fusion) {
    return HloInstruction::FusionKind::kLoop;
  }

  FusionCostModel* cost_model() { return cost_model_.get(); }

  virtual HloInstruction* Fuse(HloInstruction* producer,
                               HloInstruction* consumer,
                               bool use_multi_output_fusion);

  virtual void PopulateFusionProcessDump(FusionProcessDumpProto* dump) {}

  const AliasInfo* alias_info_;

 private:
  bool ConsumeFuel(HloInstruction* producer, HloInstruction* consumer);
  FusionDecision CanFuseConstant(HloInstruction* constant,
                                 HloInstruction* user);

  tsl::thread::ThreadPool* thread_pool_;
  std::unique_ptr<FusionCostModel> cost_model_;
  std::unique_ptr<FusionProcessDumpProto> fusion_process_dump_;
};

// Free helper used by backend `BackendCanFuse` implementations. Today
// this lives as a file-local static in priority_fusion.cc; it needs
// a header declaration now because both GPU subclass (GpuPriorityFusion)
// and later CPU subclass call it from different TUs.
bool IsFusibleBitcast(const HloInstruction& instr);

bool OperandReachableFromProducer(const HloInstruction* producer,
                                  const HloInstruction* consumer,
                                  HloDfsReachability* reachability);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_
