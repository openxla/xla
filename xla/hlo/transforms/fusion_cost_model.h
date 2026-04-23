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

#ifndef XLA_HLO_TRANSFORMS_FUSION_COST_MODEL_H_
#define XLA_HLO_TRANSFORMS_FUSION_COST_MODEL_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

class HloComputation;

// An interface for cost models used by priority-based fusion passes.
// A fusion cost model is consulted by the fusion pass to determine if
// clustering a producer instruction with its consumers is profitable, and
// what the estimated runtimes would be.
class FusionCostModel {
 public:
  struct RunTimes {
    absl::Duration unfused;
    absl::Duration fused;
  };

  virtual ~FusionCostModel() = default;

  // Called before any estimation begins, allowing the model to collect
  // computation-level information.
  virtual absl::Status Prepare(const HloComputation* computation) {
    return absl::OkStatus();
  }

  virtual absl::StatusOr<RunTimes> EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<const HloInstruction* const> consumers) = 0;

  virtual bool WouldExplodeIrSize(const HloInstruction* producer,
                                  const HloInstruction* consumer) = 0;
  virtual void PreInstructionFused(HloInstruction* producer,
                                   HloInstruction* consumer) {}

  virtual void OnInstructionFused(HloInstruction* producer,
                                  HloInstruction* consumer,
                                  HloInstruction* fusion) {}

  // Drops any cached cost information about the given instruction. This is
  // typically called when the instruction is removed or materially changed.
  virtual void Invalidate(const HloInstruction* instruction) {}

  // Recomputes cached cost information about the given instruction. This is
  // typically called when the inputs to the instruction change or are fused.
  virtual absl::Status Revisit(const HloInstruction* instruction) = 0;
  virtual void ClearCaches() {}
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_FUSION_COST_MODEL_H_
