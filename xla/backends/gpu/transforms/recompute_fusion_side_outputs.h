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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_RECOMPUTE_FUSION_SIDE_OUTPUTS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_RECOMPUTE_FUSION_SIDE_OUTPUTS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/decision.h"

namespace xla::gpu {

// A single pure elementwise side output. All operands of `expression` have
// identical expressions in `consumer`; no new fusion inputs are introduced.
struct FusionRecomputeCandidate {
  HloInstruction* producer;
  HloInstruction* consumer;
  HloInstruction* extraction;
  HloInstruction* expression;
  std::vector<HloInstruction*> operands;
};

// Matched, dependency-preserving regions with identical input/output shapes.
// The saved intermediate is internal, not an artificial benchmark output.
struct FusionRecomputeVariants {
  std::unique_ptr<HloModule> before;
  std::unique_ptr<HloModule> after;
  int64_t eliminated_bytes;
};

absl::StatusOr<FusionRecomputeVariants> ExtractFusionRecomputeVariants(
    const FusionRecomputeCandidate& candidate);

// Recomputes side outputs inside existing consumer fusions. Run after
// multi-output fusion and backend configuration assignment, before scheduling.
// Eligibility is independent of dtype, opcode cost, and whether another
// producer output is a reduction. Profitability MUST be established by the
// evaluator; a missing evaluator never rewrites HLO.
class RecomputeFusionSideOutputs : public HloModulePass {
 public:
  using Evaluator = std::function<absl::StatusOr<Decision>(
      const FusionRecomputeVariants& variants)>;

  // min_bytes and max_candidates bound tuning overhead, not profitability.
  explicit RecomputeFusionSideOutputs(Evaluator evaluator = {},
                                      int64_t min_bytes = 64 * 1024 * 1024,
                                      int64_t max_candidates = 16,
                                      bool analyze_only = false)
      : evaluator_(std::move(evaluator)),
        min_bytes_(min_bytes),
        max_candidates_(max_candidates),
        analyze_only_(analyze_only) {}

  absl::string_view name() const override {
    return "recompute-fusion-side-outputs";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Evaluator evaluator_;
  int64_t min_bytes_;
  int64_t max_candidates_;
  bool analyze_only_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_RECOMPUTE_FUSION_SIDE_OUTPUTS_H_
