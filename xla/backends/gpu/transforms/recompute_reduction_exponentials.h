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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_RECOMPUTE_REDUCTION_EXPONENTIALS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_RECOMPUTE_REDUCTION_EXPONENTIALS_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Recomputes large FP32 exponential side outputs of native reduction fusions
// inside their sole consumer, provided the consumer already computes the exact
// input to the exponential. Run after multi-output fusion. This preserves the
// reduction and avoids both materializing its exponential input and extending
// the lifetimes of the logits or row statistics.
class RecomputeReductionExponentials : public HloModulePass {
 public:
  // Keep small intermediates, which may benefit from cache reuse. The override
  // allows tests to exercise the transformation with small arrays.
  explicit RecomputeReductionExponentials(int64_t min_bytes = 64 * 1024 * 1024)
      : min_bytes_(min_bytes) {}

  absl::string_view name() const override {
    return "recompute-reduction-exponentials";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t min_bytes_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_RECOMPUTE_REDUCTION_EXPONENTIALS_H_
