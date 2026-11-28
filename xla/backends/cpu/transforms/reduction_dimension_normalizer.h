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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_REDUCTION_DIMENSION_NORMALIZER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_REDUCTION_DIMENSION_NORMALIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::cpu {

// Removes static size-one dimensions from reductions and their elementwise
// producers. Run after reshape sinking and before layout assignment/fusion so
// that size-one minor dimensions do not prevent vectorization on CPU.
class ReductionDimensionNormalizer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "cpu-reduction-dimension-normalizer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_REDUCTION_DIMENSION_NORMALIZER_H_
