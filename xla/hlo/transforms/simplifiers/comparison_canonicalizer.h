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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_COMPARISON_CANONICALIZER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_COMPARISON_CANONICALIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Rewrites scalar total-order floating-point comparisons inside `kSort`
// comparator computations whose operands both canonicalize signed zeros to +0.0
// and NaNs to the same positive NaN bit pattern into weak-order comparisons on
// the underlying operands:
//   compare(canon(lhs), canon(rhs), dir, TOTAL) -> compare(lhs, rhs, dir, WEAK)
// Only applies to floating-point types with signed zero (`HasNegativeZero`),
// where positive NaNs order greatest under `TOTAL` order.
class ComparisonCanonicalizer : public HloModulePass {
 public:
  absl::string_view name() const override { return "comparison-canonicalizer"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_COMPARISON_CANONICALIZER_H_
