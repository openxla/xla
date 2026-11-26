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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_EQUIVARIANT_ELIMINATOR_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_EQUIVARIANT_ELIMINATOR_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Eliminates or shrinks complete-bijection `collective-permute` instructions by
// conjugating rank-equivariant HLO subgraphs.
//
// Given a complete bijection P on [0, N) (all N devices participate with
// distinct sources and targets), let E_P be the instructions f whose per-device
// semantics commute with P (local pure deterministic HLO ops, approved pure
// local custom calls, and collectives whose ordered replica groups are closed
// under P), and let I_P be the values identical between every device r and
// P^{-1}(r) (constants, iotas, and cross-partition entry parameters whose
// sharding assigns identical tiles to r and P^{-1}(r)).
//
// Choosing a subset S of E_P to execute in the conjugated frame
// (u_new[r] = u_orig[P^{-1}(r)] for u in S, u_new[r] = u_orig[r] otherwise):
//   * cancels entry `CP_{P^{-1}}(x)` feeding S directly to `x`,
//   * cancels exit `CP_P(u)` leaving S directly to `u`,
//   * reads P-invariant values in I_P at zero communication cost,
//   * inserts one `CP_P(v)` for each non-invariant `v not in S` read by S, and
//   * inserts one `CP_{P^{-1}}(v)` for each `v in S` read outside S.
//
// The total post-conjugation communication cost is submodular in S and is
// minimized globally by a single s-t minimum cut per active bijection P. All
// per-device tensor values outside S and all module outputs remain bitwise
// identical (0 ULP).
class CollectivePermuteEquivariantEliminator : public HloModulePass {
 public:
  using CustomCallEquivariancePredicate =
      std::function<bool(const HloInstruction*)>;

  static constexpr absl::string_view kName =
      "collective-permute-equivariant-eliminator";
  static constexpr int64_t kDefaultPerCollectiveOverheadBytes = 64 * 1024;

  explicit CollectivePermuteEquivariantEliminator(
      CustomCallEquivariancePredicate is_equivariant_custom_call = nullptr,
      int64_t per_collective_overhead_bytes =
          kDefaultPerCollectiveOverheadBytes)
      : is_equivariant_custom_call_(std::move(is_equivariant_custom_call)),
        per_collective_overhead_bytes_(per_collective_overhead_bytes) {}

  absl::string_view name() const override { return kName; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  CustomCallEquivariancePredicate is_equivariant_custom_call_;
  int64_t per_collective_overhead_bytes_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_EQUIVARIANT_ELIMINATOR_H_
