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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONDITIONAL_RESIDUAL_ELISION_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONDITIONAL_RESIDUAL_ELISION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Removes the copies and fills of values that one conditional passes to
// another conditional taking the same branch.
//
// Partial evaluation of a conditional (JAX's cond under autodiff or remat)
// splits it into a conditional C that computes residuals and a conditional D
// that consumes them. Both are in one computation and select their branch
// with the same instruction, so D takes the branch that C took. Every branch
// of C returns the residuals of all branches: it fills the slots of the other
// branches, usually with zeros, and a branch that saves one of its operands
// returns it, which copy insertion turns into a copy. This pass rewrites:
//
//  * D's operand of branch b holds output j of C, and branch b of C returns
//    element p of its operand as output j, possibly bitcast or passed through
//    an optimization barrier: D's operand of branch b holds that element of
//    C's operand instead. D only reads its operand of branch b when it takes
//    branch b, and then C took branch b.
//  * Output j of C is only read as an element of the operands of conditionals
//    taking the same branch as C: each branch of C in which none of them reads
//    it returns an uninitialized buffer (the custom call "AllocateBuffer",
//    which the backend must implement). An output without readers is removed.
//
// A program reads the same values before and after. The second rewrite skips
// conditionals whose branch computations have other callers, so flatten the
// call graph first. Run the pass before copy insertion: the first rewrite
// makes D read a value that C also reads, and copy insertion protects it like
// any value with several readers. Run HloDCE afterwards.
class ConditionalResidualElision : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "conditional-residual-elision";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONDITIONAL_RESIDUAL_ELISION_H_
