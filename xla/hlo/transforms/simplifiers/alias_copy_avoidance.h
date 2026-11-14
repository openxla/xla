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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALIAS_COPY_AVOIDANCE_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALIAS_COPY_AVOIDANCE_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Drops the optional (kMayAlias) input/output aliases that cannot be kept
// without copying the parameter.
//
// An aliased parameter P shares its buffer with output O, so the instruction W
// that computes O overwrites P. If an instruction that reads P depends on W,
// no schedule reads P before W writes it, and copy insertion has to copy P.
// Without the alias O gets a buffer of its own and P needs no copy. Values do
// not change.
//
// An alias stays if it is required (kMustAlias), if every reader of P can run
// before W, if O is the value of a parameter or of a constant (copy insertion
// copies those into O and can do so after the readers of P), or if P is also
// returned as it is (that output needs a copy of P either way).
//
// The price is memory. With the alias the program holds one buffer for P and O
// from start to end, and the copy only for the part of the program that needs
// both the old and the new value. Without the alias it holds two buffers from
// start to end: P's, which stays with the caller, and O's, which is allocated
// before the program runs. Peak memory grows by up to the size of O, by all of
// it when the program peaks while the copy is dead. Add the pass only where
// time matters more than this memory.
//
// P does not become a buffer donor: the alias may be one that
// OptimizeInputOutputBufferAlias set up for a parameter that nobody donated.
// PjRt donates only aliased parameters, so it no longer takes P from the
// caller: the caller's buffer stays valid, and allocated, until the caller
// releases it.
//
// Run the pass after OptimizeInputOutputBufferAlias, which could set up the
// same alias again, and right before copy insertion.
class AliasCopyAvoidance : public HloModulePass {
 public:
  static constexpr absl::string_view kName = "alias-copy-avoidance";
  absl::string_view name() const override { return kName; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALIAS_COPY_AVOIDANCE_H_
