/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_WHILE_LOOP_CONSTANT_SINKING_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_WHILE_LOOP_CONSTANT_SINKING_H_

#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Sinks while loop invariant values that happen to be constants into the while
// loop body and conditional. This is probably not a win in isolation but may
// unlock further optimizations like constant folding.
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(v)
//     state = (..., v, ...)
//   }
//
// =>
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(const)
//     state = (..., v, ...)
//   }
//
// Note that it leaves the `v` in place to keep that component of the state
// tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
// `v`.
//
class WhileLoopConstantSinking : public HloModulePass {
 public:
  // Decides whether a candidate is sunk. `init_value` is the loop-invariant
  // constant (or broadcast of a constant) in the while init tuple, and
  // `body_gte` is the get-tuple-element of the same index in the while body.
  // Returning false leaves that element of the loop state as it is, in both the
  // body and the condition.
  using ShouldSinkFn = std::function<bool(const HloInstruction& init_value,
                                          const HloInstruction& body_gte)>;

  // If `should_sink` is null, every candidate is sunk.
  explicit WhileLoopConstantSinking(bool sink_broadcast_of_constants = false,
                                    bool sink_only_scalar_constants = false,
                                    ShouldSinkFn should_sink = nullptr)
      : sink_broadcast_of_constants_(sink_broadcast_of_constants),
        sink_only_scalar_constants_(sink_only_scalar_constants),
        should_sink_(std::move(should_sink)) {}

  ~WhileLoopConstantSinking() override = default;

  static constexpr absl::string_view kName = "while-loop-constant-sinking";
  absl::string_view name() const override { return kName; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> TrySinkingConstantsIntoWhileLoop(
      HloModule* module, HloInstruction* while_instr);

  const bool sink_broadcast_of_constants_;
  const bool sink_only_scalar_constants_;
  const ShouldSinkFn should_sink_;
};
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_WHILE_LOOP_CONSTANT_SINKING_H_
