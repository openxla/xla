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

#ifndef XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_
#define XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Sinks values into the while loop body and conditional that fusibles. This is
// probably not a win in isolation but may unlock further optimizations like
// fusible folding. There are two categories:

// 1. Sinks while loop invariant values into the while
// loop body and conditional.
//
//   state = (..., fusible_graph, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(v)
//     state = (..., v, ...)
//   }
//
// =>
//
//   state = (..., fusbile_graph, ..., fusible_graph_operands)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(fusibile_graph)
//     state = (..., v, ...)
//   }
//
// Note that it leaves the `v` in place to keep that component of the state
// tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
// `v`.
//
// With `sink_dynamic_slices_into_updates`, the sunk graph may also contain
// dynamic slices whose sliced operand stays outside the loop (it becomes loop
// state). This is done only when the body uses `v` solely as the update of
// dynamic-update-slices (through reshapes or bitcasts), nothing else reads
// `v`, and the loop disables hoisting (frontend attribute
// `_xla_disable_loop_instr_hoisting`; loop invariant code motion would hoist
// the slice back out of other loops). The slice then fuses into the update
// instead of being materialized before the loop, which matters most for loops
// that often run zero times. Other sunk graphs only add the operands of their
// broadcasts to the loop state; this keeps the whole sliced operand live
// through the loop instead of the slice, so it is opt in:
//
//   state = (..., dynamic-slice(big, i), ...)
//   while (pred(state)) {
//     (..., buf, v, ...) = state
//     state = (..., dynamic-update-slice(buf, v, j), v, ...)
//   }
//
// =>
//
//   state = (..., dynamic-slice(big, i), ..., big, i)
//   while (pred(state)) {
//     (..., buf, v, ..., big', i') = state
//     state = (..., dynamic-update-slice(buf, dynamic-slice(big', i'), j), ...)
//   }
//
// 2. Sinks constant-initialized value, i.e., broadcast(constant) into the while
// body. The high level idea is that we don't want to leave any element of the
// buffer after loop execution as undefined. Therefore, all the elements of the
// buffer must be written to in the body. For element-wise operation 'instr' we
// have:
// forall index i in output shape: instr[i] = f(operand1[i], ...),  where
// f is the elementwise operation.
// We can see that all the indices of the output shape is written to. These
// values can sink into the loop and fused later.
//
//   state = (..., broadcast(constant), ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     value = f(v) // f writes to the entire shape of v.
//     state = (..., value, ...)
//   }
//
// =>
//
//   state = (..., allocate-buffer(), ...)
//   while (pred(state)) {
//     i = iteration_var
//     (..., v, ...) = state
//     new_v = select(i == 0, broadcast(constant), v)
//     value = f(new_v)
//     state = (..., value, ...)
//   }
//
//   This transformation replaces the broadcast with a free AllocateBuffer
//   outside the while loop with the hope that the broadcast inside the loop
//   will be fused.
class WhileLoopFusibleSinking : public HloModulePass {
 public:
  explicit WhileLoopFusibleSinking(
      bool sink_broadcast_of_constant = true,
      bool sink_dynamic_slices_into_updates = false)
      : sink_broadcast_of_constant_(sink_broadcast_of_constant),
        sink_dynamic_slices_into_updates_(sink_dynamic_slices_into_updates) {}

  ~WhileLoopFusibleSinking() override = default;

  absl::string_view name() const override {
    return "while-loop-fusible-sinking";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Sink a fusible subgraph into a while loop.
  absl::StatusOr<bool> TrySinkingFusiblesIntoWhileLoop(
      HloInstruction* while_instr);

  // The idea is that certain constant-initialized buffers can be left as
  // uninitialized if all the elements of the buffer are written to in the loop
  // body. This way, we eliminate the need to initialize the buffer (with
  // broadcast) in the critical path of the program. To summarize, the
  // conditions to apply this optimization are:
  // 1. The buffer is a constant-initialized buffer.
  // 2. All the elements of the buffer are written to in the loop body.
  // 3. The iteration variable of the loop is monotonically increasing or
  // decreasing.
  // The optimization is applied by creating a select between the initial value
  // and the value in the body. The select is guarded by a predicate that checks
  // if the loop iteration variable is equal to the first iteration value.
  absl::StatusOr<bool> TryRewritingBroadcastAsAllocateBuffer(
      HloInstruction* while_instr);

  // Creates a loop fusion instruction containing the computation to move into
  // the while loop to avoid conflicts with actual instruction fusion, the loop
  // fusion will be defused. `IsSinkableFusion` collects the instructions to
  // fuse in `nodes` and sets `has_slice` if one of them is a dynamic slice.
  // For such a graph, pass `nodes` as `sliced_graph` to `CreateSinkableFusion`
  // so that the sliced operand and computed start indices stay outside.
  bool IsSinkableFusion(HloInstruction* while_operand,
                        absl::flat_hash_set<const HloInstruction*>& nodes,
                        bool& has_slice);
  HloInstruction* CreateSinkableFusion(
      HloInstruction* while_operand,
      const absl::flat_hash_set<const HloInstruction*>* sliced_graph);
  // Returns true if the graph `nodes` rooted at `value`, which has a dynamic
  // slice, may sink into every loop that reads `value`; those loops, with the
  // element each reads, go to `loops`.
  bool CanSinkSlice(const HloInstruction* value,
                    const absl::flat_hash_set<const HloInstruction*>& nodes,
                    std::vector<std::pair<HloInstruction*, int64_t>>& loops);

  absl::flat_hash_map<HloComputation*, int> call_counts_;
  // Loops that read a sunk dynamic slice through another loop's output, and
  // the (loop, element) pairs that already sank one.
  std::vector<HloInstruction*> slice_readers_;
  absl::flat_hash_set<std::pair<const HloInstruction*, int64_t>> sunk_slices_;
  const bool sink_broadcast_of_constant_;
  const bool sink_dynamic_slices_into_updates_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_FUSIBLE_SINKING_H_
