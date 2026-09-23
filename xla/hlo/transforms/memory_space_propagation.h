/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_MEMORY_SPACE_PROPAGATION_H_
#define XLA_HLO_TRANSFORMS_MEMORY_SPACE_PROPAGATION_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// This is a legalization pass that propagates the memory space (and associated
// split config) in the layout to the fusion computations.
//
// The pass only reads values defined inside fusion computations, where a
// value flows through nothing but get-tuple-element, tuple, add-dependency,
// domain, optimization-barrier and the nested elements of copy. Run() computes
// those values from the fusion computation itself, and builds a whole module
// HloDataflowAnalysis only when a fusion computation holds an instruction with
// other dataflow (a call, control flow or an asynchronous chain). Both ways
// see the same values.
class MemorySpacePropagation : public HloModulePass {
 public:
  explicit MemorySpacePropagation(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis = nullptr)
      : dataflow_analysis_(std::move(dataflow_analysis)) {}
  ~MemorySpacePropagation() override = default;
  absl::string_view name() const override { return "memory-space-propagation"; }

  // Propagates the memory space (and associated split config) in the layout to
  // a given fusion computation. Returns true if the computation is modified.
  // Requires the dataflow analysis given to the constructor; Run() discards
  // that analysis.
  bool RunOnComputation(HloComputation* computation);

  // Returns true if no fusion computation of module on the given execution
  // threads (all threads when empty) holds an instruction whose dataflow
  // DefiningPosition() and Positions() do not model, so that Run() needs no
  // HloDataflowAnalysis for those threads. Run() on those threads only enters
  // fusion computations on those threads: a fusion instruction and the
  // computation it calls share a thread (HloVerifier's
  // CheckCallableInstructionThreadName).
  static bool HasLocalFusionDataflow(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Returns the position defining the value at position, which must be in a
  // fusion computation with local dataflow: the same value HloDataflowAnalysis
  // with ssa_form=false and bitcast_defines_value=true reports there.
  static HloPosition DefiningPosition(HloPosition position);

  // Returns every position holding the value defined at defining, that
  // position first, under the same analysis configuration.
  static absl::InlinedVector<HloPosition, 4> Positions(
      const HloPosition& defining);

  // Returns the uses at fusion instructions of the value held at positions.
  static absl::InlinedVector<HloUse, 1> FusionUses(
      absl::Span<const HloPosition> positions);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // What Propagate reads about a value: every position holding it (the
  // defining one first) and its uses at fusion instructions.
  struct ValueView {
    absl::InlinedVector<HloPosition, 4> positions;
    absl::InlinedVector<HloUse, 1> fusion_uses;
  };

  // The position defining the value at (instruction, index), and the view
  // of the value defined at defining. Both read dataflow_analysis_ when
  // there is one and compute the answer from the fusion computation otherwise.
  HloPosition DefiningPositionAt(HloInstruction* instruction,
                                 ShapeIndexView index) const;
  ValueView ViewValue(const HloPosition& defining) const;

  // Given the shape index (operand or output) and its corresponding instruction
  // in the fused computation (parameter or root), propagates the memory space
  // (and associated split config) in the callee side. Returns true if the
  // module is modified.
  bool Propagate(ShapeIndexView index, HloInstruction* callee_instruction,
                 const Shape& src_shape,
                 absl::flat_hash_set<HloPosition>& visited) const;

  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_MEMORY_SPACE_PROPAGATION_H_
