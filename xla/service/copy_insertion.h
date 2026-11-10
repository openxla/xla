/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COPY_INSERTION_H_
#define XLA_SERVICE_COPY_INSERTION_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

// Copy insertion is a legalization HLO pass which inserts copies (kCopy
// instructions) to eliminate several kinds of problems in the HLO module.
//
//   (1) Entry parameter or a constant live out of the entry computation.  Entry
//       computation arguments and constants have different lifetimes than the
//       computation result and cannot share the same allocation. Parameters and
//       constants live out of non-entry computations do not need copies.
//
//   (2) Different values which are simultaneously live and which must be held
//       in the same buffer. This can occur in while bodies. Specifically, the
//       while loop state (the arguments to the while instruction) is updated
//       in-place and the update may clobber the value from the previous
//       iteration before the previous value is dead. Computations called from
//       kCall instructions do not need such copies because kCall has no update
//       in-place semantics.
//
//   (3) The buffer set of the root instruction of the entry computation must be
//       unambiguous and distinct. That is, InstructionAliasSet::IsAmbiguous and
//       InstructionAliasSet::IsDistinct return true.
class CopyInsertion : public HloModulePass {
 public:
  // Attachment direction for instructions that CopyInsertion treats as
  // transparent passthroughs during dataflow, alias, and copy removal
  // analysis.
  enum class PassthroughAttachment : uint8_t {
    // Not a transparent passthrough instruction.
    kNone,
    // Attaches to its producer (operand 0). Copies of the producer's output
    // for downstream users are placed after this instruction so it stays
    // attached to the producer.
    kProducer,
    // Attaches to its consumers (users). Copies of an operand for a consumer
    // are placed before this instruction so it stays attached to the consumer.
    kConsumer,
  };

  // Classifier returning the attachment direction of a single operand
  // passthrough instruction, or PassthroughAttachment::kNone when the
  // instruction is not a passthrough.
  //
  // REQUIRES: Every instruction classified as kProducer or kConsumer has
  // `operand_count() == 1` and
  // `ShapeUtil::Compatible(inst->shape(), inst->operand(0)->shape())`.
  using PassthroughClassifier =
      std::function<PassthroughAttachment(const HloInstruction*)>;

  absl::string_view name() const override { return "copy-insertion"; }
  static constexpr int64_t kUseRegionAnalysisLimit = 0;

  // backend specific function that decides whether an instruction
  // can share buffer with its operand.
  //
  // `view_color` is the layout memory space color of view values (address
  // stand ins with no storage of their own that alias their operand 0's
  // buffer, see BufferAssigner::Options::dus_view_color). When set, copy
  // removal counts the readers of a value's views as uses of that value, so
  // no producer is merged into the viewed buffer between a view and its
  // readers.
  //
  // `classify_passthrough` optionally identifies wrapper instructions (such as
  // memory space coloring markers) that forward their operand transparently
  // during dataflow, alias, and copy removal analysis while keeping inserted
  // copies on the appropriate side of each wrapper.
  //
  // TODO(b/80315712): Find a better way to tell whether a fusion can share
  // buffer.
  explicit CopyInsertion(
      const AliasInfo* alias_info,
      int64_t use_region_based_live_range_analysis = kUseRegionAnalysisLimit,
      std::function<bool(const HloInstruction* copy)> should_skip_removal =
          nullptr,
      std::optional<int64_t> view_color = std::nullopt,
      PassthroughClassifier classify_passthrough = nullptr)
      : alias_info_(alias_info),
        should_skip_removal_(std::move(should_skip_removal)),
        use_region_based_live_range_analysis_(
            use_region_based_live_range_analysis),
        view_color_(view_color),
        classify_passthrough_(std::move(classify_passthrough)) {}

  // When enabled, the copy added for the operand of an in-place operation is
  // ordered (via control dependencies) after the other readers of that operand
  // that are already predecessors of the in-place operation. This does not
  // constrain the schedule beyond what the in-place operation already
  // requires, but lets RemoveUnnecessaryCopies elide copies in chains of
  // "read, then update in place" steps on the same buffer, e.g. gather +
  // scatter updates of a loop-carried table across unrolled loop steps, where
  // the copies would otherwise block each other's removal. Copies that are not
  // removed keep these control dependencies.
  void set_order_predecessor_readers_before_in_place_copies(bool enabled) {
    order_predecessor_readers_before_in_place_copies_ = enabled;
  }

  // Try to remove as many copies from the module as possible without
  // introducing live range interference. Only copy instructions that are
  // eligible for copy elision are considered for removal.
  absl::Status RemoveUnnecessaryCopies(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {},
      bool insert_post_scheduling_control_dependencies = false);

  // A callback that allows backends to perform analysis on the HloAliasAnalysis
  // and suggest additional copies.
  using CustomBufferAnalysisFn = std::function<void(
      HloModule* module, const HloAliasAnalysis& alias_analysis,
      std::function<void(HloInstruction*, const ShapeIndex&)> add_copy_fn)>;

  // Add copies to address special constraints on the roots of computations not
  // related to live range interference:
  //
  //    (1) Entry computation root must be unambiguous and distinct.
  //
  //    (2) Any computation called by a kCall instruction must have an
  //        unambiguous root.
  //
  //    (3) Constants and parameters cannot be live out of the entry computation
  //
  absl::Status AddSpecialCaseCopies(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {},
      CustomBufferAnalysisFn custom_buffer_analysis = nullptr);

 protected:
  // Copy of a single operand edge of `user` needed when `user` reads the same
  // underlying value through multiple operands with conflicting sharing rules.
  struct OperandCopy {
    HloInstruction* user;
    int64_t operand_number;
    ShapeTree<bool> indices_to_copy;
  };

  // Special case copy plan produced by ComputeSpecialCaseCopies.
  struct SpecialCaseCopies {
    HloInstructionMap<ShapeTree<bool>> instructions_to_copy;
    std::vector<OperandCopy> operand_copies;

    bool empty() const {
      return instructions_to_copy.empty() && operand_copies.empty();
    }
  };

  // Override which requires the caller to pass in a call graph. Runs the alias
  // analysis, then ComputeSpecialCaseCopies and InsertSpecialCaseCopies.
  virtual absl::Status AddSpecialCaseCopies(
      const CallGraph& call_graph,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      HloModule* module, CustomBufferAnalysisFn custom_buffer_analysis);

  // Records the shape indices and operand edges that need special case copies
  // without inserting them. `alias_analysis` must be current for `module`.
  absl::StatusOr<SpecialCaseCopies> ComputeSpecialCaseCopies(
      const CallGraph& call_graph,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      HloModule* module, const HloAliasAnalysis& alias_analysis,
      CustomBufferAnalysisFn custom_buffer_analysis);

  // Inserts the copies computed by ComputeSpecialCaseCopies.
  absl::Status InsertSpecialCaseCopies(
      const SpecialCaseCopies& special_case_copies, HloModule* module) const;

  // Returns the configured attachment direction of `instruction`, or kNone.
  //
  // REQUIRES: `instruction` is not null.
  PassthroughAttachment ClassifyPassthrough(
      const HloInstruction* instruction) const;

  // Returns true if `instruction` is configured as a transparent passthrough.
  //
  // REQUIRES: `instruction` is not null.
  bool IsPassthrough(const HloInstruction* instruction) const {
    return ClassifyPassthrough(instruction) != PassthroughAttachment::kNone;
  }

  // Builds the HloDataflowAnalysis passthrough predicate for this pass, or
  // nullptr when none is configured.
  HloDataflowAnalysis::PassthroughPredicate MakePassthroughPredicate() const;

  // Runs HloAliasAnalysis configured with `alias_info_` and
  // `MakePassthroughPredicate()`.
  //
  // REQUIRES: `module` is not null and has a flattened call graph.
  absl::StatusOr<std::unique_ptr<HloAliasAnalysis>> RunAliasAnalysis(
      const HloModule* module) const;

  // Inserts a root copy on `computation` at `indices_to_copy`, collapsing any
  // zero user identity root bitcast wrapping a passthrough and recording any
  // retired bitcast in `dead_bitcasts` for post pass removal.
  //
  // REQUIRES: `computation` and `dead_bitcasts` are not null,
  // `computation->root_instruction()` is not null, and `indices_to_copy` is
  // shape compatible with `computation->root_instruction()->shape()`.
  absl::Status InsertRootCopyUnwrappingBitcast(
      HloComputation* computation, const ShapeTree<bool>& indices_to_copy,
      std::vector<HloInstruction*>* dead_bitcasts) const;

  // Removes retired root bitcasts collected in `dead_bitcasts`, updating
  // `module->schedule()` first when `module` is scheduled.
  //
  // REQUIRES: `module` is not null and every instruction in `dead_bitcasts` has
  // `user_count() == 0` and no control dependencies.
  absl::Status RemoveDeadRootBitcasts(
      HloModule* module, absl::Span<HloInstruction* const> dead_bitcasts) const;

  // Deep copies `instruction` for `target_user` (or for a computation root
  // when `target_user` is null), placing each copied array leaf between any
  // kProducer and kConsumer passthrough wrappers so both stay attached to their
  // targets.
  //
  // REQUIRES: `instruction` is not null and belongs to a computation, and any
  // non null `indices_to_copy` or `copies_added` is shape compatible with
  // `instruction->shape()`.
  absl::StatusOr<HloInstruction*> DeepCopyConsumerOperand(
      HloInstruction* instruction, const HloInstruction* target_user = nullptr,
      const ShapeTree<bool>* indices_to_copy = nullptr,
      ShapeTree<HloInstruction*>* copies_added = nullptr) const;

  // Deep copies `instruction` at `indices_to_copy` and rewires its downstream
  // uses (and computation root when `replace_root` is true and `instruction` or
  // an attached kProducer passthrough is the root), placing copies after any
  // attached kProducer passthroughs without introducing marker to marker edges.
  //
  // REQUIRES: `instruction` is not null and belongs to a computation, and
  // `indices_to_copy` is shape compatible with `instruction->shape()`.
  absl::Status InsertProducerCopy(
      HloInstruction* instruction, const ShapeTree<bool>& indices_to_copy,
      std::optional<absl::Span<HloInstruction* const>> users_override =
          std::nullopt,
      bool replace_root = true,
      const absl::flat_hash_set<const HloInstruction*>* skip_users =
          nullptr) const;

  // Wraps any computation root in `execution_threads` that is a configured
  // passthrough instruction in a zero cost bitcast (or tuple of leaf bitcasts)
  // so no passthrough remains a computation root after copy insertion or
  // removal.
  //
  // REQUIRES: `module` is not null.
  absl::Status WrapRootPassthroughsInBitcast(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  // Add copies for conditional instructions.
  virtual absl::Status AddCopiesForConditional(
      const HloDataflowAnalysis& dataflow, HloInstruction* conditional);

  // Adds copies for transitioning into and out of non-copyable values.
  absl::Status AddCopiesForNonCopyableTransitions(HloInstruction* chain_start);
  // Adds copies for transitioning into and out of non-copyable values for a
  // explicit non-copyable chain.
  absl::Status AddCopiesForExplicitNonCopyableTransitions(
      HloInstruction* chain_start);

  // Backend specific information about whether an instruction can share buffer
  // with its operand.
  const AliasInfo* alias_info_;

  // Function to determine whether a copy should be skipped during removal.
  std::function<bool(const HloInstruction* copy)> should_skip_removal_;

  // Run the pass on the given module. Returns whether the module was changed
  // (copies were inserted).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  friend class CopyInsertionTestPeer;

  absl::Status AddCopiesToResolveInterference(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
  // See set_order_predecessor_readers_before_in_place_copies.
  absl::Status OrderPredecessorReadersBeforeInPlaceCopies(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
  int64_t use_region_based_live_range_analysis_;
  std::optional<int64_t> view_color_;
  PassthroughClassifier classify_passthrough_;
  bool order_predecessor_readers_before_in_place_copies_ = false;
};

}  // namespace xla

#endif  // XLA_SERVICE_COPY_INSERTION_H_
