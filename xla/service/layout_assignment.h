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

#ifndef XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
#define XLA_SERVICE_LAYOUT_ASSIGNMENT_H_

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/map_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"

namespace xla {

class LayoutAssignment;

// Abstract base class representing a layout constraint in LayoutAssignment.
//
// In XLA, a layout constraint specifies a requirement or preference for how
// tensors/buffers in memory are laid out (e.g., minor-to-major dimension
// ordering, tiling, padding). Hardware backends and specific HLO operations
// (such as convolutions or matrix multiplications) impose layout constraints
// on their inputs, outputs, or computation interfaces to ensure correctness or
// optimize execution speed.
//
// During layout assignment, constraint objects are created, prioritized, and
// propagated across the HLO graph. When conflicting constraints arise, the
// constraint with higher priority takes precedence, and layout copy
// instructions (`kCopy`) are inserted where necessary to bridge incompatible
// layouts.
//
// Specific constraint types derived from LayoutConstraint include:
// - BufferLayoutConstraint: Constrains the layout of a LogicalBuffer produced
//   by an instruction.
// - OperandLayoutConstraint: Constrains the layout expected for an operand by
//   its consumer instruction.
// - ComputationLayoutConstraint: Constrains the parameter and result layouts
//   of an HloComputation interface.
class LayoutConstraint {
 public:
  LayoutConstraint(bool mandatory, bool dfs, int64_t priority)
      : mandatory_(mandatory), dfs_(dfs), priority_(priority) {}
  virtual ~LayoutConstraint() = default;

  // Returns a string representation of the constraint for debugging/logging.
  virtual std::string ToString() const = 0;

  // Returns true if this constraint is mandatory and cannot be overwritten
  // by a lower or equal priority constraint.
  bool mandatory() const { return mandatory_; }

  // Returns true if this constraint should be propagated in DFS (depth-first
  // search) order, or false for BFS (breadth-first search) propagation.
  bool dfs() const { return dfs_; }

  // Returns the priority of this constraint. When conflicting constraints are
  // encountered, higher priority constraints override lower priority ones.
  int64_t priority() const { return priority_; }

  // Priority of default/fallback layouts when not explicitly specified.
  static constexpr int64_t kDefaultPriority = -2;
  // Beginning priority level used when layout assignment starts.
  static constexpr int64_t kBeginningPriority = 0;
  // Priority assigned to user-specified layouts on the entry computation.
  static constexpr int64_t kGivenPriority = 3;

 protected:
  // Whether this constraint is mandatory (cannot be overridden).
  bool mandatory_;

  // Whether constraint propagation proceeds in DFS (true) or BFS (false)
  // order.
  bool dfs_;

  // Priority of the constraint used for resolving conflicts.
  int64_t priority_;
};

std::ostream& operator<<(std::ostream& out, const LayoutConstraint& constraint);

// Layout constraint on a single LogicalBuffer.
//
// BufferLayoutConstraint specifies the memory layout of an array buffer
// produced by a specific HloInstruction. Unlike OperandLayoutConstraint (which
// constrains a use site and can be satisfied via an inserted copy), a
// BufferLayoutConstraint directly constrains the definition of the logical
// buffer.
//
// Data Structure Usage:
// - `buffer_`: Points to the LogicalBuffer instance whose output layout is
//   constrained.
// - `layout_`: Inlined vector storing the assigned Layout object(s).
// - `enforcing_instruction_`: Optional pointer to the HloInstruction that
//   enforced or induced this buffer layout constraint.
class BufferLayoutConstraint : public LayoutConstraint {
 public:
  // Constructs a BufferLayoutConstraint for the given HloValue.
  BufferLayoutConstraint(const Layout& layout, const HloValue& buffer,
                         bool mandatory, bool dfs, int64_t priority,
                         const HloInstruction* enforcing_instruction = nullptr);

  // Returns the constrained HloValue.
  const HloValue& buffer() const { return *buffer_; }

  // Returns the target Layout for the buffer.
  const Layout& layout() const { return layout_[0]; }

  const HloInstruction* enforcing_instruction() const {
    return enforcing_instruction_;
  }
  bool negotiated() const { return negotiated_; }

  // Updates the buffer layout if allowed by priority and mandatory rules,
  // propagating changes back to LayoutAssignment.
  bool UpdateLayout(int64_t priority, const Layout& layout, bool mandatory,
                    bool dfs, LayoutAssignment* assignment,
                    const HloInstruction* from_user = nullptr);

  std::string ToString() const override;

 private:
  // The constrained layout(s) for the buffer.
  absl::InlinedVector<Layout, 2> layout_;

  // The HloValue being constrained.
  const HloValue* buffer_;

  // The instruction that enforced or induced this constraint, if any.
  const HloInstruction* enforcing_instruction_ = nullptr;

  // Whether the active layout was set via NegotiateLayout.
  bool negotiated_ = false;
};

// Layout constraint on an operand of an instruction.
//
// OperandLayoutConstraint specifies the layout required or preferred for an
// operand by a consumer HloInstruction. The constrained operand shape can be
// an array or a tuple.
//
// Unlike BufferLayoutConstraint, this is a constraint on the USE of a shaped
// value rather than a hard constraint on the instruction defining the value.
// If the defining instruction produces a different layout, copy instructions
// (`kCopy`) can be inserted between the definition and use to satisfy this
// constraint.
//
// Data Structure Usage:
// - `instruction_`: Points to the consumer HloInstruction imposing the
//   operand constraint.
// - `operand_no_`: The zero-based index of the operand being constrained.
// - `shape_layout_`: The expected ShapeLayout for the operand.
class OperandLayoutConstraint : public LayoutConstraint {
 public:
  // Constructs an OperandLayoutConstraint for the specified operand slot.
  OperandLayoutConstraint(const ShapeLayout& shape_layout,
                          const HloInstruction* instruction, int64_t operand_no,
                          bool mandatory, bool dfs, int64_t priority);

  // Returns the expected ShapeLayout for the operand.
  const ShapeLayout& shape_layout() const { return shape_layout_; }

  // Returns the consumer HloInstruction imposing this constraint.
  const HloInstruction* instruction() const { return instruction_; }

  // Returns the index of the constrained operand.
  int64_t operand_no() const { return operand_no_; }

  // Returns the operand HloInstruction being constrained.
  const HloInstruction* operand() const {
    return instruction_->operand(operand_no_);
  }

  // Updates the operand layout if allowed by priority and mandatory rules.
  bool UpdateLayout(int64_t priority, const Shape& new_shape, bool mandatory,
                    bool dfs);

  std::string ToString() const override;

 private:
  // The expected shape layout for the operand.
  ShapeLayout shape_layout_;

  // The consumer instruction imposing the constraint.
  const HloInstruction* instruction_;

  // The index of the operand in the consumer instruction.
  int64_t operand_no_;
};

// Encapsulates layout constraints on the interface of an HloComputation.
//
// ComputationLayoutConstraint specifies and tracks layout requirements on
// the inputs (parameters) and output (result) of an HloComputation. It wraps
// a ComputationLayout object alongside state flags that record which parts of
// the interface layout have been explicitly set during layout assignment.
//
// Data Structure Usage:
// - `layout_state_`: A bitmask storing layout state flags
//   (kDefaultLayoutIsUsed or kResultLayoutIsSet). Used to track whether
//   the computation result layout is explicitly constrained vs. using default
//   layouts.
// - `computation_layout_`: An instance of ComputationLayout containing the
//   actual ShapeLayout objects for all parameters and the result of the
//   computation.
class ComputationLayoutConstraint : public LayoutConstraint {
 public:
  // Layout state flags indicating which components of the computation
  // interface layout have been explicitly constrained.
  static constexpr int64_t kDefaultLayoutIsUsed = 0;
  static constexpr int64_t kResultLayoutIsSet = 1;

  // Constructs a ComputationLayoutConstraint for the given computation.
  // If computation_layout is nullptr, initializes a default layout based on
  // the computation's program shape.
  explicit ComputationLayoutConstraint(const HloComputation* computation,
                                       ComputationLayout* computation_layout,
                                       int64_t priority)
      : LayoutConstraint(/*mandatory=*/true, /*dfs=*/true, priority),
        computation_(computation),
        layout_state_((computation_layout == nullptr) ? kDefaultLayoutIsUsed
                                                      : kResultLayoutIsSet),
        computation_layout_(
            (computation_layout == nullptr)
                ? ComputationLayout(
                      computation->ComputeProgramShape(),
                      // Computation callers need layout to be set and
                      // computation parameters may miss the layout, so we
                      // cannot rely on them and need to reset/ignore the
                      // layout. Entry computation is special because unset
                      // layouts there are used to indicate that the layout
                      // should be automatically inferred.
                      /*ignore_layouts=*/!computation->IsEntryComputation())
                : *computation_layout) {}

  const HloComputation* computation() const { return computation_; }

  // Accessors for the underlying ComputationLayout.
  const ComputationLayout& computation_layout() const {
    return computation_layout_;
  }
  ComputationLayout* mutable_computation_layout() {
    return &computation_layout_;
  }

  // Resets the computation layout and priority, updating state flags for
  // the result layout according to the propagation flag.
  void ResetComputationLayout(const ComputationLayout& layout, int64_t priority,
                              bool prop_result_layout) {
    computation_layout_ = layout;
    priority_ = priority;
    if (prop_result_layout) {
      layout_state_ |= kResultLayoutIsSet;
    }
  }

  // Resets only the result shape layout, marking result_layout_is_set.
  void ResetResultLayout(const ShapeLayout& shape_layout, int64_t priority) {
    *computation_layout_.mutable_result_layout() = shape_layout;
    layout_state_ |= kResultLayoutIsSet;
    priority_ = priority;
  }

  // Returns true if the result layout has been explicitly set.
  bool result_layout_is_set() const {
    return layout_state_ & kResultLayoutIsSet;
  }

  // Returns true if the default layout is currently used.
  bool default_layout_is_used() const {
    return layout_state_ == kDefaultLayoutIsUsed;
  }

  std::string ToString() const override;

 private:
  const HloComputation* computation_;

  // Bitmask tracking whether the computation layout is using defaults, or
  // whether parameter/result layouts have been explicitly constrained.
  int64_t layout_state_;

  // The computation interface layout (parameter and result shape layouts).
  ComputationLayout computation_layout_;
};

// HLO pass which assigns layouts to all instructions in the HLO module while
// satisfying all necessary invariants and minimizing cost.
class LayoutAssignment : public HloModulePass, public HloDataflowPropagation {
 public:
  // entry_computation_layout is modified to populate a layout for the result in
  // the case that no particular layout is requested.
  explicit LayoutAssignment(ComputationLayout* entry_computation_layout);
  ~LayoutAssignment() override = default;
  using HloModulePass::Run;
  absl::string_view name() const override { return "layout-assignment"; }

  // Encapsulates all layout constraints associated with a single
  // HloComputation.
  //
  // LayoutConstraints tracks two primary categories of constraints scoped to a
  // computation:
  // 1. Computation interface layout constraints (ComputationLayoutConstraint):
  //    Specifies layout constraints on the parameters and result of the
  //    HloComputation interface.
  // 2. Operand layout constraints (OperandLayoutConstraint):
  //    Specifies layout constraints on individual operands consumed by specific
  //    instructions within the computation.
  //
  // Data Structure Usage:
  // - `computation_`: Points to the HloComputation instance whose layouts are
  //   being constrained.
  // - `computation_constraint_`: A ComputationLayoutConstraint object that
  //   wraps the ComputationLayout (parameter and result shape layouts) along
  //   with state metadata (such as priority and whether parameter/result
  //   layouts are explicitly set or defaults).
  // - `operand_constraints_`: An absl::flat_hash_map mapping an
  //   OperandConstraintKey pair (const HloInstruction*, int64_t operand_no) to
  //   a std::unique_ptr<OperandLayoutConstraint>. This lookup structure enables
  //   efficient retrieval, modification, and management of layout constraints
  //   placed on specific operand positions of instructions in the computation.
  class LayoutConstraints {
   public:
    explicit LayoutConstraints(HloComputation* computation,
                               ComputationLayout* computation_layout,
                               int64_t priority)
        : computation_(computation),
          computation_constraint_(computation, computation_layout, priority) {}
    ~LayoutConstraints() = default;

    // Returns the HloComputation associated with these layout constraints.
    const HloComputation* computation() const { return computation_; }
    HloComputation* computation() { return computation_; }

    // Returns the constrained ShapeLayout for the given instruction operand,
    // or nullptr if the operand is unconstrained.
    const ShapeLayout* OperandLayout(const HloInstruction* instruction,
                                     int64_t operand_no) const;

    // Returns the OperandLayoutConstraint for the given instruction operand,
    // or nullptr if none exists.
    const OperandLayoutConstraint* GetOperandLayoutConstraint(
        const HloInstruction* instruction, int64_t operand_no) const;

    // Returns a mutable reference to the unique_ptr<OperandLayoutConstraint>
    // for the given instruction operand slot, inserting an entry in
    // operand_constraints_ if it does not already exist.
    std::unique_ptr<OperandLayoutConstraint>& MutableOperandLayoutConstraint(
        const HloInstruction* instruction, int64_t operand_no);

    // Returns the ShapeLayout of the computation's result if set (or if this
    // is the entry computation), otherwise returns nullptr.
    const ShapeLayout* ResultLayout() const;

    // Sets the result layout for the computation and registers the updated
    // computation constraint with the LayoutAssignment pass.
    void SetResultLayout(LayoutAssignment* assignment,
                         const Shape& shape_with_layout, int64_t priority);

    // Accessors for the underlying ComputationLayout and
    // ComputationLayoutConstraint.
    const ComputationLayout& computation_layout() const {
      return computation_constraint_.computation_layout();
    }
    const ComputationLayoutConstraint& computation_constraint() const {
      return computation_constraint_;
    }
    ComputationLayoutConstraint* mutable_computation_constraint() {
      return &computation_constraint_;
    }

   private:
    // Key identifying a specific operand position of an instruction within the
    // computation.
    using OperandConstraintKey = std::pair<const HloInstruction*, int64_t>;

    // Map from (instruction, operand_no) pair to its OperandLayoutConstraint.
    absl::flat_hash_map<OperandConstraintKey,
                        std::unique_ptr<OperandLayoutConstraint>>
        operand_constraints_;

    // The computation whose values and interface are constrained.
    HloComputation* computation_;

    // Constraint on the computation interface (parameters and result layouts).
    ComputationLayoutConstraint computation_constraint_;
  };

  // Determines whether an instruction can change layouts. An instruction not
  // being able to change layout means that it requires operands with the same
  // rank as the output to have the same layout as the output.
  static bool InstructionCanChangeLayout(const HloInstruction* instruction);

  const LayoutConstraints& computation_constraints(
      const HloComputation* computation) const {
    return *FindOrDie(computation_layouts_, computation);
  }

  LayoutConstraints* mutable_computation_constraints(
      HloComputation* computation) {
    auto it = computation_layouts_.find(computation);
    LayoutConstraints* constraints = nullptr;
    if (it == computation_layouts_.end()) {
      computation_layouts_.emplace(
          computation,
          constraints = new LayoutConstraints(
              computation, nullptr, LayoutConstraint::kDefaultPriority));
    } else {
      constraints = (*it).second.get();
    }
    return constraints;
  }
  void PushAddedConstraints(const LayoutConstraint* constraint);

  // In case of an array shape returns true iff it is at most rank 1. In case of
  // a tuple shape returns true iff all leaf shapes are at most rank 1.
  static bool IsAtMostRank1(const Shape& shape);
  // Convenience wrapper around SetOperandLayout for setting the layout of a
  // operand using a Layout object. The operand must be array-shaped.
  absl::Status SetArrayOperandLayout(const Layout& layout,
                                     const HloInstruction* instruction,
                                     int64_t operand_no, bool mandatory = true,
                                     bool dfs = true) {
    return SetArrayOperandLayout(layout, instruction, operand_no, mandatory,
                                 dfs, current_priority_);
  }
  absl::Status SetArrayOperandLayout(const Layout& layout,
                                     const HloInstruction* instruction,
                                     int64_t operand_no, bool mandatory,
                                     bool dfs, int64_t priority);
  // Convenience wrapper around SetBufferLayout. Sets the layouts of all buffers
  // created by the instruction to the layouts in the given shape. The
  // instruction must define every logical buffer in its output.
  // If `allow_alias` is false, the function will check that all output buffers
  // are defined by `instruction`, not aliased to an instruction elsewhere.
  absl::Status SetInstructionLayout(const Shape& shape_with_layout,
                                    const HloInstruction* instruction,
                                    bool mandatory = true, bool dfs = true,
                                    bool allow_alias = false) {
    return SetInstructionLayout(shape_with_layout, instruction, mandatory, dfs,
                                allow_alias, current_priority_);
  }
  absl::Status SetInstructionLayout(const Shape& shape_with_layout,
                                    const HloInstruction* instruction,
                                    bool mandatory, bool dfs, bool allow_alias,
                                    int64_t priority);
  // Add a constraint on the layout of a LogicalBuffer, the layout of the
  // operand of the instruction, or the layout of the result of the computation,
  // respectively.
  absl::Status SetBufferLayout(const Layout& layout, const HloValue& buffer,
                               bool mandatory = true, bool dfs = true) {
    return SetBufferLayout(layout, buffer, mandatory, dfs, current_priority_);
  }
  absl::Status SetBufferLayout(const Layout& layout, const HloValue& buffer,
                               bool mandatory, bool dfs, int64_t priority,
                               const HloInstruction* from_user = nullptr);
  absl::Status SetOperandLayout(const Shape& shape_with_layout,
                                const HloInstruction* instruction,
                                int64_t operand_no, bool mandatory = true,
                                bool dfs = true) {
    return SetOperandLayout(shape_with_layout, instruction, operand_no,
                            mandatory, dfs, current_priority_);
  }
  absl::Status SetOperandLayout(const Shape& shape_with_layout,
                                const HloInstruction* instruction,
                                int64_t operand_no, bool mandatory, bool dfs,
                                int64_t priority);

  ComputationLayout& saved_entry_computation_layout() {
    return saved_entry_computation_layout_;
  }

  virtual bool NegotiateLayout(const HloInstruction* instruction,
                               const Layout& new_layout,
                               const Layout& existing_layout,
                               const HloInstruction* from_user,
                               const HloInstruction* orig_user) {
    return false;
  }
  // Returns true if overriding an operand buffer's layout during
  // cross computation propagation is allowed to override the existing operand
  // layout constraint on (user, operand_no).
  virtual bool CanOverrideOperandLayoutOnBufferOverride(
      const HloInstruction* user, int64_t operand_no, const Layout& new_layout,
      const Layout& existing_layout);
  // Should be made consistent with the ChooseOperandLayoutFromOutputLayout
  // except that a boolean instead of concrete layout is returned.
  virtual bool OperandLayoutAlwaysPropagateForward(const HloInstruction* user);
  // Controls when all operands of user must have the same layout.
  virtual bool OperandLayoutAlwaysPropagateToSiblings(
      const HloInstruction* user);
  // Whether to propagate the reduction layout to the operand by preserving the
  // same relative order of the dimensions that are kept, and making the
  // reduction dims the most minor dimensions.
  virtual bool PropagateReductionLayoutToOperand(const HloInstruction* user) {
    return false;
  }

 protected:
  // These methods, invoked by PropagateConstraints, propagate a layout
  // constraint to its neighbors (i.e. operands and users) in order to minimize
  // the cost of the instructions being constrainted on. New constraints are
  // added to the given constraint set.
  //
  // Backends can override these methods with backend-specific propagation
  // rules.
  virtual absl::Status PropagateBufferConstraint(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);
  virtual absl::Status PropagateOperandConstraint(
      const OperandLayoutConstraint& operand_constraint,
      LayoutConstraints* constraints);
  virtual absl::Status PropagateResultConstraint(
      const ComputationLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);

  virtual Layout GetUnconstrainedLayout(const HloValue& buffer) {
    Layout layout = LayoutUtil::GetDefaultLayoutForShape(buffer.shape());
    if (buffer.shape().has_layout()) {
      layout.set_memory_space(buffer.shape().layout().memory_space());
    }
    return layout;
  }

  const BufferLayoutConstraint* GetBufferLayoutConstraint(
      const HloValue& buffer) const;
  using BufferSet = absl::flat_hash_set<const HloValue*>;
  // Find a bufferset in the bufferset cache. This is useful since we can
  // currently create the flattened buffer set for the same instruction many
  // times, which is often slow.
  BufferSet* GetBufferSet(const HloInstruction* instruction) const;
  // Similar to above, but returns true only if all buffers associated with that
  // operand are forwarded.
  bool AllOperandBuffersForwarded(const HloInstruction* instruction,
                                  int64_t operand_no) const;
  // Returns true if any buffer in the given operand is forwarded to the output
  // of the given instruction. For example, the Tuple instruction forwards the
  // buffers of its operands and would return true for each of its operands.
  bool AnyOperandBufferForwarded(const HloInstruction* instruction,
                                 int64_t operand_no) const;
  absl::StatusOr<Layout> InferArrayLayout(const HloInstruction* instruction,
                                          const ShapeIndex& index);

  // Propagates a buffer layout constraint into the operands that use it.
  absl::Status PropagateBufferConstraintToUses(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);

  // Propagates a layout constraint on the use of the result of the given
  // instruction to the definitions of the LogicalBuffers which make up the
  // result.
  absl::Status PropagateUseConstraintToDefs(
      const ShapeLayout& shape_layout, const HloInstruction* instruction,
      LayoutConstraints* constraints, int64_t priority,
      const HloInstruction* user = nullptr);

  // Chooses a layout of operand `operand_no` of `instruction` that minimizes
  // the cost of `instruction`. `output_layout` is the layout of `instruction`.
  // Returns null if it can't decide the best layout.
  // Precondition: `instruction` and the operand are array-shaped.
  virtual std::unique_ptr<Layout> ChooseOperandLayoutFromOutputLayout(
      const Layout& output_layout, const HloInstruction* instruction,
      int64_t operand_no);
  // Given the layout of `user`'s `operand_no`-th operand, chooses a layout of
  // `user` that minimizes its cost on that operand.  Returns null if it can't
  // decide the best layout.
  // Precondition: `user` and the operand are array-shaped.
  virtual std::unique_ptr<Layout> ChooseOutputLayoutFromOperandLayout(
      const Layout& operand_layout, const HloInstruction* user,
      int64_t operand_no);

  // Convenient wrapper for InstructionCanChangeLayout.
  bool InstructionCanChangeLayoutInstance(
      const HloInstruction* instruction) const {
    return InstructionCanChangeLayout(instruction);
  }

  // For a custom-call user, propagates the operand constraint to the result
  // based on output-to-operand aliasing.
  absl::Status PropagateOperandConstraintToResultForCustomCall(
      const HloInstruction* user,
      const OperandLayoutConstraint& operand_constraint);

  // Assign layouts to the given module. Returns whether the module was changed
  // (any layouts were changed).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Initializes the layout assignment object for a new Run() call.
  absl::Status Init(HloModule* module);

  // Clones conditional computations with multiple callsites and adds copies
  // for operands of Send and layout-constrained CustomCall instructions.
  absl::Status PrepareHloForLayoutAssignment(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Verifies that the entry computation layout is compatible with the entry
  // computation shape.
  absl::Status VerifyEntryComputationLayout(const HloModule* module) const;

  // Sets up propagation by running points-to analysis, gathering computations
  // to work on, and initializing entry computation constraints.
  absl::StatusOr<std::vector<HloComputation*>> SetupPropagation(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Synchronizes, propagates, and resolves entry input_output_alias_config
  // layouts across parameters and results.
  void SynchronizeForcedInputOutputAliasing(HloModule* module);
  absl::Status PropagateConstrainedInputOutputAliasing(
      HloModule* module,
      absl::Span<HloComputation* const> computations_to_work);
  absl::Status ResolveInputOutputAliasing(HloModule* module);

  // Adds constraints which must be satisfied for correctness on all
  // backends. Called once prior to propagating constraints.
  absl::Status AddMandatoryConstraints(LayoutConstraints* constraints);

  // Adds constraints for instructions that define values with pre-existing
  // layouts.
  absl::Status AddInstructionLayoutConstraints(LayoutConstraints* constraints);

  absl::Status AddInfeedConstraints(HloInstruction* instruction);
  absl::Status AddOutfeedConstraints(HloInstruction* instruction);
  absl::Status AddParameterConstraints(HloInstruction* instruction,
                                       LayoutConstraints* constraints);
  absl::Status AddCollectiveConstraints(HloInstruction* instruction);

  // Adds constraints for instructions that call or interact with
  // sub-computations.
  absl::Status AddSubcomputationLayoutConstraints(
      LayoutConstraints* constraints);

  absl::Status AddCallConstraints(HloInstruction* instruction);
  absl::Status AddWhileConstraints(HloInstruction* instruction);
  absl::Status AddConditionalConstraints(HloInstruction* instruction);
  absl::Status AddAsyncProducerConstraints(
      HloInstruction* instruction, bool elevate_result_priority = false);
  absl::Status AddAsyncDoneConstraints(HloInstruction* instruction,
                                       LayoutConstraints* constraints);
  absl::Status SetInstructionLayout(const Shape& shape_with_layout,
                                    const HloInstruction* instruction,
                                    bool mandatory, bool dfs, bool allow_alias,
                                    int64_t priority,
                                    bool elevate_mandatory_priority);
  absl::Status SetOperandLayout(const Shape& shape_with_layout,
                                const HloInstruction* instruction,
                                int64_t operand_no, bool mandatory, bool dfs,
                                int64_t priority,
                                bool elevate_mandatory_priority);
  // Enforces a mandatory HLO structural layout on instruction output buffers or
  // operand_no, elevating priority per leaf or operand to at least any existing
  // mandatory constraint priority.
  absl::Status EnforceInstructionLayout(
      const Shape& shape_with_layout, const HloInstruction* instruction,
      bool dfs = true, bool allow_alias = false,
      std::optional<int64_t> min_priority = std::nullopt) {
    return SetInstructionLayout(shape_with_layout, instruction,
                                /*mandatory=*/true, dfs, allow_alias,
                                min_priority.value_or(current_priority_),
                                /*elevate_mandatory_priority=*/true);
  }
  absl::Status EnforceOperandLayout(
      const Shape& shape_with_layout, const HloInstruction* instruction,
      int64_t operand_no, bool dfs = true,
      std::optional<int64_t> min_priority = std::nullopt) {
    return SetOperandLayout(shape_with_layout, instruction, operand_no,
                            /*mandatory=*/true, dfs,
                            min_priority.value_or(current_priority_),
                            /*elevate_mandatory_priority=*/true);
  }
  int64_t MaxBufferPriorityAt(const HloInstruction* instruction) const;
  absl::Status SetSubcomputationOperandLayout(ShapeLayout param_layout,
                                              const HloInstruction* instruction,
                                              int64_t operand_no,
                                              int64_t min_priority);
  absl::Status ConstrainOrOverrideShapeBuffers(
      const HloInstruction* instruction, const Shape& target_shape,
      int64_t priority, const HloInstruction* enforcing_user,
      absl::flat_hash_map<HloValue::Id, int64_t>* override_counts = nullptr,
      bool* changed = nullptr,
      const HloComputation* local_root_computation = nullptr);

  // Sets the computation result layout based on constraints and
  // sub-computations.
  absl::Status AddComputationResultLayoutConstraints(
      LayoutConstraints* constraints);

  // Constrains layouts for custom calls that have specific layout requirements.
  absl::Status AddCustomCallConstraints(LayoutConstraints* constraints);

  // Initializes unconstrained_buffer_ids_ with all array HloValues in the
  // given span of computations.
  void InitUnconstrainedBuffers(absl::Span<HloComputation* const> computations);

  // HloDataflowPropagation overrides for cross computation layout propagation.
  bool HasValueAt(const HloInstruction* instruction,
                  const ShapeIndex& index) const override;
  absl::Status PropagateAcrossEdge(const HloInstruction* src_instruction,
                                   const ShapeIndex& src_index,
                                   const HloInstruction* dst_instruction,
                                   const ShapeIndex& dst_index,
                                   bool allow_override, bool* changed) override;
  const HloComputation* PreferredConditionalBranch(
      const HloInstruction* conditional) const override;
  void ResetPropagationState() override;
  absl::Status FlushComputationPropagation() override;

  // Constrains unconstrained array HloValues at `(instruction, index)`, or
  // overrides conflicting constraints when `override_counts` is provided.
  absl::Status ConstrainOrOverrideBuffersAtIndex(
      const HloInstruction* instruction, const ShapeIndex& index,
      const Layout& layout, bool dfs, int64_t priority,
      absl::flat_hash_map<HloValue::Id, int64_t>* override_counts = nullptr,
      bool* changed = nullptr, const HloInstruction* enforcing_user = nullptr);

  // Iteratively assigns layouts to remaining unconstrained buffers and
  // propagates until all buffers are constrained.
  absl::Status AssignLayoutsToUnconstrainedBuffers(
      LayoutConstraints* constraints);

  // Return a vector containing the constraints which have been added to the
  // LayoutConstraints object since the construction of the object or since the
  // last time ConsumeAddedConstraints() has been called. This is used to
  // identify newly added constraints when propagating layouts.
  std::vector<const LayoutConstraint*> ConsumeAddedConstraints() {
    std::vector<const LayoutConstraint*> ret_vec(std::move(added_constraints_));
    added_constraints_.clear();
    added_constraints_set_.clear();
    return ret_vec;
  }
  void ClearAddedConstraints() {
    added_constraints_.clear();
    added_constraints_set_.clear();
  }

  // This method can be overridden to add backend-specific constraints to the
  // layout of the instructions of a computation. This method is called after
  // all mandatory constraints have been added via AddMandatoryConstraints
  // and before propagating constraints.
  virtual absl::Status AddBackendConstraints(LayoutConstraints* constraints) {
    return absl::OkStatus();
  }

  // Assign layouts to the instructions of a computation which satisfy the given
  // layout constraints. Copies may be added to satisfy the constraints. The
  // given LayoutConstraints must have layout constraints every logical buffer
  // in the computation.
  absl::Status AssignLayouts(LayoutConstraints& constraints);

  // Propagates layout constraints from a set of initial constraints in order to
  // minimize the local cost of the computation. This propagation is *not*
  // required for correctness.
  absl::Status PropagateConstraints(LayoutConstraints* constraints);

  absl::Status PropagateBufferConstraintToOperands(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);

  // Check that all layouts in the module have been set and satisfy all
  // necessary conditions.
  absl::Status CheckLayouts(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Computes the ComputationLayout of the given constraints based on the
  // layouts assigned to parameters and root instruction.
  absl::Status CalculateComputationLayout(LayoutConstraints* constraints);

  // Propagates the layouts computed by the layout assignment pass on the given
  // computation, to the computation layout passed in to this API.
  // This API propagates missing layout, and also checks that the caller
  // specified have been respected, by comparing those with the parameters and
  // root computation instruction.
  absl::Status PropagateComputationLayouts(
      HloComputation* computation, ComputationLayout* computation_layout);

  // The pointer to the ComputationLayout passed as constructor parameter.
  ComputationLayout* entry_computation_layout_;

  // A copy of entry_computation_layout_ used to check which parameter and
  // result layouts were explicitly specified prior to running the pass.
  ComputationLayout saved_entry_computation_layout_;

 protected:
  static constexpr int64_t kSubcomputationPriorityBoost = 2;
  // Sets up the copy instruction according to the characteristic (sharding,
  // metadata, ...) of the reference instruction. The index argument is used
  // when the instruction is a tuple, and in such case the index represents
  // the location from where the copy instruction was created from.
  // If the index is empty, the whole sharding will be propagated, even in case
  // the instruction has a tuple sharding.
  static void SetupCopiedInstruction(const HloInstruction& instruction,
                                     HloInstruction* copy,
                                     const ShapeIndex& index);

  // Creates and returns a copy of the given instruction with a different
  // layout. Tuple-shaped instructions will be deep-copied, and the last Tuple
  // instruction producing the copy is returned.
  absl::StatusOr<HloInstruction*> CreateCopyWithNewLayout(
      const Shape& shape_with_layout, HloInstruction* instruction);

  // Creates a copy of the given operand if the operand's layout does not match
  // the given layout. This copy replaces the use in the given instruction.
  // Tuple operands will be deep-copied.
  virtual absl::Status CopyOperandIfLayoutsDiffer(
      const ShapeLayout& operand_layout, HloInstruction* instruction,
      int64_t operand_no);

  // Adds a copy for the operand of an instruction, unless such operand is
  // already a copy, and has a single user (which is forcibly the instruction
  // itself).
  absl::Status AddCopyForOperand(HloInstruction* instruction,
                                 int64_t operand_number);

  // Replaces dataflow_analysis_ and drops the buffer sets memoized from the
  // previous analysis.
  void SetDataflowAnalysis(std::unique_ptr<HloDataflowAnalysis> analysis);

  // Module dataflow analysis that can be updated for cloned computations.
  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;

  bool IsComputationIncluded(const HloComputation* computation) const override;

  // Returns the BufferLayoutConstraint at `(instruction, index)` if all
  // HloValues in its value set are constrained to the same minor to major
  // layout matching the subshape rank, or nullptr otherwise.
  const BufferLayoutConstraint* GetConstrainedLayoutAtIndex(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Returns all array shaped uses (instruction and operand number) of `buffer`
  // or its aliases.
  static std::vector<std::pair<const HloInstruction*, int64_t>>
  GetArrayUsesOfBuffer(const HloValue& buffer);

  std::string ToString(const LayoutConstraints& constraints) const;

 private:
  // Returns true if `computation` has an entry in `computation_layouts_` whose
  // ComputationLayout has been explicitly constrained or calculated.
  bool HasConstrainedComputationLayout(const HloComputation* computation) const;

  // Entry computation of the module currently being processed.
  HloComputation* entry_computation_ = nullptr;
  // Map containing the layouts of all computations assigned so
  // far. Computations are handled in a topological sort where computations are
  // handled before their caller instructions so the layouts of caller
  // instructions can be set to match the computation.
  absl::flat_hash_map<const HloComputation*, std::unique_ptr<LayoutConstraints>>
      computation_layouts_;

  // Subcomputations whose result layout was pinned externally by their caller.
  absl::flat_hash_set<const HloComputation*> pinned_result_computations_;

  // Array-shaped buffers which have not yet been constrained, in id order.
  absl::btree_set<HloValue::Id> unconstrained_buffer_ids_;

  // Buffer sets of dataflow_analysis_ memoized by GetBufferSet. The entries
  // point at buffers owned by the analysis, so the cache is only valid for the
  // analysis it was filled from: SetDataflowAnalysis, the only place the
  // analysis changes, clears it, and GetBufferSet checks that pairing.
  mutable absl::flat_hash_map<const HloInstruction*, std::unique_ptr<BufferSet>>
      buffer_sets_cache_;
  const HloDataflowAnalysis* buffer_sets_cache_analysis_ = nullptr;

  // Buffer layout constraints stored densely by HloValue::Id, which the
  // dataflow analysis assigns sequentially. Clear() invalidates every entry
  // in O(1) by bumping a generation, so no table is destroyed and rebuilt per
  // propagation round.
  class BufferConstraintTable {
   public:
    // The live constraint of `id`, or nullptr.
    BufferLayoutConstraint* Find(HloValue::Id id) const {
      const size_t slot = static_cast<size_t>(id);
      return slot < constraints_.size() && generations_[slot] == generation_
                 ? constraints_[slot].get()
                 : nullptr;
    }

    // Makes `constraint` the live constraint of `id`.
    BufferLayoutConstraint* Insert(
        HloValue::Id id, std::unique_ptr<BufferLayoutConstraint> constraint) {
      const size_t slot = static_cast<size_t>(id);
      if (slot >= constraints_.size()) {
        constraints_.resize(slot + 1);
        generations_.resize(slot + 1, 0);
      }
      generations_[slot] = generation_;
      constraints_[slot] = std::move(constraint);
      return constraints_[slot].get();
    }

    void Clear() { ++generation_; }

   private:
    std::vector<std::unique_ptr<BufferLayoutConstraint>> constraints_;
    std::vector<uint32_t> generations_;
    uint32_t generation_ = 1;
  };
  BufferConstraintTable buffer_constraints_;

  // A vector which holds constraints as they are added. Can be cleared with
  // ClearAddedConstraints.
  std::vector<const LayoutConstraint*> added_constraints_;
  absl::flat_hash_set<const LayoutConstraint*> added_constraints_set_;
  int64_t current_priority_ = LayoutConstraint::kBeginningPriority;

  // Stores the set of while computations that have copy disabled.
  absl::flat_hash_set<const HloComputation*> copy_disabled_while_computations_;

  // When true, PropagateConstraints also propagates newly constrained buffers
  // across computation boundaries (kCall, kWhile, kConditional, kAsyncStart)
  // into unconstrained HloValues.
  bool propagate_cross_computation_constraints_ = false;

  // Per-buffer override counts used during HloDataflowPropagation::Run.
  absl::flat_hash_map<HloValue::Id, int64_t> cross_comp_override_counts_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
