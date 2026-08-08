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

#ifndef XLA_HLO_TRANSFORMS_BFLOAT16_PROPAGATION_H_
#define XLA_HLO_TRANSFORMS_BFLOAT16_PROPAGATION_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// HLO pass which reduces the precision of some HLO instructions to BF16
// according to the backend-specific FloatSupport rule provided by the
// caller.
//
// This pass can be used to reduce instruction precision without affecting the
// numerical accuracy of the module, i.e., the final output of the module would
// be bitwise identical to that without this pass; this is possible if the
// backend already reduces precision to BF16 on some HLO instructions.
//
// This pass will not modify the signature of a computation, unless it is a
// fusion computation or its only caller is a while.
//
// !!! WARNING !!! This pass can introduce mixed precision in individual HLOs,
// which has two issues:
//
// 1) It does not guarantee to respect the passed-in FloatSupport
// specification in terms of mixed precision, so the backend may not support an
// HLO that has mixed precision produced by this pass. To address this issue,
// run FloatNormalization with the same FloatSupport after this pass.
//
// 2) In general, mixed precision may break the assumptions of some other HLO
// passes even if the specific backend supports the individual HLOs. Such
// assumptions include that there are no HLOs using mixed precision, or that the
// precision of an HLO's output is determined by its inputs. It should be used
// at the end of the HLO optimization pipeline but before
// BFloat16ConversionFolding. If other passes are needed after this pass, run
// BFloat16MixedPrecisionRemoval first to undo some of the changes made by this
// pass.
class BFloat16Propagation : public HloModulePass {
 public:
  BFloat16Propagation(const FloatSupport* bfloat16_support,
                      const AliasInfo* alias_info);

  ~BFloat16Propagation() override = default;

  static constexpr absl::string_view kName = "bfloat16-propagation";
  absl::string_view name() const override { return kName; }

  // Returns whether we should avoid changing the precision of inst regardless
  // of the producers and users.
  virtual bool ShouldKeepPrecisionUnchanged(const HloInstruction* inst);

  // Determines whether we should consider changing the precision of the given
  // instruction in the forward pass.
  virtual bool InstructionIsCandidateForBF16Output(HloInstruction* hlo);

 protected:
  const FloatSupport* bfloat16_support_;

  const AliasInfo* alias_info_;

  // Returns the original element type of the HLO instruction before
  // RunImpl starts mutating shapes in-place with changes_to_bf16_.
  PrimitiveType UnmutatedElementType(const HloInstruction* hlo) const {
    if (hlo->shape().element_type() == BF16 &&
        changes_to_bf16_.contains(const_cast<HloInstruction*>(hlo))) {
      return F32;
    }
    return hlo->shape().element_type();
  }

  // Runs the pass on the given module. Returns whether the module was changed
  // (precision reductions were added).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // ***************************
  // Function called and state produced by the forward analysis pass (from
  // parameters to root) that determines the candidate HLOs to use BF16 outputs.

  // The set of instructions to consider using bfloat16, computed in the forward
  // pass.
  absl::flat_hash_set<const HloInstruction*> consider_using_bfloat16_;

  // ***************************
  // Functions called and state produced by the backward pass (from root to
  // parameters) that finds opportunities to use BF16.

  // Determines the precision for the given instruction in the
  // opportunity-finding pass.
  void DetermineInstructionPrecision(HloInstruction* hlo, bool skip_parameters);

  // Special handling in the opportunity-finding pass for fusion computations.
  //
  // Precondition: hlo->opcode() == kFusion
  void DetermineFusionComputationPrecision(HloInstruction* fusion);

  // Reverts changes to BF16 that will not propagate outside a fusion
  // computation. This avoids BF16 casts overhead inside a fusion which won't
  // save memory bandwidth.
  //
  // Precondition: hlo->opcode() == kFusion
  void RevertIfFusionInternalBF16Changes(HloInstruction* fusion);

  // Special handling in the opportunity-finding pass for while computations.
  //
  // Precondition: hlo->opcode() == kWhile
  void DetermineWhileComputationsPrecision(HloInstruction* while_hlo);

  // Special handling in the opportunity-finding pass for associative scans.
  // Mirrors DetermineWhileComputationsPrecision because scan carries form a
  // loop-carried precision equivalence chain across the IR (carry init ->
  // body parameter -> body root -> result carry).
  //
  // Precondition: hlo->opcode() == kScan and the scan is_associative() ==
  // TRI_STATE_TRUE. Non-associative scans are expanded to while loops by the
  // ScanExpander pass and are handled by the kWhile path instead.
  void DetermineScanComputationPrecision(HloInstruction* scan_hlo);

  // Special handling in the opportunity-finding pass for conditional branches.
  //
  // Precondition: hlo->opcode() == kConditional
  void DetermineConditionalComputationsPrecision(HloInstruction* cond);

  // Special handling in the opportunity-finding pass for async computations.
  //
  // Precondition: hlo->opcode() == kAsyncStart
  void DetermineAsyncComputationsPrecision(HloInstruction* async_start);

  // Special handling in the opportunity-finding pass for called computations.
  //
  // Precondition: hlo->opcode() == kCall
  void DetermineCalledComputationsPrecision(HloInstruction* call);

  // The set of HloInstructions that have been visited in the
  // opportunity-finding pass.
  absl::flat_hash_set<const HloInstruction*>
      instructions_visited_in_backward_pass_;

  // The set of HloComputations that have been visited in the
  // opportunity-finding pass.
  absl::flat_hash_set<const HloComputation*>
      computations_visited_in_backward_pass_;

  // ***************************
  // Functions called by the final inconsistency resolving pass.

  // Adjusts the output shapes of HloInstructions such that if two
  // HloInstructions have aliasing buffers in their outputs, they must have the
  // same precision.
  void ResolveInconsistencyOfAliasingBuffers(HloModule* module);

  // Resolves inconsistency of aliasing buffers for the given computation, and
  // recursively runs on a while instruction's condition and body until a fixed
  // point is reached.
  bool ResolveInconsistencyOfAliasingBuffersHelper(
      HloComputation* computation,
      absl::flat_hash_set<const HloComputation*>* visited_computations);

  // Records that the given value must be kept in F32, bumping the state
  // versions if this is new information. All insertions into
  // values_that_must_be_kept_as_f32_ must go through this method so that the
  // state versions track every state change the resolving pass can observe.
  void KeepValueAsF32(const HloValue* value);

  // Registers an effective mutation of the state the resolving pass depends
  // on (changes_to_bf16_ or values_that_must_be_kept_as_f32_) affecting an
  // instruction or value in `computation`. Bumps the global state_version_
  // as well as the per-computation write version of `computation` and all
  // its transitive caller computations (a mutation inside a computation is
  // observable from every computation that directly or transitively calls
  // it, e.g. via called-computation parameter/root checks or forwarded
  // dataflow values).
  void BumpStateVersion(const HloComputation* computation);

  // Returns the list of computations whose write version must be bumped for
  // a mutation in `computation`: the computation itself plus all transitive
  // caller computations. Lazily computed; valid while the module structure
  // is unchanged.
  const std::vector<const HloComputation*>& GetVersionBumpList(
      const HloComputation* computation);

  // Returns true if `computation` has callers and all of them are fusion
  // instructions. Such computations are self-contained for the resolving
  // pass: fusion parameters define their own dataflow values and fused root
  // values do not escape to the caller, so all state read when resolving the
  // computation lives in the computation itself or its (fusion-called)
  // callees, and any mutation of that state bumps this computation's write
  // version via BumpStateVersion. Lazily computed.
  bool IsFusionOnlyComputation(const HloComputation* computation);

  // Returns the version of the state that resolving `computation` can
  // observe: the per-computation write version for fusion-only computations
  // (see IsFusionOnlyComputation), and the global state_version_ otherwise
  // (computations called from sequential context read caller/sibling state
  // through forwarded dataflow values, so only the global version is a sound
  // over-approximation for them).
  int64_t RelevantStateVersion(const HloComputation* computation);

  // Makes the parameters of called computations match how they are called by
  // the given HLO.
  void AdjustCalledComputationParameters(HloInstruction* hlo);

  // Makes the root instructions of called computations match how they are used
  // by the given HLO.
  void AdjustCalledComputationRoot(HloInstruction* hlo);

  // For an associative scan, aligns the precision of every carry slot across
  // its four logically-equivalent IR locations:
  //   * scan operand `num_inputs + i` (the carry init),
  //   * body parameter `num_inputs + i`,
  //   * body root tuple slot `num_outputs + i`,
  //   * scan result tuple slot `num_outputs + i`.
  // HloVerifier::HandleScan requires these four locations to share the same
  // element type even though, unlike kWhile.
  // If any of the four is F32 (after pending changes_to_bf16_),
  // all four are forced to F32. The standard
  // ResolveInconsistencyOfAliasingBuffersHelper fixed-point relies on
  // HloDataflowAnalysis aliases to propagate precision, but no such alias
  // exists for scan carries, so we enforce this verifier-level constraint
  // explicitly here. Returns whether any change was made.
  //
  // Precondition: hlo->opcode() == kScan and is_associative() ==
  // TRI_STATE_TRUE.
  bool AlignScanCarryPrecisions(HloInstruction* scan_hlo);

  // ***************************
  // Functions called after changes in changes_to_bf16_ are applied.

  // Resolves inconsistencies introduced by this pass for fusions with
  // tuple-type output.
  absl::Status ResolveInconsistentFusions(HloModule* module);

  // Resolves inconsistencies introduced by this pass for associative scans
  // where the body root tuple slot precision diverged from the scan output
  // slot precision (e.g. when the underlying body root op had multiple uses
  // with different precision demands and could not be unilaterally lowered).
  // Inserts precision-changing converts on the affected body root slots so
  // that the body root shape matches the scan output shape, satisfying
  // HloVerifier::HandleScan.
  absl::Status ResolveInconsistentScans(HloModule* module);

  // Converts the literals in kConstant HLOs which have their types changed to
  // BF16 by this pass.
  absl::Status ResolveConvertedConstants(HloModule* module);

  // Skips no-op conversions (same source and target shapes) that can be
  // produced this pass, i.e., replaces them in their uses with their operands.
  absl::Status SkipNoopConversions(HloModule* module);

  // ***************************
  // Functions called and state used by two or more passes.

  // Returns whether all uses of the given HloInstruction can consume BF16
  // input.
  bool AllUsersConsumeBF16(const HloInstruction& hlo,
                           const ShapeIndex& index) const;

  // Same as above, but takes the already-looked-up value set of (hlo, index)
  // to avoid a redundant dataflow lookup in hot loops.
  bool AllUsersConsumeBF16(const HloInstruction& hlo, const ShapeIndex& index,
                           const HloValueSet& value_set) const;

  // Memoized wrapper around the virtual ShouldKeepPrecisionUnchanged. The
  // predicate only depends on the instruction structure and the original
  // (unmutated) shapes, both of which are constant from the start of the
  // backward pass until changes_to_bf16_ is applied to the HLOs in RunImpl,
  // so within that window the result can be cached. The application loop in
  // RunImpl must keep calling the virtual method directly because it mutates
  // shapes as it goes.
  bool ShouldKeepPrecisionUnchangedCached(const HloInstruction* inst);

  // Memoized wrapper around AliasInfo::GetInPlaceInputOutputPairs, which only
  // depends on the instruction structure. The resolving pass queries it for
  // every (instruction, shape index) in every fixed-point sweep.
  const std::vector<std::pair<HloOperandIndex, ShapeIndex>>&
  GetInPlaceInputOutputPairsCached(const HloInstruction* hlo);

  // The output element type of the HLO at the given shape index after changes
  // in changes_to_bf16_ are applied.
  PrimitiveType OutputTypeAfterChange(HloInstruction* hlo,
                                      const ShapeIndex& index) const;

  // The element type of the HLO value after changes in changes_to_bf16_ are
  // applied.
  PrimitiveType ValueTypeAfterChange(const HloValue* value) const;

  // Builds value_type_after_change_, a by-value-id cache of
  // ValueTypeAfterChange kept up to date by AddToOrRemoveFromBF16ChangeSet.
  // Called at the start of the resolving pass; the cache makes the per-value
  // type lookups in the fixed-point sweeps O(1) array reads.
  void BuildValueTypeCache();

  // If target_type == BF16, adds the HLO at the given index to
  // changes_to_bf16_; otherwise, target_type must be F32 and this function
  // removes the HLO at the given index from changes_to_bf16_ if it was earlier
  // added.
  void AddToOrRemoveFromBF16ChangeSet(HloInstruction* hlo,
                                      const ShapeIndex& index,
                                      PrimitiveType target_type);

  // The set of F32 HLO values that must be kept in F32. Mutated only through
  // KeepValueAsF32 so that state_version_ tracks effective insertions.
  absl::flat_hash_set<const HloValue*> values_that_must_be_kept_as_f32_;

  // Global version counter for the state the resolving pass depends on. It
  // is bumped on every effective mutation of changes_to_bf16_ and
  // values_that_must_be_kept_as_f32_. Together with
  // resolve_completed_at_version_ below it allows
  // ResolveInconsistencyOfAliasingBuffersHelper to skip re-runs that are
  // provably no-ops.
  int64_t state_version_ = 0;

  // Per-computation write versions: how often state affecting the given
  // computation (or one of its transitive callees) has been mutated. See
  // BumpStateVersion and RelevantStateVersion.
  absl::flat_hash_map<const HloComputation*, int64_t> comp_write_version_;

  // Cache for GetVersionBumpList.
  absl::flat_hash_map<const HloComputation*, std::vector<const HloComputation*>>
      version_bump_lists_;

  // Cache for IsFusionOnlyComputation.
  absl::flat_hash_map<const HloComputation*, bool> fusion_only_computations_;

  // Maps a computation to the value of RelevantStateVersion(computation) at
  // which ResolveInconsistencyOfAliasingBuffersHelper last completed on it
  // with a final sweep that observed no state changes. Re-running the helper
  // on a computation while its relevant state version still has that value is
  // a no-op, so such runs are skipped. Without this, every fixed-point sweep
  // of a caller re-resolves all called computations (and the top-level walk
  // in ResolveInconsistencyOfAliasingBuffers resolves them yet again), which
  // multiplies across control-flow nesting levels and made this pass take
  // tens of seconds on large fusion-heavy modules.
  absl::flat_hash_map<const HloComputation*, int64_t>
      resolve_completed_at_version_;

  // Cache for ShouldKeepPrecisionUnchangedCached. Only valid while the module
  // is unmutated (i.e., until changes_to_bf16_ is applied in RunImpl).
  absl::flat_hash_map<const HloInstruction*, bool>
      keep_precision_unchanged_cache_;

  // Cache for GetInPlaceInputOutputPairsCached.
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<HloOperandIndex, ShapeIndex>>>
      inplace_input_output_pairs_cache_;

  // By-value-id cache of ValueTypeAfterChange, built by BuildValueTypeCache
  // for the resolving pass and kept in sync by
  // AddToOrRemoveFromBF16ChangeSet. Empty means "not built"; then
  // ValueTypeAfterChange computes the type directly.
  std::vector<PrimitiveType> value_type_after_change_;

  // Mapping from each HloComputation to the number of callers to it in the
  // module. Populated at the beginning of this pass.
  absl::flat_hash_map<const HloComputation*, int64_t> caller_counts_;

  // We first store the potential F32-to-BF16 changes to changes_to_bf16_, which
  // are subject to further adjustment, then finally applied to the HLOs. This
  // avoids setting changed_ to true but all changes are reverted during
  // adjustment.
  //
  // For each HloInstruction, changes_to_bf16_ stores the affected buffers in
  // the output as a map from in-place pointers to subshapes to shape indices.
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_map<Shape*, ShapeIndex>>
      changes_to_bf16_;

  // Whether the last processed HLO module has been changed by this pass.
  bool changed_ = false;

  std::unique_ptr<HloDataflowAnalysis> dataflow_;

  absl::flat_hash_set<absl::string_view> execution_threads_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_BFLOAT16_PROPAGATION_H_
