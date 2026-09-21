/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/custom_fusion_configs.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/cpu_fusion_cost_model.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {

class YnnMatcher : public LibraryMatcher {
 public:
  explicit YnnMatcher(const TargetMachineFeatures* target_machine_features,
                      const tsl::protobuf::RepeatedField<int>* fusion_types)
      : LibraryMatcher(target_machine_features, fusion_types) {}
  ~YnnMatcher() override = default;

  // Returns the set of supported HLO instructions.
  absl::flat_hash_set<HloOpcode> SupportedOps() const override {
    static const absl::NoDestructor<absl::flat_hash_set<HloOpcode>>
        kSupportedOps{[]() {
          absl::flat_hash_set<HloOpcode> supported_ops{
              HloOpcode::kDot,          HloOpcode::kReduce,
              HloOpcode::kReduceWindow, HloOpcode::kConstant,
              HloOpcode::kConvolution,  HloOpcode::kReshape,
              HloOpcode::kBitcast,      HloOpcode::kBroadcast,
              HloOpcode::kTranspose,    HloOpcode::kPad,
              HloOpcode::kIota};
          for (const auto& [op, _] : GetYnnUnaryOpMap()) {
            supported_ops.insert(op);
          }
          for (const auto& [op, _] : GetYnnBinaryOpMap()) {
            supported_ops.insert(op);
          }
          return supported_ops;
        }()};
    return *kSupportedOps;
  }

  // Returns true if the HLO instruction is supported by the library.
  absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) override {
    if (instr->IsConstant()) {
      return IsConstantSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kIota) {
      return IsIotaSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kReshape) {
      return IsReshapeOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kBitcast) {
      return IsBitcastOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kBroadcast) {
      return IsBroadcastOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kTranspose) {
      return IsTransposeOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kPad) {
      return IsPadOpSupportedByYnn(instr);
    }
    if (!IsInstructionPreferredByYnn(instr)) {
      // TODO: It might make sense sometimes that even though an instruction is
      // not preferred by YNNPACK, that we should still fuse it, if it lies
      // between two other fusions that are preferred. While it is currently
      // sometimes an advantage, it is also sometimes a regression, so for now,
      // we require every instruction to be preferred by YNNPACK.
      return false;
    }
    if (instr->opcode() == HloOpcode::kDot) {
      return IsDotSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kReduce ||
        instr->opcode() == HloOpcode::kReduceWindow) {
      return IsReduceLikeOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kConvolution) {
      return IsConvolutionOpSupportedByYnn(instr);
    }
    if (instr->IsElementwise()) {
      return IsElementwiseOpSupportedByYnn(instr);
    }
    return false;
  }

  // Returns true if the chain feeding `reduce` contains a large intermediate
  // that XLA's own loop fusion could keep out of memory.
  //
  // LibraryRewriter runs before CpuInstructionFusion. Once a reduction is
  // wrapped in a library fusion, CpuInstructionFusion treats it as opaque
  // ("Don't fuse instructions from custom fusions/calls") and every producer
  // feeding it must be materialized. For a reduction over a real buffer that
  // is a good trade: the library kernel beats our emitted loop and the buffer
  // exists either way. For a reduction over a computed intermediate it is a
  // bad one -- we pay to write and re-read a buffer that loop fusion would
  // have kept in registers, and that buffer is O(input) while the result is
  // only O(output).
  //
  // The walk looks through reduction-like and shape-only producers, because a
  // large reduction may already have been split into a reduce-window plus a
  // final reduce by the time we run: the intermediate worth saving then sits
  // above the reduce-window, not directly under the reduce.
  static bool ReductionInputIsFusibleIntermediate(
      const HloInstruction* reduce) {
    // Bound the walk so this stays cheap on deep graphs.
    static constexpr int kMaxWalkDepth = 8;

    if (!reduce->shape().IsArray()) {
      return false;
    }

    int64_t largest_intermediate = 0;
    const HloInstruction* largest_intermediate_instr = nullptr;
    const HloInstruction* instr = reduce;
    for (int depth = 0; depth < kMaxWalkDepth; ++depth) {
      if (instr->operand_count() == 0) {
        break;
      }
      const HloInstruction* input = instr->operand(0);
      if (!input->shape().IsArray()) {
        break;
      }
      // Parameters and constants are real buffers that exist regardless, so
      // there is nothing to save by declining the library fusion.
      if (input->opcode() == HloOpcode::kParameter ||
          input->opcode() == HloOpcode::kConstant) {
        break;
      }
      bool can_look_through = input->IsElementwise();
      switch (input->opcode()) {
        case HloOpcode::kBitcast:
        case HloOpcode::kBroadcast:
        case HloOpcode::kConcatenate:
        case HloOpcode::kReduce:
        case HloOpcode::kReduceWindow:
        case HloOpcode::kReshape:
        case HloOpcode::kReverse:
        case HloOpcode::kSlice:
        case HloOpcode::kTranspose:
          can_look_through = true;
          break;
        default:
          break;
      }
      if (!can_look_through) {
        break;
      }
      const int64_t input_bytes = ShapeUtil::ByteSizeOfElements(input->shape());
      if (input_bytes > largest_intermediate) {
        largest_intermediate = input_bytes;
        largest_intermediate_instr = input;
      }
      instr = input;
    }

    if (largest_intermediate_instr == nullptr) {
      return false;
    }
    // Ask the same question CpuInstructionFusion will ask: is recomputing
    // this intermediate inside the reduction loop cheaper than writing it out
    // and reading it back? If so, declining the library fusion lets loop
    // fusion keep it in registers. If not -- a small buffer, or one whose
    // arithmetic is too expensive to redo -- the library kernel is worth more
    // than the round trip it would avoid.
    //
    // A fresh model per call: its caches are keyed on instruction pointers,
    // and this runs before fusion has begun rewriting the graph.
    CpuFusionCostModel cost_model{CpuFusionCostModel::Params{}};
    return cost_model.RecomputeBeatsMaterialize(largest_intermediate_instr);
  }

  // Returns true if we should start a new fusion containing just the given HLO
  // instruction. We control the instructions that can start a fusion with the
  // `--xla_cpu_experimental_ynn_fusion_type` flag.
  bool ShouldCreateFusion(const HloInstruction* instr) override {
    if (!IsInstructionPreferredByYnn(instr)) {
      return false;
    }
    if (fuse_dot_ && instr->opcode() == HloOpcode::kDot) {
      return true;
    }
    if (fuse_conv_ && instr->opcode() == HloOpcode::kConvolution) {
      return true;
    }
    if (fuse_reduce_ && (instr->opcode() == HloOpcode::kReduce ||
                         instr->opcode() == HloOpcode::kReduceWindow)) {
      // Leave reductions over fusible intermediates to CpuInstructionFusion,
      // which can fold the producer chain into the reduction loop instead of
      // forcing it through memory.
      return !ReductionInputIsFusibleIntermediate(instr);
    }
    return fuse_eltwise_ && instr->IsElementwise();
  }

  // Returns a prefix string for the fusion op's name.
  std::string fusion_prefix() const override { return "ynn_"; }

  // Returns a string for FusionBackendConfig's fusion kind.
  absl::string_view fusion_kind() const override { return kYnnFusionKind; }

 private:
  absl::flat_hash_set<DebugOptions::LibraryFusionType> fusion_types_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_
