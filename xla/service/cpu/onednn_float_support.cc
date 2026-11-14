/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/cpu/onednn_float_support.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/onednn_contraction_rewriter.h"
#include "xla/service/float_support.h"

namespace xla {
namespace cpu {
namespace {

// Returns true if every reduce / reduce-window / map applying `hlo`'s
// computation also runs in low precision. A low precision computation applied
// by an upcast caller would leave converts in the caller's innermost loop.
// Other applying callers (e.g. scatter, sort) always support low precision.
bool IsAppliedInLowPrecision(const HloInstruction& hlo,
                             const FloatSupport& float_support) {
  if (hlo.parent() == nullptr) {
    return true;
  }
  for (const HloInstruction* caller : hlo.parent()->caller_instructions()) {
    if ((caller->opcode() == HloOpcode::kReduce ||
         caller->opcode() == HloOpcode::kReduceWindow ||
         caller->opcode() == HloOpcode::kMap) &&
        !float_support.SupportsLowPrecisionOutput(*caller)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool OneDnnFloatSupport::IsSupported(const HloInstruction& hlo) const {
  switch (hlo.opcode()) {
    // oneDNN rewritable ops
    case HloOpcode::kDot:
      return LowPrecisionType() == BF16 &&
             OneDnnContractionRewriter::ShouldRewriteDot(&hlo, true);
    case HloOpcode::kConvolution:
      return LowPrecisionType() == BF16 &&
             OneDnnContractionRewriter::ShouldRewriteConv(&hlo);
    // Collective ops.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kReduceScatter:
    // Data movement only ops.
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kCompare:
    case HloOpcode::kSelect:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    // Other special ops.
    case HloOpcode::kBitcast:
      return true;
    // Elementwise ops without precision loss.
    case HloOpcode::kAbs:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kClamp:
      return IsAppliedInLowPrecision(hlo, *this);
    case HloOpcode::kSort:
      return LowPrecisionType() == BF16;
    default:
      return false;
  }
}

}  // namespace cpu
}  // namespace xla
