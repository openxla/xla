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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/cpu_float_support.h"

#include "xla/service/cpu/onednn_matmul_rewriter.h"
#include "xla/service/cpu/onednn_memory_util.h"

namespace xla {
namespace cpu {

bool CpuFloatSupport::IsSupported(const HloInstruction& hlo) const {
  switch (hlo.opcode()) {
    // Collective ops.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kDot:
      return (LowPrecisionType() == BF16 || LowPrecisionType() == F16)
                                      && DotSupported(hlo);
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
    case HloOpcode::kSelect:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    // Other special ops.
    case HloOpcode::kBitcast:
      return true;
    default:
      return false;
  }
}

bool CpuFloatSupport::DotSupported(const HloInstruction& hlo) const {
  const Shape& lhs_shape = hlo.operand(0)->shape();
  const Shape& rhs_shape = hlo.operand(1)->shape();
  const Shape& output_shape = hlo.shape();
  // OneDNN only supports 2 <= rank <= kOneDnnMaxNDims.
  if (lhs_shape.rank() != rhs_shape.rank() ||
      rhs_shape.rank() != output_shape.rank() || lhs_shape.rank() < 2 ||
      lhs_shape.rank() > kOneDnnMaxNDims) {
    return false;
  }

  auto rank = lhs_shape.rank();
  auto rhs_dims = rhs_shape.dimensions();
  int64_t num_mac_ops = ShapeUtil::ElementsIn(lhs_shape) * rhs_dims.back();
  int mac_ops_threshold = (rank == 2) ? (1 << 23) : (1 << 18);
  return (num_mac_ops >= mac_ops_threshold);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
