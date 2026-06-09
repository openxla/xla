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

#include "xla/backends/gpu/transforms/triton_scaled_dot_support.h"

#include <cstdint>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

bool IsTritonSupportedScaledDot(const HloScaledDotInstruction& dot,
                                const se::CudaComputeCapability& cc) {
  PrimitiveType lhs = dot.operand(0)->shape().element_type();
  PrimitiveType rhs = dot.operand(1)->shape().element_type();
  const bool lhs_fp4 = (lhs == F4E2M1FN);
  const bool rhs_fp4 = (rhs == F4E2M1FN);

  // F4E2M1FN operands require Triton's fp4-specific packing, layout, and
  // target checks.
  if (lhs_fp4 || rhs_fp4) {
    // Triton's verifier derives the contracting dimension size from each
    // operand's fp4 packing. Mixing fp4 and non-fp4 makes the two sides
    // disagree.
    if (!lhs_fp4 || !rhs_fp4) {
      return false;
    }

    if (!cc.IsAtLeastHopper()) {
      return false;
    }

    // XLA emits fp4 pack flags from the operand minor dimension.
    const int64_t lhs_c =
        dot.dot_dimension_numbers().lhs_contracting_dimensions(0);
    const int64_t rhs_c =
        dot.dot_dimension_numbers().rhs_contracting_dimensions(0);
    const auto& lhs_layout = dot.operand(0)->shape().layout();
    const auto& rhs_layout = dot.operand(1)->shape().layout();
    const bool lhs_k_pack = lhs_layout.minor_to_major(0) == lhs_c;
    const bool rhs_k_pack = rhs_layout.minor_to_major(0) == rhs_c;
    PrimitiveType lhs_scale = dot.operand(2)->shape().element_type();
    PrimitiveType rhs_scale = dot.operand(3)->shape().element_type();

    // Triton's fp4 scaled-MMA path expects both scales to use the same
    // encoding.
    if (lhs_scale != rhs_scale) {
      return false;
    }

    // F4E2M1FN operands use the same payload for MXFP4 and NVFP4; the scale
    // element type selects the NVFP4 lowering.
    const bool nvfp4 = lhs_scale == F8E4M3FN;
    if (nvfp4) {
      // Triton's NVFP4 MMA lowers to mxf4nvf4, which requires both operands to
      // be K-packed so the lowering does not transpose either operand.
      return cc.IsAtLeastBlackwell() && lhs_k_pack && rhs_k_pack;
    }

    // On tcgen05 targets, the MXFP4 path accepts XLA's canonical operand
    // layout: lhs is K-packed and rhs is N-packed. Other Hopper-or-newer
    // targets can unpack fp4 operands before using the scaled dot.
    if (cc.HasTcgen05() && (!lhs_k_pack || rhs_k_pack)) {
      return false;
    }
  }

  return true;
}
}  // namespace gpu
}  // namespace xla
