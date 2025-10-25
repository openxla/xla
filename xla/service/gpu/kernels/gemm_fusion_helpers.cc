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

#include "xla/service/gpu/kernels/gemm_fusion_helpers.h"

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::Status MatchRowMajorGemm(HloDotInstruction* dot) {
  if (dot->operand(0)->shape().dimensions().size() != 2 ||
      dot->operand(1)->shape().dimensions().size() != 2) {
    return absl::InternalError("operands must have rank 2");
  }

  if (dot->shape().layout().minor_to_major().back() != 0) {
    return absl::InternalError("The dot result must have row major layout.");
  }

  auto& dot_dims = dot->dot_dimension_numbers();
  if (dot_dims.lhs_contracting_dimensions().size() != 1) {
    return absl::InternalError("Lhs contracting dimensions must be of size 1.");
  }

  if (dot_dims.rhs_contracting_dimensions().size() != 1) {
    return absl::InternalError("Rhs contracting dimensions must be of size 1.");
  }

  if (dot->operand(0)->shape().layout().minor_to_major(0) !=
      dot_dims.lhs_contracting_dimensions()[0]) {
    return absl::InternalError(
        "Lhs contracting dimension should be along the minor axis (elements "
        "that are stored contiguous in memory).");
  }

  if (dot->operand(1)->shape().layout().minor_to_major(1) !=
      dot_dims.rhs_contracting_dimensions()[0]) {
    return absl::InternalError(
        "Rhs contracting dimension should be along the major axis (elements "
        "that are NOT stored contiguous in memory).");
  }

  return absl::OkStatus();
}

absl::Status MatchSimpleGemm(HloDotInstruction* dot,
                             absl::Span<const PrimitiveType> supported_dtypes) {
  TF_RETURN_IF_ERROR(MatchRowMajorGemm(dot));

  for (PrimitiveType dtype : supported_dtypes) {
    if (dot->operand(0)->shape().element_type() == dtype &&
        dot->operand(1)->shape().element_type() == dtype &&
        dot->shape().element_type() == dtype) {
      return absl::OkStatus();
    }
  }

  return absl::InternalError("unsupported operands type");
}

}  // namespace xla::gpu
