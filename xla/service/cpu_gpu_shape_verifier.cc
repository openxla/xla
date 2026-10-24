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

#include "xla/service/cpu_gpu_shape_verifier.h"

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

// ShapeUtil::ForEachSubshape* only descend into tuples, so this check walks
// buffer-wrapped and tuple-nested sub-shapes itself. The storage complex type
// check is applied at the leaves, where it reduces to an element type check.
absl::Status VerifyShapeHasNoUnsupportedStorageComplexTypes(
    const Shape& shape, absl::string_view backend_name) {
  if (shape.IsBuffer()) {
    return VerifyShapeHasNoUnsupportedStorageComplexTypes(shape.buffer_shape(),
                                                          backend_name);
  }
  if (shape.IsTuple()) {
    for (const Shape& subshape : shape.tuple_shapes()) {
      ABSL_RETURN_IF_ERROR(
          VerifyShapeHasNoUnsupportedStorageComplexTypes(subshape,
                                                         backend_name));
    }
    return absl::OkStatus();
  }
  if (!ShapeUtil::TouchesStorageComplexTypes(shape)) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(absl::StrFormat(
      "The XLA %s backend does not support element type %s yet.", backend_name,
      primitive_util::LowercasePrimitiveTypeName(shape.element_type())));
}

}  // namespace

absl::Status VerifyNoUnsupportedStorageComplexTypes(
    const HloModule& module, absl::string_view backend_name) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      ABSL_RETURN_IF_ERROR(
          VerifyShapeHasNoUnsupportedStorageComplexTypes(
              instruction->shape(), backend_name));
    }
  }
  return absl::OkStatus();
}

absl::Status CpuGpuShapeVerifier::Preprocess(HloInstruction* hlo) {
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo->shape(), [&](const Shape& shape, const ShapeIndex&) {
        if (shape.has_layout()) {
          if (!primitive_util::IsSubByteNonPredType(shape.element_type()) &&
              shape.layout().element_size_in_bits() != 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "The XLA CPU/GPU backend does not support custom element sizes "
                "on non-sub-byte types: %s",
                hlo->ToString()));
          }
        }
        return absl::OkStatus();
      }));

  return ShapeVerifier::Preprocess(hlo);
}

}  // namespace xla
