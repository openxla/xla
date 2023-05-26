/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gpu_shape_verifier.h"

namespace xla {

Status GpuShapeVerifier::Preprocess(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(ShapeVerifier::Preprocess(hlo));

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo->shape(), [&](const Shape& shape, const ShapeIndex&) {
        if (shape.has_layout()) {
          if (LayoutUtil::IsSparseArray(shape)) {
            return InvalidArgument(
                "The XLA GPU backend does not support sparse shapes: %s",
                hlo->ToString());
          }
        }
        return OkStatus();
      }));

  return ShapeVerifier::Preprocess(hlo);
}

Status GpuShapeVerifier::HandleFusion(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(ShapeVerifier::HandleFusion(hlo));

  // Elements returned by multi-output kLoop fusions must all have the same
  // shape (ignoring element type).
  if (hlo->IsLoopFusion() && hlo->shape().IsTuple() &&
      !ShapeUtil::IsEmptyTuple(hlo->shape())) {
    const Shape& first_shape = hlo->shape().tuple_shapes(0);
    if (!absl::c_all_of(hlo->shape().tuple_shapes(), [&](const Shape& shape) {
          return ShapesSameIgnoringElementType(first_shape, shape);
        })) {
      return InternalError(
          "In a kLoop multi-output fusion, all outputs must have the same "
          "shape (ignoring element type).  Got %s",
          StringifyShape(hlo->shape()));
    }
  }
  return OkStatus();
}

// TODO(jlebar): Add additional checks here.  In particular, add checks for
// cudnn/cublas custom-calls and Triton fusions.

}  // namespace xla
