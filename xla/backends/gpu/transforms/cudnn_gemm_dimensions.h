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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_GEMM_DIMENSIONS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_GEMM_DIMENSIONS_H_

#include <cstdint>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// Describes the parameters and the output of a cuDNN GEMM fusion computation
// in terms of the dimensions of its dot (or scaled dot), in three steps:
//
// 1. Every instruction is related to the port of the dot that it contributes
//    to (one of the operands of the dot or its output) by tracking the shape
//    transformations between them (bitcasts, reshapes, transposes and
//    broadcasts) with a ShapeTracker. Copies and elementwise operations do not
//    change the logical shape.
// 2. The fusion is checked against the rules of cuDNN GEMM graphs.
// 3. The sizes and strides of every parameter and of the output are computed
//    from the physical layout of the tensor and its ShapeTracker.
class CuDnnGemmDimensions {
 public:
  // Analyzes the fusion computation that contains `dot` (a dot or a scaled
  // dot). Returns an error if the computation cannot be represented as a cuDNN
  // GEMM graph with strided rank-3 tensors.
  static absl::StatusOr<CuDnnGemmDimensions> Create(const HloInstruction& dot);

  CuDnnGemmDimensions(const CuDnnGemmDimensions&) = default;
  CuDnnGemmDimensions& operator=(const CuDnnGemmDimensions&) = default;
  CuDnnGemmDimensions(CuDnnGemmDimensions&&) = default;
  CuDnnGemmDimensions& operator=(CuDnnGemmDimensions&&) = default;

  // Returns the layout of a parameter or of the output of the computation in
  // the canonical rank-3 dimension order of cuDNN GEMMs, or nullptr if `hlo` is
  // neither. The order is [batch, M, K] for the left-hand side of the dot (and
  // its scale), [batch, K, N] for the right-hand side (and its scale) and
  // [batch, M, N] for the dot itself and everything computed from it.
  const CudnnTensorLayout* absl_nullable TensorLayout(
      const HloInstruction& hlo) const;

  // Returns the canonical sizes of the cuDNN tensor computed by `hlo`, which
  // are those of the dot operand that `hlo` contributes to, or of the dot, or
  // nullptr if `hlo` is not part of the cuDNN graph.
  const std::vector<int64_t>* absl_nullable CanonicalSizes(
      const HloInstruction& hlo) const;

 private:
  CuDnnGemmDimensions() = default;

  // Layouts of the parameters and of the output.
  absl::flat_hash_map<const HloInstruction*, CudnnTensorLayout> tensor_layouts_;
  // Canonical sizes of every instruction.
  absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>
      canonical_sizes_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_GEMM_DIMENSIONS_H_
