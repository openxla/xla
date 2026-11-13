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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_CUSTOM_REDUCTION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_CUSTOM_REDUCTION_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// Lowers an HLO reduction to keep-dims cuDNN tensors, one per stream, with the
// data types of `inputs`. The caller reshapes each result to the HLO shape. A
// reduction that XLA cannot translate returns a status, so the non-GEMM
// autotuning backend can decline the candidate.
absl::StatusOr<
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>>
LowerCudnnCustomReduction(
    const HloInstruction& reduce, cudnn_frontend::graph::Graph& graph,
    const std::vector<
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_CUSTOM_REDUCTION_H_
