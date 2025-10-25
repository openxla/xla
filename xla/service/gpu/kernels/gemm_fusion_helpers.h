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

#ifndef XLA_SERVICE_GPU_KERNELS_GEMM_FUSION_HELPERS_H_
#define XLA_SERVICE_GPU_KERNELS_GEMM_FUSION_HELPERS_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns OK if dot instruction is a simple 2D row-major gemm.
absl::Status MatchRowMajorGemm(HloDotInstruction* dot);

// Returns OK if dot instruction is a simple gemm with all operands and result
// having one of the supported data types.
absl::Status MatchSimpleGemm(HloDotInstruction* dot,
                             absl::Span<const PrimitiveType> supported_dtypes);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNELS_GEMM_FUSION_HELPERS_H_
