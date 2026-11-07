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

#ifndef XLA_SERVICE_GPU_HIPBLASLT_CUSTOM_CALLS_H_
#define XLA_SERVICE_GPU_HIPBLASLT_CUSTOM_CALLS_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

// Custom call targets for matrix multiplications that call into hipBLASLt on
// the ROCm backend.
//
// These deliberately do not share a spelling with the cuBLASLt targets in
// cublas_cudnn.h: the two backends have independent library dependencies and
// must be free to evolve separately. GemmRewriter emits these on ROCm, and
// ThunkEmitter lowers them to a CublasLtMatmulThunk, which dispatches through
// the platform-neutral se::gpu::BlasLt interface.
//
// Code that needs to recognize a BlasLt matmul regardless of backend should
// use the union predicates in blas_lt_custom_calls.h rather than these.

// A call to hipBLASLt for matrix multiplication.
inline constexpr absl::string_view kHipblasLtMatmulCallTarget =
    "__hipblaslt$matmul";

// A call to hipBLASLt for scaled matrix multiplication in FP8.
inline constexpr absl::string_view kHipblasLtMatmulF8CallTarget =
    "__hipblaslt$matmul$f8";

// A call to hipBLASLt for block-scaled matrix multiplication in MX formats.
inline constexpr absl::string_view kHipblasLtMatmulMxCallTarget =
    "__hipblaslt$matmul$mx";

// A call to the hipBLASLt extension API for grouped matrix multiplication.
inline constexpr absl::string_view kHipblasLtGroupedMatmulCallTarget =
    "__hipblaslt$groupedMatmul";

// Matrix multiplication that calls into hipBLASLt.
bool IsHipblasLtMatmul(const HloInstruction& hlo);

// Scaled matrix multiplication in FP8. Calls into hipBLASLt.
bool IsHipblasLtMatmulF8(const HloInstruction& hlo);

// Block-scaled matrix multiplication in MX formats. Calls into hipBLASLt.
bool IsHipblasLtMatmulMx(const HloInstruction& hlo);

// Grouped matrix multiplication that calls into the hipBLASLt extension API.
bool IsHipblasLtGroupedMatmul(const HloInstruction& hlo);

// Any matrix multiplication rewritten into a hipBLASLt custom call.
bool IsHipblasLtGemm(const HloInstruction& hlo);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HIPBLASLT_CUSTOM_CALLS_H_
