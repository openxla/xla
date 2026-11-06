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

#ifndef XLA_SERVICE_GPU_BLAS_LT_CUSTOM_CALLS_H_
#define XLA_SERVICE_GPU_BLAS_LT_CUSTOM_CALLS_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Platform-neutral view of the BlasLt matrix multiplication custom calls.
//
// cuBLASLt (cublas_cudnn.h) and hipBLASLt (hipblaslt_custom_calls.h) use
// separate custom call targets so that the two backends can evolve
// independently. A single XLA binary serves both, however, so compiler
// infrastructure that is not backend-specific -- the scheduler, cost models,
// layout assignment, thunk emission -- must recognize either spelling. The
// predicates below are that union; prefer them over the backend-specific ones
// unless the code really is CUDA-only or ROCm-only.
//
// The CallTarget() helpers select the spelling to *emit* for a given device.

// Matrix multiplication that calls into cuBLASLt or hipBLASLt.
bool IsBlasLtMatmul(const HloInstruction& hlo);

// Returns true if hlo is a non-fused BlasLt matmul (default epilogue, beta=0).
bool IsNonFusedBlasLtMatmul(const HloInstruction& hlo);

// Scaled matrix multiplication in FP8.
bool IsBlasLtMatmulF8(const HloInstruction& hlo);

// Block-scaled matrix multiplication in MX formats.
bool IsBlasLtMatmulMx(const HloInstruction& hlo);

// Grouped matrix multiplication.
bool IsBlasLtGroupedMatmul(const HloInstruction& hlo);

// Any matrix multiplication rewritten into a BlasLt custom call.
bool IsBlasLtGemm(const HloInstruction& hlo);

// Any matrix multiplication rewritten into a BLAS custom call, including the
// legacy cuBLAS one. Prefer IsBlasLtGemm unless you specifically need to cover
// the legacy target too.
bool IsBlasGemm(const HloInstruction& hlo);

// Returns the custom call target that `gpu_version` should emit for each kind
// of BlasLt matmul: the hipBLASLt spelling on ROCm, the cuBLASLt spelling
// otherwise. SYCL shares the cuBLASLt targets today.
//
// There is no selector for the MX target: block-scaled MX matmuls are only
// produced on ROCm, so emitters use kHipblasLtMatmulMxCallTarget directly.
absl::string_view BlasLtMatmulCallTarget(
    const stream_executor::GpuComputeCapability& gpu_version);
absl::string_view BlasLtMatmulF8CallTarget(
    const stream_executor::GpuComputeCapability& gpu_version);
absl::string_view BlasLtGroupedMatmulCallTarget(
    const stream_executor::GpuComputeCapability& gpu_version);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_BLAS_LT_CUSTOM_CALLS_H_
