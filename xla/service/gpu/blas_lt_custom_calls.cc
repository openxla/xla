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

#include "xla/service/gpu/blas_lt_custom_calls.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hipblaslt_custom_calls.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

bool IsBlasLtMatmul(const HloInstruction& hlo) {
  return IsCublasLtMatmul(hlo) || IsHipblasLtMatmul(hlo);
}

bool IsNonFusedBlasLtMatmul(const HloInstruction& hlo) {
  if (!IsBlasLtMatmul(hlo)) {
    return false;
  }
  absl::StatusOr<GpuBackendConfig> gpu_config =
      hlo.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  return gpu_config->gemm_backend_config().epilogue() ==
         GemmBackendConfig::DEFAULT;
}

bool IsBlasLtMatmulF8(const HloInstruction& hlo) {
  return IsCublasLtMatmulF8(hlo) || IsHipblasLtMatmulF8(hlo);
}

bool IsBlasLtMatmulMx(const HloInstruction& hlo) {
  // MX block-scaled matmuls are only produced on ROCm.
  return IsHipblasLtMatmulMx(hlo);
}

bool IsBlasLtGroupedMatmul(const HloInstruction& hlo) {
  return IsCublasLtGroupedMatmul(hlo) || IsHipblasLtGroupedMatmul(hlo);
}

bool IsBlasLtGemm(const HloInstruction& hlo) {
  return IsCublasLtGemm(hlo) || IsHipblasLtGemm(hlo);
}

bool IsBlasGemm(const HloInstruction& hlo) {
  return IsLegacyCublasMatmul(hlo) || IsBlasLtGemm(hlo);
}

absl::string_view BlasLtMatmulCallTarget(
    const se::GpuComputeCapability& gpu_version) {
  return gpu_version.IsRocm() ? kHipblasLtMatmulCallTarget
                              : kCublasLtMatmulCallTarget;
}

absl::string_view BlasLtMatmulF8CallTarget(
    const se::GpuComputeCapability& gpu_version) {
  return gpu_version.IsRocm() ? kHipblasLtMatmulF8CallTarget
                              : kCublasLtMatmulF8CallTarget;
}

absl::string_view BlasLtGroupedMatmulCallTarget(
    const se::GpuComputeCapability& gpu_version) {
  return gpu_version.IsRocm() ? kHipblasLtGroupedMatmulCallTarget
                              : kCublasLtGroupedMatmulCallTarget;
}

}  // namespace gpu
}  // namespace xla
