/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/gemm_rewriter_test_lib.h"

#include <gtest/gtest.h>

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hipblaslt_custom_calls.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

const stream_executor::GpuComputeCapability& GemmRewriteTestBase::Capability()
    const {
  return device_description().gpu_compute_capability();
}

stream_executor::SemanticVersion GemmRewriteTestBase::GetToolkitVersion()
    const {
  return device_description().runtime_version();
}

bool GemmRewriteTestBase::IsCuda() const { return Capability().IsCuda(); }

bool GemmRewriteTestBase::IsRocm() const { return Capability().IsRocm(); }

bool GemmRewriteTestBase::IsSycl() const { return Capability().IsOneAPI(); }

bool GemmRewriteTestBase::IsBlackwell() const {
  if (IsCuda()) {
    return Capability().cuda_compute_capability()->IsBlackwell();
  }
  return false;
}

stream_executor::GpuComputeCapability
GemmRewriteTestBase::CudaHopperOrRocmCapability() {
  // TODO(intel-tf): Add a check for oneAPI.
  if (IsCuda()) {
    return se::CudaComputeCapability::Hopper();
  }
  return stream_executor::GpuComputeCapability{
      *Capability().rocm_compute_capability()};
}

DebugOptions GemmRewriteTestBase::GetDebugOptionsForTest() const {
  DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
  // These tests test the cuBLAS rewriter so we have to make sure that we use
  // cuBLAS for them.
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
  return debug_options;
}

bool GemmRewriteTestBase::SkipGpuBlasLtTest() {
  return IsRocm() && !Capability().rocm_compute_capability()->has_hipblaslt() &&
         GetDebugOptionsForTest().xla_gpu_enable_cublaslt();
}

bool GemmRewriteTestBase::SkipGroupedGemmTest() {
  // Grouped GEMM is only supported on ROCm with hipBLASLt on gfx942 or gfx950
  const auto* rocm_cc = Capability().rocm_compute_capability();
  return IsCuda() || IsSycl() || !rocm_cc->has_hipblaslt() ||
         (rocm_cc->has_hipblaslt() && !rocm_cc->gfx9_mi300_series());
}

bool GemmRewriteTestBase::HasFp8Support() const {
  if (IsCuda()) {
    return Capability().cuda_compute_capability()->IsAtLeast(8, 9);
  } else if (IsSycl()) {
    // TODO(intel-tf): Use compute capability to find out support when
    // available.
    return false;
  } else {
    return Capability().rocm_compute_capability()->has_fp8_support();
  }
}

bool GemmRewriteTestBase::HasCudaComputeCapability(
    const stream_executor::CudaComputeCapability& cc) const {
  return IsCuda() &&
         Capability().cuda_compute_capability()->SupportsAllFeaturesOf(cc);
}

absl::string_view GemmRewriteTestBase::MatmulTarget() const {
  return IsRocm() ? kHipblasLtMatmulCallTarget : kCublasLtMatmulCallTarget;
}

absl::string_view GemmRewriteTestBase::MatmulF8Target() const {
  return IsRocm() ? kHipblasLtMatmulF8CallTarget : kCublasLtMatmulF8CallTarget;
}

absl::string_view GemmRewriteTestBase::MatmulMxTarget() const {
  // MX block-scaled matmuls are only produced on ROCm.
  return kHipblasLtMatmulMxCallTarget;
}

absl::string_view GemmRewriteTestBase::GroupedMatmulTarget() const {
  return IsRocm() ? kHipblasLtGroupedMatmulCallTarget
                  : kCublasLtGroupedMatmulCallTarget;
}

std::string GemmRewriteTestBase::Subst(absl::string_view text) const {
  const std::string matmul_target_regex =
      absl::StrReplaceAll(MatmulTarget(), {{"$", "\\$"}});
  return absl::StrReplaceAll(
      text, {{kMatmulTargetPlaceholder, MatmulTarget()},
             {kMatmulF8TargetPlaceholder, MatmulF8Target()},
             {kMatmulMxTargetPlaceholder, MatmulMxTarget()},
             {kGroupedMatmulTargetPlaceholder, GroupedMatmulTarget()},
             {kMatmulTargetRegexPlaceholder, matmul_target_regex}});
}

void GemmRewriteTestBase::MatchOptimizedHlo(absl::string_view hlo,
                                            absl::string_view pattern,
                                            bool print_operand_shape) {
  HloPjRtGpuTestBase::MatchOptimizedHlo(hlo, Subst(pattern),
                                        print_operand_shape);
}

void GemmRewriteTestBase::RunAndFilecheckHloRewrite(
    absl::string_view hlo_with_checks, HloPassInterface&& hlo_pass,
    std::function<void(HloModule*)> after_pass_checks,
    const HloModuleConfig* config) const {
  HloPjRtGpuTestBase::RunAndFilecheckHloRewrite(
      Subst(hlo_with_checks), std::move(hlo_pass), std::move(after_pass_checks),
      config);
}

void GemmRewriteTestBase::RunAndFilecheckHloRewrite(
    absl::string_view hlo_with_filecheck_lines, HloPassInterface&& hlo_pass,
    std::optional<absl::string_view> expected,
    std::function<void(HloModule*)> after_pass_checks,
    const HloModuleConfig* config,
    absl::Span<const absl::string_view> additional_check_prefixes) const {
  const std::string hlo = Subst(hlo_with_filecheck_lines);
  std::optional<std::string> substituted_expected;
  if (expected.has_value()) {
    substituted_expected = Subst(*expected);
  }
  HloPjRtGpuTestBase::RunAndFilecheckHloRewrite(
      hlo, std::move(hlo_pass), substituted_expected,
      std::move(after_pass_checks), config, additional_check_prefixes);
}

ParameterizedGemmRewriteTestBase::ParameterizedGemmRewriteTestBase() = default;

DebugOptions ParameterizedGemmRewriteTestBase::GetDebugOptionsForTest() const {
  DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_cublaslt(true);
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  return debug_options;
}

void ParameterizedGemmRewriteTestBase::MatchOptimizedHlo(
    absl::string_view hlo, const absl::string_view pattern,
    bool print_operand_shape) {
  GemmRewriteTestBase::MatchOptimizedHlo(
      hlo, absl::StrReplaceAll(pattern, replacements_), print_operand_shape);
}

absl::string_view ParameterizedGemmRewriteTestBase::CustomCallTarget() {
  return MatmulTarget();
}

void ParameterizedGemmRewriteTestBase::SetUp() {
  if (SkipGpuBlasLtTest()) {
    GTEST_SKIP() << "BlasLt is not supported on this GPU architecture";
  }
  // Populated here rather than in the constructor: MatmulTarget() queries the
  // device under test, which is not available until the fixture is set up.
  replacements_[kCustomCallTargetPlaceholder] = MatmulTarget();
}

}  // namespace xla::gpu
