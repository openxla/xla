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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GEMM_REWRITER_TEST_LIB_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GEMM_REWRITER_TEST_LIB_H_

#include <gtest/gtest.h>

#include <functional>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Base class for GEMM rewriter tests.
class GemmRewriteTestBase : public HloPjRtGpuTestBase {
 protected:
  const stream_executor::GpuComputeCapability& Capability() const;

  stream_executor::SemanticVersion GetToolkitVersion() const;

  bool IsCuda() const;

  bool IsRocm() const;

  bool IsSycl() const;

  bool IsBlackwell() const;

  stream_executor::GpuComputeCapability CudaHopperOrRocmCapability();

  DebugOptions GetDebugOptionsForTest() const override;

  bool SkipGpuBlasLtTest();

  bool SkipGroupedGemmTest();

  bool HasFp8Support() const;

  bool HasCudaComputeCapability(
      const stream_executor::CudaComputeCapability& cc) const;

  // Custom call target placeholders. Write these in HLO and FileCheck patterns
  // instead of a literal "__cublas$lt$matmul" / "__hipblaslt$matmul": Subst()
  // expands them to the spelling the GPU under test actually emits, so the
  // same test source works on CUDA, ROCm and SYCL.
  static constexpr absl::string_view kMatmulTargetPlaceholder =
      "<<BLASLT_MATMUL_TARGET>>";
  static constexpr absl::string_view kMatmulF8TargetPlaceholder =
      "<<BLASLT_MATMUL_F8_TARGET>>";
  static constexpr absl::string_view kMatmulMxTargetPlaceholder =
      "<<BLASLT_MATMUL_MX_TARGET>>";
  static constexpr absl::string_view kGroupedMatmulTargetPlaceholder =
      "<<BLASLT_GROUPED_MATMUL_TARGET>>";
  // Same as kMatmulTargetPlaceholder, but with '$' escaped so the result can
  // be used inside a FileCheck `{{...}}` regex, where a bare '$' would mean
  // end-of-line.
  static constexpr absl::string_view kMatmulTargetRegexPlaceholder =
      "<<BLASLT_MATMUL_TARGET_RE>>";

  // The custom call targets emitted for the GPU under test.
  absl::string_view MatmulTarget() const;
  absl::string_view MatmulF8Target() const;
  absl::string_view MatmulMxTarget() const;
  absl::string_view GroupedMatmulTarget() const;

  // Expands the placeholders above in `text`.
  std::string Subst(absl::string_view text) const;

  // The following shadow their base-class counterparts purely to run Subst()
  // over the HLO and FileCheck text first, so that tests can write the
  // placeholders above instead of a backend-specific target. They delegate
  // unchanged otherwise. Both RunAndFilecheckHloRewrite overloads are
  // redeclared because declaring one would hide the other.
  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false);

  void RunAndFilecheckHloRewrite(
      absl::string_view hlo_with_checks, HloPassInterface&& hlo_pass,
      std::function<void(HloModule*)> after_pass_checks = nullptr,
      const HloModuleConfig* config = nullptr) const;

  void RunAndFilecheckHloRewrite(
      absl::string_view hlo_with_filecheck_lines, HloPassInterface&& hlo_pass,
      std::optional<absl::string_view> expected,
      std::function<void(HloModule*)> after_pass_checks = nullptr,
      const HloModuleConfig* config = nullptr,
      absl::Span<const absl::string_view> additional_check_prefixes = {}) const;
};

// A test fixture class for tests which should have similar results with legacy
// cublas and cublasLt
class ParameterizedGemmRewriteTestBase : public GemmRewriteTestBase {
 public:
  ParameterizedGemmRewriteTestBase();

  DebugOptions GetDebugOptionsForTest() const override;

  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false);

  absl::string_view CustomCallTarget();

 protected:
  void SetUp() override;

 protected:
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements_;

 private:
  static constexpr absl::string_view kCustomCallTargetPlaceholder =
      "<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>";
  ;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GEMM_REWRITER_TEST_LIB_H_
