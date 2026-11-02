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

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/in_memory_store.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/autotuning/autotuner_cache.h"

namespace xla::gpu {
namespace {

// Numerical correctness of the scaled-dot lowerings, each checked against the
// same computation lowered without MX.
//
// All three backends that can claim a scaled-dot fusion are covered, including
// HIPBLASLT_FISSION, which uses no MX at all: it converts the operands and the
// scales to bf16 and runs an ordinary GEMM. The autotuner is free to select it
// in production whenever it measures fastest, so its numerics have to be right
// too.
//
// Which backend lowers the scaled dot is pinned per test rather than left to
// the autotuner, so that this test exercises a known code path instead of
// whichever candidate happened to measure fastest. Whether the autotuner picks
// an MX backend when it is free to choose is a separate question, covered by
// //xla/backends/gpu/autotuner:mx_scaled_dot_selection_test.
struct MxBackendParam {
  autotuner::Backend backend;
  // FileCheck regex, so the '$' in the custom call targets is escaped.
  absl::string_view expected_target_regex;
  absl::string_view name;
};

class MxScaledDotTest : public HloPjRtGpuTestBase,
                        public ::testing::WithParamInterface<MxBackendParam> {
 protected:
  void SetUp() override {
    const auto& gpu_cc = device_description().gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "MX execution test requires MX type support (gfx950+).";
    }
    ClearAutotuneCaches();
  }

  void TearDown() override { ClearAutotuneCaches(); }

  // The autotune caches are process-wide and keyed without debug options, so
  // without this the reference compile, or an earlier parameter of this test,
  // would hand its pick to a later compile of the same scaled dot and the
  // backend pinned below would never be consulted.
  static void ClearAutotuneCaches() {
    InMemoryStore::Clear();
    AutotunerCache::ClearAutotuneResults();
  }

  // Compares the scaled dot lowered by the backend under test against the same
  // module lowered without MX, and verifies that both took the intended path.
  //
  // `supported_by_triton` must be false for operand type combinations the
  // Triton emitter has no config for; the Triton parameter is skipped for
  // those rather than failing to compile.
  void RunMxCorrectnessTest(absl::string_view hlo_string,
                            const ErrorSpec& error_spec,
                            bool supported_by_triton = true) {
    if (!supported_by_triton &&
        GetParam().backend == autotuner::Backend::TRITON) {
      GTEST_SKIP() << "The Triton emitter produces no config for these operand "
                      "types, so there is nothing to pin it to.";
    }

    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    ASSERT_OK_AND_ASSIGN(auto ref_optimized,
                         GetOptimizedModule(hlo_string, ref_config));
    EXPECT_THAT(
        RunFileCheck(ref_optimized->ToString(),
                     R"(CHECK: {{__cublas\$lt\$matmul|__cublas\$gemm}})"),
        absl_testing::IsOkAndHolds(true));
    ClearAutotuneCaches();

    HloModuleConfig test_config = GetModuleConfigForTest();
    test_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(true);
    test_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(true);
    // Both MX lowerings require the scaled dot to be wrapped in a Triton GEMM
    // fusion first, so the two flags above stay on even for hipBLASLt.
    //
    // The backend list defaults to every backend, so it has to be cleared
    // before the one under test is added, otherwise adding is a no-op.
    // NATIVE_EMITTER stays enabled because the non-GEMM fusions in these
    // modules have no other backend.
    test_config.mutable_debug_options()
        .clear_xla_gpu_experimental_autotune_backends();
    test_config.mutable_debug_options()
        .add_xla_gpu_experimental_autotune_backends(GetParam().backend);
    test_config.mutable_debug_options()
        .add_xla_gpu_experimental_autotune_backends(
            autotuner::Backend::NATIVE_EMITTER);
    ASSERT_OK_AND_ASSIGN(auto test_optimized,
                         GetOptimizedModule(hlo_string, test_config));
    EXPECT_THAT(
        RunFileCheck(
            test_optimized->ToString(),
            absl::StrCat("CHECK: {{", GetParam().expected_target_regex, "}}")),
        absl_testing::IsOkAndHolds(true));

    EXPECT_TRUE(RunAndCompareTwoModules(std::move(test_optimized),
                                        std::move(ref_optimized), error_spec,
                                        /*run_hlo_passes=*/false));
  }
};

constexpr absl::string_view kMxFp8Hlo = R"(
HloModule mx_test
ENTRY main {
  %lhs = f8e4m3fn[32,256] parameter(0)
  %rhs = f8e4m3fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_P(MxScaledDotTest, MxFp8Correctness) {
  RunMxCorrectnessTest(kMxFp8Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp8BatchedHlo = R"(
HloModule mx_batched_test
ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotTest, MxFp8BatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp8BatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp8MixedTypesBatchedHlo = R"(
HloModule mx_fp8_mixed_types_batched_test
ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e5m2[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotTest, MxFp8MixedTypesBatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp8MixedTypesBatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4Hlo = R"(
HloModule mx_fp4_test
ENTRY main {
  %lhs = f4e2m1fn[32,256] parameter(0)
  %rhs = f4e2m1fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_P(MxScaledDotTest, MxFp4Correctness) {
  RunMxCorrectnessTest(kMxFp4Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4BatchedHlo = R"(
HloModule mx_fp4_batched_test
ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f4e2m1fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotTest, MxFp4BatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp4BatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4Fp8MixedBatchedHlo = R"(
HloModule mx_fp4_fp8_mixed_batched_test
ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotTest, MxFp4Fp8MixedBatchedCorrectness) {
  // TODO: the Triton emitter produces no config for a mixed f4e2m1fn x
  // f8e4m3fn scaled dot, so only hipBLASLt covers this case today.
  RunMxCorrectnessTest(kMxFp4Fp8MixedBatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
                       /*supported_by_triton=*/false);
}

// The scaled-dot operands are reshapes of higher-rank parameters, mimicking the
// in-graph block-scaled quantization case where the fp8 operand is produced in
// [batch, M, K/block, block] form and only flattened to [batch, M, K] by a
// reshape/bitcast that gets absorbed into the gemm fusion. This exercises the
// path where the fusion's external operands have a different rank than the
// scaled-dot operands; the MX rewrite must re-bitcast them back so the custom
// call operands stay consistent with the dot dimension numbers.
constexpr absl::string_view kMxFp8ReshapedOperandsHlo = R"(
HloModule mx_fp8_reshaped_operands_test
ENTRY main {
  %lhs_blocked = f8e4m3fn[1,32,8,32] parameter(0)
  %rhs_blocked = f8e4m3fn[1,16,8,32] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  %lhs = f8e4m3fn[1,32,256] reshape(%lhs_blocked)
  %rhs = f8e4m3fn[1,16,256] reshape(%rhs_blocked)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_P(MxScaledDotTest, MxFp8ReshapedOperandsCorrectness) {
  RunMxCorrectnessTest(kMxFp8ReshapedOperandsHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

INSTANTIATE_TEST_SUITE_P(
    MxBackends, MxScaledDotTest,
    ::testing::Values(
        MxBackendParam{autotuner::Backend::TRITON,
                       R"(__triton_nested_gemm_fusion)", "Triton"},
        MxBackendParam{autotuner::Backend::HIPBLASLT,
                       R"(__cublas\$lt\$matmul\$mx)", "HipblasLt"},
        // The trailing quote stops this matching the "$mx"
        // target, which shares the same prefix.
        MxBackendParam{autotuner::Backend::HIPBLASLT_FISSION,
                       R"(custom_call_target="__cublas\$lt\$matmul")",
                       "HipblasLtFission"}),
    [](const ::testing::TestParamInfo<MxBackendParam>& info) {
      return std::string(info.param.name);
    });

}  // namespace
}  // namespace xla::gpu
