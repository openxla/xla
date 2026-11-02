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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/in_memory_store.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/autotuning/autotuner_cache.h"

namespace xla::gpu {
namespace {

// Asserts that, given a free choice between every backend that can claim a
// scaled-dot fusion, the autotuner picks one that actually uses MX.
//
// The competitor to beat is HIPBLASLT_FISSION, which lowers the scaled dot to
// bf16 converts plus a plain __cublas$lt$matmul and so exercises no MX at all.
// On gfx950 the fastest MX candidate measures roughly 5.5-9us against 12-19us
// for the fastest fissioned one, but each candidate is timed by executing it
// and the spread of a measurement grows with machine load, so this test
// profiles several runs per candidate and compares the means.
//
// Numerical correctness of the MX lowerings is a separate question, covered by
// //xla/backends/gpu/tests:mx_scaled_dot_test.
class MxScaledDotSelectionTest : public HloPjRtGpuTestBase {
 protected:
  // Each candidate is compared by the mean of this many runs, with the ones
  // far above the median thrown out as interference. Enough samples are taken
  // for that median to be meaningful: five runs without trimming failed 2 of
  // 20 times on a loaded machine, where a burst of interference inflated the
  // measurements several times over and hit the MX candidate hardest.
  static constexpr int kProfilingRuns = 20;

  void SetUp() override {
    const auto& gpu_cc = device_description().gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "MX selection test requires MX type support (gfx950+).";
    }
    ClearAutotuneCaches();
  }

  void TearDown() override { ClearAutotuneCaches(); }

  // The autotune caches are process-wide and keyed without debug options, so
  // without this a later test would reuse an earlier one's pick instead of
  // autotuning, and would assert nothing.
  static void ClearAutotuneCaches() {
    InMemoryStore::Clear();
    AutotunerCache::ClearAutotuneResults();
  }

  void ExpectMxBackendSelected(absl::string_view hlo_string) {
    HloModuleConfig config = GetModuleConfigForTest();
    config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(true);
    config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(true);
    config.mutable_debug_options().set_xla_gpu_autotune_profiling_runs(
        kProfilingRuns);
    // The backend list is deliberately left at its default, so that Triton,
    // hipBLASLt and HIPBLASLT_FISSION all compete.
    ASSERT_OK_AND_ASSIGN(auto optimized,
                         GetOptimizedModule(hlo_string, config));

    // Either MX backend is a pass:
    //   __triton_nested_gemm_fusion -> Triton
    //   __cublas$lt$matmul$mx       -> hipBLASLt
    // A bare __cublas$lt$matmul means HIPBLASLT_FISSION won, which is the
    // failure this test exists to catch, so it is rejected explicitly rather
    // than left to the positive check.
    const std::string optimized_text = optimized->ToString();
    EXPECT_THAT(
        RunFileCheck(
            optimized_text,
            R"(CHECK: {{__triton_nested_gemm_fusion|__cublas\$lt\$matmul\$mx}})"),
        absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(optimized_text,
                ::testing::Not(::testing::HasSubstr(
                    "custom_call_target=\"__cublas$lt$matmul\"")));
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

TEST_F(MxScaledDotSelectionTest, MxFp8BeatsFission) {
  ExpectMxBackendSelected(kMxFp8Hlo);
}

// The case that regressed in practice: Triton produces no config for mixed
// f4e2m1fn x f8e4m3fn operands, so hipBLASLt MX is the only MX candidate and
// has to win against eight fissioned ones on its own.
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

TEST_F(MxScaledDotSelectionTest, MxFp4Fp8MixedBatchedBeatsFission) {
  ExpectMxBackendSelected(kMxFp4Fp8MixedBatchedHlo);
}

}  // namespace
}  // namespace xla::gpu
