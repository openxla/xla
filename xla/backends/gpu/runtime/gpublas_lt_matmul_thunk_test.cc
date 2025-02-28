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

#include <cstddef>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/executable_run_options.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/test.h"
 
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla::gpu {

namespace {
 
class GpuBlasLtMatmulThunkTest : public HloTestBase {

 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(true);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }
  void SetUp() override {
    auto dbg = GetDebugOptionsForTest();
    const auto& gpu_cc = backend().default_stream_executor()
                               ->GetDeviceDescription()
                               .gpu_compute_capability();
    if (auto* rocm = std::get_if<se::RocmComputeCapability>(&gpu_cc);
        rocm != nullptr && !rocm->has_hipblaslt()) {
      GTEST_SKIP() << "No hipblas-lt support on this architecture!";
    }
  }
};

XLA_TEST_F(GpuBlasLtMatmulThunkTest, SharedMatmulPlans) {
  absl::string_view hlo_single_plan =
      R"(
HloModule SharedMatmulPlan

ENTRY test {
  x1 = f32[101,407] parameter(0)
  x2 = f32[101,407] parameter(1)
  x3 = f32[101,407] parameter(2)
  y = f32[407,400] parameter(3)
  z = f32[407,400] parameter(4)
  w = f32[407,400] parameter(5)
  dot_a = f32[101,400] dot(x1, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_b = f32[101,400] dot(x2, z), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_c = f32[101,400] dot(x3, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul_ab = f32[101,400] multiply(dot_a, dot_b)
  ROOT abc = f32[101,400] subtract(mul_ab, dot_c)
})";

  EXPECT_TRUE(RunAndCompare(hlo_single_plan, ErrorSpec{1e-3, 1e-3}));
  // Assert that only one MatmulPlan cache entry was created.
  EXPECT_TRUE(CublasLtMatmulThunk::MatmulPlanCacheSize(0) == 1);

  absl::string_view hlo_two_plans =
      R"(
HloModule SharedMatmulPlan

ENTRY test {
  x1 = f32[101,407] parameter(0)
  x2 = f32[101,407] parameter(1)
  x3 = f32[101,407] parameter(2)
  y = f32[407,400] parameter(3)
  z = f32[407,400] parameter(4)
  w = f32[407,400] parameter(5)
  c = f32[] constant(0)
  c_bcast = f32[101,400] broadcast(c), dimensions={}
  dot_a = f32[101,400] dot(x1, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  out_a = f32[101,400] maximum(dot_a, c_bcast)
  dot_b = f32[101,400] dot(x2, z), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_c = f32[101,400] dot(x3, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul_ab = f32[101,400] multiply(out_a, dot_b)
  ROOT abc = f32[101,400] subtract(mul_ab, dot_c)
})";

  EXPECT_TRUE(RunAndCompare(hlo_two_plans, ErrorSpec{1e-3, 1e-3}));
  // Assert that we have now 2 MatmulPlans (one more created for ReLu epilogue).
  EXPECT_TRUE(CublasLtMatmulThunk::MatmulPlanCacheSize(0) == 2);
}

} // namespace
}  // namespace xla::gpu
