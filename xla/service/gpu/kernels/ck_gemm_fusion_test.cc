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

#include "xla/service/gpu/kernels/ck_gemm_fusion.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"


namespace xla::gpu {

class CkFusionTest : public HloTestBase {
};

//===----------------------------------------------------------------------===//
// Pattern matching tests
//===----------------------------------------------------------------------===//

TEST_F(CkFusionTest, RowMajorGemm) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: f16[15,19], p1: f16[19,17]) -> f16[15,17] {
      %p0 = f16[15,19]{1,0} parameter(0)
      %p1 = f16[19,17]{1,0} parameter(1)
      ROOT %r = f16[15,17]{1,0} dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* expected = R"(
    ; CHECK: %ck_gemm {{.*}} {
    ; CHECK:   [[P0:%[^ ]+]] = f16[15,19]{1,0} parameter(0)
    ; CHECK:   [[P1:%[^ ]+]] = f16[19,17]{1,0} parameter(1)
    ; CHECK:   ROOT [[DOT:%[^ ]+]] = f16[15,17]{1,0} dot([[P0]], [[P1]]),
    ; CHECK:     lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   ROOT [[FUSION:%[^ ]+]] = f16[15,17]{1,0} fusion
    ; CHECK:     kind=kCustom, calls=%ck_gemm,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{"name":"ck_gemm","kernel_index":0}
    ; CHECK:     }
    ; CHECK: }
  )";

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<CkGemmPattern>();

  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  CustomKernelFusionRewriter pass(&device, /*kernel_index=*/0, &patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

//===----------------------------------------------------------------------===//
// Run And Compare Tests
//===----------------------------------------------------------------------===//

TEST_F(CkFusionTest, BF16RowMajorGemmKernel) {
  ErrorSpec error_spec{/*aabs=*/1e-2, /*arel=*/1e-2};

  const char* hlo_text_ref = R"(
  HloModule test

  ENTRY %main (p0: bf16[256,64], p1: bf16[64,256]) -> bf16[256,256] {
    %p0 = bf16[256,64]{1,0} parameter(0)
    %p1 = bf16[64,256]{1,0} parameter(1)
    ROOT %r = bf16[256,256]{1,0} dot(%p0, %p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule ck

  ck_gemm {
    arg0 = bf16[256,64]{1,0} parameter(0)
    arg1 = bf16[64,256]{1,0} parameter(1)
    ROOT dot = bf16[256,256]{1,0} dot(arg0, arg1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    arg0 = bf16[256,64]{1,0} parameter(0)
    arg1 = bf16[64,256]{1,0} parameter(1)
    ROOT _ = bf16[256,256]{1,0} fusion(arg0, arg1), kind=kCustom, calls=ck_gemm,
      backend_config={"fusion_backend_config":{kind: "__custom_fusion", custom_fusion_config: {"name":"ck_gemm", "kernel_index":0}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}

TEST_F(CkFusionTest, FP16RowMajorGemmKernel) {
  ErrorSpec error_spec{/*aabs=*/1e-2, /*arel=*/1e-2};

  const char* hlo_text_ref = R"(
  HloModule test

  ENTRY %main (p0: f16[768,256], p1: f16[256,512]) -> f16[768,512] {
    %p0 = f16[768,256]{1,0} parameter(0)
    %p1 = f16[256,512]{1,0} parameter(1)
    ROOT %r = f16[768,512]{1,0} dot(%p0, %p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule ck

  ck_gemm {
    arg0 = f16[768,256]{1,0} parameter(0)
    arg1 = f16[256,512]{1,0} parameter(1)
    ROOT dot = f16[256,512]{1,0} dot(arg0, arg1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    arg0 = f16[768,256]{1,0} parameter(0)
    arg1 = f16[256,512]{1,0} parameter(1)
    ROOT _ = f16[768,512]{1,0} fusion(arg0, arg1), kind=kCustom, calls=ck_gemm,
      backend_config={"fusion_backend_config":{kind: "__custom_fusion", custom_fusion_config: {"name":"ck_gemm", "kernel_index":0}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}


}  // namespace xla::gpu
