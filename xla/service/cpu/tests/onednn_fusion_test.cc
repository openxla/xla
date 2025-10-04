/* Copyright 2025 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/service/cpu/onednn_util.h"

namespace xla::cpu {
namespace {

using OneDnnFusionTest = HloTestBase;

inline constexpr bool IsOneDnnGraphEnabled() {
#if defined(XLA_ONEDNN_USE_GRAPH_API)
  return true;
#endif  // XLA_ONEDNN_USE_GRAPH_API
  return false;
}

// TODO(intel-tf): Expand testing to other dtypes as library integration
// matures.
class OneDnnGraphPipelineTest
    : public HloTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_experimental_onednn_custom_call(false);
    debug_options.set_xla_cpu_use_onednn(true);
    debug_options.set_xla_cpu_use_xnnpack(false);
    debug_options.mutable_xla_cpu_experimental_onednn_fusion_type()->Add(
        DebugOptions::LIBRARY_FUSION_TYPE_DOT);
    return debug_options;
  }

  PrimitiveType dtype_;
  std::string dtypeString_;
  float atol_;
  float rtol_;

  constexpr static const char* kOneDNNBasicFusionStr = R"(
    ; CHECK:     fusion(%{{[a-z,A-Z,0-9,_,-,.]*}}, %{{[a-z,A-Z,0-9,_,-,.]*}})
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "fusion_config":{
    ; CHECK-DAG:        "kind":"__onednn_fusion"
    ; CHECK-DAG:      }
    ; CHECK-DAG:     }
    )";

  OneDnnGraphPipelineTest() {
    dtype_ = GetParam();
    atol_ = rtol_ = (dtype_ == F32) ? 1e-4 : 1e-2;
    dtypeString_ = primitive_util::LowercasePrimitiveTypeName(dtype_);
  }

  void SetUp() override {
    if (!IsOneDnnGraphEnabled()) {
      GTEST_SKIP() << "oneDNN graph is disabled!";
    }
    if (!IsSupportedType(dtype_)) {
      GTEST_SKIP() << "CPU does not support " << dtypeString_;
    }
  }

  void AdjustToleranceForDtype(PrimitiveType for_type, float atol, float rtol) {
    if (dtype_ == for_type) {
      atol_ = atol;
      rtol_ = rtol;
    }
  }

  void RunCompareAndMatchOptimizedHlo(const absl::string_view outline) {
    const std::string module_str =
        absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
    EXPECT_TRUE(RunAndCompare(module_str, ErrorSpec{atol_, rtol_}));
    MatchOptimizedHlo(module_str, kOneDNNBasicFusionStr);
  }
};

TEST_F(OneDnnFusionTest, Exponential) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule exp

    onednn_fusion {
      %p0 = f32[4] parameter(0)
      ROOT %exp = f32[4] exponential(%p0)
    }

    ENTRY entry {
      %p0 = f32[4] parameter(0)
      ROOT %fusion = f32[4] fusion(%p0), kind=kCustom, calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

// TODO(penporn): Make a parameterized BinaryEltwiseOp test instead.
TEST_F(OneDnnFusionTest, Add) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add

    onednn_fusion {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %add = f32[10] add(%p0, %p1)
    }

    ENTRY entry {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %fusion = f32[10] fusion(%p0, %p1), kind=kCustom, calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

TEST_F(OneDnnFusionTest, Mul) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule mul

    onednn_fusion {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %mul = f32[10] multiply(%p0, %p1)
    }

    ENTRY entry {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %fusion = f32[10] fusion(%p0, %p1), kind=kCustom, calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

TEST_F(OneDnnFusionTest, MatMul) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule mul

    onednn_fusion {
      %p0 = f32[10,20] parameter(0)
      %p1 = f32[20,30] parameter(1)
      ROOT %mul = f32[10,30] dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY entry {
      %p0 = f32[10,20] parameter(0)
      %p1 = f32[20,30] parameter(1)
      ROOT %fusion = f32[10,30] fusion(%p0, %p1), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

TEST_F(OneDnnFusionTest, MatMulAdd) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule mul
    onednn_fusion {
      %p0 = f32[10,20] parameter(0)
      %p1 = f32[20,30] parameter(1)
      %dot = f32[10,30] dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %p2 = f32[10,30] parameter(2)
      ROOT %add = f32[10,30] add(%dot, %p2)
    }
    ENTRY entry {
      %p0 = f32[10,20] parameter(0)
      %p1 = f32[20,30] parameter(1)
      %p2 = f32[10,30] parameter(2)
      ROOT %fusion = f32[10,30] fusion(%p0, %p1, %p2), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

TEST_P(OneDnnGraphPipelineTest, SimpleTestGemv) {
  const absl::string_view outline = R"(
  HloModule gemv
  ENTRY entry {
    arg.0 = $dtype[10,100] parameter(0)
    arg.1 = $dtype[100] parameter(1)
    ROOT dot.0 = $dtype[10] dot(arg.0, arg.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  RunCompareAndMatchOptimizedHlo(outline);
}

TEST_P(OneDnnGraphPipelineTest, MismatchedOpRankDot) {
  const absl::string_view outline = R"(
  HloModule mismatched_rank
  ENTRY entry {
    arg.0 = $dtype[100,300,300] parameter(0)
    arg.1 = $dtype[300] parameter(1)
    ROOT dot.0 = $dtype[100,300] dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  RunCompareAndMatchOptimizedHlo(outline);
}

INSTANTIATE_TEST_SUITE_P(
    OneDnnGraphPipelineTestSuite, OneDnnGraphPipelineTest,
    ::testing::Values(F32),
    [](const ::testing::TestParamInfo<OneDnnGraphPipelineTest::ParamType>&
           info) {
      auto test_name = primitive_util::LowercasePrimitiveTypeName(info.param);
      std::transform(test_name.begin(), test_name.end(), test_name.begin(),
                     [](auto c) { return std::toupper(c); });
      return test_name;
    });

}  // namespace
}  // namespace xla::cpu
