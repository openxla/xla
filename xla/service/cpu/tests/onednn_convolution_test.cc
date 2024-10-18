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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "tsl/platform/cpu_info.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/cpu/onednn_contraction_rewriter.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace cpu {

class ConvolutionTest : public HloTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_use_thunk_runtime(false);
    return debug_options;
  }

  const char* conv_rewrite_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";

  const char* conv_rewrite_bias_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:       "fusions":{
    ; CHECK-DAG:         "ops":["BIAS"]
    ; CHECK-DAG:     }
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";

  const char* fused_convolution_binary_add_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:       "fusions":{
    ; CHECK-DAG:         "ops":["BINARY_ADD"]
    ; CHECK-DAG:     }
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";

  void CheckCustomCallTypes(std::unique_ptr<HloModule>& module,
                            PrimitiveType operand0_type,
                            PrimitiveType operand1_type,
                            PrimitiveType result_type) {
    HloInstruction* custom_call =
        FindInstruction(module.get(), HloOpcode::kCustomCall);
    if (custom_call) {
      EXPECT_EQ(custom_call->operand(0)->shape().element_type(), operand0_type);
      EXPECT_EQ(custom_call->operand(1)->shape().element_type(), operand1_type);
      auto actual_type =
          custom_call->shape().IsTuple()
              ? custom_call->shape().tuple_shapes(0).element_type()
              : custom_call->shape().element_type();
      EXPECT_EQ(actual_type, result_type);
    } else {
      FAIL() << "CustomCall not found in the optimized module";
    }
  }

  void CheckCustomCallTypesMultiInstrs(std::unique_ptr<HloModule>& module,
                                      std::vector<PrimitiveType>& operand0_types,
                                      std::vector<PrimitiveType>& operand1_types,
                                      std::vector<PrimitiveType>& result_types) {
    std::vector<HloInstruction*> custom_calls = FindInstructions(module.get(), HloOpcode::kCustomCall);
    if (custom_calls.size() == 0) {
      FAIL() << "CustomCall not found in the optimized module";
    }
    for (int i = 0; i < custom_calls.size(); ++i) {
      EXPECT_EQ(custom_calls[i]->operand(0)->shape().element_type(), operand0_types[i]);
      EXPECT_EQ(custom_calls[i]->operand(1)->shape().element_type(), operand1_types[i]);
      auto actual_type =
          custom_calls[i]->shape().IsTuple()
              ? custom_calls[i]->shape().tuple_shapes(0).element_type()
              : custom_calls[i]->shape().element_type();
      EXPECT_EQ(actual_type, result_types[i]);
    }                                    
  }

};

TEST_F(ConvolutionTest, Simple2DTestF32) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32

  ENTRY convolution.test.f32 {
    arg.0 = f32[1,22,22,1] parameter(0)
    reshape.0 = f32[1,22,22,1] reshape(arg.0)
    arg.1 = f32[8,8,1,1] parameter(1)
    reshape.1 = f32[8,8,1,1] reshape(arg.1)
    convolution.0 = f32[1,11,11,1] convolution(reshape.0, reshape.1), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    reshape.2 = f32[1,11,11,1] reshape(convolution.0)
    tuple.0 = (f32[1,11,11,1]) tuple(reshape.2)
    ROOT get-tuple-element.0 = f32[1,11,11,1] get-tuple-element(tuple.0), index=0
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Simple3DTestBF16) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* convolution_module_str = R"(
  HloModule convolution.test.bf16

  ENTRY convolution.test.bf16 {
    p0 = bf16[8,4,5,5,1] parameter(0)
    p1 = bf16[3,3,3,1,32] parameter(1)
    ROOT conv = bf16[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
})";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Simple2DTestF16) {
  if (!IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }

  const char* convolution_module_str = R"(
  HloModule convolution.test.f16

  ENTRY convolution.test.bf16 {
    p0 = f16[8,4,5,5,1] parameter(0)
    p1 = f16[3,3,3,1,32] parameter(1)
    ROOT conv = f16[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
})";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Conv3DWithBiasBF16) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.with.bias.relu.bf16.3D

  ENTRY TestComputation {
    arg.0 = bf16[15,4,5,5,28] parameter(0)
    arg.1 = bf16[3,3,3,28,64] parameter(1)
    conv = bf16[15,4,5,5,64] convolution(arg.0, arg.1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = bf16[64] parameter(2)
    broadcasted_bias = bf16[15,4,5,5,64] broadcast(bias), dimensions={4}
    ROOT add = bf16[15,4,5,5,64] add(conv, broadcasted_bias)
})";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_);
}

TEST_F(ConvolutionTest, SimpleTestF32WithBinaryAddFusion1) {
  const char* convolution_module_str = R"(
  HloModule conv.binaryadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[1,22,22,1] parameter(0)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[8,8,1,1] broadcast(constant.3), dimensions={}
    convolution.0 = f32[1,11,11,1] convolution(arg0.1, broadcast.4), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[1] broadcast(constant.5), dimensions={}
    broadcast.9 = f32[1,11,11,1] broadcast(broadcast.6), dimensions={3}
    ROOT add.10 = f32[1,11,11,1] add(convolution.0, broadcast.9)
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, fused_convolution_binary_add_);
}

// This test should match BIAS + Residual Add when the residual add fusion is
// re-enabled.
TEST_F(ConvolutionTest, SimpleTestBF16WithBiasAndAddFusion) {
  const char* convolution_module_str = R"(
  HloModule convolution.add.test.bf16

  ENTRY convolution.add.test.bf16 {
    arg0.1 = bf16[1,22,22,1] parameter(0)
    arg0.2 = bf16[8,8,1,10] parameter(1)
    convolution.0 = bf16[1,11,11,10] convolution(arg0.1, arg0.2), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    const.0 = bf16[10] constant(15)
    bcast.1 = bf16[1,11,11,10] broadcast(const.0), dimensions={3}
    add.0 = bf16[1,11,11,10] add(convolution.0, bcast.1)
    const.1 = bf16[1,11,11,10] constant({...})
    ROOT add.1 = bf16[1,11,11,10] add(add.0, const.1)
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_);
}


TEST_F(ConvolutionTest, DequantizeConv2D) {
  if (!IsSupportedType(PrimitiveType::S8)) {
    GTEST_SKIP() << "CPU does not support INT8.";
  }
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32, entry_computation_layout={(s8[1,3,224,224]{3,2,1,0}, s8[64,3,7,7]{3,2,1,0})->f32[1,112,112,64]{3,2,1,0}}
  ENTRY convolution.test.f32 {
    Arg_inp = s8[1,3,224,224]{3,2,1,0} parameter(0)
    convert.194 = s32[1,3,224,224]{3,2,1,0} convert(Arg_inp)
    constant.65 = s32[] constant(-4)
    broadcast.1 = s32[1,3,224,224]{3,2,1,0} broadcast(constant.65), dimensions={}
    add = s32[1,3,224,224]{3,2,1,0} add(convert.194, broadcast.1)
    convert.196 = f32[1,3,224,224]{3,2,1,0} convert(add)
    constant.48 = f32[] constant(0.5)
    broadcast.186 = f32[1,3,224,224]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.197 = f32[1,3,224,224]{3,2,1,0} multiply(convert.196, broadcast.186)
    transpose = f32[1,224,224,3]{3,2,1,0} transpose(multiply.197), dimensions={0,2,3,1}
    Arg_9.10 = s8[64,3,7,7]{3,2,1,0} parameter(1)
    convert.205 = s32[64,3,7,7]{3,2,1,0} convert(Arg_9.10)
    constant.66 = s32[] constant(0)
    broadcast.3 = s32[64,3,7,7]{3,2,1,0} broadcast(constant.66), dimensions={}
    add.1 = s32[64,3,7,7]{3,2,1,0} add(convert.205, broadcast.3)
    convert.207 = f32[64,3,7,7]{3,2,1,0} convert(add.1)
    broadcast.163 = f32[64,3,7,7]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.208 = f32[64,3,7,7]{3,2,1,0} multiply(convert.207, broadcast.163)
    transpose.1 = f32[7,7,3,64]{3,2,1,0} transpose(multiply.208), dimensions={2,3,1,0}
    ROOT convolution = f32[1,112,112,64]{3,2,1,0} convolution(transpose, transpose.1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  std::unique_ptr<HloModule> optimized_module;
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_, false,
                    &optimized_module);
  CheckCustomCallTypes(optimized_module, S8, S8, F32);
}

TEST_F(ConvolutionTest, DequantizeConv2DBias) {
  if (!IsSupportedType(PrimitiveType::S8)) {
    GTEST_SKIP() << "CPU does not support INT8.";
  }
  const char* convolution_module_str = R"(
  HloModule DequantizeConv2DBias, alias_passthrough_params=true, entry_computation_layout={(s8[1,224,224,3]{3,2,1,0}, s8[7,7,3,64]{3,2,1,0}, f32[64]{0})->f32[1,112,112,64]{3,2,1,0}}

  ENTRY DequantizeConv2DBias {
    arg0.1 = s8[1,224,224,3]{3,2,1,0} parameter(0), parameter_replication={false}
    convert.14 = s32[1,224,224,3]{3,2,1,0} convert(arg0.1)
    constant = s32[] constant(4)
    broadcast = s32[1,224,224,3]{3,2,1,0} broadcast(constant), dimensions={}
    add = s32[1,224,224,3]{3,2,1,0} add(convert.14, broadcast)
    convert.16 = f32[1,224,224,3]{3,2,1,0} convert(add)
    constant.9 = f32[] constant(0.5)
    broadcast.10 = f32[1,224,224,3]{3,2,1,0} broadcast(constant.9), dimensions={}
    multiply.17 = f32[1,224,224,3]{3,2,1,0} multiply(convert.16, broadcast.10)
    arg1.2 = s8[7,7,3,64]{3,2,1,0} parameter(1), parameter_replication={false}
    convert.25 = s32[7,7,3,64]{3,2,1,0} convert(arg1.2)
    convert.27 = f32[7,7,3,64]{3,2,1,0} convert(convert.25)
    constant.20 = f32[] constant(0.2)
    broadcast.21 = f32[7,7,3,64]{3,2,1,0} broadcast(constant.20), dimensions={}
    multiply.28 = f32[7,7,3,64]{3,2,1,0} multiply(convert.27, broadcast.21)
    convolution.29 = f32[1,112,112,64]{3,2,1,0} convolution(multiply.17, multiply.28), window={size=7x7 stride=2x2 pad=2_3x2_3}, dim_labels=b01f_01io->b01f
    arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}
    broadcast.30 = f32[1,112,112,64]{3,2,1,0} broadcast(arg2.3), dimensions={3}
    ROOT add.31 = f32[1,112,112,64]{3,2,1,0} add(convolution.29, broadcast.30)
  })";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  std::unique_ptr<HloModule> optimized_module;
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_, false,
                    &optimized_module);
  CheckCustomCallTypes(optimized_module, S8, S8, F32);
}

TEST_F(ConvolutionTest, DequantizeConv2DBiasRequantize) {
  if (!IsSupportedType(PrimitiveType::S8)) {
    GTEST_SKIP() << "CPU does not support INT8.";
  }
  const char* convolution_module_str = R"(
  HloModule DequantizeConv2DBiasRequantize, alias_passthrough_params=true, entry_computation_layout={(s8[1,56,56,3]{3,2,1,0}, s8[3,3,3,64]{3,2,1,0}, f32[64]{0}, s8[3,3,64,64]{3,2,1,0})->f32[1,14,14,64]{3,2,1,0}}

  ENTRY DequantizeConv2DBiasRequantize {
    constant.36 = f32[] constant(-128)
    broadcast.41 = f32[1,28,28,64]{3,2,1,0} broadcast(constant.36), dimensions={}
    arg0.1 = s8[1,56,56,3]{3,2,1,0} parameter(0), parameter_replication={false}
    convert.16 = s32[1,56,56,3]{3,2,1,0} convert(arg0.1)
    constant.1 = s32[] constant(4)
    broadcast = s32[1,56,56,3]{3,2,1,0} broadcast(constant.1), dimensions={}
    add = s32[1,56,56,3]{3,2,1,0} add(convert.16, broadcast)
    convert.18 = f32[1,56,56,3]{3,2,1,0} convert(add)
    constant.11 = f32[] constant(0.5)
    broadcast.12 = f32[1,56,56,3]{3,2,1,0} broadcast(constant.11), dimensions={}
    multiply.19 = f32[1,56,56,3]{3,2,1,0} multiply(convert.18, broadcast.12)
    arg1.2 = s8[3,3,3,64]{3,2,1,0} parameter(1), parameter_replication={false}
    convert.27 = s32[3,3,3,64]{3,2,1,0} convert(arg1.2)
    convert.29 = f32[3,3,3,64]{3,2,1,0} convert(convert.27)
    constant.22 = f32[] constant(0.2)
    broadcast.23 = f32[3,3,3,64]{3,2,1,0} broadcast(constant.22), dimensions={}
    multiply.30 = f32[3,3,3,64]{3,2,1,0} multiply(convert.29, broadcast.23)
    convolution.31 = f32[1,28,28,64]{3,2,1,0} convolution(multiply.19, multiply.30), window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
    arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}
    broadcast.32 = f32[1,28,28,64]{3,2,1,0} broadcast(arg2.3), dimensions={3}
    add.33 = f32[1,28,28,64]{3,2,1,0} add(convolution.31, broadcast.32)
    constant = f32[] constant(10)
    broadcast.1 = f32[1,28,28,64]{3,2,1,0} broadcast(constant), dimensions={}
    multiply = f32[1,28,28,64]{3,2,1,0} multiply(add.33, broadcast.1)
    constant.37 = f32[] constant(127)
    broadcast.42 = f32[1,28,28,64]{3,2,1,0} broadcast(constant.37), dimensions={}
    clamp.43 = f32[1,28,28,64]{3,2,1,0} clamp(broadcast.41, multiply, broadcast.42)
    round-nearest-even.44 = f32[1,28,28,64]{3,2,1,0} round-nearest-even(clamp.43)
    convert.45 = s8[1,28,28,64]{3,2,1,0} convert(round-nearest-even.44)
    convert.54 = s32[1,28,28,64]{3,2,1,0} convert(convert.45)
    convert.56 = f32[1,28,28,64]{3,2,1,0} convert(convert.54)
    constant.49 = f32[] constant(0.1)
    broadcast.50 = f32[1,28,28,64]{3,2,1,0} broadcast(constant.49), dimensions={}
    multiply.57 = f32[1,28,28,64]{3,2,1,0} multiply(convert.56, broadcast.50)
    arg3.4 = s8[3,3,64,64]{3,2,1,0} parameter(3), parameter_replication={false}
    convert.65 = s32[3,3,64,64]{3,2,1,0} convert(arg3.4)
    convert.67 = f32[3,3,64,64]{3,2,1,0} convert(convert.65)
    broadcast.61 = f32[3,3,64,64]{3,2,1,0} broadcast(constant.22), dimensions={}
    multiply.68 = f32[3,3,64,64]{3,2,1,0} multiply(convert.67, broadcast.61)
    ROOT convolution.69 = f32[1,14,14,64]{3,2,1,0} convolution(multiply.57, multiply.68), window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
  })";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  std::unique_ptr<HloModule> optimized_module;
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_,
                    false, &optimized_module);
  std::vector<PrimitiveType> operand0_types = {S8, S8};
  std::vector<PrimitiveType> operand1_types = {S8, S8};
  std::vector<PrimitiveType> result_types = {S8, F32};                  
  CheckCustomCallTypesMultiInstrs(optimized_module, operand0_types, operand1_types, result_types);
}

TEST_F(ConvolutionTest, DequantizeConv2DBiasRequantizeQDQParamsMismatchedScale) {
  if (!IsSupportedType(PrimitiveType::S8)) {
    GTEST_SKIP() << "CPU does not support INT8.";
  }
  const char* convolution_module_str = R"(
  HloModule DequantizeConv2DBiasRequantize, alias_passthrough_params=true, entry_computation_layout={(s8[1,56,56,3]{3,2,1,0}, s8[3,3,3,64]{3,2,1,0}, f32[64]{0}, s8[3,3,64,64]{3,2,1,0})->f32[1,14,14,64]{3,2,1,0}}

  ENTRY DequantizeConv2DBiasRequantize {
    constant.36 = f32[] constant(-128)
    broadcast.41 = f32[1,28,28,64]{3,2,1,0} broadcast(constant.36), dimensions={}
    arg0.1 = s8[1,56,56,3]{3,2,1,0} parameter(0), parameter_replication={false}
    convert.16 = s32[1,56,56,3]{3,2,1,0} convert(arg0.1)
    constant.1 = s32[] constant(4)
    broadcast = s32[1,56,56,3]{3,2,1,0} broadcast(constant.1), dimensions={}
    add = s32[1,56,56,3]{3,2,1,0} add(convert.16, broadcast)
    convert.18 = f32[1,56,56,3]{3,2,1,0} convert(add)
    constant.11 = f32[] constant(0.5)
    broadcast.12 = f32[1,56,56,3]{3,2,1,0} broadcast(constant.11), dimensions={}
    multiply.19 = f32[1,56,56,3]{3,2,1,0} multiply(convert.18, broadcast.12)
    arg1.2 = s8[3,3,3,64]{3,2,1,0} parameter(1), parameter_replication={false}
    convert.27 = s32[3,3,3,64]{3,2,1,0} convert(arg1.2)
    convert.29 = f32[3,3,3,64]{3,2,1,0} convert(convert.27)
    constant.22 = f32[] constant(0.2)
    broadcast.23 = f32[3,3,3,64]{3,2,1,0} broadcast(constant.22), dimensions={}
    multiply.30 = f32[3,3,3,64]{3,2,1,0} multiply(convert.29, broadcast.23)
    convolution.31 = f32[1,28,28,64]{3,2,1,0} convolution(multiply.19, multiply.30), window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
    arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}
    broadcast.32 = f32[1,28,28,64]{3,2,1,0} broadcast(arg2.3), dimensions={3}
    add.33 = f32[1,28,28,64]{3,2,1,0} add(convolution.31, broadcast.32)
    constant = f32[] constant(10)
    broadcast.1 = f32[1,28,28,64]{3,2,1,0} broadcast(constant), dimensions={}
    multiply = f32[1,28,28,64]{3,2,1,0} multiply(add.33, broadcast.1)
    constant.37 = f32[] constant(127)
    broadcast.42 = f32[1,28,28,64]{3,2,1,0} broadcast(constant.37), dimensions={}
    clamp.43 = f32[1,28,28,64]{3,2,1,0} clamp(broadcast.41, multiply, broadcast.42)
    round-nearest-even.44 = f32[1,28,28,64]{3,2,1,0} round-nearest-even(clamp.43)
    convert.45 = s8[1,28,28,64]{3,2,1,0} convert(round-nearest-even.44)
    convert.54 = s32[1,28,28,64]{3,2,1,0} convert(convert.45)
    convert.56 = f32[1,28,28,64]{3,2,1,0} convert(convert.54)
    constant.49 = f32[] constant(0.2)
    broadcast.50 = f32[1,28,28,64]{3,2,1,0} broadcast(constant.49), dimensions={}
    multiply.57 = f32[1,28,28,64]{3,2,1,0} multiply(convert.56, broadcast.50)
    arg3.4 = s8[3,3,64,64]{3,2,1,0} parameter(3), parameter_replication={false}
    convert.65 = s32[3,3,64,64]{3,2,1,0} convert(arg3.4)
    convert.67 = f32[3,3,64,64]{3,2,1,0} convert(convert.65)
    broadcast.61 = f32[3,3,64,64]{3,2,1,0} broadcast(constant.22), dimensions={}
    multiply.68 = f32[3,3,64,64]{3,2,1,0} multiply(convert.67, broadcast.61)
    ROOT convolution.69 = f32[1,14,14,64]{3,2,1,0} convolution(multiply.57, multiply.68), window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
  })";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  std::unique_ptr<HloModule> optimized_module;
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_,
                    false, &optimized_module);
  std::vector<PrimitiveType> operand0_types = {S8, F32};
  std::vector<PrimitiveType> operand1_types = {S8, F32};
  std::vector<PrimitiveType> result_types = {F32, F32};                  
  CheckCustomCallTypesMultiInstrs(optimized_module, operand0_types, operand1_types, result_types);
}

TEST_F(ConvolutionTest, QuantizeDequantizeConv2DBiasRequantizeConstWeights) {
  const char* convolution_module_str = R"(
    HloModule QuantizeDequantizeConv2DBiasRequantizeConstWeights, alias_passthrough_params=true, entry_computation_layout={(f32[1,224,224,3]{3,2,1,0})->f32[1,56,56,64]{3,2,1,0}}

  ENTRY QuantizeDequantizeConv2DBiasRequantizeConstWeights {
    constant.34 = f32[] constant(-128)
    broadcast.39 = f32[1,112,112,64]{3,2,1,0} broadcast(constant.34), dimensions={}
    constant.5 = f32[] constant(-127)
    broadcast.10 = f32[1,224,224,3]{3,2,1,0} broadcast(constant.5), dimensions={}
    arg0.1 = f32[1,224,224,3]{3,2,1,0} parameter(0), parameter_replication={false}
    constant = f32[] constant(10)
    broadcast = f32[1,224,224,3]{3,2,1,0} broadcast(constant), dimensions={}
    multiply = f32[1,224,224,3]{3,2,1,0} multiply(arg0.1, broadcast)
    constant.6 = f32[] constant(128)
    broadcast.11 = f32[1,224,224,3]{3,2,1,0} broadcast(constant.6), dimensions={}
    clamp.12 = f32[1,224,224,3]{3,2,1,0} clamp(broadcast.10, multiply, broadcast.11)
    round-nearest-even.13 = f32[1,224,224,3]{3,2,1,0} round-nearest-even(clamp.12)
    convert.14 = s8[1,224,224,3]{3,2,1,0} convert(round-nearest-even.13)
    convert.23 = s32[1,224,224,3]{3,2,1,0} convert(convert.14)
    convert.25 = f32[1,224,224,3]{3,2,1,0} convert(convert.23)
    constant.18 = f32[] constant(0.1)
    broadcast.19 = f32[1,224,224,3]{3,2,1,0} broadcast(constant.18), dimensions={}
    multiply.26 = f32[1,224,224,3]{3,2,1,0} multiply(convert.25, broadcast.19)
    constant.27 = f32[7,7,3,64]{3,2,1,0} constant({...})
    convolution.28 = f32[1,112,112,64]{3,2,1,0} convolution(multiply.26, constant.27), window={size=7x7 stride=2x2 pad=2_3x2_3}, dim_labels=b01f_01io->b01f
    constant.29 = f32[64]{0} constant({...})
    broadcast.30 = f32[1,112,112,64]{3,2,1,0} broadcast(constant.29), dimensions={3}
    add.31 = f32[1,112,112,64]{3,2,1,0} add(convolution.28, broadcast.30)
    broadcast.1 = f32[1,112,112,64]{3,2,1,0} broadcast(constant), dimensions={}
    multiply.1 = f32[1,112,112,64]{3,2,1,0} multiply(add.31, broadcast.1)
    constant.35 = f32[] constant(127)
    broadcast.40 = f32[1,112,112,64]{3,2,1,0} broadcast(constant.35), dimensions={}
    clamp.41 = f32[1,112,112,64]{3,2,1,0} clamp(broadcast.39, multiply.1, broadcast.40)
    round-nearest-even.42 = f32[1,112,112,64]{3,2,1,0} round-nearest-even(clamp.41)
    convert.43 = s8[1,112,112,64]{3,2,1,0} convert(round-nearest-even.42)
    convert.52 = s32[1,112,112,64]{3,2,1,0} convert(convert.43)
    convert.54 = f32[1,112,112,64]{3,2,1,0} convert(convert.52)
    broadcast.48 = f32[1,112,112,64]{3,2,1,0} broadcast(constant.18), dimensions={}
    multiply.55 = f32[1,112,112,64]{3,2,1,0} multiply(convert.54, broadcast.48)
    constant.56 = f32[3,3,64,64]{3,2,1,0} constant({...})
    ROOT convolution.57 = f32[1,56,56,64]{3,2,1,0} convolution(multiply.55, constant.56), window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  std::unique_ptr<HloModule> optimized_module;
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_,
                    false, &optimized_module);
  std::vector<PrimitiveType> operand0_types = {S8, S8};
  std::vector<PrimitiveType> operand1_types = {S8, S8};
  std::vector<PrimitiveType> result_types = {S8, F32};                  
  CheckCustomCallTypesMultiInstrs(optimized_module, operand0_types, operand1_types, result_types);
}
}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
