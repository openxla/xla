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

#include "xla/backends/gpu/transforms/conv_fp8_fallback.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace se = ::stream_executor;

// FP8 ForwardGraph custom call (the most common case from
// CudnnFusedConvRewriter).
const char kFp8ConvForwardGraphHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[16,3,3,16]{3,2,1,0} parameter(1)
    %input_scale = f32[] parameter(2)
    %filter_scale = f32[] parameter(3)
    %cudnn-conv = (f8e4m3fn[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter, %input_scale, %filter_scale),
      custom_call_target="__cudnn$convForwardGraph",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0,
          "serialized_graph":"some_graph"
        }
      }
    ROOT %gte = f8e4m3fn[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// Plain FP8 Forward custom call (from ConvRewriter, not fused).
const char kFp8ConvForwardHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[16,3,3,16]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e4m3fn[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %gte = f8e4m3fn[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// Non-FP8 (F32) conv — should never be modified by the pass.
const char kF32ConvHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f32[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f32[16,3,3,16]{3,2,1,0} parameter(1)
    %cudnn-conv = (f32[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %gte = f32[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 BackwardInput custom call.
const char kFp8ConvBackwardInputHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[16,3,3,16]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e4m3fn[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convBackwardInput",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %gte = f8e4m3fn[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 BackwardFilter custom call.
const char kFp8ConvBackwardFilterHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[16,3,3,16]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e4m3fn[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convBackwardFilter",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %gte = f8e4m3fn[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 BiasActivation (fused) custom call.
const char kFp8ConvBiasActivationHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[16,3,3,16]{3,2,1,0} parameter(1)
    %bias = f32[16]{0} parameter(2)
    %cudnn-conv = (f8e4m3fn[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter, %bias),
      custom_call_target="__cudnn$convBiasActivationForward",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %gte = f8e4m3fn[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// F8E5M2 variant.
const char kFp8E5m2ConvForwardHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e5m2[16,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e5m2[16,3,3,16]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e5m2[16,54,54,16]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=b01f_i01o->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %gte = f8e5m2[16,54,54,16]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

class ConvFp8FallbackTest : public HloHardwareIndependentTestBase {
 protected:
  ConvFp8FallbackTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()) {}

  se::StreamExecutor* stream_executor_;
};

TEST_F(ConvFp8FallbackTest, NullStreamExecutorReturnsNoChange) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kFp8ConvForwardHlo));
  ConvFp8Fallback pass(/*stream_exec=*/nullptr);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConvFp8FallbackTest, F32ConvIsNotModified) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kF32ConvHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));
  EXPECT_FALSE(changed);

  // Verify the conv is unchanged.
  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
  HloInstruction* conv = root->mutable_operand(0);
  EXPECT_EQ(conv->custom_call_target(), "__cudnn$convForward");
  EXPECT_EQ(conv->shape().tuple_shapes(0).element_type(), F32);
}

// This test verifies the pass runs without error on FP8 ForwardGraph convs.
// Whether it rewrites depends on whether cuDNN supports FP8 for the given
// config on this hardware. Either outcome is valid.
TEST_F(ConvFp8FallbackTest, Fp8ForwardGraphConvRunsSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvForwardGraphHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));

  if (changed) {
    // If rewritten: verify BF16 conversion structure.
    HloInstruction* root = hlo_module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* replacement_tuple = root->mutable_operand(0);
    ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

    // First element should be convert(BF16→FP8).
    HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
    ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

    // The convert's operand should be GTE from BF16 custom call.
    HloInstruction* gte = result_convert->mutable_operand(0);
    ASSERT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);

    HloInstruction* new_conv = gte->mutable_operand(0);
    ASSERT_EQ(new_conv->opcode(), HloOpcode::kCustomCall);
    // ForwardGraph should become plain Forward.
    EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convForward");
    EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);

    // FP8 operands should be wrapped in converts to BF16.
    EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
    EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(new_conv->operand(1)->shape().element_type(), BF16);

    // Scale operands should be stripped (plain Forward doesn't use them).
    EXPECT_EQ(new_conv->operand_count(), 2);
  }
  // If not changed, the original HLO is preserved (cuDNN supports FP8).
}

TEST_F(ConvFp8FallbackTest, Fp8ForwardConvRunsSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kFp8ConvForwardHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));

  if (changed) {
    HloInstruction* root = hlo_module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* replacement_tuple = root->mutable_operand(0);
    ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

    HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
    ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

    HloInstruction* new_conv =
        result_convert->mutable_operand(0)->mutable_operand(0);
    EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convForward");
    EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
    EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
  }
}

TEST_F(ConvFp8FallbackTest, Fp8BackwardInputConvRunsSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvBackwardInputHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));

  if (changed) {
    HloInstruction* root = hlo_module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* replacement_tuple = root->mutable_operand(0);
    ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

    HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
    ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

    HloInstruction* new_conv =
        result_convert->mutable_operand(0)->mutable_operand(0);
    EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convBackwardInput");
    EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
  }
}

TEST_F(ConvFp8FallbackTest, Fp8BackwardFilterConvRunsSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvBackwardFilterHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));

  if (changed) {
    HloInstruction* root = hlo_module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* replacement_tuple = root->mutable_operand(0);
    ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

    HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
    ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

    HloInstruction* new_conv =
        result_convert->mutable_operand(0)->mutable_operand(0);
    EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convBackwardFilter");
    EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
  }
}

TEST_F(ConvFp8FallbackTest, Fp8BiasActivationConvRunsSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvBiasActivationHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));

  if (changed) {
    HloInstruction* root = hlo_module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* replacement_tuple = root->mutable_operand(0);
    ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

    HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
    ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

    HloInstruction* new_conv =
        result_convert->mutable_operand(0)->mutable_operand(0);
    // BiasActivation stays BiasActivation.
    EXPECT_EQ(new_conv->custom_call_target(),
              "__cudnn$convBiasActivationForward");
    EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
    // FP8 operands should be converted to BF16.
    EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
    EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(new_conv->operand(1)->shape().element_type(), BF16);
    // Bias (f32) should pass through without conversion.
    EXPECT_EQ(new_conv->operand(2)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(ConvFp8FallbackTest, Fp8E5m2ConvRunsSuccessfully) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kFp8E5m2ConvForwardHlo));
  ConvFp8Fallback pass(stream_executor_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(hlo_module.get()));

  if (changed) {
    HloInstruction* root = hlo_module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* replacement_tuple = root->mutable_operand(0);
    ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

    HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
    ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
    // Should convert back to the original F8E5M2 type.
    EXPECT_EQ(result_convert->shape().element_type(), F8E5M2);

    HloInstruction* new_conv =
        result_convert->mutable_operand(0)->mutable_operand(0);
    EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convForward");
    EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
    EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
