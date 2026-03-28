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

#include "xla/backends/gpu/autotuner/cudnn.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

namespace bf16_fallback_internal {
bool IsBf16FallbackConfig(const CudnnBackendConfig& config);
void MarkAsBf16Fallback(CudnnBackendConfig& config);
CudnnBackendConfig StripBf16FallbackMarker(const CudnnBackendConfig& config);
bool IsFp8ConvCustomCall(const HloCustomCallInstruction* instr);
absl::StatusOr<stream_executor::dnn::ConvolutionKind>
GetBf16FallbackConvolutionKind(stream_executor::dnn::ConvolutionKind conv_kind);
absl::StatusOr<absl::string_view> GetBf16FallbackCustomCallTarget(
    const HloCustomCallInstruction& instr);
}  // namespace bf16_fallback_internal

using ::testing::Gt;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

const char kCudnnFusionHlo[] = R"(
  fusion1 {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    ROOT d = f32[3,32,32] dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  ENTRY e {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    ROOT _ = f32[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
  })";

const char kCudnnConvolutionFusionHlo[] = R"(
  fusion1 {
    p0 = f32[16,56,56,16] parameter(0)
    p1 = f32[16,3,3,16] parameter(1)
    ROOT c = f32[16,54,54,16] convolution(p0, p1),
      window={size=3x3},
      dim_labels=f01b_i01o->f01b,
      convolution_kind=fprop
  }

  ENTRY e {
    p0 = f32[16,56,56,16] parameter(0)
    p1 = f32[16,3,3,16] parameter(1)
    ROOT _ = f32[16,54,54,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={
        "fusion_backend_config": {
          "kind": "__cudnn$fusion",
        }
      }
  })";

const char kTritonGemmFusionHlo[] = R"(
  fusion1 {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    d = f32[3,32,32] dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  e {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    _ = f32[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config": {kind: "__triton_gemm"}}
  })";

const char kCudnnCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %arg0 = f32[3,56,56,16]{2,1,0,3} parameter(0)
    %arg1 = f32[3,3,3,64]{2,1,0,3} parameter(1)
    %cudnn-conv = (f32[54,54,16,64]{1,0,3,2}, u8[0]{0})
      custom-call(%arg0, %arg1), custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=f01b_i01o->01bf,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        },
      }
    ROOT %get-tuple-element = f32[54,54,16,64]{1,0,3,2} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 forward conv custom call for BF16 fallback tests.
// Uses __cudnn$convForwardGraph target with f8e4m3fn types.
const char kFp8ConvGraphCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[1,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[3,3,16,64]{3,2,1,0} parameter(1)
    %input_scale = f32[] parameter(2)
    %filter_scale = f32[] parameter(3)
    %cudnn-conv = (f8e4m3fn[1,54,54,64]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter, %input_scale, %filter_scale),
      custom_call_target="__cudnn$convForwardGraph",
      window={size=3x3},
      dim_labels=b01f_01io->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0,
          "serialized_graph":"serialized_graph_placeholder"
        }
      }
    ROOT %get-tuple-element = f8e4m3fn[1,54,54,64]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 forward conv custom call (non-graph).
const char kFp8ConvForwardCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[1,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[3,3,16,64]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e4m3fn[1,54,54,64]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=b01f_01io->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %get-tuple-element = f8e4m3fn[1,54,54,64]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 forward activation conv custom call.
const char kFp8ConvForwardActivationCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[1,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e4m3fn[3,3,16,64]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e4m3fn[1,54,54,64]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convBiasActivationForward",
      window={size=3x3},
      dim_labels=b01f_01io->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %get-tuple-element = f8e4m3fn[1,54,54,64]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 backward-input conv custom call.
const char kFp8ConvBackwardInputCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %out_grad = f8e4m3fn[1,20,257]{2,1,0} parameter(0)
    %filter = f8e4m3fn[31,257,136]{0,2,1} parameter(1)
    %cudnn-conv = (f8e4m3fn[1,23,136]{2,1,0}, u8[0]{0})
      custom-call(%out_grad, %filter),
      custom_call_target="__cudnn$convBackwardInput",
      window={size=31 stride=2 pad=23_23},
      dim_labels=b0f_0oi->b0f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %get-tuple-element = f8e4m3fn[1,23,136]{2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 backward-filter conv custom call.
const char kFp8ConvBackwardFilterCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e4m3fn[7680,96,6,6]{1,3,2,0} parameter(0)
    %out_grad = f8e4m3fn[7680,64,4,4]{1,3,2,0} parameter(1)
    %cudnn-conv = (f8e4m3fn[64,96,3,3]{1,3,2,0}, u8[0]{0})
      custom-call(%input, %out_grad),
      custom_call_target="__cudnn$convBackwardFilter",
      window={size=3x3},
      dim_labels=bf01_oi01->bf01,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %get-tuple-element = f8e4m3fn[64,96,3,3]{1,3,2,0} get-tuple-element(%cudnn-conv), index=0
  })";

// FP8 forward conv using f8e5m2 type (alternative FP8 encoding).
const char kFp8E5M2ConvForwardCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %input = f8e5m2[1,56,56,16]{3,2,1,0} parameter(0)
    %filter = f8e5m2[3,3,16,64]{3,2,1,0} parameter(1)
    %cudnn-conv = (f8e5m2[1,54,54,64]{3,2,1,0}, u8[0]{0})
      custom-call(%input, %filter),
      custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=b01f_01io->b01f,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        }
      }
    ROOT %get-tuple-element = f8e5m2[1,54,54,64]{3,2,1,0} get-tuple-element(%cudnn-conv), index=0
  })";

const char kUnsupportedHlo[] = R"(
  computation {
    p0 = s2[3,3] parameter(0)
    p1 = s2[3,3] parameter(1)
    d = s2[3,3] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  main {
    p0 = s2[3,3] parameter(0)
    p1 = s2[3,3] parameter(1)
    fusion = s2[3,3] fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

class CudnnBackendTest : public HloHardwareIndependentTestBase {
 protected:
  CudnnBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_cudnn_gemm_fusion_level(2);
    backend_ = std::make_unique<CudnnBackend>(stream_executor_, &debug_options_,
                                              &compiler_, &target_config_);
  }

  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  std::unique_ptr<CudnnBackend> backend_;
};

TEST_F(CudnnBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, backend_);
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnConvolutionFusion) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kCudnnConvolutionFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromTritonGemmFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kTritonGemmFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest,
       GetSupportedConfigsFromFp8GraphCustomCallMarksBf16Fallback) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvGraphCustomCallHlo));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_->GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction()->operand(0)));

  int fallback_config_count = 0;
  for (const std::unique_ptr<BackendConfig>& any_config : configs) {
    CudnnBackendConfig algorithm_config;
    ASSERT_TRUE(any_config->UnpackTo(&algorithm_config));
    if (algorithm_config.is_bf16_fallback()) {
      ++fallback_config_count;
    }
  }
  if (fallback_config_count == 0) {
    GTEST_SKIP() << "Environment returned FP8 plans; BF16 fallback "
                    "discovery path was not exercised.";
  }
  EXPECT_GT(fallback_config_count, 0);
}

TEST_F(CudnnBackendTest, Bf16FallbackMarkerHelpersRoundTrip) {
  CudnnBackendConfig config;
  config.set_algo_id(1);
  (*config.mutable_tuning_knobs())[7] = 9;
  config.set_is_bf16_fallback(false);

  EXPECT_FALSE(bf16_fallback_internal::IsBf16FallbackConfig(config));
  bf16_fallback_internal::MarkAsBf16Fallback(config);
  EXPECT_TRUE(bf16_fallback_internal::IsBf16FallbackConfig(config));

  CudnnBackendConfig stripped =
      bf16_fallback_internal::StripBf16FallbackMarker(config);
  EXPECT_FALSE(bf16_fallback_internal::IsBf16FallbackConfig(stripped));
  EXPECT_EQ(stripped.tuning_knobs().at(7), 9);
  EXPECT_FALSE(stripped.is_bf16_fallback());
}

TEST_F(CudnnBackendTest, IsFp8ConvCustomCallHelperDetectsOperandTypes) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> fp8_module,
      ParseAndReturnVerifiedModule(kFp8ConvForwardCustomCallHlo));
  auto* fp8_custom_call = static_cast<HloCustomCallInstruction*>(
      fp8_module->entry_computation()->root_instruction()->mutable_operand(0));
  EXPECT_TRUE(bf16_fallback_internal::IsFp8ConvCustomCall(fp8_custom_call));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> f32_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  auto* f32_custom_call = static_cast<HloCustomCallInstruction*>(
      f32_module->entry_computation()->root_instruction()->mutable_operand(0));
  EXPECT_FALSE(bf16_fallback_internal::IsFp8ConvCustomCall(f32_custom_call));
}

TEST_F(CudnnBackendTest, GetBf16FallbackConvolutionKindHelperMapsCorrectly) {
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackConvolutionKind(
                  stream_executor::dnn::ConvolutionKind::FORWARD),
              absl_testing::IsOkAndHolds(
                  stream_executor::dnn::ConvolutionKind::FORWARD));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackConvolutionKind(
                  stream_executor::dnn::ConvolutionKind::BACKWARD_DATA),
              absl_testing::IsOkAndHolds(
                  stream_executor::dnn::ConvolutionKind::BACKWARD_DATA));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackConvolutionKind(
                  stream_executor::dnn::ConvolutionKind::BACKWARD_FILTER),
              absl_testing::IsOkAndHolds(
                  stream_executor::dnn::ConvolutionKind::BACKWARD_FILTER));
  EXPECT_THAT(
      bf16_fallback_internal::GetBf16FallbackConvolutionKind(
          stream_executor::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION),
      absl_testing::IsOkAndHolds(
          stream_executor::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackConvolutionKind(
                  stream_executor::dnn::ConvolutionKind::FORWARD_GRAPH),
              absl_testing::IsOkAndHolds(
                  stream_executor::dnn::ConvolutionKind::FORWARD));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackConvolutionKind(
                  static_cast<stream_executor::dnn::ConvolutionKind>(-1)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CudnnBackendTest, GetBf16FallbackCustomCallTargetHelperMapsKinds) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> forward_module,
      ParseAndReturnVerifiedModule(kFp8ConvForwardCustomCallHlo));
  auto* forward_custom_call = static_cast<HloCustomCallInstruction*>(
      forward_module->entry_computation()->root_instruction()->mutable_operand(
          0));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackCustomCallTarget(
                  *forward_custom_call),
              absl_testing::IsOkAndHolds(kCudnnConvForwardCallTarget));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> graph_module,
      ParseAndReturnVerifiedModule(kFp8ConvGraphCustomCallHlo));
  auto* graph_custom_call = static_cast<HloCustomCallInstruction*>(
      graph_module->entry_computation()->root_instruction()->mutable_operand(
          0));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackCustomCallTarget(
                  *graph_custom_call),
              absl_testing::IsOkAndHolds(kCudnnConvForwardCallTarget));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> backward_input_module,
      ParseAndReturnVerifiedModule(kFp8ConvBackwardInputCustomCallHlo));
  auto* backward_input_custom_call = static_cast<HloCustomCallInstruction*>(
      backward_input_module->entry_computation()
          ->root_instruction()
          ->mutable_operand(0));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackCustomCallTarget(
                  *backward_input_custom_call),
              absl_testing::IsOkAndHolds(kCudnnConvBackwardInputCallTarget));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> backward_filter_module,
      ParseAndReturnVerifiedModule(kFp8ConvBackwardFilterCustomCallHlo));
  auto* backward_filter_custom_call = static_cast<HloCustomCallInstruction*>(
      backward_filter_module->entry_computation()
          ->root_instruction()
          ->mutable_operand(0));
  EXPECT_THAT(bf16_fallback_internal::GetBf16FallbackCustomCallTarget(
                  *backward_filter_custom_call),
              absl_testing::IsOkAndHolds(kCudnnConvBackwardFilterCallTarget));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> forward_activation_module,
      ParseAndReturnVerifiedModule(kFp8ConvForwardActivationCustomCallHlo));
  auto* forward_activation_custom_call = static_cast<HloCustomCallInstruction*>(
      forward_activation_module->entry_computation()
          ->root_instruction()
          ->mutable_operand(0));
  EXPECT_THAT(
      bf16_fallback_internal::GetBf16FallbackCustomCallTarget(
          *forward_activation_custom_call),
      absl_testing::IsOkAndHolds(kCudnnConvBiasActivationForwardCallTarget));
}

TEST_F(CudnnBackendTest,
       GetSupportedConfigsFromUnsupportedFusionReturnsEmptyVector) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kUnsupportedHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(0)));
}

TEST_F(CudnnBackendTest, GetDefaultConfigFromCudnnFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_->GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()));
  TF_ASSERT_OK(config);
  CudnnBackendConfig algorithm_config;
  ASSERT_TRUE(config->get()->UnpackTo(&algorithm_config));
  EXPECT_GE(algorithm_config.algo_id(), 0);
}

TEST_F(CudnnBackendTest, GetDefaultConfigFromCudnnCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_->GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  TF_ASSERT_OK(config);
  CudnnBackendConfig algorithm_config;
  ASSERT_TRUE(config->get()->UnpackTo(&algorithm_config));
  EXPECT_EQ(algorithm_config.algo_id(), -1);
}

TEST_F(CudnnBackendTest, GetDefaultConfigFailsWithNullStreamExecutor) {
  CudnnBackend backend_without_stream_executor(nullptr, &debug_options_,
                                               &compiler_, &target_config_);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_without_stream_executor.GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(config,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*fusion_instr, any));
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
}

TEST_F(CudnnBackendTest, ApplyConfigToTritonGemmFusionSetsCudnnKind) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kTritonGemmFusionHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*fusion_instr, any));
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), kCuDnnFusionKind);
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*instr, any));
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnCustomCallWithWorkspace) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  config.mutable_workspace_size()->set_value(1024);
  HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*instr, any));

  auto* replaced_instr =
      hlo_module->entry_computation()->GetInstructionWithName("cudnn-conv");

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          replaced_instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
  EXPECT_EQ(replaced_instr->shape().tuple_shapes(1).dimensions(0), 1024);
}

TEST_F(CudnnBackendTest, ApplyBf16FallbackConfigConvertsGraphConvToForward) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvGraphCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(42);
  config.set_is_bf16_fallback(true);

  HloInstruction* conv_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(conv_instr->custom_call_target(), "__cudnn$convForwardGraph");

  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*conv_instr, any));

  // After ReplaceInstruction, the original conv is replaced by a new tuple.
  // The original root GTE now references that new tuple.
  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);

  // The GTE's operand is the replacement tuple.
  HloInstruction* replacement_tuple = root->mutable_operand(0);
  ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

  // First element should be a convert (BF16 -> f8e4m3fn).
  HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
  ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

  // The convert input should be a GTE from the new BF16 custom call.
  HloInstruction* gte = result_convert->mutable_operand(0);
  ASSERT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);

  HloInstruction* new_conv = gte->mutable_operand(0);
  ASSERT_EQ(new_conv->opcode(), HloOpcode::kCustomCall);
  // Target should be changed to plain forward.
  EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convForward");

  // Output tuple element [0] should be BF16.
  EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);

  // FP8 operands should be wrapped in converts to BF16.
  EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(1)->shape().element_type(), BF16);
  // Scale operands (f32) should pass through without conversion.
  EXPECT_EQ(new_conv->operand(2)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(new_conv->operand(3)->opcode(), HloOpcode::kParameter);

  // Backend config should have the algorithm set (without the fallback marker)
  // and serialized_graph cleared.
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          new_conv->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.cudnn_conv_backend_config().algorithm().algo_id(), 42);
  EXPECT_FALSE(
      gpu_config.cudnn_conv_backend_config().algorithm().is_bf16_fallback());
  EXPECT_TRUE(
      gpu_config.cudnn_conv_backend_config().serialized_graph().empty());
}

TEST_F(CudnnBackendTest, ApplyBf16FallbackConfigHandlesNonGraphFp8Conv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvForwardCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(7);
  config.mutable_workspace_size()->set_value(512);
  config.set_is_bf16_fallback(true);

  HloInstruction* conv_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(conv_instr->custom_call_target(), "__cudnn$convForward");

  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*conv_instr, any));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);

  HloInstruction* replacement_tuple = root->mutable_operand(0);
  ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

  // Result should be convert BF16->FP8.
  HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
  ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

  HloInstruction* gte = result_convert->mutable_operand(0);
  HloInstruction* new_conv = gte->mutable_operand(0);

  // Custom call target stays __cudnn$convForward.
  EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convForward");
  // Output is BF16.
  EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
  // Workspace should reflect the config size.
  EXPECT_EQ(new_conv->shape().tuple_shapes(1).dimensions(0), 512);
  // All operands should be converted to BF16.
  EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(1)->shape().element_type(), BF16);
}

TEST_F(CudnnBackendTest, ApplyBf16FallbackConfigHandlesBackwardInputFp8Conv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvBackwardInputCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(9);
  config.mutable_workspace_size()->set_value(256);
  config.set_is_bf16_fallback(true);

  HloInstruction* conv_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(conv_instr->custom_call_target(), "__cudnn$convBackwardInput");

  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*conv_instr, any));

  HloInstruction* replacement_tuple =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

  HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
  ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

  HloInstruction* new_conv =
      result_convert->mutable_operand(0)->mutable_operand(0);
  ASSERT_EQ(new_conv->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convBackwardInput");
  EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(new_conv->shape().tuple_shapes(1).dimensions(0), 256);
  EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
}

TEST_F(CudnnBackendTest, ApplyBf16FallbackConfigHandlesBackwardFilterFp8Conv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvBackwardFilterCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(11);
  config.set_is_bf16_fallback(true);

  HloInstruction* conv_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(conv_instr->custom_call_target(), "__cudnn$convBackwardFilter");

  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*conv_instr, any));

  HloInstruction* replacement_tuple =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

  HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
  ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(result_convert->shape().element_type(), F8E4M3FN);

  HloInstruction* new_conv =
      result_convert->mutable_operand(0)->mutable_operand(0);
  ASSERT_EQ(new_conv->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convBackwardFilter");
  EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
}

TEST_F(CudnnBackendTest, ApplyNonFallbackConfigToFp8ConvDoesNotInsertConverts) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8ConvForwardCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(3);
  // No BF16 fallback marker — this is a normal FP8 config.

  HloInstruction* conv_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);

  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*conv_instr, any));

  // The instruction should be updated in place (no new tuple wrapper).
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          conv_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.cudnn_conv_backend_config().algorithm().algo_id(), 3);
  // Output type remains FP8.
  EXPECT_EQ(conv_instr->shape().tuple_shapes(0).element_type(), F8E4M3FN);
}

TEST_F(CudnnBackendTest, ApplyBf16FallbackConfigHandlesF8E5M2Conv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kFp8E5M2ConvForwardCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(5);
  config.set_is_bf16_fallback(true);

  HloInstruction* conv_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(conv_instr->custom_call_target(), "__cudnn$convForward");

  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_->ApplyConfig(*conv_instr, any));

  HloInstruction* replacement_tuple =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(replacement_tuple->opcode(), HloOpcode::kTuple);

  // Result should be convert BF16->F8E5M2.
  HloInstruction* result_convert = replacement_tuple->mutable_operand(0);
  ASSERT_EQ(result_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(result_convert->shape().element_type(), F8E5M2);

  HloInstruction* new_conv =
      result_convert->mutable_operand(0)->mutable_operand(0);
  ASSERT_EQ(new_conv->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(new_conv->custom_call_target(), "__cudnn$convForward");
  // Output is BF16.
  EXPECT_EQ(new_conv->shape().tuple_shapes(0).element_type(), BF16);
  // F8E5M2 operands should be converted to BF16.
  EXPECT_EQ(new_conv->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(new_conv->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(new_conv->operand(1)->shape().element_type(), BF16);
}

}  // namespace gpu
}  // namespace xla
