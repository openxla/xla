/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/cudnn_fused_mha_rewriter.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/computation_layout.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cudnn_fused_mha_transpose_fusion.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = xla::match;

class CudnnFusedMhaRewriterTestHloTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    // Fake a supported compute capability to run tests,
    // we don't run any kernels in these tests so they should be safe
    // to run anywhere.
    return se::CudaComputeCapability(8, 0);
  }

  se::dnn::VersionInfo GetCudnnVersion() {
    // Fake a supported compute capability to run tests,
    // we don't run any kernels in these tests so they should be safe
    // to run anywhere.
    return se::dnn::VersionInfo(8, 8, 0);
  }

  se::dnn::VersionInfo GetCudnnVersionWithDbiasAndMaskBwdInputSupport() {
    // Fake a supported compute capability to run tests for training with dbias
    // and mask bwd input support, we don't run any kernels in these tests so
    // they should be safe to run anywhere.
    return se::dnn::VersionInfo(8, 9, 1);
  }

  CudnnFusedMhaRewriterTestHloTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    /*instruction_can_change_layout_func=*/{}) {}

 protected:
  size_t CountFusedAttentionCall(HloModule* module, bool is_backward = false) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [&](const HloInstruction* instr) {
                              if (is_backward) {
                                return IsBwdCustomCallTofMHA(*instr);
                              } else {
                                return IsFwdCustomCallTofMHA(*instr);
                              }
                            });
  }

  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    debug_options.set_xla_gpu_enable_cudnn_fmha(true);
    debug_options.set_xla_gpu_fused_attention_use_cudnn_rng(true);
    return debug_options;
  }

  HloModuleConfig GetModuleConfig() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    HloModuleConfig config_with_fmha;
    config_with_fmha.set_debug_options(debug_options);
    return config_with_fmha;
  }
};

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1Bmm2Pattern) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}


)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));
  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1Bmm2UncanonicalizedPattern) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
}


)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::GetTupleElement(
                      m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                      .WithShape(BF16, {16, 16, 256, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::GetTupleElement(
                      m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                      .WithShape(BF16, {16, 16, 256, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_rhs_contracting_dim_not_most_minor) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_lhs_contracting_dim_not_most_minor) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{2,3,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            2);
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            2);
  EXPECT_EQ(config.bmm1_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            2);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm2_non_contracting_dim_not_most_minor) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{2,3,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{2,3,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{2,3,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&fusedMhaRewriter, m.get()));
  EXPECT_TRUE(result);
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            3);
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            3);
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().lhs_contracting_dimensions()[0],
            3);
  EXPECT_EQ(config.bmm2_dot_dimension_numbers().rhs_contracting_dimensions()[0],
            3);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1Bmm2Pattern) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0},f16[16,16,256,64]{3,2,1,0})->f16[16,16,256,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = f16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = f16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = f16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = f16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = f16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}


)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(F16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.fmha_scale(), 1.0);
  EXPECT_EQ(config.dropout_rate(), 0.0);
#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHABmmBmmCallTarget}), 0)
                     .WithShape(F16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 1.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1ScaleMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit_bmm_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.14.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.26 {
  Arg_0.27 = f32[] parameter(0)
  Arg_1.28 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.27, Arg_1.28)
}

ENTRY main.38 {
  constant.10 = pred[16,16,256,256]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.33 = f32[16,16,256,256]{3,2,1,0} convert(dot.11)
  constant.6 = f32[] constant(2.1)
  broadcast.7 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
  multiply.12 = f32[16,16,256,256]{3,2,1,0} multiply(convert.33, broadcast.7)
  convert.34 = bf16[16,16,256,256]{3,2,1,0} convert(multiply.12)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
  select.13 = bf16[16,16,256,256]{3,2,1,0} select(constant.10, convert.34, broadcast.5)
  convert.36 = f32[16,16,256,256]{3,2,1,0} convert(select.13)
  constant.9 = f32[] constant(-inf)
  reduce.18 = f32[16,16,256]{2,1,0} reduce(convert.36, constant.9), dimensions={3}, to_apply=region_0.14.clone
  broadcast.22 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract.23 = f32[16,16,256,256]{3,2,1,0} subtract(convert.36, broadcast.22)
  exponential.24 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.23)
  constant.8 = f32[] constant(0)
  reduce.30 = f32[16,16,256]{2,1,0} reduce(exponential.24, constant.8), dimensions={3}, to_apply=region_1.26
  broadcast.35 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.30), dimensions={0,1,2}
  divide.36 = f32[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
  convert.49 = bf16[16,16,256,256]{3,2,1,0} convert(divide.36)
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(convert.49, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleMaskSoftmaxCallTarget}), 0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 4);

#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleMaskSoftmaxCallTarget}), 0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 4);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit_bmm_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.17.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.29 {
  Arg_0.30 = f32[] parameter(0)
  Arg_1.31 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.30, Arg_1.31)
}

ENTRY main.41 {
  constant.10 = pred[16,16,256,256]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.33 = f32[16,16,256,256]{3,2,1,0} convert(dot.11)
  constant.6 = f32[] constant(3.1)
  constant.11 = f32[] constant(1)
  broadcast.7 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
  multiply.12 = f32[16,16,256,256]{3,2,1,0} multiply(convert.33, broadcast.7)
  broadcast.11 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.11), dimensions={}
  add.15 = f32[16,16,256,256]{3,2,1,0} add(multiply.12, broadcast.11)
  convert.40 = bf16[16,16,256,256]{3,2,1,0} convert(add.15)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
  select.13 = bf16[16,16,256,256]{3,2,1,0} select(constant.10, convert.40, broadcast.5)
  convert.36 = f32[16,16,256,256]{3,2,1,0} convert(select.13)
  constant.9 = f32[] constant(-inf)
  reduce.18 = f32[16,16,256]{2,1,0} reduce(convert.36, constant.9), dimensions={3}, to_apply=region_0.17.clone
  broadcast.22 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract.23 = f32[16,16,256,256]{3,2,1,0} subtract(convert.36, broadcast.22)
  exponential.24 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.23)
  constant.8 = f32[] constant(0)
  reduce.30 = f32[16,16,256]{2,1,0} reduce(exponential.24, constant.8), dimensions={3}, to_apply=region_1.29
  broadcast.35 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.30), dimensions={0,1,2}
  divide.36 = f32[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
  convert.49 = bf16[16,16,256,256]{3,2,1,0} convert(divide.36)
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(convert.49, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);

#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasNonConstantMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit_bmm_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}

region_0.17.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.29 {
  Arg_0.30 = f32[] parameter(0)
  Arg_1.31 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.30, Arg_1.31)
}

ENTRY main.41 {
  constant.10 = pred[16,16,256,256]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.11 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.33 = f32[16,16,256,256]{3,2,1,0} convert(dot.11)
  constant.6 = f32[] constant(3.1)
  constant.11 = f32[] constant(1)
  broadcast.7 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.6), dimensions={}
  multiply.12 = f32[16,16,256,256]{3,2,1,0} multiply(convert.33, broadcast.7)
  broadcast.11 = f32[16,16,256,256]{3,2,1,0} broadcast(constant.11), dimensions={}
  add.15 = f32[16,16,256,256]{3,2,1,0} add(multiply.12, broadcast.11)
  convert.40 = bf16[16,16,256,256]{3,2,1,0} convert(add.15)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.4), dimensions={}
  compare = pred[16,16,256,256]{3,2,1,0} compare(convert.40, broadcast.5), direction=GT 
  select.13 = bf16[16,16,256,256]{3,2,1,0} select(compare, convert.40, broadcast.5)
  convert.36 = f32[16,16,256,256]{3,2,1,0} convert(select.13)
  constant.9 = f32[] constant(-inf)
  reduce.18 = f32[16,16,256]{2,1,0} reduce(convert.36, constant.9), dimensions={3}, to_apply=region_0.17.clone
  broadcast.22 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.18), dimensions={0,1,2}
  subtract.23 = f32[16,16,256,256]{3,2,1,0} subtract(convert.36, broadcast.22)
  exponential.24 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.23)
  constant.8 = f32[] constant(0)
  reduce.30 = f32[16,16,256]{2,1,0} reduce(exponential.24, constant.8), dimensions={3}, to_apply=region_1.29
  broadcast.35 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.30), dimensions={0,1,2}
  divide.36 = f32[16,16,256,256]{3,2,1,0} divide(exponential.24, broadcast.35)
  convert.49 = bf16[16,16,256,256]{3,2,1,0} convert(divide.36)
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  ROOT dot.37 = bf16[16,16,256,64]{3,2,1,0} dot(convert.49, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);

#if GOOGLE_CUDA && CUDNN_VERSION >= 8800
  // run whole pipeline
  TF_ASSERT_OK_AND_ASSIGN(
      m, ParseAndReturnVerifiedModule(module_str, GetModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(m)));

  SCOPED_TRACE(optimized_module->ToString());
  EXPECT_THAT(
      optimized_module->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(BF16, {16, 16, 256, 64})));
  TF_ASSERT_OK_AND_ASSIGN(config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 3.1);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
#endif  // GOOGLE_CUDA && CUDNN_VERSION >= 8800
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1CombinedMaskBiasSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_,
entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.32.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}

ENTRY main.61 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  transpose.6 = bf16[16,16,256,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  transpose.7 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  bitcast.35 = pred[16,256,256]{2,1,0} bitcast(Arg_4.5)
  convert.49 = s32[16,256,256]{2,1,0} convert(bitcast.35)
  constant.5 = s32[] constant(0)
  broadcast.10 = s32[16,256,256]{2,1,0} broadcast(constant.5), dimensions={}
  compare = pred[16,256,256]{2,1,0} compare(convert.49, broadcast.10), direction=GT
  constant.7 = bf16[] constant(0)
  broadcast.12 = bf16[16,256,256]{2,1,0} broadcast(constant.7), dimensions={}
  constant.9 = bf16[] constant(-9.999e+09)
  broadcast.13 = bf16[16,256,256]{2,1,0} broadcast(constant.9), dimensions={}
  select = bf16[16,256,256]{2,1,0} select(compare, broadcast.12, broadcast.13)
  convert.51 = f32[16,256,256]{2,1,0} convert(select)
  broadcast.14 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.51), dimensions={0,2,3}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  bitcast.52 = bf16[16,256,256]{2,1,0} bitcast(Arg_3.4)
  convert.52 = f32[16,256,256]{2,1,0} convert(bitcast.52)
  broadcast.15 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.52), dimensions={1,2,3}
  add.1 = f32[16,16,256,256]{3,2,1,0} add(broadcast.14, broadcast.15)
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.6, transpose.7), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  convert.55 = f32[16,16,256,256]{3,2,1,0} convert(dot.2)
  add.18 = f32[16,16,256,256]{3,2,1,0} add(convert.55, add.1)
  constant.11 = f32[] constant(-inf)
  reduce.36 = f32[16,16,256]{2,1,0} reduce(add.18, constant.11), dimensions={3}, to_apply=region_0.32.clone
  broadcast.17 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = f32[16,16,256,256]{3,2,1,0} subtract(add.18, broadcast.17)
  exponential.1 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[16,16,256]{2,1,0} reduce(exponential.1, constant.14), dimensions={3}, to_apply=region_1.44
  broadcast.18 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.48), dimensions={0,1,2}
  divide = f32[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.18)
  convert.68 = bf16[16,16,256,256]{3,2,1,0} convert(divide)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, convert.68), lhs_contracting_dims={3}, rhs_contracting_dims={3}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT transpose.8 = bf16[16,256,16,64]{3,2,1,0} transpose(dot.1), dimensions={0,3,1,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}),
                  0)))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16Bmm1ScaleBiasMaskSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0})->f16[2,6,40,64]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

region_0.34 {
  Arg_0.35 = f16[] parameter(0)
  Arg_1.36 = f16[] parameter(1)
  ROOT maximum.1 = f16[] maximum(Arg_0.35, Arg_1.36)
}

region_1.46 {
  Arg_0.47 = f32[] parameter(0)
  Arg_1.48 = f32[] parameter(1)
  ROOT add.2 = f32[] add(Arg_0.47, Arg_1.48)
}

ENTRY main.83 {
  constant.5 = u32[1]{0} constant({2718843009})
  constant.7 = u32[1]{0} constant({1272950319})
  constant.9 = u32[1]{0} constant({0})
  constant.11 = u32[1]{0} constant({2711844646})
  custom-call.59 = (u32[1]{0}, u32[1]{0}) custom-call(constant.5, constant.7, constant.9, constant.11), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.60 = u32[1]{0} get-tuple-element(custom-call.59), index=0
  bitcast.112 = u32[] bitcast(get-tuple-element.60)
  broadcast.14 = u32[9600]{0} broadcast(bitcast.112), dimensions={}
  get-tuple-element.61 = u32[1]{0} get-tuple-element(custom-call.59), index=1
  bitcast.113 = u32[] bitcast(get-tuple-element.61)
  broadcast.16 = u32[9600]{0} broadcast(bitcast.113), dimensions={}
  iota.62 = u32[19200]{0} iota(), iota_dimension=0
  slice = u32[9600]{0} slice(iota.62), slice={[0:9600]}
  slice.1 = u32[9600]{0} slice(iota.62), slice={[9600:19200]}
  custom-call.69 = (u32[9600]{0}, u32[9600]{0}) custom-call(broadcast.14, broadcast.16, slice, slice.1), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[9600]{0}, u32[9600]{0}, u32[9600]{0}, u32[9600]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\200%\000\000\000\000\000\000"
  get-tuple-element.70 = u32[9600]{0} get-tuple-element(custom-call.69), index=0
  get-tuple-element.71 = u32[9600]{0} get-tuple-element(custom-call.69), index=1
  concatenate = u32[19200]{0} concatenate(get-tuple-element.70, get-tuple-element.71), dimensions={0}
  constant.13 = u32[] constant(9)
  broadcast.18 = u32[19200]{0} broadcast(constant.13), dimensions={}
  shift-right-logical.1 = u32[19200]{0} shift-right-logical(concatenate, broadcast.18)
  constant.15 = u32[] constant(1065353216)
  broadcast.19 = u32[19200]{0} broadcast(constant.15), dimensions={}
  or.1 = u32[19200]{0} or(shift-right-logical.1, broadcast.19)
  bitcast-convert.1 = f32[19200]{0} bitcast-convert(or.1)
  constant.17 = f32[] constant(-1)
  broadcast.20 = f32[19200]{0} broadcast(constant.17), dimensions={}
  add.3 = f32[19200]{0} add(bitcast-convert.1, broadcast.20)
  constant.39 = f32[] constant(0)
  broadcast.21 = f32[19200]{0} broadcast(constant.39), dimensions={}
  maximum.2 = f32[19200]{0} maximum(add.3, broadcast.21)
  constant.28 = f32[] constant(0.8)
  broadcast.23 = f32[19200]{0} broadcast(constant.28), dimensions={}
  compare.1 = pred[19200]{0} compare(maximum.2, broadcast.23), direction=LT
  bitcast.114 = pred[2,6,40,40]{3,2,1,0} bitcast(compare.1)
  constant.34 = pred[2,6,40,40]{3,2,1,0} constant({...})
  Arg_0.1 = f16[2,6,40,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,40]{3,2,1,0} parameter(1), sharding={replicated}
  dot.30 = f16[2,6,40,40]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.35 = f16[] constant(2)
  broadcast.27 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.35), dimensions={}
  multiply.2 = f16[2,6,40,40]{3,2,1,0} multiply(dot.30, broadcast.27)
  constant.36 = f16[] constant(1)
  broadcast.29 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.36), dimensions={}
  add.5 = f16[2,6,40,40]{3,2,1,0} add(multiply.2, broadcast.29)
  constant.37 = f16[] constant(0)
  broadcast.30 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.37), dimensions={}
  select.1 = f16[2,6,40,40]{3,2,1,0} select(constant.34, add.5, broadcast.30)
  constant.38 = f16[] constant(-inf)
  reduce.38 = f16[2,6,40]{2,1,0} reduce(select.1, constant.38), dimensions={3}, to_apply=region_0.34
  broadcast.32 = f16[2,6,40,40]{3,2,1,0} broadcast(reduce.38), dimensions={0,1,2}
  subtract.1 = f16[2,6,40,40]{3,2,1,0} subtract(select.1, broadcast.32)
  exponential.1 = f16[2,6,40,40]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[2,6,40,40]{3,2,1,0} convert(exponential.1)
  reduce.50 = f32[2,6,40]{2,1,0} reduce(convert.1, constant.39), dimensions={3}, to_apply=region_1.46
  convert.2 = f16[2,6,40]{2,1,0} convert(reduce.50)
  broadcast.33 = f16[2,6,40,40]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = f16[2,6,40,40]{3,2,1,0} divide(exponential.1, broadcast.33)
  constant.40 = f16[] constant(1.25)
  broadcast.34 = f16[2,6,40,40]{3,2,1,0} broadcast(constant.40), dimensions={}
  multiply.3 = f16[2,6,40,40]{3,2,1,0} multiply(divide, broadcast.34)
  select.2 = f16[2,6,40,40]{3,2,1,0} select(bitcast.114, multiply.3, broadcast.30)
  Arg_2.3 = f16[2,6,40,64]{3,2,1,0} parameter(2), sharding={replicated}
  ROOT dot.82 = f16[2,6,40,64]{3,2,1,0} dot(select.2, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha,
                            {kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget}),
              0)
              .WithShape(F16, {2, 6, 40, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2);
  EXPECT_NEAR(config.dropout_rate(), 0.2, 1e-2);
  EXPECT_EQ(fmha->operands().size(), 5);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, F16Bmm1UnfusedSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,40,64]{3,2,1,0},f16[2,6,64,40]{3,2,1,0},f16[2,6,40,64]{3,2,1,0})->f16[2,6,40,64]{3,2,1,0}}

region_0.7 {
  Arg_0.8 = f16[] parameter(0)
  Arg_1.9 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(Arg_0.8, Arg_1.9)
}

region_1.19 {
  Arg_0.20 = f32[] parameter(0)
  Arg_1.21 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.20, Arg_1.21)
}

ENTRY main.31 {
  Arg_0.1 = f16[2,6,40,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,40]{3,2,1,0} parameter(1), sharding={replicated}
  dot = f16[2,6,40,40]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  constant = f16[] constant(-inf)
  reduce.11 = f16[2,6,40]{2,1,0} reduce(dot, constant), dimensions={3}, to_apply=region_0.7
  broadcast.3 = f16[2,6,40,40]{3,2,1,0} broadcast(reduce.11), dimensions={0,1,2}
  subtract.1 = f16[2,6,40,40]{3,2,1,0} subtract(dot, broadcast.3)
  exponential.1 = f16[2,6,40,40]{3,2,1,0} exponential(subtract.1)
  convert.1 = f32[2,6,40,40]{3,2,1,0} convert(exponential.1)
  constant.1 = f32[] constant(0)
  reduce.23 = f32[2,6,40]{2,1,0} reduce(convert.1, constant.1), dimensions={3}, to_apply=region_1.19
  convert.2 = f16[2,6,40]{2,1,0} convert(reduce.23)
  broadcast.4 = f16[2,6,40,40]{3,2,1,0} broadcast(convert.2), dimensions={0,1,2}
  divide = f16[2,6,40,40]{3,2,1,0} divide(exponential.1, broadcast.4)
  Arg_2.3 = f16[2,6,40,64]{3,2,1,0} parameter(2), sharding={replicated}
  ROOT dot.1 = f16[2,6,40,64]{3,2,1,0} dot(divide, Arg_2.3), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
                     .WithShape(F16, {2, 6, 40, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 1.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 3);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16Bmm1UnfusedSoftmaxWithConvertF32ToReduceMaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[128,6,400,64]{3,2,1,0},f16[128,6,64,400]{3,2,1,0},f16[128,6,400,64]{3,2,1,0})->f16[128,6,400,64]{3,2,1,0}}

region_0.18 {
  Arg_0.19 = f32[] parameter(0)
  Arg_1.20 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(Arg_0.19, Arg_1.20)
}

region_1.29 {
  Arg_0.30 = f32[] parameter(0)
  Arg_1.31 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.30, Arg_1.31)
}

ENTRY main.41 {
  constant.3 = pred[128,6,400,400]{3,2,1,0} constant({...})
  Arg_0.1 = f16[128,6,400,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[128,6,64,400]{3,2,1,0} parameter(1), sharding={replicated}
  constant.1 = f16[] constant(1)
  broadcast.2 = f16[128,6,400,400]{3,2,1,0} broadcast(constant.1), dimensions={}
  constant.50 = f16[] constant(2)
  broadcast.100 = f16[128,6,400,400]{3,2,1,0} broadcast(constant.50), dimensions={}
  dot = f16[128,6,400,400]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  multiply.100 = f16[128,6,400,400]{3,2,1,0} multiply(dot, broadcast.100)
  add.1 = f16[128,6,400,400]{3,2,1,0} add(multiply.100, broadcast.2)
  constant.5 = f16[] constant(0)
  broadcast.4 = f16[128,6,400,400]{3,2,1,0} broadcast(constant.5), dimensions={}
  select.1 = f16[128,6,400,400]{3,2,1,0} select(constant.3, add.1, broadcast.4)
  convert.1 = f32[128,6,400,400]{3,2,1,0} convert(select.1)
  constant.7 = f32[] constant(-inf)
  reduce.22 = f32[128,6,400]{2,1,0} reduce(convert.1, constant.7), dimensions={3}, to_apply=region_0.18
  broadcast.8 = f32[128,6,400,400]{3,2,1,0} broadcast(reduce.22), dimensions={0,1,2}
  subtract.1 = f32[128,6,400,400]{3,2,1,0} subtract(convert.1, broadcast.8)
  exponential.1 = f32[128,6,400,400]{3,2,1,0} exponential(subtract.1)
  constant.11 = f32[] constant(0)
  reduce.33 = f32[128,6,400]{2,1,0} reduce(exponential.1, constant.11), dimensions={3}, to_apply=region_1.29
  broadcast.9 = f32[128,6,400,400]{3,2,1,0} broadcast(reduce.33), dimensions={0,1,2}
  divide = f32[128,6,400,400]{3,2,1,0} divide(exponential.1, broadcast.9)
  convert.2 = f16[128,6,400,400]{3,2,1,0} convert(divide)
  Arg_2.3 = f16[128,6,400,64]{3,2,1,0} parameter(2), sharding={replicated}
  ROOT dot.1 = f16[128,6,400,64]{3,2,1,0} dot(convert.2, Arg_2.3), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(F16, {128, 6, 400, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.0);
  EXPECT_FLOAT_EQ(config.dropout_rate(), 0.0);
  EXPECT_EQ(fmha->operands().size(), 5);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1UnfusedScaleMaskBiasSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.32.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}

ENTRY main.61 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  transpose.6 = bf16[16,16,256,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  transpose.7 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  bitcast.35 = pred[16,256,256]{2,1,0} bitcast(Arg_4.5)
  convert.49 = s32[16,256,256]{2,1,0} convert(bitcast.35)
  constant.5 = s32[] constant(0)
  broadcast.10 = s32[16,256,256]{2,1,0} broadcast(constant.5), dimensions={}
  constant.50 = bf16[] constant(2)
  broadcast.100 = bf16[16,16,256,256]{3,2,1,0} broadcast(constant.50), dimensions={}
  compare = pred[16,256,256]{2,1,0} compare(convert.49, broadcast.10), direction=GT
  constant.7 = bf16[] constant(0)
  broadcast.12 = bf16[16,256,256]{2,1,0} broadcast(constant.7), dimensions={}
  constant.9 = bf16[] constant(-9.999e+09)
  broadcast.13 = bf16[16,256,256]{2,1,0} broadcast(constant.9), dimensions={}
  select = bf16[16,256,256]{2,1,0} select(compare, broadcast.12, broadcast.13)
  convert.51 = f32[16,256,256]{2,1,0} convert(select)
  broadcast.14 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.51), dimensions={0,2,3}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  bitcast.52 = bf16[16,256,256]{2,1,0} bitcast(Arg_3.4)
  convert.52 = f32[16,256,256]{2,1,0} convert(bitcast.52)
  broadcast.15 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.52), dimensions={1,2,3}
  add.1 = f32[16,16,256,256]{3,2,1,0} add(broadcast.14, broadcast.15)
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose.6, transpose.7), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  multiply.100 = bf16[16,16,256,256]{3,2,1,0} multiply(dot, broadcast.100)
  convert.55 = f32[16,16,256,256]{3,2,1,0} convert(multiply.100)
  add.10 = f32[16,16,256,256]{3,2,1,0} add(convert.55, add.1)
  constant.11 = f32[] constant(-inf)
  reduce.36 = f32[16,16,256]{2,1,0} reduce(add.10, constant.11), dimensions={3}, to_apply=region_0.32.clone
  broadcast.17 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = f32[16,16,256,256]{3,2,1,0} subtract(add.10, broadcast.17)
  exponential.1 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[16,16,256]{2,1,0} reduce(exponential.1, constant.14), dimensions={3}, to_apply=region_1.44
  broadcast.18 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.48), dimensions={0,1,2}
  divide = f32[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.18)
  convert.68 = bf16[16,16,256,256]{3,2,1,0} convert(divide)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, convert.68), lhs_contracting_dims={3}, rhs_contracting_dims={3}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT transpose.8 = bf16[16,256,16,64]{3,2,1,0} transpose(dot.1), dimensions={0,3,1,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}),
                  0)))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
  EXPECT_FLOAT_EQ(config.fmha_scale(), 2.0);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ConvertedMaskAddedAfterFirstGemmSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},pred[16,1,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.27.clone {
  Arg_0.0 = f32[] parameter(0)
  Arg_1.0 = f32[] parameter(1)
  ROOT maximum.1 = f32[] maximum(Arg_0.0, Arg_1.0)
}

region_1.39 {
  Arg_0.40 = f32[] parameter(0)
  Arg_1.41 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.40, Arg_1.41)
}

ENTRY main.56 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_2.3), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  transpose.6 = bf16[16,16,256,64]{3,2,1,0} transpose(Arg_0.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  transpose.7 = bf16[16,16,64,256]{3,2,1,0} transpose(Arg_1.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose.6, transpose.7), lhs_contracting_dims={3}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  convert.47 = f32[16,16,256,256]{3,2,1,0} convert(dot)
  Arg_3.4 = pred[16,1,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  bitcast.37 = pred[16,256,256]{2,1,0} bitcast(Arg_3.4)
  convert.42 = s32[16,256,256]{2,1,0} convert(bitcast.37)
  constant.6 = s32[] constant(0)
  broadcast.9 = s32[16,256,256]{2,1,0} broadcast(constant.6), dimensions={}
  compare = pred[16,256,256]{2,1,0} compare(convert.42, broadcast.9), direction=GT
  constant.8 = bf16[] constant(0)
  broadcast.11 = bf16[16,256,256]{2,1,0} broadcast(constant.8), dimensions={}
  constant.10 = bf16[] constant(-9.999e+09)
  broadcast.12 = bf16[16,256,256]{2,1,0} broadcast(constant.10), dimensions={}
  select = bf16[16,256,256]{2,1,0} select(compare, broadcast.11, broadcast.12)
  convert.48 = f32[16,256,256]{2,1,0} convert(select)
  broadcast.14 = f32[16,16,256,256]{3,2,1,0} broadcast(convert.48), dimensions={0,2,3}
  add.2 = f32[16,16,256,256]{3,2,1,0} add(convert.47, broadcast.14)
  constant.13 = f32[] constant(-inf)
  reduce.31 = f32[16,16,256]{2,1,0} reduce(add.2, constant.13), dimensions={3}, to_apply=region_0.27.clone
  broadcast.16 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.31), dimensions={0,1,2}
  subtract.1 = f32[16,16,256,256]{3,2,1,0} subtract(add.2, broadcast.16)
  exponential.1 = f32[16,16,256,256]{3,2,1,0} exponential(subtract.1)
  constant.14 = f32[] constant(0)
  reduce.43 = f32[16,16,256]{2,1,0} reduce(exponential.1, constant.14), dimensions={3}, to_apply=region_1.39
  broadcast.17 = f32[16,16,256,256]{3,2,1,0} broadcast(reduce.43), dimensions={0,1,2}
  divide = f32[16,16,256,256]{3,2,1,0} divide(exponential.1, broadcast.17)
  convert.63 = bf16[16,16,256,256]{3,2,1,0} convert(divide)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, convert.63), lhs_contracting_dims={3}, rhs_contracting_dims={3}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
  ROOT transpose.8 = bf16[16,256,16,64]{3,2,1,0} transpose(dot.1), dimensions={0,3,1,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Transpose(m::GetTupleElement(
                  m::CustomCall(&fmha, {kCudnnfMHAScaleBiasSoftmaxCallTarget}),
                  0)))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
}

// negative test
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_contracting_dim_not_equal_64) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,256,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,32]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,32]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&fmha, m::Dot(m::Parameter(0), m::Parameter(1)),
                                m::Parameter(2))
                             .WithShape(BF16, {16, 16, 256, 64})));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm1_non_contracting_dim_larger_than_512) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,1024,64]{3,2,1,0},bf16[16,16,1024,64]{3,2,1,0},bf16[16,16,1024,64]{3,2,1,0})->bf16[16,16,1024,64]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,1024,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,1024,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,1024,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,1024,1024]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,1024,64]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* dot;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&dot, m::Op(), m::Parameter(2))
                             .WithShape(BF16, {16, 16, 1024, 64})));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2Pattern_bmm2_rhs_non_contracting_dim_not_equal_64) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0},bf16[16,16,256,32]{3,2,1,0})->bf16[16,16,256,32]{3,2,1,0}}
ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,32]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,64]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,64]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,256,32]{3,2,1,0} dot(dot.0, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&fmha, m::Op(), m::Parameter(2))
                             .WithShape(BF16, {16, 16, 256, 32})));
}

// check if MHA is unsupported, canonicalization will not kick in
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1Bmm2PatternUncanonicalized_bmm1_contracting_dim_not_equal_64) {
  const char* module_str = R"(
HloModule fmha_test, entry_computation_layout={(bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,32]{3,2,1,0},bf16[16,16,256,64]{3,2,1,0})->bf16[16,16,64,256]{3,2,1,0}}

ENTRY main.6 {
  Arg_2.3 = bf16[16,16,256,64]{3,2,1,0} parameter(2)
  Arg_0.1 = bf16[16,16,256,32]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[16,16,256,32]{3,2,1,0} parameter(1)
  dot.0 = bf16[16,16,256,256]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
  ROOT dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(Arg_2.3, dot.0), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};

  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&fmha, m::Parameter(2), m::Op())
                             .WithShape(BF16, {16, 16, 64, 256})));
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16Bmm1BiasSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0})->bf16[16,256,16,64]{3,2,1,0}}

region_0.34 {
  Arg_0.35 = bf16[] parameter(0)
  Arg_1.36 = bf16[] parameter(1)
  ROOT maximum.37 = bf16[] maximum(Arg_0.35, Arg_1.36)
}

region_1.46 {
  Arg_0.47 = f32[] parameter(0)
  Arg_1.48 = f32[] parameter(1)
  ROOT add.49 = f32[] add(Arg_0.47, Arg_1.48)
}

ENTRY main.82 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.2 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.31 = bf16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.32 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.31), dimensions={1,2,3}
  add.33 = bf16[16,16,256,256]{3,2,1,0} add(dot, broadcast.32)
  constant.21 = bf16[] constant(-inf)
  reduce.38 = bf16[16,16,256]{2,1,0} reduce(add.33, constant.21), dimensions={3}, to_apply=region_0.34
  broadcast.42 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.38), dimensions={0,1,2}
  subtract.43 = bf16[16,16,256,256]{3,2,1,0} subtract(add.33, broadcast.42)
  exponential.44 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.43)
  convert.45 = f32[16,16,256,256]{3,2,1,0} convert(exponential.44)
  constant.9 = f32[] constant(0)
  reduce.50 = f32[16,16,256]{2,1,0} reduce(convert.45, constant.9), dimensions={3}, to_apply=region_1.46
  convert.1 = bf16[16,16,256]{2,1,0} convert(reduce.50)
  broadcast.55 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.1), dimensions={0,1,2}
  divide.56 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.44, broadcast.55)
  constant.18 = u32[1]{0} constant({255383827})
  constant.17 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.19 = u32[1]{0} constant({3213575472})
  custom-call.26 = (u32[1]{0}, u32[1]{0}) custom-call(constant.18, constant.17, constant.2, constant.19), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.27 = u32[1]{0} get-tuple-element(custom-call.26), index=0
  reshape.58 = u32[] reshape(get-tuple-element.27)
  broadcast.62 = u32[32768]{0} broadcast(reshape.58), dimensions={}
  get-tuple-element.28 = u32[1]{0} get-tuple-element(custom-call.26), index=1
  reshape.59 = u32[] reshape(get-tuple-element.28)
  broadcast.63 = u32[32768]{0} broadcast(reshape.59), dimensions={}
  iota.57 = u32[65536]{0} iota(), iota_dimension=0
  slice.60 = u32[32768]{0} slice(iota.57), slice={[0:32768]}
  slice.61 = u32[32768]{0} slice(iota.57), slice={[32768:65536]}
  custom-call.64 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.62, broadcast.63, slice.60, slice.61), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.65 = u32[32768]{0} get-tuple-element(custom-call.64), index=0
  get-tuple-element.66 = u32[32768]{0} get-tuple-element(custom-call.64), index=1
  concatenate.67 = u32[65536]{0} concatenate(get-tuple-element.65, get-tuple-element.66), dimensions={0}
  constant.15 = u32[] constant(9)
  broadcast.3 = u32[65536]{0} broadcast(constant.15), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.67, broadcast.3)
  constant.13 = u32[] constant(1065353216)
  broadcast.11 = u32[65536]{0} broadcast(constant.13), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.11)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.17 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.17)
  broadcast.18 = f32[65536]{0} broadcast(constant.9), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.18)
  constant.7 = f32[] constant(0.9)
  broadcast.19 = f32[65536]{0} broadcast(constant.7), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.19), direction=LT
  constant = bf16[] constant(1.109)
  broadcast.20 = bf16[65536]{0} broadcast(constant), dimensions={}
  constant.4 = bf16[] constant(0)
  broadcast.21 = bf16[65536]{0} broadcast(constant.4), dimensions={}
  select.1 = bf16[65536]{0} select(compare.0, broadcast.20, broadcast.21)
  reshape.19 = bf16[16,16,256]{2,1,0} reshape(select.1)
  broadcast.9 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.19), dimensions={0,1,3}
  multiply.79 = bf16[16,16,256,256]{3,2,1,0} multiply(divide.56, broadcast.9)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.2, multiply.79), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.81 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  ROOT copy.3 = bf16[16,256,16,64]{3,2,1,0} copy(transpose.81)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Copy(m::Transpose(m::Transpose(m::GetTupleElement(
                      m::CustomCall(
                          &fmha, {kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget}),
                      0))))
              .WithShape(BF16, {16, 256, 16, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 4);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16Bmm1ScaleBiasSoftmaxDropoutForm2Bmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[32,40,60,64]{3,2,1,0},bf16[32,40,60,64]{3,2,1,0},bf16[32,40,60,64]{3,2,1,0})->bf16[32,40,60,64]{3,2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

region_0.29 {
  Arg_0.30 = bf16[] parameter(0)
  Arg_1.31 = bf16[] parameter(1)
  ROOT maximum.32 = bf16[] maximum(Arg_0.30, Arg_1.31)
}

region_1.41 {
  Arg_0.42 = f32[] parameter(0)
  Arg_1.43 = f32[] parameter(1)
  ROOT add.44 = f32[] add(Arg_0.42, Arg_1.43)
}

ENTRY main.79 {
  Arg_2.3 = bf16[32,40,60,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[32,40,60,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.2 = bf16[32,60,64,40]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  constant.19 = u32[1]{0} constant({2718843009})
  constant.18 = u32[1]{0} constant({1272950319})
  constant.2 = u32[1]{0} constant({0})
  constant.20 = u32[1]{0} constant({2711844646})
  custom-call.54 = (u32[1]{0}, u32[1]{0}) custom-call(constant.19, constant.18, constant.2, constant.20), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.55 = u32[1]{0} get-tuple-element(custom-call.54), index=0
  reshape.58 = u32[] reshape(get-tuple-element.55)
  broadcast.62 = u32[1536000]{0} broadcast(reshape.58), dimensions={}
  get-tuple-element.56 = u32[1]{0} get-tuple-element(custom-call.54), index=1
  reshape.59 = u32[] reshape(get-tuple-element.56)
  broadcast.63 = u32[1536000]{0} broadcast(reshape.59), dimensions={}
  iota.57 = u32[3072000]{0} iota(), iota_dimension=0
  slice.60 = u32[1536000]{0} slice(iota.57), slice={[0:1536000]}
  slice.61 = u32[1536000]{0} slice(iota.57), slice={[1536000:3072000]}
  custom-call.64 = (u32[1536000]{0}, u32[1536000]{0}) custom-call(broadcast.62, broadcast.63, slice.60, slice.61), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1536000]{0}, u32[1536000]{0}, u32[1536000]{0}, u32[1536000]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000p\027\000\000\000\000\000"
  get-tuple-element.65 = u32[1536000]{0} get-tuple-element(custom-call.64), index=0
  get-tuple-element.66 = u32[1536000]{0} get-tuple-element(custom-call.64), index=1
  concatenate.67 = u32[3072000]{0} concatenate(get-tuple-element.65, get-tuple-element.66), dimensions={0}
  constant.16 = u32[] constant(9)
  broadcast.2 = u32[3072000]{0} broadcast(constant.16), dimensions={}
  shift-right-logical.0 = u32[3072000]{0} shift-right-logical(concatenate.67, broadcast.2)
  constant.14 = u32[] constant(1065353216)
  broadcast.6 = u32[3072000]{0} broadcast(constant.14), dimensions={}
  or.0 = u32[3072000]{0} or(shift-right-logical.0, broadcast.6)
  bitcast-convert.0 = f32[3072000]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.8 = f32[3072000]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[3072000]{0} add(bitcast-convert.0, broadcast.8)
  constant.10 = f32[] constant(0)
  broadcast.10 = f32[3072000]{0} broadcast(constant.10), dimensions={}
  maximum.0 = f32[3072000]{0} maximum(add.1, broadcast.10)
  constant.8 = f32[] constant(0.9)
  broadcast.12 = f32[3072000]{0} broadcast(constant.8), dimensions={}
  compare.0 = pred[3072000]{0} compare(maximum.0, broadcast.12), direction=LT
  reshape.18 = pred[32,60,40,40]{3,2,1,0} reshape(compare.0)
  Arg_0.1 = bf16[32,40,60,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[32,40,60,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[32,60,40,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[32,40,60,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[32,40,60,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[32,60,64,40]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[32,60,40,40]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.25 = bf16[] constant(1)
  broadcast.26 = bf16[32,60,40,40]{3,2,1,0} broadcast(constant.25), dimensions={}
  add.28 = bf16[32,60,40,40]{3,2,1,0} add(dot, broadcast.26)
  constant.24 = bf16[] constant(-inf)
  reduce.33 = bf16[32,60,40]{2,1,0} reduce(add.28, constant.24), dimensions={3}, to_apply=region_0.29
  broadcast.37 = bf16[32,60,40,40]{3,2,1,0} broadcast(reduce.33), dimensions={0,1,2}
  subtract.38 = bf16[32,60,40,40]{3,2,1,0} subtract(add.28, broadcast.37)
  exponential.39 = bf16[32,60,40,40]{3,2,1,0} exponential(subtract.38)
  convert.40 = f32[32,60,40,40]{3,2,1,0} convert(exponential.39)
  reduce.45 = f32[32,60,40]{2,1,0} reduce(convert.40, constant.10), dimensions={3}, to_apply=region_1.41
  convert.0 = bf16[32,60,40]{2,1,0} convert(reduce.45)
  broadcast.50 = bf16[32,60,40,40]{3,2,1,0} broadcast(convert.0), dimensions={0,1,2}
  divide.51 = bf16[32,60,40,40]{3,2,1,0} divide(exponential.39, broadcast.50)
  constant = bf16[] constant(1.109)
  broadcast.1 = bf16[32,60,40,40]{3,2,1,0} broadcast(constant), dimensions={}
  multiply = bf16[32,60,40,40]{3,2,1,0} multiply(divide.51, broadcast.1)
  constant.4 = bf16[] constant(0)
  broadcast.5 = bf16[32,60,40,40]{3,2,1,0} broadcast(constant.4), dimensions={}
  select.76 = bf16[32,60,40,40]{3,2,1,0} select(reshape.18, multiply, broadcast.5)
  dot.1 = bf16[32,60,64,40]{3,2,1,0} dot(transpose.2, select.76), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.78 = bf16[32,40,60,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  ROOT copy.3 = bf16[32,40,60,64]{3,2,1,0} copy(transpose.78)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Copy(m::Transpose(m::Transpose(m::GetTupleElement(
                      m::CustomCall(
                          &fmha, {kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget}),
                      0))))
              .WithShape(BF16, {32, 40, 60, 64})));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
  EXPECT_EQ(fmha->operands().size(), 4);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16TrainingBmm1Bmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->(bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0})}

ENTRY main.17 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.2 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.2, dot), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.7 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_3.4 = bf16[16,256,16,64]{3,2,1,0} parameter(3), sharding={replicated}
  copy.3 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_3.4), sharding={replicated}
  transpose.4 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  copy.4 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.12 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = bf16[16,16,256,64]{3,2,1,0} dot(dot.2, transpose.12), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.15 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = bf16[16,16,256,64]{3,2,1,0} dot(dot.2, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.13 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_3.4), sharding={replicated}
  transpose.8 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.10 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.8, dot), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.11 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.10), dimensions={0,3,1,2}
  tuple.16 = (bf16[16,256,16,64]{1,3,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{1,3,2,0}) tuple(transpose.7, transpose.15, transpose.13, transpose.11)
  get-tuple-element = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.16), index=0
  copy.6 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.16), index=1
  copy.7 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.16), index=2
  copy.8 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.16), index=3
  copy.9 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  ROOT tuple = (bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  const auto status = RunHloPass(&fusedMhaRewriter, m.get());
  const bool changed = status.value();
  EXPECT_EQ(changed, false);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->(bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[1,16,256,256]{3,2,1,0})}

region_0.54 {
  Arg_0.55 = bf16[] parameter(0)
  Arg_1.56 = bf16[] parameter(1)
  ROOT maximum.57 = bf16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = bf16[] parameter(0)
  Arg_1.116 = bf16[] parameter(1)
  ROOT add.117 = bf16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = bf16[] constant(0)
  broadcast.1 = bf16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.10 = bf16[] constant(-9.999e+09)
  broadcast.3 = bf16[16,1,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
  select.39 = bf16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = bf16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = bf16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = bf16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = bf16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  constant.31 = bf16[] constant(-inf)
  reduce.58 = bf16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = bf16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = bf16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = bf16[] constant(1.109)
  broadcast.33 = bf16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = bf16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = bf16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = bf16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = bf16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = bf16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = bf16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = bf16[] constant(1)
  broadcast.24 = bf16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = bf16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = bf16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = bf16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = bf16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = bf16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = bf16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = bf16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = bf16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = bf16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = bf16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = bf16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = bf16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = bf16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  reshape.140 = bf16[1,16,256,256]{3,2,1,0} reshape(reduce.139)
  tuple.145 = (bf16[16,256,16,64]{1,3,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{1,3,2,0}, bf16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, reshape.140)
  get-tuple-element = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = bf16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Copy(m::GetTupleElement(
              m::Tuple(
                  m::Transpose().WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(m::GetTupleElement(
                                   m::CustomCall(&fmha, {backward_target}), 0))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(
                      m::GetTupleElement(m::CustomCall({backward_target}), 1))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(m::Transpose(m::GetTupleElement(
                                   m::CustomCall({backward_target}), 2)))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Reshape(
                      m::Reshape(m::GetTupleElement(  // dbias
                          m::CustomCall({backward_target}), dbias_index)))
                      .WithShape(BF16, {1, 16, 256, 256})),
              0)),
          m::Op(), m::Op(), m::Op(), m::Op())));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},f16[16,256,16,64]{3,2,1,0})->(f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

region_0.54 {
  Arg_0.55 = f16[] parameter(0)
  Arg_1.56 = f16[] parameter(1)
  ROOT maximum.57 = f16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = f16[] parameter(0)
  Arg_1.116 = f16[] parameter(1)
  ROOT add.117 = f16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = f16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = f16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = f16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = f16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = f16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = f16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = f16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = f16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = f16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = f16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = f16[] constant(0)
  broadcast.1 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.31 = f16[] constant(-inf)
  broadcast.3 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.31), dimensions={}
  select.39 = f16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = f16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = f16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = f16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = f16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = f16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  reduce.58 = f16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = f16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = f16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = f16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = f16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = f16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = f16[] constant(1.1113)
  broadcast.33 = f16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = f16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = f16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = f16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = f16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = f16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = f16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = f16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = f16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = f16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = f16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = f16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = f16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = f16[] constant(1)
  broadcast.24 = f16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = f16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = f16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = f16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = f16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = f16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = f16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = f16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = f16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = f16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = f16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = f16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = f16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = f16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = f16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = f16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = f16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = f16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  reshape.140 = f16[1,16,256,256]{3,2,1,0} reshape(reduce.139)
  tuple.145 = (f16[16,256,16,64]{1,3,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{1,3,2,0}, f16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, reshape.140)
  get-tuple-element = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = f16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Copy(m::GetTupleElement(
              m::Tuple(
                  m::Transpose().WithShape(F16, {16, 256, 16, 64}),
                  m::Transpose(m::GetTupleElement(
                                   m::CustomCall(&fmha, {backward_target}), 0))
                      .WithShape(F16, {16, 256, 16, 64}),
                  m::Transpose(
                      m::GetTupleElement(m::CustomCall({backward_target}), 1))
                      .WithShape(F16, {16, 256, 16, 64}),
                  m::Transpose(m::Transpose(m::GetTupleElement(
                                   m::CustomCall({backward_target}), 2)))
                      .WithShape(F16, {16, 256, 16, 64}),
                  m::Reshape(
                      m::Reshape(m::GetTupleElement(  // dbias
                          m::CustomCall({backward_target}), dbias_index)))
                      .WithShape(F16, {1, 16, 256, 256})),
              0)),
          m::Op(), m::Op(), m::Op(), m::Op())));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2WithTransposeFusion) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[16,256,16,64]{3,2,1,0},f16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},f16[16,256,16,64]{3,2,1,0})->(f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true,true}

region_0.54 {
  Arg_0.55 = f16[] parameter(0)
  Arg_1.56 = f16[] parameter(1)
  ROOT maximum.57 = f16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = f16[] parameter(0)
  Arg_1.116 = f16[] parameter(1)
  ROOT add.117 = f16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = f16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = f16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = f16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = f16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = f16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = f16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = f16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = f16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = f16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = f16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = f16[] constant(0)
  broadcast.1 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.31 = f16[] constant(-inf)
  broadcast.3 = f16[16,1,256,256]{3,2,1,0} broadcast(constant.31), dimensions={}
  select.39 = f16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = f16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = f16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = f16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = f16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = f16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  reduce.58 = f16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = f16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = f16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = f16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = f16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = f16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = f16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = f16[] constant(1.1113)
  broadcast.33 = f16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = f16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = f16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = f16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = f16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = f16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = f16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = f16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = f16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = f16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = f16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = f16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = f16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = f16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = f16[] constant(1)
  broadcast.24 = f16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = f16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = f16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = f16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = f16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = f16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = f16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = f16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = f16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = f16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = f16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = f16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = f16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = f16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = f16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = f16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = f16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = f16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = f16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = f16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  reshape.140 = f16[1,16,256,256]{3,2,1,0} reshape(reduce.139)
  tuple.145 = (f16[16,256,16,64]{1,3,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{3,1,2,0}, f16[16,256,16,64]{1,3,2,0}, f16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, reshape.140)
  get-tuple-element = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = f16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = f16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = f16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = f16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[16,256,16,64]{3,2,1,0}, f16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  AlgebraicSimplifierOptions alg_sim_options;
  alg_sim_options.set_supports_non_canonical_dots(false);
  alg_sim_options.set_is_layout_sensitive(true);
  alg_sim_options.set_enable_conv_operand_swap(false);
  AlgebraicSimplifier alge_simp{alg_sim_options};

  LayoutNormalization layout_normalizer;
  HloCSE cse{/*is_layout_sensitive=*/true};
  TF_ASSERT_OK(RunHloPass(&layout_normalizer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&cse, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());

  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  CudnnFusedMHATransposeFusion fmha_transpose_fusion;

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&fmha_transpose_fusion, m.get()).status());

  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Bitcast().WithShape(F16, {16, 256, 16, 64}),
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      &fmha,
                      {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
                  0))
              .WithShape(F16, {16, 256, 16, 64}),
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
                  1))
              .WithShape(F16, {16, 256, 16, 64}),
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
                  2))
              .WithShape(F16, {16, 256, 16, 64}),
          m::GetTupleElement(  // dbias
              m::CustomCall(
                  {kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget}),
              dbias_index))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16MiniT5xTest) {
  const char* module_str = R"(
HloModule jit__lambda_, entry_computation_layout={(bf16[12,512,32,64]{3,2,1,0},bf16[12,512,2,32,64]{4,3,2,1,0},f32[12,512]{1,0},f32[12,512]{1,0})->(bf16[], bf16[12,512,32,64]{3,2,1,0}, bf16[12,512,2,32,64]{4,3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true}

region_0.51 {
  Arg_0.52 = bf16[] parameter(0)
  Arg_1.53 = bf16[] parameter(1)
  ROOT maximum.54 = bf16[] maximum(Arg_0.52, Arg_1.53)
}

region_1.63 {
  Arg_0.64 = f32[] parameter(0)
  Arg_1.65 = f32[] parameter(1)
  ROOT add.66 = f32[] add(Arg_0.64, Arg_1.65)
}

region_3.99 {
  Arg_0.100 = bf16[] parameter(0)
  Arg_1.101 = bf16[] parameter(1)
  ROOT add.102 = bf16[] add(Arg_0.100, Arg_1.101)
}

ENTRY main.129 {
  Arg_1.2 = bf16[12,512,2,32,64]{4,3,2,1,0} parameter(1), sharding={replicated}
  copy = bf16[12,512,2,32,64]{1,4,3,0,2} copy(Arg_1.2), sharding={replicated}
  slice.42 = bf16[12,512,1,32,64]{1,4,3,0,2} slice(copy), slice={[0:12], [0:512], [1:2], [0:32], [0:64]}
  reshape.44 = bf16[12,512,32,64]{1,3,2,0} reshape(slice.42)
  transpose.5 = bf16[12,32,64,512]{3,2,1,0} transpose(reshape.44), dimensions={0,2,3,1}
  Arg_0.1 = bf16[12,512,32,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[12,512,32,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  constant.5 = bf16[] constant(0.125)
  broadcast.6 = bf16[12,512,32,64]{3,1,2,0} broadcast(constant.5), dimensions={}
  multiply.45 = bf16[12,512,32,64]{3,1,2,0} multiply(copy.1, broadcast.6)
  transpose = bf16[12,32,512,64]{3,2,1,0} transpose(multiply.45), dimensions={0,2,1,3}
  copy.2 = bf16[12,512,2,32,64]{1,4,3,0,2} copy(Arg_1.2), sharding={replicated}
  slice.41 = bf16[12,512,1,32,64]{1,4,3,0,2} slice(copy.2), slice={[0:12], [0:512], [0:1], [0:32], [0:64]}
  reshape.43 = bf16[12,512,32,64]{1,3,2,0} reshape(slice.41)
  transpose.1 = bf16[12,32,64,512]{3,2,1,0} transpose(reshape.43), dimensions={0,2,3,1}
  dot = bf16[12,32,512,512]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_2.3 = f32[12,512]{1,0} parameter(2), sharding={replicated}
  constant.14 = f32[] constant(0)
  broadcast.19 = f32[12,512]{1,0} broadcast(constant.14), dimensions={}
  compare.24 = pred[12,512]{1,0} compare(Arg_2.3, broadcast.19), direction=GT
  broadcast.30 = pred[12,512,512]{2,1,0} broadcast(compare.24), dimensions={0,1}
  Arg_3.4 = f32[12,512]{1,0} parameter(3), sharding={replicated}
  compare.25 = pred[12,512]{1,0} compare(Arg_3.4, broadcast.19), direction=GT
  broadcast.33 = pred[12,512,512]{2,1,0} broadcast(compare.25), dimensions={0,2}
  and.34 = pred[12,512,512]{2,1,0} and(broadcast.30, broadcast.33)
  convert.4 = s32[12,512,512]{2,1,0} convert(and.34)
  constant.16 = s32[] constant(0)
  broadcast.21 = s32[12,512,512]{2,1,0} broadcast(constant.16), dimensions={}
  compare.0 = pred[12,512,512]{2,1,0} compare(convert.4, broadcast.21), direction=GT
  constant.20 = bf16[] constant(0)
  broadcast.22 = bf16[12,512,512]{2,1,0} broadcast(constant.20), dimensions={}
  constant.11 = bf16[] constant(-9.999e+09)
  broadcast.23 = bf16[12,512,512]{2,1,0} broadcast(constant.11), dimensions={}
  select.0 = bf16[12,512,512]{2,1,0} select(compare.0, broadcast.22, broadcast.23)
  broadcast.49 = bf16[12,32,512,512]{3,2,1,0} broadcast(select.0), dimensions={0,2,3}
  add.50 = bf16[12,32,512,512]{3,2,1,0} add(dot, broadcast.49)
  constant.22 = bf16[] constant(-inf)
  reduce.55 = bf16[12,32,512]{2,1,0} reduce(add.50, constant.22), dimensions={3}, to_apply=region_0.51
  broadcast.59 = bf16[12,32,512,512]{3,2,1,0} broadcast(reduce.55), dimensions={0,1,2}
  subtract.60 = bf16[12,32,512,512]{3,2,1,0} subtract(add.50, broadcast.59)
  exponential.61 = bf16[12,32,512,512]{3,2,1,0} exponential(subtract.60)
  convert.62 = f32[12,32,512,512]{3,2,1,0} convert(exponential.61)
  reduce.67 = f32[12,32,512]{2,1,0} reduce(convert.62, constant.14), dimensions={3}, to_apply=region_1.63
  convert.5 = bf16[12,32,512]{2,1,0} convert(reduce.67)
  broadcast.72 = bf16[12,32,512,512]{3,2,1,0} broadcast(convert.5), dimensions={0,1,2}
  divide.73 = bf16[12,32,512,512]{3,2,1,0} divide(exponential.61, broadcast.72)
  dot.1 = bf16[12,32,64,512]{3,2,1,0} dot(transpose.5, divide.73), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  convert.6 = f32[12,32,64,512]{3,2,1,0} convert(dot.1)
  reduce.83 = f32[] reduce(convert.6, constant.14), dimensions={0,3,1,2}, to_apply=region_1.63
  convert.84 = bf16[] convert(reduce.83)
  constant.2 = bf16[] constant(0.0007935)
  multiply.86 = bf16[] multiply(convert.84, constant.2)
  broadcast.9 = bf16[12,32,512,64]{3,2,1,0} broadcast(constant.2), dimensions={}
  dot.2 = bf16[12,32,512,512]{3,2,1,0} dot(broadcast.9, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  divide.109 = bf16[12,32,512,512]{3,2,1,0} divide(dot.2, broadcast.72)
  constant.10 = bf16[] constant(1)
  broadcast.24 = bf16[12,32,512]{2,1,0} broadcast(constant.10), dimensions={}
  multiply.4 = bf16[12,32,512]{2,1,0} multiply(convert.5, convert.5)
  divide.0 = bf16[12,32,512]{2,1,0} divide(broadcast.24, multiply.4)
  broadcast.96 = bf16[12,32,512,512]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.97 = bf16[12,32,512,512]{3,2,1,0} multiply(dot.2, broadcast.96)
  multiply.98 = bf16[12,32,512,512]{3,2,1,0} multiply(multiply.97, exponential.61)
  reduce.103 = bf16[12,32,512]{2,1,0} reduce(multiply.98, constant.20), dimensions={3}, to_apply=region_3.99
  negate.0 = bf16[12,32,512]{2,1,0} negate(reduce.103)
  broadcast.10 = bf16[12,32,512,512]{3,2,1,0} broadcast(negate.0), dimensions={0,1,2}
  add.118 = bf16[12,32,512,512]{3,2,1,0} add(divide.109, broadcast.10)
  multiply.119 = bf16[12,32,512,512]{3,2,1,0} multiply(add.118, exponential.61)
  transpose.9 = bf16[12,32,512,64]{2,3,1,0} transpose(reshape.43), dimensions={0,2,1,3}
  copy.3 = bf16[12,32,512,64]{3,2,1,0} copy(transpose.9)
  dot.4 = bf16[12,32,512,64]{3,2,1,0} dot(multiply.119, copy.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  broadcast.12 = bf16[12,32,512,64]{3,2,1,0} broadcast(constant.5), dimensions={}
  multiply.3 = bf16[12,32,512,64]{3,2,1,0} multiply(dot.4, broadcast.12)
  transpose.11 = bf16[12,512,32,64]{3,1,2,0} transpose(multiply.3), dimensions={0,2,1,3}
  broadcast.7 = bf16[12,32,64,512]{3,2,1,0} broadcast(constant.2), dimensions={}
  dot.90 = bf16[12,32,64,512]{3,2,1,0} dot(broadcast.7, divide.73), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.91 = bf16[12,512,32,64]{1,3,2,0} transpose(dot.90), dimensions={0,3,1,2}
  reshape.92 = bf16[12,512,1,32,64]{1,4,3,0,2} reshape(transpose.91)
  pad.93 = bf16[12,512,2,32,64]{1,4,3,0,2} pad(reshape.92, constant.20), padding=0_0x0_0x1_0x0_0x0_0
  dot.3 = bf16[12,32,512,64]{3,2,1,0} dot(multiply.119, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  copy.4 = bf16[12,32,512,64]{2,3,1,0} copy(dot.3)
  transpose.121 = bf16[12,512,32,64]{1,3,2,0} transpose(copy.4), dimensions={0,2,1,3}
  reshape.124 = bf16[12,512,1,32,64]{1,4,3,0,2} reshape(transpose.121)
  pad.125 = bf16[12,512,2,32,64]{1,4,3,0,2} pad(reshape.124, constant.20), padding=0_0x0_0x0_1x0_0x0_0
  add.126 = bf16[12,512,2,32,64]{1,4,3,0,2} add(pad.93, pad.125)
  tuple.128 = (bf16[], bf16[12,512,32,64]{3,1,2,0}, bf16[12,512,2,32,64]{1,4,3,0,2}) tuple(multiply.86, transpose.11, add.126)
  get-tuple-element = bf16[] get-tuple-element(tuple.128), index=0
  get-tuple-element.1 = bf16[12,512,32,64]{3,1,2,0} get-tuple-element(tuple.128), index=1
  copy.5 = bf16[12,512,32,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[12,512,2,32,64]{1,4,3,0,2} get-tuple-element(tuple.128), index=2
  copy.6 = bf16[12,512,2,32,64]{4,3,2,1,0} copy(get-tuple-element.2)
  ROOT tuple = (bf16[], bf16[12,512,32,64]{3,2,1,0}, bf16[12,512,2,32,64]{4,3,2,1,0}) tuple(get-tuple-element, copy.5, copy.6)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  AlgebraicSimplifierOptions alg_sim_options;
  alg_sim_options.set_supports_non_canonical_dots(false);
  alg_sim_options.set_is_layout_sensitive(true);
  alg_sim_options.set_enable_conv_operand_swap(false);
  AlgebraicSimplifier alge_simp{alg_sim_options};
  ReshapeDecomposer reshape_decomposer;
  LayoutNormalization layout_normalizer;
  HloCSE cse{/*is_layout_sensitive=*/true};
  TF_ASSERT_OK(RunHloPass(&reshape_decomposer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&layout_normalizer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&cse, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());

  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  CudnnFusedMHATransposeFusion fmha_transpose_fusion;

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&fmha_transpose_fusion, m.get()).status());

  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  EXPECT_EQ(CountFusedAttentionCall(m.get()), 1);
  EXPECT_EQ(CountFusedAttentionCall(m.get(), /*is_backward*/ true), 1);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm1ScaleBiasMaskSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,128,64]{3,2,1,0},bf16[2,6,64,128]{3,2,1,0},bf16[2,6,128,64]{3,2,1,0},bf16[2,6,128,64]{3,2,1,0})->(bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,64,128]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.38 {
  Arg_0.39 = bf16[] parameter(0)
  Arg_1.40 = bf16[] parameter(1)
  ROOT maximum.1 = bf16[] maximum(Arg_0.39, Arg_1.40)
}

region_1.50 {
  Arg_0.51 = f32[] parameter(0)
  Arg_1.52 = f32[] parameter(1)
  ROOT add.2 = f32[] add(Arg_0.51, Arg_1.52)
}

region_2.99 {
  Arg_0.100 = bf16[] parameter(0)
  Arg_1.101 = bf16[] parameter(1)
  ROOT add.3 = bf16[] add(Arg_0.100, Arg_1.101)
}

ENTRY main.126 {
  constant.6 = u32[1]{0} constant({2718843009})
  constant.8 = u32[1]{0} constant({1272950319})
  constant.10 = u32[1]{0} constant({0})
  constant.12 = u32[1]{0} constant({2711844646})
  custom-call.65 = (u32[1]{0}, u32[1]{0}) custom-call(constant.6, constant.8, constant.10, constant.12), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.66 = u32[1]{0} get-tuple-element(custom-call.65), index=0
  bitcast.343 = u32[] bitcast(get-tuple-element.66)
  broadcast.27 = u32[98304]{0} broadcast(bitcast.343), dimensions={}
  get-tuple-element.67 = u32[1]{0} get-tuple-element(custom-call.65), index=1
  bitcast.344 = u32[] bitcast(get-tuple-element.67)
  broadcast.28 = u32[98304]{0} broadcast(bitcast.344), dimensions={}
  iota.68 = u32[196608]{0} iota(), iota_dimension=0
  slice = u32[98304]{0} slice(iota.68), slice={[0:98304]}
  slice.1 = u32[98304]{0} slice(iota.68), slice={[98304:196608]}
  custom-call.75 = (u32[98304]{0}, u32[98304]{0}) custom-call(broadcast.27, broadcast.28, slice, slice.1), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[98304]{0}, u32[98304]{0}, u32[98304]{0}, u32[98304]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\001\000\000\000\000\000"
  get-tuple-element.76 = u32[98304]{0} get-tuple-element(custom-call.75), index=0
  get-tuple-element.77 = u32[98304]{0} get-tuple-element(custom-call.75), index=1
  concatenate.2 = u32[196608]{0} concatenate(get-tuple-element.76, get-tuple-element.77), dimensions={0}
  constant.56 = u32[] constant(9)
  broadcast.63 = u32[196608]{0} broadcast(constant.56), dimensions={}
  shift-right-logical.3 = u32[196608]{0} shift-right-logical(concatenate.2, broadcast.63)
  constant.57 = u32[] constant(1065353216)
  broadcast.64 = u32[196608]{0} broadcast(constant.57), dimensions={}
  or.3 = u32[196608]{0} or(shift-right-logical.3, broadcast.64)
  bitcast-convert.3 = f32[196608]{0} bitcast-convert(or.3)
  constant.58 = f32[] constant(-1)
  broadcast.65 = f32[196608]{0} broadcast(constant.58), dimensions={}
  add.10 = f32[196608]{0} add(bitcast-convert.3, broadcast.65)
  constant.48 = f32[] constant(0)
  broadcast.66 = f32[196608]{0} broadcast(constant.48), dimensions={}
  maximum.4 = f32[196608]{0} maximum(add.10, broadcast.66)
  constant.59 = f32[] constant(0.9)
  broadcast.67 = f32[196608]{0} broadcast(constant.59), dimensions={}
  compare.3 = pred[196608]{0} compare(maximum.4, broadcast.67), direction=LT
  bitcast.308 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  constant.44 = pred[2,6,128,128]{3,2,1,0} constant({...})
  Arg_0.1 = bf16[2,6,128,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = bf16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.34 = bf16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.55 = bf16[] constant(2)
  broadcast.61 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.55), dimensions={}
  multiply.8 = bf16[2,6,128,128]{3,2,1,0} multiply(dot.34, broadcast.61)
  constant.52 = bf16[] constant(1)
  broadcast.39 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.52), dimensions={}
  add.6 = bf16[2,6,128,128]{3,2,1,0} add(multiply.8, broadcast.39)
  constant.54 = bf16[] constant(0)
  broadcast.52 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.54), dimensions={}
  select.1 = bf16[2,6,128,128]{3,2,1,0} select(constant.44, add.6, broadcast.52)
  constant.41 = bf16[] constant(-inf)
  reduce.42 = bf16[2,6,128]{2,1,0} reduce(select.1, constant.41), dimensions={3}, to_apply=region_0.38
  broadcast.42 = bf16[2,6,128,128]{3,2,1,0} broadcast(reduce.42), dimensions={0,1,2}
  subtract.1 = bf16[2,6,128,128]{3,2,1,0} subtract(select.1, broadcast.42)
  exponential.1 = bf16[2,6,128,128]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,128,128]{3,2,1,0} convert(exponential.1)
  reduce.54 = f32[2,6,128]{2,1,0} reduce(convert.5, constant.48), dimensions={3}, to_apply=region_1.50
  convert.9 = bf16[2,6,128]{2,1,0} convert(reduce.54)
  broadcast.68 = bf16[2,6,128,128]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = bf16[2,6,128,128]{3,2,1,0} divide(exponential.1, broadcast.68)
  constant.60 = bf16[] constant(1.109)
  broadcast.69 = bf16[2,6,128,128]{3,2,1,0} broadcast(constant.60), dimensions={}
  multiply.20 = bf16[2,6,128,128]{3,2,1,0} multiply(divide.5, broadcast.69)
  select.8 = bf16[2,6,128,128]{3,2,1,0} select(bitcast.308, multiply.20, broadcast.52)
  Arg_2.3 = bf16[2,6,128,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.88 = bf16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  bitcast.248 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  Arg_3.4 = bf16[2,6,128,64]{3,2,1,0} parameter(3), sharding={replicated}
  dot.91 = bf16[2,6,128,128]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  select.6 = bf16[2,6,128,128]{3,2,1,0} select(bitcast.248, dot.91, broadcast.52)
  multiply.17 = bf16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.69)
  divide.4 = bf16[2,6,128,128]{3,2,1,0} divide(multiply.17, broadcast.68)
  broadcast.55 = bf16[2,6,128]{2,1,0} broadcast(constant.52), dimensions={}
  multiply.11 = bf16[2,6,128]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = bf16[2,6,128]{2,1,0} divide(broadcast.55, multiply.11)
  broadcast.56 = bf16[2,6,128]{2,1,0} broadcast(constant.60), dimensions={}
  multiply.12 = bf16[2,6,128]{2,1,0} multiply(divide.3, broadcast.56)
  broadcast.58 = bf16[2,6,128,128]{3,2,1,0} broadcast(multiply.12), dimensions={0,1,2}
  multiply.13 = bf16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.58)
  multiply.14 = bf16[2,6,128,128]{3,2,1,0} multiply(multiply.13, exponential.1)
  reduce.103 = bf16[2,6,128]{2,1,0} reduce(multiply.14, constant.54), dimensions={3}, to_apply=region_2.99
  negate.3 = bf16[2,6,128]{2,1,0} negate(reduce.103)
  broadcast.62 = bf16[2,6,128,128]{3,2,1,0} broadcast(negate.3), dimensions={0,1,2}
  add.9 = bf16[2,6,128,128]{3,2,1,0} add(divide.4, broadcast.62)
  multiply.18 = bf16[2,6,128,128]{3,2,1,0} multiply(add.9, exponential.1)
  select.7 = bf16[2,6,128,128]{3,2,1,0} select(constant.44, multiply.18, broadcast.52)
  multiply.19 = bf16[2,6,128,128]{3,2,1,0} multiply(select.7, broadcast.61)
  dot.124 = bf16[2,6,128,64]{3,2,1,0} dot(multiply.19, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = bf16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.19), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.125 = (bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0}, bf16[2,6,64,128]{3,2,1,0}, bf16[2,6,128,64]{3,2,1,0}) tuple(dot.88, dot.124, dot, dot.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&fmha, {target}), 0)
              .WithShape(BF16, {2, 6, 128, 64}),
          m::GetTupleElement(m::CustomCall(&fmha, {backward_target}), 0)
              .WithShape(BF16, {2, 6, 128, 64}),
          m::Transpose(m::GetTupleElement(m::CustomCall({backward_target}), 1))
              .WithShape(BF16, {2, 6, 64, 128}),
          m::GetTupleElement(m::CustomCall({backward_target}), 2)
              .WithShape(BF16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasMaskSoftmaxDropoutBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,128,64]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.38 {
  Arg_0.39 = f16[] parameter(0)
  Arg_1.40 = f16[] parameter(1)
  ROOT maximum.1 = f16[] maximum(Arg_0.39, Arg_1.40)
}

region_1.50 {
  Arg_0.51 = f32[] parameter(0)
  Arg_1.52 = f32[] parameter(1)
  ROOT add.2 = f32[] add(Arg_0.51, Arg_1.52)
}

region_2.99 {
  Arg_0.100 = f16[] parameter(0)
  Arg_1.101 = f16[] parameter(1)
  ROOT add.3 = f16[] add(Arg_0.100, Arg_1.101)
}

ENTRY main.126 {
  constant.6 = u32[1]{0} constant({2718843009})
  constant.8 = u32[1]{0} constant({1272950319})
  constant.10 = u32[1]{0} constant({0})
  constant.12 = u32[1]{0} constant({2711844646})
  custom-call.65 = (u32[1]{0}, u32[1]{0}) custom-call(constant.6, constant.8, constant.10, constant.12), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.66 = u32[1]{0} get-tuple-element(custom-call.65), index=0
  bitcast.343 = u32[] bitcast(get-tuple-element.66)
  broadcast.27 = u32[98304]{0} broadcast(bitcast.343), dimensions={}
  get-tuple-element.67 = u32[1]{0} get-tuple-element(custom-call.65), index=1
  bitcast.344 = u32[] bitcast(get-tuple-element.67)
  broadcast.28 = u32[98304]{0} broadcast(bitcast.344), dimensions={}
  iota.68 = u32[196608]{0} iota(), iota_dimension=0
  slice = u32[98304]{0} slice(iota.68), slice={[0:98304]}
  slice.1 = u32[98304]{0} slice(iota.68), slice={[98304:196608]}
  custom-call.75 = (u32[98304]{0}, u32[98304]{0}) custom-call(broadcast.27, broadcast.28, slice, slice.1), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[98304]{0}, u32[98304]{0}, u32[98304]{0}, u32[98304]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\001\000\000\000\000\000"
  get-tuple-element.76 = u32[98304]{0} get-tuple-element(custom-call.75), index=0
  get-tuple-element.77 = u32[98304]{0} get-tuple-element(custom-call.75), index=1
  concatenate.2 = u32[196608]{0} concatenate(get-tuple-element.76, get-tuple-element.77), dimensions={0}
  constant.56 = u32[] constant(9)
  broadcast.63 = u32[196608]{0} broadcast(constant.56), dimensions={}
  shift-right-logical.3 = u32[196608]{0} shift-right-logical(concatenate.2, broadcast.63)
  constant.57 = u32[] constant(1065353216)
  broadcast.64 = u32[196608]{0} broadcast(constant.57), dimensions={}
  or.3 = u32[196608]{0} or(shift-right-logical.3, broadcast.64)
  bitcast-convert.3 = f32[196608]{0} bitcast-convert(or.3)
  constant.58 = f32[] constant(-1)
  broadcast.65 = f32[196608]{0} broadcast(constant.58), dimensions={}
  add.10 = f32[196608]{0} add(bitcast-convert.3, broadcast.65)
  constant.48 = f32[] constant(0)
  broadcast.66 = f32[196608]{0} broadcast(constant.48), dimensions={}
  maximum.4 = f32[196608]{0} maximum(add.10, broadcast.66)
  constant.59 = f32[] constant(0.9)
  broadcast.67 = f32[196608]{0} broadcast(constant.59), dimensions={}
  compare.3 = pred[196608]{0} compare(maximum.4, broadcast.67), direction=LT
  bitcast.308 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  constant.44 = pred[2,6,128,128]{3,2,1,0} constant({...})
  Arg_0.1 = f16[2,6,128,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.34 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.55 = f16[] constant(2)
  broadcast.61 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.55), dimensions={}
  multiply.8 = f16[2,6,128,128]{3,2,1,0} multiply(dot.34, broadcast.61)
  constant.52 = f16[] constant(1)
  broadcast.39 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.52), dimensions={}
  add.6 = f16[2,6,128,128]{3,2,1,0} add(multiply.8, broadcast.39)
  constant.54 = f16[] constant(0)
  broadcast.52 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.54), dimensions={}
  select.1 = f16[2,6,128,128]{3,2,1,0} select(constant.44, add.6, broadcast.52)
  constant.41 = f16[] constant(-inf)
  reduce.42 = f16[2,6,128]{2,1,0} reduce(select.1, constant.41), dimensions={3}, to_apply=region_0.38
  broadcast.42 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.42), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(select.1, broadcast.42)
  exponential.1 = f16[2,6,128,128]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,128,128]{3,2,1,0} convert(exponential.1)
  reduce.54 = f32[2,6,128]{2,1,0} reduce(convert.5, constant.48), dimensions={3}, to_apply=region_1.50
  convert.9 = f16[2,6,128]{2,1,0} convert(reduce.54)
  broadcast.68 = f16[2,6,128,128]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = f16[2,6,128,128]{3,2,1,0} divide(exponential.1, broadcast.68)
  constant.60 = f16[] constant(1.1113)
  broadcast.69 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.60), dimensions={}
  multiply.20 = f16[2,6,128,128]{3,2,1,0} multiply(divide.5, broadcast.69)
  select.8 = f16[2,6,128,128]{3,2,1,0} select(bitcast.308, multiply.20, broadcast.52)
  Arg_2.3 = f16[2,6,128,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.88 = f16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  bitcast.248 = pred[2,6,128,128]{3,2,1,0} bitcast(compare.3)
  Arg_3.4 = f16[2,6,128,64]{3,2,1,0} parameter(3), sharding={replicated}
  dot.91 = f16[2,6,128,128]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  select.6 = f16[2,6,128,128]{3,2,1,0} select(bitcast.248, dot.91, broadcast.52)
  multiply.17 = f16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.69)
  divide.4 = f16[2,6,128,128]{3,2,1,0} divide(multiply.17, broadcast.68)
  broadcast.55 = f16[2,6,128]{2,1,0} broadcast(constant.52), dimensions={}
  multiply.11 = f16[2,6,128]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = f16[2,6,128]{2,1,0} divide(broadcast.55, multiply.11)
  broadcast.56 = f16[2,6,128]{2,1,0} broadcast(constant.60), dimensions={}
  multiply.12 = f16[2,6,128]{2,1,0} multiply(divide.3, broadcast.56)
  broadcast.58 = f16[2,6,128,128]{3,2,1,0} broadcast(multiply.12), dimensions={0,1,2}
  multiply.13 = f16[2,6,128,128]{3,2,1,0} multiply(select.6, broadcast.58)
  multiply.14 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.13, exponential.1)
  reduce.103 = f16[2,6,128]{2,1,0} reduce(multiply.14, constant.54), dimensions={3}, to_apply=region_2.99
  negate.3 = f16[2,6,128]{2,1,0} negate(reduce.103)
  broadcast.62 = f16[2,6,128,128]{3,2,1,0} broadcast(negate.3), dimensions={0,1,2}
  add.9 = f16[2,6,128,128]{3,2,1,0} add(divide.4, broadcast.62)
  multiply.18 = f16[2,6,128,128]{3,2,1,0} multiply(add.9, exponential.1)
  select.7 = f16[2,6,128,128]{3,2,1,0} select(constant.44, multiply.18, broadcast.52)
  multiply.19 = f16[2,6,128,128]{3,2,1,0} multiply(select.7, broadcast.61)
  dot.124 = f16[2,6,128,64]{3,2,1,0} dot(multiply.19, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.19), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(select.8, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.125 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}) tuple(dot.88, dot.124, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&fmha, {target}), 0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::GetTupleElement(m::CustomCall(&fmha, {backward_target}), 0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::Transpose(m::GetTupleElement(m::CustomCall({backward_target}), 1))
              .WithShape(F16, {2, 6, 64, 128}),
          m::GetTupleElement(m::CustomCall({backward_target}), 2)
              .WithShape(F16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       F16TrainingBmm1ScaleBiasMaskSoftmaxBmm2) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f16[2,6,128,64]{3,2,1,0},f16[2,6,64,128]{3,2,1,0},f16[2,6,128,64]{3,2,1,0},f16[2,6,128,64]{3,2,1,0})->(f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.21 {
  Arg_0.22 = f16[] parameter(0)
  Arg_1.23 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(Arg_0.22, Arg_1.23)
}

region_1.33 {
  Arg_0.34 = f32[] parameter(0)
  Arg_1.35 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.34, Arg_1.35)
}

region_2.55 {
  Arg_0.56 = f16[] parameter(0)
  Arg_1.57 = f16[] parameter(1)
  ROOT add.1 = f16[] add(Arg_0.56, Arg_1.57)
}

ENTRY main.82 {
  constant.18 = pred[2,6,128,128]{3,2,1,0} constant({...})
  Arg_0.1 = f16[2,6,128,64]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f16[2,6,64,128]{3,2,1,0} parameter(1), sharding={replicated}
  dot.17 = f16[2,6,128,128]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.22 = f16[] constant(2)
  broadcast.24 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.22), dimensions={}
  multiply.2 = f16[2,6,128,128]{3,2,1,0} multiply(dot.17, broadcast.24)
  constant.19 = f16[] constant(1)
  broadcast.13 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.19), dimensions={}
  add.3 = f16[2,6,128,128]{3,2,1,0} add(multiply.2, broadcast.13)
  constant.21 = f16[] constant(0)
  broadcast.23 = f16[2,6,128,128]{3,2,1,0} broadcast(constant.21), dimensions={}
  select.1 = f16[2,6,128,128]{3,2,1,0} select(constant.18, add.3, broadcast.23)
  constant.15 = f16[] constant(-inf)
  reduce.25 = f16[2,6,128]{2,1,0} reduce(select.1, constant.15), dimensions={3}, to_apply=region_0.21
  broadcast.17 = f16[2,6,128,128]{3,2,1,0} broadcast(reduce.25), dimensions={0,1,2}
  subtract.1 = f16[2,6,128,128]{3,2,1,0} subtract(select.1, broadcast.17)
  exponential.1 = f16[2,6,128,128]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,128,128]{3,2,1,0} convert(exponential.1)
  constant.17 = f32[] constant(0)
  reduce.37 = f32[2,6,128]{2,1,0} reduce(convert.5, constant.17), dimensions={3}, to_apply=region_1.33
  convert.9 = f16[2,6,128]{2,1,0} convert(reduce.37)
  broadcast.26 = f16[2,6,128,128]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = f16[2,6,128,128]{3,2,1,0} divide(exponential.1, broadcast.26)
  Arg_2.3 = f16[2,6,128,64]{3,2,1,0} parameter(2), sharding={replicated}
  dot.46 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = f16[2,6,128,64]{3,2,1,0} parameter(3), sharding={replicated}
  dot.49 = f16[2,6,128,128]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  divide.4 = f16[2,6,128,128]{3,2,1,0} divide(dot.49, broadcast.26)
  broadcast.20 = f16[2,6,128]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.3 = f16[2,6,128]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = f16[2,6,128]{2,1,0} divide(broadcast.20, multiply.3)
  broadcast.21 = f16[2,6,128,128]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.4 = f16[2,6,128,128]{3,2,1,0} multiply(dot.49, broadcast.21)
  multiply.5 = f16[2,6,128,128]{3,2,1,0} multiply(multiply.4, exponential.1)
  reduce.59 = f16[2,6,128]{2,1,0} reduce(multiply.5, constant.21), dimensions={3}, to_apply=region_2.55
  negate.2 = f16[2,6,128]{2,1,0} negate(reduce.59)
  broadcast.25 = f16[2,6,128,128]{3,2,1,0} broadcast(negate.2), dimensions={0,1,2}
  add.5 = f16[2,6,128,128]{3,2,1,0} add(divide.4, broadcast.25)
  multiply.8 = f16[2,6,128,128]{3,2,1,0} multiply(add.5, exponential.1)
  select.3 = f16[2,6,128,128]{3,2,1,0} select(constant.18, multiply.8, broadcast.23)
  multiply.9 = f16[2,6,128,128]{3,2,1,0} multiply(select.3, broadcast.24)
  dot.80 = f16[2,6,128,64]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = f16[2,6,64,128]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = f16[2,6,128,64]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.81 = (f16[2,6,128,64]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}, f16[2,6,64,128]{3,2,1,0}, f16[2,6,128,64]{3,2,1,0}) tuple(dot.46, dot.80, dot, dot.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHAScaleBiasMaskSoftmaxCallTarget}),
              0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::GetTupleElement(
              m::CustomCall(&fmha,
                            {kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget}),
              0)
              .WithShape(F16, {2, 6, 128, 64}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall(
                      {kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget}),
                  1))
              .WithShape(F16, {2, 6, 64, 128}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget}),
              2)
              .WithShape(F16, {2, 6, 128, 64}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
}

TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm1ScaleBiasSoftmaxDropoutBmm2DbiasShouldHaveUserShape) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0},bf16[1,16,256,256]{3,2,1,0},pred[16,1,256,256]{3,2,1,0},bf16[16,256,16,64]{3,2,1,0})->(bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[1,16,256,256]{3,2,1,0})}

region_0.54 {
  Arg_0.55 = bf16[] parameter(0)
  Arg_1.56 = bf16[] parameter(1)
  ROOT maximum.57 = bf16[] maximum(Arg_0.55, Arg_1.56)
}

region_1.66 {
  Arg_0.67 = f32[] parameter(0)
  Arg_1.68 = f32[] parameter(1)
  ROOT add.69 = f32[] add(Arg_0.67, Arg_1.68)
}

region_2.114 {
  Arg_0.115 = bf16[] parameter(0)
  Arg_1.116 = bf16[] parameter(1)
  ROOT add.117 = bf16[] add(Arg_0.115, Arg_1.116)
}

ENTRY main.146 {
  Arg_2.3 = bf16[16,256,16,64]{3,2,1,0} parameter(2), sharding={replicated}
  copy = bf16[16,256,16,64]{1,3,2,0} copy(Arg_2.3), sharding={replicated}
  transpose.5 = bf16[16,16,64,256]{3,2,1,0} transpose(copy), dimensions={0,2,3,1}
  Arg_0.1 = bf16[16,256,16,64]{3,2,1,0} parameter(0), sharding={replicated}
  copy.1 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_0.1), sharding={replicated}
  transpose = bf16[16,16,256,64]{3,2,1,0} transpose(copy.1), dimensions={0,2,1,3}
  Arg_1.2 = bf16[16,256,16,64]{3,2,1,0} parameter(1), sharding={replicated}
  copy.2 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.1 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.2), dimensions={0,2,3,1}
  dot = bf16[16,16,256,256]{3,2,1,0} dot(transpose, transpose.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_4.5 = pred[16,1,256,256]{3,2,1,0} parameter(4), sharding={replicated}
  convert.35 = s32[16,1,256,256]{3,2,1,0} convert(Arg_4.5)
  constant.28 = s32[] constant(0)
  broadcast.29 = s32[16,1,256,256]{3,2,1,0} broadcast(constant.28), dimensions={}
  compare.36 = pred[16,1,256,256]{3,2,1,0} compare(convert.35, broadcast.29), direction=GT
  constant.30 = bf16[] constant(0)
  broadcast.1 = bf16[16,1,256,256]{3,2,1,0} broadcast(constant.30), dimensions={}
  constant.10 = bf16[] constant(-9.999e+09)
  broadcast.3 = bf16[16,1,256,256]{3,2,1,0} broadcast(constant.10), dimensions={}
  select.39 = bf16[16,1,256,256]{3,2,1,0} select(compare.36, broadcast.1, broadcast.3)
  reshape.41 = bf16[16,256,256]{2,1,0} reshape(select.39)
  broadcast.42 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.41), dimensions={0,2,3}
  Arg_3.4 = bf16[1,16,256,256]{3,2,1,0} parameter(3), sharding={replicated}
  reshape.44 = bf16[16,256,256]{2,1,0} reshape(Arg_3.4)
  broadcast.45 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.44), dimensions={1,2,3}
  add.46 = bf16[16,16,256,256]{3,2,1,0} add(broadcast.42, broadcast.45)
  add.53 = bf16[16,16,256,256]{3,2,1,0} add(dot, add.46)
  constant.31 = bf16[] constant(-inf)
  reduce.58 = bf16[16,16,256]{2,1,0} reduce(add.53, constant.31), dimensions={3}, to_apply=region_0.54
  broadcast.62 = bf16[16,16,256,256]{3,2,1,0} broadcast(reduce.58), dimensions={0,1,2}
  subtract.63 = bf16[16,16,256,256]{3,2,1,0} subtract(add.53, broadcast.62)
  exponential.64 = bf16[16,16,256,256]{3,2,1,0} exponential(subtract.63)
  convert.65 = f32[16,16,256,256]{3,2,1,0} convert(exponential.64)
  constant.11 = f32[] constant(0)
  reduce.70 = f32[16,16,256]{2,1,0} reduce(convert.65, constant.11), dimensions={3}, to_apply=region_1.66
  convert.4 = bf16[16,16,256]{2,1,0} convert(reduce.70)
  broadcast.75 = bf16[16,16,256,256]{3,2,1,0} broadcast(convert.4), dimensions={0,1,2}
  divide.76 = bf16[16,16,256,256]{3,2,1,0} divide(exponential.64, broadcast.75)
  constant.22 = u32[1]{0} constant({255383827})
  constant.21 = u32[1]{0} constant({267815257})
  constant.2 = u32[1]{0} constant({0})
  constant.23 = u32[1]{0} constant({3213575472})
  custom-call.49 = (u32[1]{0}, u32[1]{0}) custom-call(constant.22, constant.21, constant.2, constant.23), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[1]{0}, u32[1]{0}, u32[1]{0}, u32[1]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\001\000\000\000\000\000\000\000"
  get-tuple-element.50 = u32[1]{0} get-tuple-element(custom-call.49), index=0
  reshape.80 = u32[] reshape(get-tuple-element.50)
  broadcast.84 = u32[32768]{0} broadcast(reshape.80), dimensions={}
  get-tuple-element.51 = u32[1]{0} get-tuple-element(custom-call.49), index=1
  reshape.81 = u32[] reshape(get-tuple-element.51)
  broadcast.85 = u32[32768]{0} broadcast(reshape.81), dimensions={}
  iota.79 = u32[65536]{0} iota(), iota_dimension=0
  slice.82 = u32[32768]{0} slice(iota.79), slice={[0:32768]}
  slice.83 = u32[32768]{0} slice(iota.79), slice={[32768:65536]}
  custom-call.86 = (u32[32768]{0}, u32[32768]{0}) custom-call(broadcast.84, broadcast.85, slice.82, slice.83), custom_call_target="cu_threefry2x32", operand_layout_constraints={u32[32768]{0}, u32[32768]{0}, u32[32768]{0}, u32[32768]{0}}, api_version=API_VERSION_STATUS_RETURNING, backend_config="\000\200\000\000\000\000\000\000"
  get-tuple-element.87 = u32[32768]{0} get-tuple-element(custom-call.86), index=0
  get-tuple-element.88 = u32[32768]{0} get-tuple-element(custom-call.86), index=1
  concatenate.89 = u32[65536]{0} concatenate(get-tuple-element.87, get-tuple-element.88), dimensions={0}
  constant.17 = u32[] constant(9)
  broadcast.13 = u32[65536]{0} broadcast(constant.17), dimensions={}
  shift-right-logical.0 = u32[65536]{0} shift-right-logical(concatenate.89, broadcast.13)
  constant.15 = u32[] constant(1065353216)
  broadcast.21 = u32[65536]{0} broadcast(constant.15), dimensions={}
  or.0 = u32[65536]{0} or(shift-right-logical.0, broadcast.21)
  bitcast-convert.0 = f32[65536]{0} bitcast-convert(or.0)
  constant.3 = f32[] constant(-1)
  broadcast.30 = f32[65536]{0} broadcast(constant.3), dimensions={}
  add.1 = f32[65536]{0} add(bitcast-convert.0, broadcast.30)
  broadcast.31 = f32[65536]{0} broadcast(constant.11), dimensions={}
  maximum.0 = f32[65536]{0} maximum(add.1, broadcast.31)
  constant.9 = f32[] constant(0.9)
  broadcast.32 = f32[65536]{0} broadcast(constant.9), dimensions={}
  compare.0 = pred[65536]{0} compare(maximum.0, broadcast.32), direction=LT
  constant = bf16[] constant(1.109)
  broadcast.33 = bf16[65536]{0} broadcast(constant), dimensions={}
  broadcast.34 = bf16[65536]{0} broadcast(constant.30), dimensions={}
  select.2 = bf16[65536]{0} select(compare.0, broadcast.33, broadcast.34)
  reshape.39 = bf16[16,16,256]{2,1,0} reshape(select.2)
  broadcast.9 = bf16[16,16,256,256]{3,2,1,0} broadcast(reshape.39), dimensions={0,1,3}
  multiply.101 = bf16[16,16,256,256]{3,2,1,0} multiply(divide.76, broadcast.9)
  dot.1 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.5, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.103 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.1), dimensions={0,3,1,2}
  Arg_5.6 = bf16[16,256,16,64]{3,2,1,0} parameter(5), sharding={replicated}
  copy.3 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.4 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.3), dimensions={0,2,1,3}
  dot.2 = bf16[16,16,256,256]{3,2,1,0} dot(transpose.4, transpose.5), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  multiply.108 = bf16[16,16,256,256]{3,2,1,0} multiply(dot.2, broadcast.9)
  divide.124 = bf16[16,16,256,256]{3,2,1,0} divide(multiply.108, broadcast.75)
  constant.19 = bf16[] constant(1)
  broadcast.24 = bf16[16,16,256]{2,1,0} broadcast(constant.19), dimensions={}
  multiply.2 = bf16[16,16,256]{2,1,0} multiply(convert.4, convert.4)
  divide.0 = bf16[16,16,256]{2,1,0} divide(broadcast.24, multiply.2)
  broadcast.111 = bf16[16,16,256,256]{3,2,1,0} broadcast(divide.0), dimensions={0,1,2}
  multiply.112 = bf16[16,16,256,256]{3,2,1,0} multiply(multiply.108, broadcast.111)
  multiply.113 = bf16[16,16,256,256]{3,2,1,0} multiply(multiply.112, exponential.64)
  reduce.118 = bf16[16,16,256]{2,1,0} reduce(multiply.113, constant.30), dimensions={3}, to_apply=region_2.114
  negate.1 = bf16[16,16,256]{2,1,0} negate(reduce.118)
  broadcast.11 = bf16[16,16,256,256]{3,2,1,0} broadcast(negate.1), dimensions={0,1,2}
  add.133 = bf16[16,16,256,256]{3,2,1,0} add(divide.124, broadcast.11)
  multiply.134 = bf16[16,16,256,256]{3,2,1,0} multiply(add.133, exponential.64)
  copy.4 = bf16[16,256,16,64]{3,1,2,0} copy(Arg_1.2), sharding={replicated}
  transpose.9 = bf16[16,16,256,64]{3,2,1,0} transpose(copy.4), dimensions={0,2,1,3}
  dot.4 = bf16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose.9), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.144 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.4), dimensions={0,2,1,3}
  dot.3 = bf16[16,16,256,64]{3,2,1,0} dot(multiply.134, transpose), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.142 = bf16[16,256,16,64]{3,1,2,0} transpose(dot.3), dimensions={0,2,1,3}
  copy.5 = bf16[16,256,16,64]{1,3,2,0} copy(Arg_5.6), sharding={replicated}
  transpose.104 = bf16[16,16,64,256]{3,2,1,0} transpose(copy.5), dimensions={0,2,3,1}
  dot.106 = bf16[16,16,64,256]{3,2,1,0} dot(transpose.104, multiply.101), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.107 = bf16[16,256,16,64]{1,3,2,0} transpose(dot.106), dimensions={0,3,1,2}
  reduce.139 = bf16[16,256,256]{2,1,0} reduce(multiply.134, constant.30), dimensions={0}, to_apply=region_2.114
  bitcast.111 = bf16[1,16,256,256]{3,2,1,0} bitcast(reduce.139)
  all-reduce = bf16[1,16,256,256]{3,2,1,0} all-reduce(bitcast.111), channel_id=85, replica_groups={{0}}, to_apply=region_2.114
  tuple.145 = (bf16[16,256,16,64]{1,3,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{3,1,2,0}, bf16[16,256,16,64]{1,3,2,0}, bf16[1,16,256,256]{3,2,1,0}) tuple(transpose.103, transpose.144, transpose.142, transpose.107, all-reduce)
  get-tuple-element = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=0
  copy.6 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element)
  get-tuple-element.1 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=1
  copy.7 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.1)
  get-tuple-element.2 = bf16[16,256,16,64]{3,1,2,0} get-tuple-element(tuple.145), index=2
  copy.8 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.2)
  get-tuple-element.3 = bf16[16,256,16,64]{1,3,2,0} get-tuple-element(tuple.145), index=3
  copy.9 = bf16[16,256,16,64]{3,2,1,0} copy(get-tuple-element.3)
  get-tuple-element.4 = bf16[1,16,256,256]{3,2,1,0} get-tuple-element(tuple.145), index=4
  ROOT tuple = (bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[16,256,16,64]{3,2,1,0}, bf16[1,16,256,256]{3,2,1,0}) tuple(copy.6, copy.7, copy.8, copy.9, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  const absl::string_view backward_target =
      kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
  auto dbias_index = 5;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Copy(m::GetTupleElement(
              m::Tuple(
                  m::Transpose().WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(m::GetTupleElement(
                                   m::CustomCall(&fmha, {backward_target}), 0))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(
                      m::GetTupleElement(m::CustomCall({backward_target}), 1))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::Transpose(m::Transpose(m::GetTupleElement(
                                   m::CustomCall({backward_target}), 2)))
                      .WithShape(BF16, {16, 256, 16, 64}),
                  m::AllReduce(m::Bitcast(
                      m::Reshape(
                          m::GetTupleElement(  // dbias
                              m::CustomCall({backward_target}), dbias_index))
                          .WithShape(BF16, {16, 256, 256})))),
              0)),
          m::Op(), m::Op(), m::Op(), m::Op())));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 5);
  EXPECT_NEAR(config.dropout_rate(), 0.1, 1e-2);
}


// flash attention
TEST_F(CudnnFusedMhaRewriterTestHloTest,
       BF16TrainingBmm1CausalMaskSoftmaxBmm2Pattern) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,128,2048]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0},bf16[2,6,2048,128]{3,2,1,0})->(bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_0.32 {
  Arg_0.33 = bf16[] parameter(0)
  Arg_1.34 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(Arg_0.33, Arg_1.34)
}

region_1.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0.45, Arg_1.46)
}

region_2.66 {
  Arg_0.67 = bf16[] parameter(0)
  Arg_1.68 = bf16[] parameter(1)
  ROOT add.1 = bf16[] add(Arg_0.67, Arg_1.68)
}

ENTRY main.92 {
  Arg_0.1 = bf16[2,6,2048,128]{3,2,1,0} parameter(0), sharding={replicated}
  Arg_1.2 = bf16[2,6,128,2048]{3,2,1,0} parameter(1), sharding={replicated}
  dot.14 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_0.1, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  constant.17 = bf16[] constant(2)
  broadcast.29 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(constant.17), dimensions={}
  multiply.2 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.14, broadcast.29)
  iota.2 = s32[2048,2048]{1,0} iota(), iota_dimension=0
  iota.5 = s32[2048,2048]{1,0} iota(), iota_dimension=1
  compare.1 = pred[2048,2048]{1,0} compare(iota.2, iota.5), direction=LT
  constant.6 = bf16[] constant(-2.366e+38)
  broadcast.16 = bf16[2048,2048]{1,0} broadcast(constant.6), dimensions={}
  constant.16 = bf16[] constant(0)
  broadcast.17 = bf16[2048,2048]{1,0} broadcast(constant.16), dimensions={}
  select.2 = bf16[2048,2048]{1,0} select(compare.1, broadcast.16, broadcast.17)
  broadcast.19 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(select.2), dimensions={2,3}
  add.3 = bf16[2,6,2048,2048]{3,2,1,0} add(multiply.2, broadcast.19)
  constant.10 = bf16[] constant(-inf)
  reduce.36 = bf16[2,6,2048]{2,1,0} reduce(add.3, constant.10), dimensions={3}, to_apply=region_0.32
  broadcast.21 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(reduce.36), dimensions={0,1,2}
  subtract.1 = bf16[2,6,2048,2048]{3,2,1,0} subtract(add.3, broadcast.21)
  exponential.1 = bf16[2,6,2048,2048]{3,2,1,0} exponential(subtract.1)
  convert.5 = f32[2,6,2048,2048]{3,2,1,0} convert(exponential.1)
  constant.14 = f32[] constant(0)
  reduce.48 = f32[2,6,2048]{2,1,0} reduce(convert.5, constant.14), dimensions={3}, to_apply=region_1.44
  convert.9 = bf16[2,6,2048]{2,1,0} convert(reduce.48)
  broadcast.32 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(convert.9), dimensions={0,1,2}
  divide.5 = bf16[2,6,2048,2048]{3,2,1,0} divide(exponential.1, broadcast.32)
  Arg_2.3 = bf16[2,6,2048,128]{3,2,1,0} parameter(2), sharding={replicated}
  dot.57 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  Arg_3.4 = bf16[2,6,2048,128]{3,2,1,0} parameter(3), sharding={replicated}
  dot.60 = bf16[2,6,2048,2048]{3,2,1,0} dot(Arg_3.4, Arg_2.3), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  divide.4 = bf16[2,6,2048,2048]{3,2,1,0} divide(dot.60, broadcast.32)
  constant.15 = bf16[] constant(1)
  broadcast.25 = bf16[2,6,2048]{2,1,0} broadcast(constant.15), dimensions={}
  multiply.3 = bf16[2,6,2048]{2,1,0} multiply(convert.9, convert.9)
  divide.3 = bf16[2,6,2048]{2,1,0} divide(broadcast.25, multiply.3)
  broadcast.26 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.4 = bf16[2,6,2048,2048]{3,2,1,0} multiply(dot.60, broadcast.26)
  multiply.5 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.4, exponential.1)
  reduce.70 = bf16[2,6,2048]{2,1,0} reduce(multiply.5, constant.16), dimensions={3}, to_apply=region_2.66
  negate.2 = bf16[2,6,2048]{2,1,0} negate(reduce.70)
  broadcast.31 = bf16[2,6,2048,2048]{3,2,1,0} broadcast(negate.2), dimensions={0,1,2}
  add.5 = bf16[2,6,2048,2048]{3,2,1,0} add(divide.4, broadcast.31)
  multiply.8 = bf16[2,6,2048,2048]{3,2,1,0} multiply(add.5, exponential.1)
  multiply.9 = bf16[2,6,2048,2048]{3,2,1,0} multiply(multiply.8, broadcast.29)
  dot.90 = bf16[2,6,2048,128]{3,2,1,0} dot(multiply.9, Arg_1.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  dot = bf16[2,6,128,2048]{3,2,1,0} dot(Arg_0.1, multiply.9), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  dot.1 = bf16[2,6,2048,128]{3,2,1,0} dot(divide.5, Arg_3.4), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  ROOT tuple.91 = (bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}, bf16[2,6,128,2048]{3,2,1,0}, bf16[2,6,2048,128]{3,2,1,0}) tuple(dot.57, dot.90, dot, dot.1)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  CudnnFusedMHARewriter fusedMhaRewriter{
      GetCudaComputeCapability(),
      GetCudnnVersionWithDbiasAndMaskBwdInputSupport()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());
  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  const HloInstruction* fmha;
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHASoftmaxCallTarget}), 0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::GetTupleElement(
              m::CustomCall(&fmha, {kCudnnfMHASoftmaxBackwardCallTarget}), 0)
              .WithShape(BF16, {2, 6, 2048, 128}),
          m::Transpose(
              m::GetTupleElement(
                  m::CustomCall({kCudnnfMHASoftmaxBackwardCallTarget}), 1))
              .WithShape(BF16, {2, 6, 128, 2048}),
          m::GetTupleElement(
              m::CustomCall({kCudnnfMHASoftmaxBackwardCallTarget}), 2)
              .WithShape(BF16, {2, 6, 2048, 128}))));
  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          fmha->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(fmha->operands().size(), 6);
  EXPECT_NEAR(config.dropout_rate(), 0, 1e-2);
  EXPECT_EQ(config.is_flash_attention(), true);
}

// flash attention
// GPT3 pattern
TEST_F(CudnnFusedMhaRewriterTestHloTest, BF16TrainingGPT3) {
  const char* module_str = R"(
HloModule jit__unnamed_wrapped_function_, entry_computation_layout={((s32[], bf16[4,2048,768]{2,1,0}, bf16[12,3072]{1,0}, bf16[12,768,3072]{2,1,0}, bf16[12,768]{1,0}, bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, bf16[12,3072]{1,0}, bf16[12,768,3072]{2,1,0}, bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, bf16[12,4,2048,768]{3,2,1,0}, bf16[4,1,2048,2048]{3,2,0,1}, bf16[4,2048]{1,0}))->(s32[], bf16[4,2048,768]{2,1,0}, bf16[12,3072]{1,0}, bf16[12,768,3072]{2,1,0}, bf16[12,768]{1,0}, bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, bf16[12,3072]{1,0}, bf16[12,768,3072]{2,1,0}, bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, bf16[12,4,2048,768]{3,2,1,0}, bf16[4,1,2048,2048]{3,2,0,1}, bf16[4,2048]{1,0})}

region_8.643 {
  Arg_0.644 = f32[] parameter(0)
  Arg_1.645 = f32[] parameter(1)
  ROOT add.646 = f32[] add(Arg_0.644, Arg_1.645)
}

region_23.860 {
  Arg_0.861 = bf16[] parameter(0)
  Arg_1.862 = bf16[] parameter(1)
  ROOT add.863 = bf16[] add(Arg_0.861, Arg_1.862)
}

region_33.931 {
  Arg_0.932 = f32[] parameter(0)
  Arg_1.933 = f32[] parameter(1)
  ROOT maximum.934 = f32[] maximum(Arg_0.932, Arg_1.933)
}

ENTRY main.92 {
  arg_tuple.1060 = (s32[], bf16[4,2048,768]{2,1,0}, bf16[12,3072]{1,0}, bf16[12,768,3072]{2,1,0}, bf16[12,768]{1,0}, /*index=5*/bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, /*index=10*/bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, bf16[12,3072]{1,0}, /*index=15*/bf16[12,768,3072]{2,1,0}, bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, /*index=20*/bf16[12,768]{1,0}, bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, /*index=25*/bf16[12,4,2048,768]{3,2,1,0}, bf16[4,1,2048,2048]{3,2,0,1}, bf16[4,2048]{1,0}) parameter(0)
  get-tuple-element.1061 = s32[] get-tuple-element(arg_tuple.1060), index=0
  constant.1121 = s32[] constant(1)
  add.1568 = s32[] add(get-tuple-element.1061, constant.1121)
  get-tuple-element.1062 = bf16[4,2048,768]{2,1,0} get-tuple-element(arg_tuple.1060), index=1
  get-tuple-element.1083 = bf16[12,3,768,12,64]{4,3,2,1,0} get-tuple-element(arg_tuple.1060), index=22
  constant.178 = s32[] constant(11)
  subtract.6 = s32[] subtract(constant.178, get-tuple-element.1061)
  constant.1120 = s32[] constant(0)
  compare.1161 = pred[] compare(subtract.6, constant.1120), direction=LT
  constant.205 = s32[] constant(23)
  subtract.10 = s32[] subtract(constant.205, get-tuple-element.1061)
  select.1163 = s32[] select(compare.1161, subtract.10, subtract.6)
  dynamic-slice.1164 = bf16[1,3,768,12,64]{4,3,2,1,0} dynamic-slice(get-tuple-element.1083, select.1163, constant.1120, constant.1120, constant.1120, /*index=5*/constant.1120), dynamic_slice_sizes={1,3,768,12,64}
  reshape.1165 = bf16[3,768,12,64]{3,2,1,0} reshape(dynamic-slice.1164)
  transpose.12 = bf16[3,12,64,768]{2,1,3,0} transpose(reshape.1165), dimensions={0,2,3,1}
  reshape.35 = bf16[2304,768]{1,0} reshape(transpose.12)
  get-tuple-element.1086 = bf16[12,4,2048,768]{3,2,1,0} get-tuple-element(arg_tuple.1060), index=25
  dynamic-slice.1178 = bf16[1,4,2048,768]{3,2,1,0} dynamic-slice(get-tuple-element.1086, select.1163, constant.1120, constant.1120, constant.1120), dynamic_slice_sizes={1,4,2048,768}
  reshape.1179 = bf16[4,2048,768]{2,1,0} reshape(dynamic-slice.1178)
  convert.1196 = f32[4,2048,768]{2,1,0} convert(reshape.1179)
  constant.1117 = f32[] constant(0)
  reduce.1197 = f32[4,2048]{1,0} reduce(convert.1196, constant.1117), dimensions={2}, to_apply=region_8.643
  constant.41 = f32[] constant(0.00130208337)
  broadcast.367 = f32[4,2048]{1,0} broadcast(constant.41), dimensions={}
  multiply.44 = f32[4,2048]{1,0} multiply(reduce.1197, broadcast.367)
  broadcast.1211 = f32[4,2048,768]{2,1,0} broadcast(multiply.44), dimensions={0,1}
  subtract.1212 = f32[4,2048,768]{2,1,0} subtract(convert.1196, broadcast.1211)
  multiply.1204 = f32[4,2048,768]{2,1,0} multiply(subtract.1212, subtract.1212)
  reduce.1206 = f32[4,2048]{1,0} reduce(multiply.1204, constant.1117), dimensions={2}, to_apply=region_8.643
  multiply.45 = f32[4,2048]{1,0} multiply(reduce.1206, broadcast.367)
  constant.1111 = f32[] constant(1e-05)
  broadcast.469 = f32[4,2048]{1,0} broadcast(constant.1111), dimensions={}
  add.92 = f32[4,2048]{1,0} add(multiply.45, broadcast.469)
  reshape.779 = f32[4,2048,1]{1,0,2} reshape(add.92)
  rsqrt.1214 = f32[4,2048,1]{1,0,2} rsqrt(reshape.779)
  reshape.1218 = f32[4,2048]{1,0} reshape(rsqrt.1214)
  broadcast.1219 = f32[4,2048,768]{2,1,0} broadcast(reshape.1218), dimensions={0,1}
  multiply.1220 = f32[4,2048,768]{2,1,0} multiply(subtract.1212, broadcast.1219)
  convert.1221 = bf16[4,2048,768]{2,1,0} convert(multiply.1220)
  get-tuple-element.1081 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=20
  dynamic-slice.1155 = bf16[1,768]{1,0} dynamic-slice(get-tuple-element.1081, select.1163, constant.1120), dynamic_slice_sizes={1,768}
  constant.1090 = bf16[] constant(1)
  broadcast.370 = bf16[1,768]{1,0} broadcast(constant.1090), dimensions={}
  add.44 = bf16[1,768]{1,0} add(dynamic-slice.1155, broadcast.370)
  reshape.1225 = bf16[768]{0} reshape(add.44)
  broadcast.1226 = bf16[4,2048,768]{2,1,0} broadcast(reshape.1225), dimensions={2}
  multiply.1227 = bf16[4,2048,768]{2,1,0} multiply(convert.1221, broadcast.1226)
  get-tuple-element.1080 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=19
  dynamic-slice.1151 = bf16[1,768]{1,0} dynamic-slice(get-tuple-element.1080, select.1163, constant.1120), dynamic_slice_sizes={1,768}
  reshape.1230 = bf16[768]{0} reshape(dynamic-slice.1151)
  broadcast.1231 = bf16[4,2048,768]{2,1,0} broadcast(reshape.1230), dimensions={2}
  add.1232 = bf16[4,2048,768]{2,1,0} add(multiply.1227, broadcast.1231)
  transpose.13 = bf16[768,4,2048]{0,2,1} transpose(add.1232), dimensions={2,0,1}
  reshape.36 = bf16[768,8192]{0,1} reshape(transpose.13)
  dot.6 = bf16[2304,8192]{1,0} dot(reshape.35, reshape.36), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.37 = bf16[3,12,64,4,2048]{4,2,1,3,0} reshape(dot.6)
  get-tuple-element.1082 = bf16[12,3,12,64]{3,2,1,0} get-tuple-element(arg_tuple.1060), index=21
  dynamic-slice.1160 = bf16[1,3,12,64]{3,2,1,0} dynamic-slice(get-tuple-element.1082, select.1163, constant.1120, constant.1120, constant.1120), dynamic_slice_sizes={1,3,12,64}
  reshape.1237 = bf16[3,12,64]{2,1,0} reshape(dynamic-slice.1160)
  broadcast.372 = bf16[3,12,64,4,2048]{4,2,1,3,0} broadcast(reshape.1237), dimensions={0,1,2}
  add.45 = bf16[3,12,64,4,2048]{4,2,1,3,0} add(reshape.37, broadcast.372)
  transpose.67 = bf16[3,4,2048,12,64]{2,4,3,1,0} transpose(add.45), dimensions={0,3,4,1,2}
  // V
  slice.1244 = bf16[1,4,2048,12,64]{2,4,3,1,0} slice(transpose.67), slice={[2:3], [0:4], [0:2048], [0:12], [0:64]}
  reshape.1245 = bf16[4,2048,12,64]{1,3,2,0} reshape(slice.1244)
  transpose.16 = bf16[4,12,64,2048]{3,2,1,0} transpose(reshape.1245), dimensions={0,2,3,1}
  // Q
  slice.1240 = bf16[1,4,2048,12,64]{2,4,3,1,0} slice(transpose.67), slice={[0:1], [0:4], [0:2048], [0:12], [0:64]}
  constant.1105 = bf16[] constant(0.125)
  broadcast.374 = bf16[1,4,2048,12,64]{2,4,3,1,0} broadcast(constant.1105), dimensions={}
  multiply.42 = bf16[1,4,2048,12,64]{2,4,3,1,0} multiply(slice.1240, broadcast.374)
  reshape.458 = bf16[4,2048,12,64]{1,3,2,0} reshape(multiply.42)
  transpose.14 = bf16[4,12,2048,64]{2,3,1,0} transpose(reshape.458), dimensions={0,2,1,3}
  copy = bf16[4,12,2048,64]{3,2,1,0} copy(transpose.14)
  // K
  slice.1242 = bf16[1,4,2048,12,64]{2,4,3,1,0} slice(transpose.67), slice={[1:2], [0:4], [0:2048], [0:12], [0:64]}
  reshape.1243 = bf16[4,2048,12,64]{1,3,2,0} reshape(slice.1242)
  transpose.15 = bf16[4,12,64,2048]{3,2,1,0} transpose(reshape.1243), dimensions={0,2,3,1}
  // Q K -> S
  dot.7 = bf16[4,12,2048,2048]{3,2,1,0} dot(copy, transpose.15), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  convert.1251 = f32[4,12,2048,2048]{3,2,1,0} convert(dot.7)
  get-tuple-element.1087 = bf16[4,1,2048,2048]{3,2,0,1} get-tuple-element(arg_tuple.1060), index=26
  // causal mask
  iota.6 = s32[2048,2048]{1,0} iota(), iota_dimension=0
  iota.9 = s32[2048,2048]{1,0} iota(), iota_dimension=1
  compare.1188 = pred[2048,2048]{1,0} compare(iota.6, iota.9), direction=LT
  constant.1118 = bf16[] constant(-2.366e+38)
  broadcast.1119 = bf16[2048,2048]{1,0} broadcast(constant.1118), dimensions={}
  constant.1089 = bf16[] constant(0)
  broadcast.284 = bf16[2048,2048]{1,0} broadcast(constant.1089), dimensions={}
  select.168 = bf16[2048,2048]{1,0} select(compare.1188, broadcast.1119, broadcast.284)
  broadcast.1194 = bf16[4,1,2048,2048]{3,2,0,1} broadcast(select.168), dimensions={2,3}
  minimum.1195 = bf16[4,1,2048,2048]{3,2,0,1} minimum(get-tuple-element.1087, broadcast.1194)
  reshape.1247 = bf16[4,2048,2048]{2,1,0} reshape(minimum.1195)
  convert.96 = f32[4,2048,2048]{2,1,0} convert(reshape.1247)
  broadcast.1255 = f32[4,12,2048,2048]{3,2,1,0} broadcast(convert.96), dimensions={0,2,3}
  add.1256 = f32[4,12,2048,2048]{3,2,1,0} add(convert.1251, broadcast.1255)
  // softmax
  constant.1104 = f32[] constant(-inf)
  reduce.1257 = f32[4,12,2048]{2,1,0} reduce(add.1256, constant.1104), dimensions={3}, to_apply=region_33.931
  broadcast.1261 = f32[4,12,2048,2048]{3,2,1,0} broadcast(reduce.1257), dimensions={0,1,2}
  subtract.1262 = f32[4,12,2048,2048]{3,2,1,0} subtract(add.1256, broadcast.1261)
  exponential.1263 = f32[4,12,2048,2048]{3,2,1,0} exponential(subtract.1262)
  reduce.1264 = f32[4,12,2048]{2,1,0} reduce(exponential.1263, constant.1117), dimensions={3}, to_apply=region_8.643
  broadcast.1268 = f32[4,12,2048,2048]{3,2,1,0} broadcast(reduce.1264), dimensions={0,1,2}
  divide.1269 = f32[4,12,2048,2048]{3,2,1,0} divide(exponential.1263, broadcast.1268)
  convert.1272 = bf16[4,12,2048,2048]{3,2,1,0} convert(divide.1269)
  // V P -> O
  dot.8 = bf16[4,12,64,2048]{3,2,1,0} dot(transpose.16, convert.1272), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  transpose.18 = bf16[4,2048,64,12]{1,2,3,0} transpose(dot.8), dimensions={0,3,2,1}
  reshape.44 = bf16[8192,768]{1,0} reshape(transpose.18)
  get-tuple-element.1085 = bf16[12,768,12,64]{3,2,1,0} get-tuple-element(arg_tuple.1060), index=24
  dynamic-slice.1173 = bf16[1,768,12,64]{3,2,1,0} dynamic-slice(get-tuple-element.1085, select.1163, constant.1120, constant.1120, constant.1120), dynamic_slice_sizes={1,768,12,64}
  reshape.1174 = bf16[768,12,64]{2,1,0} reshape(dynamic-slice.1173)
  transpose.19 = bf16[64,12,768]{0,1,2} transpose(reshape.1174), dimensions={2,1,0}
  reshape.45 = bf16[768,768]{1,0} reshape(transpose.19)
  dot.9 = bf16[8192,768]{1,0} dot(reshape.44, reshape.45), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  get-tuple-element.1084 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=23
  dynamic-slice.1169 = bf16[1,768]{1,0} dynamic-slice(get-tuple-element.1084, select.1163, constant.1120), dynamic_slice_sizes={1,768}
  reshape.1279 = bf16[768]{0} reshape(dynamic-slice.1169)
  broadcast.383 = bf16[8192,768]{1,0} broadcast(reshape.1279), dimensions={1}
  add.46 = bf16[8192,768]{1,0} add(dot.9, broadcast.383)
  reshape.462 = bf16[4,2048,768]{2,1,0} reshape(add.46)
  add.1283 = bf16[4,2048,768]{2,1,0} add(reshape.462, reshape.1179)
  convert.1285 = f32[4,2048,768]{2,1,0} convert(add.1283)
  reduce.1286 = f32[4,2048]{1,0} reduce(convert.1285, constant.1117), dimensions={2}, to_apply=region_8.643
  multiply.46 = f32[4,2048]{1,0} multiply(reduce.1286, broadcast.367)
  broadcast.1291 = f32[4,2048,768]{2,1,0} broadcast(multiply.46), dimensions={0,1}
  subtract.1292 = f32[4,2048,768]{2,1,0} subtract(convert.1285, broadcast.1291)
  multiply.1293 = f32[4,2048,768]{2,1,0} multiply(subtract.1292, subtract.1292)
  reduce.1295 = f32[4,2048]{1,0} reduce(multiply.1293, constant.1117), dimensions={2}, to_apply=region_8.643
  multiply.47 = f32[4,2048]{1,0} multiply(reduce.1295, broadcast.367)
  add.93 = f32[4,2048]{1,0} add(multiply.47, broadcast.469)
  reshape.785 = f32[4,2048,1]{1,0,2} reshape(add.93)
  rsqrt.1303 = f32[4,2048,1]{1,0,2} rsqrt(reshape.785)
  reshape.1307 = f32[4,2048]{1,0} reshape(rsqrt.1303)
  broadcast.1308 = f32[4,2048,768]{2,1,0} broadcast(reshape.1307), dimensions={0,1}
  multiply.1309 = f32[4,2048,768]{2,1,0} multiply(subtract.1292, broadcast.1308)
  convert.1310 = bf16[4,2048,768]{2,1,0} convert(multiply.1309)
  get-tuple-element.1079 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=18
  dynamic-slice.1146 = bf16[1,768]{1,0} dynamic-slice(get-tuple-element.1079, select.1163, constant.1120), dynamic_slice_sizes={1,768}
  add.47 = bf16[1,768]{1,0} add(dynamic-slice.1146, broadcast.370)
  reshape.1314 = bf16[768]{0} reshape(add.47)
  broadcast.1315 = bf16[4,2048,768]{2,1,0} broadcast(reshape.1314), dimensions={2}
  multiply.1316 = bf16[4,2048,768]{2,1,0} multiply(convert.1310, broadcast.1315)
  get-tuple-element.1078 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=17
  dynamic-slice.1142 = bf16[1,768]{1,0} dynamic-slice(get-tuple-element.1078, select.1163, constant.1120), dynamic_slice_sizes={1,768}
  reshape.1319 = bf16[768]{0} reshape(dynamic-slice.1142)
  broadcast.1320 = bf16[4,2048,768]{2,1,0} broadcast(reshape.1319), dimensions={2}
  add.1321 = bf16[4,2048,768]{2,1,0} add(multiply.1316, broadcast.1320)
  reshape.47 = bf16[8192,768]{1,0} reshape(add.1321)
  get-tuple-element.1076 = bf16[12,768,3072]{2,1,0} get-tuple-element(arg_tuple.1060), index=15
  dynamic-slice.1132 = bf16[1,768,3072]{2,1,0} dynamic-slice(get-tuple-element.1076, select.1163, constant.1120, constant.1120), dynamic_slice_sizes={1,768,3072}
  reshape.1133 = bf16[768,3072]{1,0} reshape(dynamic-slice.1132)
  dot.10 = bf16[8192,3072]{1,0} dot(reshape.47, reshape.1133), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  get-tuple-element.1075 = bf16[12,3072]{1,0} get-tuple-element(arg_tuple.1060), index=14
  dynamic-slice.1128 = bf16[1,3072]{1,0} dynamic-slice(get-tuple-element.1075, select.1163, constant.1120), dynamic_slice_sizes={1,3072}
  reshape.1326 = bf16[3072]{0} reshape(dynamic-slice.1128)
  broadcast.387 = bf16[8192,3072]{1,0} broadcast(reshape.1326), dimensions={1}
  add.48 = bf16[8192,3072]{1,0} add(dot.10, broadcast.387)
  reshape.469 = bf16[4,2048,3072]{2,1,0} reshape(add.48)
  broadcast.389 = bf16[4,2048]{1,0} broadcast(constant.1090), dimensions={}
  get-tuple-element.1088 = bf16[4,2048]{1,0} get-tuple-element(arg_tuple.1060), index=27
  subtract.3 = bf16[4,2048]{1,0} subtract(broadcast.389, get-tuple-element.1088)
  broadcast.1349 = bf16[4,2048,768]{2,1,0} broadcast(subtract.3), dimensions={0,1}
  multiply.1350 = bf16[4,2048,768]{2,1,0} multiply(get-tuple-element.1062, broadcast.1349)
  reshape.53 = bf16[8192,768]{1,0} reshape(multiply.1350)
  get-tuple-element.1077 = bf16[12,3072,768]{2,1,0} get-tuple-element(arg_tuple.1060), index=16
  dynamic-slice.1137 = bf16[1,3072,768]{2,1,0} dynamic-slice(get-tuple-element.1077, select.1163, constant.1120, constant.1120), dynamic_slice_sizes={1,3072,768}
  reshape.1138 = bf16[3072,768]{1,0} reshape(dynamic-slice.1137)
  dot.12 = bf16[8192,3072]{1,0} dot(reshape.53, reshape.1138), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  reshape.55 = bf16[4,2048,3072]{2,1,0} reshape(dot.12)
  broadcast.1360 = bf16[4,2048,3072]{2,1,0} broadcast(subtract.3), dimensions={0,1}
  multiply.1361 = bf16[4,2048,3072]{2,1,0} multiply(reshape.55, broadcast.1360)
  multiply.1362 = bf16[4,2048,3072]{2,1,0} multiply(reshape.469, multiply.1361)
  constant.1092 = bf16[] constant(0.5)
  broadcast.1093 = bf16[4,2048,3072]{2,1,0} broadcast(constant.1092), dimensions={}
  multiply.1363 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1362, broadcast.1093)
  broadcast.1095 = bf16[4,2048,3072]{2,1,0} broadcast(constant.1090), dimensions={}
  multiply.1329 = bf16[4,2048,3072]{2,1,0} multiply(reshape.469, reshape.469)
  multiply.1330 = bf16[4,2048,3072]{2,1,0} multiply(reshape.469, multiply.1329)
  constant.1098 = bf16[] constant(0.04468)
  broadcast.1099 = bf16[4,2048,3072]{2,1,0} broadcast(constant.1098), dimensions={}
  multiply.1333 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1330, broadcast.1099)
  add.1334 = bf16[4,2048,3072]{2,1,0} add(reshape.469, multiply.1333)
  constant.1096 = bf16[] constant(0.7969)
  broadcast.1097 = bf16[4,2048,3072]{2,1,0} broadcast(constant.1096), dimensions={}
  multiply.1335 = bf16[4,2048,3072]{2,1,0} multiply(add.1334, broadcast.1097)
  tanh.1336 = bf16[4,2048,3072]{2,1,0} tanh(multiply.1335)
  subtract.1337 = bf16[4,2048,3072]{2,1,0} subtract(broadcast.1095, tanh.1336)
  multiply.1364 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1363, subtract.1337)
  multiply.1365 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1364, tanh.1336)
  add.1366 = bf16[4,2048,3072]{2,1,0} add(multiply.1364, multiply.1365)
  multiply.1367 = bf16[4,2048,3072]{2,1,0} multiply(add.1366, broadcast.1097)
  constant.66 = bf16[] constant(0.03564)
  broadcast.289 = bf16[4,2048,3072]{2,1,0} broadcast(constant.66), dimensions={}
  multiply.1368 = bf16[4,2048,3072]{2,1,0} multiply(add.1366, broadcast.289)
  constant.1100 = bf16[] constant(3)
  broadcast.1101 = bf16[4,2048,3072]{2,1,0} broadcast(constant.1100), dimensions={}
  multiply.1332 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1329, broadcast.1101)
  multiply.1369 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1368, multiply.1332)
  add.1370 = bf16[4,2048,3072]{2,1,0} add(multiply.1367, multiply.1369)
  add.1338 = bf16[4,2048,3072]{2,1,0} add(tanh.1336, broadcast.1095)
  multiply.1339 = bf16[4,2048,3072]{2,1,0} multiply(add.1338, broadcast.1093)
  multiply.1371 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1361, multiply.1339)
  add.1372 = bf16[4,2048,3072]{2,1,0} add(add.1370, multiply.1371)
  reshape.59 = bf16[8192,3072]{1,0} reshape(add.1372)
  dot.14 = bf16[8192,768]{1,0} dot(reshape.59, reshape.1133), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  reshape.61 = bf16[4,2048,768]{2,1,0} reshape(dot.14)
  multiply.1390 = bf16[4,2048,768]{2,1,0} multiply(reshape.61, broadcast.1315)
  convert.1391 = f32[4,2048,768]{2,1,0} convert(multiply.1390)
  multiply.1392 = f32[4,2048,768]{2,1,0} multiply(subtract.1292, convert.1391)
  reduce.1393 = f32[4,2048]{1,0} reduce(multiply.1392, constant.1117), dimensions={2}, to_apply=region_8.643
  reshape.1394 = f32[4,2048,1]{1,0,2} reshape(reduce.1393)
  divide.1304 = f32[4,2048,1]{1,0,2} divide(rsqrt.1303, reshape.785)
  constant.1109 = f32[] constant(-0.5)
  broadcast.1110 = f32[4,2048,1]{1,0,2} broadcast(constant.1109), dimensions={}
  multiply.1305 = f32[4,2048,1]{1,0,2} multiply(divide.1304, broadcast.1110)
  multiply.1395 = f32[4,2048,1]{1,0,2} multiply(reshape.1394, multiply.1305)
  constant.206 = f32[] constant(0.00260416674)
  broadcast.474 = f32[4,2048,1]{1,0,2} broadcast(constant.206), dimensions={}
  multiply.48 = f32[4,2048,1]{1,0,2} multiply(multiply.1395, broadcast.474)
  reshape.510 = f32[4,2048]{1,0} reshape(multiply.48)
  broadcast.293 = f32[4,2048,768]{2,1,0} broadcast(reshape.510), dimensions={0,1}
  multiply.1399 = f32[4,2048,768]{2,1,0} multiply(subtract.1292, broadcast.293)
  multiply.1406 = f32[4,2048,768]{2,1,0} multiply(convert.1391, broadcast.1308)
  add.1410 = f32[4,2048,768]{2,1,0} add(multiply.1399, multiply.1406)
  negate.1400 = f32[4,2048,768]{2,1,0} negate(multiply.1399)
  reduce.1401 = f32[4,2048]{1,0} reduce(negate.1400, constant.1117), dimensions={2}, to_apply=region_8.643
  negate.1407 = f32[4,2048,768]{2,1,0} negate(multiply.1406)
  reduce.1408 = f32[4,2048]{1,0} reduce(negate.1407, constant.1117), dimensions={2}, to_apply=region_8.643
  add.49 = f32[4,2048]{1,0} add(reduce.1401, reduce.1408)
  multiply.74 = f32[4,2048]{1,0} multiply(add.49, broadcast.367)
  broadcast.1414 = f32[4,2048,768]{2,1,0} broadcast(multiply.74), dimensions={0,1}
  add.1415 = f32[4,2048,768]{2,1,0} add(add.1410, broadcast.1414)
  convert.1416 = bf16[4,2048,768]{2,1,0} convert(add.1415)
  add.1417 = bf16[4,2048,768]{2,1,0} add(get-tuple-element.1062, convert.1416)
  reshape.65 = bf16[8192,768]{1,0} reshape(add.1417)
  reshape.66 = bf16[768,768]{1,0} reshape(dynamic-slice.1173)
  dot.16 = bf16[8192,768]{1,0} dot(reshape.65, reshape.66), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.67 = bf16[4,2048,12,64]{3,1,2,0} reshape(dot.16)
  transpose.34 = bf16[4,12,2048,64]{3,2,1,0} transpose(reshape.67), dimensions={0,2,1,3}
  // dO V -> dP
  dot.17 = bf16[4,12,2048,2048]{3,2,1,0} dot(transpose.34, transpose.16), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  convert.1427 = f32[4,12,2048,2048]{3,2,1,0} convert(dot.17)
  constant.1102 = f32[] constant(1)
  broadcast.495 = f32[4,12,2048]{2,1,0} broadcast(constant.1102), dimensions={}
  multiply.73 = f32[4,12,2048]{2,1,0} multiply(reduce.1264, reduce.1264)
  divide.3 = f32[4,12,2048]{2,1,0} divide(broadcast.495, multiply.73)
  broadcast.1430 = f32[4,12,2048,2048]{3,2,1,0} broadcast(divide.3), dimensions={0,1,2}
  multiply.1431 = f32[4,12,2048,2048]{3,2,1,0} multiply(convert.1427, broadcast.1430)
  multiply.1432 = f32[4,12,2048,2048]{3,2,1,0} multiply(multiply.1431, exponential.1263)
  reduce.1433 = f32[4,12,2048]{2,1,0} reduce(multiply.1432, constant.1117), dimensions={3}, to_apply=region_8.643
  negate.38 = f32[4,12,2048]{2,1,0} negate(reduce.1433)
  broadcast.1437 = f32[4,12,2048,2048]{3,2,1,0} broadcast(negate.38), dimensions={0,1,2}
  divide.1441 = f32[4,12,2048,2048]{3,2,1,0} divide(convert.1427, broadcast.1268)
  add.1442 = f32[4,12,2048,2048]{3,2,1,0} add(broadcast.1437, divide.1441)
  multiply.1443 = f32[4,12,2048,2048]{3,2,1,0} multiply(add.1442, exponential.1263)
  convert.1444 = bf16[4,12,2048,2048]{3,2,1,0} convert(multiply.1443)
  copy.1 = bf16[4,12,2048,64]{3,2,1,0} copy(transpose.14)
  // dS Q -> dK
  dot.18 = bf16[4,12,2048,64]{3,2,1,0} dot(convert.1444, copy.1), lhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  transpose.1446 = bf16[4,2048,12,64]{3,1,2,0} transpose(dot.18), dimensions={0,2,1,3}
  reshape.1448 = bf16[1,4,2048,12,64]{4,2,3,1,0} reshape(transpose.1446)
  pad.1449 = bf16[3,4,2048,12,64]{4,2,3,1,0} pad(reshape.1448, constant.1089), padding=1_1x0_0x0_0x0_0x0_0
  transpose.39 = bf16[4,12,2048,64]{2,3,1,0} transpose(reshape.1243), dimensions={0,2,1,3}
  copy.2 = bf16[4,12,2048,64]{3,2,1,0} copy(transpose.39)
  // dS K -> dQ
  dot.19 = bf16[4,12,2048,64]{3,2,1,0} dot(convert.1444, copy.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  broadcast.395 = bf16[4,12,2048,64]{3,2,1,0} broadcast(constant.1105), dimensions={}
  multiply.43 = bf16[4,12,2048,64]{3,2,1,0} multiply(dot.19, broadcast.395)
  transpose.70 = bf16[4,2048,12,64]{3,1,2,0} transpose(multiply.43), dimensions={0,2,1,3}
  reshape.1454 = bf16[1,4,2048,12,64]{4,2,3,1,0} reshape(transpose.70)
  pad.1455 = bf16[3,4,2048,12,64]{4,2,3,1,0} pad(reshape.1454, constant.1089), padding=0_2x0_0x0_0x0_0x0_0
  add.1456 = bf16[3,4,2048,12,64]{4,2,3,1,0} add(pad.1449, pad.1455)
  transpose.1425 = bf16[4,12,64,2048]{2,3,1,0} transpose(reshape.67), dimensions={0,2,3,1}
  copy.3 = bf16[4,12,64,2048]{3,2,1,0} copy(transpose.1425)
  // dO P -> dV
  dot.1457 = bf16[4,12,64,2048]{3,2,1,0} dot(copy.3, convert.1272), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  copy.4 = bf16[4,12,64,2048]{2,3,1,0} copy(dot.1457)
  transpose.1458 = bf16[4,2048,12,64]{3,1,2,0} transpose(copy.4), dimensions={0,3,1,2}
  reshape.1460 = bf16[1,4,2048,12,64]{4,2,3,1,0} reshape(transpose.1458)
  pad.1461 = bf16[3,4,2048,12,64]{4,2,3,1,0} pad(reshape.1460, constant.1089), padding=2_0x0_0x0_0x0_0x0_0
  add.1462 = bf16[3,4,2048,12,64]{4,2,3,1,0} add(add.1456, pad.1461)
  transpose.40 = bf16[4,2048,3,12,64]{4,1,3,0,2} transpose(add.1462), dimensions={1,2,0,3,4}
  reshape.77 = bf16[8192,2304]{1,0} reshape(transpose.40)
  dot.20 = bf16[8192,768]{1,0} dot(reshape.77, reshape.35), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.79 = bf16[4,2048,768]{2,1,0} reshape(dot.20)
  multiply.1478 = bf16[4,2048,768]{2,1,0} multiply(reshape.79, broadcast.1226)
  convert.1479 = f32[4,2048,768]{2,1,0} convert(multiply.1478)
  multiply.1480 = f32[4,2048,768]{2,1,0} multiply(subtract.1212, convert.1479)
  reduce.1481 = f32[4,2048]{1,0} reduce(multiply.1480, constant.1117), dimensions={2}, to_apply=region_8.643
  reshape.1482 = f32[4,2048,1]{1,0,2} reshape(reduce.1481)
  divide.1215 = f32[4,2048,1]{1,0,2} divide(rsqrt.1214, reshape.779)
  multiply.1216 = f32[4,2048,1]{1,0,2} multiply(divide.1215, broadcast.1110)
  multiply.1483 = f32[4,2048,1]{1,0,2} multiply(reshape.1482, multiply.1216)
  multiply.49 = f32[4,2048,1]{1,0,2} multiply(multiply.1483, broadcast.474)
  reshape.515 = f32[4,2048]{1,0} reshape(multiply.49)
  broadcast.298 = f32[4,2048,768]{2,1,0} broadcast(reshape.515), dimensions={0,1}
  multiply.1487 = f32[4,2048,768]{2,1,0} multiply(subtract.1212, broadcast.298)
  multiply.1494 = f32[4,2048,768]{2,1,0} multiply(convert.1479, broadcast.1219)
  add.1498 = f32[4,2048,768]{2,1,0} add(multiply.1487, multiply.1494)
  negate.1488 = f32[4,2048,768]{2,1,0} negate(multiply.1487)
  reduce.1489 = f32[4,2048]{1,0} reduce(negate.1488, constant.1117), dimensions={2}, to_apply=region_8.643
  negate.1495 = f32[4,2048,768]{2,1,0} negate(multiply.1494)
  reduce.1496 = f32[4,2048]{1,0} reduce(negate.1495, constant.1117), dimensions={2}, to_apply=region_8.643
  add.50 = f32[4,2048]{1,0} add(reduce.1489, reduce.1496)
  multiply.75 = f32[4,2048]{1,0} multiply(add.50, broadcast.367)
  broadcast.1502 = f32[4,2048,768]{2,1,0} broadcast(multiply.75), dimensions={0,1}
  add.1503 = f32[4,2048,768]{2,1,0} add(add.1498, broadcast.1502)
  convert.1504 = bf16[4,2048,768]{2,1,0} convert(add.1503)
  add.1505 = bf16[4,2048,768]{2,1,0} add(add.1417, convert.1504)
  get-tuple-element.1063 = bf16[12,3072]{1,0} get-tuple-element(arg_tuple.1060), index=2
  reduce.1373 = bf16[3072]{0} reduce(add.1372, constant.1089), dimensions={0,1}, to_apply=region_23.860
  reshape.1508 = bf16[1,3072]{1,0} reshape(reduce.1373)
  dynamic-update-slice.1512 = bf16[12,3072]{1,0} dynamic-update-slice(get-tuple-element.1063, reshape.1508, select.1163, constant.1120)
  get-tuple-element.1064 = bf16[12,768,3072]{2,1,0} get-tuple-element(arg_tuple.1060), index=3
  transpose.26 = bf16[3072,4,2048]{0,2,1} transpose(add.1372), dimensions={2,0,1}
  reshape.56 = bf16[3072,8192]{0,1} reshape(transpose.26)
  dot.29 = bf16[768,3072]{1,0} dot(reshape.47, reshape.56), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  reshape.1513 = bf16[1,768,3072]{2,1,0} reshape(dot.29)
  dynamic-update-slice.1517 = bf16[12,768,3072]{2,1,0} dynamic-update-slice(get-tuple-element.1064, reshape.1513, select.1163, constant.1120, constant.1120)
  get-tuple-element.1065 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=4
  reduce.1351 = bf16[768]{0} reduce(multiply.1350, constant.1089), dimensions={0,1}, to_apply=region_23.860
  reshape.1518 = bf16[1,768]{1,0} reshape(reduce.1351)
  dynamic-update-slice.1522 = bf16[12,768]{1,0} dynamic-update-slice(get-tuple-element.1065, reshape.1518, select.1163, constant.1120)
  get-tuple-element.1066 = bf16[12,3072,768]{2,1,0} get-tuple-element(arg_tuple.1060), index=5
  multiply.1340 = bf16[4,2048,3072]{2,1,0} multiply(reshape.469, multiply.1339)
  multiply.1345 = bf16[4,2048,3072]{2,1,0} multiply(multiply.1340, broadcast.1360)
  reshape.51 = bf16[8192,3072]{1,0} reshape(multiply.1345)
  transpose.22 = bf16[768,4,2048]{0,2,1} transpose(multiply.1350), dimensions={2,0,1}
  reshape.50 = bf16[768,8192]{0,1} reshape(transpose.22)
  dot.30 = bf16[3072,768]{1,0} dot(reshape.51, reshape.50), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  reshape.1523 = bf16[1,3072,768]{2,1,0} reshape(dot.30)
  dynamic-update-slice.1527 = bf16[12,3072,768]{2,1,0} dynamic-update-slice(get-tuple-element.1066, reshape.1523, select.1163, constant.1120, constant.1120)
  get-tuple-element.1067 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=6
  reduce.1380 = bf16[768]{0} reduce(dot.14, constant.1089), dimensions={0}, to_apply=region_23.860
  reshape.1528 = bf16[1,768]{1,0} reshape(reduce.1380)
  dynamic-update-slice.1532 = bf16[12,768]{1,0} dynamic-update-slice(get-tuple-element.1067, reshape.1528, select.1163, constant.1120)
  get-tuple-element.1068 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=7
  multiply.1383 = bf16[4,2048,768]{2,1,0} multiply(convert.1310, reshape.61)
  reduce.1384 = bf16[768]{0} reduce(multiply.1383, constant.1089), dimensions={0,1}, to_apply=region_23.860
  reshape.1533 = bf16[1,768]{1,0} reshape(reduce.1384)
  dynamic-update-slice.1537 = bf16[12,768]{1,0} dynamic-update-slice(get-tuple-element.1068, reshape.1533, select.1163, constant.1120)
  get-tuple-element.1069 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=8
  reduce.1468 = bf16[768]{0} reduce(dot.20, constant.1089), dimensions={0}, to_apply=region_23.860
  reshape.1538 = bf16[1,768]{1,0} reshape(reduce.1468)
  dynamic-update-slice.1542 = bf16[12,768]{1,0} dynamic-update-slice(get-tuple-element.1069, reshape.1538, select.1163, constant.1120)
  get-tuple-element.1070 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=9
  multiply.1471 = bf16[4,2048,768]{2,1,0} multiply(convert.1221, reshape.79)
  reduce.1472 = bf16[768]{0} reduce(multiply.1471, constant.1089), dimensions={0,1}, to_apply=region_23.860
  reshape.1543 = bf16[1,768]{1,0} reshape(reduce.1472)
  dynamic-update-slice.1547 = bf16[12,768]{1,0} dynamic-update-slice(get-tuple-element.1070, reshape.1543, select.1163, constant.1120)
  get-tuple-element.1071 = bf16[12,3,12,64]{3,2,1,0} get-tuple-element(arg_tuple.1060), index=10
  reduce.1463 = bf16[3,12,64]{2,1,0} reduce(add.1462, constant.1089), dimensions={1,2}, to_apply=region_23.860
  reshape.1548 = bf16[1,3,12,64]{3,2,1,0} reshape(reduce.1463)
  dynamic-update-slice.1552 = bf16[12,3,12,64]{3,2,1,0} dynamic-update-slice(get-tuple-element.1071, reshape.1548, select.1163, constant.1120, constant.1120, /*index=5*/constant.1120)
  get-tuple-element.1072 = bf16[12,3,768,12,64]{4,3,2,1,0} get-tuple-element(arg_tuple.1060), index=11
  transpose.42 = bf16[3,12,64,4,2048]{2,4,1,3,0} transpose(add.1462), dimensions={0,3,4,1,2}
  reshape.80 = bf16[2304,8192]{1,0} reshape(transpose.42)
  reshape.81 = bf16[8192,768]{1,0} reshape(add.1232)
  dot.21 = bf16[2304,768]{1,0} dot(reshape.80, reshape.81), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.82 = bf16[3,12,64,768]{2,1,3,0} reshape(dot.21)
  transpose.1507 = bf16[3,768,12,64]{3,2,1,0} transpose(reshape.82), dimensions={0,3,1,2}
  reshape.1553 = bf16[1,3,768,12,64]{4,3,2,1,0} reshape(transpose.1507)
  dynamic-update-slice.1557 = bf16[12,3,768,12,64]{4,3,2,1,0} dynamic-update-slice(get-tuple-element.1072, reshape.1553, select.1163, constant.1120, constant.1120, /*index=5*/constant.1120, constant.1120)
  get-tuple-element.1073 = bf16[12,768]{1,0} get-tuple-element(arg_tuple.1060), index=12
  reduce.1419 = bf16[768]{0} reduce(add.1417, constant.1089), dimensions={0,1}, to_apply=region_23.860
  reshape.1558 = bf16[1,768]{1,0} reshape(reduce.1419)
  dynamic-update-slice.1562 = bf16[12,768]{1,0} dynamic-update-slice(get-tuple-element.1073, reshape.1558, select.1163, constant.1120)
  get-tuple-element.1074 = bf16[12,768,12,64]{3,2,1,0} get-tuple-element(arg_tuple.1060), index=13
  transpose.30 = bf16[768,4,2048]{0,2,1} transpose(add.1417), dimensions={2,0,1}
  reshape.62 = bf16[768,8192]{0,1} reshape(transpose.30)
  transpose.31 = bf16[4,2048,12,64]{1,3,2,0} transpose(dot.8), dimensions={0,3,1,2}
  reshape.63 = bf16[8192,768]{1,0} reshape(transpose.31)
  dot.15 = bf16[768,768]{1,0} dot(reshape.62, reshape.63), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  reshape.1563 = bf16[1,768,12,64]{3,2,1,0} reshape(dot.15)
  dynamic-update-slice.1567 = bf16[12,768,12,64]{3,2,1,0} dynamic-update-slice(get-tuple-element.1074, reshape.1563, select.1163, constant.1120, constant.1120, /*index=5*/constant.1120)
  ROOT tuple.1569 = (s32[], bf16[4,2048,768]{2,1,0}, bf16[12,3072]{1,0}, bf16[12,768,3072]{2,1,0}, bf16[12,768]{1,0}, /*index=5*/bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, /*index=10*/bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, bf16[12,3072]{1,0}, /*index=15*/bf16[12,768,3072]{2,1,0}, bf16[12,3072,768]{2,1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, bf16[12,768]{1,0}, /*index=20*/bf16[12,768]{1,0}, bf16[12,3,12,64]{3,2,1,0}, bf16[12,3,768,12,64]{4,3,2,1,0}, bf16[12,768]{1,0}, bf16[12,768,12,64]{3,2,1,0}, /*index=25*/bf16[12,4,2048,768]{3,2,1,0}, bf16[4,1,2048,2048]{3,2,0,1}, bf16[4,2048]{1,0}) tuple(add.1568, add.1505, dynamic-update-slice.1512, dynamic-update-slice.1517, dynamic-update-slice.1522, /*index=5*/dynamic-update-slice.1527, dynamic-update-slice.1532, dynamic-update-slice.1537, dynamic-update-slice.1542, dynamic-update-slice.1547, /*index=10*/dynamic-update-slice.1552, dynamic-update-slice.1557, dynamic-update-slice.1562, dynamic-update-slice.1567, get-tuple-element.1075, /*index=15*/get-tuple-element.1076, get-tuple-element.1077, get-tuple-element.1078, get-tuple-element.1079, get-tuple-element.1080, /*index=20*/get-tuple-element.1081, get-tuple-element.1082, get-tuple-element.1083, get-tuple-element.1084, get-tuple-element.1085, /*index=25*/get-tuple-element.1086, get-tuple-element.1087, get-tuple-element.1088)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  AlgebraicSimplifierOptions alg_sim_options;
  alg_sim_options.set_supports_non_canonical_dots(false);
  alg_sim_options.set_is_layout_sensitive(true);
  alg_sim_options.set_enable_conv_operand_swap(false);
  AlgebraicSimplifier alge_simp{alg_sim_options};
  ReshapeDecomposer reshape_decomposer;
  LayoutNormalization layout_normalizer;
  HloCSE cse{/*is_layout_sensitive=*/true};
  TF_ASSERT_OK(RunHloPass(&reshape_decomposer, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&layout_normalizer, m.get()).status());
  // TF_ASSERT_OK(RunHloPass(&cse, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&cse, m.get()).status());
  CudnnFusedMHARewriter fusedMhaRewriter{GetCudaComputeCapability(),
                                         GetCudnnVersion()};
  TF_ASSERT_OK(RunHloPass(&fusedMhaRewriter, m.get()).status());

  CudnnFusedMHATransposeFusion fmha_transpose_fusion;

  HloDCE dce;
  TF_ASSERT_OK(RunHloPass(&alge_simp, m.get()).status());
  TF_ASSERT_OK(RunHloPass(&fmha_transpose_fusion, m.get()).status());

  TF_ASSERT_OK(RunHloPass(&dce, m.get()).status());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  HloInstruction* fwd_instruction = nullptr;
  HloInstruction* bwd_instruction = nullptr;
  SCOPED_TRACE(m->ToString());
  for (HloInstruction* instr :
       m->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == kCudnnfMHASoftmaxCallTarget) {
      fwd_instruction = instr;
    }
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == kCudnnfMHASoftmaxBackwardCallTarget) {
      bwd_instruction = instr;
    }
  }
  EXPECT_NE(fwd_instruction, nullptr);
  EXPECT_NE(bwd_instruction, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(
      auto config, fwd_instruction->backend_config<CudnnfMHABackendConfig>());
  EXPECT_EQ(config.is_flash_attention(), true);
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
