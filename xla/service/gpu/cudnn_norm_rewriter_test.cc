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

#include "xla/service/gpu/cudnn_norm_rewriter.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cudnn/cudnn.h"
#endif

#include "tsl/lib/core/status_test_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/filecheck.h"

namespace xla {
namespace gpu {
namespace {

class CudnnNormRewriterTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

 protected:
  void TestNorm(std::string hlo_text, std::string optimized_hlo) {
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
    MatchOptimizedHlo(hlo_text, optimized_hlo);
  }
};

TEST_F(CudnnNormRewriterTest, LayerNorm2N1) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4] parameter(0)
        multiply3 = f32[2,4] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2] reduce(multiply3, c0), dimensions={1}, to_apply=apply
        c1 = f32[] constant(0.25)
        c1_bcast = f32[2] broadcast(c1), dimensions={}
        multiply9 = f32[2] multiply(reduce1, c1_bcast)
        reduce = f32[2] reduce(input, c0),dimensions={1}, to_apply=apply
        multiply8 = f32[2] multiply(reduce, c1_bcast)
        multiply4 = f32[2] multiply(multiply8, multiply8)
        subtract = f32[2] subtract(multiply9, multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2] broadcast(c2), dimensions={}
        add3 = f32[2] add(subtract, c2_bcast)
        rsqrt1 = f32[2] rsqrt(add3)
        broadcast15 = f32[2,4] broadcast(rsqrt1), dimensions={0}
        broadcast4 = f32[2,4] broadcast(multiply8), dimensions={0}
        subtract1 = f32[2,4] subtract(input, broadcast4)
        multiply6 = f32[2,4] multiply(broadcast15, subtract1)
        scale = f32[4] parameter(1)
        broadcast17 = f32[2,4] broadcast(scale), dimensions={1}
        multiply7 = f32[2,4] multiply(multiply6, broadcast17)
        bias = f32[4] parameter(2)
        broadcast18 = f32[2,4] broadcast(bias), dimensions={1}
        ROOT out = f32[2,4] add(multiply7, broadcast18)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4], scale: f32[4], bias: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[2,4,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[GTE_BITCAST:%[^ ]+]] = f32[2,4]{1,0} bitcast([[GTE]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D3) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        multiply3 = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2,4,6] reduce(multiply3, c0), dimensions={3}, to_apply=apply
        c1 = f32[] constant(0.125)
        c1_bcast = f32[2,4,6] broadcast(c1), dimensions={}
        multiply9 = f32[2,4,6] multiply(reduce1, c1_bcast)
        reduce = f32[2,4,6] reduce(input,c0), dimensions={3}, to_apply=apply
        multiply8 = f32[2,4,6] multiply(reduce, c1_bcast)
        multiply4 = f32[2,4,6] multiply(multiply8, multiply8)
        subtract = f32[2,4,6] subtract(multiply9, multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2,4,6] broadcast(c2), dimensions={}
        add3 = f32[2,4,6] add(subtract, c2_bcast)
        rsqrt1 = f32[2,4,6] rsqrt(add3)
        broadcast15 = f32[2,4,6,8] broadcast(rsqrt1), dimensions={0,1,2}
        broadcast4 = f32[2,4,6,8] broadcast(multiply8), dimensions={0,1,2}
        subtract1 = f32[2,4,6,8] subtract(input, broadcast4)
        multiply6 = f32[2,4,6,8] multiply(broadcast15, subtract1)
        scale = f32[8] parameter(1)
        broadcast17 = f32[2,4,6,8] broadcast(scale), dimensions={3}
        multiply7 = f32[2,4,6,8] multiply(multiply6, broadcast17)
        bias = f32[8] parameter(2)
        broadcast18 = f32[2,4,6,8] broadcast(bias), dimensions={3}
        ROOT out = f32[2,4,6,8] add(multiply7, broadcast18)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4,6,8], scale: f32[8], bias: f32[8]) -> f32[2,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[8]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[48,8,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[GTE_BITCAST:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} bitcast([[GTE]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D2) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        multiply3 = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2,4,8] reduce(multiply3, c0), dimensions={2}, to_apply=apply
        c1 = f32[] constant(0.166667)
        c1_bcast = f32[2,4,8] broadcast(c1), dimensions={}
        multiply9 = f32[2,4,8] multiply(reduce1, c1_bcast)
        reduce = f32[2,4,8] reduce(input,c0), dimensions={2}, to_apply=apply
        multiply8 = f32[2,4,8] multiply(reduce, c1_bcast)
        multiply4 = f32[2,4,8] multiply(multiply8, multiply8)
        subtract = f32[2,4,8] subtract(multiply9, multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2,4,8] broadcast(c2), dimensions={}
        add3 = f32[2,4,8] add(subtract, c2_bcast)
        rsqrt1 = f32[2,4,8] rsqrt(add3)
        broadcast15 = f32[2,4,6,8] broadcast(rsqrt1), dimensions={0,1,3}
        broadcast4 = f32[2,4,6,8] broadcast(multiply8), dimensions={0,1,3}
        subtract1 = f32[2,4,6,8] subtract(input, broadcast4)
        multiply6 = f32[2,4,6,8] multiply(broadcast15, subtract1)
        scale = f32[6] parameter(1)
        broadcast17 = f32[2,4,6,8] broadcast(scale), dimensions={2}
        multiply7 = f32[2,4,6,8] multiply(multiply6, broadcast17)
        bias = f32[6] parameter(2)
        broadcast18 = f32[2,4,6,8] broadcast(bias), dimensions={2}
        ROOT out = f32[2,4,6,8] add(multiply7, broadcast18)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4,6,8], scale: f32[6], bias: f32[6]) -> f32[2,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[2,4,8,6]{3,2,1,0} transpose([[P0]]), dimensions={0,1,3,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[6]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[6]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[64,6,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[FUSION:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} fusion([[GTE]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D12) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        multiply3 = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2,8] reduce(multiply3, c0), dimensions={1,2}, to_apply=apply
        c1 = f32[] constant(0.041667)
        c1_bcast = f32[2,8] broadcast(c1), dimensions={}
        multiply9 = f32[2,8] multiply(reduce1, c1_bcast)
        reduce = f32[2,8] reduce(input,c0), dimensions={1,2}, to_apply=apply
        multiply8 = f32[2,8] multiply(reduce, c1_bcast)
        multiply4 = f32[2,8] multiply(multiply8, multiply8)
        subtract = f32[2,8] subtract(multiply9, multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2,8] broadcast(c2), dimensions={}
        add3 = f32[2,8] add(subtract,c2_bcast)
        rsqrt1 = f32[2,8] rsqrt(add3)
        broadcast15 = f32[2,4,6,8] broadcast(rsqrt1), dimensions={0,3}
        broadcast4 = f32[2,4,6,8] broadcast(multiply8), dimensions={0,3}
        subtract1 = f32[2,4,6,8] subtract(input, broadcast4)
        multiply6 = f32[2,4,6,8] multiply(broadcast15, subtract1)
        scale = f32[4,6] parameter(1)
        broadcast17 = f32[2,4,6,8] broadcast(scale), dimensions={1,2}
        multiply7 = f32[2,4,6,8] multiply(multiply6, broadcast17)
        bias = f32[4,6] parameter(2)
        broadcast18 = f32[2,4,6,8] broadcast(bias), dimensions={1,2}
        ROOT out = f32[2,4,6,8] add(multiply7, broadcast18)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4,6,8], scale: f32[4,6], bias: f32[4,6]) -> f32[2,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[2,8,4,6]{3,2,1,0} transpose([[P0]]), dimensions={0,3,1,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,6]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,4,6,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT  [[FUSION:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} fusion([[GTE]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain2D1) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4] parameter(0)
        multiply3 = f32[2,4] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2] reduce(multiply3, c0), dimensions={1}, to_apply=apply
        c1 = f32[] constant(0.25)
        c1_bcast = f32[2] broadcast(c1), dimensions={}
        multiply9 = f32[2] multiply(reduce1,c1_bcast)
        reduce = f32[2] reduce(input,c0), dimensions={1}, to_apply=apply
        multiply8 = f32[2] multiply(reduce,c1_bcast)
        multiply4 = f32[2] multiply(multiply8,multiply8)
        subtract = f32[2] subtract(multiply9,multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2] broadcast(c2), dimensions={}
        add3 = f32[2] add(subtract,c2_bcast)
        rsqrt1 = f32[2] rsqrt(add3)
        broadcast15 = f32[2,4] broadcast(rsqrt1), dimensions={0}
        broadcast4 = f32[2,4] broadcast(multiply8), dimensions={0}
        subtract1 = f32[2,4] subtract(input,broadcast4)
        multiply6 = f32[2,4] multiply(broadcast15,subtract1)
        scale = f32[4] parameter(1)
        broadcast17 = f32[2,4] broadcast(scale), dimensions={1}
        multiply7 = f32[2,4] multiply(multiply6,broadcast17)
        bias = f32[4] parameter(2)
        broadcast18 = f32[2,4] broadcast(bias), dimensions={1}
        add = f32[2,4] add(multiply7,broadcast18)
        divide = f32[2] divide(rsqrt1, add3)
        ROOT out = (f32[2,4], f32[2], f32[2], f32[2]) tuple(add, multiply8, divide, rsqrt1)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4], scale: f32[4], bias: f32[4]) -> (f32[2,4], f32[2], f32[2], f32[2]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[2,4,1,1]{3,2,1,0}, f32[2,1,1,1]{3,2,1,0}, f32[2,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[GTE0_BITCAST:%[^ ]+]] = f32[2,4]{1,0} bitcast([[GTE0]])
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[2,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2]{0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[2]{0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2]{0} bitcast([[GTE2]])
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}) tuple([[GTE0_BITCAST]], [[GTE1_BITCAST]], [[FUSION]], [[GTE2_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain4D3) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        multiply3 = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2,4,6] reduce(multiply3, c0), dimensions={3}, to_apply=apply
        c1 = f32[] constant(0.125)
        c1_bcast = f32[2,4,6] broadcast(c1), dimensions={}
        multiply9 = f32[2,4,6] multiply(reduce1,c1_bcast)
        reduce = f32[2,4,6] reduce(input,c0), dimensions={3}, to_apply=apply
        multiply8 = f32[2,4,6] multiply(reduce,c1_bcast)
        multiply4 = f32[2,4,6] multiply(multiply8,multiply8)
        subtract = f32[2,4,6] subtract(multiply9,multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2,4,6] broadcast(c2), dimensions={}
        add3 = f32[2,4,6] add(subtract,c2_bcast)
        rsqrt1 = f32[2,4,6] rsqrt(add3)
        broadcast15 = f32[2,4,6,8] broadcast(rsqrt1), dimensions={0,1,2}
        broadcast4 = f32[2,4,6,8] broadcast(multiply8), dimensions={0,1,2}
        subtract1 = f32[2,4,6,8] subtract(input,broadcast4)
        multiply6 = f32[2,4,6,8] multiply(broadcast15,subtract1)
        scale = f32[8] parameter(1)
        broadcast17 = f32[2,4,6,8] broadcast(scale), dimensions={3}
        multiply7 = f32[2,4,6,8] multiply(multiply6,broadcast17)
        bias = f32[8] parameter(2)
        broadcast18 = f32[2,4,6,8] broadcast(bias), dimensions={3}
        add = f32[2,4,6,8] add(multiply7,broadcast18)
        divide = f32[2,4,6] divide(rsqrt1, add3)
        ROOT out = (f32[2,4,6,8], f32[2,4,6], f32[2,4,6], f32[2,4,6]) tuple(add, multiply8, divide, rsqrt1)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4,6,8], scale: f32[8], bias: f32[8]) -> (f32[2,4,6,8], f32[2,4,6], f32[2,4,6], f32[2,4,6]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[8]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[48,8,1,1]{3,2,1,0}, f32[2,4,6,1]{3,2,1,0}, f32[2,4,6,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[GTE0_BITCAST:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} bitcast([[GTE0]])
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[2,4,6,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2,4,6]{2,1,0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,4,6,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[2,4,6]{2,1,0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2,4,6]{2,1,0} bitcast([[GTE2]])
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6]{2,1,0}, f32[2,4,6]{2,1,0}, f32[2,4,6]{2,1,0}) tuple([[GTE0_BITCAST]], [[GTE1_BITCAST]], [[FUSION]], [[GTE2_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain4D12) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Layer norm kernels require Ampere or newer architecture.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        multiply3 = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        reduce1 = f32[2,8] reduce(multiply3, c0), dimensions={1,2}, to_apply=apply
        c1 = f32[] constant(0.041667)
        c1_bcast = f32[2,8] broadcast(c1), dimensions={}
        multiply9 = f32[2,8] multiply(reduce1, c1_bcast)
        reduce = f32[2,8] reduce(input,c0), dimensions={1,2}, to_apply=apply
        multiply8 = f32[2,8] multiply(reduce, c1_bcast)
        multiply4 = f32[2,8] multiply(multiply8, multiply8)
        subtract = f32[2,8] subtract(multiply9, multiply4)
        c2 = f32[] constant(0.001)
        c2_bcast = f32[2,8] broadcast(c2), dimensions={}
        add3 = f32[2,8] add(subtract,c2_bcast)
        rsqrt1 = f32[2,8] rsqrt(add3)
        broadcast15 = f32[2,4,6,8] broadcast(rsqrt1), dimensions={0,3}
        broadcast4 = f32[2,4,6,8] broadcast(multiply8), dimensions={0,3}
        subtract1 = f32[2,4,6,8] subtract(input, broadcast4)
        multiply6 = f32[2,4,6,8] multiply(broadcast15, subtract1)
        scale = f32[4,6] parameter(1)
        broadcast17 = f32[2,4,6,8] broadcast(scale), dimensions={1,2}
        multiply7 = f32[2,4,6,8] multiply(multiply6, broadcast17)
        bias = f32[4,6] parameter(2)
        broadcast18 = f32[2,4,6,8] broadcast(bias), dimensions={1,2}
        add = f32[2,4,6,8] add(multiply7, broadcast18)
        divide = f32[2,8] divide(rsqrt1, add3)
        ROOT out = (f32[2,4,6,8], f32[2,8], f32[2,8], f32[2,8]) tuple(add, multiply8, divide, rsqrt1)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test (input: f32[2,4,6,8], scale: f32[4,6], bias: f32[4,6]) -> (f32[2,4,6,8], f32[2,8], f32[2,8], f32[2,8]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[2,8,4,6]{3,2,1,0} transpose([[P0]]), dimensions={0,3,1,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,6]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,4,6,1]{3,2,1,0}, f32[2,8,1,1]{3,2,1,0}, f32[2,8,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[FUSION0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} fusion([[GTE0]]), kind=kLoop, calls=[[FUSED_COMPUTATION0:%[^ ]+]]
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[2,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2,8]{1,0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[FUSION1:%[^ ]+]] = f32[2,8]{1,0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION1:%[^ ]+]]
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2,8]{1,0} bitcast([[GTE2]])
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,8]{1,0}, f32[2,8]{1,0}, f32[2,8]{1,0}) tuple([[FUSION0]], [[GTE1_BITCAST]], [[FUSION1]], [[GTE2_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
