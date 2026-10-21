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

#include "xla/backends/gpu/transforms/algebraic_simplifier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuAlgebraicSimplifierTest : public HloHardwareIndependentTestBase {
 public:
  se::CudaComputeCapability Ampere() {
    return se::CudaComputeCapability::Ampere();
  }
  se::CudaComputeCapability Hopper() {
    return se::CudaComputeCapability::Hopper();
  }
};

TEST_F(GpuAlgebraicSimplifierTest, SinkBroadcastOperandsOfChainedAdds) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={1}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  ASSERT_TRUE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::AddAnyOrder(
                  m::Broadcast(m::Add(m::Constant(), m::Constant())),
                  m::Convolution(m::Op(), m::Op()))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkBroadcastOperandsOfChainedAddsWhenDisabled) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={1}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(false);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkBroadcastOperandsOfChainedAddsWithoutConvolution) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      p = bf16[4, 4] parameter(0)
      const0 = bf16[4] constant({0, 0.25, 0.5, 0.75})
      bcast0 = bf16[4,4] broadcast(const0), dimensions={0}
      add0 = bf16[4,4] add(p, bcast0)
      const1 = bf16[4] constant({1, 1.25, 1.5, 1.75})
      bcast1 = bf16[4,4] broadcast(const1), dimensions={0}
      ROOT add1 = bf16[4,4] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(
    GpuAlgebraicSimplifierTest,
    DoNotSinkBroadcastOperandsOfChainedAddsWithMismatchedBroadcastDimensions) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={2}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_X3_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32_x3
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_X6_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(
    GpuAlgebraicSimplifierTest,
    DotToMultiplyRewriteForZeroContractingDimWith_BF16_BF16_F32_X6_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT dot = f32[] dot(a, b),
      algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
  constexpr absl::string_view kPattern = R"(
    CHECK-COUNT-6: %[[partial_result:.*]] = bf16[] multiply
  )";
  ASSERT_OK_AND_ASSIGN(bool matched, RunFileCheck(m->ToString(), kPattern));
  EXPECT_TRUE(matched);
}

class DotOutputScaleFoldingTest : public GpuAlgebraicSimplifierTest {
 protected:
  AlgebraicSimplifierOptions Options(bool enable = true) {
    AlgebraicSimplifierOptions options;
    options.set_enable_dot_output_scale_folding(enable);
    return options;
  }

  absl::StatusOr<bool> Run(HloModule* module, bool enable = true) {
    return GpuAlgebraicSimplifier(Options(enable), Hopper()).Run(module);
  }
};

TEST_F(DotOutputScaleFoldingTest, FoldsScaleIntoDequantizedRhs) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,8,10240] parameter(0)
      qw = s8[4,10240,256] parameter(1)
      w = bf16[4,10240,256] convert(qw)
      dot = bf16[32,8,4,256] dot(x, w), lhs_contracting_dims={2},
        rhs_contracting_dims={1}
      scale = bf16[4,256] parameter(2)
      bscale = bf16[32,8,4,256] broadcast(scale), dimensions={2,3}
      ROOT mul = bf16[32,8,4,256] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* broadcast = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Parameter(0),
                        m::Multiply(m::Convert(m::Parameter(1)),
                                    m::Broadcast(&broadcast, m::Parameter(2))
                                        .WithShape(BF16, {4, 10240, 256})))
                     .WithShape(BF16, {32, 8, 4, 256})))
      << module->ToString();
  EXPECT_THAT(broadcast->dimensions(), ::testing::ElementsAre(0, 2));
}

TEST_F(DotOutputScaleFoldingTest, FoldsScaleIntoDequantizedLhs) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      qw = s8[4,10240,256] parameter(0)
      w = bf16[4,10240,256] convert(qw)
      x = bf16[32,8,10240] parameter(1)
      dot = bf16[4,256,32,8] dot(w, x), lhs_contracting_dims={1},
        rhs_contracting_dims={2}
      scale = bf16[4,256] parameter(2)
      bscale = bf16[4,256,32,8] broadcast(scale), dimensions={0,1}
      ROOT mul = bf16[4,256,32,8] multiply(bscale, dot)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* broadcast = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Multiply(m::Convert(m::Parameter(0)),
                                    m::Broadcast(&broadcast, m::Parameter(2))
                                        .WithShape(BF16, {4, 10240, 256})),
                        m::Parameter(1))))
      << module->ToString();
  EXPECT_THAT(broadcast->dimensions(), ::testing::ElementsAre(0, 2));
}

TEST_F(DotOutputScaleFoldingTest, FoldsScaleThroughTransposeAndReshape) {
  // kv_einsum-like pattern: bsd,ckdh->cbskh with k == 1 removed by a reshape
  // and the output transposed before the scale is applied.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,8,10240] parameter(0)
      qw = s8[2,10240,256] parameter(1)
      w = bf16[2,10240,256] convert(qw)
      dot = bf16[32,8,2,256] dot(x, w), lhs_contracting_dims={2},
        rhs_contracting_dims={1}
      transpose = bf16[2,32,8,256] transpose(dot), dimensions={2,0,1,3}
      reshape = bf16[2,32,8,1,256] reshape(transpose)
      scale = bf16[2,256] parameter(2)
      bscale = bf16[2,32,8,1,256] broadcast(scale), dimensions={0,4}
      ROOT mul = bf16[2,32,8,1,256] multiply(reshape, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* broadcast = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Transpose(m::Dot(
                  m::Parameter(0),
                  m::Multiply(m::Convert(m::Parameter(1)),
                              m::Broadcast(&broadcast, m::Parameter(2))
                                  .WithShape(BF16, {2, 10240, 256})))))));
  EXPECT_THAT(broadcast->dimensions(), ::testing::ElementsAre(0, 2));
}

TEST_F(DotOutputScaleFoldingTest, DropsDegenerateScaleDimensions) {
  // The degenerate dimension of the scale is broadcast along a non-contracting
  // dimension of the LHS; it does not prevent folding into the RHS.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,1,4096] parameter(0)
      qw = s8[4,4096,256] parameter(1)
      w = bf16[4,4096,256] convert(qw)
      dot = bf16[32,1,4,256] dot(x, w), lhs_contracting_dims={2},
        rhs_contracting_dims={1}
      scale = bf16[1,4,256] parameter(2)
      bscale = bf16[32,1,4,256] broadcast(scale), dimensions={1,2,3}
      ROOT mul = bf16[32,1,4,256] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  // Other simplifications may remove the degenerate dimension from the dot, so
  // only check that the scale is applied to the dequantized RHS and no longer
  // to the result.
  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kDot);
  ASSERT_NE(dot, nullptr);
  const HloInstruction* broadcast = nullptr;
  ASSERT_THAT(
      dot->operand(1),
      GmockMatch(m::Multiply(
          m::Convert(m::Parameter(1)),
          m::Broadcast(&broadcast,
                       m::Reshape(m::Parameter(2)).WithShape(BF16, {4, 256}))
              .WithShape(BF16, {4, 4096, 256}))))
      << module->ToString();
  EXPECT_THAT(broadcast->dimensions(), ::testing::ElementsAre(0, 2));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::testing::Not(GmockMatch(m::Multiply())));
}

TEST_F(DotOutputScaleFoldingTest, FoldsScaleIntoTransposedDequantizedLhs) {
  // kv_einsum-like pattern (bsd,ckdh->cbskh) with the quantized weight as the
  // LHS of the dot: the output is transposed and the scale has a degenerate
  // dimension that maps to the degenerate k dimension of the weight.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      qw = s8[2,1,10240,256] parameter(0)
      w = bf16[2,1,10240,256] convert(qw)
      x = bf16[32,8,10240] parameter(1)
      dot = bf16[2,1,256,32,8] dot(w, x), lhs_contracting_dims={2},
        rhs_contracting_dims={2}
      transpose = bf16[2,32,8,1,256] transpose(dot), dimensions={0,3,4,1,2}
      scale = bf16[2,1,256] parameter(2)
      bscale = bf16[2,32,8,1,256] broadcast(scale), dimensions={0,3,4}
      ROOT mul = bf16[2,32,8,1,256] multiply(transpose, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  // Other simplifications may remove the degenerate dimension from the dot, so
  // only check that the scale is applied to the dequantized LHS and no longer
  // to the result.
  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kDot);
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(dot->operand(0),
              GmockMatch(m::Multiply(
                  m::Op(), m::Broadcast(m::Op().WithShape(BF16, {2, 256})))));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(m::Op())));
}

TEST_F(DotOutputScaleFoldingTest, FoldsScaleWhenDequantizedOperandIsReshaped) {
  // The degenerate dimension of the quantized weight is removed by a reshape
  // between the convert and the dot (as DegenerateDimensionRewriter and the
  // dot canonicalization in AlgebraicSimplifier do).
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,8,10240] parameter(0)
      qw = s8[2,1,10240,256] parameter(1)
      w = bf16[2,1,10240,256] convert(qw)
      w_reshape = bf16[2,10240,256] reshape(w)
      dot = bf16[32,8,2,256] dot(x, w_reshape), lhs_contracting_dims={2},
        rhs_contracting_dims={1}
      scale = bf16[2,256] parameter(2)
      bscale = bf16[32,8,2,256] broadcast(scale), dimensions={2,3}
      ROOT mul = bf16[32,8,2,256] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* broadcast = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Parameter(0),
                        m::Multiply(m::Reshape(m::Convert(m::Parameter(1))),
                                    m::Broadcast(&broadcast, m::Parameter(2))
                                        .WithShape(BF16, {2, 10240, 256})))))
      << module->ToString();
  EXPECT_THAT(broadcast->dimensions(), ::testing::ElementsAre(0, 2));
}

TEST_F(DotOutputScaleFoldingTest, FoldsScaleOverBatchAndNonContractingDims) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[2,32,1024] parameter(0)
      qw = s8[2,1024,512] parameter(1)
      w = bf16[2,1024,512] convert(qw)
      dot = bf16[2,32,512] dot(x, w), lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
      scale = bf16[2,512] parameter(2)
      bscale = bf16[2,32,512] broadcast(scale), dimensions={0,2}
      ROOT mul = bf16[2,32,512] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* broadcast = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Parameter(0),
                  m::Multiply(m::Convert(m::Parameter(1)),
                              m::Broadcast(&broadcast, m::Parameter(2))))))
      << module->ToString();
  EXPECT_THAT(broadcast->dimensions(), ::testing::ElementsAre(0, 2));
}

TEST_F(DotOutputScaleFoldingTest, DoesNotFoldWhenDisabled) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,10240] parameter(0)
      qw = s8[10240,1024] parameter(1)
      w = bf16[10240,1024] convert(qw)
      dot = bf16[32,1024] dot(x, w), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
      scale = bf16[1024] parameter(2)
      bscale = bf16[32,1024] broadcast(scale), dimensions={1}
      ROOT mul = bf16[32,1024] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get(), /*enable=*/false));
  EXPECT_FALSE(changed);
}

TEST_F(DotOutputScaleFoldingTest, DoesNotFoldScaleOverOtherOperandDims) {
  // The scale varies along the LHS non-contracting dimension, but only the RHS
  // is dequantized.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,10240] parameter(0)
      qw = s8[10240,1024] parameter(1)
      w = bf16[10240,1024] convert(qw)
      dot = bf16[32,1024] dot(x, w), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
      scale = bf16[32] parameter(2)
      bscale = bf16[32,1024] broadcast(scale), dimensions={0}
      ROOT mul = bf16[32,1024] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotOutputScaleFoldingTest, DoesNotFoldWithoutDequantization) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,10240] parameter(0)
      w32 = f32[10240,1024] parameter(1)
      w = bf16[10240,1024] convert(w32)
      dot = bf16[32,1024] dot(x, w), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
      scale = bf16[1024] parameter(2)
      bscale = bf16[32,1024] broadcast(scale), dimensions={1}
      ROOT mul = bf16[32,1024] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotOutputScaleFoldingTest, DoesNotFoldWhenDotHasOtherUsers) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,10240] parameter(0)
      qw = s8[10240,1024] parameter(1)
      w = bf16[10240,1024] convert(qw)
      dot = bf16[32,1024] dot(x, w), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
      scale = bf16[1024] parameter(2)
      bscale = bf16[32,1024] broadcast(scale), dimensions={1}
      mul = bf16[32,1024] multiply(dot, bscale)
      ROOT tuple = (bf16[32,1024], bf16[32,1024]) tuple(mul, dot)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotOutputScaleFoldingTest, DoesNotFoldWhenDotOutputTypeIsWider) {
  // Folding would require rounding the f32 scale to bf16.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,10240] parameter(0)
      qw = s8[10240,1024] parameter(1)
      w = bf16[10240,1024] convert(qw)
      dot = f32[32,1024] dot(x, w), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
      scale = f32[1024] parameter(2)
      bscale = f32[32,1024] broadcast(scale), dimensions={1}
      ROOT mul = f32[32,1024] multiply(dot, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotOutputScaleFoldingTest, DoesNotFoldThroughMergingReshape) {
  // The reshape merges two non-contracting dimensions of the dequantized
  // operand, so the scale dimension cannot be mapped to a dot dimension.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      x = bf16[32,10240] parameter(0)
      qw = s8[10240,4,256] parameter(1)
      w = bf16[10240,4,256] convert(qw)
      dot = bf16[32,4,256] dot(x, w), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
      reshape = bf16[32,1024] reshape(dot)
      scale = bf16[1024] parameter(2)
      bscale = bf16[32,1024] broadcast(scale), dimensions={1}
      ROOT mul = bf16[32,1024] multiply(reshape, bscale)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla::gpu
