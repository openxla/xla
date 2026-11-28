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

#include "xla/backends/gpu/hlo/transforms/dot_bf16_demotion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "xla/backends/gpu/hlo/transforms/dot_operand_converter.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;

class DotBf16DemotionTest : public HloPjRtGpuTestBase {};

TEST_F(DotBf16DemotionTest, DemotesUpcastS8AndBf16DotToBf16) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[64,8,128]{2,1,0} parameter(0)
    p1 = bf16[32,8,128]{2,1,0} parameter(1)
    c0 = f32[64,8,128]{2,1,0} convert(p0), metadata={op_name="upcast_s8"}
    rp = bf16[32,8,128]{2,1,0} reduce-precision(p1), exponent_bits=8,
                                                     mantissa_bits=7
    c1 = f32[32,8,128]{2,1,0} convert(rp)
    ROOT dot = f32[8,64,32]{2,1,0} dot(c0, c1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(AllOf(op::Convert(op::Parameter(0)),
                                  op::Shape("bf16[64,8,128]{2,1,0}")),
                            op::ReducePrecision(op::Parameter(1))),
                    op::Shape("f32[8,64,32]{2,1,0}")));
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->metadata()
                .op_name(),
            "upcast_s8");
}

TEST_F(DotBf16DemotionTest, DemotesUpcastBf16AndSmallIntDotsToBf16) {
  for (PrimitiveType int_type : {S4, U4, S8, U8}) {
    for (PrimitiveType out_type : {F32, BF16}) {
      SCOPED_TRACE(absl::StrCat(
          "int=", primitive_util::LowercasePrimitiveTypeName(int_type),
          ", out=", primitive_util::LowercasePrimitiveTypeName(out_type)));
      absl::string_view module_tmpl = R"(
      HloModule module

      ENTRY main {
        p0 = bf16[2,3]{1,0} parameter(0)
        p1 = $0[3,2]{1,0} parameter(1)
        c0 = f32[2,3]{1,0} convert(p0)
        c1 = f32[3,2]{1,0} convert(p1)
        ROOT dot = $1[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                             rhs_contracting_dims={0}
      })";
      std::string out_type_str(
          primitive_util::LowercasePrimitiveTypeName(out_type));
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<HloModule> module,
          ParseAndReturnVerifiedModule(absl::Substitute(
              module_tmpl, primitive_util::LowercasePrimitiveTypeName(int_type),
              out_type_str)));
      ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
      EXPECT_TRUE(changed);
      EXPECT_THAT(
          module->entry_computation()->root_instruction(),
          AllOf(op::Dot(op::Parameter(0), AllOf(op::Convert(op::Parameter(1)),
                                                op::Shape("bf16[3,2]{1,0}"))),
                op::Shape(absl::Substitute("$0[2,2]{1,0}", out_type_str))));
    }
  }
}

TEST_F(DotBf16DemotionTest,
       DemotesF32ReducedToBf16PrecisionAndLooksThroughDataMovement) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[3,2]{1,0} parameter(0)
    p1 = s8[2,3]{1,0} parameter(1)
    rp = f32[3,2]{1,0} reduce-precision(p0), exponent_bits=8, mantissa_bits=7
    t0 = f32[2,3]{1,0} transpose(rp), dimensions={1,0}
    c1 = f32[2,3]{1,0} convert(p1)
    t1 = f32[3,2]{1,0} transpose(c1), dimensions={1,0}
    ROOT dot = f32[2,2]{1,0} dot(t0, t1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0},
                                          operand_precision={HIGH, DEFAULT}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Dot(AllOf(op::Transpose(AllOf(op::Convert(op::ReducePrecision()),
                                        op::Shape("bf16[3,2]{1,0}"))),
                    op::Shape("bf16[2,3]{1,0}")),
              AllOf(op::Transpose(AllOf(op::Convert(op::Parameter(1)),
                                        op::Shape("bf16[2,3]{1,0}"))),
                    op::Shape("bf16[3,2]{1,0}"))));
}

TEST_F(DotBf16DemotionTest, DoesNotDemoteDotWithNonBf16ExactOperand) {
  for (absl::string_view rhs :
       {"p1 = f32[3,2]{1,0} parameter(1)\n c1 = f32[3,2]{1,0} negate(p1)",
        "p1 = s16[3,2]{1,0} parameter(1)\n c1 = f32[3,2]{1,0} convert(p1)",
        "p1 = f16[3,2]{1,0} parameter(1)\n c1 = f32[3,2]{1,0} convert(p1)",
        "p1 = f32[3,2]{1,0} parameter(1)\n"
        " c1 = f32[3,2]{1,0} reduce-precision(p1), exponent_bits=8, "
        "mantissa_bits=8",
        "p1 = f32[3,2]{1,0} parameter(1)\n"
        " c1 = f32[3,2]{1,0} reduce-precision(p1), exponent_bits=8, "
        "mantissa_bits=10",
        "p1 = s8[3,2,4]{2,1,0} parameter(1)\n"
        " c1 = f32[3,2]{1,0} bitcast(p1)"}) {
    SCOPED_TRACE(rhs);
    absl::string_view module_tmpl = R"(
    HloModule module

    ENTRY main {
      p0 = s8[2,3]{1,0} parameter(0)
      c0 = f32[2,3]{1,0} convert(p0)
      $0
      ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
    })";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(absl::Substitute(module_tmpl, rhs)));
    ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(DotBf16DemotionTest, DoesNotDemoteDotWithoutTf32Precision) {
  for (absl::string_view precision :
       {"operand_precision={HIGHEST, DEFAULT}",
        "operand_precision={DEFAULT, HIGHEST}", "algorithm=dot_f32_f32_f32",
        "algorithm=dot_tf32_tf32_f32"}) {
    SCOPED_TRACE(precision);
    absl::string_view module_tmpl = R"(
    HloModule module

    ENTRY main {
      p0 = s8[2,3]{1,0} parameter(0)
      p1 = bf16[3,2]{1,0} parameter(1)
      c0 = f32[2,3]{1,0} convert(p0)
      c1 = f32[3,2]{1,0} convert(p1)
      ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}, $0
    })";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(absl::Substitute(module_tmpl, precision)));
    ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(DotBf16DemotionTest,
       DoesNotDemoteWhenTensorFloat32DisabledOrBf16DotUnsupported) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = bf16[3,2]{1,0} parameter(1)
    c0 = f32[2,3]{1,0} convert(p0)
    c1 = f32[3,2]{1,0} convert(p1)
    ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(module_string));
    ASSERT_OK_AND_ASSIGN(
        bool changed,
        DotBf16Demotion(se::CudaComputeCapability::Volta()).Run(module.get()));
    EXPECT_FALSE(changed);
  }
  bool orig_tf32 = tsl::tensor_float_32_execution_enabled();
  tsl::enable_tensor_float_32_execution(false);
  absl::Cleanup restore_tf32 = [orig_tf32] {
    tsl::enable_tensor_float_32_execution(orig_tf32);
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotBf16DemotionTest,
       DoesNotDuplicateSharedMaterializedF32OperandWithNonDemotableDot) {
  // `c_mask` is shared between a demotable dot (`dot_ref`) and a non-demotable
  // dot (`dot_alt`, whose LHS is an arbitrary F32 parameter). Demoting only
  // `dot_ref` would require keeping both F32 and BF16 copies of `mask`.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ref = bf16[128,16]{1,0} parameter(0)
    alt = f32[128,16]{1,0} parameter(1)
    mask = pred[128,8,4]{2,1,0} parameter(2)
    c_ref = f32[128,16]{1,0} convert(ref)
    c_mask = f32[128,8,4]{2,1,0} convert(mask)
    dot_ref = f32[16,8,4]{2,1,0} dot(c_ref, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot_alt = f32[16,8,4]{2,1,0} dot(alt, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT out = (f32[16,8,4]{2,1,0}, f32[16,8,4]{2,1,0}) tuple(dot_ref, dot_alt)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotBf16DemotionTest,
       DoesNotDuplicateSharedMaterializedF32OperandWithF32Consumer) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[4,8]{1,0} parameter(0)
    p1 = bf16[8,4]{1,0} parameter(1)
    bias = f32[4,8]{1,0} parameter(2)
    c0 = f32[4,8]{1,0} convert(p0)
    c1 = f32[8,4]{1,0} convert(p1)
    add = f32[4,8]{1,0} add(c0, bias)
    dot = f32[4,4]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                     rhs_contracting_dims={0}
    ROOT out = (f32[4,8]{1,0}, f32[4,4]{1,0}) tuple(add, dot)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotBf16DemotionTest, BypassesSharedFreeBf16SourceWhenAlsoUsedByF32Op) {
  // `c1 = convert<f32>(p1: bf16)` is shared with `add`, so `c1` must stay F32.
  // Because `p1` is already materialized in BF16, `dot` reads `p1` directly
  // with zero extra allocation while `c0` narrows in-place to BF16.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[4,8]{1,0} parameter(0)
    p1 = bf16[8,4]{1,0} parameter(1)
    bias = f32[8,4]{1,0} parameter(2)
    c0 = f32[4,8]{1,0} convert(p0)
    c1 = f32[8,4]{1,0} convert(p1)
    add = f32[8,4]{1,0} add(c1, bias)
    dot = f32[4,4]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                     rhs_contracting_dims={0}
    ROOT out = (f32[8,4]{1,0}, f32[4,4]{1,0}) tuple(add, dot)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Add(AllOf(op::Convert(op::Parameter(1)),
                                      op::Shape("f32[8,4]{1,0}")),
                                op::Parameter(2)),
                        op::Dot(AllOf(op::Convert(op::Parameter(0)),
                                      op::Shape("bf16[4,8]{1,0}")),
                                op::Parameter(1))));
}

TEST_F(DotBf16DemotionTest,
       DemotesTwoDotsSharingMaterializedOperandInPlaceWithoutDuplication) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ref = bf16[128,16]{1,0} parameter(0)
    alt = bf16[128,16]{1,0} parameter(1)
    mask = pred[128,8,4]{2,1,0} parameter(2)
    c_ref = f32[128,16]{1,0} convert(ref)
    c_alt = f32[128,16]{1,0} convert(alt)
    c_mask = f32[128,8,4]{2,1,0} convert(mask)
    dot_ref = f32[16,8,4]{2,1,0} dot(c_ref, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot_alt = f32[16,8,4]{2,1,0} dot(c_alt, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT out = (f32[16,8,4]{2,1,0}, f32[16,8,4]{2,1,0}) tuple(dot_ref, dot_alt)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_TRUE(changed);
  const std::string hlo = module->ToString();
  EXPECT_THAT(hlo, HasSubstr("bf16[128,8,4]{2,1,0} convert(%mask)"));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[128,8,4]")));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[128,16]")));
}

TEST_F(DotBf16DemotionTest, PropagatesF32ConstraintsTransitivelyAcrossDots) {
  // dot3 forces `z` to stay F32 -> rejects dot2 -> forces `y` to stay F32 ->
  // rejects dot1. None of the three dots should demote.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p_x = s8[4,4]{1,0} parameter(0)
    p_y = s8[4,4]{1,0} parameter(1)
    p_z = s8[4,4]{1,0} parameter(2)
    p_f32 = f32[4,4]{1,0} parameter(3)
    x = f32[4,4]{1,0} convert(p_x)
    y = f32[4,4]{1,0} convert(p_y)
    z = f32[4,4]{1,0} convert(p_z)
    d1 = f32[4,4]{1,0} dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    d2 = f32[4,4]{1,0} dot(y, z), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    d3 = f32[4,4]{1,0} dot(z, p_f32), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT out = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(d1, d2, d3)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotBf16DemotionTest,
       DemotesHelixConditionalWithAlignAltAndSharedMaskAndScalarZero) {
  // Mirrors Helix PASVariantScorer: `alt` flows through a conditional that
  // slices and zero-pads it in one branch, while `mask` is shared between
  // `dot_ref` and `dot_alt`, and scalar `zero` is also used by an F32 add.
  absl::string_view module_string = R"(
  HloModule module

  align_branch {
    arg = (bf16[128,16]{1,0}, f32[]) parameter(0)
    alt_in = bf16[128,16]{1,0} get-tuple-element(arg), index=0
    zero = f32[] get-tuple-element(arg), index=1
    alt_f32 = f32[128,16]{1,0} convert(alt_in)
    sliced = f32[120,16]{1,0} slice(alt_f32), slice={[0:120], [0:16]}
    padded = f32[128,16]{1,0} pad(sliced, zero), padding=0_8_0x0_0_0
    ROOT res = (f32[128,16]{1,0}) tuple(padded)
  }

  identity_branch {
    arg = bf16[128,16]{1,0} parameter(0)
    alt_f32 = f32[128,16]{1,0} convert(arg)
    ROOT res = (f32[128,16]{1,0}) tuple(alt_f32)
  }

  ENTRY main {
    is_indel = pred[] parameter(0)
    ref = bf16[128,16]{1,0} parameter(1)
    alt = bf16[128,16]{1,0} parameter(2)
    mask = pred[128,8,4]{2,1,0} parameter(3)
    other = f32[] parameter(4)
    zero = f32[] constant(0)
    other_plus_zero = f32[] add(other, zero)
    align_arg = (bf16[128,16]{1,0}, f32[]) tuple(alt, zero)
    cond = (f32[128,16]{1,0}) conditional(is_indel, align_arg, alt),
        true_computation=align_branch, false_computation=identity_branch
    alt_aligned = f32[128,16]{1,0} get-tuple-element(cond), index=0
    c_ref = f32[128,16]{1,0} convert(ref)
    c_mask = f32[128,8,4]{2,1,0} convert(mask)
    dot_ref = f32[16,8,4]{2,1,0} dot(c_ref, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot_alt = f32[16,8,4]{2,1,0} dot(alt_aligned, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT out = (f32[16,8,4]{2,1,0}, f32[16,8,4]{2,1,0}, f32[])
        tuple(dot_ref, dot_alt, other_plus_zero)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_TRUE(changed);
  const std::string hlo = module->ToString();
  EXPECT_THAT(hlo, HasSubstr("bf16[128,8,4]{2,1,0} convert(%mask)"));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[128,8,4]")));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[128,16]")));
  EXPECT_THAT(hlo, HasSubstr("(bf16[128,16]{1,0}) conditional("));
}

TEST_F(DotBf16DemotionTest,
       DoesNotDemoteConditionalWhenOneBranchIsNotBf16Exact) {
  absl::string_view module_string = R"(
  HloModule module

  exact_branch {
    arg = bf16[128,16]{1,0} parameter(0)
    ROOT c = f32[128,16]{1,0} convert(arg)
  }

  inexact_branch {
    arg = f32[128,16]{1,0} parameter(0)
    ROOT neg = f32[128,16]{1,0} negate(arg)
  }

  ENTRY main {
    pred_cond = pred[] parameter(0)
    ref = bf16[128,16]{1,0} parameter(1)
    alt_bf16 = bf16[128,16]{1,0} parameter(2)
    alt_f32 = f32[128,16]{1,0} parameter(3)
    mask = pred[128,8,4]{2,1,0} parameter(4)
    cond = f32[128,16]{1,0} conditional(pred_cond, alt_bf16, alt_f32),
        true_computation=exact_branch, false_computation=inexact_branch
    c_ref = f32[128,16]{1,0} convert(ref)
    c_mask = f32[128,8,4]{2,1,0} convert(mask)
    dot_ref = f32[16,8,4]{2,1,0} dot(c_ref, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot_alt = f32[16,8,4]{2,1,0} dot(cond, c_mask),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT out = (f32[16,8,4]{2,1,0}, f32[16,8,4]{2,1,0}) tuple(dot_ref, dot_alt)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotBf16DemotionTest, DemotesWhileLoopCarryingBf16ExactValue) {
  absl::string_view module_string = R"(
  HloModule module

  while_cond {
    state = (s32[], f32[8,4]{1,0}, bf16[2,4]{1,0}) parameter(0)
    idx = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(4)
    ROOT cmp = pred[] compare(idx, limit), direction=LT
  }

  while_body {
    state = (s32[], f32[8,4]{1,0}, bf16[2,4]{1,0}) parameter(0)
    idx = s32[] get-tuple-element(state), index=0
    buf = f32[8,4]{1,0} get-tuple-element(state), index=1
    upd_bf16 = bf16[2,4]{1,0} get-tuple-element(state), index=2
    upd_f32 = f32[2,4]{1,0} convert(upd_bf16)
    zero = s32[] constant(0)
    new_buf = f32[8,4]{1,0} dynamic-update-slice(buf, upd_f32, idx, zero)
    one = s32[] constant(1)
    next_idx = s32[] add(idx, one)
    ROOT next = (s32[], f32[8,4]{1,0}, bf16[2,4]{1,0})
        tuple(next_idx, new_buf, upd_bf16)
  }

  ENTRY main {
    init_s8 = s8[8,4]{1,0} parameter(0)
    upd = bf16[2,4]{1,0} parameter(1)
    rhs_bf16 = bf16[4,4]{1,0} parameter(2)
    init_f32 = f32[8,4]{1,0} convert(init_s8)
    zero = s32[] constant(0)
    init = (s32[], f32[8,4]{1,0}, bf16[2,4]{1,0}) tuple(zero, init_f32, upd)
    loop = (s32[], f32[8,4]{1,0}, bf16[2,4]{1,0}) while(init),
        condition=while_cond, body=while_body
    lhs = f32[8,4]{1,0} get-tuple-element(loop), index=1
    rhs = f32[4,4]{1,0} convert(rhs_bf16)
    ROOT dot = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_TRUE(changed);
  const std::string hlo = module->ToString();
  EXPECT_THAT(hlo, HasSubstr("bf16[8,4]{1,0} dynamic-update-slice("));
  EXPECT_THAT(hlo, HasSubstr("(s32[], bf16[8,4]{1,0}, bf16[2,4]{1,0}) while("));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[8,4]{1,0} convert")));
}

TEST_F(DotBf16DemotionTest,
       DoesNotDemoteWhileLoopWhenBackedgeProducesNonBf16ExactValue) {
  absl::string_view module_string = R"(
  HloModule module

  while_cond {
    state = (s32[], f32[8,4]{1,0}) parameter(0)
    idx = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(4)
    ROOT cmp = pred[] compare(idx, limit), direction=LT
  }

  while_body {
    state = (s32[], f32[8,4]{1,0}) parameter(0)
    idx = s32[] get-tuple-element(state), index=0
    buf = f32[8,4]{1,0} get-tuple-element(state), index=1
    new_buf = f32[8,4]{1,0} add(buf, buf)
    one = s32[] constant(1)
    next_idx = s32[] add(idx, one)
    ROOT next = (s32[], f32[8,4]{1,0}) tuple(next_idx, new_buf)
  }

  ENTRY main {
    init_s8 = s8[8,4]{1,0} parameter(0)
    rhs_bf16 = bf16[4,4]{1,0} parameter(1)
    init_f32 = f32[8,4]{1,0} convert(init_s8)
    zero = s32[] constant(0)
    init = (s32[], f32[8,4]{1,0}) tuple(zero, init_f32)
    loop = (s32[], f32[8,4]{1,0}) while(init),
        condition=while_cond, body=while_body
    lhs = f32[8,4]{1,0} get-tuple-element(loop), index=1
    rhs = f32[4,4]{1,0} convert(rhs_bf16)
    ROOT dot = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotBf16Demotion().Run(module.get()));
  EXPECT_FALSE(changed);
}

constexpr absl::string_view kUpcastS8Bf16DotTemplate = R"(
  HloModule module

  ENTRY main {
    p0 = s8[512,4,1024]{2,1,0} parameter(0)
    p1 = bf16[32,4,1024]{2,1,0} parameter(1)
    c0 = f32[512,4,1024]{2,1,0} convert(p0)
    c1 = f32[32,4,1024]{2,1,0} convert(p1)
    ROOT dot = f32[4,512,32]{2,1,0} dot(c0, c1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}$0
  })";

TEST_F(DotBf16DemotionTest, CompilerDemotesUpcastS8AndBf16DotToBf16) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      GetOptimizedModule(absl::Substitute(kUpcastS8Bf16DotTemplate, "")));
  const std::string optimized = module->ToString();
  EXPECT_THAT(optimized, HasSubstr("bf16[512,4,"));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[512,4,")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[32,4,")));
}

TEST_F(DotBf16DemotionTest, CompilerDemotesUpcastThroughDataMovementToBf16) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[4,1024,512]{2,1,0} parameter(0)
    p1 = bf16[32,1024]{1,0} parameter(1)
    c0 = f32[4,1024,512]{2,1,0} convert(p0)
    t0 = f32[512,4,1024]{2,1,0} transpose(c0), dimensions={2,0,1}
    c1 = f32[32,1024]{1,0} convert(p1)
    b1 = f32[32,4,1024]{2,1,0} broadcast(c1), dimensions={0,2}
    ROOT dot = f32[4,512,32]{2,1,0} dot(t0, b1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(module_string));
  const std::string optimized = module->ToString();
  EXPECT_THAT(optimized, Not(HasSubstr("f32[4,1024,512]")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[512,4,")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[32,1024]")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[32,4,")));
}

TEST_F(DotBf16DemotionTest, DemotedDotMatchesIeeeF32Dot) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      absl::Substitute(kUpcastS8Bf16DotTemplate, ""),
      absl::Substitute(kUpcastS8Bf16DotTemplate,
                       ", operand_precision={HIGHEST, HIGHEST}"),
      ErrorSpec{1e-3, 1e-5}));
}

TEST_F(DotBf16DemotionTest,
       CompilerDemotesHelixConditionalWithDuplicateMaskConvertsNoTriton) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  // Two un-CSE'd converts of the same pred mask plus a conditional-aligned
  // alt branch, compiled with Triton GEMM disabled (cuBLAS GemmRewriter).
  // HloCSE deduplicates the mask converts before DotBf16Demotion runs, so both
  // dots demote to BF16 and no F32 copy of the mask survives.
  absl::string_view module_string = R"(
  HloModule module

  align_branch {
    arg = bf16[1024,32]{1,0} parameter(0)
    alt_f32 = f32[1024,32]{1,0} convert(arg)
    sliced = f32[1000,32]{1,0} slice(alt_f32), slice={[0:1000], [0:32]}
    zero = f32[] constant(0)
    ROOT padded = f32[1024,32]{1,0} pad(sliced, zero), padding=0_24_0x0_0_0
  }

  identity_branch {
    arg = bf16[1024,32]{1,0} parameter(0)
    ROOT alt_f32 = f32[1024,32]{1,0} convert(arg)
  }

  ENTRY main {
    is_indel = pred[] parameter(0)
    ref = bf16[1024,32]{1,0} parameter(1)
    alt = bf16[1024,32]{1,0} parameter(2)
    mask = pred[1024,8,16]{2,1,0} parameter(3)
    alt_aligned = f32[1024,32]{1,0} conditional(is_indel, alt, alt),
        true_computation=align_branch, false_computation=identity_branch
    c_ref = f32[1024,32]{1,0} convert(ref)
    c_mask_ref = f32[1024,8,16]{2,1,0} convert(mask)
    c_mask_alt = f32[1024,8,16]{2,1,0} convert(mask)
    dot_ref = f32[32,8,16]{2,1,0} dot(c_ref, c_mask_ref),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot_alt = f32[32,8,16]{2,1,0} dot(alt_aligned, c_mask_alt),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT out = (f32[32,8,16]{2,1,0}, f32[32,8,16]{2,1,0}) tuple(dot_ref, dot_alt)
  })";
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(module_string, config));
  const std::string hlo = optimized->ToString();
  EXPECT_THAT(hlo, HasSubstr("__cublas$"));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[1024,8,16]")));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[1024,128]")));
  EXPECT_THAT(hlo, Not(HasSubstr("f32[1024,32]")));
  EXPECT_THAT(hlo, HasSubstr("bf16[1024,"));
}

TEST_F(DotBf16DemotionTest,
       CompilerAvoidsMaskDuplicationWhenAltIsNonDemotableNoTriton) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  // Same un-CSE'd pred mask pattern, except `alt` is an arbitrary F32 input.
  // After HloCSE merges `c_mask_ref` and `c_mask_alt`, DotBf16Demotion keeps
  // both dots in F32 so no duplicate BF16 copy of `mask` is allocated.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ref = bf16[1024,32]{1,0} parameter(0)
    alt = f32[1024,32]{1,0} parameter(1)
    mask = pred[1024,8,16]{2,1,0} parameter(2)
    c_ref = f32[1024,32]{1,0} convert(ref)
    c_mask_ref = f32[1024,8,16]{2,1,0} convert(mask)
    c_mask_alt = f32[1024,8,16]{2,1,0} convert(mask)
    dot_ref = f32[32,8,16]{2,1,0} dot(c_ref, c_mask_ref),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot_alt = f32[32,8,16]{2,1,0} dot(alt, c_mask_alt),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT out = (f32[32,8,16]{2,1,0}, f32[32,8,16]{2,1,0}) tuple(dot_ref, dot_alt)
  })";
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(module_string, config));
  const std::string hlo = optimized->ToString();
  EXPECT_THAT(hlo, HasSubstr("__cublas$"));
  EXPECT_THAT(hlo, Not(HasSubstr("bf16[1024,8,16]")));
  EXPECT_THAT(hlo, Not(HasSubstr("bf16[1024,128]")));
  EXPECT_THAT(hlo, HasSubstr("f32[1024,"));
}

}  // namespace
}  // namespace xla::gpu
