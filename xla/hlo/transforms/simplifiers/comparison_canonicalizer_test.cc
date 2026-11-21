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

#include "xla/hlo/transforms/simplifiers/comparison_canonicalizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::TestParamInfo;
using ::testing::Values;
using ::testing::WithParamInterface;
using ::tsl::proto_testing::EqualsProto;

using ComparisonCanonicalizerTest = HloHardwareIndependentTestBase;

constexpr absl::string_view kTypedSortComparatorHloTemplate = R"hlo(
  HloModule m

  comparator {
    p0 = $0[] parameter(0)
    p1 = $0[] parameter(1)
    zero = $0[] constant(0)
    nan_lhs = $0[] constant(nan)
    nan_rhs = $0[] constant(nan)
    p0_is_zero = pred[] compare(p0, zero), direction=EQ
    p0_no_neg_zero = $0[] select(p0_is_zero, zero, p0)
    p0_is_nan = pred[] compare(p0, p0), direction=NE
    p0_canon = $0[] select(p0_is_nan, nan_lhs, p0_no_neg_zero)
    p1_is_zero = pred[] compare(p1, zero), direction=EQ
    p1_no_neg_zero = $0[] select(p1_is_zero, zero, p1)
    p1_is_nan = pred[] compare(p1, p1), direction=NE
    p1_canon = $0[] select(p1_is_nan, nan_rhs, p1_no_neg_zero)
    ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=TOTAL
  }

  ENTRY main {
    keys = $0[8] parameter(0)
    ROOT sorted = $0[8] sort(keys), dimensions={0}, to_apply=comparator
  }
)hlo";

class ComparisonCanonicalizerSignedZeroTypeTest
    : public HloHardwareIndependentTestBase,
      public WithParamInterface<PrimitiveType> {};

TEST_P(ComparisonCanonicalizerSignedZeroTypeTest,
       RewritesScalarSortComparator) {
  const std::string hlo =
      absl::Substitute(kTypedSortComparatorHloTemplate,
                       primitive_util::LowercasePrimitiveTypeName(GetParam()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                             /*expect_change=*/true));

  const HloComputation* comp =
      module->entry_computation()->root_instruction()->to_apply();
  const HloInstruction* root = comp->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCompare);
  EXPECT_EQ(root->comparison_direction(), ComparisonDirection::kLt);
  EXPECT_EQ(root->comparison_order(), ComparisonOrder::kWeak);
  EXPECT_EQ(root->operand(0), comp->parameter_instruction(0));
  EXPECT_EQ(root->operand(1), comp->parameter_instruction(1));
  EXPECT_EQ(comp->instruction_count(), 3);
}

INSTANTIATE_TEST_SUITE_P(SignedZeroFloatTypes,
                         ComparisonCanonicalizerSignedZeroTypeTest,
                         Values(F32, BF16, F16, F64, F8E4M3FN, F8E5M2, F8E3M4,
                                F8E4M3),
                         [](const TestParamInfo<PrimitiveType>& info) {
                           return PrimitiveType_Name(info.param);
                         });

class ComparisonCanonicalizerUnsignedZeroTypeTest
    : public HloHardwareIndependentTestBase,
      public WithParamInterface<PrimitiveType> {};

TEST_P(ComparisonCanonicalizerUnsignedZeroTypeTest,
       DoesNotRewriteSortComparator) {
  const std::string hlo =
      absl::Substitute(kTypedSortComparatorHloTemplate,
                       primitive_util::LowercasePrimitiveTypeName(GetParam()));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

INSTANTIATE_TEST_SUITE_P(UnsignedZeroFloatTypes,
                         ComparisonCanonicalizerUnsignedZeroTypeTest,
                         Values(F8E4M3FNUZ, F8E5M2FNUZ, F8E4M3B11FNUZ),
                         [](const TestParamInfo<PrimitiveType>& info) {
                           return PrimitiveType_Name(info.param);
                         });

class ComparisonCanonicalizerDirectionTest
    : public HloHardwareIndependentTestBase,
      public WithParamInterface<ComparisonDirection> {};

TEST_P(ComparisonCanonicalizerDirectionTest, RewritesSortComparator) {
  constexpr absl::string_view kHloTemplate = R"hlo(
    HloModule m

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)
      p0_is_zero = pred[] compare(p0, zero), direction=EQ
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE
      p0_canon = f32[] select(p0_is_nan, nan_c, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE
      p1_canon = f32[] select(p1_is_nan, nan_c, p1_no_neg_zero)
      ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=$0, order=TOTAL
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";
  const ComparisonDirection direction = GetParam();
  const std::string hlo =
      absl::Substitute(kHloTemplate, ComparisonDirectionToString(direction));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                             /*expect_change=*/true));

  const HloComputation* comp =
      module->entry_computation()->root_instruction()->to_apply();
  const HloInstruction* root = comp->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCompare);
  EXPECT_EQ(root->comparison_direction(), direction);
  EXPECT_EQ(root->comparison_order(), ComparisonOrder::kWeak);
  EXPECT_EQ(comp->instruction_count(), 3);
}

INSTANTIATE_TEST_SUITE_P(
    AllDirections, ComparisonCanonicalizerDirectionTest,
    Values(ComparisonDirection::kLt, ComparisonDirection::kLe,
           ComparisonDirection::kGt, ComparisonDirection::kGe,
           ComparisonDirection::kEq, ComparisonDirection::kNe),
    [](const TestParamInfo<ComparisonDirection>& info) {
      return ComparisonDirectionToString(info.param);
    });

class ComparisonCanonicalizerNonTotalOuterOrderTest
    : public HloHardwareIndependentTestBase,
      public WithParamInterface<ComparisonOrder> {};

TEST_P(ComparisonCanonicalizerNonTotalOuterOrderTest,
       DoesNotRewriteSortComparator) {
  constexpr absl::string_view kHloTemplate = R"hlo(
    HloModule m

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)
      p0_is_zero = pred[] compare(p0, zero), direction=EQ
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE
      p0_canon = f32[] select(p0_is_nan, nan_c, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE
      p1_canon = f32[] select(p1_is_nan, nan_c, p1_no_neg_zero)
      ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=$0
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";
  const std::string hlo =
      absl::Substitute(kHloTemplate, ComparisonOrderToShortString(GetParam()));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

INSTANTIATE_TEST_SUITE_P(
    NonTotalOuterOrders, ComparisonCanonicalizerNonTotalOuterOrderTest,
    Values(ComparisonOrder::kPartial, ComparisonOrder::kWeak),
    [](const TestParamInfo<ComparisonOrder>& info) {
      return std::string(ComparisonOrderToShortString(info.param));
    });

class ComparisonCanonicalizerNonPartialInnerOrderTest
    : public HloHardwareIndependentTestBase,
      public WithParamInterface<ComparisonOrder> {};

TEST_P(ComparisonCanonicalizerNonPartialInnerOrderTest,
       DoesNotRewriteNonPartialInnerNanCompare) {
  constexpr absl::string_view kHloTemplate = R"hlo(
    HloModule m

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)
      p0_is_zero = pred[] compare(p0, zero), direction=EQ
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE, order=$0
      p0_canon = f32[] select(p0_is_nan, nan_c, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE, order=$0
      p1_canon = f32[] select(p1_is_nan, nan_c, p1_no_neg_zero)
      ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=TOTAL
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";
  const std::string hlo =
      absl::Substitute(kHloTemplate, ComparisonOrderToShortString(GetParam()));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_P(ComparisonCanonicalizerNonPartialInnerOrderTest,
       DoesNotRewriteNonPartialInnerZeroCompare) {
  constexpr absl::string_view kHloTemplate = R"hlo(
    HloModule m

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)
      p0_is_zero = pred[] compare(p0, zero), direction=EQ, order=$0
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE
      p0_canon = f32[] select(p0_is_nan, nan_c, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ, order=$0
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE
      p1_canon = f32[] select(p1_is_nan, nan_c, p1_no_neg_zero)
      ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=TOTAL
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";
  const std::string hlo =
      absl::Substitute(kHloTemplate, ComparisonOrderToShortString(GetParam()));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

INSTANTIATE_TEST_SUITE_P(
    NonPartialInnerOrders, ComparisonCanonicalizerNonPartialInnerOrderTest,
    Values(ComparisonOrder::kTotal, ComparisonOrder::kWeak),
    [](const TestParamInfo<ComparisonOrder>& info) {
      return std::string(ComparisonOrderToShortString(info.param));
    });

TEST_F(ComparisonCanonicalizerTest, RewritesMatchingCustomPositiveNanPayloads) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      zero = f32[] constant(0)
      nan_lhs = f32[] constant(nan(0x123))
      nan_rhs = f32[] constant(nan(0x123))
      p0_is_zero = pred[] compare(p0, zero), direction=EQ
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE
      p0_canon = f32[] select(p0_is_nan, nan_lhs, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE
      p1_canon = f32[] select(p1_is_nan, nan_rhs, p1_no_neg_zero)
      ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=TOTAL
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       RunAndCheckHloRewrite(kHlo, ComparisonCanonicalizer(),
                                             /*expect_change=*/true));

  const HloComputation* comp =
      module->entry_computation()->root_instruction()->to_apply();
  const HloInstruction* root = comp->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCompare);
  EXPECT_EQ(root->comparison_order(), ComparisonOrder::kWeak);
  EXPECT_EQ(comp->instruction_count(), 3);
}

TEST_F(ComparisonCanonicalizerTest,
       RewritesMultiKeyLexicographicSortComparatorWithSharedOperands) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    comparator {
      k0_lhs = f32[] parameter(0)
      k0_rhs = f32[] parameter(1)
      k1_lhs = f32[] parameter(2)
      k1_rhs = f32[] parameter(3)
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)

      k0_lhs_z = pred[] compare(k0_lhs, zero), direction=EQ
      k0_lhs_nz = f32[] select(k0_lhs_z, zero, k0_lhs)
      k0_lhs_n = pred[] compare(k0_lhs, k0_lhs), direction=NE
      k0_lhs_c = f32[] select(k0_lhs_n, nan_c, k0_lhs_nz)

      k0_rhs_z = pred[] compare(k0_rhs, zero), direction=EQ
      k0_rhs_nz = f32[] select(k0_rhs_z, zero, k0_rhs)
      k0_rhs_n = pred[] compare(k0_rhs, k0_rhs), direction=NE
      k0_rhs_c = f32[] select(k0_rhs_n, nan_c, k0_rhs_nz)

      k1_lhs_z = pred[] compare(k1_lhs, zero), direction=EQ
      k1_lhs_nz = f32[] select(k1_lhs_z, zero, k1_lhs)
      k1_lhs_n = pred[] compare(k1_lhs, k1_lhs), direction=NE
      k1_lhs_c = f32[] select(k1_lhs_n, nan_c, k1_lhs_nz)

      k1_rhs_z = pred[] compare(k1_rhs, zero), direction=EQ
      k1_rhs_nz = f32[] select(k1_rhs_z, zero, k1_rhs)
      k1_rhs_n = pred[] compare(k1_rhs, k1_rhs), direction=NE
      k1_rhs_c = f32[] select(k1_rhs_n, nan_c, k1_rhs_nz)

      // CHECK:      comparator
      // CHECK-DAG:  %[[K0_LT:.*]] = pred[] compare(%k0_lhs, %k0_rhs), direction=LT, order=WEAK
      // CHECK-DAG:  %[[K0_EQ:.*]] = pred[] compare(%k0_lhs, %k0_rhs), direction=EQ, order=WEAK
      // CHECK-DAG:  %[[K1_LT:.*]] = pred[] compare(%k1_lhs, %k1_rhs), direction=LT, order=WEAK
      // CHECK-DAG:  %[[TIE:.*]] = pred[] and(%[[K0_EQ]], %[[K1_LT]])
      // CHECK:      ROOT {{.*}} = pred[] or(%[[K0_LT]], %[[TIE]])
      k0_lt = pred[] compare(k0_lhs_c, k0_rhs_c), direction=LT, order=TOTAL
      k0_eq = pred[] compare(k0_lhs_c, k0_rhs_c), direction=EQ, order=TOTAL
      k1_lt = pred[] compare(k1_lhs_c, k1_rhs_c), direction=LT, order=TOTAL
      tie = pred[] and(k0_eq, k1_lt)
      ROOT out = pred[] or(k0_lt, tie)
    }

    ENTRY main {
      k0 = f32[8] parameter(0)
      k1 = f32[8] parameter(1)
      ROOT sorted = (f32[8], f32[8]) sort(k0, k1), dimensions={0}, to_apply=comparator
    }
  )hlo";

  RunAndFilecheckHloRewrite(kHlo, ComparisonCanonicalizer(), [](HloModule* m) {
    EXPECT_EQ(m->entry_computation()
                  ->root_instruction()
                  ->to_apply()
                  ->instruction_count(),
              9);
  });
}

TEST_F(ComparisonCanonicalizerTest,
       PreservesOriginalValueShardingMetadataAndControlDependencies) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      tok = token[] after-all()
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)
      p0_is_zero = pred[] compare(p0, zero), direction=EQ
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE
      p0_canon = f32[] select(p0_is_nan, nan_c, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE
      p1_canon = f32[] select(p1_is_nan, nan_c, p1_no_neg_zero)
      cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=TOTAL,
          control-predecessors={tok},
          sharding={replicated},
          frontend_attributes={test_attr="val"},
          origin={{"cmp_orig"}},
          metadata={op_name="my_cmp"}
      tok_out = token[] after-all(), control-predecessors={cmp}
      ROOT res = pred[] copy(cmp), control-predecessors={tok_out}
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       RunAndCheckHloRewrite(kHlo, ComparisonCanonicalizer(),
                                             /*expect_change=*/true));

  const HloInstruction* res = module->entry_computation()
                                  ->root_instruction()
                                  ->to_apply()
                                  ->root_instruction();
  const HloInstruction* cmp = res->operand(0);
  ASSERT_EQ(res->control_predecessors().size(), 1);
  const HloInstruction* tok_out = res->control_predecessors()[0];
  ASSERT_EQ(cmp->opcode(), HloOpcode::kCompare);
  EXPECT_EQ(cmp->comparison_order(), ComparisonOrder::kWeak);
  ASSERT_NE(cmp->original_value(), nullptr);
  EXPECT_EQ(cmp->original_value()->ToString(), R"({"cmp_orig"})");
  ASSERT_TRUE(cmp->has_sharding());
  EXPECT_EQ(cmp->sharding().ToString(), "{replicated}");
  EXPECT_THAT(cmp->frontend_attributes(),
              EqualsProto(R"pb(map { key: "test_attr" value: "val" })pb"));
  EXPECT_THAT(cmp->metadata(), EqualsProto(R"pb(op_name: "my_cmp")pb"));
  ASSERT_EQ(cmp->control_predecessors().size(), 1);
  EXPECT_EQ(cmp->control_predecessors()[0]->opcode(), HloOpcode::kAfterAll);
  ASSERT_EQ(tok_out->control_predecessors().size(), 1);
  EXPECT_EQ(tok_out->control_predecessors()[0], cmp);
}

absl::StatusOr<std::string> WithF32ComparatorReplacement(
    absl::string_view target, absl::string_view replacement) {
  const std::string base = absl::Substitute(kTypedSortComparatorHloTemplate,
                                            /*$0=*/"f32");
  std::string modified = absl::StrReplaceAll(base, {{target, replacement}});
  if (modified == base) {
    return absl::InvalidArgumentError(
        absl::StrCat("Replacement had no effect: ", target));
  }
  return modified;
}

TEST_F(ComparisonCanonicalizerTest, DoesNotRewriteOutsideSortComparator) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      zero = f32[] constant(0)
      nan_c = f32[] constant(nan)
      p0_is_zero = pred[] compare(p0, zero), direction=EQ
      p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)
      p0_is_nan = pred[] compare(p0, p0), direction=NE
      p0_canon = f32[] select(p0_is_nan, nan_c, p0_no_neg_zero)
      p1_is_zero = pred[] compare(p1, zero), direction=EQ
      p1_no_neg_zero = f32[] select(p1_is_zero, zero, p1)
      p1_is_nan = pred[] compare(p1, p1), direction=NE
      p1_canon = f32[] select(p1_is_nan, nan_c, p1_no_neg_zero)
      ROOT cmp = pred[] compare(p0_canon, p1_canon), direction=LT, order=TOTAL
    }
  )hlo";
  EXPECT_OK(RunAndCheckHloRewrite(kHlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest,
       DoesNotRewriteNonScalarCompareInSortComparator) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    or_reduce {
      a = pred[] parameter(0)
      b = pred[] parameter(1)
      ROOT out = pred[] or(a, b)
    }

    comparator {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      b0 = f32[2] broadcast(p0), dimensions={}
      b1 = f32[2] broadcast(p1), dimensions={}
      zero_s = f32[] constant(0)
      nan_s = f32[] constant(nan)
      zero = f32[2] broadcast(zero_s), dimensions={}
      nan_c = f32[2] broadcast(nan_s), dimensions={}
      b0_is_zero = pred[2] compare(b0, zero), direction=EQ
      b0_no_neg_zero = f32[2] select(b0_is_zero, zero, b0)
      b0_is_nan = pred[2] compare(b0, b0), direction=NE
      b0_canon = f32[2] select(b0_is_nan, nan_c, b0_no_neg_zero)
      b1_is_zero = pred[2] compare(b1, zero), direction=EQ
      b1_no_neg_zero = f32[2] select(b1_is_zero, zero, b1)
      b1_is_nan = pred[2] compare(b1, b1), direction=NE
      b1_canon = f32[2] select(b1_is_nan, nan_c, b1_no_neg_zero)
      cmp_vec = pred[2] compare(b0_canon, b1_canon), direction=LT, order=TOTAL
      false_c = pred[] constant(false)
      ROOT res = pred[] reduce(cmp_vec, false_c), dimensions={0}, to_apply=or_reduce
    }

    ENTRY main {
      keys = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(keys), dimensions={0}, to_apply=comparator
    }
  )hlo";
  EXPECT_OK(RunAndCheckHloRewrite(kHlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest, DoesNotRewriteNegativeNanReplacement) {
  ASSERT_OK_AND_ASSIGN(
      const std::string hlo,
      WithF32ComparatorReplacement("constant(nan)", "constant(-nan)"));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest, DoesNotRewriteMismatchedNanPayloads) {
  ASSERT_OK_AND_ASSIGN(
      const std::string hlo,
      WithF32ComparatorReplacement("nan_lhs = f32[] constant(nan)",
                                   "nan_lhs = f32[] constant(nan(0x1))"));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest, DoesNotRewriteNegativeZeroReplacement) {
  ASSERT_OK_AND_ASSIGN(
      const std::string hlo,
      WithF32ComparatorReplacement(
          "p0_no_neg_zero = f32[] select(p0_is_zero, zero, p0)",
          "neg_zero = f32[] constant(-0.0)\n"
          "    p0_no_neg_zero = f32[] select(p0_is_zero, neg_zero, p0)"));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest,
       DoesNotRewriteMismatchedNanPredicateOperand) {
  ASSERT_OK_AND_ASSIGN(
      const std::string hlo,
      WithF32ComparatorReplacement("compare(p0, p0), direction=NE",
                                   "compare(p0, p1), direction=NE"));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest,
       DoesNotRewriteMismatchedZeroPredicateOperand) {
  ASSERT_OK_AND_ASSIGN(
      const std::string hlo,
      WithF32ComparatorReplacement("compare(p0, zero), direction=EQ",
                                   "compare(p1, zero), direction=EQ"));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

TEST_F(ComparisonCanonicalizerTest, DoesNotRewriteUncanonicalizedRhsOperand) {
  ASSERT_OK_AND_ASSIGN(const std::string hlo, WithF32ComparatorReplacement(
                                                  "compare(p0_canon, p1_canon)",
                                                  "compare(p0_canon, p1)"));
  EXPECT_OK(RunAndCheckHloRewrite(hlo, ComparisonCanonicalizer(),
                                  /*expect_change=*/false));
}

}  // namespace
}  // namespace xla
