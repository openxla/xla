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

#include "xla/hlo/transforms/expanders/permutation_sort_expander.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::absl_testing::IsOkAndHolds;
using PermutationSortExpanderTest = HloHardwareIndependentTestBase;

TEST_F(PermutationSortExpanderTest, ReplacePermutationSortWithScatter) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(true));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Tuple(op::Iota(),
                      op::Scatter(
                          op::Broadcast(op::Constant()),
                          op::Concatenate(op::Iota(),
                                          op::Reshape(op::GetTupleElement(
                                              op::Sort(), /*tuple_index=*/1))),
                          op::Iota())));
}

TEST_F(PermutationSortExpanderTest, DontReplaceIfWrongComparisonDirection) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest, DontReplaceIfComparingWrongParameters) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.rhs, p.0.lhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest, DontReplacePermutationSortIfNonIntegral) {
  // Same as ReplacePermutationSortWithScatter except that the iota has F32
  // type.
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = f32[] parameter(2)
      p.1.rhs = f32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = f32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, f32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = f32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (f32[64,8732]{1,0}, f32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest, DontReplacePermutationSortWrongDimensions) {
  // Same as ReplacePermutationSortWithScatter except that the sort dimensions
  // don't match.
  const char* hlo_string = R"(
   HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={0}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest,
       ScatterUpdateComputationUsesSameElementTypeAsUpdates) {
  const char* hlo_string = R"(
    HloModule permutation_sort_int64

    lt_f32_s64 {
      k_lhs = f32[] parameter(0)
      k_rhs = f32[] parameter(1)
      v_lhs = s64[] parameter(2)
      v_rhs = s64[] parameter(3)
      ROOT lt = pred[] compare(k_lhs, k_rhs), direction=LT, order=TOTAL
    }

    lt_s64 {
      p0 = s64[] parameter(0)
      p1 = s64[] parameter(1)
      p2 = s64[] parameter(2)
      p3 = s64[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[10]{0} parameter(0)
      values = s64[10]{0} iota(), iota_dimension=0
      sort1 = (f32[10]{0}, s64[10]{0})
          sort(keys, values), dimensions={0}, to_apply=lt_f32_s64
      perm = s64[10]{0} get-tuple-element(sort1), index=1
      ROOT sort2 = (s64[10]{0}, s64[10]{0})
          sort(perm, values), dimensions={0}, to_apply=lt_s64
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PermutationSortExpander().Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* scatter = FindInstruction(module.get(), HloOpcode::kScatter);
  ASSERT_NE(scatter, nullptr);

  HloComputation* update_comp = scatter->to_apply();
  ASSERT_NE(update_comp, nullptr);

  EXPECT_EQ(update_comp->parameter_instruction(0)->shape().element_type(), S64);
  EXPECT_EQ(update_comp->parameter_instruction(1)->shape().element_type(), S64);
}

Literal MakeDeterministicF32(absl::Span<const int64_t> dims, uint32_t seed) {
  Literal lit = LiteralUtil::CreateFromDimensions(F32, dims);
  auto data = lit.data<float>();
  uint32_t s = seed;
  for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
    s = s * 1664525u + 1013904223u;
    data[i] = static_cast<float>(static_cast<int32_t>(s >> 9) - (1 << 22));
  }
  return lit;
}

void ExpectBitwiseLiteralEq(const Literal& actual, const Literal& expected) {
  ASSERT_EQ(actual.shape(), expected.shape());
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  const auto* a = static_cast<const uint8_t*>(actual.untyped_data());
  const auto* b = static_cast<const uint8_t*>(expected.untyped_data());
  for (int64_t i = 0; i < actual.size_bytes(); ++i) {
    ASSERT_EQ(a[i], b[i]) << "byte mismatch at " << i;
  }
}

TEST_F(PermutationSortExpanderTest,
       RejectsSortAxisOverflowDynamicShapesAndSideEffectComparator) {
  // Reject overflowing sort-axis iotas, dynamic shapes, and side effects.
  const absl::string_view kRejected[] = {
      R"(
        HloModule s8_overflow
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = s8[] parameter(2)
          p3 = s8[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = s8[] parameter(0)
          p1 = s8[] parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[256]{0} parameter(0)
          v = f32[256]{0} parameter(1)
          i = s8[256]{0} iota(), iota_dimension=0
          sort1 = (f32[256]{0}, s8[256]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
          p = s8[256]{0} get-tuple-element(sort1), index=1
          ROOT sort2 = (s8[256]{0}, f32[256]{0}) sort(p, v), dimensions={0}, to_apply=lt_k
        }
      )",
      R"(
        HloModule u8_overflow
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = u8[] parameter(2)
          p3 = u8[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = u8[] parameter(0)
          p1 = u8[] parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[300]{0} parameter(0)
          v = f32[300]{0} parameter(1)
          i = u8[300]{0} iota(), iota_dimension=0
          sort1 = (f32[300]{0}, u8[300]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
          p = u8[300]{0} get-tuple-element(sort1), index=1
          ROOT sort2 = (u8[300]{0}, f32[300]{0}) sort(p, v), dimensions={0}, to_apply=lt_k
        }
      )",
      R"(
        HloModule s32_overflow
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = s32[] parameter(2)
          p3 = s32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = s32[] parameter(0)
          p1 = s32[] parameter(1)
          p2 = s32[] parameter(2)
          p3 = s32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[2147483649]{0} parameter(0)
          i = s32[2147483649]{0} iota(), iota_dimension=0
          sort1 = (f32[2147483649]{0}, s32[2147483649]{0}) sort(k, i),
            dimensions={0}, to_apply=lt_f32
          p = s32[2147483649]{0} get-tuple-element(sort1), index=1
          ROOT sort2 = (s32[2147483649]{0}, s32[2147483649]{0}) sort(p, i),
            dimensions={0}, to_apply=lt_k
        }
      )",
      R"(
        HloModule dynamic_sort
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = s32[] parameter(2)
          p3 = s32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = s32[] parameter(0)
          p1 = s32[] parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[<=8]{0} parameter(0)
          v = f32[<=8]{0} parameter(1)
          i = s32[8]{0} iota(), iota_dimension=0
          s = s32[] get-dimension-size(k), dimensions={0}
          id = s32[<=8]{0} set-dimension-size(i, s), dimensions={0}
          sort1 = (f32[<=8]{0}, s32[<=8]{0}) sort(k, id), dimensions={0}, to_apply=lt_f32
          p = s32[<=8]{0} get-tuple-element(sort1), index=1
          ROOT sort2 = (s32[<=8]{0}, f32[<=8]{0}) sort(p, v), dimensions={0}, to_apply=lt_k
        }
      )",
      R"(
        HloModule side_effect_cmp
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = s32[] parameter(2)
          p3 = s32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = s32[] parameter(0)
          p1 = s32[] parameter(1)
          p2 = s32[] parameter(2)
          p3 = s32[] parameter(3)
          se = s32[] custom-call(p0), custom_call_target="side_effect",
            custom_call_has_side_effect=true
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[8]{0} parameter(0)
          i = s32[8]{0} iota(), iota_dimension=0
          sort1 = (f32[8]{0}, s32[8]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
          p = s32[8]{0} get-tuple-element(sort1), index=1
          ROOT sort2 = (s32[8]{0}, s32[8]{0}) sort(p, i), dimensions={0}, to_apply=lt_k
        }
      )",
  };
  for (absl::string_view hlo : kRejected) {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(PermutationSortExpander().Run(m.get()), IsOkAndHolds(false));
  }

  // Exact type boundaries s8[128] and u8[256] still expand with 0-ULP parity,
  // preserving -0.0f, subnormals, and NaN payloads bit-for-bit.
  for (const auto& [n, hlo] :
       std::vector<std::pair<int64_t, absl::string_view>>{{128, R"(
             HloModule s8_max
             lt_f32 {
               p0 = f32[] parameter(0)
               p1 = f32[] parameter(1)
               p2 = s8[] parameter(2)
               p3 = s8[] parameter(3)
               ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
             }
             lt_k {
               p0 = s8[] parameter(0)
               p1 = s8[] parameter(1)
               p2 = f32[] parameter(2)
               p3 = f32[] parameter(3)
               ROOT lt = pred[] compare(p0, p1), direction=LT
             }
             ENTRY main {
               k = f32[128]{0} parameter(0)
               v = f32[128]{0} parameter(1)
               i = s8[128]{0} iota(), iota_dimension=0
               sort1 = (f32[128]{0}, s8[128]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
               p = s8[128]{0} get-tuple-element(sort1), index=1
               ROOT sort2 = (s8[128]{0}, f32[128]{0}) sort(p, v), dimensions={0}, to_apply=lt_k
             }
           )"},
                                                          {256, R"(
             HloModule u8_max
             lt_f32 {
               p0 = f32[] parameter(0)
               p1 = f32[] parameter(1)
               p2 = u8[] parameter(2)
               p3 = u8[] parameter(3)
               ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
             }
             lt_k {
               p0 = u8[] parameter(0)
               p1 = u8[] parameter(1)
               p2 = f32[] parameter(2)
               p3 = f32[] parameter(3)
               ROOT lt = pred[] compare(p0, p1), direction=LT
             }
             ENTRY main {
               k = f32[256]{0} parameter(0)
               v = f32[256]{0} parameter(1)
               i = u8[256]{0} iota(), iota_dimension=0
               sort1 = (f32[256]{0}, u8[256]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
               p = u8[256]{0} get-tuple-element(sort1), index=1
               ROOT sort2 = (u8[256]{0}, f32[256]{0}) sort(p, v), dimensions={0}, to_apply=lt_k
             }
           )"}}) {
    Literal k = MakeDeterministicF32({n}, 17u);
    Literal v = MakeDeterministicF32({n}, 99u);
    v.data<float>()[0] = std::bit_cast<float>(0x80000000u);
    v.data<float>()[1] = std::bit_cast<float>(0x00000001u);
    v.data<float>()[2] = std::bit_cast<float>(0x7fc01234u);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    TF_ASSERT_OK_AND_ASSIGN(Literal before,
                            HloEvaluator().Evaluate(*m, {&k, &v}));
    EXPECT_THAT(PermutationSortExpander().Run(m.get()), IsOkAndHolds(true));
    TF_ASSERT_OK_AND_ASSIGN(Literal after,
                            HloEvaluator().Evaluate(*m, {&k, &v}));
    std::vector<Literal> before_parts = before.DecomposeTuple();
    std::vector<Literal> after_parts = after.DecomposeTuple();
    ExpectBitwiseLiteralEq(after_parts[0], before_parts[0]);
    ExpectBitwiseLiteralEq(after_parts[1], before_parts[1]);
  }
}

TEST_F(PermutationSortExpanderTest,
       WidensBatchAxisScatterIndicesWithBitwiseParity) {
  struct BatchCase {
    int64_t rows;
    int64_t cols;
    absl::string_view hlo;
  };
  const BatchCase kCases[] = {
      {200, 16, R"(
        HloModule s8_batch
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = s8[] parameter(2)
          p3 = s8[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = s8[] parameter(0)
          p1 = s8[] parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[200,16]{1,0} parameter(0)
          v = f32[200,16]{1,0} parameter(1)
          i = s8[200,16]{1,0} iota(), iota_dimension=1
          sort1 = (f32[200,16]{1,0}, s8[200,16]{1,0}) sort(k, i), dimensions={1}, to_apply=lt_f32
          p = s8[200,16]{1,0} get-tuple-element(sort1), index=1
          ROOT sort2 = (s8[200,16]{1,0}, f32[200,16]{1,0}) sort(p, v), dimensions={1}, to_apply=lt_k
        }
      )"},
      {300, 16, R"(
        HloModule u8_batch
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = u8[] parameter(2)
          p3 = u8[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = u8[] parameter(0)
          p1 = u8[] parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[300,16]{1,0} parameter(0)
          v = f32[300,16]{1,0} parameter(1)
          i = u8[300,16]{1,0} iota(), iota_dimension=1
          sort1 = (f32[300,16]{1,0}, u8[300,16]{1,0}) sort(k, i), dimensions={1}, to_apply=lt_f32
          p = u8[300,16]{1,0} get-tuple-element(sort1), index=1
          ROOT sort2 = (u8[300,16]{1,0}, f32[300,16]{1,0}) sort(p, v), dimensions={1}, to_apply=lt_k
        }
      )"},
      {300, 200, R"(
        HloModule u8_high_bit
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = u8[] parameter(2)
          p3 = u8[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = u8[] parameter(0)
          p1 = u8[] parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[300,200]{1,0} parameter(0)
          v = f32[300,200]{1,0} parameter(1)
          i = u8[300,200]{1,0} iota(), iota_dimension=1
          sort1 = (f32[300,200]{1,0}, u8[300,200]{1,0}) sort(k, i), dimensions={1}, to_apply=lt_f32
          p = u8[300,200]{1,0} get-tuple-element(sort1), index=1
          ROOT sort2 = (u8[300,200]{1,0}, f32[300,200]{1,0}) sort(p, v), dimensions={1}, to_apply=lt_k
        }
      )"},
      {20, 8, R"(
        HloModule s4_subbyte
        lt_f32 {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          p2 = s4[]{:E(4)} parameter(2)
          p3 = s4[]{:E(4)} parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
        }
        lt_k {
          p0 = s4[]{:E(4)} parameter(0)
          p1 = s4[]{:E(4)} parameter(1)
          p2 = f32[] parameter(2)
          p3 = f32[] parameter(3)
          ROOT lt = pred[] compare(p0, p1), direction=LT
        }
        ENTRY main {
          k = f32[20,8]{1,0} parameter(0)
          v = f32[20,8]{1,0} parameter(1)
          i = s4[20,8]{1,0:E(4)} iota(), iota_dimension=1
          sort1 = (f32[20,8]{1,0}, s4[20,8]{1,0:E(4)}) sort(k, i), dimensions={1}, to_apply=lt_f32
          p = s4[20,8]{1,0:E(4)} get-tuple-element(sort1), index=1
          ROOT sort2 = (s4[20,8]{1,0:E(4)}, f32[20,8]{1,0}) sort(p, v), dimensions={1}, to_apply=lt_k
        }
      )"},
  };
  for (const BatchCase& c : kCases) {
    Literal k = MakeDeterministicF32({c.rows, c.cols}, 123u);
    Literal v = MakeDeterministicF32({c.rows, c.cols}, 456u);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(c.hlo));
    TF_ASSERT_OK_AND_ASSIGN(Literal before,
                            HloEvaluator().Evaluate(*m, {&k, &v}));
    EXPECT_THAT(PermutationSortExpander().Run(m.get()), IsOkAndHolds(true));
    HloVerifier gpu_verifier(std::make_unique<CpuGpuVerifierMetadata>(
        HloVerifierOpts{}.MakeLayoutSensitive()));
    EXPECT_THAT(gpu_verifier.Run(m.get()), IsOkAndHolds(false));
    const HloInstruction* scatter =
        FindInstruction(m.get(), HloOpcode::kScatter);
    ASSERT_NE(scatter, nullptr);
    EXPECT_EQ(scatter->operand(1)->shape().element_type(), S32);
    EXPECT_EQ(scatter->operand(1)->shape().layout().element_size_in_bits(), 0);
    TF_ASSERT_OK_AND_ASSIGN(Literal after,
                            HloEvaluator().Evaluate(*m, {&k, &v}));
    std::vector<Literal> before_parts = before.DecomposeTuple();
    std::vector<Literal> after_parts = after.DecomposeTuple();
    EXPECT_EQ(after_parts[0], before_parts[0]);
    ExpectBitwiseLiteralEq(after_parts[1], before_parts[1]);
  }

  // Batch dim > 2^31 widens scatter_indices to S64 (while 2^31 stays S32).
  constexpr absl::string_view kHugeBatch = R"(
    HloModule s64_batch
    lt_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s8[] parameter(2)
      p3 = s8[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
    }
    lt_k {
      p0 = s8[] parameter(0)
      p1 = s8[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[2147483649,2]{1,0} parameter(0)
      v = f32[2147483649,2]{1,0} parameter(1)
      i = s8[2147483649,2]{1,0} iota(), iota_dimension=1
      sort1 = (f32[2147483649,2]{1,0}, s8[2147483649,2]{1,0}) sort(k, i),
        dimensions={1}, to_apply=lt_f32
      p = s8[2147483649,2]{1,0} get-tuple-element(sort1), index=1
      ROOT sort2 = (s8[2147483649,2]{1,0}, f32[2147483649,2]{1,0}) sort(p, v),
        dimensions={1}, to_apply=lt_k
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto huge, ParseAndReturnVerifiedModule(kHugeBatch));
  EXPECT_THAT(PermutationSortExpander().Run(huge.get()), IsOkAndHolds(true));
  EXPECT_EQ(FindInstruction(huge.get(), HloOpcode::kScatter)
                ->operand(1)
                ->shape()
                .element_type(),
            S64);
}

TEST_F(PermutationSortExpanderTest,
       PreservesLayoutsMixedPrecisionShardingsThreadAndControlDeps) {
  // Tiled keys, non-default output leaf layouts/memory spaces, tuple sharding,
  // and non-main execution thread.
  constexpr absl::string_view kLayoutSharding = R"(
    HloModule layout_sharding
    lt_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
    }, execution_thread="worker"
    lt_k {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }, execution_thread="worker"
    async_body {
      k = f32[16,128]{1,0:T(8,128)} parameter(0)
      v = f32[16,128]{1,0:T(8,128)} parameter(1)
      i = s32[16,128]{1,0:T(8,128)} iota(), iota_dimension=1
      sort1 = (f32[16,128]{1,0:T(8,128)}, s32[16,128]{1,0:T(8,128)}) sort(k, i),
        dimensions={1}, to_apply=lt_f32
      p = s32[16,128]{1,0:T(8,128)} get-tuple-element(sort1), index=1
      ROOT sort2 = (s32[16,128]{0,1:S(1)}, bf16[16,128]{0,1:S(2)}) sort(p, v),
        dimensions={1}, to_apply=lt_k,
        sharding={{devices=[2,1]<=[2]}, {maximal device=1}}
    }, execution_thread="worker"
    ENTRY main {
      k = f32[16,128]{1,0:T(8,128)} parameter(0)
      v = f32[16,128]{1,0:T(8,128)} parameter(1)
      st = ((f32[16,128]{1,0:T(8,128)}, f32[16,128]{1,0:T(8,128)}),
            (s32[16,128]{0,1:S(1)}, bf16[16,128]{0,1:S(2)}), s32[])
        async-start(k, v), async_execution_thread="worker", calls=async_body
      ROOT done = (s32[16,128]{0,1:S(1)}, bf16[16,128]{0,1:S(2)}) async-done(st)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnUnverifiedModule(kLayoutSharding));
  HloVerifier strict_verifier(/*layout_sensitive=*/true,
                              /*allow_mixed_precision=*/true);
  EXPECT_THAT(strict_verifier.Run(m.get()), IsOkAndHolds(false));
  EXPECT_THAT(PermutationSortExpander().Run(m.get(), {"worker"}),
              IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m.get()),
              IsOkAndHolds(false));
  const HloComputation* body = m->GetComputationWithName("async_body");
  ASSERT_NE(body, nullptr);
  const HloInstruction* root = body->root_instruction();
  EXPECT_EQ(
      root->operand(0)->shape(),
      ShapeUtil::MakeShapeWithDenseLayout(S32, {16, 128}, {0, 1}, {}, 1, 0, 1));
  EXPECT_EQ(root->operand(1)->shape(),
            ShapeUtil::MakeShapeWithDenseLayout(BF16, {16, 128}, {0, 1}, {}, 1,
                                                0, 2));
  EXPECT_TRUE(root->operand(0)->has_sharding());
  EXPECT_EQ(root->operand(1)->sharding(), HloSharding::SingleDevice(1));
  EXPECT_EQ(root->operand(1)->operand(0)->operand(0)->sharding(),
            HloSharding::SingleDevice(1));
  EXPECT_EQ(root->operand(1)->to_apply()->execution_thread(), "worker");

  // Unreduced sharding must not land on iota or scalar zero; manual sharding
  // propagates cleanly.
  constexpr absl::string_view kUnreduced = R"(
    HloModule unreduced_sort
    lt_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
    }
    lt_k {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[4,8]{1,0} parameter(0)
      v = f32[4,8]{1,0} parameter(1)
      i = s32[4,8]{1,0} iota(), iota_dimension=1
      sort1 = (f32[4,8]{1,0}, s32[4,8]{1,0}) sort(k, i), dimensions={1}, to_apply=lt_f32
      p = s32[4,8]{1,0} get-tuple-element(sort1), index=1
      ROOT sort2 = (s32[4,8]{1,0}, f32[4,8]{1,0}) sort(p, v), dimensions={1},
        to_apply=lt_k, sharding={{unreduced}, {unreduced}}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto um, ParseAndReturnVerifiedModule(kUnreduced));
  EXPECT_THAT(PermutationSortExpander().Run(um.get()), IsOkAndHolds(true));
  for (const HloInstruction* inst : um->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kIota ||
        inst->opcode() == HloOpcode::kConstant) {
      EXPECT_FALSE(inst->has_sharding() && inst->sharding().IsUnreduced());
    }
  }

  // Control deps before->sort2->after expand and survive TupleSimplifier.
  constexpr absl::string_view kCtrlOk = R"(
    HloModule ctrl_ok
    lt_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
    }
    lt_k {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[8]{0} parameter(0)
      v = f32[8]{0} parameter(1)
      before = token[] after-all()
      i = s32[8]{0} iota(), iota_dimension=0
      sort1 = (f32[8]{0}, s32[8]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
      p = s32[8]{0} get-tuple-element(sort1), index=1
      sort2 = (s32[8]{0}, f32[8]{0}) sort(p, v), dimensions={0}, to_apply=lt_k,
        control-predecessors={before}
      after = token[] after-all(), control-predecessors={sort2}
      g0 = s32[8]{0} get-tuple-element(sort2), index=0, control-predecessors={after}
      g1 = f32[8]{0} get-tuple-element(sort2), index=1
      ROOT out = (s32[8]{0}, f32[8]{0}, token[]) tuple(g0, g1, after)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto cm, ParseAndReturnVerifiedModule(kCtrlOk));
  EXPECT_THAT(PermutationSortExpander().Run(cm.get()), IsOkAndHolds(true));
  EXPECT_THAT(TupleSimplifier().Run(cm.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(false, false).Run(cm.get()), IsOkAndHolds(false));
  const HloInstruction* sc = FindInstruction(cm.get(), HloOpcode::kScatter);
  ASSERT_NE(sc, nullptr);
  EXPECT_EQ(sc->control_predecessors().size(), 1);
  EXPECT_EQ(sc->control_successors().size(), 1);
  EXPECT_EQ(cm->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->control_predecessors()
                .size(),
            2);

  // Control chain sort2->mid->gte1 declines to avoid scatter<->mid cycle.
  constexpr absl::string_view kCtrlCycle = R"(
    HloModule ctrl_cycle
    lt_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT, order=TOTAL
    }
    lt_k {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[8]{0} parameter(0)
      v = f32[8]{0} parameter(1)
      i = s32[8]{0} iota(), iota_dimension=0
      sort1 = (f32[8]{0}, s32[8]{0}) sort(k, i), dimensions={0}, to_apply=lt_f32
      p = s32[8]{0} get-tuple-element(sort1), index=1
      sort2 = (s32[8]{0}, f32[8]{0}) sort(p, v), dimensions={0}, to_apply=lt_k
      mid = token[] after-all(), control-predecessors={sort2}
      ROOT g1 = f32[8]{0} get-tuple-element(sort2), index=1,
        control-predecessors={mid}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto cyc, ParseAndReturnVerifiedModule(kCtrlCycle));
  EXPECT_THAT(PermutationSortExpander().Run(cyc.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
