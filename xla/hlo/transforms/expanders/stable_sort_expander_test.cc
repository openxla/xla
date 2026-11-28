/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/stable_sort_expander.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = match;

using StableSortExpanderTest = HloHardwareIndependentTestBase;

// Checks whether 'a' and 'b' are roots of equivalent computations, except that
// parameters 2 * i and 2 * i + 1 are switched.
bool IsSameComputationExceptParams(const HloInstruction* a,
                                   const HloInstruction* b) {
  if (a->opcode() != b->opcode() || a->operand_count() != b->operand_count()) {
    return false;
  }
  if (a->opcode() == HloOpcode::kParameter) {
    // Check that parameters were switched.
    return a->parameter_number() == (b->parameter_number() ^ 1);
  }
  // If the operation has no operands, it should actually be the same.
  if (a->operand_count() == 0) {
    return a == b;
  }
  // Otherwise recursively compare all operands.
  for (int64_t i = 0; i < a->operand_count(); ++i) {
    if (!IsSameComputationExceptParams(a->operand(i), b->operand(i))) {
      return false;
    }
  }
  return true;
}

// Check that the comparison computation has been modified to add a tie breaker
// using 'iota_parameter'.
void CheckComputationHasTieBreaker(const HloInstruction* root,
                                   int64_t iota_parameter) {
  // With the tie breaker, the root instruction should be
  //   Select(Eq(Comp(), CompReverse()), Lt(), Comp())
  // with Comp() being the original comparison function, and CompReverse() being
  // the copied comparison function where the parameters are reversed. Lt() is
  // the tie breaker comparison using the Iota operand.
  ASSERT_EQ(root->opcode(), HloOpcode::kSelect);
  ASSERT_EQ(root->operand(0)->opcode(), HloOpcode::kCompare);
  ASSERT_EQ(root->operand(0)->comparison_direction(), ComparisonDirection::kEq);

  // Check that the tie breaker instruction is correct.
  EXPECT_THAT(root->operand(1),
              GmockMatch(m::Lt(m::Parameter(iota_parameter * 2),
                               m::Parameter(iota_parameter * 2 + 1))));
  EXPECT_EQ(root->operand(2), root->operand(0)->operand(0));

  // Check that Comp() and CompReverse() are equivalent except that
  // CompReverse() has reversed parameters.
  EXPECT_TRUE(IsSameComputationExceptParams(root->operand(0)->operand(0),
                                            root->operand(0)->operand(1)));
}

TEST_F(StableSortExpanderTest, StabilizeSortReuseIotaOperand) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0)));
  const HloInstruction* sort_instr = root->operand(0);
  ASSERT_EQ(sort_instr->opcode(), HloOpcode::kSort);
  EXPECT_FALSE(Cast<const HloSortInstruction>(sort_instr)->is_stable());
  CheckComputationHasTieBreaker(sort_instr->to_apply()->root_instruction(),
                                /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest,
       StabilizeSortReuseIotaOperandComplicatedComparison) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     max = u32[] constant(2147483647)
     zero = s32[] constant(0)
     lhs.signed = s32[] bitcast-convert(p.0.lhs)
     lhs.unsigned = u32[] bitcast-convert(p.0.lhs)
     lhs.flipped = u32[] subtract(max, lhs.unsigned)
     lhs.flipped.signed = s32[] bitcast-convert(lhs.flipped)
     lhs.is_negative = pred[] compare(lhs.flipped.signed, zero), direction=LT
     lhs.converted = s32[] select(lhs.is_negative, lhs.flipped.signed, lhs.signed)
     rhs.signed = s32[] bitcast-convert(p.0.rhs)
     rhs.unsigned = u32[] bitcast-convert(p.0.rhs)
     rhs.flipped = u32[] subtract(max, rhs.unsigned)
     rhs.flipped.signed = s32[] bitcast-convert(rhs.flipped)
     rhs.is_negative = pred[] compare(rhs.flipped.signed, zero), direction=LT
     rhs.converted = s32[] select(rhs.is_negative, rhs.flipped.signed, rhs.signed)
     ROOT lt = pred[] compare(lhs.converted, rhs.converted), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(), /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest, StabilizeSortAddIotaOperandAndChangeRoot) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} parameter(1)
     ROOT sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, GmockMatch(m::Tuple(
                m::GetTupleElement(
                    m::Sort(m::Parameter(0), m::Parameter(1), m::Iota()), 0),
                m::GetTupleElement(
                    m::Sort(m::Parameter(0), m::Parameter(1), m::Iota()), 1))));
  const HloInstruction* sort_instr = root->operand(0)->operand(0);
  ASSERT_EQ(sort_instr->opcode(), HloOpcode::kSort);
  EXPECT_FALSE(Cast<const HloSortInstruction>(sort_instr)->is_stable());
  CheckComputationHasTieBreaker(sort_instr->to_apply()->root_instruction(),
                                /*iota_parameter=*/2);
}

TEST_F(StableSortExpanderTest, HonorIsStableFlag) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=false
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_FALSE(stabilizer.Run(module.get()).value());
}

TEST_F(StableSortExpanderTest,
       StabilizeSortDontReuseIotaOperandWrongDimension) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=0
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  // Simplify away the "wrapper" tuple around the new sort.
  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota(), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/2);
}

TEST_F(StableSortExpanderTest, StabilizeSortDontReuseIotaOperandWrongType) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = f32[] parameter(2)
     p.1.rhs = f32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = f32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, f32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  // Simplify away the "wrapper" tuple around the new sort.
  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota(), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/2);
}

TEST_F(StableSortExpanderTest, StabilizeSortR1) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = s32[] parameter(0)
     p.0.rhs = s32[] parameter(1)
     mask = s32[] constant(65535)
     lhs = s32[] and(p.0.lhs, mask)
     rhs = s32[] and(p.0.rhs, mask)
     ROOT lt = pred[] compare(lhs, rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = s32[64,8732]{1,0} parameter(0)
     ROOT sort = s32[64,8732]{1,0} sort(keys), dimensions={0}, to_apply=compare,
       is_stable=true
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(), /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest, StabilizeSortR1NoRoot) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = s32[] parameter(0)
     p.0.rhs = s32[] parameter(1)
     mask = s32[] constant(65535)
     lhs = s32[] and(p.0.lhs, mask)
     rhs = s32[] and(p.0.rhs, mask)
     ROOT lt = pred[] compare(lhs, rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = s32[64,8732]{1,0} parameter(0)
     sort = s32[64,8732]{1,0} sort(keys), dimensions={0}, to_apply=compare,
       is_stable=true
     ROOT neg = s32[64,8732]{1,0} negate(sort)
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Negate(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0))));
  CheckComputationHasTieBreaker(
      root->operand(0)->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest, MultipleSortsSingleComputationNoIota) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    compare {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys.0 = f32[64,8732]{1,0} parameter(0)
      keys.1 = f32[64,8732]{1,0} parameter(1)
      sort.0 = f32[64,8732]{1,0} sort(keys.0),
        dimensions={1}, to_apply=compare, is_stable=true
      sort.1 = f32[64,8732]{1,0} sort(keys.1),
        dimensions={1}, to_apply=compare, is_stable=true
      ROOT add = f32[64,8732]{1,0} add(sort.0, sort.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::Add(
                  m::GetTupleElement(m::Sort(m::Parameter(0), m::Iota()), 0),
                  m::GetTupleElement(m::Sort(m::Parameter(1), m::Iota()), 0))));
  EXPECT_NE(root->operand(0)->operand(0)->to_apply()->unique_id(),
            root->operand(1)->operand(0)->to_apply()->unique_id());
}

TEST_F(StableSortExpanderTest, MultipleSortsSingleComputationWithIota) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    compare {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys.0 = f32[64,8732]{1,0} parameter(0)
      keys.1 = f32[64,8732]{1,0} parameter(1)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort.0 = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys.0, values),
        dimensions={1}, to_apply=compare, is_stable=true
      sort.1 = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys.1, values),
        dimensions={1}, to_apply=compare, is_stable=true
      gte.0 = f32[64,8732]{1,0} get-tuple-element(sort.0), index=0
      gte.1 = f32[64,8732]{1,0} get-tuple-element(sort.1), index=0
      ROOT add = f32[64,8732]{1,0} add(gte.0, gte.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::Add(
                  m::GetTupleElement(m::Sort(m::Parameter(0), m::Iota()), 0),
                  m::GetTupleElement(m::Sort(m::Parameter(1), m::Iota()), 0))));
  EXPECT_NE(root->operand(0)->operand(0)->to_apply()->unique_id(),
            root->operand(1)->operand(0)->to_apply()->unique_id());
}

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST_F(StableSortExpanderTest,
       ExpandsZeroOperandConstantAndParameterComparatorRoots) {
  constexpr absl::string_view kConstFalse = R"(
    HloModule const_false_cmp
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT c = pred[] constant(false)
    }
    ENTRY main {
      x = f32[5]{0} parameter(0)
      ROOT s = f32[5]{0} sort(x), dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  // Includes -0.0f, +0.0f, subnormal, and two distinct NaN payloads.
  const uint32_t kBits[5] = {0x80000000u, 0x00000000u, 0x7fc00456u, 0x00000001u,
                             0x7fc00123u};
  Literal input = LiteralUtil::CreateFromDimensions(F32, {5});
  for (int i = 0; i < 5; ++i) {
    input.data<float>()[i] = std::bit_cast<float>(kBits[i]);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kConstFalse));
  TF_ASSERT_OK_AND_ASSIGN(Literal before,
                          HloEvaluator().Evaluate(*m, {&input}));
  EXPECT_THAT(StableSortExpander().Run(m.get()), IsOkAndHolds(true));
  const HloInstruction* sort =
      m->entry_computation()->root_instruction()->operand(0);
  CheckComputationHasTieBreaker(sort->to_apply()->root_instruction(),
                                /*iota_parameter=*/1);
  TF_ASSERT_OK_AND_ASSIGN(Literal after, HloEvaluator().Evaluate(*m, {&input}));
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(std::bit_cast<uint32_t>(after.data<float>()[i]), kBits[i]);
    EXPECT_EQ(std::bit_cast<uint32_t>(before.data<float>()[i]), kBits[i]);
  }

  constexpr absl::string_view kParamRoot = R"(
    HloModule param_root_cmp
    cmp {
      ROOT p0 = pred[] parameter(0)
      p1 = pred[] parameter(1)
    }
    ENTRY main {
      x = pred[4]{0} parameter(0)
      ROOT s = pred[4]{0} sort(x), dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto pm, ParseAndReturnVerifiedModule(kParamRoot));
  EXPECT_THAT(StableSortExpander().Run(pm.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(true, false).Run(pm.get()), IsOkAndHolds(false));
  CheckComputationHasTieBreaker(pm->entry_computation()
                                    ->root_instruction()
                                    ->operand(0)
                                    ->to_apply()
                                    ->root_instruction(),
                                /*iota_parameter=*/1);
  Literal p_in = LiteralUtil::CreateR1<bool>({false, true, false, true});
  TF_ASSERT_OK_AND_ASSIGN(Literal p_out, HloEvaluator().Evaluate(*pm, {&p_in}));
  EXPECT_EQ(p_out, LiteralUtil::CreateR1<bool>({true, true, false, false}));
}

TEST_F(StableSortExpanderTest,
       HandlesLargeSortDimensionsAndReusesWideIotasInPriorityOrder) {
  // Exact 2^31 boundary stays S32; 2^31 + 1 uses S64 and refuses wrapped S32.
  constexpr absl::string_view kS32Max = R"(
    HloModule s32_max
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[2147483648]{0} parameter(0)
      ROOT s = f32[2147483648]{0} sort(k), dimensions={0}, is_stable=true,
        to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_max, ParseAndReturnVerifiedModule(kS32Max));
  EXPECT_THAT(StableSortExpander().Run(m_max.get()), IsOkAndHolds(true));
  EXPECT_EQ(m_max->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->operand(1)
                ->shape()
                .element_type(),
            S32);

  constexpr absl::string_view kWrappedS32 = R"(
    HloModule wrapped_s32
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[2147483649]{0} parameter(0)
      i = s32[2147483649]{0} iota(), iota_dimension=0
      ROOT s = (f32[2147483649]{0}, s32[2147483649]{0}) sort(k, i),
        dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_wrap,
                          ParseAndReturnVerifiedModule(kWrappedS32));
  EXPECT_THAT(StableSortExpander().Run(m_wrap.get()), IsOkAndHolds(true));
  const HloInstruction* wrap_sort =
      m_wrap->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_EQ(wrap_sort->operand_count(), 3);
  EXPECT_EQ(wrap_sort->operand(2)->shape().element_type(), S64);
  CheckComputationHasTieBreaker(wrap_sort->to_apply()->root_instruction(), 2);

  // Reuse U32 at 2^31 + 1, refuse wrapped U32 at 2^32 + 1 (adding S64), and
  // prefer S32 over S64 when both are present.
  constexpr absl::string_view kReuseU32 = R"(
    HloModule reuse_u32
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = u32[] parameter(2)
      p3 = u32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[2147483649]{0} parameter(0)
      i = u32[2147483649]{0} iota(), iota_dimension=0
      ROOT s = (f32[2147483649]{0}, u32[2147483649]{0}) sort(k, i),
        dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_u32, ParseAndReturnVerifiedModule(kReuseU32));
  EXPECT_THAT(StableSortExpander().Run(m_u32.get()), IsOkAndHolds(true));
  const HloInstruction* u32_sort =
      m_u32->entry_computation()->root_instruction();
  EXPECT_EQ(u32_sort->operand_count(), 2);
  CheckComputationHasTieBreaker(u32_sort->to_apply()->root_instruction(), 1);

  constexpr absl::string_view kWrappedU32 = R"(
    HloModule wrapped_u32
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = u32[] parameter(2)
      p3 = u32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[4294967297]{0} parameter(0)
      i = u32[4294967297]{0} iota(), iota_dimension=0
      ROOT s = (f32[4294967297]{0}, u32[4294967297]{0}) sort(k, i),
        dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_wu32,
                          ParseAndReturnVerifiedModule(kWrappedU32));
  EXPECT_THAT(StableSortExpander().Run(m_wu32.get()), IsOkAndHolds(true));
  const HloInstruction* wu32_sort =
      m_wu32->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_EQ(wu32_sort->operand_count(), 3);
  EXPECT_EQ(wu32_sort->operand(2)->shape().element_type(), S64);

  constexpr absl::string_view kPreferS32OverS64 = R"(
    HloModule prefer_s32
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s64[] parameter(2)
      p3 = s64[] parameter(3)
      p4 = s32[] parameter(4)
      p5 = s32[] parameter(5)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[8]{0} parameter(0)
      i64 = s64[8]{0} iota(), iota_dimension=0
      i32 = s32[8]{0} iota(), iota_dimension=0
      ROOT s = (f32[8]{0}, s64[8]{0}, s32[8]{0}) sort(k, i64, i32),
        dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_pref,
                          ParseAndReturnVerifiedModule(kPreferS32OverS64));
  EXPECT_THAT(StableSortExpander().Run(m_pref.get()), IsOkAndHolds(true));
  const HloInstruction* pref_sort =
      m_pref->entry_computation()->root_instruction();
  EXPECT_EQ(pref_sort->operand_count(), 3);
  CheckComputationHasTieBreaker(pref_sort->to_apply()->root_instruction(), 2);

  // Reuse S64 and U64 iotas with exact HloEvaluator tie-break parity.
  constexpr absl::string_view kReuseU64 = R"(
    HloModule reuse_u64
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = u64[] parameter(2)
      p3 = u64[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[6]{0} parameter(0)
      i = u64[6]{0} iota(), iota_dimension=0
      ROOT s = (f32[6]{0}, u64[6]{0}) sort(k, i), dimensions={0},
        is_stable=true, to_apply=cmp
    }
  )";
  Literal ties =
      LiteralUtil::CreateR1<float>({2.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f});
  TF_ASSERT_OK_AND_ASSIGN(auto m_u64, ParseAndReturnVerifiedModule(kReuseU64));
  TF_ASSERT_OK_AND_ASSIGN(Literal b_u64,
                          HloEvaluator().Evaluate(*m_u64, {&ties}));
  EXPECT_THAT(StableSortExpander().Run(m_u64.get()), IsOkAndHolds(true));
  EXPECT_EQ(m_u64->entry_computation()->root_instruction()->operand_count(), 2);
  TF_ASSERT_OK_AND_ASSIGN(Literal a_u64,
                          HloEvaluator().Evaluate(*m_u64, {&ties}));
  EXPECT_EQ(a_u64, b_u64);

  // Unbounded dynamic sort fails cleanly without mutating the module.
  constexpr absl::string_view kUnbounded = R"(
    HloModule unbounded_sort
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[?]{0} parameter(0)
      ROOT s = f32[?]{0} sort(k), dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_unb,
                          ParseAndReturnUnverifiedModule(kUnbounded));
  const int64_t unb_count = m_unb->entry_computation()->instruction_count();
  EXPECT_THAT(StableSortExpander().Run(m_unb.get()),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_EQ(m_unb->entry_computation()->instruction_count(), unb_count);
}

TEST_F(StableSortExpanderTest,
       PreservesSubByteTiledDynamicAndNonDefaultOutputLayouts) {
  // Sub-byte s4[16]{0:E(4)}: synthesized S32 iota clears element_size_in_bits.
  constexpr absl::string_view kSubByte = R"(
    HloModule s4_stable
    cmp {
      p0 = s4[]{:E(4)} parameter(0)
      p1 = s4[]{:E(4)} parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = s4[16]{0:E(4)} parameter(0)
      ROOT s = s4[16]{0:E(4)} sort(k), dimensions={0}, is_stable=true,
        to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto ms4, ParseAndReturnVerifiedModule(kSubByte));
  EXPECT_THAT(StableSortExpander().Run(ms4.get()), IsOkAndHolds(true));
  HloVerifier gpu_verifier(std::make_unique<CpuGpuVerifierMetadata>(
      HloVerifierOpts{}.MakeLayoutSensitive()));
  EXPECT_THAT(gpu_verifier.Run(ms4.get()), IsOkAndHolds(false));
  const HloInstruction* s4_iota =
      ms4->entry_computation()->root_instruction()->operand(0)->operand(1);
  EXPECT_EQ(s4_iota->shape().element_type(), S32);
  EXPECT_EQ(s4_iota->shape().layout().element_size_in_bits(), 0);

  // Tiled bf16 with sub-tile (8,128)(2,1) and non-default variadic output
  // layouts/memory spaces pass layout-sensitive verification.
  constexpr absl::string_view kTiledAndLayouts = R"(
    HloModule tiled_layouts
    cmp {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      p2 = s16[] parameter(2)
      p3 = s16[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = bf16[16,128]{1,0:T(8,128)(2,1)} parameter(0)
      v = s16[16,128]{1,0:T(8,128)(2,1)} parameter(1)
      ROOT s = (bf16[16,128]{0,1:S(1)}, s16[16,128]{0,1:S(2)}) sort(k, v),
        dimensions={1}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m_tl,
                          ParseAndReturnVerifiedModule(kTiledAndLayouts));
  EXPECT_THAT(StableSortExpander().Run(m_tl.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(true, false).Run(m_tl.get()), IsOkAndHolds(false));
  const HloInstruction* tl_root = m_tl->entry_computation()->root_instruction();
  EXPECT_EQ(tl_root->operand(0)->shape().layout().memory_space(), 1);
  EXPECT_EQ(tl_root->operand(1)->shape().layout().memory_space(), 2);
  EXPECT_TRUE(tl_root->operand(0)
                  ->operand(0)
                  ->operand(2)
                  ->shape()
                  .layout()
                  .tiles()
                  .empty());

  // Bounded dynamic sort synthesizes static iota + SetDimensionSize and
  // evaluates identically on active slice; bound > INT32_MAX keeps static S64.
  constexpr absl::string_view kBoundedDyn = R"(
    HloModule bounded_dyn
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[6]{0} parameter(0)
      v = s32[6]{0} parameter(1)
      n = s32[] parameter(2)
      kd = f32[<=6]{0} set-dimension-size(k, n), dimensions={0}
      vd = s32[<=6]{0} set-dimension-size(v, n), dimensions={0}
      ROOT s = (f32[<=6]{0}, s32[<=6]{0}) sort(kd, vd), dimensions={0},
        is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto md, ParseAndReturnVerifiedModule(kBoundedDyn));
  EXPECT_THAT(StableSortExpander().Run(md.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(true, false).Run(md.get()), IsOkAndHolds(false));
  const HloInstruction* dyn_sort =
      md->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* dyn_iota = dyn_sort->operand(2);
  ASSERT_EQ(dyn_iota->opcode(), HloOpcode::kSetDimensionSize);
  EXPECT_TRUE(dyn_iota->shape().is_dynamic_dimension(0));
  EXPECT_EQ(dyn_iota->operand(0)->opcode(), HloOpcode::kIota);
  EXPECT_FALSE(dyn_iota->operand(0)->shape().is_dynamic());
  EXPECT_EQ(dyn_iota->operand(1)->opcode(), HloOpcode::kGetDimensionSize);
  CheckComputationHasTieBreaker(dyn_sort->to_apply()->root_instruction(), 2);

  constexpr absl::string_view kHugeDyn = R"(
    HloModule huge_dyn
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[<=2147483649]{0} parameter(0)
      ROOT s = f32[<=2147483649]{0} sort(k), dimensions={0}, is_stable=true,
        to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto mhd, ParseAndReturnVerifiedModule(kHugeDyn));
  EXPECT_THAT(StableSortExpander().Run(mhd.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(true, false).Run(mhd.get()), IsOkAndHolds(false));
}

TEST_F(StableSortExpanderTest,
       PreservesShardingsMetadataControlDepsSharedEntryAndRejectsSideEffects) {
  // Variadic sharding, op metadata, control deps (including sort->mid->gte0),
  // and intra-comparator control edges.
  constexpr absl::string_view kShardedCtrl = R"(
    HloModule sharded_ctrl
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s16[] parameter(2)
      p3 = s16[] parameter(3)
      n0 = f32[] abs(p0)
      n1 = f32[] abs(p1), control-predecessors={n0}
      ROOT lt = pred[] compare(n0, n1), direction=LT
    }
    ENTRY main {
      k = f32[8,4]{1,0} parameter(0)
      v = s16[8,4]{1,0} parameter(1)
      before = token[] after-all()
      s = (f32[8,4]{1,0}, s16[8,4]{1,0}) sort(k, v), dimensions={1},
        is_stable=true, to_apply=cmp, control-predecessors={before},
        sharding={{devices=[2,1]<=[2]}, {replicated}},
        metadata={op_name="stable_sort_op"}
      mid = token[] after-all(), control-predecessors={s}
      g0 = f32[8,4]{1,0} get-tuple-element(s), index=0,
        control-predecessors={mid}
      g1 = s16[8,4]{1,0} get-tuple-element(s), index=1
      ROOT out = (f32[8,4]{1,0}, s16[8,4]{1,0}, token[]) tuple(g0, g1, mid)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto ms, ParseAndReturnVerifiedModule(kShardedCtrl));
  EXPECT_THAT(StableSortExpander().Run(ms.get()), IsOkAndHolds(true));
  EXPECT_THAT(TupleSimplifier().Run(ms.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(true, false).Run(ms.get()), IsOkAndHolds(false));
  const HloInstruction* new_sort = FindInstruction(ms.get(), HloOpcode::kSort);
  ASSERT_NE(new_sort, nullptr);
  EXPECT_FALSE(Cast<HloSortInstruction>(new_sort)->is_stable());
  ASSERT_TRUE(new_sort->has_sharding());
  EXPECT_EQ(new_sort->sharding().tuple_elements().size(), 3);
  EXPECT_TRUE(new_sort->sharding().tuple_elements()[1].IsReplicated());
  EXPECT_EQ(new_sort->sharding().tuple_elements()[2],
            new_sort->sharding().tuple_elements()[0]);
  EXPECT_EQ(new_sort->operand(2)->sharding(),
            new_sort->sharding().tuple_elements()[0]);
  EXPECT_EQ(new_sort->operand(2)->metadata().op_name(), "stable_sort_op");
  EXPECT_EQ(new_sort->to_apply()->root_instruction()->metadata().op_name(),
            "stable_sort_op");
  EXPECT_EQ(new_sort->control_predecessors().size(), 1);
  EXPECT_EQ(new_sort->control_successors().size(), 1);
  int cloned_ctrl = 0;
  for (const HloInstruction* inst : new_sort->to_apply()->instructions()) {
    if (inst->opcode() == HloOpcode::kAbs &&
        !inst->control_predecessors().empty()) {
      ++cloned_ctrl;
    }
  }
  EXPECT_EQ(cloned_ctrl, 2);

  // Unreduced sharding is not placed on synthesized iota or get-dimension-size.
  constexpr absl::string_view kUnreduced = R"(
    HloModule unreduced_stable
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[<=8]{0} parameter(0)
      ROOT s = f32[<=8]{0} sort(k), dimensions={0}, is_stable=true,
        to_apply=cmp, sharding={unreduced}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto mu, ParseAndReturnVerifiedModule(kUnreduced));
  EXPECT_THAT(StableSortExpander().Run(mu.get()), IsOkAndHolds(true));
  for (const HloInstruction* inst : mu->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kIota ||
        inst->opcode() == HloOpcode::kGetDimensionSize ||
        inst->opcode() == HloOpcode::kSetDimensionSize) {
      EXPECT_FALSE(inst->has_sharding() && inst->sharding().IsUnreduced());
    }
  }

  // Comparator that is also the module entry computation is cloned rather than
  // mutating ENTRY in place.
  constexpr absl::string_view kSharedEntry = R"(
    HloModule shared_entry
    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    sub {
      k = f32[4]{0} parameter(0)
      i = s32[4]{0} iota(), iota_dimension=0
      ROOT s = (f32[4]{0}, s32[4]{0}) sort(k, i), dimensions={0},
        is_stable=true, to_apply=main
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto me, ParseAndReturnVerifiedModule(kSharedEntry));
  EXPECT_THAT(StableSortExpander().Run(me.get()), IsOkAndHolds(true));
  EXPECT_EQ(me->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kCompare);
  EXPECT_NE(me->GetComputationWithName("sub")->root_instruction()->to_apply(),
            me->entry_computation());

  // Side-effecting comparator returns FailedPrecondition before any mutation.
  constexpr absl::string_view kSideEffect = R"(
    HloModule side_effect_stable
    cmp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      se = f32[] custom-call(p0), custom_call_target="side_effect",
        custom_call_has_side_effect=true
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }
    ENTRY main {
      k = f32[8]{0} parameter(0)
      ROOT s = f32[8]{0} sort(k), dimensions={0}, is_stable=true, to_apply=cmp
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto mse, ParseAndReturnVerifiedModule(kSideEffect));
  const int64_t se_insts = mse->entry_computation()->instruction_count();
  const int64_t se_comps = mse->computation_count();
  EXPECT_THAT(StableSortExpander().Run(mse.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(mse->entry_computation()->instruction_count(), se_insts);
  EXPECT_EQ(mse->computation_count(), se_comps);
  EXPECT_TRUE(
      Cast<HloSortInstruction>(mse->entry_computation()->root_instruction())
          ->is_stable());
}

}  // namespace
}  // namespace xla
