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

#include "xla/hlo/transforms/simplifiers/slice_sinker.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace {

namespace m = match;
using ::testing::ElementsAre;

class SliceSinkerTest : public HloHardwareIndependentTestBase {};

TEST_F(SliceSinkerTest, TernaryOperation) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = pred[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      p2 = f32[8,9] parameter(2)
      s00 = pred[2,9] slice(pred[8,9] p0), slice={[0:2], [0:9]}
      s01 = pred[6,9] slice(pred[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      s20 = f32[2,9] slice(f32[8,9] p2), slice={[0:2], [0:9]}
      s21 = f32[6,9] slice(f32[8,9] p2), slice={[2:8], [0:9]}
      sel0 = f32[2,9] select(pred[2,9] s00, f32[2,9] s10, f32[2,9] s20)
      sel1 = f32[6,9] select(pred[6,9] s01, f32[6,9] s11, f32[6,9] s21)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(sel0, sel1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  EXPECT_THAT(inst,
              GmockMatch(m::Tuple(
                  m::Slice(&slice0, m::Select(m::Parameter(0), m::Parameter(1),
                                              m::Parameter(2))),
                  m::Slice(&slice1, m::Select(m::Parameter(0), m::Parameter(1),
                                              m::Parameter(2))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, OverlappingPartialSlicesBeneficial) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[5,9] slice(f32[8,9] p0), slice={[3:8], [0:9]}
      s02 = f32[8,4] slice(f32[8,9] p0), slice={[0:8], [0:4]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[5,9] slice(f32[8,9] p1), slice={[3:8], [0:9]}
      s12 = f32[8,4] slice(f32[8,9] p1), slice={[0:8], [0:4]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[5,9] add(f32[5,9] s01, f32[5,9] s11)
      add2 = f32[8,4] add(f32[8,4] s02, f32[8,4] s12)
      ROOT tuple = (f32[2,9], f32[5,9], f32[8,4]) tuple(add0, add1, add2)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  EXPECT_THAT(
      inst, GmockMatch(m::Tuple(
                m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice2, m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(3, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(8, 4));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, SameSliceSourcesTwoPeerGroups) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s02 = f32[8,2] slice(f32[8,9] p0), slice={[0:8], [0:2]}
      s03 = f32[8,7] slice(f32[8,9] p0), slice={[0:8], [2:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      s12 = f32[8,2] slice(f32[8,9] p1), slice={[0:8], [0:2]}
      s13 = f32[8,7] slice(f32[8,9] p1), slice={[0:8], [2:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] s11)
      mul0 = f32[8,2] multiply(f32[8,2] s02, f32[8,2] s12)
      mul1 = f32[8,7] multiply(f32[8,7] s03, f32[8,7] s13)
      ROOT tuple = (f32[2,9], f32[6,9], f32[8,2], f32[8,7]) tuple(add0, add1, mul0, mul1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  const HloInstruction* slice3;
  EXPECT_THAT(
      inst,
      GmockMatch(m::Tuple(
          m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice2, m::Multiply(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice3, m::Multiply(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(8, 2));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice3->slice_starts(), ElementsAre(0, 2));
  EXPECT_THAT(slice3->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice3->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, OverlappingMultipleSlices) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[5,9] slice(f32[8,9] p0), slice={[3:8], [0:9]}
      s02 = f32[3,9] slice(f32[8,9] p0), slice={[2:5], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[5,9] slice(f32[8,9] p1), slice={[3:8], [0:9]}
      s12 = f32[3,9] slice(f32[8,9] p1), slice={[2:5], [0:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[5,9] add(f32[5,9] s01, f32[5,9] s11)
      add2 = f32[3,9] add(f32[3,9] s02, f32[3,9] s12)
      ROOT tuple = (f32[2,9], f32[5,9], f32[3,9]) tuple(add0, add1, add2)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  EXPECT_THAT(
      inst, GmockMatch(m::Tuple(
                m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice2, m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(3, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(5, 9));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, DisjointedPartialSlices) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[5,9] slice(f32[8,9] p0), slice={[2:7], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[5,9] slice(f32[8,9] p1), slice={[2:7], [0:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[5,9] add(f32[5,9] s01, f32[5,9] s11)
      ROOT tuple = (f32[2,9], f32[5,9]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, OverlappingPartialSlicesNotBeneficial) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,7] slice(f32[8,9] p0), slice={[0:2], [0:7]}
      s01 = f32[6,7] slice(f32[8,9] p0), slice={[2:8], [0:7]}
      s10 = f32[2,7] slice(f32[8,9] p1), slice={[0:2], [0:7]}
      s11 = f32[6,7] slice(f32[8,9] p1), slice={[2:8], [0:7]}
      add0 = f32[2,7] add(f32[2,7] s00, f32[2,7] s10)
      add1 = f32[6,7] add(f32[6,7] s01, f32[6,7] s11)
      ROOT tuple = (f32[2,7], f32[6,7]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, DifferentOrderingOfSliceSources) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,7] parameter(0)
      p1 = f32[8,7] parameter(1)
      s00 = f32[2,7] slice(f32[8,7] p0), slice={[0:2], [0:7]}
      s01 = f32[6,7] slice(f32[8,7] p0), slice={[2:8], [0:7]}
      s10 = f32[2,7] slice(f32[8,7] p1), slice={[0:2], [0:7]}
      s11 = f32[6,7] slice(f32[8,7] p1), slice={[2:8], [0:7]}
      add0 = f32[2,7] add(f32[2,7] s00, f32[2,7] s10)
      add1 = f32[6,7] add(f32[6,7] s11, f32[6,7] s01)
      ROOT tuple = (f32[2,7], f32[6,7]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SlicesFromDifferentIndices) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[4,9] slice(f32[8,9] p0), slice={[0:4], [0:9]}
      s01 = f32[4,9] slice(f32[8,9] p0), slice={[4:8], [0:9]}
      s10 = f32[4,9] slice(f32[8,9] p1), slice={[0:4], [0:9]}
      s11 = f32[4,9] slice(f32[8,9] p1), slice={[4:8], [0:9]}
      add0 = f32[4,9] add(f32[4,9] s01, f32[4,9] s10)
      add1 = f32[4,9] add(f32[4,9] s00, f32[4,9] s11)
      ROOT tuple = (f32[4,9], f32[4,9]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, DifferentOperator) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      mul = f32[2,9] multiply(f32[2,9] s00, f32[2,9] s10)
      add = f32[6,9] add(f32[6,9] s01, f32[6,9] s11)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(mul, add)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SameOperatorDifferentAttributes) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      cmp1 = pred[2,9] compare(f32[2,9] s00, f32[2,9] s10), direction=GT
      cmp2 = pred[6,9] compare(f32[6,9] s01, f32[6,9] s11), direction=LT
      ROOT tuple = (pred[2,9], pred[6,9]) tuple(cmp1, cmp2)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SlicesWithMultiUsers) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] s11)
      mul0 = f32[2,9] multiply(f32[2,9] s00, f32[2,9] s10)
      mul1 = f32[6,9] multiply(f32[6,9] s01, f32[6,9] s11)
      ROOT tuple = (f32[2,9], f32[6,9], f32[2,9], f32[6,9]) tuple(add0, add1, mul0, mul1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  const HloInstruction* slice3;
  EXPECT_THAT(
      inst,
      GmockMatch(m::Tuple(
          m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice2, m::Multiply(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice3, m::Multiply(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice3->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice3->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice3->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, NonElementWise) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8] parameter(0)
      s00 = f32[2] slice(f32[8] p0), slice={[0:2]}
      s01 = f32[6] slice(f32[8] p0), slice={[2:8]}
      bc0 = f32[2,9] broadcast(f32[2] s00), dimensions={0}
      bc1 = f32[6,9] broadcast(f32[6] s01), dimensions={0}
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(bc0, bc1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SlicesWithNontrivialStrides) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[4,9] slice(f32[8,9] p0), slice={[0:7:2], [0:9]}
      s01 = f32[4,9] slice(f32[8,9] p0), slice={[1:8:2], [0:9]}
      s10 = f32[4,9] slice(f32[8,9] p1), slice={[0:7:2], [0:9]}
      s11 = f32[4,9] slice(f32[8,9] p1), slice={[1:8:2], [0:9]}
      add0 = f32[4,9] add(f32[4,9] s00, f32[4,9] s10)
      add1 = f32[4,9] add(f32[4,9] s01, f32[4,9] s11)
      ROOT tuple = (f32[4,9], f32[4,9]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  EXPECT_THAT(
      inst, GmockMatch(m::Tuple(
                m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(7, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(2, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(1, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(2, 1));
}

TEST_F(SliceSinkerTest, NotAllSliceOperand) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[2,9] parameter(1)
      p2 = f32[6,9] parameter(2)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      abs0 = f32[2,9] abs(f32[2,9] p1)
      abs1 = f32[6,9] abs(f32[6,9] p2)
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] abs0)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] abs1)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, Cascade) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      abs0 = f32[2,9] abs(f32[2,9] s10)
      abs1 = f32[6,9] abs(f32[6,9] s11)
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] abs0)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] abs1)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(add0, add1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  EXPECT_THAT(
      inst,
      GmockMatch(m::Tuple(
          m::Slice(&slice0, m::Add(m::Parameter(0), m::Abs(m::Parameter(1)))),
          m::Slice(&slice1,
                   m::Add(m::Parameter(0), m::Abs(m::Parameter(1)))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, SameOpcodeDifferentResultElementTypes) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      convert0 = s32[2,9] convert(f32[2,9] s00)
      convert1 = s64[6,9] convert(f32[6,9] s01)
      ROOT tuple = (s32[2,9], s64[6,9]) tuple(convert0, convert1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest,
       DenseSubrangeWithOffsetAndCroppedCascadeAndScalarAllowlist) {
  const char* kModuleStr = R"(
    HloModule m
    ENTRY main {
      p0 = u32[8] parameter(0)
      p1 = u32[8] parameter(1)
      s0 = u32[1] slice(p0), slice={[2:3]}
      s1 = u32[1] slice(p0), slice={[3:4]}
      s2 = u32[1] slice(p0), slice={[4:5]}
      s3 = u32[1] slice(p0), slice={[5:6]}
      r0 = u32[] reshape(s0)
      r1 = u32[] reshape(s1)
      r2 = u32[] reshape(s2)
      r3 = u32[] reshape(s3)
      t0 = u32[1] slice(p1), slice={[2:3]}
      t1 = u32[1] slice(p1), slice={[3:4]}
      t2 = u32[1] slice(p1), slice={[4:5]}
      t3 = u32[1] slice(p1), slice={[5:6]}
      q0 = u32[] reshape(t0)
      q1 = u32[] reshape(t1)
      q2 = u32[] reshape(t2)
      q3 = u32[] reshape(t3)
      z0 = u32[] clz(r0)
      z1 = u32[] clz(r1)
      z2 = u32[] clz(r2)
      z3 = u32[] clz(r3)
      c_a = u32[] constant(1)
      c_b = u32[] constant(1)
      sh0 = u32[] shift-right-logical(q0, c_a)
      sh1 = u32[] shift-right-logical(q1, c_b)
      sh2 = u32[] shift-right-logical(q2, c_a)
      sh3 = u32[] shift-right-logical(q3, c_b)
      a0 = u32[] add(z0, sh0)
      a1 = u32[] add(z1, sh1)
      a2 = u32[] add(z2, sh2)
      a3 = u32[] add(z3, sh3)
      f0 = f32[] convert(a0)
      f1 = f32[] convert(a1)
      f2 = f32[] convert(a2)
      f3 = f32[] convert(a3)
      l0 = f32[] log(f0)
      l1 = f32[] log(f1)
      l2 = f32[] log(f2)
      l3 = f32[] log(f3)
      // Also combine cropped f32[4] with uncropped p0[2:6] using r0..r3 twice
      // (multi-use reshape peeling + mixed [8] and cropped [4] sources).
      g0 = f32[] convert(r0)
      g1 = f32[] convert(r1)
      g2 = f32[] convert(r2)
      g3 = f32[] convert(r3)
      m0 = f32[] add(l0, g0)
      m1 = f32[] add(l1, g1)
      m2 = f32[] add(l2, g2)
      m3 = f32[] add(l3, g3)
      ROOT out = (f32[], f32[], f32[], f32[]) tuple(m0, m1, m2, m3)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  int clz_count = 0;
  int log_count = 0;
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->user_count() == 0 &&
        inst != module->entry_computation()->root_instruction()) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kClz) {
      ++clz_count;
      EXPECT_THAT(inst->shape().dimensions(), ElementsAre(4));
    }
    if (inst->opcode() == HloOpcode::kLog) {
      ++log_count;
      EXPECT_THAT(inst->shape().dimensions(), ElementsAre(4));
    }
  }
  EXPECT_EQ(clz_count, 1);
  EXPECT_EQ(log_count, 1);
}

TEST_F(SliceSinkerTest,
       OperandZeroInvariantBroadcastPerSliceScalarBroadcastAndMultiUseReshape) {
  const char* kModuleStr = R"(
    HloModule m
    ENTRY main {
      logits = f32[4,8,16] parameter(0)
      cot = f32[8] parameter(1)
      scale = f32[4] parameter(2)
      s0 = f32[1,8,16] slice(logits), slice={[0:1], [0:8], [0:16]}
      s1 = f32[1,8,16] slice(logits), slice={[1:2], [0:8], [0:16]}
      s2 = f32[1,8,16] slice(logits), slice={[2:3], [0:8], [0:16]}
      s3 = f32[1,8,16] slice(logits), slice={[3:4], [0:8], [0:16]}
      r0 = f32[8,16] reshape(s0)
      r1 = f32[8,16] reshape(s1)
      r2 = f32[8,16] reshape(s2)
      r3 = f32[8,16] reshape(s3)
      e0 = f32[8,16] exponential(r0)
      e1 = f32[8,16] exponential(r1)
      e2 = f32[8,16] exponential(r2)
      e3 = f32[8,16] exponential(r3)
      bc0 = f32[8,16] broadcast(cot), dimensions={0}
      bc1 = f32[8,16] broadcast(cot), dimensions={0}
      m0 = f32[8,16] multiply(bc0, e0)
      m1 = f32[8,16] multiply(bc1, e1)
      m2 = f32[8,16] multiply(bc0, e2)
      m3 = f32[8,16] multiply(bc1, e3)
      k0 = f32[] reshape(f32[1] slice(scale), slice={[0:1]})
      k1 = f32[] reshape(f32[1] slice(scale), slice={[1:2]})
      k2 = f32[] reshape(f32[1] slice(scale), slice={[2:3]})
      k3 = f32[] reshape(f32[1] slice(scale), slice={[3:4]})
      kb0 = f32[8,16] broadcast(k0), dimensions={}
      kb1 = f32[8,16] broadcast(k1), dimensions={}
      kb2 = f32[8,16] broadcast(k2), dimensions={}
      kb3 = f32[8,16] broadcast(k3), dimensions={}
      // Reuse r0..r3 in a second op to verify multi-use reshape sinking.
      d0 = f32[8,16] add(m0, kb0)
      d1 = f32[8,16] add(m1, kb1)
      d2 = f32[8,16] add(m2, kb2)
      d3 = f32[8,16] add(m3, kb3)
      u0 = f32[8,16] subtract(d0, r0)
      u1 = f32[8,16] subtract(d1, r1)
      u2 = f32[8,16] subtract(d2, r2)
      u3 = f32[8,16] subtract(d3, r3)
      ROOT out = (f32[8,16], f32[8,16], f32[8,16], f32[8,16]) tuple(u0, u1, u2, u3)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  int live_sub = 0;
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->user_count() > 0 && inst->opcode() == HloOpcode::kSubtract) {
      ++live_sub;
      EXPECT_THAT(inst->shape().dimensions(), ElementsAre(4, 8, 16));
    }
  }
  EXPECT_EQ(live_sub, 1);
}

TEST_F(SliceSinkerTest, RejectsLiftedTensorExceedingEightMibCap) {
  const char* kModuleStr = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,1024,1024] parameter(0)
      p1 = f32[2,1024,1024] parameter(1)
      // Full source is 2 * 1024 * 1024 * 4 = 8,388,608 B (allowed); 3 slices
      // of [1,1024,1024] from f32[3,1024,1024] is 12 MiB > 8 MiB (rejected).
      q0 = f32[3,1024,1024] parameter(2)
      q1 = f32[3,1024,1024] parameter(3)
      a0 = f32[1,1024,1024] slice(q0), slice={[0:1], [0:1024], [0:1024]}
      a1 = f32[1,1024,1024] slice(q0), slice={[1:2], [0:1024], [0:1024]}
      a2 = f32[1,1024,1024] slice(q0), slice={[2:3], [0:1024], [0:1024]}
      b0 = f32[1,1024,1024] slice(q1), slice={[0:1], [0:1024], [0:1024]}
      b1 = f32[1,1024,1024] slice(q1), slice={[1:2], [0:1024], [0:1024]}
      b2 = f32[1,1024,1024] slice(q1), slice={[2:3], [0:1024], [0:1024]}
      x0 = f32[1,1024,1024] add(a0, b0)
      x1 = f32[1,1024,1024] add(a1, b1)
      x2 = f32[1,1024,1024] add(a2, b2)
      ROOT out = (f32[2,1024,1024], f32[2,1024,1024], f32[1,1024,1024], f32[1,1024,1024], f32[1,1024,1024]) tuple(p0, p1, x0, x1, x2)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, RejectsNonZeroUlpScalarLifts) {
  for (const char* hlo : {
           // Float-to-unsigned scalar convert (NaN divergence).
           R"(
             HloModule f2u
             ENTRY main {
               p = f32[2] parameter(0)
               a = u32[1] convert(f32[1] slice(p), slice={[0:1]})
               b = u32[1] convert(f32[1] slice(p), slice={[1:2]})
               ROOT t = (u32[1], u32[1]) tuple(a, b)
             })",
           // Narrowing F32 -> BF16 scalar convert.
           R"(
             HloModule f2bf16
             ENTRY main {
               p = f32[2] parameter(0)
               a = bf16[1] convert(f32[1] slice(p), slice={[0:1]})
               b = bf16[1] convert(f32[1] slice(p), slice={[1:2]})
               ROOT t = (bf16[1], bf16[1]) tuple(a, b)
             })",
           // BF16 1-element arithmetic.
           R"(
             HloModule bf16_scalar
             ENTRY main {
               p = bf16[2] parameter(0)
               s0 = bf16[1] slice(p), slice={[0:1]}
               s1 = bf16[1] slice(p), slice={[1:2]}
               ROOT t = (bf16[1], bf16[1]) tuple(bf16[1] add(s0, s0), bf16[1] add(s1, s1))
             })",
           // Scalar ceil and clamp (NaN and signed-zero divergence).
           R"(
             HloModule ceil_scalar
             ENTRY main {
               p = f32[2] parameter(0)
               a = f32[1] ceil(f32[1] slice(p), slice={[0:1]})
               b = f32[1] ceil(f32[1] slice(p), slice={[1:2]})
               ROOT t = (f32[1], f32[1]) tuple(a, b)
             })",
           R"(
             HloModule clamp_scalar
             ENTRY main {
               p = f32[2] parameter(0)
               lo = f32[1] constant({0.0})
               hi = f32[1] constant({1.0})
               a = f32[1] clamp(lo, f32[1] slice(p), slice={[0:1]}, hi)
               b = f32[1] clamp(lo, f32[1] slice(p), slice={[1:2]}, hi)
               ROOT t = (f32[1], f32[1]) tuple(a, b)
             })",
           // Total-order float compare and copy.
           R"(
             HloModule totalorder_compare
             ENTRY main {
               p = f32[2] parameter(0)
               q = f32[2] parameter(1)
               a = pred[1] compare(f32[1] slice(p), slice={[0:1]}, f32[1] slice(q), slice={[0:1]}), direction=LT, type=TOTALORDER
               b = pred[1] compare(f32[1] slice(p), slice={[1:2]}, f32[1] slice(q), slice={[1:2]}), direction=LT, type=TOTALORDER
               ROOT t = (pred[1], pred[1]) tuple(a, b)
             })",
           R"(
             HloModule copy_excluded
             ENTRY main {
               p = f32[8] parameter(0)
               a = f32[4] copy(f32[4] slice(p), slice={[0:4]})
               b = f32[4] copy(f32[4] slice(p), slice={[4:8]})
               ROOT t = (f32[4], f32[4]) tuple(a, b)
             })",
           // Scalar F32 divide over convert operand (promoted-BF16 guard).
           R"(
             HloModule promoted_divide
             ENTRY main {
               p = f32[2] parameter(0)
               d = bf16[] parameter(1)
               df = f32[] convert(d)
               r0 = f32[] reshape(f32[1] slice(p), slice={[0:1]})
               r1 = f32[] reshape(f32[1] slice(p), slice={[1:2]})
               ROOT t = (f32[], f32[]) tuple(f32[] divide(r0, df), f32[] divide(r1, df))
             })",
           // Tiled post-layout slices are ignored.
           R"(
             HloModule tiled_layout_excluded
             ENTRY main {
               p = f32[8,128]{1,0:T(8,128)} parameter(0)
               s0 = f32[4,128]{1,0:T(4,128)} slice(p), slice={[0:4], [0:128]}
               s1 = f32[4,128]{1,0:T(4,128)} slice(p), slice={[4:8], [0:128]}
               ROOT t = (f32[4,128], f32[4,128]) tuple(f32[4,128]{1,0:T(4,128)} negate(s0), f32[4,128]{1,0:T(4,128)} negate(s1))
             })",
       }) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    SliceSinker slice_sinker;
    ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
    EXPECT_FALSE(result) << module->name();
  }
}

TEST_F(SliceSinkerTest, RejectsControlDependencyCycleAndRoleTwoOnlyGroup) {
  const char* kControlDepHlo = R"(
    HloModule control_dep_cycle
    ENTRY main {
      p = f32[4] parameter(0)
      c = f32[] constant(1.0)
      r0 = f32[] reshape(f32[1] slice(p), slice={[0:1]})
      r1 = f32[] reshape(f32[1] slice(p), slice={[1:2]})
      r2 = f32[] reshape(f32[1] slice(p), slice={[2:3]})
      r3 = f32[] reshape(f32[1] slice(p), slice={[3:4]})
      a0 = f32[] add(r0, c)
      u = f32[] negate(a0)
      a1 = f32[] add(r1, c), control-predecessors={u}
      // Role-ii-only group (per-slice scalar broadcasts + invariant broadcast,
      // zero primary slices) must not lift.
      iota = f32[8] parameter(1)
      b0 = f32[8] broadcast(r2), dimensions={}
      b1 = f32[8] broadcast(r3), dimensions={}
      ib = f32[8] broadcast(c), dimensions={}
      m0 = f32[8] add(b0, ib)
      m1 = f32[8] add(b1, ib)
      ROOT out = (f32[], f32[], f32[8], f32[8], f32[8]) tuple(u, a1, m0, m1, iota)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kControlDepHlo));
  SliceSinker slice_sinker;
  ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace xla
