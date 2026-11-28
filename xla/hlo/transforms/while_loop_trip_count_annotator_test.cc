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

#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;

class TripCountAnnotatorTest : public HloHardwareIndependentTestBase {};

TEST_F(TripCountAnnotatorTest, KnownSmallTripCount) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_TRUE(config.has_known_induction_variable());
  EXPECT_TRUE(config.has_known_init_step());
  EXPECT_EQ(config.known_trip_count().n(), 10);
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 0);
  EXPECT_EQ(config.known_init_step().init(), 0);
  EXPECT_EQ(config.known_init_step().step(), 1);
}

TEST_F(TripCountAnnotatorTest, KnownLargeTripCount) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(1000000)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 1000000);
}

TEST_F(TripCountAnnotatorTest, NonzeroStartStep) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      two = s32[] constant(2)
      i_plus_two = s32[] add(i, two)
      ROOT tuple = (s32[]) tuple(i_plus_two)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      max_i = s32[] constant(1000000)
      ROOT done = pred[] compare(i, max_i), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(10)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 499995);
  EXPECT_TRUE(config.has_known_init_step());
  EXPECT_EQ(config.known_init_step().init(), 10);
  EXPECT_EQ(config.known_init_step().step(), 2);
}

TEST_F(TripCountAnnotatorTest, LessThanOrEqualTo) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(1000000)
      ROOT done = pred[] compare(i, trip_count), direction=LE
    }

    ENTRY test {
      i_start = s32[] constant(10)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 999991);
}

TEST_F(TripCountAnnotatorTest, Int64Overflow) {
  // for(i = INT64_MIN; i < INT64_MAX; ++i)
  //
  // We store the trip count as an int64_t, so this loop is unanalyzable.
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s64[]) parameter(0)
      i = s64[] get-tuple-element(param), index=0
      one = s64[] constant(1)
      i_plus_one = s64[] add(i, one)
      ROOT tuple = (s64[]) tuple(i_plus_one)
    }

    Cond {
      param = (s64[]) parameter(0)
      i = s64[] get-tuple-element(param), index=0
      trip_count = s64[] constant(9223372036854775807) // 2^63-1
      ROOT done = pred[] compare(i, trip_count), direction=LE
    }

    ENTRY test {
      i_start = s64[] constant(-9223372036854775808)  // -2^63
      initial_tuple = (s64[]) tuple(i_start)
      ROOT while = (s64[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_FALSE(config.has_known_trip_count());
  EXPECT_FALSE(config.has_known_init_step());
  EXPECT_TRUE(config.has_known_induction_variable());
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 0);
}

TEST_F(TripCountAnnotatorTest, FillsDynamicVariableInitStep) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], s32[], f32[20,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      counter = s32[] get-tuple-element(param), index=1
      buf = f32[20,8] get-tuple-element(param), index=2
      one = s32[] constant(1)
      zero = s32[] constant(0)
      i_plus = s32[] add(i, one)
      c_plus = s32[] add(counter, one)
      to_host = f32[20,8] custom-call(buf), custom_call_target="MoveToHost"
      slice = f32[1,8] dynamic-slice(to_host, counter, zero), dynamic_slice_sizes={1,8}
      to_dev = f32[1,8] custom-call(slice), custom_call_target="MoveToDevice"
      next_buf = f32[20,8] dynamic-update-slice(buf, to_dev, counter, zero)
      ROOT tuple = (s32[], s32[], f32[20,8]) tuple(i_plus, c_plus, next_buf)
    }

    Cond {
      param = (s32[], s32[], f32[20,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      ten = s32[] constant(10)
      ROOT done = pred[] compare(i, ten), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      c_start = s32[] constant(5)
      buf_start = f32[20,8] parameter(0)
      initial_tuple = (s32[], s32[], f32[20,8]) tuple(i_start, c_start, buf_start)
      ROOT while = (s32[], s32[], f32[20,8]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_THAT(config, EqualsProto(R"pb(
                known_trip_count { n: 10 }
                known_induction_variable { tuple_index: 0 }
                known_init_step { init: 0 step: 1 }
                dynamic_variables { tuple_index: 1 init: 5 step: 1 }
              )pb"));
}

TEST_F(TripCountAnnotatorTest, FillsDynamicVariableInitStepFromPrimaryCopy) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], s32[], f32[20,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      shadow = s32[] get-tuple-element(param), index=1
      buf = f32[20,8] get-tuple-element(param), index=2
      one = s32[] constant(1)
      zero = s32[] constant(0)
      i_plus = s32[] add(i, one)
      to_host = f32[20,8] custom-call(buf), custom_call_target="MoveToHost"
      slice = f32[1,8] dynamic-slice(to_host, shadow, zero), dynamic_slice_sizes={1,8}
      to_dev = f32[1,8] custom-call(slice), custom_call_target="MoveToDevice"
      next_buf = f32[20,8] dynamic-update-slice(buf, to_dev, shadow, zero)
      ROOT tuple = (s32[], s32[], f32[20,8]) tuple(i_plus, i, next_buf)
    }

    Cond {
      param = (s32[], s32[], f32[20,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      ten = s32[] constant(10)
      ROOT done = pred[] compare(i, ten), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(1)
      shadow_init = s32[] constant(0)
      buf_start = f32[20,8] parameter(0)
      initial_tuple = (s32[], s32[], f32[20,8]) tuple(i_start, shadow_init, buf_start)
      ROOT while = (s32[], s32[], f32[20,8]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_THAT(config, EqualsProto(R"pb(
                known_trip_count { n: 9 }
                known_induction_variable { tuple_index: 0 }
                known_init_step { init: 1 step: 1 }
                dynamic_variables { tuple_index: 1 init: 0 step: 1 }
              )pb"));
}

TEST_F(TripCountAnnotatorTest, NonZeroTupleIndex) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[], s32[]) tuple(one, i_plus_one)
    }

    Cond {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=1
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[], s32[]) tuple(i_start, i_start)
      ROOT while = (s32[], s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 1);
}

TEST_F(TripCountAnnotatorTest, InductionVarForwardedToConstant) {
  // for(i = 0; i < 10; ++i) {}
  // use(GTE(while, 0))  --> should become use(constant(10))
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      data = f32[10] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[], f32[10]) tuple(i_plus_one, data)
    }

    Cond {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      data = f32[10] parameter(0)
      initial_tuple = (s32[], f32[10]) tuple(i_start, data)
      while = (s32[], f32[10]) while(initial_tuple), condition=Cond, body=Body
      i_final = s32[] get-tuple-element(while), index=0
      ROOT result = f32[10] get-tuple-element(while), index=1
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // Find the GTE that used to extract index 0 -- it should now have no users
  // and a constant should exist with value 10.
  HloComputation* entry = m->entry_computation();
  bool found_constant = false;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions().empty() &&
        instr->literal().Get<int32_t>({}) == 10) {
      found_constant = true;
    }
  }
  EXPECT_TRUE(found_constant);
}

TEST_F(TripCountAnnotatorTest, InductionVarForwardedNonZeroInit) {
  // for(i = 5; i < 15; i += 2) {} -> trip_count = 5, final_value = 15
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      data = f32[10] get-tuple-element(param), index=1
      two = s32[] constant(2)
      i_plus_two = s32[] add(i, two)
      ROOT tuple = (s32[], f32[10]) tuple(i_plus_two, data)
    }

    Cond {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(15)
      ROOT done = pred[] compare(i, limit), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(5)
      data = f32[10] parameter(0)
      initial_tuple = (s32[], f32[10]) tuple(i_start, data)
      while = (s32[], f32[10]) while(initial_tuple), condition=Cond, body=Body
      i_final = s32[] get-tuple-element(while), index=0
      ROOT result = f32[10] get-tuple-element(while), index=1
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // trip_count = (15 - 5) / 2 = 5, final_value = 5 + 2 * 5 = 15
  HloComputation* entry = m->entry_computation();
  bool found_constant = false;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions().empty() &&
        instr->literal().Get<int32_t>({}) == 15) {
      found_constant = true;
    }
  }
  EXPECT_TRUE(found_constant);
}

TEST_F(TripCountAnnotatorTest, InductionVarMultipleGTEsForwarded) {
  // Multiple GTEs extracting the induction variable should all be replaced.
  const char* kModuleStr = R"(
    HloModule test

    add_comp {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    Body {
      param = (s32[], s32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      data = s32[4] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[], s32[4]) tuple(i_plus_one, data)
    }

    Cond {
      param = (s32[], s32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT done = pred[] compare(i, limit), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      data = s32[4] parameter(0)
      initial_tuple = (s32[], s32[4]) tuple(i_start, data)
      while = (s32[], s32[4]) while(initial_tuple), condition=Cond, body=Body
      i_final_1 = s32[] get-tuple-element(while), index=0
      i_final_2 = s32[] get-tuple-element(while), index=0
      data_out = s32[4] get-tuple-element(while), index=1
      zero = s32[] constant(0)
      ds1 = s32[1] dynamic-slice(data_out, i_final_1), dynamic_slice_sizes={1}
      ds2 = s32[1] dynamic-slice(data_out, i_final_2), dynamic_slice_sizes={1}
      ROOT result = s32[] reduce(ds1, zero), dimensions={0}, to_apply=add_comp
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // Both GTEs at index 0 should have been replaced with constant(4).
  HloComputation* entry = m->entry_computation();
  int constant_4_count = 0;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions().empty() &&
        instr->literal().Get<int32_t>({}) == 4) {
      constant_4_count++;
    }
  }
  // We create one constant per GTE replaced.
  EXPECT_GE(constant_4_count, 2);

  // Verify the dynamic-slice ops now consume constants, not GTEs.
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kDynamicSlice) {
      EXPECT_EQ(instr->operand(1)->opcode(), HloOpcode::kConstant);
    }
  }
}

TEST_F(TripCountAnnotatorTest, InductionVarNotForwardedForBruteForce) {
  // When trip count is only known via brute-force (no init/step), we don't
  // forward to constant because we can't compute the final value.
  // We use a loop where MatchTrivialLoopRange fails but brute force succeeds.
  // A loop with s64 and large constant won't match trivial range due to
  // overflow concerns, but brute force can handle it if trip count is small.
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(5)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // This particular loop will match the trivial range, so the constant
  // should be created. This test just validates the pass doesn't crash.
  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 5);
}

TEST_F(TripCountAnnotatorTest, InductionVarNonZeroTupleIndexForwarded) {
  // Induction variable at tuple index 1 should be forwarded correctly.
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (f32[10], s32[]) parameter(0)
      data = f32[10] get-tuple-element(param), index=0
      i = s32[] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (f32[10], s32[]) tuple(data, i_plus_one)
    }

    Cond {
      param = (f32[10], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=1
      limit = s32[] constant(7)
      ROOT done = pred[] compare(i, limit), direction=LT
    }

    ENTRY test {
      data = f32[10] parameter(0)
      i_start = s32[] constant(0)
      initial_tuple = (f32[10], s32[]) tuple(data, i_start)
      while = (f32[10], s32[]) while(initial_tuple), condition=Cond, body=Body
      i_final = s32[] get-tuple-element(while), index=1
      ROOT result = f32[10] get-tuple-element(while), index=0
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // final_value = 0 + 1 * 7 = 7
  HloComputation* entry = m->entry_computation();
  bool found_constant_7 = false;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions().empty() &&
        instr->literal().Get<int32_t>({}) == 7) {
      found_constant_7 = true;
    }
  }
  EXPECT_TRUE(found_constant_7);
}

TEST_F(TripCountAnnotatorTest, OverwritesPrePopulatedBackendConfig) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while_loop = (s32[]) while(initial_tuple), condition=Cond, body=Body,
        backend_config={"known_trip_count":{"n":"99"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_op = m->entry_computation()->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  ASSERT_OK_AND_ASSIGN(WhileLoopBackendConfig config,
                       while_op->backend_config<WhileLoopBackendConfig>());
  EXPECT_THAT(config, EqualsProto(R"pb(
                known_induction_variable { tuple_index: 0 }
                known_trip_count { n: 10 }
                known_init_step { init: 0 step: 1 }
              )pb"));
}

std::string MakeCountedLoopHlo(absl::string_view ty, absl::string_view init,
                               absl::string_view bound, absl::string_view step,
                               absl::string_view dir, bool indvar_on_lhs) {
  const absl::string_view cmp_lhs = indvar_on_lhs ? "i" : "bound";
  const absl::string_view cmp_rhs = indvar_on_lhs ? "bound" : "i";
  return absl::StrFormat(
      R"hlo(
    HloModule counted_loop
    Body {
      param = (%1$s[]) parameter(0)
      i = %1$s[] get-tuple-element(param), index=0
      st = %1$s[] constant(%4$s)
      next = %1$s[] add(i, st)
      ROOT tuple = (%1$s[]) tuple(next)
    }
    Cond {
      param = (%1$s[]) parameter(0)
      i = %1$s[] get-tuple-element(param), index=0
      bound = %1$s[] constant(%3$s)
      ROOT done = pred[] compare(%6$s, %7$s), direction=%5$s
    }
    ENTRY main {
      init = %1$s[] constant(%2$s)
      init_tuple = (%1$s[]) tuple(init)
      while_loop = (%1$s[]) while(init_tuple), condition=Cond, body=Body
      ROOT out = %1$s[] get-tuple-element(while_loop), index=0
    })hlo",
      ty, init, bound, step, dir, cmp_lhs, cmp_rhs);
}

std::string MakeSelfAffineDynVarHlo(absl::string_view ty,
                                    absl::string_view c_init,
                                    absl::string_view c_update_body,
                                    int64_t trip_count) {
  return absl::StrFormat(
      R"hlo(
    HloModule self_affine
    Body {
      param = (s32[], %1$s[], f32[20,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      c = %1$s[] get-tuple-element(param), index=1
      buf = f32[20,8] get-tuple-element(param), index=2
      one = s32[] constant(1)
      zero = %1$s[] constant(0)
      i_next = s32[] add(i, one)
      %3$s
      to_host = f32[20,8] custom-call(buf), custom_call_target="MoveToHost"
      slice = f32[1,8] dynamic-slice(to_host, c, zero), dynamic_slice_sizes={1,8}
      to_dev = f32[1,8] custom-call(slice), custom_call_target="MoveToDevice"
      next_buf = f32[20,8] dynamic-update-slice(buf, to_dev, c, zero)
      ROOT tuple = (s32[], %1$s[], f32[20,8]) tuple(i_next, c_next, next_buf)
    }
    Cond {
      param = (s32[], %1$s[], f32[20,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      lim = s32[] constant(%4$d)
      ROOT done = pred[] compare(i, lim), direction=LT
    }
    ENTRY main {
      i0 = s32[] constant(0)
      c0 = %1$s[] constant(%2$s)
      buf0 = f32[20,8] parameter(0)
      init = (s32[], %1$s[], f32[20,8]) tuple(i0, c0, buf0)
      ROOT w = (s32[], %1$s[], f32[20,8]) while(init), condition=Cond, body=Body
    })hlo",
      ty, c_init, c_update_body, trip_count);
}

TEST_F(TripCountAnnotatorTest, ZeroTripForwardSignedLoop) {
  struct Case {
    const char* init;
    const char* bound;
    const char* step;
    int32_t expected_init;
  };
  for (const Case& c : {Case{"5", "4", "1", 5}, Case{"8", "5", "2", 8},
                        Case{"100", "1", "1", 100}}) {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(MakeCountedLoopHlo(
                             "s32", c.init, c.bound, c.step, "LT", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);

    HloInstruction* while_op =
        m->entry_computation()->GetInstructionWithName("while_loop");
    ASSERT_OK_AND_ASSIGN(auto config,
                         while_op->backend_config<WhileLoopBackendConfig>());
    ASSERT_TRUE(config.has_known_trip_count());
    EXPECT_EQ(config.known_trip_count().n(), 0);
    EXPECT_FALSE(config.has_known_init_step());

    const HloInstruction* root = m->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kConstant);
    EXPECT_EQ(root->literal().Get<int32_t>({}), c.expected_init);
  }
}

TEST_F(TripCountAnnotatorTest, ZeroTripNarrowAndUnsignedConstantValueWrap) {
  // u32: i=0, i<0, step=1 and i=5, i<1, step=20.
  for (const auto& [init_str, bound_str, step_str, want] :
       {std::tuple{"0", "0", "1", uint32_t{0}},
        std::tuple{"5", "1", "20", uint32_t{5}}}) {
    ASSERT_OK_AND_ASSIGN(
        auto m, ParseAndReturnVerifiedModule(MakeCountedLoopHlo(
                    "u32", init_str, bound_str, step_str, "LT", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_TRUE(cfg.has_known_trip_count());
    EXPECT_EQ(cfg.known_trip_count().n(), 0);
    EXPECT_FALSE(cfg.has_known_init_step());
    ASSERT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConstant);
    EXPECT_EQ(
        m->entry_computation()->root_instruction()->literal().Get<uint32_t>({}),
        want);
  }

  // s32: i=0, i<INT32_MIN, step=1.
  {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(MakeCountedLoopHlo(
                             "s32", "0", "-2147483648", "1", "LT", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_TRUE(cfg.has_known_trip_count());
    EXPECT_EQ(cfg.known_trip_count().n(), 0);
    EXPECT_FALSE(cfg.has_known_init_step());
    ASSERT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConstant);
    EXPECT_EQ(
        m->entry_computation()->root_instruction()->literal().Get<int32_t>({}),
        0);
  }

  // s8: i=0, i<-120, step=20.
  {
    ASSERT_OK_AND_ASSIGN(
        auto m, ParseAndReturnVerifiedModule(
                    MakeCountedLoopHlo("s8", "0", "-120", "20", "LT", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_TRUE(cfg.has_known_trip_count());
    EXPECT_EQ(cfg.known_trip_count().n(), 0);
    EXPECT_FALSE(cfg.has_known_init_step());
    ASSERT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConstant);
    EXPECT_EQ(
        m->entry_computation()->root_instruction()->literal().Get<int8_t>({}),
        0);
  }

  // Valid swapped-operand forward loop: s32 i=0; 1000000 > i; i += 1.
  {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(MakeCountedLoopHlo(
                             "s32", "0", "1000000", "1", "GT", false)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    EXPECT_EQ(cfg.known_trip_count().n(), 1000000);
    EXPECT_EQ(cfg.known_init_step().init(), 0);
    EXPECT_EQ(cfg.known_init_step().step(), 1);
    ASSERT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConstant);
    EXPECT_EQ(
        m->entry_computation()->root_instruction()->literal().Get<int32_t>({}),
        1000000);
  }
}

TEST_F(TripCountAnnotatorTest, ReversedConditionLoopSafety) {
  // s32: i=5, i>=3, step=1 runs until signed overflow; no static trip count.
  {
    ASSERT_OK_AND_ASSIGN(
        auto m, ParseAndReturnVerifiedModule(
                    MakeCountedLoopHlo("s32", "5", "3", "1", "GE", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    EXPECT_FALSE(cfg.has_known_trip_count());
    EXPECT_FALSE(cfg.has_known_init_step());
    EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kGetTupleElement);
  }

  // s32: i=0, i>42, step=1 exits on iteration 0 with value 0 (not 42).
  {
    ASSERT_OK_AND_ASSIGN(
        auto m, ParseAndReturnVerifiedModule(
                    MakeCountedLoopHlo("s32", "0", "42", "1", "GT", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_TRUE(cfg.has_known_trip_count());
    EXPECT_EQ(cfg.known_trip_count().n(), 0);
    EXPECT_FALSE(cfg.has_known_init_step());
    ASSERT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConstant);
    EXPECT_EQ(
        m->entry_computation()->root_instruction()->literal().Get<int32_t>({}),
        0);
  }
}

TEST_F(TripCountAnnotatorTest, S64BoundaryLoops) {
  // s64: i=0, i<INT64_MAX, step=1 executes INT64_MAX trips, ending at
  // INT64_MAX.
  {
    ASSERT_OK_AND_ASSIGN(
        auto m, ParseAndReturnVerifiedModule(MakeCountedLoopHlo(
                    "s64", "0", "9223372036854775807", "1", "LT", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    EXPECT_EQ(cfg.known_trip_count().n(), std::numeric_limits<int64_t>::max());
    EXPECT_EQ(cfg.known_init_step().init(), 0);
    EXPECT_EQ(cfg.known_init_step().step(), 1);
    ASSERT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConstant);
    EXPECT_EQ(
        m->entry_computation()->root_instruction()->literal().Get<int64_t>({}),
        std::numeric_limits<int64_t>::max());
  }

  // Wrapping s64 loops leave known_trip_count unset without signed overflow.
  for (const auto& [bound_str, step_str] :
       {std::pair{"9223372036854775806", "2"},
        std::pair{"9223372036854775807", "1"}}) {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(MakeCountedLoopHlo(
                             "s64", "0", bound_str, step_str, "LE", true)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->GetInstructionWithName("while_loop")
                             ->backend_config<WhileLoopBackendConfig>());
    EXPECT_FALSE(cfg.has_known_trip_count());
    EXPECT_FALSE(cfg.has_known_init_step());
    EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kGetTupleElement);
  }
}

TEST_F(TripCountAnnotatorTest, DecomposeLinearAndLaggedS64NoUbsan) {
  // subtract(c, INT64_MIN) and add(add(c, INT64_MAX), 1) with 3 trips wrap at
  // k=2 and must not trigger signed int64_t overflow.
  for (absl::string_view update :
       {absl::string_view{"k = s64[] constant(-9223372036854775808)\n"
                          "c_next = s64[] subtract(c, k)"},
        absl::string_view{"kmax = s64[] constant(9223372036854775807)\n"
                          "k1 = s64[] constant(1)\n"
                          "t = s64[] add(c, kmax)\n"
                          "c_next = s64[] add(t, k1)"}}) {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(
                             MakeSelfAffineDynVarHlo("s64", "0", update, 3)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->root_instruction()
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_EQ(cfg.dynamic_variables_size(), 1);
    EXPECT_FALSE(cfg.dynamic_variables(0).has_init());
    EXPECT_FALSE(cfg.dynamic_variables(0).has_step());
  }

  // Lagged s64 copy with primary_init=2, primary_step=1, add(i, INT64_MAX).
  constexpr absl::string_view kLaggedS64 = R"hlo(
    HloModule lagged_s64
    Body {
      param = (s64[], s64[], f32[20,8]) parameter(0)
      i = s64[] get-tuple-element(param), index=0
      shadow = s64[] get-tuple-element(param), index=1
      buf = f32[20,8] get-tuple-element(param), index=2
      one = s64[] constant(1)
      zero = s64[] constant(0)
      kmax = s64[] constant(9223372036854775807)
      i_next = s64[] add(i, one)
      s_next = s64[] add(i, kmax)
      to_host = f32[20,8] custom-call(buf), custom_call_target="MoveToHost"
      slice = f32[1,8] dynamic-slice(to_host, shadow, zero), dynamic_slice_sizes={1,8}
      to_dev = f32[1,8] custom-call(slice), custom_call_target="MoveToDevice"
      next_buf = f32[20,8] dynamic-update-slice(buf, to_dev, shadow, zero)
      ROOT tuple = (s64[], s64[], f32[20,8]) tuple(i_next, s_next, next_buf)
    }
    Cond {
      param = (s64[], s64[], f32[20,8]) parameter(0)
      i = s64[] get-tuple-element(param), index=0
      lim = s64[] constant(5)
      ROOT done = pred[] compare(i, lim), direction=LT
    }
    ENTRY main {
      i0 = s64[] constant(2)
      s0 = s64[] constant(0)
      buf0 = f32[20,8] parameter(0)
      init = (s64[], s64[], f32[20,8]) tuple(i0, s0, buf0)
      ROOT w = (s64[], s64[], f32[20,8]) while(init), condition=Cond, body=Body
    })hlo";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kLaggedS64));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(auto cfg,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  ASSERT_EQ(cfg.dynamic_variables_size(), 1);
  EXPECT_FALSE(cfg.dynamic_variables(0).has_init());
  EXPECT_FALSE(cfg.dynamic_variables(0).has_step());
}

TEST_F(TripCountAnnotatorTest, NarrowAndUnsignedDynamicVariables) {
  // Wrapping updates leave dynamic_variables init/step unset.
  struct RejectCase {
    const char* ty;
    const char* init;
    const char* body;
  };
  for (const RejectCase& rc : {
           RejectCase{"s32", "0",
                      "k = s32[] constant(1500000000)\n"
                      "t = s32[] add(c, k)\n"
                      "c_next = s32[] add(t, k)"},
           RejectCase{"s32", "1000000000",
                      "k = s32[] constant(1500000000)\n"
                      "c_next = s32[] add(c, k)"},
           RejectCase{"u64", "0",
                      "k = u64[] constant(9223372036854775807)\n"
                      "c_next = u64[] add(c, k)"},
       }) {
    ASSERT_OK_AND_ASSIGN(
        auto m, ParseAndReturnVerifiedModule(
                    MakeSelfAffineDynVarHlo(rc.ty, rc.init, rc.body, 3)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->root_instruction()
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_EQ(cfg.dynamic_variables_size(), 1);
    EXPECT_FALSE(cfg.dynamic_variables(0).has_init());
    EXPECT_FALSE(cfg.dynamic_variables(0).has_step());
  }

  // S8 lagged copy wrapping mid-loop (i=100..104, shadow=126, add(i, 27)).
  constexpr absl::string_view kS8Lagged = R"hlo(
    HloModule s8_lagged
    Body {
      param = (s8[], s8[], f32[20,8]) parameter(0)
      i = s8[] get-tuple-element(param), index=0
      shadow = s8[] get-tuple-element(param), index=1
      buf = f32[20,8] get-tuple-element(param), index=2
      one = s8[] constant(1)
      zero = s8[] constant(0)
      d = s8[] constant(27)
      i_next = s8[] add(i, one)
      s_next = s8[] add(i, d)
      to_host = f32[20,8] custom-call(buf), custom_call_target="MoveToHost"
      slice = f32[1,8] dynamic-slice(to_host, shadow, zero), dynamic_slice_sizes={1,8}
      to_dev = f32[1,8] custom-call(slice), custom_call_target="MoveToDevice"
      next_buf = f32[20,8] dynamic-update-slice(buf, to_dev, shadow, zero)
      ROOT tuple = (s8[], s8[], f32[20,8]) tuple(i_next, s_next, next_buf)
    }
    Cond {
      param = (s8[], s8[], f32[20,8]) parameter(0)
      i = s8[] get-tuple-element(param), index=0
      lim = s8[] constant(105)
      ROOT done = pred[] compare(i, lim), direction=LT
    }
    ENTRY main {
      i0 = s8[] constant(100)
      s0 = s8[] constant(126)
      buf0 = f32[20,8] parameter(0)
      init = (s8[], s8[], f32[20,8]) tuple(i0, s0, buf0)
      ROOT w = (s8[], s8[], f32[20,8]) while(init), condition=Cond, body=Body
    })hlo";
  {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kS8Lagged));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->root_instruction()
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_EQ(cfg.dynamic_variables_size(), 1);
    EXPECT_FALSE(cfg.dynamic_variables(0).has_init());
    EXPECT_FALSE(cfg.dynamic_variables(0).has_step());
  }

  // U32 countdown add(c, 0xFFFFFFFF), U64 countdown add(c, UINT64_MAX), and
  // wrapped-to-1 S32 update (2147483647 + 2147483647 + 3 == 1 mod 2^32).
  struct AcceptCase {
    const char* ty;
    const char* init;
    const char* body;
    int64_t trips;
    int64_t want_init;
    int64_t want_step;
  };
  for (const AcceptCase& ac : {
           AcceptCase{"u32", "10",
                      "m1 = u32[] constant(4294967295)\n"
                      "c_next = u32[] add(c, m1)",
                      5, 10, -1},
           AcceptCase{"u64", "10",
                      "m1 = u64[] constant(18446744073709551615)\n"
                      "c_next = u64[] add(c, m1)",
                      5, 10, -1},
           AcceptCase{"s32", "0",
                      "kmax = s32[] constant(2147483647)\n"
                      "k3 = s32[] constant(3)\n"
                      "t1 = s32[] add(c, kmax)\n"
                      "t2 = s32[] add(t1, kmax)\n"
                      "c_next = s32[] add(t2, k3)",
                      10, 0, 1},
       }) {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(MakeSelfAffineDynVarHlo(
                             ac.ty, ac.init, ac.body, ac.trips)));
    WhileLoopTripCountAnnotator pass;
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK_AND_ASSIGN(auto cfg,
                         m->entry_computation()
                             ->root_instruction()
                             ->backend_config<WhileLoopBackendConfig>());
    ASSERT_EQ(cfg.dynamic_variables_size(), 1);
    EXPECT_EQ(cfg.dynamic_variables(0).init(), ac.want_init);
    EXPECT_EQ(cfg.dynamic_variables(0).step(), ac.want_step);
  }
}

TEST_F(TripCountAnnotatorTest,
       ForwardedGtePreservesControlDepsLayoutAndSharding) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule gte_attrs
    Body {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      v = s32[] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_next = s32[] add(i, one)
      ROOT tuple = (s32[], s32[]) tuple(i_next, v)
    }
    Cond {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      lim = s32[] constant(6)
      ROOT done = pred[] compare(i, lim), direction=LT
    }
    ENTRY main {
      p = s32[] parameter(0)
      i0 = s32[] constant(0)
      init = (s32[], s32[]) tuple(i0, p)
      w = (s32[], s32[]) while(init), condition=Cond, body=Body
      gte_pred = s32[] get-tuple-element(w), index=0, control-predecessors={p}
      gte_succ = s32[] get-tuple-element(w), index=0
      sink = s32[] add(p, i0), control-predecessors={gte_succ}
      gte_clean = s32[]{:S(1)} get-tuple-element(w), index=0,
        sharding={replicated}, metadata={op_name="indvar_meta"},
        frontend_attributes={keep_attr="1", _xla_compute_type="host"}
      sum1 = s32[] add(gte_pred, gte_succ)
      sum2 = s32[] add(sum1, sink)
      ROOT out = s32[] add(sum2, gte_clean)
    })hlo";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);

  HloComputation* entry = m->entry_computation();
  EXPECT_NE(entry->GetInstructionWithName("gte_pred"), nullptr);
  EXPECT_NE(entry->GetInstructionWithName("gte_succ"), nullptr);
  EXPECT_EQ(entry->GetInstructionWithName("gte_clean"), nullptr);

  const HloInstruction* forwarded = entry->root_instruction()->operand(1);
  ASSERT_EQ(forwarded->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(forwarded->literal().Get<int32_t>({}), 6);
  ASSERT_TRUE(forwarded->shape().has_layout());
  EXPECT_EQ(forwarded->shape().layout().memory_space(), 1);
  ASSERT_TRUE(forwarded->has_sharding());
  EXPECT_EQ(forwarded->sharding(), HloSharding::Replicate());
  EXPECT_EQ(forwarded->metadata().op_name(), "indvar_meta");
  EXPECT_EQ(forwarded->get_frontend_attribute("keep_attr"), "1");
  EXPECT_FALSE(
      forwarded->get_frontend_attribute("_xla_compute_type").has_value());
}

TEST_F(TripCountAnnotatorTest, SecondRunIsIdempotent) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule idempotent
    Body {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      v = s32[] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_next = s32[] add(i, one)
      ROOT tuple = (s32[], s32[]) tuple(i_next, v)
    }
    Cond {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      lim = s32[] constant(4)
      ROOT done = pred[] compare(i, lim), direction=LT
    }
    ENTRY main {
      p = s32[] parameter(0)
      i0 = s32[] constant(0)
      init = (s32[], s32[]) tuple(i0, p)
      w = (s32[], s32[]) while(init), condition=Cond, body=Body
      gte_fwd = s32[] get-tuple-element(w), index=0
      gte_dead = s32[] get-tuple-element(w), index=0
      gte_ctrl = s32[] get-tuple-element(w), index=0, control-predecessors={p}
      ROOT out = s32[] add(gte_fwd, gte_ctrl)
    })hlo";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed1, RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed1);
  const int64_t count_after_first = m->entry_computation()->instruction_count();

  ASSERT_OK_AND_ASSIGN(bool changed2, RunHloPass(&pass, m.get()));
  EXPECT_FALSE(changed2);
  EXPECT_EQ(m->entry_computation()->instruction_count(), count_after_first);

  // Also check pure-annotation loop with no GTE users.
  constexpr absl::string_view kPureHlo = R"hlo(
    HloModule idempotent_pure
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      ROOT tuple = (s32[]) tuple(s32[] add(i, one))
    }
    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      lim = s32[] constant(8)
      ROOT done = pred[] compare(i, lim), direction=LT
    }
    ENTRY main {
      init = (s32[]) tuple(s32[] constant(0))
      ROOT w = (s32[]) while(init), condition=Cond, body=Body
    })hlo";
  ASSERT_OK_AND_ASSIGN(auto m2, ParseAndReturnVerifiedModule(kPureHlo));
  ASSERT_OK_AND_ASSIGN(bool pure_changed1, RunHloPass(&pass, m2.get()));
  EXPECT_TRUE(pure_changed1);
  ASSERT_OK_AND_ASSIGN(bool pure_changed2, RunHloPass(&pass, m2.get()));
  EXPECT_FALSE(pure_changed2);
}

}  // namespace
}  // namespace xla
