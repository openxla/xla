/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_emission_utils.h"

#include <gmock/gmock.h>

#include <cstdint>
#include <memory>

#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using IrEmitterTest = HloHardwareIndependentTestBase;

TEST_F(IrEmitterTest, ConvWithZeroSizedKernelNotImplementedAsEigen) {
  const char* const hlo_string = R"(
HloModule ModuleWithConv

ENTRY Conv {
  input = f32[32,50,28,28]{3,2,1,0} parameter(0)
  kernel = f32[50,0,5,5]{3,2,1,0} parameter(1)
  ROOT convolution = f32[32,0,24,24]{3,2,1,0} convolution(input, kernel),
    window={size=5x5},
    dim_labels=bf01_io01->bf01
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* entry_computation = module->entry_computation();

  HloInstruction* conv_instr = entry_computation->root_instruction();
  cpu::TargetMachineFeaturesStub target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  EXPECT_FALSE(cpu::PotentiallyImplementedAsEigenConvolution(
      *conv_instr, target_machine_features));
}

TEST_F(IrEmitterTest, CpuScatterPackedSubByteType) {
  for (PrimitiveType ty : {S2, U2, S4, U4, F4E2M1FN}) {
    EXPECT_TRUE(cpu::IsCpuScatterPackedSubByteType(ty)) << ty;
  }
  for (PrimitiveType ty : {S1, U1, PRED, S8, U8, F16, BF16, F32, C128}) {
    EXPECT_FALSE(cpu::IsCpuScatterPackedSubByteType(ty)) << ty;
  }
}

TEST_F(IrEmitterTest, CpuScatterOverwriteReducer) {
  auto is_overwrite = [&](absl::string_view hlo) -> absl::StatusOr<bool> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    return cpu::IsCpuScatterOverwriteReducer(*module->entry_computation());
  };

  EXPECT_THAT(is_overwrite(R"hlo(
    HloModule overwrite
    ENTRY main {
      lhs = f32[] parameter(0)
      ROOT rhs = f32[] parameter(1)
    }
  )hlo"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(is_overwrite(R"hlo(
    HloModule dead_lhs_user
    ENTRY main {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      dead = f32[] add(lhs, rhs)
      ROOT out = f32[] copy(rhs)
    }
  )hlo"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(is_overwrite(R"hlo(
    HloModule keep_lhs
    ENTRY main {
      ROOT lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
    }
  )hlo"),
              absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(is_overwrite(R"hlo(
    HloModule add_reducer
    ENTRY main {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
  )hlo"),
              absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(is_overwrite(R"hlo(
    HloModule unary_reducer
    ENTRY main {
      ROOT lhs = f32[] parameter(0)
    }
  )hlo"),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(IrEmitterTest, CpuScatterDirectAtomicRmwReducer) {
  auto is_direct = [&](absl::string_view hlo,
                       PrimitiveType elem_type) -> absl::StatusOr<bool> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    return cpu::IsCpuScatterDirectAtomicRmwReducer(*module->entry_computation(),
                                                   elem_type);
  };

  // Byte addressable integral ops in forward and reverse operand order.
  for (absl::string_view hlo : {
           R"hlo(
             HloModule add_fwd
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] add(lhs, rhs)
             }
           )hlo",
           R"hlo(
             HloModule add_rev
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] add(rhs, lhs)
             }
           )hlo",
           R"hlo(
             HloModule and_rev
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] and(rhs, lhs)
             }
           )hlo",
           R"hlo(
             HloModule or_fwd
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] or(lhs, rhs)
             }
           )hlo",
           R"hlo(
             HloModule xor_rev
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] xor(rhs, lhs)
             }
           )hlo",
           R"hlo(
             HloModule sub_fwd
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] subtract(lhs, rhs)
             }
           )hlo",
       }) {
    EXPECT_THAT(is_direct(hlo, S8), absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(is_direct(hlo, U16), absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(is_direct(hlo, S32), absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(is_direct(hlo, U64), absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(is_direct(hlo, F32), absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(is_direct(hlo, BF16), absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(is_direct(hlo, C64), absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(is_direct(hlo, S4), absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(is_direct(hlo, U2), absl_testing::IsOkAndHolds(false));
  }

  // Non commutative subtract in reverse order, multiply, maximum, duplicate
  // operand, multi instruction body, and wrong arity fall back to CAS.
  for (absl::string_view hlo : {
           R"hlo(
             HloModule sub_rev
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] subtract(rhs, lhs)
             }
           )hlo",
           R"hlo(
             HloModule mul_fwd
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] multiply(lhs, rhs)
             }
           )hlo",
           R"hlo(
             HloModule max_fwd
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] maximum(lhs, rhs)
             }
           )hlo",
           R"hlo(
             HloModule dup_operand
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               ROOT out = s32[] add(lhs, lhs)
             }
           )hlo",
           R"hlo(
             HloModule multi_instr
             ENTRY main {
               lhs = s32[] parameter(0)
               rhs = s32[] parameter(1)
               tmp = s32[] add(lhs, rhs)
               ROOT out = s32[] add(tmp, rhs)
             }
           )hlo",
           R"hlo(
             HloModule unary
             ENTRY main {
               lhs = s32[] parameter(0)
               ROOT out = s32[] negate(lhs)
             }
           )hlo",
       }) {
    EXPECT_THAT(is_direct(hlo, S32), absl_testing::IsOkAndHolds(false));
  }

  // PRED supports only bitwise and, or, and xor.
  for (absl::string_view hlo : {
           R"hlo(
             HloModule pred_and
             ENTRY main {
               lhs = pred[] parameter(0)
               rhs = pred[] parameter(1)
               ROOT out = pred[] and(lhs, rhs)
             }
           )hlo",
           R"hlo(
             HloModule pred_or_rev
             ENTRY main {
               lhs = pred[] parameter(0)
               rhs = pred[] parameter(1)
               ROOT out = pred[] or(rhs, lhs)
             }
           )hlo",
           R"hlo(
             HloModule pred_xor
             ENTRY main {
               lhs = pred[] parameter(0)
               rhs = pred[] parameter(1)
               ROOT out = pred[] xor(lhs, rhs)
             }
           )hlo",
       }) {
    EXPECT_THAT(is_direct(hlo, PRED), absl_testing::IsOkAndHolds(true));
  }
  EXPECT_THAT(is_direct(R"hlo(
    HloModule pred_eq
    ENTRY main {
      lhs = pred[] parameter(0)
      rhs = pred[] parameter(1)
      ROOT out = pred[] compare(lhs, rhs), direction=EQ
    }
  )hlo",
                        PRED),
              absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(is_direct(R"hlo(
    HloModule pred_max
    ENTRY main {
      lhs = pred[] parameter(0)
      rhs = pred[] parameter(1)
      ROOT out = pred[] maximum(lhs, rhs)
    }
  )hlo",
                        PRED),
              absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
