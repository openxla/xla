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

#include "xla/service/cpu/onednn_float_support.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class OneDnnFloatSupportTest : public HloHardwareIndependentTestBase {
 protected:
  // Mirrors the oneDNN pipeline: normalization leaves f32 -> bf16 -> f32 round
  // trips at the reducer boundary, which SimplifyFPConversions removes.
  bool Normalize(HloModule* module, bool simplify_conversions) {
    OneDnnFloatSupport float_support(
        BF16, [](const HloInstruction&) { return false; });
    FloatNormalization float_normalization(&float_support);
    absl::StatusOr<bool> changed = float_normalization.Run(module);
    EXPECT_TRUE(changed.ok()) << changed.status();
    if (simplify_conversions) {
      EXPECT_TRUE(SimplifyFPConversions().Run(module).ok());
    }
    return changed.value_or(false);
  }
};

// Reduce-window is upcast, so its max reducer must be upcast too. A bf16 max
// would keep f32 -> bf16 -> max -> f32 converts on every window element.
TEST_F(OneDnnFloatSupportTest, ReduceWindowMaxReducerUpcast) {
  constexpr absl::string_view kHlo = R"(
HloModule test_module

max {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] maximum(a, b)
}

ENTRY main {
  p0 = bf16[1,16,16,8] parameter(0)
  init = bf16[] constant(-inf)
  ROOT rw = bf16[1,16,16,8] reduce-window(p0, init), window={size=1x5x5x1 pad=0_0x2_2x2_2x0_0}, to_apply=max
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  Normalize(module.get(), /*simplify_conversions=*/true);

  HloInstruction* rw = FindInstruction(module.get(), HloOpcode::kReduceWindow);
  ASSERT_NE(rw, nullptr);
  EXPECT_EQ(rw->shape().element_type(), F32);
  const HloInstruction* root = rw->to_apply()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMaximum);
  EXPECT_EQ(root->shape().element_type(), F32);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->shape().element_type(), F32);
}

TEST_F(OneDnnFloatSupportTest, ReduceMaxAndMinReducersUpcast) {
  constexpr absl::string_view kHlo = R"(
HloModule test_module

max {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] maximum(a, b)
}

min {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] minimum(a, b)
}

ENTRY main {
  p0 = bf16[16,32] parameter(0)
  ninf = bf16[] constant(-inf)
  pinf = bf16[] constant(inf)
  rmax = bf16[16] reduce(p0, ninf), dimensions={1}, to_apply=max
  rmin = bf16[16] reduce(p0, pinf), dimensions={1}, to_apply=min
  ROOT t = (bf16[16], bf16[16]) tuple(rmax, rmin)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  Normalize(module.get(), /*simplify_conversions=*/true);

  int num_reduces = 0;
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kReduce) continue;
    ++num_reduces;
    EXPECT_EQ(instr->shape().element_type(), F32);
    const HloInstruction* root = instr->to_apply()->root_instruction();
    EXPECT_EQ(root->shape().element_type(), F32) << root->ToString();
    EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter)
        << root->ToString();
  }
  EXPECT_EQ(num_reduces, 2);
}

// Scatter supports bf16, so its max combiner stays in bf16.
TEST_F(OneDnnFloatSupportTest, ScatterMaxCombinerNotUpcast) {
  constexpr absl::string_view kHlo = R"(
HloModule test_module

max {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] maximum(a, b)
}

ENTRY main {
  operand = bf16[16,8] parameter(0)
  indices = s32[4] parameter(1)
  updates = bf16[4,8] parameter(2)
  ROOT scatter = bf16[16,8] scatter(operand, indices, updates),
      update_window_dims={1}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=max
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(Normalize(module.get(), /*simplify_conversions=*/false));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->shape().element_type(), BF16);
  EXPECT_EQ(root->to_apply()->root_instruction()->shape().element_type(),
            BF16);
}

// Top-level max and clamp stay bf16 so ReLU / ReLU6 can be matched as bf16
// post-ops.
TEST_F(OneDnnFloatSupportTest, TopLevelClampAndMaxNotUpcast) {
  constexpr absl::string_view kHlo = R"(
HloModule test_module

ENTRY main {
  p0 = bf16[100] parameter(0)
  c0 = bf16[] constant(0)
  c6 = bf16[] constant(6)
  zero = bf16[100] broadcast(c0), dimensions={}
  six = bf16[100] broadcast(c6), dimensions={}
  max = bf16[100] maximum(p0, zero)
  ROOT clamp = bf16[100] clamp(zero, max, six)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(Normalize(module.get(), /*simplify_conversions=*/false));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kClamp);
  EXPECT_EQ(root->shape().element_type(), BF16);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kMaximum);
  EXPECT_EQ(root->operand(1)->shape().element_type(), BF16);
}

}  // namespace
}  // namespace xla::cpu
