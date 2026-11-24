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

#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout_util.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;

using SubByteNormalizationTest = HloHardwareIndependentTestBase;

TEST_F(SubByteNormalizationTest,
       SetElementSizePacksPlainSubByteAndSkipsBitPlanes) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, entry_computation_layout={(u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))}, u4[16,128]{1,0})->(u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))}, u4[16,128]{1,0})}

ENTRY main {
  p0 = u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))} parameter(0)
  p1 = u4[16,128]{1,0} parameter(1)
  ROOT out = (u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))}, u4[16,128]{1,0}) tuple(p0, p1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(SubByteNormalization(SubByteNormalization::SET_ELEMENT_SIZE)
                  .Run(module.get()),
              IsOkAndHolds(true));

  const HloInstruction* p0 =
      module->entry_computation()->parameter_instruction(0);
  const HloInstruction* p1 =
      module->entry_computation()->parameter_instruction(1);
  EXPECT_EQ(p0->shape().layout().element_size_in_bits(), 0);
  EXPECT_EQ(p0->shape().layout().bit_planes().size(), 2);
  EXPECT_THAT(LayoutUtil::ValidateLayoutInShape(p0->shape()), IsOk());

  EXPECT_EQ(p1->shape().layout().element_size_in_bits(), 4);
  EXPECT_TRUE(p1->shape().layout().bit_planes().empty());

  EXPECT_EQ(module->entry_computation_layout()
                .parameter_layout(0)
                .shape()
                .layout()
                .element_size_in_bits(),
            0);
  EXPECT_EQ(module->entry_computation_layout()
                .parameter_layout(1)
                .shape()
                .layout()
                .element_size_in_bits(),
            4);
  EXPECT_THAT(LayoutUtil::ValidateLayoutInShape(
                  module->entry_computation_layout().result_layout().shape()),
              IsOk());

  // Running again on a module with only bit-plane sub-byte layouts is a no-op.
  constexpr absl::string_view kBitPlaneOnlyHlo = R"hlo(
HloModule bp_only, entry_computation_layout={(u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))})->u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))}}

ENTRY main {
  ROOT p0 = u4[16,128]{1,0:T(16,128)BP(2:(16,1))(2:(16,1))} parameter(0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> bp_module,
                       ParseAndReturnVerifiedModule(kBitPlaneOnlyHlo));
  EXPECT_THAT(SubByteNormalization(SubByteNormalization::SET_ELEMENT_SIZE)
                  .Run(bp_module.get()),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
