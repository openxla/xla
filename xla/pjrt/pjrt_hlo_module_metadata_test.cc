/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/pjrt/pjrt_hlo_module_metadata.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/service/hlo_parser.h"

namespace xla {
namespace {
constexpr absl::string_view kTestHloModule = R"(
HloModule IrToHlo.40, entry_computation_layout={(f32[4,2]{0,1})->(s32[<=8,2]{1,0})}

%add_S32.12 (lhs.13: s32[], rhs.14: s32[]) -> s32[] {
  %lhs.13 = s32[] parameter(0)
  %rhs.14 = s32[] parameter(1)
  ROOT %add.15 = s32[] add(s32[] %lhs.13, s32[] %rhs.14)
}

%compare-greater-than.21 (p.0.lhs.22: s32[], p.0.rhs.23: s32[], p.1.lhs.24: s32[], p.1.rhs.25: s32[], p.2.lhs.26: s32[], p.2.rhs.27: s32[]) -> pred[] {
  %p.1.lhs.24 = s32[] parameter(2)
  %p.1.rhs.25 = s32[] parameter(3)
  %p.2.lhs.26 = s32[] parameter(4)
  %p.2.rhs.27 = s32[] parameter(5)
  %constant.28 = pred[] constant(true)
  %broadcast.29 = pred[] broadcast(pred[] %constant.28), dimensions={}
  %p.0.lhs.22 = s32[] parameter(0)
  %p.0.rhs.23 = s32[] parameter(1)
  %compare.30 = pred[] compare(s32[] %p.0.lhs.22, s32[] %p.0.rhs.23), direction=GT
  ROOT %select.31 = pred[] select(pred[] %broadcast.29, pred[] %compare.30, pred[] %broadcast.29)
}

ENTRY %IrToHlo.40 (p0.1: f32[4,2]) -> (s32[<=8,2]) {
  %p0.1 = f32[4,2]{0,1} parameter(0)
  %constant.2 = f32[] constant(0)
  %broadcast.3 = f32[4,2]{1,0} broadcast(f32[] %constant.2), dimensions={}
  %compare.4 = pred[4,2]{1,0} compare(f32[4,2]{0,1} %p0.1, f32[4,2]{1,0} %broadcast.3), direction=NE
  %reshape.5 = pred[8]{0} reshape(pred[4,2]{1,0} %compare.4)
  %convert.6 = s32[8]{0} convert(pred[8]{0} %reshape.5)
  %iota.17 = s32[4,2]{1,0} iota(), iota_dimension=0
  %reshape.18 = s32[8]{0} reshape(s32[4,2]{1,0} %iota.17)
  %iota.19 = s32[4,2]{1,0} iota(), iota_dimension=1
  %reshape.20 = s32[8]{0} reshape(s32[4,2]{1,0} %iota.19)
  %sort.32 = (s32[8]{0}, s32[8]{0}, s32[8]{0}) sort(s32[8]{0} %convert.6, s32[8]{0} %reshape.18, s32[8]{0} %reshape.20), dimensions={0}, is_stable=true, to_apply=%compare-greater-than.21
  %get-tuple-element.33 = s32[8]{0} get-tuple-element((s32[8]{0}, s32[8]{0}, s32[8]{0}) %sort.32), index=1
  %reshape.34 = s32[8,1]{1,0} reshape(s32[8]{0} %get-tuple-element.33)
  %get-tuple-element.35 = s32[8]{0} get-tuple-element((s32[8]{0}, s32[8]{0}, s32[8]{0}) %sort.32), index=2
  %reshape.36 = s32[8,1]{1,0} reshape(s32[8]{0} %get-tuple-element.35)
  %concatenate.37 = s32[8,2]{1,0} concatenate(s32[8,1]{1,0} %reshape.34, s32[8,1]{1,0} %reshape.36), dimensions={1}
  %constant.7 = s32[] constant(0)
  %broadcast.8 = s32[8]{0} broadcast(s32[] %constant.7), dimensions={}
  %compare.9 = pred[8]{0} compare(s32[8]{0} %convert.6, s32[8]{0} %broadcast.8), direction=GT
  %convert.10 = s32[8]{0} convert(pred[8]{0} %compare.9)
  %constant.11 = s32[] constant(0)
  %reduce.16 = s32[] reduce(s32[8]{0} %convert.10, s32[] %constant.11), dimensions={0}, to_apply=%add_S32.12
  %set-dimension-size.38 = s32[<=8,2]{1,0} set-dimension-size(s32[8,2]{1,0} %concatenate.37, s32[] %reduce.16), dimensions={0}
  ROOT %tuple.39 = (s32[<=8,2]{1,0}) tuple(s32[<=8,2]{1,0} %set-dimension-size.38)
}
)";

TEST(PjrtHloModuleMetadataTest, CreateFromHloModule) {
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       xla::ParseAndReturnUnverifiedModule(kTestHloModule, {}));
  ASSERT_OK_AND_ASSIGN(auto pjrt_hlo_module_metadata,
                       PjrtHloModuleMetadata::CreateFromHloModule(*hlo_module));
  EXPECT_EQ(pjrt_hlo_module_metadata->name(), "IrToHlo.40");
  EXPECT_EQ(pjrt_hlo_module_metadata->num_replicas(), 1);
  EXPECT_EQ(pjrt_hlo_module_metadata->num_partitions(), 1);
  EXPECT_EQ(pjrt_hlo_module_metadata->num_parameters(), 1);
  EXPECT_EQ(pjrt_hlo_module_metadata->launch_id(), 0);
  EXPECT_EQ(pjrt_hlo_module_metadata->hlo_first_parameter_instruction_shape()
                .ToString(),
            "f32[4,2]");
  EXPECT_EQ(pjrt_hlo_module_metadata->result_shape().ToString(),
            "(s32[<=8,2])");
}

}  // namespace

}  // namespace xla
