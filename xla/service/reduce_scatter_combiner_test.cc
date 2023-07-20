/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/reduce_scatter_combiner.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

constexpr int64_t kMaxCombineCount = 256;
constexpr int64_t kMaxByteCount = 10 * 1024 * 1024;

class ReduceScatterCombinerTest : public HloTestBase,
                                  public ::testing::WithParamInterface<bool> {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, bool expect_change,
      int64_t byte_threshold = kMaxByteCount,
      int64_t count_threshold = kMaxCombineCount) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed = ReduceScatterCombiner(byte_threshold, count_threshold)
                       .Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t ReduceScatterCount(std::unique_ptr<HloModule>& module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<HloOpcode::kReduceScatter>);
  }

 protected:
  bool HasSchedule() const { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(Paramtests, ReduceScatterCombinerTest,
                         ::testing::Values(false, true));

TEST_P(ReduceScatterCombinerTest, Simple) {
  std::string hlo_string =
      absl::Substitute(R"(
HloModule m$0

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  ROOT t = (f32[4], f32[4]) tuple(rs0, rs1)
}
)",
                       HasSchedule() ? ", is_scheduled=true" : "");
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(ReduceScatterCount(module), 1);
}

TEST_P(ReduceScatterCombinerTest, SimpleMultipleGroups) {
  std::string hlo_string =
      absl::Substitute(R"(
HloModule m$0

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  p1 = f32[8, 8] parameter(1)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs1 = f32[4, 8] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs2 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1}, to_apply=sum
  rs3 = f32[8, 4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={1}, to_apply=sum
  ROOT t = (f32[4, 8], f32[4, 8], f32[8, 4], f32[8, 4]) tuple(rs0, rs1, rs2, rs3)
}
)",
                       HasSchedule() ? ", is_scheduled=true" : "");
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(ReduceScatterCount(module), 2);
}

// Test that dependent reduce-scatter do not get combined.
TEST_P(ReduceScatterCombinerTest, DependentReduceScatter) {
  std::string hlo_string =
      absl::Substitute(R"(
HloModule m$0

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs1 = f32[2, 8] reduce-scatter(rs0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  ROOT t = (f32[4, 8], f32[2, 8]) tuple(rs0, rs1)
}
)",
                       HasSchedule() ? ", is_scheduled=true" : "");
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_P(ReduceScatterCombinerTest, DoNotCombineMismatched) {
  std::string hlo_string =
      absl::Substitute(R"(
HloModule m$0

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), replica_groups={{1,0}}, dimensions={0}, to_apply=sum
  ROOT t = (f32[4], f32[4]) tuple(rs0, rs1)
}
)",
                       HasSchedule() ? ", is_scheduled=true" : "");
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_P(ReduceScatterCombinerTest, DoNotCombineWithoutReductionKind) {
  std::string hlo_string =
      absl::Substitute(R"(
HloModule TestModule$0

region_0 {
  Arg_1 = bf16[] parameter(1)
  Arg_0 = bf16[] parameter(0)
  convert_1 = f32[] convert(Arg_1)
  convert_0 = f32[] convert(Arg_0)
  add0 = f32[] add(convert_1, convert_0)
  ROOT convert_2 = bf16[] convert(add0)
}

region_1 {
  Arg_1 = bf16[] parameter(1)
  Arg_0 = bf16[] parameter(0)
  convert_1 = f32[] convert(Arg_1)
  convert_0 = f32[] convert(Arg_0)
  add0 = f32[] add(convert_1, convert_0)
  ROOT convert_2 = bf16[] convert(add0)
}

ENTRY entry{
 param0 = bf16[512,256]{1,0} parameter(0)
 param1 = bf16[512,256]{1,0} parameter(1)
 reduce-scatter.0 = bf16[512,256]{1,0} reduce-scatter(param0), replica_groups={{0}}, dimensions={0}, to_apply=region_0
 reduce-scatter.1 = bf16[512,256]{1,0} reduce-scatter(param1), replica_groups={{0}}, dimensions={0}, to_apply=region_1
 ROOT add.0 = tuple(reduce-scatter.0, reduce-scatter.1)
}
)",
                       HasSchedule() ? ", is_scheduled=true" : "");
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

}  // namespace
}  // namespace xla
