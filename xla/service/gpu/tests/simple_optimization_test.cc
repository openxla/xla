/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/strings/string_view.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

class SimpleOptimizationTest : public HloTestBase {};

TEST_F(SimpleOptimizationTest, OptimizeModule) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
}

TEST_F(SimpleOptimizationTest, DegeneratedAllReduceRemoval) {
  constexpr absl::string_view kHloText = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

main {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  ROOT all-reduce = f32[8,16] all-reduce(p0),
    channel_id=1,
    use_global_device_ids=true,
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}},
    to_apply=sum
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(kHloText));
  EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
