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

#include "xla/service/cpu/cpu_performance_model.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"

namespace xla::cpu {
namespace {

class CpuPerformanceModelTest : public HloHardwareIndependentTestBase {
 public:
  CpuPerformanceModelTest() { options_.count_multiple_input_accesses = true; }

  CpuPerformanceModel::RunTimes EstimateRunTimes(absl::string_view hlo_string,
                                                 absl::string_view producer) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    CpuHloCostAnalysis analysis(options_);
    CHECK_OK(module->entry_computation()->Accept(&analysis));
    const HloInstruction* instr = FindInstruction(module.get(), producer);
    std::vector<const HloInstruction*> consumers(instr->users().begin(),
                                                 instr->users().end());
    return model_.EstimateRunTimes(instr, &analysis, consumers);
  }

 protected:
  HloCostAnalysis::Options options_;
  CpuPerformanceModel model_{CpuPerformanceModel::DefaultDeviceInfo()};
};

TEST_F(CpuPerformanceModelTest, FusingCheapProducerIntoBroadcastIsFaster) {
  // The producer is read three times per element. Recomputing a sqrt is
  // cheaper than writing and reading back 4 MiB.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

fused {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024,3] parameter(1)
  b = f32[1024,1024,3] broadcast(p0), dimensions={0,1}
  d = f32[1024,1024,3] divide(p1, b)
  c = f32[] constant(0)
  ROOT r = f32[1024,3] reduce(d, c), dimensions={1}, to_apply=add
}

ENTRY e {
  x = f32[1024,1024] parameter(0)
  y = f32[1024,1024,3] parameter(1)
  sqrt = f32[1024,1024] sqrt(x)
  ROOT fusion = f32[1024,3] fusion(sqrt, y), kind=kLoop, calls=fused
})";
  CpuPerformanceModel::RunTimes run_times =
      EstimateRunTimes(hlo_string, "sqrt");
  EXPECT_LT(run_times.time_fused, run_times.time_unfused);
}

TEST_F(CpuPerformanceModelTest, FusingExpensiveProducerIntoHeavyReuseIsSlower) {
  // Fusing the exponential would recompute it 64 times per element.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

fused {
  p0 = f32[131072] parameter(0)
  b = f32[131072,64] broadcast(p0), dimensions={0}
  c = f32[] constant(0)
  ROOT r = f32[64] reduce(b, c), dimensions={0}, to_apply=add
}

ENTRY e {
  x = f32[131072] parameter(0)
  exp = f32[131072] exponential(x)
  ROOT fusion = f32[64] fusion(exp), kind=kLoop, calls=fused
})";
  CpuPerformanceModel::RunTimes run_times = EstimateRunTimes(hlo_string, "exp");
  EXPECT_GT(run_times.time_fused, run_times.time_unfused);
}

TEST_F(CpuPerformanceModelTest, CustomCallIsNeverFused) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY e {
  x = f32[1024] parameter(0)
  cc = f32[1024] custom-call(x), custom_call_target="foo"
  ROOT n = f32[1024] negate(cc)
})";
  CpuPerformanceModel::RunTimes run_times = EstimateRunTimes(hlo_string, "cc");
  EXPECT_EQ(run_times.time_fused, absl::InfiniteDuration());
}

}  // namespace
}  // namespace xla::cpu
