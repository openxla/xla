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

#include "xla/hlo/pass/hlo_pass_interface.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

using HloComputationPassTest = HloHardwareIndependentTestBase;

// Records the name of every computation it is run on. Reports a change for the
// computation named `computation_to_change` and fails on the computation named
// `computation_to_fail`.
class RecordingComputationPass : public HloComputationPass {
 public:
  explicit RecordingComputationPass(std::string computation_to_change = "",
                                    std::string computation_to_fail = "")
      : computation_to_change_(std::move(computation_to_change)),
        computation_to_fail_(std::move(computation_to_fail)) {}

  absl::string_view name() const override { return "recording"; }

  const std::vector<std::string>& visited() const { return visited_; }

 protected:
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation) override {
    visited_.push_back(std::string(computation->name()));
    if (computation->name() == computation_to_fail_) {
      return Internal("Failed on %s", computation->name());
    }
    return computation->name() == computation_to_change_;
  }

 private:
  std::string computation_to_change_;
  std::string computation_to_fail_;
  std::vector<std::string> visited_;
};

constexpr absl::string_view kModuleWithFusionAndCall = R"(
HloModule m

%fused_computation (p: f32[2]) -> f32[2] {
  %p = f32[2] parameter(0)
  ROOT %neg = f32[2] negate(%p)
}

%callee (p: f32[2]) -> f32[2] {
  %p = f32[2] parameter(0)
  ROOT %fusion = f32[2] fusion(%p), kind=kLoop, calls=%fused_computation
}

ENTRY %main (p: f32[2]) -> f32[2] {
  %p = f32[2] parameter(0)
  ROOT %call = f32[2] call(%p), to_apply=%callee
}
)";

constexpr absl::string_view kModuleWithParallelThread = R"(
HloModule m

%async_builder {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  ROOT %foo = add(%p0, %p1)
}, execution_thread="parallel_thread"

ENTRY %main (p0: f32[10], p1: f32[10]) -> f32[10] {
  %p0 = f32[10] parameter(0)
  %p1 = f32[10] parameter(1)
  %async-start = ((f32[10], f32[10]), f32[10], s32[]) async-start(f32[10] %p0, f32[10] %p1), async_execution_thread="parallel_thread", calls=%async_builder
  ROOT %done = f32[10]{0} async-done(((f32[10], f32[10]), f32[10], s32[]) %async-start), async_execution_thread="parallel_thread", calls=%async_builder
}
)";

TEST_F(HloComputationPassTest, VisitsNonFusionComputationsInPostOrder) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleWithFusionAndCall));
  RecordingComputationPass pass;

  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
  EXPECT_THAT(pass.visited(), ElementsAre("callee", "main"));
}

TEST_F(HloComputationPassTest, ReportsChangeIfAnyComputationChanged) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleWithFusionAndCall));
  RecordingComputationPass pass(/*computation_to_change=*/"callee");

  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(pass.visited(), ElementsAre("callee", "main"));
}

TEST_F(HloComputationPassTest, StopsAndPropagatesError) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleWithFusionAndCall));
  RecordingComputationPass pass(/*computation_to_change=*/"",
                                /*computation_to_fail=*/"callee");

  EXPECT_THAT(pass.Run(module.get()), StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(pass.visited(), ElementsAre("callee"));
}

TEST_F(HloComputationPassTest, VisitsAllExecutionThreadsByDefault) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleWithParallelThread));
  RecordingComputationPass pass;

  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
  EXPECT_THAT(pass.visited(), ElementsAre("async_builder", "main"));
}

TEST_F(HloComputationPassTest, VisitsOnlyRequestedExecutionThreads) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleWithParallelThread));

  RecordingComputationPass parallel_pass;
  EXPECT_THAT(parallel_pass.Run(module.get(), {"parallel_thread"}),
              IsOkAndHolds(false));
  EXPECT_THAT(parallel_pass.visited(), ElementsAre("async_builder"));

  RecordingComputationPass main_pass;
  EXPECT_THAT(
      main_pass.Run(module.get(), {HloInstruction::kMainExecutionThread}),
      IsOkAndHolds(false));
  EXPECT_THAT(main_pass.visited(), ElementsAre("main"));

  RecordingComputationPass unknown_thread_pass;
  EXPECT_THAT(unknown_thread_pass.Run(module.get(), {"unknown_thread"}),
              IsOkAndHolds(false));
  EXPECT_THAT(unknown_thread_pass.visited(), IsEmpty());
}

}  // namespace
}  // namespace xla
