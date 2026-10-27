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

#include "xla/service/scheduler_memory_fencing_autotuner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/scheduler_memory_fencing.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
constexpr int64_t kMiB = 1024 * 1024;

class SchedulerMemoryFencingAutotunerTest
    : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> MakeModule(int count) {
    std::string hlo =
        "HloModule m, is_scheduled=true\nENTRY main {\n"
        "p = f32[4] parameter(0)\n";
    for (int i = 0; i < count; ++i) {
      absl::StrAppend(&hlo, "n", i, " = f32[4] negate(p)\n");
    }
    absl::StrAppend(&hlo,
                    "gate = f32[4] negate(p)\n"
                    "ROOT r = f32[4] add(p, gate)\n}");
    return ParseAndReturnVerifiedModule(hlo);
  }

  HloInstruction* Gate(HloModule* module) {
    return module->entry_computation()->GetInstructionWithName("gate");
  }

  std::vector<SchedulerMemoryFencing::BufferFences> Buffers(
      HloModule* module, const std::vector<int64_t>& sizes) {
    std::vector<SchedulerMemoryFencing::BufferFences> buffers;
    for (int i = 0; i < sizes.size(); ++i) {
      buffers.push_back({sizes[i],
                         {{module->entry_computation()->GetInstructionWithName(
                               absl::StrCat("n", i)),
                           Gate(module)}}});
    }
    return buffers;
  }
};

TEST_F(SchedulerMemoryFencingAutotunerTest,
       RefinesWinnerAndBoundsSchedulingWork) {
  ASSERT_OK_AND_ASSIGN(auto module, MakeModule(12));
  auto buffers =
      Buffers(module.get(), {2048 * kMiB, 1024 * kMiB, 768 * kMiB, 512 * kMiB,
                             384 * kMiB, 256 * kMiB, 192 * kMiB, 128 * kMiB,
                             96 * kMiB, 64 * kMiB, 32 * kMiB, 16 * kMiB});
  std::vector<int> trials;
  HloInstruction* original_gate = Gate(module.get());
  ASSERT_OK(AutotuneSchedulerMemoryFencing(
      module.get(), buffers, 200ULL * 1024 * kMiB,
      [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
        HloInstruction* gate = Gate(trial);
        EXPECT_TRUE(gate->metadata().op_name().empty());
        gate->mutable_metadata().set_op_name("scheduled");
        const int edges = gate->control_predecessors().size();
        trials.push_back(edges);
        EXPECT_OK(trial->schedule().Verify());
        // An intermediate threshold wins: more fencing is not always better.
        const int64_t peak = edges == 5   ? 20
                             : edges == 3 ? 50
                             : edges == 4 ? 100
                                          : 300;
        return SchedulerMemoryFencingCost{peak, edges == 5 ? 999.0 : 1.0};
      }));
  EXPECT_THAT(trials, ElementsAre(0, 1, 2, 4, 6, 8, 5, 3, 5));
  EXPECT_EQ(Gate(module.get()), original_gate);
  EXPECT_EQ(original_gate->control_predecessors().size(), 5);
  EXPECT_EQ(original_gate->metadata().op_name(), "scheduled");
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(SchedulerMemoryFencingAutotunerTest, BaselineWinsMemoryAndExactTies) {
  for (int64_t fenced_peak : {100, 101}) {
    ASSERT_OK_AND_ASSIGN(auto module, MakeModule(1));
    auto buffers = Buffers(module.get(), {512 * kMiB});
    std::vector<int> trials;
    ASSERT_OK(AutotuneSchedulerMemoryFencing(
        module.get(), buffers, 1024 * kMiB,
        [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
          int edges = Gate(trial)->control_predecessors().size();
          trials.push_back(edges);
          return SchedulerMemoryFencingCost{edges ? fenced_peak : 100, 10};
        }));
    EXPECT_THAT(trials, ElementsAre(0, 1, 0));
    EXPECT_THAT(Gate(module.get())->control_predecessors(), IsEmpty());
  }
}

TEST_F(SchedulerMemoryFencingAutotunerTest, LatencyBreaksMemoryTie) {
  ASSERT_OK_AND_ASSIGN(auto module, MakeModule(1));
  auto buffers = Buffers(module.get(), {512 * kMiB});
  ASSERT_OK(AutotuneSchedulerMemoryFencing(
      module.get(), buffers, 1024 * kMiB,
      [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
        return SchedulerMemoryFencingCost{
            100, Gate(trial)->control_predecessors().empty() ? 20.0 : 10.0};
      }));
  EXPECT_EQ(Gate(module.get())->control_predecessors().size(), 1);
}

TEST_F(SchedulerMemoryFencingAutotunerTest, ClampsAndDeduplicatesFenceSets) {
  ASSERT_OK_AND_ASSIGN(auto module, MakeModule(1));
  auto buffers = Buffers(module.get(), {512 * kMiB});
  // A second buffer contributes the same edge; no extra scheduling trial.
  buffers.push_back({256 * kMiB, buffers.front().edges});
  int calls = 0;
  ASSERT_OK(AutotuneSchedulerMemoryFencing(
      module.get(), buffers, 1,
      [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
        ++calls;
        return SchedulerMemoryFencingCost{
            Gate(trial)->control_predecessors().empty() ? 200 : 100, 1};
      }));
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(Gate(module.get())->control_predecessors().size(), 1);
}

TEST_F(SchedulerMemoryFencingAutotunerTest,
       RefinementFindsBuffersBelowAllAnchors) {
  ASSERT_OK_AND_ASSIGN(auto module, MakeModule(1));
  auto buffers = Buffers(module.get(), {4 * kMiB});
  std::vector<int> trials;
  ASSERT_OK(AutotuneSchedulerMemoryFencing(
      module.get(), buffers, 100ULL * 1024 * kMiB,
      [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
        int edges = Gate(trial)->control_predecessors().size();
        trials.push_back(edges);
        return SchedulerMemoryFencingCost{edges ? 100 : 200, 1};
      }));
  EXPECT_THAT(trials, ElementsAre(0, 1, 1));
}

TEST_F(SchedulerMemoryFencingAutotunerTest, EmptyPlanAndZeroBudgetRunOnlyOnce) {
  for (bool zero_budget : {false, true}) {
    ASSERT_OK_AND_ASSIGN(auto module, MakeModule(1));
    auto buffers = Buffers(module.get(), {512 * kMiB});
    if (!zero_budget) buffers.clear();
    int calls = 0;
    ASSERT_OK(AutotuneSchedulerMemoryFencing(
        module.get(), buffers, zero_budget ? 0 : 1024 * kMiB,
        [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
          EXPECT_EQ(trial, module.get());
          ++calls;
          return SchedulerMemoryFencingCost{100, 1};
        }));
    EXPECT_EQ(calls, 1);
  }
}

TEST_F(SchedulerMemoryFencingAutotunerTest,
       FailedTrialLeavesOriginalUntouched) {
  ASSERT_OK_AND_ASSIGN(auto module, MakeModule(1));
  auto buffers = Buffers(module.get(), {512 * kMiB});
  const std::string original = module->ToString();
  int calls = 0;
  absl::Status status = AutotuneSchedulerMemoryFencing(
      module.get(), buffers, 1024 * kMiB,
      [&](HloModule* trial) -> absl::StatusOr<SchedulerMemoryFencingCost> {
        ++calls;
        Gate(trial)->mutable_metadata().set_op_name("trial mutation");
        if (calls == 2) return absl::InternalError("test scheduling failure");
        return SchedulerMemoryFencingCost{100, 1};
      });
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("test scheduling failure"));
  EXPECT_EQ(module->ToString(), original);
  EXPECT_EQ(calls, 2);
}

}  // namespace
}  // namespace xla
