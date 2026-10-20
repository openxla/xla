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

#include "xla/service/gpu/scheduler_memory_fencing_fallback.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kHlo = R"(
HloModule memory_fencing

ENTRY main {
  p = f32[128]{0} parameter(0)
  a = f32[128]{0} negate(p)
  b = f32[128]{0} exponential(p)
  use_a = f32[1]{0} slice(a), slice={[0:1]}
  use_b = f32[1]{0} slice(b), slice={[0:1]}
  ROOT result = (f32[1]{0}, f32[1]{0}) tuple(use_a, use_b)
}
)";

class SchedulerMemoryFencingFallbackTest
    : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> MakeModule() {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
    DebugOptions& options = module->mutable_config().mutable_debug_options();
    options.set_xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes(
        0);
    options.set_xla_gpu_enable_latency_hiding_scheduler(true);
    return module;
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> Prepare(HloModule* module,
                                                            bool overlap) {
    ++preparations_;
    HloInstruction* p = FindInstruction(module, "p");
    HloInstruction* a = FindInstruction(module, "a");
    HloInstruction* b = FindInstruction(module, "b");
    HloInstruction* use_a = FindInstruction(module, "use_a");
    HloInstruction* use_b = FindInstruction(module, "use_b");
    HloInstruction* result = FindInstruction(module, "result");
    if (module->config()
                .debug_options()
                .xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes() >=
            0 &&
        a->control_successors().empty()) {
      ABSL_RETURN_IF_ERROR(a->AddControlDependencyTo(b));
    }
    HloSchedule schedule(module);
    schedule.set_sequence(
        module->entry_computation(),
        overlap ? HloInstructionSequence({p, a, b, use_a, use_b, result})
                : HloInstructionSequence({p, a, use_a, b, use_b, result}));
    ABSL_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
    return BufferAssigner::Run(
        module, std::make_unique<SequentialHloOrdering>(module->schedule()),
        [](const BufferValue& value) {
          return ShapeUtil::ByteSizeOf(value.shape(), /*pointer_size=*/8);
        },
        &alias_info_, [](LogicalBuffer::Color) { return 256; });
  }

  AliasInfo alias_info_;
  int preparations_ = 0;
};

TEST_F(SchedulerMemoryFencingFallbackTest, FallsBackOnLargerTemporaryHeap) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  ASSERT_OK_AND_ASSIGN(
      BufferAssignedModule result,
      ScheduleWithMemoryFencingFallback(std::move(module), [&](HloModule*
                                                                   candidate) {
        return Prepare(
            candidate,
            candidate->config()
                    .debug_options()
                    .xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes() >=
                0);
      }));
  EXPECT_EQ(preparations_, 2);
  EXPECT_EQ(&result.buffer_assignment->module(), result.module.get());
  EXPECT_TRUE(FindInstruction(result.module.get(), "b")
                  ->control_predecessors()
                  .empty());
  EXPECT_EQ(
      result.buffer_assignment->GetStats().preallocated_temp_allocation_bytes,
      512);
  EXPECT_EQ(
      result.module->config()
          .debug_options()
          .xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes(),
      0);
}

TEST_F(SchedulerMemoryFencingFallbackTest, KeepsSmallerFencedHeap) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  HloModule* original = module.get();
  ASSERT_OK_AND_ASSIGN(
      BufferAssignedModule result,
      ScheduleWithMemoryFencingFallback(std::move(module), [&](HloModule*
                                                                   candidate) {
        return Prepare(
            candidate,
            candidate->config()
                    .debug_options()
                    .xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes() <
                0);
      }));
  EXPECT_EQ(preparations_, 2);
  EXPECT_EQ(result.module.get(), original);
  EXPECT_EQ(&result.buffer_assignment->module(), original);
  EXPECT_EQ(
      result.buffer_assignment->GetStats().preallocated_temp_allocation_bytes,
      512);
}

TEST_F(SchedulerMemoryFencingFallbackTest, KeepsFencedHeapOnTie) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  HloModule* original = module.get();
  ASSERT_OK_AND_ASSIGN(BufferAssignedModule result,
                       ScheduleWithMemoryFencingFallback(
                           std::move(module), [&](HloModule* candidate) {
                             return Prepare(candidate, /*overlap=*/false);
                           }));
  EXPECT_EQ(preparations_, 2);
  EXPECT_EQ(result.module.get(), original);
  EXPECT_EQ(&result.buffer_assignment->module(), original);
}

TEST_F(SchedulerMemoryFencingFallbackTest,
       ChecksFinalAllocationsNotLiveRanges) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  ASSERT_OK_AND_ASSIGN(
      BufferAssignedModule result,
      ScheduleWithMemoryFencingFallback(
          std::move(module),
          [&](HloModule* candidate)
              -> absl::StatusOr<std::unique_ptr<BufferAssignment>> {
            // Identical schedules/live ranges, but a larger final assignment
            // models packing overhead that a live-memory comparison misses.
            ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                                  Prepare(candidate, /*overlap=*/false));
            if (preparations_ == 1) {
              assignment->NewEmptyAllocation(256, LogicalBuffer::Color(0));
            }
            return assignment;
          }));
  EXPECT_TRUE(FindInstruction(result.module.get(), "b")
                  ->control_predecessors()
                  .empty());
  EXPECT_EQ(
      result.buffer_assignment->GetStats().preallocated_temp_allocation_bytes,
      512);
}

TEST_F(SchedulerMemoryFencingFallbackTest, DoesNotTradeMemoryBetweenColors) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  ASSERT_OK_AND_ASSIGN(
      BufferAssignedModule result,
      ScheduleWithMemoryFencingFallback(
          std::move(module),
          [&](HloModule* candidate)
              -> absl::StatusOr<std::unique_ptr<BufferAssignment>> {
            ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                                  Prepare(candidate, /*overlap=*/false));
            // Fencing reduces the combined size, but grows color 0.
            assignment->NewEmptyAllocation(
                preparations_ == 1 ? 256 : 512,
                LogicalBuffer::Color(preparations_ == 1 ? 0 : 1));
            return assignment;
          }));
  EXPECT_TRUE(FindInstruction(result.module.get(), "b")
                  ->control_predecessors()
                  .empty());
}

TEST_F(SchedulerMemoryFencingFallbackTest,
       RejectsLargerSingleAllocationWithEqualTotalBytes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  ASSERT_OK_AND_ASSIGN(
      BufferAssignedModule result,
      ScheduleWithMemoryFencingFallback(
          std::move(module),
          [&](HloModule* candidate)
              -> absl::StatusOr<std::unique_ptr<BufferAssignment>> {
            ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                                  Prepare(candidate, /*overlap=*/false));
            if (preparations_ == 1) {
              assignment->NewEmptyAllocation(1024, LogicalBuffer::Color(0));
            } else {
              assignment->NewEmptyAllocation(512, LogicalBuffer::Color(0));
              assignment->NewEmptyAllocation(512, LogicalBuffer::Color(0));
            }
            return assignment;
          }));
  EXPECT_TRUE(FindInstruction(result.module.get(), "b")
                  ->control_predecessors()
                  .empty());
}

TEST_F(SchedulerMemoryFencingFallbackTest, DisabledFencingPreparesOnlyOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes(-1);
  ASSERT_OK_AND_ASSIGN(BufferAssignedModule result,
                       ScheduleWithMemoryFencingFallback(
                           std::move(module), [&](HloModule* candidate) {
                             return Prepare(candidate, /*overlap=*/false);
                           }));
  EXPECT_EQ(preparations_, 1);
}

TEST_F(SchedulerMemoryFencingFallbackTest, UnsetFencingPreparesOnlyOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes();
  ASSERT_OK_AND_ASSIGN(BufferAssignedModule result,
                       ScheduleWithMemoryFencingFallback(
                           std::move(module), [&](HloModule* candidate) {
                             return Prepare(candidate, /*overlap=*/false);
                           }));
  EXPECT_EQ(preparations_, 1);
}

TEST_F(SchedulerMemoryFencingFallbackTest, DisabledLhsPreparesOnlyOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_latency_hiding_scheduler(false);
  ASSERT_OK_AND_ASSIGN(BufferAssignedModule result,
                       ScheduleWithMemoryFencingFallback(
                           std::move(module), [&](HloModule* candidate) {
                             return Prepare(candidate, /*overlap=*/false);
                           }));
  EXPECT_EQ(preparations_, 1);
}

TEST_F(SchedulerMemoryFencingFallbackTest, ScheduledInputPreparesOnlyOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> initial_assignment,
                       Prepare(module.get(), /*overlap=*/false));
  initial_assignment.reset();
  preparations_ = 0;
  ASSERT_OK_AND_ASSIGN(BufferAssignedModule result,
                       ScheduleWithMemoryFencingFallback(
                           std::move(module), [&](HloModule* candidate) {
                             return Prepare(candidate, /*overlap=*/false);
                           }));
  EXPECT_EQ(preparations_, 1);
}

TEST_F(SchedulerMemoryFencingFallbackTest, PropagatesFencedPreparationError) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  EXPECT_THAT(
      ScheduleWithMemoryFencingFallback(
          std::move(module),
          [](HloModule*) -> absl::StatusOr<std::unique_ptr<BufferAssignment>> {
            return absl::InternalError("fenced preparation failed");
          }),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("fenced preparation failed")));
}

TEST_F(SchedulerMemoryFencingFallbackTest,
       KeepsFencedIfUnfencedPreparationFails) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, MakeModule());
  HloModule* original = module.get();
  ASSERT_OK_AND_ASSIGN(
      BufferAssignedModule result,
      ScheduleWithMemoryFencingFallback(
          std::move(module),
          [&](HloModule* candidate)
              -> absl::StatusOr<std::unique_ptr<BufferAssignment>> {
            if (preparations_ != 0) {
              return absl::InternalError("unfenced preparation failed");
            }
            return Prepare(candidate, /*overlap=*/false);
          }));
  EXPECT_EQ(result.module.get(), original);
  EXPECT_EQ(&result.buffer_assignment->module(), original);
}

}  // namespace
}  // namespace xla::gpu
