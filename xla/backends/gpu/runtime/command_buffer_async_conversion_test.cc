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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointee;

MATCHER_P(ThunkKindIs, kind, "") { return arg.kind() == kind; }

template <typename... Kinds>
auto ThunkKindsAre(Kinds... kinds) {
  return ElementsAre(Pointee(ThunkKindIs(kinds))...);
}

class RejectingAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    return absl::InternalError("Unexpected allocation during conversion");
  }
};

class CommandBufferAsyncConversionTest : public testing::Test {
 protected:
  Thunk::ThunkInfo Info() {
    Thunk::ThunkInfo info;
    info.thunk_id = ids_.GetNextThunkId();
    return info;
  }

  void AddCommand(ThunkSequence& thunks) {
    thunks.Emplace<ReplicaIdThunk>(Info(), slice_);
  }

  AsyncStartThunk* Start(ThunkSequence& thunks, bool convertible = true) {
    ThunkSequence body;
    if (convertible) {
      AddCommand(body);
    } else {
      // Memset thunks are not eligible for conversion in this pass.
      body.Emplace<Memset32BitValueThunk>(Info(), 0, slice_);
    }
    auto start = std::make_unique<AsyncStartThunk>(
        Info(), ComputationStreamId(0), std::move(body));
    auto* result = start.get();
    thunks.push_back(std::move(start));
    return result;
  }

  void Done(ThunkSequence& thunks, AsyncStartThunk* start) {
    thunks.Emplace<AsyncDoneThunk>(Info(), start->async_execution());
  }

  absl::StatusOr<bool> Convert(ThunkSequence& thunks) {
    DebugOptions options;
    options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
    options.set_xla_gpu_graph_min_graph_size(1);
    options.set_xla_gpu_command_buffer_scheduling_mode(DebugOptions::LHS);
    se::DeviceDescription device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
    RejectingAllocator allocator;
    return CommandBufferConversionPass("test").Run(
        &thunks, options, /*hlo_module=*/nullptr, device, allocator);
  }

  BufferAllocation allocation_{0, sizeof(int32_t), 0};
  BufferAllocation::Slice slice_{&allocation_, 0, sizeof(int32_t)};
  ThunkIdGenerator ids_;
};

TEST_F(CommandBufferAsyncConversionTest, KeepsUnsupportedRegionsIntact) {
  for (bool crossed : {false, true}) {
    SCOPED_TRACE(crossed);
    ThunkSequence thunks;
    AddCommand(thunks);
    auto* outer = Start(thunks, /*convertible=*/false);
    auto* inner = Start(thunks);
    Done(thunks, crossed ? outer : inner);
    Done(thunks, crossed ? inner : outer);
    AddCommand(thunks);

    ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
    EXPECT_TRUE(changed);
    // Capturing the inner start/done alone would drop its stream ordering
    // against the unsupported outer operation. Capture resumes after both
    // joins, and ordinary commands before the region can still be captured.
    EXPECT_THAT(thunks,
                ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kAsyncStart,
                              Thunk::kAsyncStart, Thunk::kAsyncDone,
                              Thunk::kAsyncDone, Thunk::kCommandBuffer));
  }
}

TEST_F(CommandBufferAsyncConversionTest, KeepsControlFlowInsideOpenRegion) {
  ThunkSequence thunks;
  auto* start = Start(thunks);
  ThunkSequence body;
  AddCommand(body);
  // WHILE conversion is disabled, but converting its body in isolation would
  // still be possible if the enclosing async boundary were discarded.
  auto loop = std::make_unique<WhileThunk>(
      Info(), BufferAllocation::Slice(&allocation_, 0, 1), ThunkSequence{},
      std::move(body), /*trip_count=*/1);
  auto* loop_ptr = loop.get();
  thunks.push_back(std::move(loop));
  Done(thunks, start);
  AddCommand(thunks);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kWhile,
                                    Thunk::kAsyncDone, Thunk::kCommandBuffer));
  EXPECT_THAT(loop_ptr->body_executor().thunks(),
              ThunkKindsAre(Thunk::kReplicaId));
}

TEST_F(CommandBufferAsyncConversionTest, KeepsTailAfterUnmatchedStart) {
  ThunkSequence thunks;
  AddCommand(thunks);
  Start(thunks);  // Its done can belong to a different pipelined computation.
  AddCommand(thunks);
  auto* inner = Start(thunks);
  Done(thunks, inner);
  AddCommand(thunks);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kAsyncStart,
                                    Thunk::kReplicaId, Thunk::kAsyncStart,
                                    Thunk::kAsyncDone, Thunk::kReplicaId));
}

TEST_F(CommandBufferAsyncConversionTest, MatchesCanonicalAsyncExecution) {
  ThunkSequence thunks;
  auto* canonical = Start(thunks);
  Done(thunks, canonical);
  ThunkSequence body;
  AddCommand(body);
  thunks.push_back(std::make_unique<AsyncStartThunk>(
      Info(), ComputationStreamId(0), std::move(body),
      canonical->async_execution()));
  Done(thunks, canonical);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));
}

TEST_F(CommandBufferAsyncConversionTest,
       DoesNotConvertOverlappingSharedExecution) {
  ThunkSequence thunks;
  auto* canonical = Start(thunks);
  ThunkSequence body;
  AddCommand(body);
  thunks.push_back(std::make_unique<AsyncStartThunk>(
      Info(), ComputationStreamId(0), std::move(body),
      canonical->async_execution()));
  Done(thunks, canonical);
  Done(thunks, canonical);
  AddCommand(thunks);

  // Runtime permits only one outstanding start per AsyncExecution. Do not
  // close and capture a region at the first done when a duplicate start exists.
  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_FALSE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart,
                                    Thunk::kAsyncDone, Thunk::kAsyncDone,
                                    Thunk::kReplicaId));
}

}  // namespace
}  // namespace xla::gpu
