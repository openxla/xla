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

#include "xla/backends/gpu/runtime/async_thunk_passes.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/debug_options_flags.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class FakeThunkPassBufferAllocator : public ThunkPassBufferAllocator {
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) final {
    return absl::UnimplementedError("NewEmptyAllocation is not implemented.");
  }
};

// A thunk with configurable buffer use for testing conflict detection.
class FakeThunk : public Thunk {
 public:
  explicit FakeThunk(std::optional<BufferUse> buffer_use = std::nullopt)
      : Thunk(Thunk::kKernel, Thunk::ThunkInfo()),
        buffer_use_(std::move(buffer_use)) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override {
    if (buffer_use_.has_value()) {
      return {*buffer_use_};
    }
    return {};
  }

 private:
  std::optional<BufferUse> buffer_use_;
};

// Creates a FakeThunk with an optional buffer use.
std::unique_ptr<FakeThunk> MakeFakeThunk(
    std::optional<BufferUse> buffer_use = std::nullopt) {
  if (buffer_use.has_value()) {
    return std::make_unique<FakeThunk>(std::move(*buffer_use));
  }
  return std::make_unique<FakeThunk>();
}

// Creates an AsyncStartThunk wrapping a single FakeThunk with an optional
// buffer use, so that the start thunk transitively reports it via Walk.
std::unique_ptr<AsyncStartThunk> MakeAsyncStart() {
  ThunkSequence nested;
  nested.push_back(MakeFakeThunk());
  return std::make_unique<AsyncStartThunk>(Thunk::ThunkInfo(),
                                           AsyncStartThunk::AsyncKind::kCompute,
                                           std::move(nested));
}

std::unique_ptr<AsyncStartThunk> MakeAsyncStart(BufferUse buffer_use) {
  ThunkSequence nested;
  nested.push_back(MakeFakeThunk(std::move(buffer_use)));
  return std::make_unique<AsyncStartThunk>(Thunk::ThunkInfo(),
                                           AsyncStartThunk::AsyncKind::kCompute,
                                           std::move(nested));
}

// Creates an AsyncDoneThunk matching the given start thunk.
std::unique_ptr<AsyncDoneThunk> MakeAsyncDone(AsyncStartThunk* start) {
  return std::make_unique<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                          start->async_execution());
}

//===----------------------------------------------------------------------===//
// RemoveRedundantAsyncThunkPass tests.
//===----------------------------------------------------------------------===//

TEST(RemoveRedundantAsyncThunkPassTest, InlinesAdjacentStartDone) {
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(std::move(start));
  thunks.push_back(std::move(done));

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  RemoveRedundantAsyncThunkPass pass;

  ASSERT_OK_AND_ASSIGN(bool changed,
                       pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                                device_info, allocator));
  EXPECT_TRUE(changed);
  // The start/done pair should be replaced by the single nested thunk.
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kKernel);
}

TEST(RemoveRedundantAsyncThunkPassTest, PreservesNonAdjacentStartDone) {
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(std::move(start));
  thunks.push_back(MakeFakeThunk());
  thunks.push_back(std::move(done));

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  RemoveRedundantAsyncThunkPass pass;

  ASSERT_OK_AND_ASSIGN(bool changed,
                       pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                                device_info, allocator));
  EXPECT_FALSE(changed);
  EXPECT_EQ(thunks.size(), 3);
}

//===----------------------------------------------------------------------===//
// ExpandAsyncScopeThunkPass tests.
//===----------------------------------------------------------------------===//

// Helper to run the expand pass and return whether it changed the sequence.
absl::StatusOr<bool> RunExpandPass(ThunkSequence* thunks) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  ExpandAsyncScopeThunkPass pass;
  return pass.Run(thunks, debug_options, /*hlo_module=*/nullptr, device_info,
                  allocator);
}

TEST(ExpandAsyncScopeThunkPassTest, MovesStartUpAndDoneDown) {
  // No buffer uses on any thunk => no conflicts => start moves to front,
  // done moves to back.
  //
  //   kernel0, kernel1, start, kernel2, done, kernel3
  //   => start, kernel0, kernel1, kernel2, kernel3, done
  auto start = MakeAsyncStart();
  auto* start_ptr = start.get();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(MakeFakeThunk());
  thunks.push_back(MakeFakeThunk());
  thunks.push_back(std::move(start));
  thunks.push_back(MakeFakeThunk());
  thunks.push_back(std::move(done));
  thunks.push_back(MakeFakeThunk());

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_TRUE(changed);
  ASSERT_EQ(thunks.size(), 6);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(thunks[0].get(), start_ptr);
  EXPECT_EQ(thunks[5]->kind(), Thunk::kAsyncDone);
}

TEST(ExpandAsyncScopeThunkPassTest, StartStopsAtConflictingThunk) {
  // start writes buffer 0; kernel0 also writes buffer 0 => conflict.
  // start should not move past kernel0.
  //
  //   kernel0(write buf0), start(write buf0), done
  //   => kernel0(write buf0), start(write buf0), done  (unchanged for start)
  //   done can move... but there is nothing after it.
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  auto slice = BufferAllocation::Slice(&alloc, 0, 1024);
  auto shape = ShapeUtil::MakeShape(F32, {256});

  auto start = MakeAsyncStart(BufferUse::Write(slice, shape));
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(MakeFakeThunk(BufferUse::Write(slice, shape)));
  thunks.push_back(std::move(start));
  thunks.push_back(std::move(done));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_FALSE(changed);
  EXPECT_EQ(thunks[0]->kind(),
            Thunk::kKernel);  // FakeThunk
  EXPECT_EQ(thunks[1]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(thunks[2]->kind(), Thunk::kAsyncDone);
}

TEST(ExpandAsyncScopeThunkPassTest, DoneStopsAtConflictingThunk) {
  // done (inherits start's write to buf0) cannot move past kernel that also
  // writes buf0.
  //
  //   start(write buf0), done, kernel(write buf0)
  //   => start(write buf0), done, kernel(write buf0)  (unchanged)
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  auto slice = BufferAllocation::Slice(&alloc, 0, 1024);
  auto shape = ShapeUtil::MakeShape(F32, {256});

  auto start = MakeAsyncStart(BufferUse::Write(slice, shape));
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(std::move(start));
  thunks.push_back(std::move(done));
  thunks.push_back(MakeFakeThunk(BufferUse::Write(slice, shape)));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_FALSE(changed);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(thunks[1]->kind(), Thunk::kAsyncDone);
  EXPECT_EQ(thunks[2]->kind(), Thunk::kKernel);
}

TEST(ExpandAsyncScopeThunkPassTest, StartMovesUpPastNonConflicting) {
  // start writes buf0, kernel writes buf1 => no conflict => start moves up.
  //
  //   kernel(write buf1), start(write buf0), done
  //   => start(write buf0), kernel(write buf1), done
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto slice0 = BufferAllocation::Slice(&alloc0, 0, 1024);
  auto slice1 = BufferAllocation::Slice(&alloc1, 0, 1024);
  auto shape = ShapeUtil::MakeShape(F32, {256});

  auto start = MakeAsyncStart(BufferUse::Write(slice0, shape));
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(MakeFakeThunk(BufferUse::Write(slice1, shape)));
  thunks.push_back(std::move(start));
  thunks.push_back(std::move(done));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_TRUE(changed);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(thunks[1]->kind(), Thunk::kKernel);  // FakeThunk
  EXPECT_EQ(thunks[2]->kind(), Thunk::kAsyncDone);
}

TEST(ExpandAsyncScopeThunkPassTest, ReadReadDoesNotConflict) {
  // Both start and kernel only read buf0 => no conflict => start moves up,
  // done moves down.
  //
  //   kernel(read buf0), start(read buf0), done, kernel2(read buf0)
  //   => start, kernel, kernel2, done
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  auto slice = BufferAllocation::Slice(&alloc, 0, 1024);
  auto shape = ShapeUtil::MakeShape(F32, {256});

  auto start = MakeAsyncStart(BufferUse::Read(slice, shape));
  auto done = MakeAsyncDone(start.get());

  ThunkSequence thunks;
  thunks.push_back(MakeFakeThunk(BufferUse::Read(slice, shape)));
  thunks.push_back(std::move(start));
  thunks.push_back(std::move(done));
  thunks.push_back(MakeFakeThunk(BufferUse::Read(slice, shape)));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_TRUE(changed);
  ASSERT_EQ(thunks.size(), 4);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(thunks[3]->kind(), Thunk::kAsyncDone);
}

//===----------------------------------------------------------------------===//
// Helpers for running a pass on a single ThunkSequence.
//===----------------------------------------------------------------------===//

absl::StatusOr<bool> RunRemovePass(ThunkSequence* thunks) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;
  RemoveRedundantAsyncThunkPass pass;
  return pass.Run(thunks, debug_options, /*hlo_module=*/nullptr, device_info,
                  allocator);
}

//===----------------------------------------------------------------------===//
// Nested control flow tests: SequentialThunk.
//===----------------------------------------------------------------------===//

TEST(RemoveRedundantAsyncThunkPassTest, RecursesIntoSequentialThunk) {
  // Place an adjacent start/done pair inside a SequentialThunk and verify the
  // pass inlines it.
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence inner;
  inner.push_back(std::move(start));
  inner.push_back(std::move(done));

  auto seq =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(inner));
  auto* seq_ptr = seq.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(seq));

  ASSERT_OK_AND_ASSIGN(bool changed, RunRemovePass(&thunks));
  EXPECT_TRUE(changed);
  // The SequentialThunk is still there, but its inner sequence should now
  // contain only the inlined kernel thunk.
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kSequential);
  ASSERT_EQ(seq_ptr->thunks().size(), 1);
  EXPECT_EQ(seq_ptr->thunks()[0]->kind(), Thunk::kKernel);
}

TEST(ExpandAsyncScopeThunkPassTest, RecursesIntoSequentialThunk) {
  // Place kernel, start, kernel, done, kernel inside a SequentialThunk and
  // verify the expand pass widens the async scope within it.
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence inner;
  inner.push_back(MakeFakeThunk());
  inner.push_back(std::move(start));
  inner.push_back(MakeFakeThunk());
  inner.push_back(std::move(done));
  inner.push_back(MakeFakeThunk());

  auto seq =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(inner));
  auto* seq_ptr = seq.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(seq));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_TRUE(changed);
  // start should be at position 0 and done at position 4 within the
  // SequentialThunk's inner sequence.
  ASSERT_EQ(seq_ptr->thunks().size(), 5);
  EXPECT_EQ(seq_ptr->thunks()[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(seq_ptr->thunks()[4]->kind(), Thunk::kAsyncDone);
}

//===----------------------------------------------------------------------===//
// Nested control flow tests: WhileThunk.
//===----------------------------------------------------------------------===//

TEST(RemoveRedundantAsyncThunkPassTest, RecursesIntoWhileThunkBody) {
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence body;
  body.push_back(std::move(start));
  body.push_back(std::move(done));

  // WhileThunk needs a condition buffer and a condition thunk sequence.
  BufferAllocation alloc(/*index=*/0, /*size=*/1, /*color=*/0);
  auto cond_slice = BufferAllocation::Slice(&alloc, 0, 1);

  ThunkSequence condition;
  condition.push_back(MakeFakeThunk());

  auto while_thunk = std::make_unique<WhileThunk>(
      Thunk::ThunkInfo(), cond_slice, std::move(condition), std::move(body));
  auto* while_ptr = while_thunk.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(while_thunk));

  ASSERT_OK_AND_ASSIGN(bool changed, RunRemovePass(&thunks));
  EXPECT_TRUE(changed);
  ASSERT_EQ(while_ptr->body_executor().thunks().size(), 1);
  EXPECT_EQ(while_ptr->body_executor().thunks()[0]->kind(), Thunk::kKernel);
}

TEST(ExpandAsyncScopeThunkPassTest, RecursesIntoWhileThunkBody) {
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence body;
  body.push_back(MakeFakeThunk());
  body.push_back(std::move(start));
  body.push_back(MakeFakeThunk());
  body.push_back(std::move(done));
  body.push_back(MakeFakeThunk());

  BufferAllocation alloc(/*index=*/0, /*size=*/1, /*color=*/0);
  auto cond_slice = BufferAllocation::Slice(&alloc, 0, 1);

  ThunkSequence condition;
  condition.push_back(MakeFakeThunk());

  auto while_thunk = std::make_unique<WhileThunk>(
      Thunk::ThunkInfo(), cond_slice, std::move(condition), std::move(body));
  auto* while_ptr = while_thunk.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(while_thunk));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_TRUE(changed);
  auto& body_thunks = while_ptr->body_executor().thunks();
  ASSERT_EQ(body_thunks.size(), 5);
  EXPECT_EQ(body_thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(body_thunks[4]->kind(), Thunk::kAsyncDone);
}

//===----------------------------------------------------------------------===//
// Nested control flow tests: ConditionalThunk.
//===----------------------------------------------------------------------===//

TEST(RemoveRedundantAsyncThunkPassTest, RecursesIntoConditionalThunkBranches) {
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence branch0;
  branch0.push_back(std::move(start));
  branch0.push_back(std::move(done));

  ThunkSequence branch1;
  branch1.push_back(MakeFakeThunk());

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(branch0));
  branches.push_back(std::move(branch1));

  BufferAllocation alloc(/*index=*/0, /*size=*/4, /*color=*/0);
  auto index_slice = BufferAllocation::Slice(&alloc, 0, 4);
  ShapedSlice shaped_slice{index_slice, ShapeUtil::MakeShape(S32, {})};

  auto cond_thunk = std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo(), shaped_slice, std::move(branches));
  auto* cond_ptr = cond_thunk.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(cond_thunk));

  ASSERT_OK_AND_ASSIGN(bool changed, RunRemovePass(&thunks));
  EXPECT_TRUE(changed);
  // Branch 0 should have the start/done inlined to a single kernel thunk.
  ASSERT_EQ(cond_ptr->branch_executors()[0].thunks().size(), 1);
  EXPECT_EQ(cond_ptr->branch_executors()[0].thunks()[0]->kind(),
            Thunk::kKernel);
  // Branch 1 should be unchanged.
  ASSERT_EQ(cond_ptr->branch_executors()[1].thunks().size(), 1);
  EXPECT_EQ(cond_ptr->branch_executors()[1].thunks()[0]->kind(),
            Thunk::kKernel);
}

TEST(ExpandAsyncScopeThunkPassTest, RecursesIntoConditionalThunkBranches) {
  auto start = MakeAsyncStart();
  auto done = MakeAsyncDone(start.get());

  ThunkSequence branch0;
  branch0.push_back(MakeFakeThunk());
  branch0.push_back(std::move(start));
  branch0.push_back(MakeFakeThunk());
  branch0.push_back(std::move(done));
  branch0.push_back(MakeFakeThunk());

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(branch0));

  BufferAllocation alloc(/*index=*/0, /*size=*/4, /*color=*/0);
  auto index_slice = BufferAllocation::Slice(&alloc, 0, 4);
  ShapedSlice shaped_slice{index_slice, ShapeUtil::MakeShape(S32, {})};

  auto cond_thunk = std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo(), shaped_slice, std::move(branches));
  auto* cond_ptr = cond_thunk.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(cond_thunk));

  ASSERT_OK_AND_ASSIGN(bool changed, RunExpandPass(&thunks));
  EXPECT_TRUE(changed);
  auto& branch_thunks = cond_ptr->branch_executors()[0].thunks();
  ASSERT_EQ(branch_thunks.size(), 5);
  EXPECT_EQ(branch_thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(branch_thunks[4]->kind(), Thunk::kAsyncDone);
}

}  // namespace
}  // namespace xla::gpu
