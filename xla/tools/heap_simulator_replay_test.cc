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

#include "xla/tools/heap_simulator_replay.h"

#include <cstdint>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/service/hlo.pb.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

void AddBuffer(BufferAllocationProto* allocation, int64_t id, int64_t offset,
               int64_t size) {
  auto* buffer = allocation->add_assigned();
  buffer->set_logical_buffer_id(id);
  buffer->set_offset(offset);
  buffer->set_size(size);
}

void AddEvent(HeapSimulatorTrace* trace, HeapSimulatorTrace::Event::Kind kind,
              int64_t id) {
  auto* event = trace->add_events();
  event->set_kind(kind);
  event->set_buffer_id(id);
}

HloProto MakeProto() {
  HloProto proto;
  auto* assignment = proto.mutable_buffer_assignment();
  auto* allocation = assignment->add_buffer_allocations();
  allocation->set_index(7);
  allocation->set_size(7);
  AddBuffer(allocation, 10, 0, 4);
  // CombineTempAllocations appends an untraced allocation after the heap.
  AddBuffer(allocation, 11, 4, 3);
  auto* trace = assignment->add_heap_simulator_traces();
  // A dump can retain the old index after allocations are renumbered.
  trace->set_buffer_allocation_index(3);
  AddEvent(trace, HeapSimulatorTrace::Event::ALLOC, 10);
  AddEvent(trace, HeapSimulatorTrace::Event::FREE, 10);
  return proto;
}

TEST(HeapSimulatorReplayTest, StaleIndexAndUntracedSuffix) {
  const HloProto proto = MakeProto();
  const std::string original = proto.SerializeAsString();
  std::ostringstream output;
  const absl::Status status = ReplayHeapSimulator(proto, 4, output);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(proto.SerializeAsString(), original);
  EXPECT_THAT(output.str(), HasSubstr("recorded_trace_bytes=4"));
  EXPECT_THAT(output.str(), HasSubstr("untraced_suffix_bytes=3"));
  EXPECT_THAT(output.str(), HasSubstr("traced_buffers=1 assigned_buffers=2"));
  EXPECT_THAT(output.str(), HasSubstr("| size x duration"));
}

TEST(HeapSimulatorReplayTest, ReplaysColocation) {
  HloProto proto = MakeProto();
  auto* assignment = proto.mutable_buffer_assignment();
  AddBuffer(assignment->mutable_buffer_allocations(0), 12, 0, 4);
  auto* trace = assignment->mutable_heap_simulator_traces(0);
  trace->clear_events();
  AddEvent(trace, HeapSimulatorTrace::Event::ALLOC, 10);
  AddEvent(trace, HeapSimulatorTrace::Event::SHARE_WITH, 12);
  trace->mutable_events(1)->set_share_with_canonical_id(10);
  AddEvent(trace, HeapSimulatorTrace::Event::FREE, 10);
  AddEvent(trace, HeapSimulatorTrace::Event::FREE, 12);
  std::ostringstream output;
  const absl::Status status = ReplayHeapSimulator(proto, 4, output);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_THAT(output.str(), HasSubstr("traced_buffers=2 assigned_buffers=3"));
}

TEST(HeapSimulatorReplayTest, RejectsMultipleTraces) {
  HloProto proto = MakeProto();
  auto* assignment = proto.mutable_buffer_assignment();
  const HeapSimulatorTrace copy = assignment->heap_simulator_traces(0);
  *assignment->add_heap_simulator_traces() = copy;
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsTraceCrossingAllocations) {
  HloProto proto = MakeProto();
  AddEvent(proto.mutable_buffer_assignment()->mutable_heap_simulator_traces(0),
           HeapSimulatorTrace::Event::ALLOC, 99);
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsIncompleteLifetime) {
  HloProto proto = MakeProto();
  auto* trace =
      proto.mutable_buffer_assignment()->mutable_heap_simulator_traces(0);
  trace->clear_events();
  AddEvent(trace, HeapSimulatorTrace::Event::ALLOC, 10);
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsFreeWithoutAllocation) {
  HloProto proto = MakeProto();
  proto.mutable_buffer_assignment()
      ->mutable_heap_simulator_traces(0)
      ->mutable_events(0)
      ->set_kind(HeapSimulatorTrace::Event::FREE);
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsUnknownCanonicalBuffer) {
  HloProto proto = MakeProto();
  auto* event = proto.mutable_buffer_assignment()
                    ->mutable_heap_simulator_traces(0)
                    ->mutable_events(0);
  event->set_kind(HeapSimulatorTrace::Event::SHARE_WITH);
  event->set_share_with_canonical_id(99);
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsUntracedBufferInsideHeap) {
  HloProto proto = MakeProto();
  proto.mutable_buffer_assignment()
      ->mutable_buffer_allocations(0)
      ->mutable_assigned(1)
      ->set_offset(0);
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsOutOfBoundsBuffer) {
  HloProto proto = MakeProto();
  proto.mutable_buffer_assignment()
      ->mutable_buffer_allocations(0)
      ->mutable_assigned(0)
      ->set_size(8);
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(proto, 4, output).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeapSimulatorReplayTest, RejectsNonpositiveAlignment) {
  std::ostringstream output;
  EXPECT_EQ(ReplayHeapSimulator(MakeProto(), 0, output).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace xla
