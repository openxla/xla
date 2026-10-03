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

#include "xla/pjrt/se/buffer_sequencing_event.h"

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/se/event_pool.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;

TEST(EventPoolHandleTest, DefaultConstructedHandleHasZeroSequenceNumber) {
  EventPool::Handle handle;
  EXPECT_EQ(handle.sequence_number(), 0);
  EXPECT_EQ(handle.event(), nullptr);
}

TEST(BufferSequencingEventTest, DefaultConstructedEventPoolHandleIsComplete) {
  auto seq_event = BufferSequencingEvent::Create(/*async_work_runner=*/nullptr);
  seq_event->SetSequencingEvent(EventPool::Handle(), /*stream=*/nullptr);

  EXPECT_EQ(seq_event->sequence_number(), 0);
  EXPECT_TRUE(seq_event->IsComplete());
  EXPECT_THAT(seq_event->WaitForEventOnExternalStream(0), IsOk());
}

TEST(BufferSequencingEventTest, CApiFunctionTableReturnsZeroSequenceNumber) {
  auto seq_event = BufferSequencingEvent::Create(/*async_work_runner=*/nullptr);
  seq_event->SetSequencingEvent(EventPool::Handle(), /*stream=*/nullptr);

  const PJRT_DeviceEvent_FunctionTable* vtable =
      internal::GetBuiltinDeviceEventCApiFunctionTable<BufferSequencingEvent>();
  ASSERT_NE(vtable, nullptr);
  ASSERT_NE(vtable->get_definition_stream, nullptr);

  uint64_t sequence_number = 0xdeadbeef;
  intptr_t stream_handle = vtable->get_definition_stream(
      seq_event.GetAsyncValue(), &sequence_number);
  EXPECT_EQ(stream_handle, 0);
  EXPECT_EQ(sequence_number, 0);
}

TEST(BufferSequencingEventTest, ComparisonOperators) {
  auto seq_event1 =
      BufferSequencingEvent::Create(/*async_work_runner=*/nullptr);
  seq_event1->SetSequencingEvent(EventPool::Handle(), /*stream=*/nullptr);

  auto seq_event2 =
      BufferSequencingEvent::Create(/*async_work_runner=*/nullptr);
  seq_event2->SetSequencingEvent(EventPool::Handle(), /*stream=*/nullptr);

  EXPECT_FALSE(*seq_event1 < *seq_event2);
  EXPECT_FALSE(*seq_event1 > *seq_event2);
  EXPECT_TRUE(*seq_event1 <= *seq_event2);
  EXPECT_TRUE(*seq_event1 >= *seq_event2);
}

}  // namespace
}  // namespace xla
