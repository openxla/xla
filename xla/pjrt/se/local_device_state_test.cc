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

#include "xla/pjrt/se/local_device_state.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <variant>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

class ErroredEvent : public se::Event {
 public:
  Status PollForStatus() override { return Status::kError; }
};

// Like a stream host callback, a polled callback still runs if its event
// reports an error and no error callback was given.
TEST(LocalDeviceStateTest, PolledCallbackRunsOnEventErrorWithoutErrorCallback) {
  NiceMock<se::MockPlatform> platform;
  se::DeviceDescription description;
  description.set_confidential_computing_enabled(true);
  NiceMock<se::MockStreamExecutor> executor;
  ON_CALL(executor, GetPlatform()).WillByDefault(Return(&platform));
  ON_CALL(executor, GetDeviceDescription())
      .WillByDefault(ReturnRef(description));
  ON_CALL(executor, CreateStream(_))
      .WillByDefault([&executor](
                         std::optional<std::variant<se::StreamPriority, int>>) {
        auto stream = std::make_unique<NiceMock<se::MockStream>>();
        ON_CALL(*stream, parent()).WillByDefault(Return(&executor));
        return absl::StatusOr<std::unique_ptr<se::Stream>>(std::move(stream));
      });
  ON_CALL(executor, CreateEvent()).WillByDefault([]() {
    return absl::StatusOr<std::unique_ptr<se::Event>>(
        std::make_unique<ErroredEvent>());
  });
  LocalDeviceState local_device_state(
      &executor, /*client=*/nullptr,
      LocalDeviceState::AllocationModel::kSynchronous,
      /*max_inflight_computations=*/std::nullopt,
      /*allow_event_reuse=*/true, /*use_callback_stream=*/false);
  ASSERT_TRUE(local_device_state.uses_event_polling_callbacks());

  absl::Notification called;
  ASSERT_OK(local_device_state.ThenExecuteCallback(
      local_device_state.compute_stream(), [&]() { called.Notify(); }));
  EXPECT_TRUE(called.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

}  // namespace
}  // namespace xla
