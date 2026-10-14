// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/server/host_buffer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/python/ifrt_proxy/common/transfer_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Pointee;

TEST(HostBufferStoreTest, ReadAfterWrite) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  ASSERT_THAT(store.Store(kHandle, "foo"), absl_testing::IsOk());
  EXPECT_THAT(store.Lookup(kHandle),
              absl_testing::IsOkAndHolds(Pointee(std::string("foo"))));

  ASSERT_THAT(store.Delete(kHandle), absl_testing::IsOk());
  EXPECT_THAT(store.Lookup(kHandle),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(HostBufferStoreTest, WriteAfterReadStarted) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  auto [lookup_promise, lookup_fut] =
      tsl::MakePromise<HostBufferStore::MemRegion>();

  absl::Notification closure_started;
  tsl::Env::Default()->SchedClosure(
      [&, promise = std::move(lookup_promise)]() mutable {
        closure_started.Notify();
        promise.Set(store.Lookup(kHandle, /*timeout=*/absl::Seconds(10)));
      });

  closure_started.WaitForNotification();
  absl::SleepFor(absl::Seconds(1));

  ASSERT_THAT(store.Store(kHandle, "foo"), absl_testing::IsOk());
  EXPECT_THAT(lookup_fut.Await(),
              absl_testing::IsOkAndHolds(Pointee(std::string("foo"))));
}

TEST(HostBufferStoreTest, ShutdownAfterReadStarted) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  auto [lookup_promise, lookup_fut] =
      tsl::MakePromise<HostBufferStore::MemRegion>();

  absl::Notification closure_started;
  tsl::Env::Default()->SchedClosure([&, promise = std::move(
                                            lookup_promise)]() mutable {
    closure_started.Notify();
    promise.Set(store.Lookup(kHandle, /*timeout=*/absl::InfiniteDuration()));
  });

  closure_started.WaitForNotification();
  absl::SleepFor(absl::Seconds(1));

  store.Shutdown("test");
  EXPECT_THAT(lookup_fut.Await(),
              absl_testing::StatusIs(Not(absl::StatusCode::kOk)));
}

TEST(HostBufferStoreTest, WriteAfterShutdown) {
  HostBufferStore store;
  const uint64_t kHandle = 1;
  store.Shutdown("test");
  EXPECT_THAT(store.Store(kHandle, "foo"),
              absl_testing::StatusIs(Not(absl::StatusCode::kOk)));
}

TEST(HostBufferStoreTest, LookupAfterShutdown) {
  HostBufferStore store;
  const uint64_t kHandle = 1;
  ASSERT_THAT(store.Store(kHandle, "foo"), absl_testing::IsOk());
  store.Shutdown("test");
  EXPECT_THAT(store.Lookup(kHandle, /*timeout=*/absl::InfiniteDuration()),
              absl_testing::StatusIs(Not(absl::StatusCode::kOk)));
}

TEST(HostBufferStoreTest, ReadFromDiskToleratesDeletedFile) {
  HostBufferStore store;
  const uint64_t kHandle = 1;
  const std::optional<std::string> file_path = LargeTransferFilePath(kHandle);
  ASSERT_TRUE(file_path.has_value());
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), *file_path, "foo"),
              absl_testing::IsOk());

  ASSERT_THAT(store.ReadFromDisk(kHandle), absl_testing::IsOk());
  // Clients delete the file as soon as `ReadFromDisk()` returns.
  ASSERT_THAT(tsl::Env::Default()->DeleteFile(*file_path),
              absl_testing::IsOk());
  EXPECT_THAT(store.Lookup(kHandle),
              absl_testing::IsOkAndHolds(Pointee(std::string("foo"))));

  // Releasing the buffer should not warn about the file being gone.
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("Failed to delete file")))
      .Times(0);
  log.StartCapturingLogs();
  EXPECT_THAT(store.Delete(kHandle), absl_testing::IsOk());
}

TEST(HostBufferStoreTest, ReadFromDiskDeletesFileOnRelease) {
  HostBufferStore store;
  const uint64_t kHandle = 2;
  const std::optional<std::string> file_path = LargeTransferFilePath(kHandle);
  ASSERT_TRUE(file_path.has_value());
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), *file_path, "foo"),
              absl_testing::IsOk());

  // Older clients rely on the server to delete the file.
  ASSERT_THAT(store.ReadFromDisk(kHandle), absl_testing::IsOk());
  ASSERT_THAT(store.Delete(kHandle), absl_testing::IsOk());
  EXPECT_THAT(tsl::Env::Default()->FileExists(*file_path),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(HostBufferStoreTest, UnknownHandle) {
  HostBufferStore store;
  const uint64_t kHandle = 1;

  EXPECT_THAT(store.Lookup(kHandle),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(store.Delete(kHandle),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
