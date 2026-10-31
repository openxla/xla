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

#include "xla/pjrt/gpu/collective_graph_invalidation.h"

#include <array>
#include <memory>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

class TestStore : public InMemoryKeyValueStore {
 public:
  absl::StatusOr<std::string> Get(absl::string_view key,
                                  absl::Duration) override {
    return InMemoryKeyValueStore::Get(key, absl::Seconds(5));
  }
};

class CollectiveGraphInvalidationTest : public ::testing::Test {
 protected:
  CollectiveGraphInvalidationTest() {
    for (int process = 0; process != 2; ++process) {
      callbacks_[process] = CreateCollectiveGraphInvalidationCallback(
          ProcessId(process), devices_, store_);
    }
  }

  absl::StatusOr<std::array<bool, 2>> RunRound(absl::string_view signature,
                                               std::array<bool, 2> dirty) {
    std::array<absl::StatusOr<bool>, 2> results;
    std::thread remote([&] {
      results[1] = callbacks_[1](signature, participants_, dirty[1]);
    });
    results[0] = callbacks_[0](signature, participants_, dirty[0]);
    remote.join();
    ABSL_ASSIGN_OR_RETURN(bool local, results[0]);
    ABSL_ASSIGN_OR_RETURN(bool other, results[1]);
    return std::array<bool, 2>{local, other};
  }

  const absl::flat_hash_map<GlobalDeviceId, ProcessId> devices_ = {
      {GlobalDeviceId(0), ProcessId(0)}, {GlobalDeviceId(1), ProcessId(1)}};
  const std::array<GlobalDeviceId, 2> participants_ = {GlobalDeviceId(0),
                                                       GlobalDeviceId(1)};
  std::shared_ptr<TestStore> store_ = std::make_shared<TestStore>();
  std::array<CollectiveGraphInvalidationCallback, 2> callbacks_;
};

TEST_F(CollectiveGraphInvalidationTest, SuccessiveRoundsDoNotReuseDirtyState) {
  EXPECT_THAT(RunRound("cliques", {false, true}),
              IsOkAndHolds(ElementsAre(true, true)));
  EXPECT_THAT(RunRound("cliques", {false, false}),
              IsOkAndHolds(ElementsAre(false, false)));
  EXPECT_THAT(RunRound("cliques", {true, false}),
              IsOkAndHolds(ElementsAre(true, true)));
}

TEST_F(CollectiveGraphInvalidationTest, IndependentSignaturesCanInterleave) {
  const std::array<GlobalDeviceId, 1> local = {GlobalDeviceId(0)};
  EXPECT_THAT(callbacks_[0]("local_work", local, true), IsOkAndHolds(true));

  absl::StatusOr<std::array<bool, 2>> first;
  std::thread other([&] { first = RunRound("first", {true, false}); });
  auto second = RunRound("second", {false, false});
  other.join();
  EXPECT_THAT(first, IsOkAndHolds(ElementsAre(true, true)));
  EXPECT_THAT(second, IsOkAndHolds(ElementsAre(false, false)));
}

class CleanupStore : public TestStore {
 public:
  absl::Status Set(absl::string_view key, absl::string_view value) override {
    ABSL_RETURN_IF_ERROR(TestStore::Set(key, value));
    if (absl::EndsWith(key, "/dirty/0")) {
      prefix = std::string(absl::StripSuffix(key, "dirty/0"));
      input_published.Notify();
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> Get(absl::string_view key,
                                  absl::Duration timeout) override {
    if (absl::EndsWith(key, "/ack/1")) {
      waiting_for_ack.Notify();
    }
    return TestStore::Get(key, timeout);
  }

  std::string prefix;
  absl::Notification input_published;
  absl::Notification waiting_for_ack;
};

TEST_F(CollectiveGraphInvalidationTest, CleanupWaitsForEveryResultReader) {
  auto store = std::make_shared<CleanupStore>();
  auto coordinator =
      CreateCollectiveGraphInvalidationCallback(ProcessId(0), devices_, store);
  absl::StatusOr<bool> result;
  std::thread worker(
      [&] { result = coordinator("cliques", participants_, false); });

  bool published =
      store->input_published.WaitForNotificationWithTimeout(absl::Seconds(5));
  EXPECT_TRUE(published);
  if (published) {
    EXPECT_OK(store->Set(absl::StrCat(store->prefix, "dirty/1"), "1"));
    EXPECT_TRUE(store->waiting_for_ack.WaitForNotificationWithTimeout(
        absl::Seconds(5)));
    EXPECT_THAT(store->TryGet(absl::StrCat(store->prefix, "result")),
                IsOkAndHolds("1"));
    EXPECT_OK(store->Set(absl::StrCat(store->prefix, "ack/1"), ""));
  }
  worker.join();
  EXPECT_THAT(result, IsOkAndHolds(true));
  EXPECT_THAT(store->TryGet(absl::StrCat(store->prefix, "result")),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla::gpu
