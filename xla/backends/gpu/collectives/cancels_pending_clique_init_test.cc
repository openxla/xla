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

#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/mock_stream_executor.h"

namespace xla::gpu {
namespace {

using ::testing::NiceMock;
using ::testing::Return;

namespace se = ::stream_executor;

// Blocks inside CreateCommunicatorsWithCancel the way PollUntilDone blocks:
// returns only after the test releases it, and reports whether the shared
// token was cancelled while it was blocked.
class BlockingInitCollectives : public GpuCollectives {
 public:
  bool IsImplemented() const override { return true; }

  absl::StatusOr<CliqueIdCallback> InitializeTopology(
      const Topology&) override {
    return absl::UnimplementedError("unused");
  }

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const override {
    return CliqueId("pending-init");
  }

  absl::StatusOr<std::unique_ptr<Communicator>> CreateCommunicator() override {
    return absl::UnimplementedError("unused");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey&, const std::optional<CliqueIds>&,
                      absl::Span<const DeviceRank>,
                      const Collectives::Config&) override {
    return absl::UnimplementedError("unused");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const>, int32_t, absl::Span<const RankId>,
      const Collectives::Config&, absl::Span<const DeviceRank>) override {
    return absl::UnimplementedError("unused");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicatorsWithCancel(const CliqueKey&,
                                const std::optional<CliqueIds>&,
                                absl::Span<const DeviceRank>,
                                const Collectives::Config&,
                                std::shared_ptr<CancellationToken> cancel)
      override {
    {
      absl::MutexLock lock(mu_);
      cancel_ = cancel;
    }
    entered_.Notify();
    release_.WaitForNotification();
    if (cancel->IsCancelled()) {
      return absl::CancelledError("pending clique init cancelled");
    }
    return absl::InternalError(
        "pending clique init was not cancelled by task failure");
  }

  bool WaitUntilEntered(absl::Duration timeout) {
    return entered_.WaitForNotificationWithTimeout(timeout);
  }

  std::shared_ptr<CancellationToken> token() const {
    absl::MutexLock lock(mu_);
    return cancel_;
  }

  void Release() {
    if (!release_.HasBeenNotified()) {
      release_.Notify();
    }
  }

 private:
  mutable absl::Mutex mu_;
  std::shared_ptr<CancellationToken> cancel_;
  absl::Notification entered_;
  absl::Notification release_;
};

coordination::TaskInfo MakeConnectedTask(int task_id, uint64_t incarnation) {
  coordination::TaskInfo info;
  info.set_task_id(task_id);
  info.set_state(coordination::TaskState::CONNECTED);
  info.set_incarnation(incarnation);
  return info;
}

void ResetProcessTaskState() {
  std::vector<coordination::TaskInfo> empty;
  (void)UpdateGlobalProcessInfo(absl::MakeSpan(empty));
}

// A task failure during CreateCommunicatorsWithCancel must cancel the token
// stored in pending_cliques. LockableGpuClique does not exist yet. No GPU and
// no NCCL are involved.
TEST(PendingCliqueInitTest, TaskFailureCancelsInProgressInit) {
  BlockingInitCollectives collectives;
  auto cleanup = absl::MakeCleanup([&] {
    collectives.Release();
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  constexpr uint64_t kIncarnation = 10;
  std::vector<coordination::TaskInfo> infos = {
      MakeConnectedTask(/*task_id=*/0, kIncarnation),
  };
  ASSERT_OK(UpdateGlobalProcessInfo(absl::MakeSpan(infos)));

  NiceMock<se::MockStreamExecutor> executor;
  ON_CALL(executor, SynchronizeAllActivity).WillByDefault(Return(true));
  ON_CALL(executor, device_ordinal).WillByDefault(Return(0));

  GpuCliqueKey key({GlobalDeviceId(0)}, /*num_local_participants=*/1,
                   CommunicationId(0),
                   /*incarnations=*/{IncarnationId(kIncarnation)});
  std::vector<std::vector<GlobalDeviceId>> groups = {{GlobalDeviceId(0)}};
  AcquiredCliquesMap acquired;

  absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> acquire_result;
  std::thread worker([&] {
    acquire_result = AcquireClique(
        &collectives, &executor, RunId(0), key, groups,
        [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
          return CliqueIds(CliqueId("pending-init"));
        },
        RankId(0), acquired);
  });
  auto join_worker = absl::MakeCleanup([&] {
    collectives.Release();
    if (worker.joinable()) {
      worker.join();
    }
  });

  ASSERT_TRUE(collectives.WaitUntilEntered(absl::Seconds(30)))
      << "Init never reached CreateCommunicatorsWithCancel.";

  ASSERT_OK(AbortTaskCliques(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated task death")));

  std::shared_ptr<CancellationToken> token = collectives.token();
  ASSERT_NE(token, nullptr);
  const bool cancelled = token->IsCancelled();

  // Unblock init before asserting so a red result still exits.
  collectives.Release();
  worker.join();

  EXPECT_TRUE(cancelled)
      << "Task failure did not cancel the pending clique init token. "
         "Acquire finished with: "
      << acquire_result.status();
}

}  // namespace
}  // namespace xla::gpu
