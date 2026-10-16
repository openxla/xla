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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
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
#include "xla/core/collectives/reduction_kind.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::NiceMock;
using ::testing::Return;

namespace se = ::stream_executor;

// Communicator returned for a parent clique so a later acquire can split it.
// Collective ops are unused. Abort succeeds and is counted so a test can see
// whether the parent was aborted while its lock was still held. The counter
// outlives the clique: a later abort destroys the communicator after Abort().
class StubCommunicator : public Communicator {
 public:
  static void ResetAbortCount() { abort_count_.store(0); }
  static int AbortCount() { return abort_count_.load(); }

  absl::Status Abort() override {
    abort_count_.fetch_add(1);
    return absl::OkStatus();
  }

  Future<> AllReduce(se::DeviceAddressBase, se::DeviceAddressBase,
                     PrimitiveType, size_t, ReductionKind,
                     const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> Broadcast(se::DeviceAddressBase, se::DeviceAddressBase,
                     PrimitiveType, size_t, RankId, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> ReduceScatter(se::DeviceAddressBase, se::DeviceAddressBase,
                         PrimitiveType, size_t, ReductionKind,
                         const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> AllGather(se::DeviceAddressBase, se::DeviceAddressBase,
                     PrimitiveType, size_t, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> CollectivePermute(se::DeviceAddressBase, se::DeviceAddressBase,
                             PrimitiveType, size_t, std::optional<RankId>,
                             absl::Span<const RankId>,
                             const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4>,
                    absl::InlinedVector<se::DeviceAddressBase, 4>,
                    PrimitiveType, size_t, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> Send(se::DeviceAddressBase, PrimitiveType, size_t, RankId,
                const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> Recv(se::DeviceAddressBase, PrimitiveType, size_t, RankId,
                const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::StatusOr<size_t> NumRanks() const override { return 1; }

  std::string ToString() const override { return "stub"; }

 private:
  static std::atomic<int> abort_count_;
};

std::atomic<int> StubCommunicator::abort_count_{0};

// Blocks inside CreateCommunicatorsWithCancel or SplitCommunicatorsWithCancel
// the way PollUntilDone blocks: returns only after the test releases it, and
// reports whether the shared token was cancelled while it was blocked.
class BlockingInitCollectives : public GpuCollectives {
 public:
  enum class PendingKind { kInit, kSplit };

  explicit BlockingInitCollectives(PendingKind pending = PendingKind::kInit)
      : pending_(pending) {}

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
  CreateCommunicatorsWithCancel(
      const CliqueKey&, const std::optional<CliqueIds>&,
      absl::Span<const DeviceRank> ranks, const Collectives::Config&,
      std::shared_ptr<CancellationToken> cancel) override {
    if (pending_ == PendingKind::kSplit) {
      std::vector<std::unique_ptr<Communicator>> comms;
      comms.reserve(ranks.size());
      for (size_t i = 0; i < ranks.size(); ++i) {
        comms.push_back(std::make_unique<StubCommunicator>());
      }
      return comms;
    }
    return BlockUntilReleased(std::move(cancel), "pending clique init");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  SplitCommunicatorsWithCancel(
      absl::Span<const Communicator* const>, int32_t, absl::Span<const RankId>,
      const Collectives::Config&, absl::Span<const DeviceRank>,
      std::shared_ptr<CancellationToken> cancel) override {
    if (pending_ != PendingKind::kSplit) {
      return absl::UnimplementedError("unused");
    }
    return BlockUntilReleased(std::move(cancel), "pending clique split");
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
  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> BlockUntilReleased(
      std::shared_ptr<CancellationToken> cancel, absl::string_view what) {
    {
      absl::MutexLock lock(mu_);
      cancel_ = cancel;
    }
    entered_.Notify();
    release_.WaitForNotification();
    if (cancel->IsCancelled()) {
      return absl::CancelledError(absl::StrCat(what, " cancelled"));
    }
    return absl::InternalError(
        absl::StrCat(what, " was not cancelled by task failure"));
  }

  PendingKind pending_;
  mutable absl::Mutex mu_;
  std::shared_ptr<CancellationToken> cancel_ ABSL_GUARDED_BY(mu_);
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

// A task failure during SplitCommunicatorsWithCancel must cancel the pending
// token immediately. The parent lock is held until the split returns, so the
// parent communicator stays alive across that call. Once the lock is released,
// a later abort of the same already-failed task must abort the parent once.
// No GPU and no NCCL are involved.
TEST(PendingCliqueInitTest, TaskFailureCancelsInProgressSplit) {
  StubCommunicator::ResetAbortCount();
  BlockingInitCollectives collectives(
      BlockingInitCollectives::PendingKind::kSplit);
  AcquiredCliquesMap acquired;
  absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> acquire_result;
  std::thread worker;
  auto cleanup = absl::MakeCleanup([&] {
    collectives.Release();
    if (worker.joinable()) {
      worker.join();
    }
    acquire_result = absl::CancelledError("cleanup");
    acquired.clear();
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

  const std::vector<IncarnationId> incarnations = {IncarnationId(kIncarnation)};
  GpuCliqueKey parent_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                          /*num_local_participants=*/1, CommunicationId(0),
                          incarnations);
  GpuCliqueKey child_key({GlobalDeviceId(0)}, /*num_local_participants=*/1,
                         CommunicationId(0), incarnations);
  // device_groups must cover every device of the parent, or the child is
  // created instead of split.
  std::vector<std::vector<GlobalDeviceId>> parent_devices = {
      {GlobalDeviceId(0), GlobalDeviceId(1)}};

  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<LockableGpuClique::Lock> parent_lock,
      AcquireClique(
          &collectives, &executor, RunId(0), parent_key, parent_devices,
          [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
            return CliqueIds(CliqueId("pending-split-parent"));
          },
          RankId(0), acquired));
  acquired.emplace(parent_key, parent_lock);

  worker = std::thread([&] {
    acquire_result = AcquireClique(
        &collectives, &executor, RunId(1), child_key, parent_devices,
        [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
          return CliqueIds(CliqueId("pending-split-child"));
        },
        RankId(0), acquired);
  });

  ASSERT_TRUE(collectives.WaitUntilEntered(absl::Seconds(30)))
      << "Split never reached SplitCommunicatorsWithCancel.";

  ASSERT_OK(AbortTaskCliques(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated task death")));

  std::shared_ptr<CancellationToken> token = collectives.token();
  ASSERT_NE(token, nullptr);
  EXPECT_TRUE(token->IsCancelled())
      << "Task failure did not cancel the pending clique split token.";
  // The split has not returned, so the parent lock is still held.
  const int aborts_while_locked = StubCommunicator::AbortCount();
  EXPECT_EQ(aborts_while_locked, 0)
      << "Parent communicator was aborted while the split still held its lock.";

  collectives.Release();
  worker.join();
  parent_lock.reset();
  acquired.clear();

  EXPECT_EQ(acquire_result.status().code(), absl::StatusCode::kCancelled)
      << acquire_result.status();

  // The task is already ERROR. Releasing the parent lock must still let a
  // later abort reclaim that parent, once.
  ASSERT_OK(AbortTaskCliques(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated task death")));
  EXPECT_EQ(StubCommunicator::AbortCount(), aborts_while_locked + 1)
      << "Parent communicator was not aborted once after its lock was "
         "released.";
}

}  // namespace
}  // namespace xla::gpu
