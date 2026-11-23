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

#include "xla/backends/gpu/collectives/gpu_clique.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/barrier.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/mock_gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_stream_executor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

class TestResource final : public GpuClique::Resource {
 public:
  explicit TestResource(std::function<void(bool)> retire)
      : retire_(std::move(retire)) {}
  void Retire(bool healthy) override { retire_(healthy); }

 private:
  std::function<void(bool)> retire_;
};

std::unique_ptr<GpuClique> MakeResourceClique(int ranks = 1) {
  absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
  std::vector<GlobalDeviceId> devices;
  for (int rank = 0; rank < ranks; ++rank) {
    comms[RankId(rank)] = std::make_unique<NiceMock<MockGpuCommunicator>>();
    devices.emplace_back(rank);
  }
  return std::make_unique<GpuClique>(GpuCliqueKey(std::move(devices), ranks),
                                     std::nullopt, std::move(comms), false,
                                     std::make_shared<CancellationToken>());
}

TEST(GpuCliqueTest, ResourcesAreReusedAndIsolatedByNameRankAndClique) {
  int created = 0;
  std::atomic<int> retired{0};
  auto create = [&]() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
    ++created;
    return std::make_unique<TestResource>([&](bool healthy) {
      EXPECT_TRUE(healthy);
      ++retired;
    });
  };
  auto clique = MakeResourceClique(2);
  ASSERT_OK_AND_ASSIGN(auto* first,
                       clique->GetOrCreateResource(RankId(0), "a", create));
  ASSERT_OK_AND_ASSIGN(auto* again,
                       clique->GetOrCreateResource(RankId(0), "a", create));
  EXPECT_EQ(first, again);
  ASSERT_OK_AND_ASSIGN(auto* other_name,
                       clique->GetOrCreateResource(RankId(0), "b", create));
  ASSERT_OK_AND_ASSIGN(auto* other_rank,
                       clique->GetOrCreateResource(RankId(1), "a", create));
  EXPECT_NE(first, other_name);
  EXPECT_NE(first, other_rank);

  // Identical device groups in different clique instances do not share state.
  auto replacement = MakeResourceClique(2);
  ASSERT_OK_AND_ASSIGN(auto* other_clique, replacement->GetOrCreateResource(
                                               RankId(0), "a", create));
  EXPECT_NE(first, other_clique);
  EXPECT_EQ(created, 4);
  EXPECT_EQ(retired, 0);
  clique.reset();
  EXPECT_EQ(retired, 3);
  replacement.reset();
  EXPECT_EQ(retired, 4);
}

TEST(GpuCliqueTest, ConcurrentLookupCreatesOnce) {
  auto clique = MakeResourceClique();
  int created = 0;
  auto create = [&]() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
    ++created;
    return std::make_unique<TestResource>([](bool healthy) {});
  };
  absl::Barrier ready(2);
  auto lookup = [&] {
    ready.Block();
    EXPECT_OK(clique->GetOrCreateResource(RankId(0), "a", create));
  };
  std::thread first(lookup);
  std::thread second(lookup);
  first.join();
  second.join();
  EXPECT_EQ(created, 1);
}

TEST(GpuCliqueTest, FactoryFailureIsNotCached) {
  auto clique = MakeResourceClique();
  int attempts = 0;
  auto create = [&]() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
    if (++attempts == 1) {
      return absl::ResourceExhaustedError("allocation failed");
    }
    return std::make_unique<TestResource>([](bool healthy) {});
  };
  EXPECT_THAT(clique->GetOrCreateResource(RankId(0), "a", create),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("allocation failed")));
  EXPECT_OK(clique->GetOrCreateResource(RankId(0), "a", create));
  EXPECT_EQ(attempts, 2);
  auto null_create =
      []() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
    return nullptr;
  };
  EXPECT_THAT(clique->GetOrCreateResource(RankId(0), "null", null_create),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("null")));
  EXPECT_THAT(clique->GetOrCreateResource(RankId(0), "", create),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("name")));
  EXPECT_THAT(clique->GetOrCreateResource(RankId(9), "a", create),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("rank")));
}

TEST(GpuCliqueTest, RetiresInNameOrderBeforeCommunicatorDestruction) {
  std::vector<std::string> order;
  class TrackedCommunicator : public NiceMock<MockGpuCommunicator> {
   public:
    explicit TrackedCommunicator(std::vector<std::string>& order)
        : order_(order) {}
    ~TrackedCommunicator() override { order_.push_back("communicator"); }

   private:
    std::vector<std::string>& order_;
  };
  absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
  comms[RankId(0)] = std::make_unique<TrackedCommunicator>(order);
  {
    GpuClique clique(GpuCliqueKey({GlobalDeviceId(0)}, 1), std::nullopt,
                     std::move(comms), false,
                     std::make_shared<CancellationToken>());
    for (const std::string& name : {"b", "a"}) {
      EXPECT_OK(clique.GetOrCreateResource(
          RankId(0), name,
          [&]() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
            return std::make_unique<TestResource>([&, name](bool healthy) {
              EXPECT_TRUE(healthy);
              order.push_back(name);
              // Would deadlock if the resource-map mutex were held.
              EXPECT_EQ(clique.device_comm(RankId(0), {}), std::nullopt);
            });
          }));
    }
  }
  EXPECT_THAT(order, ElementsAre("a", "b", "communicator"));
}

TEST(GpuCliqueTest, LocalRanksCanRetireCollectively) {
  auto clique = MakeResourceClique(2);
  absl::Barrier retirement(2);
  std::atomic<int> retired{0};
  for (int rank = 0; rank < 2; ++rank) {
    EXPECT_OK(clique->GetOrCreateResource(
        RankId(rank), "a",
        [&]() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
          return std::make_unique<TestResource>([&](bool healthy) {
            EXPECT_TRUE(healthy);
            retirement.Block();
            ++retired;
          });
        }));
  }
  clique.reset();
  EXPECT_EQ(retired, 2);
}

TEST(GpuCliqueTest, CancellationAbortAndHealthErrorsRetireAsUnhealthy) {
  for (int failure = 0; failure < 3; ++failure) {
    SCOPED_TRACE(failure);
    auto clique = MakeResourceClique();
    int retired = 0;
    auto create =
        [&]() -> absl::StatusOr<std::unique_ptr<GpuClique::Resource>> {
      return std::make_unique<TestResource>([&](bool healthy) {
        EXPECT_FALSE(healthy);
        ++retired;
      });
    };
    EXPECT_OK(clique->GetOrCreateResource(RankId(0), "a", create));
    auto* comm = static_cast<MockGpuCommunicator*>(*clique->comm(RankId(0)));
    if (failure == 0) {
      clique->Cancel();
    } else if (failure == 1) {
      EXPECT_CALL(*comm, Abort())
          .WillOnce(Return(absl::InternalError("abort failed")));
      EXPECT_THAT(clique->Abort(), StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("abort failed")));
      EXPECT_TRUE(clique->IsCancelled());
    } else {
      ON_CALL(*comm, HealthCheck())
          .WillByDefault(Return(absl::InternalError("communicator failed")));
    }
    if (failure != 2) {
      EXPECT_CALL(*comm, HealthCheck()).Times(0);
      EXPECT_THAT(clique->GetOrCreateResource(RankId(0), "a", create),
                  StatusIs(absl::StatusCode::kFailedPrecondition,
                           HasSubstr("cancelled")));
    }
    clique.reset();
    EXPECT_EQ(retired, 1);
  }
}

TEST(GpuCliqueTest, ReleaseSymmetricMemoryOnlyOnMatchingDevice) {
  NiceMock<se::MockStreamExecutor> executor0;
  NiceMock<se::MockStreamExecutor> executor1;
  ON_CALL(executor0, device_ordinal()).WillByDefault(Return(0));
  ON_CALL(executor1, device_ordinal()).WillByDefault(Return(1));

  auto comm0 = std::make_unique<StrictMock<MockGpuCommunicator>>();
  auto comm1 = std::make_unique<StrictMock<MockGpuCommunicator>>();
  auto comm_without_executor =
      std::make_unique<StrictMock<MockGpuCommunicator>>();
  EXPECT_CALL(*comm0, stream_executor()).WillRepeatedly(Return(&executor0));
  EXPECT_CALL(*comm1, stream_executor()).WillRepeatedly(Return(&executor1));
  EXPECT_CALL(*comm_without_executor, stream_executor())
      .WillRepeatedly(Return(nullptr));

  se::DeviceAddressBase addr(reinterpret_cast<void*>(0x1000), 1024);
  EXPECT_CALL(*comm1, ReleaseSymmetricMemory(addr));

  absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
  comms[RankId(0)] = std::move(comm0);
  comms[RankId(1)] = std::move(comm1);
  comms[RankId(2)] = std::move(comm_without_executor);
  GpuClique clique(
      GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1), GlobalDeviceId(2)},
                   /*num_local_participants=*/3),
      /*ids=*/std::nullopt, std::move(comms), /*peer_access_enabled=*/false,
      std::make_shared<CancellationToken>());

  clique.ReleaseSymmetricMemory(LocalDeviceId(1), addr);
}

}  // namespace
}  // namespace xla::gpu
