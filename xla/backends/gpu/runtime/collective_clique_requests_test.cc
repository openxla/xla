/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_clique_requests.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(CollectiveCliqueRequestsTest, OrderedRequests) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);
  GlobalDeviceId d2 = GlobalDeviceId(2);
  GlobalDeviceId d3 = GlobalDeviceId(3);

  GpuCliqueKey k0({d2, d3}, 2);
  GpuCliqueKey k1({d0, d1}, 2);
  GpuCliqueKey k2({d0, d1, d2, d3}, 4);

  std::vector<std::vector<GlobalDeviceId>> rg0 = {{d2, d3}};
  std::vector<std::vector<GlobalDeviceId>> rg1 = {{d0, d1}};
  std::vector<std::vector<GlobalDeviceId>> rg2 = {{d0, d1, d2, d3}};

  CollectiveCliqueRequests requests;
  ASSERT_OK(requests.RequestClique(k0, rg0));
  ASSERT_OK(requests.RequestClique(k1, rg1));
  ASSERT_OK(requests.RequestClique(k2, rg2));

  // Check that we acquire larger cliques first, and then cliques with smaller
  // id first, as acquiring cliques according to natural clique key order might
  // lead to deadlocks during communicator splitting.
  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 3);
  EXPECT_EQ(ordered_requests[0].key, k2);
  EXPECT_EQ(ordered_requests[1].key, k0);
  EXPECT_EQ(ordered_requests[2].key, k1);
}

TEST(CollectiveCliqueRequestsTest, RequestDevComms) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);

  GpuCliqueKey k0({d0, d1}, 2);

  std::vector<std::vector<GlobalDeviceId>> rg0 = {{d0, d1}};

  GpuDeviceCommunicator::Requirements dev_comm0{8};
  GpuDeviceCommunicator::Requirements dev_comm1{16};

  CollectiveCliqueRequests requests;
  ASSERT_OK(requests.RequestClique(k0, rg0, {dev_comm0}));
  ASSERT_OK(requests.RequestClique(k0, rg0, {dev_comm1}));

  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 1);
  EXPECT_EQ(ordered_requests[0].key, k0);
  ASSERT_EQ(ordered_requests[0].dev_comms.size(), 2);
  EXPECT_TRUE(ordered_requests[0].dev_comms.contains(dev_comm0));
  EXPECT_TRUE(ordered_requests[0].dev_comms.contains(dev_comm1));
}

TEST(CollectiveCliqueRequestsTest, DeviceGroupsMismatch) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);
  GlobalDeviceId d2 = GlobalDeviceId(2);
  GlobalDeviceId d3 = GlobalDeviceId(3);

  GpuCliqueKey k0({d0, d1}, 2);

  // Callers must pass pre-sorted device groups.
  std::vector<std::vector<GlobalDeviceId>> dg0a = {{d0, d1}};
  std::vector<std::vector<GlobalDeviceId>> dg0b = {{d0, d1}, {d2, d3}};

  CollectiveCliqueRequests requests;
  ASSERT_OK(requests.RequestClique(k0, dg0a));

  ASSERT_THAT(
      requests.RequestClique(k0, dg0b),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("different device groups")));
}

TEST(CollectiveCliqueRequestsTest, SingletonDeviceGroupsMismatchAllowed) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);
  GlobalDeviceId d2 = GlobalDeviceId(2);
  GlobalDeviceId d3 = GlobalDeviceId(3);

  // Singleton clique key (single device, no cross-device communication).
  GpuCliqueKey k_singleton({d0}, /*num_local_participants=*/1);

  // Same singleton key requested with different surrounding device groups, as
  // produced by connected-component collective-permute on different
  // instructions. This must be tolerated (regression test for the clique
  // acquisition deadlock).
  std::vector<std::vector<GlobalDeviceId>> dg_a = {{d0}, {d1, d2, d3}};
  std::vector<std::vector<GlobalDeviceId>> dg_b = {{d0}, {d1}, {d2, d3}};

  CollectiveCliqueRequests requests;
  ASSERT_OK(requests.RequestClique(k_singleton, dg_a));
  EXPECT_FALSE(requests.HasMixedDeviceGroups());
  EXPECT_OK(requests.RequestClique(k_singleton, dg_b));
  EXPECT_TRUE(requests.HasMixedDeviceGroups());

  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 1);
  EXPECT_EQ(ordered_requests[0].key, k_singleton);
}

TEST(CollectiveCliqueRequestsTest, UniformDeviceGroups) {
  GlobalDeviceId d0(0), d1(1), d2(2), d3(3);
  std::vector<std::vector<GlobalDeviceId>> groups = {{d0, d1}, {d2, d3}};
  GpuCliqueKey key({d0, d1}, /*num_local_participants=*/2);

  CollectiveCliqueRequests requests;
  EXPECT_FALSE(requests.HasMixedDeviceGroups());
  ASSERT_OK(requests.RequestClique(key, groups));
  std::string signature = requests.CollectiveGraphSignature();
  ASSERT_OK(requests.RequestClique(key, groups));
  EXPECT_EQ(requests.CollectiveGraphSignature(), signature);
  EXPECT_FALSE(requests.HasMixedDeviceGroups());

  GpuCliqueKey other_communication({d0, d1}, /*num_local_participants=*/2,
                                   CommunicationId(1));
  ASSERT_OK(requests.RequestClique(other_communication, groups));
  EXPECT_NE(requests.CollectiveGraphSignature(), signature);
  EXPECT_FALSE(requests.HasMixedDeviceGroups());
}

TEST(CollectiveCliqueRequestsTest, GraphSignatureIsRankIndependent) {
  GlobalDeviceId d0(0), d1(1), d2(2), d3(3);
  std::vector<std::vector<GlobalDeviceId>> groups_a = {{d0}, {d1, d2, d3}};
  std::vector<std::vector<GlobalDeviceId>> groups_b = {{d0}, {d1}, {d2, d3}};

  CollectiveCliqueRequests rank0;
  GpuCliqueKey singleton({d0}, /*num_local_participants=*/1);
  ASSERT_OK(rank0.RequestClique(singleton, groups_a));
  std::string signature = rank0.CollectiveGraphSignature();
  ASSERT_OK(rank0.RequestClique(singleton, groups_b));
  EXPECT_NE(rank0.CollectiveGraphSignature(), signature);
  EXPECT_EQ(rank0.size(), 1);

  CollectiveCliqueRequests rank1;
  ASSERT_OK(rank1.RequestClique(
      GpuCliqueKey({d1, d2, d3}, /*num_local_participants=*/2), groups_a));
  ASSERT_OK(rank1.RequestClique(
      GpuCliqueKey({d1}, /*num_local_participants=*/1), groups_b));
  EXPECT_EQ(rank1.size(), 2);
  EXPECT_TRUE(rank1.HasMixedDeviceGroups());
  EXPECT_EQ(rank0.CollectiveGraphSignature(), rank1.CollectiveGraphSignature());
}

TEST(CollectiveCliqueRequestsTest, GraphParticipantsFollowTransitiveGroups) {
  GlobalDeviceId d0(0), d1(1), d2(2), d3(3), d4(4), d5(5), d6(6), d7(7);
  std::vector<std::vector<GlobalDeviceId>> first = {
      {d0}, {d1, d2}, {d3}, {d4, d5}, {d6}, {d7}};
  std::vector<std::vector<GlobalDeviceId>> second = {
      {d0, d1}, {d2, d3}, {d4}, {d5, d6}, {d7}};
  CollectiveCliqueRequests requests;
  ASSERT_OK(requests.RequestClique(GpuCliqueKey({d0}, 1), first));
  ASSERT_OK(requests.RequestClique(GpuCliqueKey({d0, d1}, 2), second));

  EXPECT_THAT(requests.CollectiveGraphParticipants(d0),
              ElementsAre(d0, d1, d2, d3));
  EXPECT_THAT(requests.CollectiveGraphParticipants(d3),
              ElementsAre(d0, d1, d2, d3));
  EXPECT_THAT(requests.CollectiveGraphParticipants(d4), ElementsAre(d4, d5, d6));
  EXPECT_THAT(requests.CollectiveGraphParticipants(d7), ElementsAre(d7));
}

TEST(CollectiveCliqueRequestsTest, GraphParticipantsWithoutRequests) {
  CollectiveCliqueRequests requests;
  EXPECT_THAT(requests.CollectiveGraphParticipants(GlobalDeviceId(3)),
              ElementsAre(GlobalDeviceId(3)));
}

TEST(CollectiveCliqueRequestsTest, RejectEmptyDeviceGroups) {
  CollectiveCliqueRequests requests;
  EXPECT_THAT(
      requests.RequestClique(GpuCliqueKey({GlobalDeviceId(0)}, 1), {}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("nonempty device group")));
}

TEST(CollectiveCliqueRequestsTest, DeviceGroupsLexicographicSort) {
  GlobalDeviceId d0(0), d1(1), d2(2), d3(3), d4(4), d5(5);

  // Just a sanity check for our DCHECKs in clique requests.S
  std::vector<std::vector<GlobalDeviceId>> groups = {
      {d4, d5},
      {d0, d1},
      {d2, d3},
  };
  absl::c_sort(groups);

  EXPECT_EQ(groups[0], (std::vector{d0, d1}));
  EXPECT_EQ(groups[1], (std::vector{d2, d3}));
  EXPECT_EQ(groups[2], (std::vector{d4, d5}));
}

TEST(CollectiveCliqueRequestsTest, RequestCliqueSemaphore) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);

  GpuCliqueKey k0({d0, d1}, 2);
  std::vector<std::vector<GlobalDeviceId>> rg0 = {{d0, d1}};

  CollectiveCliqueRequests::BarrierRequirements barrier_reqs;
  barrier_reqs.use_barrier_semaphores = true;

  CollectiveCliqueRequests requests;
  ASSERT_OK(
      requests.RequestClique(k0, rg0,
                             CollectiveCliqueRequests::CliqueRequirements{
                                 /*dev_comm=*/std::nullopt, barrier_reqs}));

  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 1);
  EXPECT_TRUE(ordered_requests[0].use_barrier_semaphores);
  EXPECT_FALSE(ordered_requests[0].use_cross_device_barrier_requested);
}

TEST(CollectiveCliqueRequestsTest, RequestCliqueSemaphoreMultipleRequests) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);

  GpuCliqueKey k0({d0, d1}, 2);
  std::vector<std::vector<GlobalDeviceId>> rg0 = {{d0, d1}};

  CollectiveCliqueRequests::BarrierRequirements barrier_reqs1;
  barrier_reqs1.use_cross_device_barrier = true;

  CollectiveCliqueRequests::BarrierRequirements barrier_reqs2;
  barrier_reqs2.use_barrier_semaphores = true;

  CollectiveCliqueRequests requests;
  ASSERT_OK(
      requests.RequestClique(k0, rg0,
                             CollectiveCliqueRequests::CliqueRequirements{
                                 /*dev_comm=*/std::nullopt, barrier_reqs1}));
  ASSERT_OK(
      requests.RequestClique(k0, rg0,
                             CollectiveCliqueRequests::CliqueRequirements{
                                 /*dev_comm=*/std::nullopt, barrier_reqs2}));

  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 1);
  EXPECT_TRUE(ordered_requests[0].use_barrier_semaphores);
  EXPECT_TRUE(ordered_requests[0].use_cross_device_barrier_requested);
}

}  // namespace
}  // namespace xla::gpu
