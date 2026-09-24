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

#include "xla/pjrt/common_pjrt_client.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/runtime/device_id.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::xla::gpu::CommunicationId;
using ::xla::gpu::CrossHostTransferCliqueKey;
using ::xla::gpu::GpuCliqueKey;

absl::StatusOr<gpu::GpuCliqueKey> TransferKey(
    int src_process_index, int dst_process_index, GlobalDeviceId src_device,
    GlobalDeviceId dst_device,
    const absl::flat_hash_map<int, IncarnationId>& incarnations) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<IncarnationId> transfer_incarnations,
      SortedTransferIncarnations(src_process_index, dst_process_index,
                                 incarnations));
  return CrossHostTransferCliqueKey(src_device, dst_device,
                                    transfer_incarnations);
}

TEST(SortedTransferIncarnationsTest, EmptySnapshotKeepsHistoricalCliqueKey) {
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuCliqueKey key,
      TransferKey(/*src_process_index=*/1, /*dst_process_index=*/2,
                  GlobalDeviceId(4), GlobalDeviceId(5), {}));

  GpuCliqueKey historical({GlobalDeviceId(4), GlobalDeviceId(5)},
                          /*num_local_participants=*/1);
  EXPECT_EQ(key, historical);
  EXPECT_THAT(key.incarnations(), IsEmpty());
}

TEST(SortedTransferIncarnationsTest, SnapshotChangesCliqueKey) {
  absl::flat_hash_map<int, IncarnationId> incarnations = {
      {2, IncarnationId(20)},
      {1, IncarnationId(10)},
  };
  const GlobalDeviceId src(9);
  const GlobalDeviceId dst(4);

  ASSERT_OK_AND_ASSIGN(
      gpu::GpuCliqueKey key,
      TransferKey(/*src_process_index=*/2, /*dst_process_index=*/1, src, dst,
                  incarnations));
  GpuCliqueKey historical({src, dst}, /*num_local_participants=*/1);

  EXPECT_NE(key, historical);
  EXPECT_THAT(key.devices(), ElementsAre(src, dst));
  EXPECT_EQ(key.num_local_participants(), 1);
  EXPECT_EQ(key.communication_id(), CommunicationId(0));
  EXPECT_THAT(key.incarnations(),
              ElementsAre(IncarnationId(10), IncarnationId(20)));

  ASSERT_OK_AND_ASSIGN(
      gpu::GpuCliqueKey swapped,
      TransferKey(/*src_process_index=*/1, /*dst_process_index=*/2, dst, src,
                  incarnations));
  EXPECT_THAT(swapped.incarnations(),
              ElementsAre(IncarnationId(10), IncarnationId(20)));
  EXPECT_THAT(swapped.devices(), ElementsAre(dst, src));
}

TEST(SortedTransferIncarnationsTest, SameTaskUsesOneIncarnation) {
  absl::flat_hash_map<int, IncarnationId> incarnations = {
      {3, IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      gpu::GpuCliqueKey key,
      TransferKey(/*src_process_index=*/3, /*dst_process_index=*/3,
                  GlobalDeviceId(1), GlobalDeviceId(2), incarnations));

  EXPECT_THAT(key.incarnations(), ElementsAre(IncarnationId(7)));
  EXPECT_NE(key, GpuCliqueKey({GlobalDeviceId(1), GlobalDeviceId(2)},
                              /*num_local_participants=*/1));
}

TEST(SortedTransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<int, IncarnationId> incarnations = {
      {1, IncarnationId(10)},
  };

  EXPECT_THAT(SortedTransferIncarnations(/*src_process_index=*/1,
                                         /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace xla
