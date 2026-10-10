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
#include "xla/runtime/device_id.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(SortedTransferIncarnationsTest, EmptySnapshotReturnsNoIncarnations) {
  ASSERT_OK_AND_ASSIGN(
      std::vector<IncarnationId> transfer_incarnations,
      SortedTransferIncarnations(/*src_process_index=*/1,
                                 /*dst_process_index=*/2, {}));
  EXPECT_THAT(transfer_incarnations, IsEmpty());
}

TEST(SortedTransferIncarnationsTest, SnapshotReturnsSortedIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      std::vector<IncarnationId> forward,
      SortedTransferIncarnations(/*src_process_index=*/2,
                                 /*dst_process_index=*/1, incarnations));
  EXPECT_THAT(forward, ElementsAre(IncarnationId(10), IncarnationId(20)));

  ASSERT_OK_AND_ASSIGN(
      std::vector<IncarnationId> swapped,
      SortedTransferIncarnations(/*src_process_index=*/1,
                                 /*dst_process_index=*/2, incarnations));
  EXPECT_THAT(swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(SortedTransferIncarnationsTest, SameTaskUsesOneIncarnation) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      std::vector<IncarnationId> transfer_incarnations,
      SortedTransferIncarnations(/*src_process_index=*/3,
                                 /*dst_process_index=*/3, incarnations));
  EXPECT_THAT(transfer_incarnations, ElementsAre(IncarnationId(7)));
}

TEST(SortedTransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(SortedTransferIncarnations(/*src_process_index=*/1,
                                         /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace xla
