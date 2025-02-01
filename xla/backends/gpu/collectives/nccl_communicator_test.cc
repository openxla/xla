/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/nccl_communicator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/utility/utility.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

// Creates a non-blocking NCCL communicator.
absl::StatusOr<NcclCommunicator> CreateNonBlockingCommunicator() {
  // Create a unique NCCL Id.
  ncclUniqueId id;
  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclGetUniqueId(&id)));

  // Initialize a communicator.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;  // non-blocking
  ncclComm_t comm;
  ncclResult_t r =
      ncclCommInitRankConfig(&comm, /*nranks=*/1, id, /*rank=*/0, &config);
  if (r != ncclSuccess && r != ncclInProgress) {
    return XLA_NCCL_STATUS(r);
  }

  // Wait for the communicator to finish initializing.
  ncclResult_t state = ncclInProgress;
  while (state == ncclInProgress) {
    TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclCommGetAsyncError(comm, &state)));
  }
  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(state));

  // Wrap and return the communicator.
  return absl::StatusOr<NcclCommunicator>(absl::in_place_t(), comm);
}

TEST(NcclCommunicator, AbortSucceeds) {
  absl::StatusOr<NcclCommunicator> comm = CreateNonBlockingCommunicator();
  TF_ASSERT_OK(comm.status());
  TF_ASSERT_OK(comm->Abort());
}

TEST(NcclCommunicator, DoubleAbortFails) {
  absl::StatusOr<NcclCommunicator> comm = CreateNonBlockingCommunicator();
  TF_ASSERT_OK(comm.status());
  TF_ASSERT_OK(comm->Abort());
  ASSERT_THAT(comm->Abort(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("aborted")));
}

TEST(NcclCommunicator, OperationAfterAbortFails) {
  absl::StatusOr<NcclCommunicator> comm = CreateNonBlockingCommunicator();
  TF_ASSERT_OK(comm.status());
  TF_ASSERT_OK(comm->Abort());
  ASSERT_THAT(comm->NumRanks(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("aborted")));
}

}  // namespace
}  // namespace xla::gpu
