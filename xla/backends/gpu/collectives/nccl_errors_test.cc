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

#include "xla/backends/gpu/collectives/nccl_errors.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(NcclErrorsTest, ReportsQueryFailureWithUnderlyingError) {
  CancellationToken cancel;
  // Inject ncclSystemError from the status query to verify query-failure
  // reporting.
  auto query = [](ncclComm_t, ncclResult_t* state) {
    *state = ncclInProgress;
    return ncclSystemError;
  };

  EXPECT_THAT(internal::PollUntilDoneWithQuery(nullptr, cancel, query),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("ncclCommGetAsyncError failed"),
                             HasSubstr(ncclGetErrorString(ncclSystemError)))));
}

TEST(NcclErrorsTest, DistinguishesAsyncOperationErrorFromQueryFailure) {
  CancellationToken cancel;
  auto query = [](ncclComm_t, ncclResult_t* state) {
    *state = ncclRemoteError;
    return ncclSuccess;
  };

  EXPECT_THAT(internal::PollUntilDoneWithQuery(nullptr, cancel, query),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr(ncclGetErrorString(ncclRemoteError)),
                             Not(HasSubstr("ncclCommGetAsyncError failed")))));
}

}  // namespace
}  // namespace xla::gpu
