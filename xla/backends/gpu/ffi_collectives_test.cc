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

#include "xla/backends/gpu/ffi_collectives.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/mock_gpu_collectives.h"
#include "xla/backends/gpu/collectives/mock_gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

struct ResourceState {
  int created = 0;
  int retired = 0;
  XLA_FFI_Communicator_Resource_State retirement_state =
      XLA_FFI_COMMUNICATOR_RESOURCE_HEALTHY;
};

XLA_FFI_Error* CreateResource(XLA_FFI_Communicator_Resource_Create_Args* args) {
  EXPECT_EQ(args->struct_size,
            XLA_FFI_Communicator_Resource_Create_Args_STRUCT_SIZE);
  EXPECT_EQ(args->extension_start, nullptr);
  EXPECT_EQ(args->resource, nullptr);
  EXPECT_NE(args->api, nullptr);
  auto* state = static_cast<ResourceState*>(args->user_data);
  ++state->created;
  args->resource = state;
  return nullptr;
}

void RetireResource(XLA_FFI_Communicator_Resource_Retire_Args* args) {
  EXPECT_EQ(args->struct_size,
            XLA_FFI_Communicator_Resource_Retire_Args_STRUCT_SIZE);
  EXPECT_EQ(args->extension_start, nullptr);
  auto* state = static_cast<ResourceState*>(args->resource);
  ++state->retired;
  state->retirement_state = args->state;
}

class CommunicatorResourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    ON_CALL(executor_, GetPlatform()).WillByDefault(Return(&platform_));
    ON_CALL(platform_, Name()).WillByDefault(ReturnRef(platform_name_));
    gpu_options_.set_collectives(&collectives_);
    assignment_(0, 0) = 0;
    run_options_.mutable_run_options()->set_stream(&stream_);
    run_options_.mutable_run_options()->set_device_assignment(&assignment_);
    run_options_.mutable_run_options()->set_gpu_executable_run_options(
        &gpu_options_);
    ASSERT_OK_AND_ASSIGN(auto params, CollectiveParams::Create(
                                          run_options_, {}, LocalDeviceId(0)));
    params_.emplace(std::move(params));
    absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
    comms[RankId(0)] = std::make_unique<NiceMock<MockGpuCommunicator>>();
    GpuCliqueKey key({GlobalDeviceId(0)}, 1);
    clique_ = std::make_unique<LockableGpuClique>(
        key, std::nullopt, std::move(comms), false,
        std::make_shared<CancellationToken>());
    AcquiredCliquesMap acquired;
    acquired.emplace(
        key, std::make_shared<LockableGpuClique::Lock>(clique_->Acquire()));
    cliques_ = CollectiveCliques(std::move(acquired));
    state_.collective_params = &*params_;
    state_.collective_cliques = &cliques_;
    ext_ = MakeCollectivesExtension(&state_);
    args_.struct_size = XLA_FFI_Communicator_Resource_Get_Args_STRUCT_SIZE;
    args_.group_mode = XLA_FFI_GROUP_CROSS_REPLICA;
    args_.name = name_.data();
    args_.name_size = name_.size();
    args_.user_data = &resource_;
    args_.create = CreateResource;
    args_.retire = RetireResource;
  }

  absl::Status GetResource() {
    return ffi::TakeError(ext_.get_communicator_resource(&ext_, &args_));
  }

  void DestroyClique() {
    state_.collective_cliques = nullptr;
    cliques_ = CollectiveCliques();
    clique_.reset();
  }

  std::string platform_name_ = "test";
  std::string name_ = "test.resource.v1";
  NiceMock<se::MockPlatform> platform_;
  NiceMock<se::MockStreamExecutor> executor_;
  NiceMock<se::MockStream> stream_;
  NiceMock<MockGpuCollectives> collectives_;
  DeviceAssignment assignment_{1, 1};
  GpuExecutableRunOptions gpu_options_;
  ServiceExecutableRunOptions run_options_;
  std::optional<CollectiveParams> params_;
  ResourceState resource_;
  std::unique_ptr<LockableGpuClique> clique_;
  CollectiveCliques cliques_;
  GpuCollectivesState state_;
  XLA_FFI_Collectives_Extension ext_ = {};
  XLA_FFI_Communicator_Resource_Get_Args args_ = {};
};

TEST_F(CommunicatorResourceTest, ReusesStateAndRetiresThroughCAbi) {
  ASSERT_OK(GetResource());
  EXPECT_EQ(args_.resource, &resource_);
  ResourceState unused;
  args_.user_data = &unused;
  ASSERT_OK(GetResource());
  EXPECT_EQ(args_.resource, &resource_);
  EXPECT_EQ(resource_.created, 1);
  EXPECT_EQ(unused.created, 0);
  EXPECT_EQ(resource_.retired, 0);
  DestroyClique();
  EXPECT_EQ(resource_.retired, 1);
  EXPECT_EQ(resource_.retirement_state, XLA_FFI_COMMUNICATOR_RESOURCE_HEALTHY);
}

TEST_F(CommunicatorResourceTest, RejectsMismatchedCallbacks) {
  ASSERT_OK(GetResource());
  args_.retire = +[](XLA_FFI_Communicator_Resource_Retire_Args*) {};
  EXPECT_THAT(GetResource(), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("different callbacks")));
  EXPECT_EQ(args_.resource, nullptr);
  DestroyClique();
  EXPECT_EQ(resource_.retired, 1);
}

TEST_F(CommunicatorResourceTest,
       PropagatesCreationErrorsAndRejectsNullResources) {
  args_.create = +[](XLA_FFI_Communicator_Resource_Create_Args* args) {
    XLA_FFI_Error_Create_Args error = {};
    error.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    error.errc = XLA_FFI_Error_Code_RESOURCE_EXHAUSTED;
    error.message = "factory failure";
    return args->api->XLA_FFI_Error_Create(&error);
  };
  EXPECT_THAT(GetResource(), StatusIs(absl::StatusCode::kResourceExhausted,
                                      HasSubstr("factory failure")));
  args_.create =
      +[](XLA_FFI_Communicator_Resource_Create_Args*) -> XLA_FFI_Error* {
    return nullptr;
  };
  EXPECT_THAT(GetResource(), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("returned null")));
  args_.create = CreateResource;
  EXPECT_OK(GetResource());
  EXPECT_EQ(resource_.created, 1);
}

TEST_F(CommunicatorResourceTest,
       CancelledGroupRejectsLookupAndRetiresUnhealthy) {
  ASSERT_OK(GetResource());
  clique_->Cancel();
  EXPECT_THAT(GetResource(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("cancelled")));
  DestroyClique();
  EXPECT_EQ(resource_.retired, 1);
  EXPECT_EQ(resource_.retirement_state,
            XLA_FFI_COMMUNICATOR_RESOURCE_UNHEALTHY);
}

TEST_F(CommunicatorResourceTest, ValidatesStageArgumentsAndAcquiredGroup) {
  args_.struct_size = 0;
  EXPECT_THAT(GetResource(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("size")));
  args_.struct_size = XLA_FFI_Communicator_Resource_Get_Args_STRUCT_SIZE;
  args_.retire = nullptr;
  EXPECT_THAT(GetResource(), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("required")));
  args_.retire = RetireResource;
  args_.communication_id = 7;
  EXPECT_THAT(GetResource(),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("No clique")));
  args_.communication_id = 0;
  state_.collective_cliques = nullptr;
  EXPECT_THAT(GetResource(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("acquired")));
  EXPECT_EQ(resource_.created, 0);
}

}  // namespace
}  // namespace xla::gpu
