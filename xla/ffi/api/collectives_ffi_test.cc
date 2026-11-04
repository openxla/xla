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

#include "xla/ffi/api/collectives_ffi.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

static XLA_FFI_Communicator* const kFakeCommunicator =
    reinterpret_cast<XLA_FFI_Communicator*>(0xC0FFEE);
static XLA_FFI_Window* const kFakeWindow =
    reinterpret_cast<XLA_FFI_Window*>(0xDEC0DE);
static constexpr size_t kFakeWindowOffset = 42;

// Fake backend: implements the extension callbacks and records what the handler
// requested so the test can check the C++ wrapper builds the C args correctly.
struct FakeBackend {
  bool logical_device_info_called = false;
  bool request_called = false;
  XLA_FFI_CollectiveGroupMode requested_group_mode =
      XLA_FFI_GROUP_CROSS_REPLICA;
  size_t requested_num_groups = 0;
  int64_t requested_communication_id = -1;
  std::vector<int64_t> requested_ids;
  XLA_FFI_CollectiveGroupMode get_group_mode = XLA_FFI_GROUP_CROSS_REPLICA;
  int64_t get_communication_id = -1;
  std::vector<std::vector<int64_t>> get_groups;

  bool request_window_called = false;
  XLA_FFI_CollectiveGroupMode requested_window_group_mode =
      XLA_FFI_GROUP_CROSS_REPLICA;
  size_t requested_window_num_groups = 0;
  int64_t requested_window_communication_id = -1;
  std::vector<int64_t> requested_window_ids;
  std::vector<XLA_FFI_CollectiveMemoryRegion> requested_regions;
  const void* get_window_buffer = nullptr;
};

static XLA_FFI_Error* FakeGetLogicalDeviceInfo(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_LogicalDeviceInfo_Get_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->logical_device_info_called = true;
  EXPECT_EQ(args->struct_size, XLA_FFI_LogicalDeviceInfo_Get_Args_STRUCT_SIZE);
  EXPECT_EQ(args->extension_start, nullptr);
  args->replica_id = 1;
  args->partition_id = 2;
  args->replica_count = 3;
  args->partition_count = 4;
  return nullptr;
}

static XLA_FFI_Error* FakeRequestCommunicator(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Request_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->request_called = true;
  backend->requested_group_mode = args->group_mode;
  backend->requested_num_groups = args->num_groups;
  backend->requested_communication_id = args->communication_id;
  backend->requested_ids.clear();
  for (size_t i = 0; i < args->num_groups; ++i) {
    for (size_t j = 0; j < args->groups[i].size; ++j) {
      backend->requested_ids.push_back(args->groups[i].ids[j]);
    }
  }
  return nullptr;
}

static XLA_FFI_Error* FakeGetCommunicator(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Communicator_Get_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  EXPECT_EQ(args->struct_size, XLA_FFI_Communicator_Get_Args_STRUCT_SIZE);
  EXPECT_EQ(args->extension_start, nullptr);
  backend->get_group_mode = args->group_mode;
  backend->get_communication_id = args->communication_id;
  backend->get_groups.clear();
  for (size_t i = 0; i < args->num_groups; ++i) {
    backend->get_groups.emplace_back(
        args->groups[i].ids, args->groups[i].ids + args->groups[i].size);
  }
  args->communicator = kFakeCommunicator;
  args->rank = 2;
  args->size = 3;
  args->local_participant_count = 1;
  return nullptr;
}

static XLA_FFI_Error* FakeRequestWindow(
    const XLA_FFI_Collectives_Extension* self,
    XLA_FFI_Window_Request_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->request_window_called = true;
  backend->requested_window_group_mode = args->group_mode;
  backend->requested_window_num_groups = args->num_groups;
  backend->requested_window_communication_id = args->communication_id;
  backend->requested_window_ids.clear();
  for (size_t i = 0; i < args->num_groups; ++i) {
    for (size_t j = 0; j < args->groups[i].size; ++j) {
      backend->requested_window_ids.push_back(args->groups[i].ids[j]);
    }
  }
  backend->requested_regions.clear();
  for (size_t i = 0; i < args->num_regions; ++i) {
    backend->requested_regions.push_back(args->regions[i]);
  }
  return nullptr;
}

static XLA_FFI_Error* FakeGetWindow(const XLA_FFI_Collectives_Extension* self,
                                    XLA_FFI_Window_Get_Args* args) {
  auto* backend = reinterpret_cast<FakeBackend*>(self->state);
  backend->get_window_buffer = args->buffer;
  args->window = kFakeWindow;
  args->window_offset = kFakeWindowOffset;
  return nullptr;
}

// Builds a collectives extension for the fake backend. Mirrors the small
// builder each backend uses to publish the extension (see the GPU backend).
static XLA_FFI_Collectives_Extension MakeFakeCollectivesExtension(
    XLA_FFI_CollectivesState* state,
    XLA_FFI_Communicator_Request* request_communicator,
    XLA_FFI_Communicator_Get* get_communicator,
    XLA_FFI_Window_Request* request_window = nullptr,
    XLA_FFI_Window_Get* get_window = nullptr) {
  XLA_FFI_Collectives_Extension ext = {};
  ext.extension_base =
      MakeExtensionHeader<internal::CollectivesExtensionBase<void>>();
  ext.state = state;
  ext.request_communicator = request_communicator;
  ext.get_communicator = get_communicator;
  ext.request_window = request_window;
  ext.get_window = get_window;
  ext.get_logical_device_info = FakeGetLogicalDeviceInfo;
  return ext;
}

TEST(CollectivesFfiTest, ExtensionVersionCompatibility) {
  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [](Communicator comm) -> Error {
        return comm.RequestCommunicator(GroupMode::kCrossReplica, {},
                                        /*communication_id=*/0);
      });
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  const struct {
    int32_t major;
    int32_t minor;
    bool supported;
  } cases[] = {
      {Collectives::kMajorVersion, 2, true},
      {Collectives::kMajorVersion, Collectives::kMinorVersion, true},
      {Collectives::kMajorVersion, Collectives::kMinorVersion + 1, true},
      {Collectives::kMajorVersion, 1, false},
      {Collectives::kMajorVersion + 1, Collectives::kMinorVersion, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.major);
    SCOPED_TRACE(test_case.minor);
    FakeBackend backend;
    XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
        reinterpret_cast<XLA_FFI_CollectivesState*>(&backend),
        FakeRequestCommunicator, nullptr);
    ext.extension_base.id.major_version = test_case.major;
    ext.extension_base.id.minor_version = test_case.minor;
    InvokeContext context;
    context.extension_start = &ext.extension_base;

    absl::Status status = Invoke(Api(), *handler, call_frame, context);
    if (test_case.supported) {
      EXPECT_OK(status);
    } else {
      EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                                   HasSubstr("Extension version mismatch")));
    }
    EXPECT_EQ(backend.request_called, test_case.supported);
  }
}

TEST(CollectivesFfiTest, LogicalDeviceInfoWithoutResourceRequests) {
  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [](Communicator comm) -> Error {
        ErrorOr<LogicalDeviceInfo> info = comm.GetLogicalDeviceInfo();
        if (info.has_error()) {
          return info.error();
        }
        EXPECT_EQ(info->replica_id, 1);
        EXPECT_EQ(info->partition_id, 2);
        EXPECT_EQ(info->replica_count, 3);
        EXPECT_EQ(info->partition_count, 4);
        return Error::Success();
      });
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  FakeBackend backend;
  XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
      reinterpret_cast<XLA_FFI_CollectivesState*>(&backend),
      FakeRequestCommunicator, FakeGetCommunicator, FakeRequestWindow,
      FakeGetWindow);
  InvokeContext context;
  context.extension_start = &ext.extension_base;

  ASSERT_OK(Invoke(Api(), *handler, call_frame, context));
  EXPECT_TRUE(backend.logical_device_info_called);
  EXPECT_FALSE(backend.request_called);
  EXPECT_FALSE(backend.request_window_called);
}

TEST(CollectivesFfiTest, LogicalDeviceInfoRejectsOlderExtension) {
  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [](Communicator comm) -> Error {
        ErrorOr<LogicalDeviceInfo> info = comm.GetLogicalDeviceInfo();
        return info.has_error() ? info.error() : Error::Success();
      });
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  FakeBackend backend;
  XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
      reinterpret_cast<XLA_FFI_CollectivesState*>(&backend), nullptr, nullptr);
  ext.extension_base.id.minor_version = 2;
  ext.extension_base.struct_size =
      XLA_FFI_STRUCT_SIZE(XLA_FFI_Collectives_Extension, get_window);
  InvokeContext context;
  context.extension_start = &ext.extension_base;

  EXPECT_THAT(
      Invoke(Api(), *handler, call_frame, context),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Logical device information is not supported")));
  EXPECT_FALSE(backend.logical_device_info_called);
}

TEST(CollectivesFfiTest, LogicalDeviceInfoUnsupportedCallback) {
  for (bool truncated : {false, true}) {
    SCOPED_TRACE(truncated);
    FakeBackend backend;
    XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
        reinterpret_cast<XLA_FFI_CollectivesState*>(&backend), nullptr,
        nullptr);
    if (truncated) {
      ext.extension_base.struct_size =
          XLA_FFI_STRUCT_SIZE(XLA_FFI_Collectives_Extension, get_window);
    } else {
      ext.get_logical_device_info = nullptr;
    }

    ErrorOr<LogicalDeviceInfo> info =
        Communicator(Api(), &ext).GetLogicalDeviceInfo();
    ASSERT_TRUE(info.has_error());
    EXPECT_EQ(info.error().errc(), ErrorCode::kUnimplemented);
    EXPECT_THAT(info.error().message(), HasSubstr("not supported"));
    EXPECT_FALSE(backend.logical_device_info_called);
  }
}

TEST(CollectivesFfiTest, LogicalDeviceInfoPropagatesCallbackError) {
  XLA_FFI_Collectives_Extension ext =
      MakeFakeCollectivesExtension(nullptr, nullptr, nullptr);
  ext.get_logical_device_info = [](const XLA_FFI_Collectives_Extension*,
                                   XLA_FFI_LogicalDeviceInfo_Get_Args*) {
    return internal::CreateError(
        Api(), Error(ErrorCode::kFailedPrecondition,
                     "logical device assignment unavailable"));
  };

  ErrorOr<LogicalDeviceInfo> info =
      Communicator(Api(), &ext).GetLogicalDeviceInfo();
  ASSERT_TRUE(info.has_error());
  EXPECT_EQ(info.error().errc(), ErrorCode::kFailedPrecondition);
  EXPECT_EQ(info.error().message(), "logical device assignment unavailable");
}

TEST(CollectivesFfiTest, GetCommunicatorPropagatesCallbackError) {
  XLA_FFI_Collectives_Extension ext =
      MakeFakeCollectivesExtension(nullptr, nullptr, nullptr);
  ext.get_communicator = [](const XLA_FFI_Collectives_Extension*,
                            XLA_FFI_Communicator_Get_Args*) {
    return internal::CreateError(
        Api(), Error::InvalidArgument("invalid communicator group"));
  };

  ErrorOr<CommunicatorInfo> info =
      Communicator(Api(), &ext)
          .GetCommunicator(GroupMode::kCrossReplica, {},
                           /*communication_id=*/0);
  ASSERT_TRUE(info.has_error());
  EXPECT_EQ(info.error().errc(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(info.error().message(), "invalid communicator group");
}

TEST(CollectivesFfiTest, GetCommunicatorRequiresMetadataSupport) {
  XLA_FFI_Collectives_Extension ext =
      MakeFakeCollectivesExtension(nullptr, nullptr, nullptr);
  ext.extension_base.id.minor_version = 2;
  ext.get_communicator = [](const XLA_FFI_Collectives_Extension*,
                            XLA_FFI_Communicator_Get_Args* args) {
    args->communicator = kFakeCommunicator;
    return static_cast<XLA_FFI_Error*>(nullptr);
  };

  ErrorOr<CommunicatorInfo> info =
      Communicator(Api(), &ext)
          .GetCommunicator(GroupMode::kCrossReplica, {},
                           /*communication_id=*/0);
  ASSERT_TRUE(info.has_error());
  EXPECT_EQ(info.error().errc(), ErrorCode::kUnimplemented);
  EXPECT_THAT(
      info.error().message(),
      HasSubstr("Communicator metadata requires collectives extension"));
}

TEST(CollectivesFfiTest, RequestAndGetCommunicator) {
  bool called = false;
  CommunicatorInfo got = {};

  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [&](Communicator comm) -> Error {
        called = true;
        if (Error status = comm.RequestCommunicator(GroupMode::kFlattenedId,
                                                    {{5, 1, 3}, {2, 0, 4}},
                                                    /*communication_id=*/7);
            status.failure()) {
          return status;
        }
        ErrorOr<CommunicatorInfo> comm_or = comm.GetCommunicator(
            GroupMode::kFlattenedId, {{5, 1, 3}, {2, 0, 4}},
            /*communication_id=*/7);
        if (comm_or.has_error()) {
          return comm_or.error();
        }
        got = comm_or.value();
        return Error::Success();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  FakeBackend backend;
  XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
      reinterpret_cast<XLA_FFI_CollectivesState*>(&backend),
      FakeRequestCommunicator, FakeGetCommunicator);

  InvokeContext context;
  context.extension_start = &ext.extension_base;

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kExecute);

  ASSERT_OK(status);
  EXPECT_TRUE(called);
  EXPECT_EQ(got.communicator, kFakeCommunicator);
  EXPECT_EQ(got.rank, 2);
  EXPECT_EQ(got.size, 3);
  EXPECT_EQ(got.local_participant_count, 1);
  EXPECT_TRUE(backend.request_called);
  EXPECT_EQ(backend.requested_group_mode, XLA_FFI_GROUP_FLATTENED_ID);
  EXPECT_EQ(backend.requested_num_groups, 2u);
  EXPECT_EQ(backend.requested_communication_id, 7);
  EXPECT_THAT(backend.requested_ids, ElementsAre(5, 1, 3, 2, 0, 4));
  EXPECT_EQ(backend.get_group_mode, XLA_FFI_GROUP_FLATTENED_ID);
  EXPECT_EQ(backend.get_communication_id, 7);
  EXPECT_THAT(backend.get_groups,
              ElementsAre(ElementsAre(5, 1, 3), ElementsAre(2, 0, 4)));
}

TEST(CollectivesFfiTest, RequestAndGetWindow) {
  static uint32_t fake_buffer0 = 0;
  static uint32_t fake_buffer1 = 0;
  const std::vector<CollectiveMemoryRegion> regions = {
      {&fake_buffer0, sizeof(fake_buffer0)},
      {&fake_buffer1, sizeof(fake_buffer1)},
  };

  bool called = false;
  WindowLookup got_lookup = {nullptr, 0};

  auto handler = Ffi::Bind().Ctx<Extension<Collectives>>().To(
      [&](Communicator comm) -> Error {
        called = true;
        if (Error status = comm.RequestWindow(GroupMode::kFlattenedId, {{0, 1}},
                                              /*communication_id=*/9, regions);
            status.failure()) {
          return status;
        }
        ErrorOr<WindowLookup> lookup_or =
            comm.GetWindow(GroupMode::kFlattenedId, {{0, 1}},
                           /*communication_id=*/9, &fake_buffer0);
        if (lookup_or.has_error()) {
          return lookup_or.error();
        }
        got_lookup = lookup_or.value();
        return Error::Success();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  FakeBackend backend;
  XLA_FFI_Collectives_Extension ext = MakeFakeCollectivesExtension(
      reinterpret_cast<XLA_FFI_CollectivesState*>(&backend),
      FakeRequestCommunicator, FakeGetCommunicator, FakeRequestWindow,
      FakeGetWindow);

  InvokeContext context;
  context.extension_start = &ext.extension_base;

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kExecute);

  ASSERT_OK(status);
  EXPECT_TRUE(called);
  EXPECT_TRUE(backend.request_window_called);
  EXPECT_EQ(backend.requested_window_group_mode, XLA_FFI_GROUP_FLATTENED_ID);
  EXPECT_EQ(backend.requested_window_num_groups, 1u);
  EXPECT_EQ(backend.requested_window_communication_id, 9);
  EXPECT_THAT(backend.requested_window_ids, ElementsAre(0, 1));
  ASSERT_EQ(backend.requested_regions.size(), 2u);
  EXPECT_EQ(backend.requested_regions[0].buffer, &fake_buffer0);
  EXPECT_EQ(backend.requested_regions[0].byte_size, sizeof(fake_buffer0));
  EXPECT_EQ(backend.requested_regions[1].buffer, &fake_buffer1);
  EXPECT_EQ(backend.requested_regions[1].byte_size, sizeof(fake_buffer1));
  EXPECT_EQ(backend.get_window_buffer, &fake_buffer0);
  EXPECT_EQ(got_lookup.window, kFakeWindow);
  EXPECT_EQ(got_lookup.offset, kFakeWindowOffset);
}

}  // namespace
}  // namespace xla::ffi
