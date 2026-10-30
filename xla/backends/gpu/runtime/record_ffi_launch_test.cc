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

#include <cstdint>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/record_ffi.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/mock_command_buffer.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::A;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Optional;
using ::testing::Return;
using ::testing::StrictMock;

XLA_FFI_LaunchConfig LaunchConfig() {
  return {XLA_FFI_LaunchConfig_STRUCT_SIZE,
          {{8, 1, 1}, {32, 1, 1}, {2, 1, 1}},
          64,
          1,
          1,
          {4, 1, 1}};
}

class RecordFfiLaunchTest : public ::testing::Test {
 protected:
  absl::Status Create(const XLA_FFI_LaunchConfig& config) {
    return ffi::TakeError(api_->create_launch_with_config(
        &context_, "test", &command_, 0, XLA_FFI_SourceFormat_FUNCTION_PTR,
        &config, &args_, nullptr, 0, &ffi_command_));
  }

  absl::Status Update(const XLA_FFI_LaunchConfig& config) {
    return ffi::TakeError(api_->update_launch_with_config(
        &context_, ffi_command_, &config, &args_));
  }

  absl::Status UpdateArguments() {
    return ffi::TakeError(api_->update_launch(&context_, ffi_command_, &args_));
  }

  struct TestCommand : se::CommandBuffer::Command {
  } command_;
  StrictMock<se::MockCommandBuffer> command_buffer_;
  NiceMock<se::MockStreamExecutor> executor_;
  XLA_FFI_RecordContext context_{&command_buffer_, &executor_, {}, true};
  XLA_FFI_KernelArgs args_{nullptr, 0};
  const XLA_FFI_RecordApi* api_ = GetXlaFfiRecordApi();
  const XLA_FFI_Command* ffi_command_ = nullptr;
};

TEST_F(RecordFfiLaunchTest, ForwardsLaunchConfiguration) {
  int32_t value = 42;
  XLA_FFI_KernelArg arg{&value, sizeof(value), XLA_FFI_KernelArgType_HostValue};
  args_ = {&arg, 1};
  EXPECT_CALL(command_buffer_,
              CreateLaunch(se::ThreadDim(32), se::BlockDim(8),
                           Optional(se::ClusterDim(2)),
                           A<const se::NativeKernel&>(), _, _, _))
      .WillOnce([&](const auto&, const auto&, const auto&,
                    const se::NativeKernel& kernel,
                    const se::KernelArgsPackedArrayBase& args, auto,
                    auto) -> absl::StatusOr<const se::CommandBuffer::Command*> {
        EXPECT_EQ(kernel.device_fn, &command_);
        EXPECT_EQ(kernel.name(), "test");
        EXPECT_TRUE(kernel.pdl);
        EXPECT_TRUE(kernel.cooperative);
        EXPECT_THAT(kernel.preferred_cluster_dims, Optional(se::ClusterDim(4)));
        EXPECT_EQ(args.number_of_shared_bytes(), 64);
        EXPECT_EQ(*static_cast<const int32_t*>(args.argument_addresses()[0]),
                  42);
        return &command_;
      });
  ASSERT_OK(Create(LaunchConfig()));
  EXPECT_EQ(ffi_command_, reinterpret_cast<const XLA_FFI_Command*>(&command_));
}

TEST_F(RecordFfiLaunchTest, UpdatesDimensionsAndSharedMemory) {
  EXPECT_CALL(command_buffer_,
              CreateLaunch(_, _, _, A<const se::NativeKernel&>(), _, _, _))
      .WillOnce(Return(&command_));
  XLA_FFI_LaunchConfig config = LaunchConfig();
  ASSERT_OK(Create(config));
  config.launch_dims.grid.x = 16;
  config.launch_dims.block.x = 64;
  config.shared_mem_bytes = 128;
  int32_t value = 42;
  XLA_FFI_KernelArg arg{&value, sizeof(value), XLA_FFI_KernelArgType_HostValue};
  args_ = {&arg, 1};
  EXPECT_CALL(command_buffer_,
              UpdateLaunch(&command_, se::ThreadDim(64), se::BlockDim(16),
                           Optional(se::ClusterDim(2)),
                           A<const se::NativeKernel&>(), _))
      .Times(2)
      .WillRepeatedly([](auto, const auto&, const auto&, const auto&,
                         const se::NativeKernel& kernel,
                         const se::KernelArgsPackedArrayBase& args) {
        EXPECT_TRUE(kernel.cooperative);
        EXPECT_TRUE(kernel.pdl);
        EXPECT_THAT(kernel.preferred_cluster_dims, Optional(se::ClusterDim(4)));
        EXPECT_EQ(args.number_of_shared_bytes(), 128);
        EXPECT_EQ(*static_cast<const int32_t*>(args.argument_addresses()[0]),
                  42);
        return absl::OkStatus();
      });
  ASSERT_OK(Update(config));
  EXPECT_OK(UpdateArguments());
}

TEST_F(RecordFfiLaunchTest, FailedUpdatePreservesCachedConfiguration) {
  EXPECT_CALL(command_buffer_,
              CreateLaunch(_, _, _, A<const se::NativeKernel&>(), _, _, _))
      .WillOnce(Return(&command_));
  XLA_FFI_LaunchConfig config = LaunchConfig();
  ASSERT_OK(Create(config));
  config.launch_dims.grid.x = 16;
  config.shared_mem_bytes = 128;
  EXPECT_CALL(command_buffer_, UpdateLaunch(&command_, _, se::BlockDim(16), _,
                                            A<const se::NativeKernel&>(), _))
      .WillOnce(Return(absl::InternalError("driver rejected update")));
  EXPECT_THAT(Update(config), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("driver rejected update")));
  EXPECT_CALL(command_buffer_,
              UpdateLaunch(&command_, se::ThreadDim(32), se::BlockDim(8), _,
                           A<const se::NativeKernel&>(), _))
      .WillOnce([](auto, const auto&, const auto&, const auto&, const auto&,
                   const se::KernelArgsPackedArrayBase& args) {
        EXPECT_EQ(args.number_of_shared_bytes(), 64);
        return absl::OkStatus();
      });
  EXPECT_OK(UpdateArguments());
}

TEST_F(RecordFfiLaunchTest, RejectsLaunchAttributeUpdates) {
  EXPECT_CALL(command_buffer_,
              CreateLaunch(_, _, _, A<const se::NativeKernel&>(), _, _, _))
      .WillOnce(Return(&command_));
  const XLA_FFI_LaunchConfig config = LaunchConfig();
  ASSERT_OK(Create(config));
  const auto changes = {
      +[](XLA_FFI_LaunchConfig& c) { c.launch_dims.cluster.x = 1; },
      +[](XLA_FFI_LaunchConfig& c) { c.preferred_cluster.x = 8; },
      +[](XLA_FFI_LaunchConfig& c) { c.uses_pdl = 0; },
      +[](XLA_FFI_LaunchConfig& c) { c.cooperative = 0; }};
  for (auto change : changes) {
    XLA_FFI_LaunchConfig updated = config;
    change(updated);
    EXPECT_THAT(Update(updated), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("cannot change")));
  }
  EXPECT_CALL(command_buffer_,
              UpdateLaunch(&command_, se::ThreadDim(32), se::BlockDim(8),
                           Optional(se::ClusterDim(2)),
                           A<const se::NativeKernel&>(), _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(UpdateArguments());
}

TEST_F(RecordFfiLaunchTest, RejectsInvalidConfigurations) {
  struct InvalidConfig {
    void (*change)(XLA_FFI_LaunchConfig&);
    const char* message;
  };
  const InvalidConfig invalid_configs[] = {
      {[](auto& c) { --c.struct_size; }, "truncated"},
      {[](auto& c) { c.launch_dims.grid.x = 0; }, "Grid"},
      {[](auto& c) { c.launch_dims.block.y = -1; }, "block"},
      {[](auto& c) { c.launch_dims.cluster.z = 0; }, "Cluster"},
      {[](auto& c) { c.preferred_cluster.y = 0; }, "Preferred"},
      {[](auto& c) { c.launch_dims.cluster = {}; }, "Preferred"},
      {[](auto& c) { c.launch_dims.grid.x = 6; }, "Preferred"},
      {[](auto& c) {
         c.launch_dims.grid.x = 12;
         c.launch_dims.cluster.x = 3;
       },
       "Preferred"},
      {[](auto& c) {
         c.launch_dims.grid.z = 2;
         c.preferred_cluster.z = 2;
       },
       "Preferred cluster"},
      {[](auto& c) { c.uses_pdl = 2; }, "PDL"},
      {[](auto& c) { c.cooperative = -1; }, "Cooperative"},
  };
  for (const InvalidConfig& invalid : invalid_configs) {
    SCOPED_TRACE(invalid.message);
    XLA_FFI_LaunchConfig config = LaunchConfig();
    invalid.change(config);
    EXPECT_THAT(Create(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                         HasSubstr(invalid.message)));
  }
}

TEST_F(RecordFfiLaunchTest, LegacyCallbacksPreservePdlPolicy) {
  context_.pdl_enabled = false;
  EXPECT_CALL(command_buffer_,
              CreateLaunch(se::ThreadDim(32), se::BlockDim(8), Eq(std::nullopt),
                           A<const se::NativeKernel&>(), _, _, _))
      .WillOnce([&](const auto&, const auto&, const auto&,
                    const se::NativeKernel& kernel, const auto&, auto,
                    auto) -> absl::StatusOr<const se::CommandBuffer::Command*> {
        EXPECT_FALSE(kernel.pdl);
        EXPECT_FALSE(kernel.cooperative);
        EXPECT_EQ(kernel.preferred_cluster_dims, std::nullopt);
        return &command_;
      });
  ASSERT_OK(ffi::TakeError(api_->create_launch(
      &context_, "test", &command_, 0, XLA_FFI_SourceFormat_FUNCTION_PTR,
      {{8, 1, 1}, {32, 1, 1}, {}}, 0, 1, &args_, nullptr, 0, &ffi_command_)));
  EXPECT_CALL(command_buffer_,
              UpdateLaunch(&command_, se::ThreadDim(32), se::BlockDim(8),
                           Eq(std::nullopt), A<const se::NativeKernel&>(), _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(UpdateArguments());
}

}  // namespace
}  // namespace xla::gpu
