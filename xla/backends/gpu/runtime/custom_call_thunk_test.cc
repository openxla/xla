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

#include "xla/backends/gpu/runtime/custom_call_thunk.h"

#include <cstddef>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {
using absl_testing::StatusIs;
using ::testing::HasSubstr;

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  TF_ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
  TF_ASSIGN_OR_RETURN(auto* platform,
                      se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(0);
}

TEST(CustomCallThunkTest, SimpleCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  bool was_called = false;

  CustomCallThunk::CustomCallTarget target =
      [&](se::Stream* stream_in_callback, void** args, const char* target_name,
          size_t num_args, XlaCustomCallStatus* status) {
        was_called = true;
        EXPECT_THAT(stream_in_callback, ::testing::Eq(stream.get()));
      };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CustomCallThunk::Create(Thunk::ThunkInfo(), "target_name",
                                          target, {}, {}, ""));
  se::StreamExecutorMemoryAllocator allocator(executor);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
      stream.get(), stream.get(), nullptr, nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(Thunk::ExecuteParams(params)),
              absl_testing::IsOk());
  EXPECT_TRUE(was_called);
}

TEST(CustomCallThunkTest, CustomCallOnCustomStream) {
  // Whitebox test to ensure that custom calls respect execution_stream_id
  // assignments.
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> extra_stream,
                          executor->CreateStream());
  // Setup the additional streams.
  Thunk::ExecutionStreamIdMap additional_compute_streams = {};
  additional_compute_streams[ExecutionStreamId(1)] = extra_stream.get();
  se::StreamExecutorMemoryAllocator allocator(executor);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
      stream.get(), stream.get(), nullptr, nullptr, additional_compute_streams);

  CustomCallThunk::CustomCallTarget target =
      [&](se::Stream* stream_in_callback, void** args, const char* target_name,
          size_t num_args, XlaCustomCallStatus* status) {
        // We should be launching on the extra stream and not the default one.
        EXPECT_THAT(stream_in_callback, ::testing::Eq(extra_stream.get()));
      };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CustomCallThunk::Create(Thunk::ThunkInfo(), "target_name",
                                          target, {}, {}, ""));
  // Setting this tells the thunk to dispatch on one of the additional streams.
  thunk->set_execution_stream_id(ExecutionStreamId(1));
  EXPECT_THAT(thunk->ExecuteOnStream(Thunk::ExecuteParams(params)),
              absl_testing::IsOk());
}

// A simple callback function that always returns an error.
absl::Status ReturnError() {
  return absl::UnknownError("Custom call was executed!");
}

XLA_FFI_DEFINE_HANDLER(kReturnError, ReturnError, ffi::Ffi::Bind(),
                       {ffi::Traits::kCmdBufferCompatible});

constexpr absl::string_view kReturnErrorCustomCallName =
    "__xla_test$$return_error";

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kReturnErrorCustomCallName,
                         "CUDA", kReturnError);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kReturnErrorCustomCallName,
                         "ROCM", kReturnError);

TEST(CustomCallThunkTest, ResolvesFFICustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/std::string(kReturnErrorCustomCallName),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{},
          /*called_computation=*/nullptr,
          /*platform_name=*/executor->GetPlatform()->Name()));

  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations empty_unused_allocations({}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), empty_unused_allocations,
      /*stream=*/stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(params),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("Custom call was executed!")));
}

// A simple callback function that always returns an error and has the function
// signature for a legacy custom call.
void Callback_WithStatusFailed(void* /*stream*/, void** /*buffers*/,
                               const char* /*opaque*/, size_t /*opaque_len*/,
                               XlaCustomCallStatus* status) {
  constexpr absl::string_view kErrorMessage =
      "Legacy Custom call was executed!";
  XlaCustomCallStatusSetFailure(status, kErrorMessage.data(),
                                kErrorMessage.size());
}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_WithStatusFailed, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_WithStatusFailed, "ROCM");

TEST(CustomCallThunkTest, ResolvesLegacyCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/"Callback_WithStatusFailed",
          /*operands=*/{},
          /*results=*/{}, /*opaque=*/"",
          CustomCallApiVersion::API_VERSION_STATUS_RETURNING,
          /*platform_name=*/executor->GetPlatform()->Name()));

  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations empty_unused_allocations({}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), empty_unused_allocations,
      /*stream=*/stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(params),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Legacy Custom call was executed!")));
}

}  // namespace
}  // namespace xla::gpu
