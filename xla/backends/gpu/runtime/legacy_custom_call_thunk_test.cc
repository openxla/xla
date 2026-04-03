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

#include "xla/backends/gpu/runtime/legacy_custom_call_thunk.h"

#include <cstddef>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {
using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::HasSubstr;

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
  ASSIGN_OR_RETURN(auto* platform, se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(0);
}

TEST(LegacyCustomCallThunkTest, SimpleCustomCall) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  bool was_called = false;

  LegacyCustomCallThunk::CustomCallTarget target =
      [&](se::Stream* stream_in_callback, void** args, const char* target_name,
          size_t num_args, XlaCustomCallStatus* status) {
        was_called = true;
        EXPECT_THAT(stream_in_callback, ::testing::Eq(stream.get()));
      };

  ASSERT_OK_AND_ASSIGN(
      auto thunk, LegacyCustomCallThunk::Create(
                      Thunk::ThunkInfo(), "target_name", target, {}, {}, ""));
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
      stream.get(), stream.get(), nullptr, nullptr, nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(Thunk::ExecuteParams(params)),
              absl_testing::IsOk());
  EXPECT_TRUE(was_called);
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

TEST(LegacyCustomCallThunkTest, ResolvesLegacyCustomCall) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LegacyCustomCallThunk> thunk,
                       LegacyCustomCallThunk::Create(
                           Thunk::ThunkInfo(),
                           /*target_name=*/"Callback_WithStatusFailed",
                           /*operands=*/{},
                           /*results=*/{}, /*opaque=*/"",
                           CustomCallApiVersion::API_VERSION_STATUS_RETURNING,
                           /*platform_name=*/executor->GetPlatform()->Name()));

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations empty_unused_allocations({}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), empty_unused_allocations,
      /*stream=*/stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(params),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Legacy Custom call was executed!")));
}

TEST(LegacyCustomCallThunkTest, ParseLegacyProtoWithNonUtf8Opaque) {
  // This test ensures that legacy custom calls can contain non-UTF-8 opaque
  // data, and these will be correctly parsed (and not fail).

  CustomCallThunkProto proto =
      tsl::proto_testing::ParseTextProtoOrDie<CustomCallThunkProto>(
          R"pb(
            target_name: "Callback_WithStatusFailed"
            api_version: API_VERSION_STATUS_RETURNING
            opaque: "\xfe"
          )pb");

  std::string serialized_to_wire_format;
  proto.SerializeToString(&serialized_to_wire_format);

  CustomCallThunkProto reconstructed_proto;
  EXPECT_TRUE(reconstructed_proto.ParseFromString(serialized_to_wire_format));
}

TEST(LegacyCustomCallThunkTest, LegacyCustomCallRoundTrip) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LegacyCustomCallThunk> original_thunk,
                       LegacyCustomCallThunk::Create(
                           Thunk::ThunkInfo(),
                           /*target_name=*/"Callback_WithStatusFailed",
                           /*operands=*/{},
                           /*results=*/{}, /*opaque=*/"opaque",
                           CustomCallApiVersion::API_VERSION_STATUS_RETURNING,
                           /*platform_name=*/executor->GetPlatform()->Name()));

  ASSERT_OK_AND_ASSIGN(ThunkProto proto, original_thunk->ToProto());
  original_thunk.reset();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LegacyCustomCallThunk> new_thunk,
      LegacyCustomCallThunk::FromProto(
          Thunk::ThunkInfo(), proto.custom_call_thunk(),
          /*buffer_allocations=*/{}, executor->GetPlatform()->Name()));

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations empty_unused_allocations({}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), empty_unused_allocations,
      /*stream=*/stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  // We check that the new thunk behaves like the original one (returning
  // internal error with specific message).
  EXPECT_THAT(new_thunk->ExecuteOnStream(params),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Legacy Custom call was executed!")));
}

}  // namespace
}  // namespace xla::gpu
