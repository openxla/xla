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

#include "xla/stream_executor/sycl/sycl_conv_utils.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/protobuf/dnn.pb.h"

namespace stream_executor::sycl {
namespace {

using ::testing::HasSubstr;

// A 3x3 NHWC forward convolution with an OHWI filter. oneDNN prefers a blocked
// filter layout for it, so the workspace holds a pre-packed filter and is
// non-empty.
OneDnnConvConfig MakeForwardConvConfig() {
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSpatial = 56;
  constexpr int64_t kFeatures = 64;
  constexpr int64_t kFilterSize = 3;

  dnn::BatchDescriptor input(2);
  input.set_count(kBatch)
      .set_feature_map_count(kFeatures)
      .set_spatial_dim(dnn::DimIndex::X, kSpatial)
      .set_spatial_dim(dnn::DimIndex::Y, kSpatial)
      .set_layout(dnn::DataLayout::kBatchYXDepth);

  dnn::FilterDescriptor filter(2);
  filter.set_output_feature_map_count(kFeatures)
      .set_input_feature_map_count(kFeatures)
      .set_spatial_dim(dnn::DimIndex::X, kFilterSize)
      .set_spatial_dim(dnn::DimIndex::Y, kFilterSize)
      .set_layout(dnn::FilterLayout::kOutputYXInput);

  dnn::ConvolutionDescriptor conv(2);
  conv.set_zero_padding(dnn::DimIndex::X, 1)
      .set_zero_padding(dnn::DimIndex::Y, 1);

  return OneDnnConvConfig{dnn::ConvolutionKind::FORWARD,
                          dnn::DataType::kFloat,
                          dnn::DataType::kFloat,
                          input,
                          filter,
                          /*output_descriptor=*/input,
                          conv};
}

class SyclConvUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(Platform * platform,
                         PlatformManager::PlatformWithId(kSyclPlatformId));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    ASSERT_OK_AND_ASSIGN(pd_, CreateOneDnnConvPrimitiveDesc(
                                  MakeForwardConvConfig(), stream_.get()));
    ASSERT_GT(pd_.workspace.total_size, 0);
  }

  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
  OneDnnConvPrimitiveDesc pd_;
};

TEST_F(SyclConvUtilsTest, RejectsEmptyWorkspace) {
  EXPECT_THAT(
      CreateOneDnnConvPrimitive(pd_, stream_.get(), OnednnConvBufferPointers{},
                                DeviceAddressBase()),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("needs a workspace of")));
}

TEST_F(SyclConvUtilsTest, RejectsUndersizedWorkspace) {
  DeviceAddressBase workspace =
      executor_->AllocateArray<uint8_t>(pd_.workspace.total_size);
  ASSERT_FALSE(workspace.is_null());

  EXPECT_THAT(
      CreateOneDnnConvPrimitive(
          pd_, stream_.get(), OnednnConvBufferPointers{},
          DeviceAddressBase(workspace.opaque(), pd_.workspace.total_size - 1)),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("needs a workspace of")));
  executor_->Deallocate(&workspace);
}

TEST_F(SyclConvUtilsTest, RejectsNullWorkspace) {
  EXPECT_THAT(CreateOneDnnConvPrimitive(
                  pd_, stream_.get(), OnednnConvBufferPointers{},
                  DeviceAddressBase(nullptr, pd_.workspace.total_size)),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("null workspace buffer")));
}

TEST_F(SyclConvUtilsTest, AcceptsWorkspaceOfRequiredSize) {
  DeviceAddressBase input =
      executor_->AllocateArray<uint8_t>(pd_.src_md.get_size());
  DeviceAddressBase filter =
      executor_->AllocateArray<uint8_t>(pd_.filter_md.get_size());
  DeviceAddressBase output =
      executor_->AllocateArray<uint8_t>(pd_.dst_md.get_size());
  DeviceAddressBase workspace =
      executor_->AllocateArray<uint8_t>(pd_.workspace.total_size);
  ASSERT_FALSE(input.is_null() || filter.is_null() || output.is_null() ||
               workspace.is_null());

  EXPECT_THAT(CreateOneDnnConvPrimitive(
                  pd_, stream_.get(),
                  OnednnConvBufferPointers{input.opaque(), filter.opaque(),
                                           output.opaque()},
                  workspace),
              absl_testing::IsOk());

  executor_->Deallocate(&input);
  executor_->Deallocate(&filter);
  executor_->Deallocate(&output);
  executor_->Deallocate(&workspace);
}

}  // namespace
}  // namespace stream_executor::sycl
