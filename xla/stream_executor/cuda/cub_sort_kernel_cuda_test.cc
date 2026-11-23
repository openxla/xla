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

#include "xla/stream_executor/cuda/cub_sort_kernel_cuda.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::cuda {
namespace {

namespace ffi = ::xla::ffi;
namespace se = ::stream_executor;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;

struct CubSortTestParam {
  int64_t batch_size;
  int64_t segment_size;
  bool descending;
};

size_t ComputeLegacyScratchBytes(size_t cub_temp_bytes, int64_t batch_size) {
  if (batch_size <= 1) {
    return std::max<size_t>(1, cub_temp_bytes);
  }
  const size_t aligned_temp_bytes =
      ((cub_temp_bytes + alignof(int32_t) - 1) / alignof(int32_t)) *
      alignof(int32_t);
  return aligned_temp_bytes +
         static_cast<size_t>(batch_size + 1) * sizeof(int32_t);
}

class CubSortKernelCudaFfiTest
    : public ::testing::TestWithParam<CubSortTestParam> {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_,
                         se::PlatformManager::PlatformWithName("CUDA"));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
  }

  se::Platform* platform_ = nullptr;
  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
};

TEST(CubSortKernelCudaRegistrationTest,
     LegacyExecuteHandlersRegisteredAndInstantiateHandlersRemoved) {
  EXPECT_THAT(
      ffi::FindHandler(
          xla::gpu::kCubDeviceRadixSortKeysUnassignedScratchSizeTarget, "CUDA"),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(ffi::FindHandler(
                  xla::gpu::kCubDeviceRadixSortPairsUnassignedScratchSizeTarget,
                  "CUDA"),
              StatusIs(absl::StatusCode::kNotFound));

  ASSERT_OK_AND_ASSIGN(
      ffi::HandlerRegistration keys_reg,
      ffi::FindHandler(xla::gpu::kCubDeviceRadixSortKeysTarget, "CUDA"));
  EXPECT_EQ(keys_reg.bundle.instantiate, nullptr);
  EXPECT_NE(keys_reg.bundle.execute, nullptr);

  ASSERT_OK_AND_ASSIGN(
      ffi::HandlerRegistration pairs_reg,
      ffi::FindHandler(xla::gpu::kCubDeviceRadixSortPairsTarget, "CUDA"));
  EXPECT_EQ(pairs_reg.bundle.instantiate, nullptr);
  EXPECT_NE(pairs_reg.bundle.execute, nullptr);
}

TEST_P(CubSortKernelCudaFfiTest, SortKeysExecute) {
  const CubSortTestParam param = GetParam();
  const int64_t num_items = param.batch_size * param.segment_size;

  std::vector<float> host_keys_in(num_items);
  for (int64_t b = 0; b < param.batch_size; ++b) {
    for (int64_t i = 0; i < param.segment_size; ++i) {
      host_keys_in[b * param.segment_size + i] =
          static_cast<float>(((i * 37 + b * 13) % 97) - 48) +
          static_cast<float>(b) * 0.25f;
    }
  }

  std::vector<float> expected_keys = host_keys_in;
  for (int64_t b = 0; b < param.batch_size; ++b) {
    auto begin = expected_keys.begin() + b * param.segment_size;
    auto end = begin + param.segment_size;
    if (param.descending) {
      std::sort(begin, end, std::greater<float>());
    } else {
      std::sort(begin, end);
    }
  }

  size_t cub_temp_bytes = 0;
  ASSERT_OK(ToStatus(CubSortKeys<float>(nullptr, cub_temp_bytes, nullptr,
                                        nullptr, num_items, param.descending,
                                        param.batch_size, nullptr)));
  const size_t scratch_bytes =
      ComputeLegacyScratchBytes(cub_temp_bytes, param.batch_size);

  se::DeviceAddress<float> d_keys_in =
      executor_->AllocateArray<float>(num_items);
  auto keys_in_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_keys_in); });
  se::DeviceAddress<float> d_keys_out =
      executor_->AllocateArray<float>(num_items);
  auto keys_out_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_keys_out); });
  se::DeviceAddressBase d_scratch = executor_->Allocate(scratch_bytes);
  auto scratch_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_scratch); });

  const size_t keys_bytes = static_cast<size_t>(num_items) * sizeof(float);
  ASSERT_OK(stream_->Memcpy(&d_keys_in, host_keys_in.data(), keys_bytes));

  ASSERT_OK_AND_ASSIGN(
      ffi::HandlerRegistration registration,
      ffi::FindHandler(xla::gpu::kCubDeviceRadixSortKeysTarget, "CUDA"));

  const std::vector<int64_t> dims =
      param.batch_size == 1
          ? std::vector<int64_t>{param.segment_size}
          : std::vector<int64_t>{param.batch_size, param.segment_size};
  ffi::CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/2);
  builder.AddBufferArg(d_keys_in, xla::F32, dims);
  builder.AddBufferRet(d_keys_out, xla::F32, dims);
  builder.AddBufferRet(d_scratch, xla::U8,
                       {static_cast<int64_t>(scratch_bytes)});
  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("descending", param.descending);
  attrs.Insert("batch_size", param.batch_size);
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  ffi::InvokeContext context{};
  context.backend_context = ffi::InvokeContext::GpuContext{stream_.get()};
  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), registration.bundle.execute,
                        call_frame, context, XLA_FFI_ExecutionStage_EXECUTE));

  std::vector<float> host_keys_out(num_items);
  ASSERT_OK(stream_->Memcpy(host_keys_out.data(), d_keys_out, keys_bytes));
  ASSERT_OK(stream_->BlockHostUntilDone());

  EXPECT_THAT(host_keys_out, ElementsAreArray(expected_keys));
}

TEST_P(CubSortKernelCudaFfiTest, SortPairsExecute) {
  const CubSortTestParam param = GetParam();
  const int64_t num_items = param.batch_size * param.segment_size;

  std::vector<float> host_keys_in(num_items);
  std::vector<int32_t> host_vals_in(num_items);
  for (int64_t b = 0; b < param.batch_size; ++b) {
    for (int64_t i = 0; i < param.segment_size; ++i) {
      const int64_t idx = b * param.segment_size + i;
      host_keys_in[idx] = static_cast<float>(((i * 29 + b * 7) % 31) - 15);
      host_vals_in[idx] = static_cast<int32_t>(idx);
    }
  }

  std::vector<float> expected_keys(num_items);
  std::vector<int32_t> expected_vals(num_items);
  for (int64_t b = 0; b < param.batch_size; ++b) {
    std::vector<std::pair<float, int32_t>> pairs(param.segment_size);
    for (int64_t i = 0; i < param.segment_size; ++i) {
      const int64_t idx = b * param.segment_size + i;
      pairs[i] = {host_keys_in[idx], host_vals_in[idx]};
    }
    auto cmp = [&](const std::pair<float, int32_t>& a,
                   const std::pair<float, int32_t>& b_pair) {
      return param.descending ? (a.first > b_pair.first)
                              : (a.first < b_pair.first);
    };
    std::stable_sort(pairs.begin(), pairs.end(), cmp);
    for (int64_t i = 0; i < param.segment_size; ++i) {
      const int64_t idx = b * param.segment_size + i;
      expected_keys[idx] = pairs[i].first;
      expected_vals[idx] = pairs[i].second;
    }
  }

  size_t cub_temp_bytes = 0;
  ASSERT_OK(ToStatus(CubSortPairs<float, uint32_t>(
      nullptr, cub_temp_bytes, nullptr, nullptr, nullptr, nullptr, num_items,
      param.descending, param.batch_size, nullptr)));
  const size_t scratch_bytes =
      ComputeLegacyScratchBytes(cub_temp_bytes, param.batch_size);

  se::DeviceAddress<float> d_keys_in =
      executor_->AllocateArray<float>(num_items);
  auto keys_in_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_keys_in); });
  se::DeviceAddress<int32_t> d_vals_in =
      executor_->AllocateArray<int32_t>(num_items);
  auto vals_in_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_vals_in); });
  se::DeviceAddress<float> d_keys_out =
      executor_->AllocateArray<float>(num_items);
  auto keys_out_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_keys_out); });
  se::DeviceAddress<int32_t> d_vals_out =
      executor_->AllocateArray<int32_t>(num_items);
  auto vals_out_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_vals_out); });
  se::DeviceAddressBase d_scratch = executor_->Allocate(scratch_bytes);
  auto scratch_cleanup =
      absl::MakeCleanup([&]() { executor_->Deallocate(&d_scratch); });

  const size_t keys_bytes = static_cast<size_t>(num_items) * sizeof(float);
  const size_t vals_bytes = static_cast<size_t>(num_items) * sizeof(int32_t);
  ASSERT_OK(stream_->Memcpy(&d_keys_in, host_keys_in.data(), keys_bytes));
  ASSERT_OK(stream_->Memcpy(&d_vals_in, host_vals_in.data(), vals_bytes));

  ASSERT_OK_AND_ASSIGN(
      ffi::HandlerRegistration registration,
      ffi::FindHandler(xla::gpu::kCubDeviceRadixSortPairsTarget, "CUDA"));

  const std::vector<int64_t> dims =
      param.batch_size == 1
          ? std::vector<int64_t>{param.segment_size}
          : std::vector<int64_t>{param.batch_size, param.segment_size};
  ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/3);
  builder.AddBufferArg(d_keys_in, xla::F32, dims);
  builder.AddBufferArg(d_vals_in, xla::S32, dims);
  builder.AddBufferRet(d_keys_out, xla::F32, dims);
  builder.AddBufferRet(d_vals_out, xla::S32, dims);
  builder.AddBufferRet(d_scratch, xla::U8,
                       {static_cast<int64_t>(scratch_bytes)});
  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("descending", param.descending);
  attrs.Insert("batch_size", param.batch_size);
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  ffi::InvokeContext context{};
  context.backend_context = ffi::InvokeContext::GpuContext{stream_.get()};
  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), registration.bundle.execute,
                        call_frame, context, XLA_FFI_ExecutionStage_EXECUTE));

  std::vector<float> host_keys_out(num_items);
  std::vector<int32_t> host_vals_out(num_items);
  ASSERT_OK(stream_->Memcpy(host_keys_out.data(), d_keys_out, keys_bytes));
  ASSERT_OK(stream_->Memcpy(host_vals_out.data(), d_vals_out, vals_bytes));
  ASSERT_OK(stream_->BlockHostUntilDone());

  EXPECT_THAT(host_keys_out, ElementsAreArray(expected_keys));
  EXPECT_THAT(host_vals_out, ElementsAreArray(expected_vals));
}

INSTANTIATE_TEST_SUITE_P(
    BatchAndDirection, CubSortKernelCudaFfiTest,
    ::testing::Values(CubSortTestParam{1, 256, false},
                      CubSortTestParam{1, 256, true},
                      CubSortTestParam{4, 128, false},
                      CubSortTestParam{4, 128, true}),
    [](const ::testing::TestParamInfo<CubSortTestParam>& info) {
      return absl::StrFormat("batch%d_seg%d_%s", info.param.batch_size,
                             info.param.segment_size,
                             info.param.descending ? "desc" : "asc");
    });

}  // namespace
}  // namespace stream_executor::cuda
