/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/device_memory_allocator.h"

#include <complex>
#include <cstdint>
#include <string>
#include <tuple>

#include "absl/strings/str_cat.h"
#include "Eigen/Core"  // from @eigen_archive
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/float8.h"
#include "tsl/platform/test.h"
#include "tsl/protobuf/dnn.pb.h"

namespace stream_executor {
namespace {

struct AllocateTypedTest
    : testing::TestWithParam<std::tuple<dnn::DataType, int>> {
  static inline constexpr int kDeviceOrdinal = 0;

  Platform *platform = MultiPlatformManager::PlatformWithName("Host").value();
  StreamExecutor *executor =
      platform->ExecutorForDevice(kDeviceOrdinal).value();
  StreamExecutorMemoryAllocator allocator{executor};

  template <typename T>
  void AllocateGivesCorrectSizeAndValue(int num_elements) {
    const auto allocation =
        allocator.Allocate<T>(kDeviceOrdinal, num_elements).value();
    static_assert(
        std::is_same_v<decltype(allocation), const ScopedDeviceMemory<T>>);
    ASSERT_NE(allocation, nullptr);
    ASSERT_EQ(allocation.allocator(), &allocator);
    ASSERT_EQ(allocation.device_ordinal(), kDeviceOrdinal);

    const auto &allocation_cref = allocation.cref();
    static_assert(
        std::is_same_v<decltype(allocation_cref), const DeviceMemory<T> &>);
    ASSERT_FALSE(allocation_cref.is_null());
    ASSERT_EQ(allocation_cref.size(), num_elements * sizeof(T));
    ASSERT_EQ(allocation_cref.ElementCount(), num_elements);

    if (num_elements == 1) {
      ASSERT_TRUE(allocation_cref.IsScalar());
    } else {
      ASSERT_FALSE(allocation_cref.IsScalar());
    }
  }
};

TEST_P(AllocateTypedTest, AllocateGivesCorrectSizeAndValue) {
  const auto &[type, num_elements] = GetParam();

  switch (type) {
    case dnn::kFloat:
      return AllocateGivesCorrectSizeAndValue<float>(num_elements);
    case dnn::kDouble:
      return AllocateGivesCorrectSizeAndValue<double>(num_elements);
    case dnn::kHalf:
      return AllocateGivesCorrectSizeAndValue<Eigen::half>(num_elements);
    case dnn::kInt8:
      return AllocateGivesCorrectSizeAndValue<int8_t>(num_elements);
    case dnn::kInt32:
      return AllocateGivesCorrectSizeAndValue<int32_t>(num_elements);
    case dnn::kComplexFloat:
      return AllocateGivesCorrectSizeAndValue<std::complex<float>>(
          num_elements);
    case dnn::kComplexDouble:
      return AllocateGivesCorrectSizeAndValue<std::complex<double>>(
          num_elements);
    case dnn::kBF16:
      return AllocateGivesCorrectSizeAndValue<Eigen::bfloat16>(num_elements);
    case dnn::kF8E5M2:
      return AllocateGivesCorrectSizeAndValue<tsl::float8_e5m2>(num_elements);
    case dnn::kF8E4M3FN:
      return AllocateGivesCorrectSizeAndValue<tsl::float8_e4m3fn>(num_elements);

    default:
      FAIL() << "unsupported type";
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllocateTyped, AllocateTypedTest,
    testing::Combine(testing::Values(dnn::kFloat, dnn::kDouble, dnn::kHalf,
                                     dnn::kInt8, dnn::kInt32,
                                     dnn::kComplexFloat, dnn::kComplexDouble,
                                     dnn::kBF16, dnn::kF8E5M2, dnn::kF8E4M3FN),
                     testing::Values(1, 4)),
    [](const testing::TestParamInfo<std::tuple<dnn::DataType, int>> &info) {
      return absl::StrCat("T_", dnn::DataType_Name(std::get<0>(info.param)),
                          "_N_", std::get<1>(info.param));
    });

}  // namespace
}  // namespace stream_executor
