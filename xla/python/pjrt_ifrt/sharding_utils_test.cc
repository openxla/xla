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

#define EIGEN_USE_THREADS

#include "xla/python/pjrt_ifrt/sharding_utils.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/python/ifrt/shape.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace {
using ::testing::ElementsAreArray;

template <typename T>
std::vector<T> GetTfTensorData(const tensorflow::Tensor& tensor) {
  return std::vector<T>(tensor.flat<T>().data(),
                        tensor.flat<T>().data() + tensor.NumElements());
}

TEST(ShardingUtilsTest, EvenSplit) {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);

  // split a 4 x 4 to 2 x 2
  alignas(64)
      int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  Shape in_shape({4, 4});
  Shape out_shape({2, 2});
  int num_partitions = 4;

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, (ReplicateOrSplit<int32_t, 2>(num_partitions, data, in_shape,
                                                 out_shape, device)));
  ASSERT_EQ(result.size(), 4);

  EXPECT_THAT(GetTfTensorData<int32_t>(result[0]),
              ElementsAreArray({1, 2, 5, 6}));
  EXPECT_THAT(GetTfTensorData<int32_t>(result[1]),
              ElementsAreArray({9, 10, 13, 14}));
  EXPECT_THAT(GetTfTensorData<int32_t>(result[2]),
              ElementsAreArray({3, 4, 7, 8}));
  EXPECT_THAT(GetTfTensorData<int32_t>(result[3]),
              ElementsAreArray({11, 12, 15, 16}));
}

TEST(ShardingUtilsTest, UnevenSplit) {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);

  // split a 4 x 4 to 2 x 4
  alignas(64)
      int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  Shape in_shape({4, 4});
  Shape out_shape({2, 4});
  int num_partitions = 2;

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, (ReplicateOrSplit<int32_t, 2>(num_partitions, data, in_shape,
                                                 out_shape, device)));
  ASSERT_EQ(result.size(), 2);

  EXPECT_THAT(GetTfTensorData<int32_t>(result[0]),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(GetTfTensorData<int32_t>(result[1]),
              ElementsAreArray({9, 10, 11, 12, 13, 14, 15, 16}));
}

TEST(ShardingUtilsTest, Replicate) {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);

  // Replicates 2x2
  alignas(64) int32_t data[] = {1, 2, 3, 4};
  Shape in_shape({2, 2});
  Shape out_shape({2, 2});
  int num_partitions = 2;

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, (ReplicateOrSplit<int32_t, 2>(num_partitions, data, in_shape,
                                                 out_shape, device)));
  ASSERT_EQ(result.size(), 2);

  EXPECT_THAT(GetTfTensorData<int32_t>(result[0]),
              ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(GetTfTensorData<int32_t>(result[1]),
              ElementsAreArray({1, 2, 3, 4}));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
