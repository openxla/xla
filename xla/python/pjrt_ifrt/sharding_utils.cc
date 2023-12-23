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

#include "xla/python/pjrt_ifrt/sharding_utils.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace xla {
namespace ifrt {
namespace internal {
// Enumerates all possible slices on each axis.
// For example, for a 4x4 to 2x2 partition,
// `num_partitions_per_axis` = {2 ,2 } and all possibles slices notated by
// indices are [0, 0], [0, 1], [1, 0], [1, 1].
// Use recursion to enumerate for simplicity since we do not expect the total
// number of slices to be huge.
std::vector<std::vector<int64_t>> EnumerateSlices(
    std::vector<int64_t> num_partitions_per_axis) {
  if (num_partitions_per_axis.empty()) {
    return {};
  }

  if (num_partitions_per_axis.size() == 1) {
    std::vector<std::vector<int64_t>> result;
    for (int64_t p = 0; p < num_partitions_per_axis[0]; ++p) {
      result.push_back({p});
    }
    return result;
  }

  int64_t last_dim = num_partitions_per_axis.back();

  num_partitions_per_axis.pop_back();

  std::vector<std::vector<int64_t>> subslices =
      EnumerateSlices(num_partitions_per_axis);

  std::vector<std::vector<int64_t>> result;
  result.reserve(last_dim * subslices.size());
  for (int64_t p = 0; p < last_dim; ++p) {
    for (const auto& subslice : subslices) {
      std::vector<int64_t> slice = subslice;
      slice.push_back(p);
      result.push_back(std::move(slice));
    }
  }
  return result;
}

}  // namespace internal
}  // namespace ifrt
}  // namespace xla
