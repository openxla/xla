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

#ifndef XLA_PYTHON_PJRT_IFRT_SHARDING_UTILS_H_
#define XLA_PYTHON_PJRT_IFRT_SHARDING_UTILS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/python/ifrt/shape.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_types.h"
#include "third_party/tensorflow/core/framework/types.h"
#include "third_party/tensorflow/core/framework/types.pb.h"

namespace xla {
namespace ifrt {
namespace internal {

template <typename DType, int64_t Rank>
tensorflow::Tensor CreateTensor(
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape) {
  tensorflow::TensorShape tensor_shape;
  for (int i = 0; i < Rank; ++i) {
    tensor_shape.AddDim(slice_shape[i]);
  }
  tensorflow::Tensor tensor(tensorflow::template DataTypeToEnum<DType>::v(),
                            tensor_shape);
  return tensor;
}

// Forwarding declarations.
// Enumerates all possible slices on each axis.
// For example, for a 4x4 to 2x2 partition,
// `num_partitions_per_axis` = {2 ,2 } and all possibles slices notated by
// indices are [0, 0], [0, 1], [1, 0], [1, 1].
// Use recursion to enumerate for simplicity since we do not expect the total
// number of slices to be huge.
std::vector<std::vector<int64_t>> EnumerateSlices(
    std::vector<int64_t> num_partitions_per_axis);

template <typename DType, int64_t Rank>
tensorflow::Tensor SplitMaybePad(
    typename tensorflow::TTypes<DType, Rank>::Tensor input_tensor,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, Rank>& slice_shape,
    const Eigen::ThreadPoolDevice& device) {
  Eigen::DSizes<Eigen::DenseIndex, Rank> non_padded_slice_shape;
  for (int64_t r = 0; r < Rank; ++r) {
    int64_t dim_size = input_tensor.dimension(r);
    int out_dim = slice_shape[r];
    int64_t non_padded_dim = 0;
    if (slice_indices[r] + out_dim > dim_size) {
      // Partial padding.
      non_padded_dim = dim_size - slice_indices[r];
    } else {
      non_padded_dim = out_dim;
    }
    non_padded_slice_shape[r] = non_padded_dim;
  }

  tensorflow::Tensor tensor = CreateTensor<DType, Rank>(slice_shape);
  VLOG(1) << "Slice to " << non_padded_slice_shape << " from "
          << tensor.shape();

  tensor.tensor<DType, Rank>()
      .slice(Eigen::DSizes<Eigen::DenseIndex, Rank>(), non_padded_slice_shape)
      .device(device) =
      input_tensor.slice(slice_indices, non_padded_slice_shape);

  return tensor;
}
}  // namespace internal

// Replicate or split specified data into `num_devices` slices of shape
// `out_shape`.
//
// If `out_shape` = `in_shape`, replicates `num_partitions` of input data.
// If `out_shape` < `in_shape`, splits input data into `num_partitions`. Pads
// data if it is not evenly splittable.
template <typename DType, int64_t Rank>
absl::StatusOr<std::vector<tensorflow::Tensor>> ReplicateOrSplit(
    const int num_partitions, DType* data, const Shape& in_shape,
    const Shape& out_shape, const Eigen::ThreadPoolDevice& device) {
  Eigen::DSizes<Eigen::DenseIndex, Rank> in_shape_dsizes;
  Eigen::DSizes<Eigen::DenseIndex, Rank> out_shape_dsizes;
  for (int64_t dim = 0; dim < Rank; ++dim) {
    in_shape_dsizes[dim] = in_shape.dims()[dim];
    out_shape_dsizes[dim] = out_shape.dims()[dim];
  }

  typename tensorflow::TTypes<DType, Rank>::Tensor input_tensor(
      data, in_shape_dsizes);

  std::vector<int64_t> num_partitions_per_axis;
  num_partitions_per_axis.reserve(Rank);
  // Create partitions per_axis in reverse order so that the slices are in
  // natural order.
  for (int64_t dim = 0; dim < Rank; dim++) {
    int64_t dim_size = in_shape.dims()[dim];
    int64_t out_dim = out_shape.dims()[dim];

    int64_t num_partitions = (dim_size + out_dim - 1) / out_dim;
    num_partitions_per_axis.push_back(num_partitions);
    VLOG(1) << "At dimesion  " << dim << " , split into " << num_partitions
            << " partitions. (in_size, out_size): " << dim_size << ", "
            << out_dim;
  }

  std::vector<std::vector<int64_t>> all_slices =
      internal::EnumerateSlices(num_partitions_per_axis);
  VLOG(1) << "Number of slices: " << all_slices.size();

  std::vector<tensorflow::Tensor> result;
  result.reserve(num_partitions);
  for (const auto& slice : all_slices) {
    Eigen::DSizes<Eigen::DenseIndex, Rank> slice_indices;
    for (int64_t dim = 0; dim < Rank; dim++) {
      slice_indices[dim] = slice[dim] * out_shape.dims()[dim];
    }

    auto tensor = internal::SplitMaybePad<DType, Rank>(
        input_tensor, slice_indices, out_shape_dsizes, device);
    result.push_back(std::move(tensor));
  }

  if (all_slices.size() != num_partitions) {
    // This is a replicate case.
    if (all_slices.size() == 1 && num_partitions > 1 && in_shape == out_shape) {
      VLOG(1) << "Replicate by " << num_partitions;
      tensorflow::Tensor replicated_tensor = result[0];
      for (int i = 1; i < num_partitions; ++i) {
        result.push_back(replicated_tensor);
      }
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot split or replicate ", in_shape.DebugString(), " to ",
          out_shape.DebugString(), " across ", num_partitions, " devices."));
    }
  }
  return result;
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_SHARDING_UTILS_H_
