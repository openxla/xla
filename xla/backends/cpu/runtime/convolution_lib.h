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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_LIB_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/backends/cpu/runtime/work_queue.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/framework/convolution/eigen_spatial_convolutions.h"  // IWYU pragma: keep

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu::internal {

// Done callback is called when the Convolution computation is complete.
using DoneCallback = absl::AnyInvocable<void()>;

template <typename ScalarType>
void EigenConv2D(const Eigen::ThreadPoolDevice* device, ScalarType* out,
                 const ScalarType* lhs, const ScalarType* rhs,
                 Eigen::Index input_batch, Eigen::Index input_x,
                 Eigen::Index input_y, Eigen::Index input_channels,
                 Eigen::Index kernel_x, Eigen::Index kernel_y,
                 Eigen::Index kernel_channels, Eigen::Index kernel_filters,
                 Eigen::Index output_x, Eigen::Index output_y,
                 Eigen::Index x_stride, Eigen::Index y_stride,
                 Eigen::Index padding_x_before, Eigen::Index padding_x_after,
                 Eigen::Index padding_y_before, Eigen::Index padding_y_after,
                 Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,
                 Eigen::Index rhs_x_dilation, Eigen::Index rhs_y_dilation,
                 Eigen::Index feature_group_count, DoneCallback done);

template <typename ScalarType>
void EigenConv3D(const Eigen::ThreadPoolDevice* device, ScalarType* out,
                 const ScalarType* lhs, const ScalarType* rhs,
                 Eigen::Index input_batch, Eigen::Index input_x,
                 Eigen::Index input_y, Eigen::Index input_z,
                 Eigen::Index input_channels, Eigen::Index kernel_x,
                 Eigen::Index kernel_y, Eigen::Index kernel_z,
                 Eigen::Index kernel_channels, Eigen::Index kernel_filters,
                 Eigen::Index output_x, Eigen::Index output_y,
                 Eigen::Index output_z, Eigen::Index x_stride,
                 Eigen::Index y_stride, Eigen::Index z_stride,
                 Eigen::Index padding_x_before, Eigen::Index padding_x_after,
                 Eigen::Index padding_y_before, Eigen::Index padding_y_after,
                 Eigen::Index padding_z_before, Eigen::Index padding_z_after,
                 Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,
                 Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,
                 Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,
                 Eigen::Index feature_group_count, DoneCallback done);

//===----------------------------------------------------------------------===//
// Convolution 2D implementation details.
//===----------------------------------------------------------------------===//

inline bool UseParallelDevice(const Eigen::ThreadPoolDevice* device) {
  return device != nullptr && device->getPool() != nullptr &&
         device->numThreads() > 0;
}

inline bool MultiplyWithinLimit(std::initializer_list<Eigen::Index> factors,
                                uint64_t limit, uint64_t* product = nullptr) {
  uint64_t acc = 1;
  for (Eigen::Index f : factors) {
    if (f < 0 || (f > 0 && acc > limit / static_cast<uint64_t>(f))) {
      return false;
    }
    acc *= static_cast<uint64_t>(f);
  }
  if (product != nullptr) {
    *product = acc;
  }
  return true;
}

template <typename ScalarType>
bool TryFinishZeroConv(ScalarType* out,
                       std::initializer_list<Eigen::Index> out_dims,
                       std::initializer_list<Eigen::Index> in_dims,
                       DoneCallback& done) {
  constexpr uint64_t kMax = std::min<uint64_t>(
      static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max()),
      std::numeric_limits<size_t>::max() / sizeof(ScalarType));
  uint64_t n = 0;
  CHECK(MultiplyWithinLimit(out_dims, kMax, &n));
  const bool zero_in = std::any_of(in_dims.begin(), in_dims.end(),
                                   [](Eigen::Index d) { return d <= 0; });
  if (n > 0 && !zero_in) {
    return false;
  }
  if (n > 0) {
    std::fill_n(out, static_cast<size_t>(n), ScalarType(0.0f));
  }
  std::move(done)();
  return true;
}

// Clamps filter range [d_begin, d_end) so `origin + d` stays in [0, out_size).
inline bool ComputePatchRange(Eigen::Index idx, Eigen::Index stride,
                              Eigen::Index pad, Eigen::Index filter,
                              Eigen::Index out, Eigen::Index* origin_out,
                              Eigen::Index* d_begin, Eigen::Index* d_end) {
  const Eigen::Index off = idx * stride, shift = off - (filter - 1);
  if (pad < -off ||
      (shift >= 0 ? pad > (out - 1) - shift
                  : (pad > out - 1 && pad - (out - 1) > -shift))) {
    return false;
  }
  using U = std::make_unsigned_t<Eigen::Index>;
  const auto origin =
      static_cast<Eigen::Index>(static_cast<U>(pad) + static_cast<U>(shift));
  *origin_out = origin;
  *d_begin = origin < 0 ? -origin : 0;
  *d_end = origin <= out - filter ? filter : out - origin;
  return true;
}

// Returns in 'out_data' (assumes to be zero-initialized) image patch in storage
// order (width, height, depth), constructed from patches in 'conv_matrix',
// which is required to be in storage order (in_width * in_height, filter_width,
// filter_height, out_depth).
// Based on TF implementation by Yangqing Jia (jiayq).
// TODO(adambanas): The original implementation implicitly rotates the kernel by
// 180 degrees, but to be backwards compatible, we cannot do that in XLA. This
// results in counterintuitive operations on conv_matrix, which is also 15-20%
// slower. Try alternative approaches (e.g. rotate kernel before matrix
// multiplication in the calling function).
template <typename T>
void Pack2DPatches(const T* conv_matrix, Eigen::Index depth,
                   Eigen::Index height, Eigen::Index width,
                   Eigen::Index filter_h, Eigen::Index filter_w,
                   Eigen::Index pad_top, Eigen::Index pad_left,
                   Eigen::Index stride_h, Eigen::Index stride_w,
                   Eigen::Index h_patches_number, Eigen::Index w_patches_number,
                   T* __restrict out_im_data) {
  if (depth <= 0 || height <= 0 || width <= 0 || filter_h <= 0 ||
      filter_w <= 0 || stride_h <= 0 || stride_w <= 0 ||
      h_patches_number <= 0 || w_patches_number <= 0) {
    return;
  }
  const Eigen::Index filter_spatial_size = filter_h * filter_w;
  const Eigen::Index patch_stride = filter_spatial_size * depth;

  for (Eigen::Index w = 0; w < w_patches_number; ++w) {
    Eigen::Index w_begin = 0, dw_begin = 0, dw_end = 0;
    if (!ComputePatchRange(w, stride_w, pad_left, filter_w, width, &w_begin,
                           &dw_begin, &dw_end)) {
      continue;
    }
    for (Eigen::Index h = 0; h < h_patches_number; ++h) {
      Eigen::Index h_begin = 0, dh_begin = 0, dh_end = 0;
      if (!ComputePatchRange(h, stride_h, pad_top, filter_h, height, &h_begin,
                             &dh_begin, &dh_end)) {
        continue;
      }
      const Eigen::Index patch_base = (w * h_patches_number + h) * patch_stride;
      for (Eigen::Index dw = dw_begin; dw < dw_end; ++dw) {
        for (Eigen::Index dh = dh_begin; dh < dh_end; ++dh) {
          const T* src =
              conv_matrix + patch_base +
              (filter_spatial_size - 1 - (dw * filter_h + dh)) * depth;
          T* dst =
              out_im_data + ((w_begin + dw) * height + h_begin + dh) * depth;
          for (Eigen::Index i = 0; i < depth; ++i) {
            dst[i] += src[i];
          }
        }
      }
    }
  }
}

template <typename ScalarType>
bool CanUseCustomTransposedConv(
    Eigen::Index input_batch, Eigen::Index input_x, Eigen::Index input_y,
    Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_filters,
    Eigen::Index output_x, Eigen::Index output_y, Eigen::Index x_stride,
    Eigen::Index y_stride, Eigen::Index lhs_x_dilation,
    Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,
    Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count) {
  if ((lhs_x_dilation <= 1 && lhs_y_dilation <= 1) || lhs_x_dilation < 1 ||
      lhs_y_dilation < 1 || rhs_x_dilation != 1 || rhs_y_dilation != 1 ||
      feature_group_count != 1 || x_stride != 1 || y_stride != 1 ||
      input_batch <= 0 || input_x <= 0 || input_y <= 0 || kernel_x <= 0 ||
      kernel_y <= 0 || kernel_filters <= 0 || output_x <= 0 || output_y <= 0) {
    return false;
  }
  constexpr uint64_t kMaxIndex =
      static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max());
  if (static_cast<uint64_t>(input_x - 1) >
          kMaxIndex / static_cast<uint64_t>(lhs_x_dilation) ||
      static_cast<uint64_t>(input_y - 1) >
          kMaxIndex / static_cast<uint64_t>(lhs_y_dilation)) {
    return false;
  }
  constexpr uint64_t kMaxConvMatrixSize = uint64_t{8} << 30;  // 8 GiB
  constexpr uint64_t kMaxAddressableElems = std::min<uint64_t>(
      kMaxIndex, std::numeric_limits<size_t>::max() / sizeof(ScalarType));
  constexpr uint64_t kMaxConvMatrixElems = std::min<uint64_t>(
      kMaxConvMatrixSize / sizeof(ScalarType), kMaxAddressableElems);
  return MultiplyWithinLimit({input_batch, input_x, input_y, kernel_x, kernel_y,
                              kernel_filters},
                             kMaxConvMatrixElems) &&
         MultiplyWithinLimit({input_batch, output_x, output_y, kernel_filters},
                             kMaxAddressableElems);
}

// This implementation is based on TF algorithm with parallel contraction.
// TODO(adambanas): There are other variants of this algorithm, 10% performance
// improvement was observed on 1D case when not using parallel contraction.
// Explore these alternatives.
// TODO(adambanas): Add support for feature group count.
template <typename ScalarType>
void EigenTransposedConv2D(
    const Eigen::ThreadPoolDevice* device, ScalarType* out,
    const ScalarType* lhs, const ScalarType* rhs, Eigen::Index input_batch,
    Eigen::Index input_x, Eigen::Index input_y, Eigen::Index input_channels,
    Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_channels,
    Eigen::Index kernel_filters, Eigen::Index output_x, Eigen::Index output_y,
    Eigen::Index padding_x_before, Eigen::Index padding_x_after,
    Eigen::Index padding_y_before, Eigen::Index padding_y_after,
    Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,
    DoneCallback done) {
  // Grouped convolutions are not supported yet.
  CHECK(kernel_channels == input_channels);

  using TensorMap3D =
      Eigen::TensorMap<Eigen::Tensor<ScalarType, 3, Eigen::RowMajor>,
                       Eigen::Unaligned>;
  using ConstTensorMap3D =
      Eigen::TensorMap<Eigen::Tensor<const ScalarType, 3, Eigen::RowMajor>,
                       Eigen::Unaligned>;
  using ConstTensorMap2D =
      Eigen::TensorMap<Eigen::Tensor<const ScalarType, 2, Eigen::RowMajor>,
                       Eigen::Unaligned>;

  // Total spatial dimensions.
  const Eigen::Index input_image_size = input_x * input_y;
  const Eigen::Index output_image_size = output_x * output_y;
  // Kernel dimensions per input channel.
  const Eigen::Index kernel_total_size = kernel_x * kernel_y * kernel_filters;
  const Eigen::Index input_offset = input_image_size * kernel_total_size;
  const Eigen::Index output_offset = output_image_size * kernel_filters;

  // Initialize output to zero.
  ScalarType* out_data = out;
  const size_t output_size =
      static_cast<size_t>(input_batch) * static_cast<size_t>(output_offset);
  if (output_size > 0) {
    std::fill_n(out_data, output_size, ScalarType(0.0f));
  }

  // Intermediate buffer (convolution matrix).
  const size_t buffer_size =
      static_cast<size_t>(input_batch) * static_cast<size_t>(input_offset);
  if (buffer_size == 0 || output_size == 0 || input_channels == 0) {
    done();
    return;
  }

  auto conv_matrix = std::make_unique<ScalarType[]>(buffer_size);
  ScalarType* conv_matrix_data = conv_matrix.get();

  // Initialize contraction dims (we need to transpose 'B' below, the dimension
  // we need to contract is 'kernel_channels').
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims = {
      Eigen::IndexPair<Eigen::DenseIndex>(1, 1)};

  // Compute intermediate results (convolution matrix) into conv_matrix.
  TensorMap3D C(conv_matrix_data, input_batch * input_image_size,
                kernel_x * kernel_y, kernel_filters);

  ConstTensorMap2D A(lhs, input_batch * input_image_size, input_channels);
  ConstTensorMap3D B(rhs, kernel_x * kernel_y, kernel_channels, kernel_filters);

  // Pack the calculated patches into the output buffer.
  // NOTE: The ownership of the conv_matrix is transferred to the lambda without
  // data copy or reallocation. Thanks to that, conv_matrix_data pointer remains
  // valid, and that is important because 'C' matrix is referencing it.
  auto pack_patches = [=, conv_matrix = std::move(conv_matrix),
                       done = std::move(done)]() mutable {
    const ScalarType* conv_matrix_data = conv_matrix.get();
    ScalarType* local_out_data = out_data;

    // TODO(adambanas): Run this part in parallel.
    for (Eigen::Index image_id = 0; image_id < input_batch; ++image_id) {
      Pack2DPatches<ScalarType>(conv_matrix_data, kernel_filters, output_y,
                                output_x, kernel_y, kernel_x, padding_y_before,
                                padding_x_before, lhs_y_dilation,
                                lhs_x_dilation, /*h_patches_number=*/input_y,
                                /*w_patches_number=*/input_x, local_out_data);

      conv_matrix_data += input_offset;
      local_out_data += output_offset;
    }

    // Signal completion of the work once the patches are packed.
    done();
  };

  Eigen::ThreadPoolDevice single_thread_device(/*pool=*/nullptr,
                                               /*num_cores=*/1);
  C.device(UseParallelDevice(device) ? *device : single_thread_device,
           std::move(pack_patches)) = A.contract(B, contract_dims);
}

// Algorithm that works for all types of 2D convolutions. Even though it works
// for transposed convolutions, the custom algorithm should be used whenever
// applicable, because it is faster.
template <bool is_grouped, typename ScalarType>
void EigenGenericConv2D(
    const Eigen::ThreadPoolDevice* device, ScalarType* out,
    const ScalarType* lhs, const ScalarType* rhs, Eigen::Index input_batch,
    Eigen::Index input_x, Eigen::Index input_y, Eigen::Index input_channels,
    Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_channels,
    Eigen::Index kernel_filters, Eigen::Index output_x, Eigen::Index output_y,
    Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index padding_x_before,
    Eigen::Index padding_x_after, Eigen::Index padding_y_before,
    Eigen::Index padding_y_after, Eigen::Index lhs_x_dilation,
    Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,
    Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count,
    DoneCallback done) {
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  //   kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
  pre_contract_dims[0] = output_y * output_x * input_batch;
  pre_contract_dims[1] = kernel_channels * kernel_y * kernel_x;

  // IMPORTANT: Below, the row and column dimensions must be flipped when passed
  // to Eigen.

  // For non-grouped convolutions, we can optimize Eigen expressions and avoid
  // introducing an extra dimension of size `1`.
  if constexpr (!is_grouped) {
    DCHECK_EQ(feature_group_count, 1) << "Expected feature group count to be 1";

    const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 4, Eigen::RowMajor>,
                           Eigen::Unaligned>
        input(lhs, input_batch, input_x, input_y, input_channels);
    const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 2, Eigen::RowMajor>,
                           Eigen::Unaligned>
        kernel(rhs, pre_contract_dims[1], kernel_filters);
    Eigen::TensorMap<Eigen::Tensor<ScalarType, 2, Eigen::RowMajor>,
                     Eigen::Unaligned>
        output(out, pre_contract_dims[0], kernel_filters);

    auto convolved =
        input
            .extract_image_patches(
                kernel_y, kernel_x, y_stride, x_stride, rhs_y_dilation,
                rhs_x_dilation, lhs_y_dilation, lhs_x_dilation,
                padding_y_before, padding_y_after, padding_x_before,
                padding_x_after, static_cast<ScalarType>(0.0f))
            .reshape(pre_contract_dims)
            .contract(kernel, contract_dims);

    Eigen::ThreadPoolDevice single_thread_device(/*pool=*/nullptr,
                                                 /*num_cores=*/1);
    output.device(UseParallelDevice(device) ? *device : single_thread_device,
                  std::move(done)) = convolved;
  } else {
    const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 5, Eigen::RowMajor>,
                           Eigen::Unaligned>
        input(lhs, input_batch, input_x, input_y, feature_group_count,
              input_channels / feature_group_count);
    const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 3, Eigen::RowMajor>,
                           Eigen::Unaligned>
        kernel(rhs, pre_contract_dims[1], feature_group_count,
               kernel_filters / feature_group_count);

    // Signal done callback when all feature groups are done.
    tsl::CountDownAsyncValueRef<tsl::Chain> count_down(feature_group_count);
    count_down.AsPtr().AndThen([done = std::move(done)]() mutable { done(); });

    auto convolve_range = [=](const Eigen::ThreadPoolDevice& dev,
                              Eigen::Index start, Eigen::Index end) {
      Eigen::TensorMap<Eigen::Tensor<ScalarType, 3, Eigen::RowMajor>,
                       Eigen::Unaligned>
          output(out, pre_contract_dims[0], feature_group_count,
                 kernel_filters / feature_group_count);
      for (Eigen::Index i = start; i < end; ++i) {
        auto on_done = [count_down]() mutable { count_down.CountDown(); };
        auto convolved =
            input.chip(i, 3)
                .extract_image_patches(
                    kernel_y, kernel_x, y_stride, x_stride, rhs_y_dilation,
                    rhs_x_dilation, lhs_y_dilation, lhs_x_dilation,
                    padding_y_before, padding_y_after, padding_x_before,
                    padding_x_after, static_cast<ScalarType>(0.0f))
                .reshape(pre_contract_dims)
                .contract(kernel.chip(i, 1), contract_dims);
        output.chip(i, 1).device(dev, std::move(on_done)) = convolved;
      }
    };

    // For grouped convolutions, we need to execute multiple Eigen expressions
    // and we might use thread pool to run them concurrently.
    if (UseParallelDevice(device)) {
      // Although we schedule at most one tasks for each thread, individual
      // convolution might also schedule more tasks into the same thread pool.
      const auto max_tasks = static_cast<Eigen::Index>(device->numThreads());
      const Eigen::Index task_size = (feature_group_count - 1) / max_tasks + 1;
      const Eigen::Index num_tasks = (feature_group_count - 1) / task_size + 1;

      Worker::Parallelize(device->getPool(), /*num_workers=*/num_tasks,
                          num_tasks, [=](Eigen::Index task_index) {
                            const Eigen::Index start = task_index * task_size;
                            const Eigen::Index end =
                                start + std::min(task_size,
                                                 feature_group_count - start);
                            convolve_range(*device, start, end);
                          });

    } else {
      // Convolve all feature groups sequentially in the caller thread.
      Eigen::ThreadPoolDevice single_thread_device(/*pool=*/nullptr,
                                                   /*num_cores=*/1);
      convolve_range(single_thread_device, 0, feature_group_count);
    }
  }
}

template <typename ScalarType>
void EigenConv2D(const Eigen::ThreadPoolDevice* device, ScalarType* out,
                 const ScalarType* lhs, const ScalarType* rhs,
                 Eigen::Index input_batch, Eigen::Index input_x,
                 Eigen::Index input_y, Eigen::Index input_channels,
                 Eigen::Index kernel_x, Eigen::Index kernel_y,
                 Eigen::Index kernel_channels, Eigen::Index kernel_filters,
                 Eigen::Index output_x, Eigen::Index output_y,
                 Eigen::Index x_stride, Eigen::Index y_stride,
                 Eigen::Index padding_x_before, Eigen::Index padding_x_after,
                 Eigen::Index padding_y_before, Eigen::Index padding_y_after,
                 Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,
                 Eigen::Index rhs_x_dilation, Eigen::Index rhs_y_dilation,
                 Eigen::Index feature_group_count, DoneCallback done) {
  CHECK_GT(feature_group_count, 0);
  if (TryFinishZeroConv(out, {input_batch, output_x, output_y, kernel_filters},
                        {input_x, input_y, input_channels, kernel_x, kernel_y,
                         kernel_channels},
                        done)) {
    return;
  }
  if (CanUseCustomTransposedConv<ScalarType>(
          input_batch, input_x, input_y, kernel_x, kernel_y, kernel_filters,
          output_x, output_y, x_stride, y_stride, lhs_x_dilation,
          lhs_y_dilation, rhs_x_dilation, rhs_y_dilation,
          feature_group_count)) {
    EigenTransposedConv2D(device, out, lhs, rhs, input_batch, input_x, input_y,
                          input_channels, kernel_x, kernel_y, kernel_channels,
                          kernel_filters, output_x, output_y, padding_x_before,
                          padding_x_after, padding_y_before, padding_y_after,
                          lhs_x_dilation, lhs_y_dilation, std::move(done));
    return;
  }
  (feature_group_count == 1
       ? EigenGenericConv2D<false, ScalarType>
       : EigenGenericConv2D<true, ScalarType>)(device, out, lhs, rhs,
                                               input_batch, input_x, input_y,
                                               input_channels, kernel_x,
                                               kernel_y, kernel_channels,
                                               kernel_filters, output_x,
                                               output_y, x_stride, y_stride,
                                               padding_x_before,
                                               padding_x_after,
                                               padding_y_before,
                                               padding_y_after, lhs_x_dilation,
                                               lhs_y_dilation, rhs_x_dilation,
                                               rhs_y_dilation,
                                               feature_group_count,
                                               std::move(done));
}

//===----------------------------------------------------------------------===//
// Convolution 3D implementation details.
//===----------------------------------------------------------------------===//

template <typename ScalarType>
void EigenConv3D(const Eigen::ThreadPoolDevice* device, ScalarType* out,
                 const ScalarType* lhs, const ScalarType* rhs,
                 Eigen::Index input_batch, Eigen::Index input_x,
                 Eigen::Index input_y, Eigen::Index input_z,
                 Eigen::Index input_channels, Eigen::Index kernel_x,
                 Eigen::Index kernel_y, Eigen::Index kernel_z,
                 Eigen::Index kernel_channels, Eigen::Index kernel_filters,
                 Eigen::Index output_x, Eigen::Index output_y,
                 Eigen::Index output_z, Eigen::Index x_stride,
                 Eigen::Index y_stride, Eigen::Index z_stride,
                 Eigen::Index padding_x_before, Eigen::Index padding_x_after,
                 Eigen::Index padding_y_before, Eigen::Index padding_y_after,
                 Eigen::Index padding_z_before, Eigen::Index padding_z_after,
                 Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,
                 Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,
                 Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,
                 Eigen::Index feature_group_count, DoneCallback done) {
  CHECK_GT(feature_group_count, 0);
  if (TryFinishZeroConv(
          out, {input_batch, output_x, output_y, output_z, kernel_filters},
          {input_x, input_y, input_z, input_channels, kernel_x, kernel_y,
           kernel_z, kernel_channels},
          done)) {
    return;
  }
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  //   kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
  pre_contract_dims[0] = output_x * output_y * output_z * input_batch;
  pre_contract_dims[1] = kernel_channels * kernel_x * kernel_y * kernel_z;

  const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 6, Eigen::RowMajor>,
                         Eigen::Unaligned>
      input(lhs, input_batch, input_x, input_y, input_z, feature_group_count,
            input_channels / feature_group_count);

  const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 3, Eigen::RowMajor>,
                         Eigen::Unaligned>
      kernel(rhs, pre_contract_dims[1], feature_group_count,
             kernel_filters / feature_group_count);

  Eigen::TensorMap<Eigen::Tensor<ScalarType, 3, Eigen::RowMajor>,
                   Eigen::Unaligned>
      output(out, pre_contract_dims[0], feature_group_count,
             kernel_filters / feature_group_count);

  // Signal done callback when all feature groups are done.
  tsl::CountDownAsyncValueRef<tsl::Chain> count_down(feature_group_count);
  count_down.AsPtr().AndThen([done = std::move(done)]() mutable { done(); });

  Eigen::ThreadPoolDevice single_thread_device(/*pool=*/nullptr,
                                               /*num_cores=*/1);
  const Eigen::ThreadPoolDevice& dev =
      UseParallelDevice(device) ? *device : single_thread_device;

  for (Eigen::Index i = 0; i < feature_group_count; ++i) {
    // The dimension order must be flipped when passed to Eigen.
    auto input_chip = input.chip(i, 4);
    auto patches =
        Eigen::TensorVolumePatchOp<Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::Dynamic, decltype(input_chip)>(
            input_chip, kernel_z, kernel_y, kernel_x, z_stride, y_stride,
            x_stride, rhs_z_dilation, rhs_y_dilation, rhs_x_dilation,
            lhs_z_dilation, lhs_y_dilation, lhs_x_dilation, padding_z_before,
            padding_z_after, padding_y_before, padding_y_after,
            padding_x_before, padding_x_after, static_cast<ScalarType>(0.0f));

    auto convolved = patches.reshape(pre_contract_dims)
                         .contract(kernel.chip(i, 1), contract_dims);

    auto on_done = [count_down]() mutable { count_down.CountDown(); };
    output.chip(i, 1).device(dev, std::move(on_done)) = convolved;
  }
}

}  // namespace xla::cpu::internal

// Define Conv2D template for all supported data types.
#define XLA_CPU_DECLARE_CONV2D(SCALAR_TYPE)                                 \
  extern template void xla::cpu::internal::EigenConv2D<SCALAR_TYPE>(        \
      const Eigen::ThreadPoolDevice* device, SCALAR_TYPE* out,              \
      const SCALAR_TYPE* lhs, const SCALAR_TYPE* rhs,                       \
      Eigen::Index input_batch, Eigen::Index input_x, Eigen::Index input_y, \
      Eigen::Index input_channels, Eigen::Index kernel_x,                   \
      Eigen::Index kernel_y, Eigen::Index kernel_channels,                  \
      Eigen::Index kernel_filters, Eigen::Index output_x,                   \
      Eigen::Index output_y, Eigen::Index x_stride, Eigen::Index y_stride,  \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,          \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,          \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,             \
      Eigen::Index rhs_x_dilation, Eigen::Index rhs_y_dilation,             \
      Eigen::Index feature_group_count, DoneCallback done)

XLA_CPU_DECLARE_CONV2D(Eigen::half);
XLA_CPU_DECLARE_CONV2D(float);

#undef XLA_CPU_DECLARE_CONV2D

// Define Conv3D template for all supported data types.
#define XLA_CPU_DECLARE_CONV3D(SCALAR_TYPE)                                 \
  extern template void xla::cpu::internal::EigenConv3D<SCALAR_TYPE>(        \
      const Eigen::ThreadPoolDevice* device, SCALAR_TYPE* out,              \
      const SCALAR_TYPE* lhs, const SCALAR_TYPE* rhs,                       \
      Eigen::Index input_batch, Eigen::Index input_x, Eigen::Index input_y, \
      Eigen::Index input_z, Eigen::Index input_channels,                    \
      Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_z,  \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,            \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index output_z,  \
      Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index z_stride,  \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,          \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,          \
      Eigen::Index padding_z_before, Eigen::Index padding_z_after,          \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,             \
      Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,             \
      Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,             \
      Eigen::Index feature_group_count, DoneCallback done)

XLA_CPU_DECLARE_CONV3D(Eigen::half);
XLA_CPU_DECLARE_CONV3D(float);

#undef XLA_CPU_DECLARE_CONV3D

#define XLA_CPU_DEFINE_CONV2D(SCALAR_TYPE)                                  \
  template void xla::cpu::internal::EigenConv2D<SCALAR_TYPE>(               \
      const Eigen::ThreadPoolDevice* device, SCALAR_TYPE* out,              \
      const SCALAR_TYPE* lhs, const SCALAR_TYPE* rhs,                       \
      Eigen::Index input_batch, Eigen::Index input_x, Eigen::Index input_y, \
      Eigen::Index input_channels, Eigen::Index kernel_x,                   \
      Eigen::Index kernel_y, Eigen::Index kernel_channels,                  \
      Eigen::Index kernel_filters, Eigen::Index output_x,                   \
      Eigen::Index output_y, Eigen::Index x_stride, Eigen::Index y_stride,  \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,          \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,          \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,             \
      Eigen::Index rhs_x_dilation, Eigen::Index rhs_y_dilation,             \
      Eigen::Index feature_group_count, DoneCallback done)

#define XLA_CPU_DEFINE_CONV3D(SCALAR_TYPE)                                  \
  template void xla::cpu::internal::EigenConv3D<SCALAR_TYPE>(               \
      const Eigen::ThreadPoolDevice* device, SCALAR_TYPE* out,              \
      const SCALAR_TYPE* lhs, const SCALAR_TYPE* rhs,                       \
      Eigen::Index input_batch, Eigen::Index input_x, Eigen::Index input_y, \
      Eigen::Index input_z, Eigen::Index input_channels,                    \
      Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_z,  \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,            \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index output_z,  \
      Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index z_stride,  \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,          \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,          \
      Eigen::Index padding_z_before, Eigen::Index padding_z_after,          \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,             \
      Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,             \
      Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,             \
      Eigen::Index feature_group_count, DoneCallback done)

#endif  // XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_LIB_H_
