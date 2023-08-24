/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/cpu/runtime_ducc_fft.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/base/attributes.h"
#include "ducc/google/fft.h"  // from @ducc
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive  // For ThreadPoolDevice.
#include "unsupported/Eigen/CXX11/ThreadPool"  // from @eigen_archive
#include "xla/executable_run_options.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_DuccFft(
    const void *run_options_ptr, void *out, void *operand, int32_t fft_type,
    int32_t double_precision, int32_t fft_rank, int64_t input_batch,
    int64_t fft_length0, int64_t fft_length1, int64_t fft_length2) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);

  bool forward = (fft_type == /*FFT*/ 0 || fft_type == /*RFFT*/ 2);
  bool real = (fft_type == /*RFFT*/ 2 || fft_type == /*IRFFT*/ 3);

  using Shape = std::vector<std::size_t>;
  using Stride = std::vector<std::ptrdiff_t>;

  Shape in_shape(fft_rank + 1);
  Stride in_stride(fft_rank + 1);
  Shape out_shape(fft_rank + 1);
  Stride out_stride(fft_rank + 1);
  Shape axes(fft_rank);
  int64_t fft_length[3] = {fft_length0, fft_length1, fft_length2};

  in_shape[fft_rank] = (real && !forward) ? fft_length[fft_rank - 1] / 2 + 1
                                          : fft_length[fft_rank - 1];
  in_stride[fft_rank] = 1;
  out_shape[fft_rank] = (real && forward) ? fft_length[fft_rank - 1] / 2 + 1
                                          : fft_length[fft_rank - 1];
  out_stride[fft_rank] = 1;
  for (int i = fft_rank; i-- > 1;) {
    in_shape[i] = fft_length[i - 1];
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
    out_shape[i] = fft_length[i - 1];
    out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    axes[i] = i + 1;
  }
  in_shape[0] = input_batch;
  in_stride[0] = in_stride[1] * in_shape[1];
  out_shape[0] = input_batch;
  out_stride[0] = out_stride[1] * out_shape[1];
  axes[0] = 1;

  double inv_scale = 1.0;
  for (int i = 0; i < fft_rank; ++i) {
    inv_scale *= out_shape[axes[i]];
  }
  double scale = forward ? 1.0 : 1.0 / inv_scale;

  Eigen::ThreadPoolInterface *thread_pool =
      run_options == nullptr ? nullptr
      : run_options->intra_op_thread_pool() == nullptr
          ? nullptr
          : run_options->intra_op_thread_pool()->getPool();

  if (!real) {
    if (double_precision) {
      ducc0::google::c2c(static_cast<const std::complex<double> *>(operand),
                         in_shape, in_stride,
                         static_cast<std::complex<double> *>(out), out_shape,
                         out_stride, axes, forward, scale, thread_pool);
    } else {
      ducc0::google::c2c(
          static_cast<const std::complex<float> *>(operand), in_shape,
          in_stride, static_cast<std::complex<float> *>(out), out_shape,
          out_stride, axes, forward, static_cast<float>(scale), thread_pool);
    }
  } else if (forward) {
    if (double_precision) {
      ducc0::google::r2c(static_cast<double *>(operand), in_shape, in_stride,
                         static_cast<std::complex<double> *>(out), out_shape,
                         out_stride, axes, forward, scale, thread_pool);
    } else {
      ducc0::google::r2c(static_cast<float *>(operand), in_shape, in_stride,
                         static_cast<std::complex<float> *>(out), out_shape,
                         out_stride, axes, forward, static_cast<float>(scale),
                         thread_pool);
    }
  } else {
    if (double_precision) {
      ducc0::google::c2r(static_cast<const std::complex<double> *>(operand),
                         in_shape, in_stride, static_cast<double *>(out),
                         out_shape, out_stride, axes, forward, scale,
                         thread_pool);
    } else {
      ducc0::google::c2r(static_cast<const std::complex<float> *>(operand),
                         in_shape, in_stride, static_cast<float *>(out),
                         out_shape, out_stride, axes, forward,
                         static_cast<float>(scale), thread_pool);
    }
  }
}
