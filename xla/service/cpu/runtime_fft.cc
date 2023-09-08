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

#include "xla/service/cpu/runtime_fft.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "ducc/google/threading.h"  // from @ducc
#include "ducc/src/ducc0/fft/fft.h"  // from @ducc
#include "ducc/src/ducc0/fft/fft1d_impl.h"  // from @ducc  // NOLINT: required for fft definitions.
#include "ducc/src/ducc0/fft/fftnd_impl.h"  // from @ducc  // NOLINT: required for fft definitions.
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/executable_run_options.h"
#include "xla/service/cpu/runtime_lightweight_check.h"

namespace {

enum class FftKind {
  kR2C,
  kC2R,
  kC2C,
};

using shape_t = ducc0::fmav_info::shape_t;
using stride_t = ducc0::fmav_info::stride_t;

void FftImpl(int nthreads, void* out, void* operand, int32_t fft_type,
             int32_t double_precision, int32_t fft_rank, int64_t input_batch,
             int64_t const* fft_lengths) {
  shape_t shape(fft_rank + 1);
  shape[0] = static_cast<size_t>(input_batch);
  for (int i = 0; i < fft_rank; ++i) {
    shape[i + 1] = static_cast<size_t>(fft_lengths[i]);
  }

  FftKind kind;
  bool forward;
  switch (static_cast<xla::internal::FftType>(fft_type)) {
    case xla::internal::FftType::FFT:
      kind = FftKind::kC2C;
      forward = true;
      break;
    case xla::internal::FftType::IFFT:
      kind = FftKind::kC2C;
      forward = false;
      break;
    case xla::internal::FftType::RFFT:
      kind = FftKind::kR2C;
      forward = true;
      break;
    case xla::internal::FftType::IRFFT:
      kind = FftKind::kC2R;
      forward = false;
      break;
  }

  shape_t axes;
  axes.resize(fft_rank);
  std::iota(axes.begin(), axes.end(), 1);

  double scale = 1.0;
  if (!forward) {
    for (int i = 0; i < fft_rank; ++i) {
      scale *= shape[i + 1];
    }
    scale = 1.0 / scale;
  }

  auto compute_strides = [](const shape_t& shape) {
    stride_t strides(shape.size());
    ptrdiff_t stride = 1;
    auto stride_it = strides.rbegin();
    for (auto shape_it = shape.rbegin(); shape_it != shape.rend(); ++shape_it) {
      *stride_it = stride;
      ++stride_it;

      stride *= *shape_it;
    }
    return strides;
  };
  stride_t strides = compute_strides(shape);

  switch (kind) {
    case FftKind::kC2C:
      if (!double_precision) {
        ducc0::cfmav<std::complex<float>> m_in(
            reinterpret_cast<std::complex<float>*>(operand), shape, strides);
        ducc0::vfmav<std::complex<float>> m_out(
            reinterpret_cast<std::complex<float>*>(out), shape, strides);
        ducc0::c2c(m_in, m_out, axes, forward, static_cast<float>(scale),
                   nthreads);
      } else {
        ducc0::cfmav<std::complex<double>> m_in(
            reinterpret_cast<std::complex<double>*>(operand), shape, strides);
        ducc0::vfmav<std::complex<double>> m_out(
            reinterpret_cast<std::complex<double>*>(out), shape, strides);
        ducc0::c2c(m_in, m_out, axes, forward, scale, nthreads);
      }
      break;
    case FftKind::kC2R: {
      shape_t shape_in = shape;
      shape_in.back() = shape_in.back() / 2 + 1;
      stride_t strides_in = compute_strides(shape_in);

      const stride_t& strides_out = strides;

      if (!double_precision) {
        ducc0::cfmav<std::complex<float>> m_in(
            reinterpret_cast<std::complex<float>*>(operand), shape_in,
            strides_in);
        ducc0::vfmav<float> m_out(reinterpret_cast<float*>(out), shape,
                                  strides_out);
        ducc0::c2r(m_in, m_out, axes, forward, static_cast<float>(scale),
                   nthreads);
      } else {
        ducc0::cfmav<std::complex<double>> m_in(
            reinterpret_cast<std::complex<double>*>(operand), shape_in,
            strides_in);
        ducc0::vfmav<double> m_out(reinterpret_cast<double*>(out), shape,
                                   strides_out);
        ducc0::c2r(m_in, m_out, axes, forward, scale, nthreads);
      }
      break;
    }
    case FftKind::kR2C: {
      shape_t shape_out = shape;
      shape_out.back() = shape_out.back() / 2 + 1;
      stride_t strides_out = compute_strides(shape_out);

      if (!double_precision) {
        ducc0::cfmav<float> m_in(reinterpret_cast<float*>(operand), shape,
                                 strides);
        ducc0::vfmav<std::complex<float>> m_out(
            reinterpret_cast<std::complex<float>*>(out), shape_out,
            strides_out);
        ducc0::r2c(m_in, m_out, axes, forward, static_cast<float>(scale),
                   nthreads);
      } else {
        ducc0::cfmav<double> m_in(reinterpret_cast<double*>(operand), shape,
                                  strides);
        ducc0::vfmav<std::complex<double>> m_out(
            reinterpret_cast<std::complex<double>*>(out), shape_out,
            strides_out);
        ducc0::r2c(m_in, m_out, axes, forward, scale, nthreads);
      }
      break;
    }
  }
}

}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_DuccFft(
    const void* run_options_ptr, void* out, void* operand, int32_t fft_type,
    int32_t double_precision, int32_t fft_rank, int64_t input_batch,
    int64_t const* fft_lengths) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  // Set DUCC to use the current device threadpool.  Since this is a
  // thread-local setting, this is thread-safe.
  ducc0::google::EigenThreadPool thread_pool(
      *run_options->intra_op_thread_pool()->getPool());
  ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(thread_pool);

  size_t nthreads = thread_pool.nthreads();

  FftImpl(nthreads, out, operand, fft_type, double_precision, fft_rank,
          input_batch, fft_lengths);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_DuccSingleThreadedFft(
    const void* run_options_ptr, void* out, void* operand, int32_t fft_type,
    int32_t double_precision, int32_t fft_rank, int64_t input_batch,
    int64_t const* fft_lengths) {
  FftImpl(/*nthreads=*/1, out, operand, fft_type, double_precision, fft_rank,
          input_batch, fft_lengths);
}
