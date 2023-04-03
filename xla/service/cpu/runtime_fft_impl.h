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
#ifndef XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
#define XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_

#include <array>

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/types.h"
#include "ducc/src/ducc0/fft/fft.h"

namespace xla {

namespace internal {

enum class FftType : int32_t {
  FFT = 0,    // Forward FFT; complex in, complex out.
  IFFT = 1,   // Inverse FFT; complex in, complex out.
  RFFT = 2,   // Forward real FFT; real in, fft_length / 2 + 1 complex out
  IRFFT = 3,  // Inverse real FFT; fft_length / 2 + 1 complex in,
              //                   fft_length real out
};
inline constexpr int FftTypeArraySize() { return 4; }

// Computes either a forward or reverse complex-to-complex FFT.
template <bool Forward, int FFTRank, typename EigenDevice, typename Complex>
void EigenFftC2C(const EigenDevice& device, Complex* out, Complex* operand,
                 int64_t input_batch, int64_t fft_length0, int64_t fft_length1,
                 int64_t fft_length2) {
  // Create the axes (which are always trailing).
  const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
  constexpr auto direction = Forward ? Eigen::FFT_FORWARD : Eigen::FFT_REVERSE;

  const std::array<int64_t, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> dims;
  dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    dims[i + 1] = fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, dims);
  Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, dims);
  output.device(device) = input.template fft<Eigen::BothParts, direction>(axes);
}

// Computes a forward real->complex FFT, slicing out redundant negative
// frequencies from the innermost dimension.
template <int FFTRank, typename EigenDevice, typename Real, typename Complex>
void EigenFftR2C(const EigenDevice& device, Complex* out, Real* operand,
                 int64_t input_batch, int64_t fft_length0, int64_t fft_length1,
                 int64_t fft_length2) {
  const std::array<int64_t, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> in_dims;
  in_dims[0] = input_batch;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> out_dims;
  out_dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    in_dims[i + 1] = fft_shape[i];
    out_dims[i + 1] = i == FFTRank - 1 ? fft_shape[i] / 2 + 1 : fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<Real, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, in_dims);
  Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, out_dims);

  // Create the axes (which are always trailing).
  const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);

  // Compute the full FFT using a temporary tensor.
  Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor> full_fft(in_dims);

  const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;
  full_fft.device(device) =
      input.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);

  // Slice away the negative frequency components.
  output.device(device) = full_fft.slice(zero_start_indices, out_dims);
}

// Computes a reverse complex->real FFT, reconstructing redundant negative
// frequencies using reverse conjugate on innermost dimension after doing IFFT
// on outer dimensions.
template <int FFTRank, typename EigenDevice, typename Complex, typename Real>
void EigenFftC2R(const EigenDevice& device, Real* out, Complex* operand,
                 int64_t input_batch, int64_t fft_length0, int64_t fft_length1,
                 int64_t fft_length2) {
  const std::array<int64_t, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> in_dims;
  in_dims[0] = input_batch;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> out_dims;
  out_dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    in_dims[i + 1] = i == FFTRank - 1 ? fft_shape[i] / 2 + 1 : fft_shape[i];
    out_dims[i + 1] = fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, in_dims);
  Eigen::TensorMap<Eigen::Tensor<Real, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, out_dims);

  // Calculate the shape of the temporary tensor for the full FFT and the
  // region we will slice from input given fft_shape. We slice input to
  // fft_shape on its inner-most dimensions, except the last (which we
  // slice to fft_shape[-1] / 2 + 1).
  Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor> full_fft(out_dims);

  // Calculate the starting point and range of the source of
  // negative frequency part.
  auto neg_sizes = in_dims;
  neg_sizes[FFTRank] = fft_shape[FFTRank - 1] - in_dims[FFTRank];
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_target_indices;
  neg_target_indices[FFTRank] = in_dims[FFTRank];

  const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_start_indices;
  neg_start_indices[FFTRank] = 1;

  full_fft.slice(zero_start_indices, in_dims).device(device) = input;

  // First, conduct IFFTs on outer dimensions. We save computation (and
  // avoid touching uninitialized memory) by slicing full_fft to the
  // subregion we wrote input to.
  if (FFTRank > 1) {
    const auto outer_axes =
        Eigen::ArrayXi::LinSpaced(FFTRank - 1, 1, FFTRank - 1);
    full_fft.slice(zero_start_indices, in_dims).device(device) =
        full_fft.slice(zero_start_indices, in_dims)
            .template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(outer_axes);
  }

  // Reconstruct the full FFT by appending reversed and conjugated
  // spectrum as the negative frequency part.
  Eigen::array<bool, FFTRank + 1> reverse_last_axis;
  for (auto i = 0; i <= FFTRank; i++) {
    reverse_last_axis[i] = i == FFTRank;
  }

  if (neg_sizes[FFTRank] != 0) {
    full_fft.slice(neg_target_indices, neg_sizes).device(device) =
        full_fft.slice(neg_start_indices, neg_sizes)
            .reverse(reverse_last_axis)
            .conjugate();
  }

  auto inner_axis = Eigen::array<int, 1>{FFTRank};
  output.device(device) =
      full_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(inner_axis);
}

template <int FFTRank, typename EigenDevice>
void EigenFftWithRank(const EigenDevice& device, void* out, void* operand,
                      FftType fft_type, bool double_precision,
                      int64_t input_batch, int64_t fft_length0,
                      int64_t fft_length1, int64_t fft_length2) {
  switch (fft_type) {
    case FftType::FFT:
      if (double_precision) {
        EigenFftC2C<true, FFTRank, EigenDevice, complex128>(
            device, static_cast<complex128*>(out),
            static_cast<complex128*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftC2C<true, FFTRank, EigenDevice, complex64>(
            device, static_cast<complex64*>(out),
            static_cast<complex64*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      }
      break;
    case FftType::IFFT:
      if (double_precision) {
        EigenFftC2C<false, FFTRank, EigenDevice, complex128>(
            device, static_cast<complex128*>(out),
            static_cast<complex128*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftC2C<false, FFTRank, EigenDevice, complex64>(
            device, static_cast<complex64*>(out),
            static_cast<complex64*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      }
      break;
    case FftType::RFFT:
      if (double_precision) {
        EigenFftR2C<FFTRank, EigenDevice, double, complex128>(
            device, static_cast<complex128*>(out),
            static_cast<double*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftR2C<FFTRank, EigenDevice, float, complex64>(
            device, static_cast<complex64*>(out), static_cast<float*>(operand),
            input_batch, fft_length0, fft_length1, fft_length2);
      }
      break;
    case FftType::IRFFT:
      if (double_precision) {
        EigenFftC2R<FFTRank, EigenDevice, complex128, double>(
            device, static_cast<double*>(out),
            static_cast<complex128*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftC2R<FFTRank, EigenDevice, complex64, float>(
            device, static_cast<float*>(out), static_cast<complex64*>(operand),
            input_batch, fft_length0, fft_length1, fft_length2);
      }
      break;
    default:
      // Unsupported FFT type
      abort();
  }
}

}  // namespace internal

template <typename EigenDevice>
void EigenFftImpl(const EigenDevice& device, void* out, void* operand,
                  internal::FftType fft_type, bool double_precision,
                  int32_t fft_rank, int64_t input_batch, int64_t fft_length0,
                  int64_t fft_length1, int64_t fft_length2) {
  switch (fft_rank) {
    case 1:
      internal::EigenFftWithRank<1, EigenDevice>(device, out, operand, fft_type,
                                                 double_precision, input_batch,
                                                 fft_length0, 0, 0);
      break;
    case 2:
      internal::EigenFftWithRank<2, EigenDevice>(device, out, operand, fft_type,
                                                 double_precision, input_batch,
                                                 fft_length0, fft_length1, 0);
      break;
    case 3:
      internal::EigenFftWithRank<3, EigenDevice>(
          device, out, operand, fft_type, double_precision, input_batch,
          fft_length0, fft_length1, fft_length2);
      break;
    default:
      // Unsupported FFT rank
      abort();
  }
}

using shape_t = ducc0::fmav_info::shape_t;
using stride_t = ducc0::fmav_info::stride_t;

// TODO: add thread pool
void DuccFftImpl(void *out, void *in, internal::FftType fft_type,
                 bool double_precision, int32_t fft_rank, int64_t input_batch, 
                 int64_t fft_length0, int64_t fft_length1, int64_t fft_length2) {
  // What we do: similar to the Eigen-based implementation we reinterpret 
  // the shape as [input_batch, ..fft_lengths] with some additional transform
  // to the last out/in shape axis for R2C and C2R respectively.
  //
  // TODO: convert this to generics?
  auto forward = fft_type == internal::FftType::FFT || fft_type == internal::FftType::RFFT;
  shape_t shape;  
  shape_t axes;
  double scale = 1.;
  // While the parameter computation is inexpensive, one wonders whether it is worth 
  // amortizing the cost at IR emission time...
  switch (fft_rank) {
    case 1:
      shape = {input_batch, fft_length0};
      axes = {1};
      if (!forward) {
        scale = 1. / fft_length0;
      }
      break;
    case 2:
      shape = {input_batch, fft_length0, fft_length1};
      axes = {1, 2};
      if (!forward) {
        scale = 1. / (fft_length0 * fft_length1);
      }
      break;
    case 3:
      shape = {input_batch, fft_length0, fft_length1, fft_length2};
      axes = {1, 2, 3};
      if (!forward) {
        scale = 1. / (fft_length0 * fft_length1 * fft_length2);
      }
      break;
    default:
      // Unsupported FFT rank
      abort();
  }
  scale = 100.;

  switch (fft_type) {
  case internal::FftType::IFFT:
  case internal::FftType::FFT:
    // TODO(jon-chuang): swap to double first
    if (!double_precision) {
      ducc0::cfmav<std::complex<float>> m_in(
          reinterpret_cast<std::complex<float> *>(in), shape);
      ducc0::vfmav<std::complex<float>> m_out(
          reinterpret_cast<std::complex<float> *>(out), shape);
      ducc0::c2c(m_in, m_out, axes, forward, static_cast<float>(scale));
    } else {
      ducc0::cfmav<std::complex<double>> m_in(
          reinterpret_cast<std::complex<double> *>(in), shape);
      ducc0::vfmav<std::complex<double>> m_out(
          reinterpret_cast<std::complex<double> *>(out), shape);
      ducc0::c2c(m_in, m_out, axes, forward, scale);
    }
    break;
  case internal::FftType::IRFFT:
    // C2R
    if (!double_precision) {
      auto shape_in = shape;
      shape_in[axes.back()] = shape_in[axes.back()] / 2 + 1;
      ducc0::cfmav<std::complex<float>> m_in(
          reinterpret_cast<std::complex<float> *>(in), shape_in);
      ducc0::vfmav<float> m_out(reinterpret_cast<float *>(out), shape);
      ducc0::c2r(m_in, m_out, axes, false, static_cast<float>(scale));
    } else {
      auto shape_in = shape;
      shape_in[axes.back()] = shape_in[axes.back()] / 2 + 1;
      ducc0::cfmav<std::complex<double>> m_in(
          reinterpret_cast<std::complex<double> *>(in), shape_in);
      ducc0::vfmav<double> m_out(reinterpret_cast<double *>(out), shape);
      ducc0::c2r(m_in, m_out, axes, false, scale);
    }
    break;
  case internal::FftType::RFFT:
    // R2C
    if (!double_precision) {
      auto shape_out = shape;
      shape_out[axes.back()] = shape_out[axes.back()] / 2 + 1;
      ducc0::cfmav<float> m_in(reinterpret_cast<float *>(in), shape);
      ducc0::vfmav<std::complex<float>> m_out(
          reinterpret_cast<std::complex<float> *>(out), shape_out);
      ducc0::r2c(m_in, m_out, axes, true, static_cast<float>(scale));
    } else {
      auto shape_out = shape;
      shape_out[axes.back()] = shape_out[axes.back()] / 2 + 1;
      ducc0::cfmav<double> m_in(reinterpret_cast<double *>(in), shape);
      ducc0::vfmav<std::complex<double>> m_out(
          reinterpret_cast<std::complex<double> *>(out), shape_out);
      ducc0::r2c(m_in, m_out, axes, true, scale);
    }
    break;
  default:
    // Unsupported FFT type
    abort();
  }
}

}  // namespace xla

#endif  // XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
