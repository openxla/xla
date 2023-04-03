
#include "xla/service/cpu/runtime_fft_impl.h"
#include "ducc/src/ducc0/fft/fft.h"

namespace xla {

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