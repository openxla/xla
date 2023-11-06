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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "Eigen/Core"  // from @eigen_archive
#include "absl/types/span.h"
#include "xla/service/gpu/runtime/resize_bicubic_kernel_common.h"

namespace xla::gpu {
namespace {

#define C10_MAX_THREADS_PER_BLOCK(val)           \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) \
  __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t area_pixel_compute_source_index(
    accscalar_t scale, int dst_index, bool align_corners, bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
                          static_cast<accscalar_t>(0.5);
    // See Note[Follow Opencv resize logic]
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
               ? static_cast<accscalar_t>(0)
               : src_idx;
  }
}

template <typename scalar_t>
__device__ __forceinline__ static scalar_t upsample_get_value_bounded(
    const scalar_t* data, int batch, int channel, int height, int width, int y,
    int x, int64_t strides0, int64_t strides1, int64_t strides2,
    int64_t strides3) {
  int access_y = max(min(y, height - 1), 0);
  int access_x = max(min(x, width - 1), 0);
  const int64_t offset = batch * strides0 + channel * strides1 +
                         access_y * strides2 + access_x * strides3;
  return data[offset];
}

static inline __device__ void gpuAtomicAddNoRet(float* address, float val) {
  atomicAdd(address, val);
}

#if defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
static inline __device__ void gpuAtomicAddNoRet(Eigen::half* address,
                                                Eigen::half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    uint16_t hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum << 16)
                              : (old & 0xffff0000) | hsum;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#else
static inline __device__ void gpuAtomicAddNoRet(Eigen::half* address,
                                                Eigen::half val) {
  atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
}
#endif

#if defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)))
static inline __device__ void gpuAtomicAddNoRet(Eigen::bfloat16* address,
                                                Eigen::bfloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    uint16_t hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;

    old = (size_t)address & 2 ? (old & 0xffff) | (hsum << 16)
                              : (old & 0xffff0000) | hsum;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#else
static inline __device__ void gpuAtomicAddNoRet(Eigen::bfloat16* address,
                                                Eigen::bfloat16 val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16*>(address),
            static_cast<__nv_bfloat16>(val));
}
#endif

/* Used by UpSampleBicubic2d.cu */
template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static void upsample_increment_value_bounded(
    scalar_t* data, int batch, int channel, int height, int width, int y, int x,
    accscalar_t value, int64_t strides0, int64_t strides1, int64_t strides2,
    int64_t strides3) {
  int access_y = max(min(y, height - 1), 0);
  int access_x = max(min(x, width - 1), 0);
  /* TODO: result here is truncated to scalar_t,
     check: https://github.com/pytorch/pytorch/pull/19630#discussion_r281426912
   */
  const int64_t offset = batch * strides0 + channel * strides1 +
                         access_y * strides2 + access_x * strides3;
  gpuAtomicAddNoRet(reinterpret_cast<scalar_t*>(&data[offset]),
                    static_cast<scalar_t>(value));
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_convolution1(
    accscalar_t x, accscalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_convolution2(
    accscalar_t x, accscalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename accscalar_t>
__device__ __forceinline__ static void get_cubic_upsampling_coefficients(
    accscalar_t coeffs[4], accscalar_t t) {
  accscalar_t A = -0.75;

  accscalar_t x1 = t;
  coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);

  // opposite coefficients
  accscalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0, A);
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_interp1d(
    scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, accscalar_t t) {
  accscalar_t coeffs[4];
  get_cubic_upsampling_coefficients<accscalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bicubic2d_out_frame(
    const int num_elements, const int batchsize, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const accscalar_t height_scale,
    const accscalar_t width_scale, const bool align_corners, scalar_t* idata,
    scalar_t* odata, const int64_t istrides0, const int64_t istrides1,
    const int64_t istrides2, const int64_t istrides3, const int64_t ostrides0,
    const int64_t ostrides1, const int64_t ostrides2, const int64_t ostrides3) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= num_elements) {
    return;
  }
  // Special case: input and output are the same size, just copy
  const int output_x = index % output_width;
  const int output_y = index / output_width;
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; c++) {
        int64_t offset = istrides0 * n + istrides1 * c + istrides2 * output_y +
                         istrides3 * output_x;
        idata[offset] = odata[offset];
      }
    }
    return;
  }
  // Interpolation kernel
  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int in_x = floorf(real_x);
  accscalar_t t_x = real_x - in_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int in_y = floorf(real_y);
  accscalar_t t_y = real_y - in_y;

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; c++) {
      accscalar_t coefficients[4];

      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x - 1,
                istrides0, istrides1, istrides2, istrides3),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 0,
                istrides0, istrides1, istrides2, istrides3),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 1,
                istrides0, istrides1, istrides2, istrides3),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 2,
                istrides0, istrides1, istrides2, istrides3),
            t_x);
      }
      int64_t offset = ostrides0 * n + ostrides1 * c + ostrides2 * output_y +
                       ostrides3 * output_x;
      odata[offset] = static_cast<scalar_t>(
          cubic_interp1d(coefficients[0], coefficients[1], coefficients[2],
                         coefficients[3], t_y));
    }
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bicubic2d_backward_out_frame(
    const int num_elements, const int batchsize, const int channels,
    const int input_height, const int input_width, const int output_height,
    const int output_width, const accscalar_t height_scale,
    const accscalar_t width_scale, const bool align_corners, scalar_t* grad_idata,
    scalar_t* grad_odata, const int64_t istrides0, const int64_t istrides1,
    const int64_t istrides2, const int64_t istrides3, const int64_t ostrides0,
    const int64_t ostrides1, const int64_t ostrides2, const int64_t ostrides3) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= num_elements) {
    return;
  }

  const int output_x = index % output_width;
  const int output_y = index / output_width;
  // special case: output_xust copy
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        int64_t offset = istrides0 * n + istrides1 * c + istrides2 * output_y +
                         istrides3 * output_x;
        grad_idata[offset] = grad_odata[offset];
      }
    }
    return;
  }

  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int input_x = floorf(real_x);
  accscalar_t t_x = real_x - input_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int input_y = floorf(real_y);
  accscalar_t t_y = real_y - input_y;
  
  accscalar_t x_coeffs[4];
  accscalar_t y_coeffs[4];

  get_cubic_upsampling_coefficients(x_coeffs, t_x);
  get_cubic_upsampling_coefficients(y_coeffs, t_y);

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; ++c) {
      int64_t offset = ostrides0 * n + ostrides1 * c + ostrides2 * output_y +
                       ostrides3 * output_x;
      scalar_t out_value = grad_odata[offset];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsample_increment_value_bounded<scalar_t, accscalar_t>(
              grad_idata, n, c, input_height, input_width, input_y - 1 + i,
              input_x - 1 + j, out_value * y_coeffs[i] * x_coeffs[j], istrides0,
              istrides1, istrides2, istrides3);
        }
      }
    }
  }
}

}  // namespace

template <typename scalar_t>
void* GetResizeBicubicKernel() {
  return reinterpret_cast<void*>(
      &upsample_bicubic2d_out_frame<scalar_t, float>);
}

template <typename scalar_t>
void* GetResizeBicubicGradKernel() {
  return reinterpret_cast<void*>(
      &upsample_bicubic2d_backward_out_frame<scalar_t, float>);
}


template void* GetResizeBicubicKernel<Eigen::half>();
template void* GetResizeBicubicKernel<Eigen::bfloat16>();
template void* GetResizeBicubicKernel<float>();

template void* GetResizeBicubicGradKernel<Eigen::half>();
template void* GetResizeBicubicGradKernel<Eigen::bfloat16>();
template void* GetResizeBicubicGradKernel<float>();

}  // namespace xla::gpu
