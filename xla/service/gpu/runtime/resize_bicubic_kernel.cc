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

#include "xla/xla_data.pb.h"
#include "xla/service/gpu/runtime/resize_bicubic_kernel.h"

#include <algorithm>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/service/gpu/runtime/resize_bicubic_kernel_common.h"

namespace xla::gpu {
using ::stream_executor::gpu::GpuStreamHandle;

float compute_scales_value(double scale, int64_t src_size, int64_t dst_size) {
  return (scale > 0.) ? (float)(1.0 / scale) : (float)src_size / dst_size;
}

float area_pixel_compute_scale(int input_size, int output_size,
                               bool align_corners, double scale) {
  if (align_corners) {
    if (output_size > 1) {
      return (float)(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<float>(0);
    }
  } else {
    return compute_scales_value(scale, input_size, output_size);
  }
}

absl::Status RunResizeBicubicImpl(::tensorflow::se::gpu::GpuStreamHandle stream,
                                  int threads_per_block_limit,
                                  xla::runtime::StridedMemrefView input,
                                  xla::runtime::StridedMemrefView output,
                                  bool align_corners, float scales_h,
                                  float scales_w) {
  int output_height = output.sizes[2];
  int output_width = output.sizes[3];

  int input_height = input.sizes[2];
  int input_width = input.sizes[3];

  int num_output_elements = output_height * output_width;
  int threads_per_block = std::min(threads_per_block_limit, 1024);
  int blocks_per_grid =
      (num_output_elements + threads_per_block - 1) / threads_per_block;
  int shmem_size = 0;

  // Get scaling factors
  float rheight = area_pixel_compute_scale(input_height, output_height,
                                           align_corners, scales_h);
  float rwidth = area_pixel_compute_scale(input_width, output_width,
                                          align_corners, scales_w);
  void* kernel = nullptr;
  // kernel = GetResizeBicubicKernel<float>();
  xla::PrimitiveType dtype = static_cast<xla::PrimitiveType>(input.dtype);
  switch (dtype) {
    case xla::PrimitiveType::F32:
      kernel = GetResizeBicubicKernel<float>();
      break;

    case xla::PrimitiveType::BF16:
      kernel = GetResizeBicubicKernel<Eigen::bfloat16>();
      break;

    case xla::PrimitiveType::F16:
      kernel = GetResizeBicubicKernel<Eigen::half>();
      break;

    default:
      return absl::UnimplementedError(
          absl::StrCat("ResizeBicubic not implemented for this dtype: ", dtype,
                       " PrimitiveType::F32: ", xla::PrimitiveType::F32,
                       "equals ", xla::PrimitiveType::F32 == dtype));
  }

  absl::Span<const int64_t>* isizes = &input.sizes;
  absl::Span<const int64_t>* osizes = &output.sizes;
  int batchsize = (*isizes)[0];
  int channels = (*isizes)[1];
  int64_t istrides0 = input.strides[0];
  int64_t istrides1 = input.strides[1];
  int64_t istrides2 = input.strides[2];
  int64_t istrides3 = input.strides[3];
  int64_t ostrides0 = output.strides[0];
  int64_t ostrides1 = output.strides[1];
  int64_t ostrides2 = output.strides[2];
  int64_t ostrides3 = output.strides[3];
  void* kernel_args[] = {&num_output_elements, &batchsize,   &channels,
                         &input_height,        &input_width, &output_height,
                         &output_width,        &rheight,     &rwidth,
                         &align_corners,       &input.data,  &output.data,
                         &istrides0,           &istrides1,   &istrides2,
                         &istrides3,           &ostrides0,   &ostrides1,
                         &ostrides2,           &ostrides3};
  cudaError_t launch_status =
      cudaLaunchKernel(kernel, blocks_per_grid, threads_per_block, kernel_args,
                       shmem_size, stream);
  if (launch_status != cudaSuccess) {
    return absl::InternalError(absl::StrCat("Failed to launch kernel: ",
                                            cudaGetErrorString(launch_status)));
  }
  return absl::OkStatus();
}

absl::Status RunResizeBicubicGradImpl(
    ::tensorflow::se::gpu::GpuStreamHandle stream, int threads_per_block_limit,
    xla::runtime::StridedMemrefView grad_input,
    xla::runtime::StridedMemrefView grad_output, bool align_corners,
    float scales_h, float scales_w) {
  int output_height = grad_output.sizes[2];
  int output_width = grad_output.sizes[3];

  int input_height = grad_input.sizes[2];
  int input_width = grad_input.sizes[3];
  int num_output_elements = output_height * output_width;
  int threads_per_block = std::min(threads_per_block_limit, 1024);
  int blocks_per_grid =
      (num_output_elements + threads_per_block - 1) / threads_per_block;
  int shmem_size = 0;

  // Get scaling factors
  float rheight = area_pixel_compute_scale(input_height, output_height,
                                           align_corners, scales_h);
  float rwidth = area_pixel_compute_scale(input_width, output_width,
                                          align_corners, scales_w);

  void* kernel = nullptr;
  // kernel = GetResizeBicubicGradKernel<float>();
  xla::PrimitiveType dtype = static_cast<xla::PrimitiveType>(grad_input.dtype);
  switch (dtype) {
    case xla::PrimitiveType::F32:
      kernel = GetResizeBicubicGradKernel<float>();
      break;

    case xla::PrimitiveType::BF16:
      kernel = GetResizeBicubicGradKernel<Eigen::bfloat16>();
      break;

    case xla::PrimitiveType::F16:
      kernel = GetResizeBicubicGradKernel<Eigen::half>();
      break;

    default:
      return absl::UnimplementedError(absl::StrCat(
          "ResizeBicubicGrad not implemented for this dtype: ", dtype,
          " PrimitiveType::F32: ", xla::PrimitiveType::F32, "equal",
          dtype == xla::PrimitiveType::F32));
  }

  absl::Span<const int64_t>* isizes = &grad_input.sizes;
  absl::Span<const int64_t>* osizes = &grad_output.sizes;
  int batchsize = (*isizes)[0];
  int channels = (*isizes)[1];
  int64_t istrides0 = grad_input.strides[0];
  int64_t istrides1 = grad_input.strides[1];
  int64_t istrides2 = grad_input.strides[2];
  int64_t istrides3 = grad_input.strides[3];
  int64_t ostrides0 = grad_output.strides[0];
  int64_t ostrides1 = grad_output.strides[1];
  int64_t ostrides2 = grad_output.strides[2];
  int64_t ostrides3 = grad_output.strides[3];
  void* kernel_args[] = {
      &num_output_elements, &batchsize,       &channels,
      &input_height,        &input_width,     &output_height,
      &output_width,        &rheight,         &rwidth,
      &align_corners,       &grad_input.data, &grad_output.data,
      &istrides0,           &istrides1,       &istrides2,
      &istrides3,           &ostrides0,       &ostrides1,
      &ostrides2,           &ostrides3};
  cudaError_t launch_status =
      cudaLaunchKernel(kernel, blocks_per_grid, threads_per_block, kernel_args,
                       shmem_size, stream);
  if (launch_status != cudaSuccess) {
    return absl::InternalError(absl::StrCat("Failed to launch kernel: ",
                                            cudaGetErrorString(launch_status)));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
