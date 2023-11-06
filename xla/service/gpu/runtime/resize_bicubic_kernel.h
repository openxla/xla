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

#ifndef XLA_SERVICE_GPU_RUNTIME_RESIZE_BICUBIC_KERNEL_H_
#define XLA_SERVICE_GPU_RUNTIME_RESIZE_BICUBIC_KERNEL_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/status/status.h"
#include "xla/runtime/memref_view.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Input:
//  - input: [batch_size, channels, input_height, input_width] dtype
// Output:
//  - output: [batch_size, channels, output_height, output_width] dtype
absl::Status RunResizeBicubicImpl(::tensorflow::se::gpu::GpuStreamHandle stream,
                                  int threads_per_block_limit,
                                  xla::runtime::StridedMemrefView input,
                                  xla::runtime::StridedMemrefView output,
                                  bool align_corners, float scales_h,
                                  float scales_w);
// Input:
// - input: [batch_size, channels, output_height, output_width] dtype
// Output:
// - output: [batch_size, channels, input_height, input_width] dtype
absl::Status RunResizeBicubicGradImpl(
    ::tensorflow::se::gpu::GpuStreamHandle stream, int threads_per_block_limit,
    xla::runtime::StridedMemrefView grad_input,
    xla::runtime::StridedMemrefView grad_output, bool align_corners,
    float scales_h, float scales_w);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_RESIZE_BICUBIC_KERNEL_H_
