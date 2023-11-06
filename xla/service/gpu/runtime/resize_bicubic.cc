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

#include "xla/service/gpu/runtime/resize_bicubic.h"

#include <stdint.h>

#include <cstddef>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/runtime/resize_bicubic_kernel.h"
// #include "xla/runtime/custom_call_registry.h"

#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
using ::xla::runtime::CustomCall;
using ::xla::runtime::StridedMemrefView;

static absl::Status ResizeBicubicImpl(
    const ServiceExecutableRunOptions* run_options, StridedMemrefView input,
    StridedMemrefView output, bool align_corners) {
  float scales_h =
      static_cast<float>(output.sizes[2]) / static_cast<float>(input.sizes[2]);
  float scales_w =
      static_cast<float>(output.sizes[3]) / static_cast<float>(input.sizes[3]);
  se::StreamExecutor* executor = run_options->stream()->parent();
  return RunResizeBicubicImpl(
      se::gpu::AsGpuStreamValue(run_options->stream()),
      executor->GetDeviceDescription().threads_per_block_limit(), input, output,
      align_corners, scales_h, scales_w);
}

static absl::Status ResizeBicubicGradImpl(
    const ServiceExecutableRunOptions* run_options,
    StridedMemrefView grad_output, StridedMemrefView grad_input,
    bool align_corners) {
  float scales_h = static_cast<float>(grad_output.sizes[2]) /
                   static_cast<float>(grad_input.sizes[2]);
  float scales_w = static_cast<float>(grad_output.sizes[3]) /
                   static_cast<float>(grad_input.sizes[3]);
  se::StreamExecutor* executor = run_options->stream()->parent();
  return RunResizeBicubicGradImpl(
      se::gpu::AsGpuStreamValue(run_options->stream()),
      executor->GetDeviceDescription().threads_per_block_limit(), grad_input,
      grad_output, align_corners, scales_h, scales_w);
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ResizeBicubic, FunctionWrapper<ResizeBicubicImpl>(), checks,
    CustomCall::Bind("__gpu$ResizeBicubic")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()  // input
        .Arg<StridedMemrefView>()  // output
        .Attr<bool>("align_corners"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ResizeBicubicGrad, FunctionWrapper<ResizeBicubicGradImpl>(), checks,
    CustomCall::Bind("__gpu$ResizeBicubicGrad")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()  // grad_output
        .Arg<StridedMemrefView>()  // grad_input
        .Attr<bool>("align_corners"));

void RegisterResizeBicubicCustomCall(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("__gpu$ResizeBicubic", ResizeBicubic);
  registry.Register("__gpu$ResizeBicubicGrad", ResizeBicubicGrad);
}

}  // namespace xla::gpu
