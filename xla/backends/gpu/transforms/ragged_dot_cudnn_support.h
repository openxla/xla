/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_DOT_CUDNN_SUPPORT_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_DOT_CUDNN_SUPPORT_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Returns an allowed Decision iff cuDNN positively reports an execution plan
// for the cuDNN fusion that RaggedDotFusionRewriter would build for
// `ragged_dot` on `device_description`. The ragged dot is cloned into a scratch
// module and RaggedDotFusionRewriter is run on it, then cuDNN is probed:
//  - with a `stream_exec`, against its live cuDNN handle
//    (CuDnnFusionCompiler::GetAvailablePlanCount);
//  - without one (deviceless/AOT) and if the loaded cuDNN supports deviceless
//    compilation, from the device description alone
//    (CuDnnFusionCompiler::SupportsFusionDeviceless);
//  - otherwise there is no way to probe.
// Anything but a positive answer -- kUnsupported, kUnknown (e.g. cuDNN headers
// older than 9.15), a probe failure or no way to probe -- returns a forbidden
// Decision with the reason, so the
// caller falls back to the regular dense expansion.
//
// The probe runs on the shapes/layouts `ragged_dot` has now; layout assignment
// may later pick different layouts for the real fusion.
Decision IsRaggedDotSupportedByCuDnn(
    const HloInstruction& ragged_dot,
    const stream_executor::DeviceDescription& device_description,
    stream_executor::StreamExecutor* stream_exec = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_DOT_CUDNN_SUPPORT_H_
