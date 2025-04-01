/* Copyright 2025 The OpenXLA Authors.

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

#include <optional>

#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"

namespace xla {
namespace gpu {

std::optional<std::unique_ptr<mlir::Pass>> MaybeCreateConvertFloatPass(
    const se::DeviceDescription& device_description) {
  se::SemanticVersion ptx_version =
      nvptx::DetermineHighestSupportedPtxVersionFromCudaVersion(
          device_description.runtime_version());
  se::CudaComputeCapability cc = device_description.cuda_compute_capability();

  // FP8 conversion intrinsics are available on sm89 since ptx 8.1
  // Older ptx versions only support FP8 conversion for sm90
  if ((ptx_version >= se::SemanticVersion(8, 1, 0) && cc.IsAtLeast(8, 9)) ||
      (ptx_version >= se::SemanticVersion(7, 8, 0) && cc.IsAtLeast(9, 0))) {
    return CreateConvertFloatNvidiaPass();
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
