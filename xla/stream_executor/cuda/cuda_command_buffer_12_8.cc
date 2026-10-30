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

#include <cstdint>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::gpu {

absl::Status CudaCommandBuffer::SetPreferredClusterDimensions(
    GraphNodeHandle node_handle, const ClusterDim& dimensions) {
  CUlaunchAttributeValue value{};
  value.preferredClusterDim.x = static_cast<uint32_t>(dimensions.x);
  value.preferredClusterDim.y = static_cast<uint32_t>(dimensions.y);
  value.preferredClusterDim.z = static_cast<uint32_t>(dimensions.z);
  CUgraphNode node = absl::bit_cast<CUgraphNode>(node_handle);
  CUresult result = cuGraphKernelNodeSetAttribute(
      node, CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &value);
  if (result == CUDA_ERROR_INVALID_VALUE) {
    // Some drivers support preferred clusters for launches but not graph nodes.
    // Distinguish an unsupported attribute from an invalid launch
    // configuration.
    CUlaunchAttributeValue current{};
    if (cuGraphKernelNodeGetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &current) ==
        CUDA_ERROR_INVALID_VALUE) {
      return absl::UnimplementedError(
          "CUDA driver does not support preferred cluster graph attributes");
    }
  }
  return cuda::ToStatus(
      result, "Failed to set kernel node preferred cluster dimensions");
}

}  // namespace stream_executor::gpu
