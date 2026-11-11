/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/command_buffer_kernels.h"

#include <cstdint>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda_device_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {
namespace {

// Sets conditional handle values to 1 for the active branch and 0 for all
// others in [0, num_handles). Handles are passed in batches of up to 8; when
// enable_default_condition is true (the final batch), an out of range index
// activates handles[num_handles - 1].
__global__ void SetCaseCondition(
    cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
    cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3,
    cudaGraphConditionalHandle h4, cudaGraphConditionalHandle h5,
    cudaGraphConditionalHandle h6, cudaGraphConditionalHandle h7,
    const void* index, bool index_is_bool, int32_t index_offset,
    int32_t num_handles, bool enable_default_condition) {
  const cudaGraphConditionalHandle handles[8] = {h0, h1, h2, h3,
                                                 h4, h5, h6, h7};

  const int32_t index_int32 =
      index_is_bool
          ? static_cast<int32_t>(*static_cast<const uint8_t*>(index) != 0)
          : *static_cast<const int32_t*>(index);

  int64_t effective_index =
      static_cast<int64_t>(index_int32) - static_cast<int64_t>(index_offset);
  if (enable_default_condition &&
      (index_int32 < 0 || effective_index >= num_handles)) {
    effective_index = num_handles - 1;
  }

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    if (i < num_handles) {
      cudaGraphSetConditional(handles[i], effective_index == i ? 1u : 0u);
    }
  }
}

__global__ void SetWhileCondition(cudaGraphConditionalHandle handle,
                                  const uint8_t* predicate) {
  cudaGraphSetConditional(handle, *predicate != 0 ? 1u : 0u);
}

__global__ void NoOp() {}

}  // namespace

absl::StatusOr<KernelLoaderSpec> GetSetCaseConditionKernelLoaderSpec() {
  return KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&SetCaseCondition), "set_case_condition", 13);
}

absl::StatusOr<KernelLoaderSpec> GetSetWhileConditionKernelLoaderSpec() {
  return KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&SetWhileCondition), "set_while_condition", 2);
}

absl::StatusOr<KernelLoaderSpec> GetNoOpKernelLoaderSpec() {
  return KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&NoOp), "noop", 0);
}

}  // namespace stream_executor::cuda
