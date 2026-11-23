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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_THUNK_EMITTER_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_THUNK_EMITTER_H_

#include <cstdint>
#include <optional>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Parameters describing a CUDA CUB radix sort to lower into a flat
// ThunkSequence of CustomKernelThunks (or DeviceToDeviceCopyThunks for
// segment_size == 1).
struct CubSortThunkEmitParams {
  // Produces a fresh ThunkInfo (with unique ThunkId and profile annotation)
  // for each emitted thunk in the returned sequence.
  absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info;
  const stream_executor::DeviceDescription& device_description;
  int devices_in_process = 1;
  PrimitiveType key_type = PRIMITIVE_TYPE_INVALID;
  std::optional<PrimitiveType> value_type;
  bool descending = false;
  int64_t batch_size = 1;
  ShapedSlice keys_in;
  ShapedSlice keys_out;
  std::optional<ShapedSlice> values_in;
  std::optional<ShapedSlice> values_out;
  ShapedSlice scratch;

  // Optional overrides forwarded to PlanCubSortExecution for unit tests.
  std::optional<int32_t> portion_size_override;
  std::optional<int32_t> max_grid_x_override;
};

// Emits a flat ThunkSequence of embedded-CUBIN CustomKernelThunks (or
// DeviceToDeviceCopyThunks for segment_size == 1, or empty for num_items == 0)
// implementing a CUDA CUB radix sort.
//
// In non-CUDA builds this resolves to a stub returning UnimplementedError
// (selected via BUILD `if_cuda_is_configured`, without `#ifdef`s in C++).
absl::StatusOr<ThunkSequence> EmitCudaCubSortThunkSequence(
    const CubSortThunkEmitParams& params);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_THUNK_EMITTER_H_
