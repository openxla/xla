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
#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Input/output buffer slice pair for keys or values in a CUB radix sort.
struct CubSortBufferPair {
  ShapedSlice in;
  ShapedSlice out;
};

// Value-typed parameters describing a CUDA CUB radix sort to lower into a flat
// ThunkSequence of CustomKernelThunks. Key and optional value element types are
// derived from `keys.in.shape` and `values->in.shape`.
struct CubSortThunkEmitParams {
  int devices_in_process = 1;
  bool descending = false;
  int64_t batch_size = 1;
  CubSortBufferPair keys;
  std::optional<CubSortBufferPair> values;
  ShapedSlice scratch;
};

// Emits a flat ThunkSequence of embedded-CUBIN CustomKernelThunks implementing
// a CUDA CUB radix sort (requires num_items > 0 and segment_size >= 2).
// `make_thunk_info` produces a fresh ThunkInfo (with unique ThunkId and
// profile annotation) for each emitted thunk in the returned sequence.
absl::StatusOr<ThunkSequence> EmitCudaCubSortThunkSequence(
    const stream_executor::DeviceDescription& device_description,
    const CubSortThunkEmitParams& params,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    CubSortPlanOptions plan_options = {});

// Native custom-call handler lowering `xla.gpu.ext.cub_sort_keys` and
// `xla.gpu.ext.cub_sort_pairs` directly into embedded-CUBIN
// CustomKernelThunks.
absl::StatusOr<ThunkSequence> EmitCudaCubSortNativeCustomCall(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_THUNK_EMITTER_H_
