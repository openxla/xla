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

#ifndef XLA_BACKENDS_GPU_RUNTIME_OPTIONAL_BUFFER_USE_H_
#define XLA_BACKENDS_GPU_RUNTIME_OPTIONAL_BUFFER_USE_H_

#include <optional>

#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::gpu {

// Helpers for `Thunk::buffer_uses()` overrides of thunks with optional operands
// (e.g. optional bias, scale or workspace buffers). They replace repetitive
// `if (x.has_value()) res.push_back(BufferUse::Read(...));` blocks.

// Factory for a `BufferUse`, e.g. `&BufferUse::Read` or `&BufferUse::Write`.
using BufferUseFactory = BufferUse (*)(BufferAllocation::Slice slice,
                                       Shape shape);

// Appends `make_use(slice->slice, slice->shape)` to `uses` if `slice` has a
// value. `T` is `ShapedSlice` or `const ShapedSlice`.
//
// Example:
//
//   AppendOptionalBufferUse(uses, &BufferUse::Read, bias_);
template <typename T>
void AppendOptionalBufferUse(Thunk::BufferUses& uses, BufferUseFactory make_use,
                             const std::optional<T>& slice) {
  if (slice.has_value()) {
    uses.push_back(make_use(slice->slice, slice->shape));
  }
}

// Appends `make_use(*slice, *shape)` to `uses` if `slice` has a value. `shape`
// must have a value whenever `slice` has one.
//
// Example:
//
//   AppendOptionalBufferUse(uses, &BufferUse::Read, bias_buffer_,
//                           descriptor_.bias_shape);
void AppendOptionalBufferUse(
    Thunk::BufferUses& uses, BufferUseFactory make_use,
    const std::optional<BufferAllocation::Slice>& slice,
    const std::optional<Shape>& shape);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_OPTIONAL_BUFFER_USE_H_
