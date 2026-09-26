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

#include "xla/backends/gpu/runtime/optional_buffer_use.h"

#include <optional>

#include "absl/log/check.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::gpu {

void AppendOptionalBufferUse(
    Thunk::BufferUses& uses, BufferUseFactory make_use,
    const std::optional<BufferAllocation::Slice>& slice,
    const std::optional<Shape>& shape) {
  if (!slice.has_value()) {
    return;
  }
  CHECK(shape.has_value()) << "Missing shape for buffer slice "
                           << slice->ToString();
  uses.push_back(make_use(*slice, *shape));
}

}  // namespace xla::gpu
