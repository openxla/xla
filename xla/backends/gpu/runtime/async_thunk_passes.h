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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ASYNC_THUNK_PASSES_H_
#define XLA_BACKENDS_GPU_RUNTIME_ASYNC_THUNK_PASSES_H_

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// A collection of thunk passes for optimizing async execution of XLA:GPU. At
// compile time we rely on various passes (i.e. latency hiding scheduler) to
// come up with an optimized schedule that overlaps compute with communication
// (or in some cases compute with another compute). However the resulting thunk
// sequence might still have optimization opportunities, depending on the exact
// scheduling and buffer assignment produced by the compiler, and by doing one
// more round of optimizations on top of the thunk sequence we can correct
// suboptimal decisions made when compiling HLO programs.

//===----------------------------------------------------------------------===//
// RemoveRedundantAsyncThunkPass.
//===----------------------------------------------------------------------===//

// Removes redundant async start/done thunk pairs from a thunk sequence. When an
// AsyncDoneThunk immediately follows its matching AsyncStartThunk, there is no
// actual asynchronous execution: the done thunk just waits for an event that
// was recorded right before it. In this case, we can inline the nested thunk
// sequence from the start thunk, avoiding the overhead of creating an async
// execution scope (recording events and synchronizing streams).
//
// Example:
//
//   %start = AsyncStartThunk([thunk-sequence])
//   %done  = AsyncDoneThunk(%start)
//
// is replaced by:
//
//   [thunk-sequence]
//
class RemoveRedundantAsyncThunkPass : public ThunkPassInterface {
 public:
  absl::string_view name() const override { return "remove-redundant-async"; }

  absl::StatusOr<bool> Run(ThunkSequence* thunk_sequence,
                           const DebugOptions& debug_options,
                           const HloModule* absl_nullable hlo_module,
                           const se::DeviceDescription& device_info,
                           ThunkPassBufferAllocator& allocator) override;
};

//===----------------------------------------------------------------------===//
// ExpandAsyncScopeThunkPass.
//===----------------------------------------------------------------------===//

// Expands async execution scopes by moving AsyncStartThunk as far up the thunk
// sequence as possible and the corresponding AsyncDoneThunk as far down as
// possible, maximizing the window of asynchronous overlap.
//
// Motivation: The latency hiding scheduler operates at the HLO level before
// buffer assignment, so it makes scheduling decisions without knowing the final
// buffer layout. After buffer assignment, some async start/done pairs may end
// up closer together than necessary -- e.g., a collective-start might be placed
// just one or two thunks before its done, leaving little room to overlap
// communication with compute. By widening the gap between start and done at the
// thunk level, this pass recovers latency hiding opportunities that the HLO
// scheduler could not exploit.
//
// Example:
//
//   %kernel0 = KernelThunk()
//   %kernel1 = KernelThunk()
//   %start   = AsyncStartThunk()
//   %kernel2 = KernelThunk()
//   %done    = AsyncDoneThunk(%start)
//   %kernel3 = KernelThunk()
//
// is replaced by:
//
//   %start   = AsyncStartThunk()
//   %kernel0 = KernelThunk()
//   %kernel1 = KernelThunk()
//   %kernel2 = KernelThunk()
//   %kernel3 = KernelThunk()
//   %done    = AsyncDoneThunk(%start)
//
// The pass uses `buffer_uses()` and `resource_uses()` (collected transitively
// via the `Thunk::Walk` API to account for nested thunk sequences) to detect
// read-write conflicts between thunks. A start or done thunk can be moved past
// another thunk only when their buffer and resource use sets do not conflict.
//
// AsyncDoneThunk does not report buffer or resource uses of its own; for
// conflict resolution purposes this pass assigns it the same buffer and
// resource uses as the corresponding AsyncStartThunk, since the done thunk
// logically waits for the start thunk's work to complete and therefore touches
// the same buffers and resources.
class ExpandAsyncScopeThunkPass : public ThunkPassInterface {
 public:
  absl::string_view name() const override { return "expand-async-scope"; }

  absl::StatusOr<bool> Run(ThunkSequence* thunk_sequence,
                           const DebugOptions& debug_options,
                           const HloModule* absl_nullable hlo_module,
                           const se::DeviceDescription& device_info,
                           ThunkPassBufferAllocator& allocator) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ASYNC_THUNK_PASSES_H_
