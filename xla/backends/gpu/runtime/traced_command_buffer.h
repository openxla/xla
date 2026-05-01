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

#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

// A cache for traced command buffers that will re-trace on change in buffer
// allocations that are relevant for `buffers` passed to constructor. We use a
// very simple most-recently-used cache of traced command buffers as in practice
// subsequent calls to XLA executable tend to reuse the same allocations.
class TracedCommandBuffer : public CommandState {
 public:
  explicit TracedCommandBuffer(const Command* trace_cmd,
                               Command::BufferUses buffers,
                               int64_t capacity = 16);

  // Returns true if a cache entry already exists for the buffer addresses
  // implied by `buffer_allocation`. Read-only; does not trace or mutate the
  // cache. Used by collective operations to vote on cache state across ranks
  // before deciding whether to use a cached graph or re-trace together.
  bool HasEntry(const BufferAllocations* buffer_allocation) const;

  // Returns cached command buffer traced using the same buffer addresses or
  // traces and caches a new command buffer using user provided callback.
  absl::StatusOr<se::CommandBuffer*> GetOrTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

  // Always traces and updates the cache, even if a matching entry already
  // exists. Used by collective operations on a coordinated cache miss: every
  // rank must re-trace together (to keep NCCL calls symmetric) AND the cache
  // must be populated so the next iteration can take the fast path.
  absl::StatusOr<se::CommandBuffer*> ForceTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

 private:
  struct Entry {
    std::vector<se::DeviceAddressBase> recorded_allocs;
    std::unique_ptr<se::CommandBuffer> command_buffer;
  };

  // Collects current device addresses for the allocations tracked by this
  // cache from `buffer_allocation`, in the order of `allocs_indices_`.
  absl::InlinedVector<se::DeviceAddressBase, 4> CollectAllocs(
      const BufferAllocations* buffer_allocation) const;

  // Returns the index of the cache entry whose recorded allocations match
  // `allocs` and which holds a valid command buffer, or std::nullopt if no
  // such entry exists.
  std::optional<size_t> FindMatchingIndex(
      absl::Span<const se::DeviceAddressBase> allocs) const;

  // Moves entry at `i` to the front (LRU update) and shifts entries in
  // [0, i) one slot to the right. Returns a reference to the now-front entry.
  Entry& ShiftToFront(size_t i);

  // Shared body for `GetOrTraceCommandBuffer` and `ForceTraceCommandBuffer`.
  // - When `force_retrace` is false: if a matching entry exists, returns the
  //   cached command buffer; otherwise traces into an empty/evicted slot.
  // - When `force_retrace` is true: always invokes `trace`. If a matching
  //   entry exists it is overwritten in place; otherwise traces into an
  //   empty/evicted slot. Used to keep ranks symmetric on a coordinated
  //   cache miss.
  absl::StatusOr<se::CommandBuffer*> Trace(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority, bool force_retrace);

  std::vector<BufferAllocation::Index> allocs_indices_;
  const Command* trace_cmd_;
  int64_t capacity_;
  std::vector<Entry> entries_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_
