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

#include "xla/backends/gpu/runtime/traced_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

TracedCommandBuffer::TracedCommandBuffer(const Command* trace_cmd,
                                         Command::BufferUses buffers,
                                         int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) {
    allocs_indices.insert(buffer.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::InlinedVector<se::DeviceAddressBase, 4>
TracedCommandBuffer::CollectAllocs(
    const BufferAllocations* buffer_allocation) const {
  absl::InlinedVector<se::DeviceAddressBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }
  return allocs;
}

std::optional<size_t> TracedCommandBuffer::FindMatchingIndex(
    absl::Span<const se::DeviceAddressBase> allocs) const {
  for (size_t i = 0; i < capacity_; ++i) {
    if (ABSL_PREDICT_TRUE(absl::c_equal(entries_[i].recorded_allocs, allocs) &&
                          entries_[i].command_buffer)) {
      return i;
    }
  }
  return std::nullopt;
}

TracedCommandBuffer::Entry& TracedCommandBuffer::ShiftToFront(size_t i) {
  if (i == 0) {
    return entries_[0];
  }
  Entry entry = std::move(entries_[i]);
  do {
    entries_[i] = std::move(entries_[i - 1]);
  } while (--i > 0);
  return entries_[0] = std::move(entry);
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::Trace(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority, bool force_retrace) {
  auto allocs = CollectAllocs(buffer_allocation);

  // (Re)trace into entry `i`, optionally updating its `recorded_allocs`, and
  // apply `priority` if non-default. Returns the index passed in for chaining
  // through `ShiftToFront`.
  auto retrace_into = [&](size_t i,
                          bool assign_allocs) -> absl::StatusOr<size_t> {
    ASSIGN_OR_RETURN(
        entries_[i].command_buffer,
        se::TraceCommandBufferFactory::Create(executor, stream, trace));
    if (assign_allocs) {
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
    }
    if (priority != se::StreamPriority::Default) {
      TF_RETURN_IF_ERROR(entries_[i].command_buffer->SetPriority(priority));
    }
    return i;
  };

  // Matching entry exists: either return it as-is, or re-trace into the same
  // slot (allocations unchanged) when forced.
  if (auto i = FindMatchingIndex(allocs); i.has_value()) {
    if (!force_retrace) {
      VLOG(6) << "Command buffer trace cache hit for command "
              << trace_cmd_->ToString(0);
      return ShiftToFront(*i).command_buffer.get();
    }
    ASSIGN_OR_RETURN(size_t idx, retrace_into(*i, /*assign_allocs=*/false));
    VLOG(6) << "Command buffer trace cache force-retrace for command "
            << trace_cmd_->ToString(0);
    return ShiftToFront(idx).command_buffer.get();
  }

  // No match: trace into the first empty slot if one exists.
  for (size_t i = 0; i < capacity_; ++i) {
    if (entries_[i].command_buffer == nullptr) {
      ASSIGN_OR_RETURN(size_t idx, retrace_into(i, /*assign_allocs=*/true));
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString(0);
      return ShiftToFront(idx).command_buffer.get();
    }
  }

  // Cache full: evict the oldest (last) entry and trace into it.
  ASSIGN_OR_RETURN(size_t idx,
                   retrace_into(capacity_ - 1, /*assign_allocs=*/true));
  VLOG(6) << "Command buffer trace cache does replacement for command "
          << trace_cmd_->ToString(0);
  return ShiftToFront(idx).command_buffer.get();
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  return Trace(buffer_allocation, executor, stream, trace, priority,
               /*force_retrace=*/false);
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::ForceTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  return Trace(buffer_allocation, executor, stream, trace, priority,
               /*force_retrace=*/true);
}

bool TracedCommandBuffer::HasEntry(
    const BufferAllocations* buffer_allocation) const {
  return FindMatchingIndex(CollectAllocs(buffer_allocation)).has_value();
}

}  // namespace xla::gpu
