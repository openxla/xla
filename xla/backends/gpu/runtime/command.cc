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

#include "xla/backends/gpu/runtime/command.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
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
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/debug_options_flags.h"
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

namespace {

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

// A cache for traced command buffers that re-traces on change in buffer
// allocations relevant for a given command. Uses a simple MRU cache.
class TracedCommandBuffer : public CommandState {
 public:
  explicit TracedCommandBuffer(const Command* trace_cmd,
                               Command::BufferUses buffers,
                               int64_t capacity = 16);

  absl::StatusOr<se::CommandBuffer*> GetOrTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

 private:
  std::vector<BufferAllocation::Index> allocs_indices_;
  struct Entry {
    std::vector<se::DeviceAddressBase> recorded_allocs;
    std::unique_ptr<se::CommandBuffer> command_buffer;
  };
  const Command* trace_cmd_;
  int64_t capacity_;
  std::vector<Entry> entries_;
};

TracedCommandBuffer::TracedCommandBuffer(const Command* trace_cmd,
                                         Command::BufferUses buffers,
                                         int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) {
    allocs_indices.insert(buffer.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  absl::InlinedVector<se::DeviceAddressBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  auto shift_right = [&](size_t i) -> Entry& {
    if (i == 0) return entries_[0];
    Entry entry = std::move(entries_[i]);
    do {
      entries_[i] = std::move(entries_[i - 1]);
    } while (--i > 0);
    return entries_[0] = std::move(entry);
  };

  for (size_t i = 0; i < capacity_; ++i) {
    if (ABSL_PREDICT_TRUE(absl::c_equal(entries_[i].recorded_allocs, allocs) &&
                          entries_[i].command_buffer)) {
      VLOG(6) << "Command buffer trace cache hit for command "
              << trace_cmd_->ToString(0);
      return shift_right(i).command_buffer.get();
    }
    if (entries_[i].command_buffer == nullptr) {
      TF_ASSIGN_OR_RETURN(
          entries_[i].command_buffer,
          se::TraceCommandBufferFactory::Create(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
      if (priority != se::StreamPriority::Default) {
        TF_RETURN_IF_ERROR(entries_[i].command_buffer->SetPriority(priority));
      }
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString(0);
      return shift_right(i).command_buffer.get();
    }
  }

  TF_ASSIGN_OR_RETURN(
      entries_[capacity_ - 1].command_buffer,
      se::TraceCommandBufferFactory::Create(executor, stream, trace));
  entries_[capacity_ - 1].recorded_allocs.assign(allocs.begin(), allocs.end());
  VLOG(6) << "Command buffer trace cache does replacement for command "
          << trace_cmd_->ToString(0);
  return shift_right(capacity_ - 1).command_buffer.get();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Command::Handle
//===----------------------------------------------------------------------===//

/*static*/ absl::StatusOr<const se::CommandBuffer::Command*> Command::Handle(
    RecordAction action,
    absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>(
        absl::Span<const se::CommandBuffer::Command* const>)>
        create_command,
    absl::FunctionRef<absl::Status(const se::CommandBuffer::Command*)>
        update_command) {
  if (auto* create = std::get_if<RecordCreate>(&action)) {
    return create_command(create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&action)) {
    TF_RETURN_IF_ERROR(update_command(update->command));
    return update->command;
  }
  return absl::InternalError("Invalid record action");
}

//===----------------------------------------------------------------------===//
// Command::RecordTracedCommand
//===----------------------------------------------------------------------===//

absl::StatusOr<const se::CommandBuffer::Command*> Command::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd = record_params.state.GetOrCreate<TracedCommandBuffer>(
      this, command_buffer, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffer_uses(),
            debug_options.xla_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace, priority()));

  VLOG(5) << "Record traced command into command buffer: " << command_buffer;
  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// CommandType helpers
//===----------------------------------------------------------------------===//

std::string CommandTypeString(CommandType type) {
  switch (type) {
#define CASE_CMD_STRING(enum_name, cmd_name, ...) \
  case CommandType::enum_name:                    \
    return cmd_name;
    XLA_GPU_COMMAND_LIST(CASE_CMD_STRING)
#undef CASE_CMD_STRING
  }
}

bool IsCollectiveCommand(CommandType type) {
  switch (type) {
    case CommandType::kAllGatherCmd:
    case CommandType::kAllReduceCmd:
    case CommandType::kAllToAllCmd:
    case CommandType::kCollectiveBroadcastCmd:
    case CommandType::kCollectivePermuteCmd:
    case CommandType::kRaggedAllToAllCmd:
    case CommandType::kReduceScatterCmd:
    case CommandType::kRecvCmd:
    case CommandType::kSendCmd:
      return true;
    default:
      return false;
  }
}

}  // namespace xla::gpu
