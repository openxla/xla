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

#include "xla/backends/gpu/runtime/command_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandThunk
//===----------------------------------------------------------------------===//

std::vector<const se::CommandBuffer::Command*>
CommandThunk::CommandBufferDependencies(
    const CommandBufferParams& record_params) const {
  // If no executor is set in record_params, return empty dependencies.
  if (record_params.executor == nullptr) {
    return {};
  }

  // If the current command is a source command, use the executor dependencies
  // specified in record_params.
  if (record_params.executor->IsSource(this)) {
    return record_params.external_dependencies;
  }

  // Otherwise, follow the same method as CommandThunkExecutor::Dependencies
  // to get the dependencies.
  return record_params.executor->Dependencies(record_params, this);
}

absl::Status CommandThunk::HandleCmdCreateOrUpdate(
    CommandBufferParams& record_params, CreateCommand create_command,
    UpdateCommand update_command) {
  // Delegate to the executor to handle the create or update.
  return record_params.executor->HandleCmdCreateOrUpdate(
      record_params, this, create_command, update_command);
}

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

std::string CommandBufferCmdSequence::ToString(int indent) const {
  std::string result;
  for (const auto& cmd : *this) {
    result += cmd->ToString(indent) + "\n";
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CommandOperation (helper for ExecutionGraph)
//===----------------------------------------------------------------------===//

namespace {
// An adaptor from CommandThunk to ExecutionGraph::Operation for building an
// execution graph from a command sequence.
class CommandOperation : public ExecutionGraph::Operation {
 public:
  explicit CommandOperation(CommandThunk::BufferUseVector buffers,
                            const CommandThunk* cmd)
      : name_(absl::StrFormat("cmd %s: %s", cmd->ToString(0),
                              cmd->profile_annotation())),
        buffers_(std::move(buffers)),
        cmd_(cmd),
        resources_(cmd_->resources()) {}

  absl::string_view name() const final { return name_; }
  absl::Span<const BufferUse> BufferUses() const final { return buffers_; }
  absl::Span<const ResourceUse> ResourceUses() const final {
    return resources_;
  }
  void add_resouce_use(ResourceUse resource_use) {
    resources_.push_back(resource_use);
  }

  const CommandThunk* cmd() const { return cmd_; }

  std::string ToString() const final {
    std::vector<std::string> resource_reprs;
    resource_reprs.reserve(resources_.size());
    for (const ResourceUse& use : resources_) {
      absl::string_view access =
          use.access() == ResourceUse::kRead ? "read" : "write";
      absl::string_view kind = Resource::ToString(use.resource()->kind());
      resource_reprs.push_back(
          absl::StrFormat("%s@%p(%s)", kind, use.resource().get(), access));
    }
    return absl::StrFormat("%s resources=[%s]", cmd_->ToString(0),
                           absl::StrJoin(resource_reprs, ", "));
  }

 private:
  std::string name_;
  CommandThunk::BufferUseVector buffers_;
  const CommandThunk* cmd_;
  ResourceUseVector resources_;

  // The token resource is used to specify dependency other than buffer data
  // flow, e.g, LHS topology will use token resouce to specify dependency across
  // commands.
  std::shared_ptr<Resource> token_;
};

static std::vector<CommandOperation> CreateCommandOperations(
    const CommandBufferCmdSequence& commands,
    CommandBufferCmdExecutor::SynchronizationMode synchronization_mode) {
  std::vector<CommandOperation> operations;
  operations.reserve(commands.size());
  VLOG(3) << "CreateCommandOperations with synchronization mode: "
          << (synchronization_mode ==
                      CommandBufferCmdExecutor::SynchronizationMode::kConcurrent
                  ? "Concurrent"
                  : "LHS");
  if (synchronization_mode ==
      CommandBufferCmdExecutor::SynchronizationMode::kConcurrent) {
    // For concurrent synchronization mode, pass in buffer and resouces for
    // dependency inference.
    for (const auto& cmd : commands) {
      operations.emplace_back(cmd->buffer_uses(), cmd.get());
    }
  }

  if (synchronization_mode ==
      CommandBufferCmdExecutor::SynchronizationMode::kLHS) {
    // For LHS mode, don't pass in buffers.
    // Will use token resource to specify dependency across commands.
    for (const auto& cmd : commands) {
      operations.emplace_back(CommandThunk::BufferUseVector{}, cmd.get());
    }

    // Find the matching async start command for an async done command by
    // comparing async_events pointers.
    auto find_async_start_id = [&](int64_t done_id) -> int64_t {
      // Get async_events from the done command via virtual method.
      auto done_events = commands[done_id]->async_events();
      if (done_events == nullptr) {
        return -1;
      }
      for (int64_t j = done_id - 1; j >= 0; --j) {
        if (commands[j]->IsAsyncStart() &&
            commands[j]->async_events() == done_events) {
          return j;
        }
      }
      return -1;
    };

    for (int64_t i = 0; i < static_cast<int64_t>(operations.size()); ++i) {
      if (operations[i].cmd()->IsAsyncStart()) {
        for (int64_t j = i - 1; j >= 0; --j) {
          if (operations[j].cmd()->IsAsyncStart()) {
            continue;
          }
          operations[i].add_resouce_use(
              ResourceUse::Read(commands[j]->token()));
          break;
        }
      } else if (operations[i].cmd()->IsAsyncDone()) {
        int64_t async_start_cmd_id = find_async_start_id(i);
        CHECK_NE(async_start_cmd_id, -1);
        operations[i].add_resouce_use(
            ResourceUse::Read(commands[async_start_cmd_id]->token()));
        CHECK_GT(i, 0);
        if ((i - 1) != async_start_cmd_id) {
          operations[i].add_resouce_use(
              ResourceUse::Read(commands[i - 1]->token()));
        }
      } else {
        for (int64_t j = i - 1; j >= 0; --j) {
          if (operations[j].cmd()->IsAsyncStart()) {
            // The first command in the async group does not depend on the async
            // command
            continue;
          }
          operations[i].add_resouce_use(
              ResourceUse::Read(commands[j]->token()));
          break;
        }
      }
    }
  }

  if (VLOG_IS_ON(2)) {
    for (const CommandOperation& op : operations) {
      VLOG(2) << op.ToString();
    }
  }

  return operations;
}
}  // namespace

//===----------------------------------------------------------------------===//
// CommandBufferCmdExecutor
//===----------------------------------------------------------------------===//

absl::StatusOr<CommandBufferCmdExecutor> CommandBufferCmdExecutor::Create(
    CommandBufferCmdSequence commands,
    SynchronizationMode synchronization_mode) {
  std::optional<ExecutionGraph> execution_graph = std::nullopt;

  // In automatic synchronization mode construct an execution graph for the
  // sequence of commands and derive the structure of command dependencies
  // from the buffer use conflicts.
  if (synchronization_mode != SynchronizationMode::kSerialize) {
    auto operations = CreateCommandOperations(commands, synchronization_mode);
    TF_ASSIGN_OR_RETURN(execution_graph,
                        ExecutionGraph::Create<CommandOperation>(operations));
    VLOG(3) << "Execution graph: " << execution_graph->ToString();
  }

  return CommandBufferCmdExecutor(synchronization_mode, std::move(commands),
                                  std::move(execution_graph));
}

CommandBufferCmdExecutor::CommandBufferCmdExecutor(
    SynchronizationMode synchronization_mode, CommandBufferCmdSequence commands,
    std::optional<ExecutionGraph> execution_graph)
    : synchronization_mode_(synchronization_mode),
      commands_(std::move(commands)),
      execution_graph_(std::move(execution_graph)) {
  // Buffer allocations referenced by commands in this sequence.
  absl::btree_set<BufferAllocation::Index> allocs_indices;

  for (const auto& cmd : commands_) {
    absl::btree_set<BufferAllocation::Index> cmd_allocs_indices;

    for (const BufferUse& buffer : cmd->buffer_uses()) {
      buffers_.insert(buffer);
      allocs_indices.insert(buffer.slice().index());
      cmd_allocs_indices.insert(buffer.slice().index());
    }

    // Record buffer allocations indices referenced by the `cmd`.
    cmd_allocs_indices_.emplace_back(cmd_allocs_indices.begin(),
                                     cmd_allocs_indices.end());
  }

  // Record all buffer allocations indices referenced by all commands in this
  // sequence.
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::Status CommandBufferCmdExecutor::Prepare(
    const Thunk::PrepareParams& params) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdExecutor::Initialize(
    const Thunk::InitializeParams& params) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdExecutor::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) const {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  VLOG(3) << "Record " << commands_.size() << " commands into command buffer";

  // Set the executor in record_params for dependency resolution.
  record_params->executor = this;

  if (record_params->command_buffer->state() ==
      se::CommandBuffer::State::kFinalized) {
    TF_RETURN_IF_ERROR(record_params->command_buffer->Update());
  }

  // Check if command `id` has to be updated based on the buffer allocations
  // that changed since the last call to `Record`. We keep intersection vector
  // outside of a lambda to avoid repeated heap allocations on every call.
  std::vector<BufferAllocation::Index> alloc_intersection;
  auto skip_command_update = [&](CommandId id) {
    // If we don't know what allocations changed since the last call to
    // `Record` we must always update the command.
    if (!record_params->updated_allocs) {
      return false;
    }

    // We always update commands that require initialization, even if buffer
    // allocations didn't change.
    const CommandThunk* command = commands_[id].get();
    if (command->command_buffer_requires_initialization() &&
        record_params->is_initialization) {
      return false;
    }

    if (command->command_buffer_force_update()) {
      return false;
    }

    DCHECK(absl::c_is_sorted(*record_params->updated_allocs))
        << "Updated allocs must be sorted: "
        << absl::StrJoin(*record_params->updated_allocs, ", ");

    DCHECK(absl::c_is_sorted(cmd_allocs_indices_[id]))
        << "Command allocs must be sorted: "
        << absl::StrJoin(cmd_allocs_indices_[id], ", ");

    alloc_intersection.clear();
    absl::c_set_intersection(cmd_allocs_indices_[id],
                             *record_params->updated_allocs,
                             std::back_inserter(alloc_intersection));
    return alloc_intersection.empty();
  };

  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  size_t num_skipped_command_updates = 0;

  for (CommandId id = 0; id < static_cast<CommandId>(commands_.size()); ++id) {
    CommandThunk* command = commands_[id].get();

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    // Skip recording collective commands if mock collectives are enabled.
    if (execute_params.mock_collectives && command->IsCollective()) {
      continue;
    }

    // Skip updating command if it doesn't use any of the updated allocations.
    if (skip_command_update(id)) {
      VLOG(3) << "Skip updating command " << command->ToString(0);
      ++num_skipped_command_updates;
      continue;
    }

    TF_RETURN_IF_ERROR(command->ExecuteOnStream(execute_params));
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(1) << absl::StrFormat("Created %d commands in %d μs", commands_.size(),
                             end_micros - start_micros);

  if (record_params->is_finalize) {
    return record_params->command_buffer->Finalize();
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdExecutor::CheckCommandBufferState(
    se::CommandBuffer* command_buffer,
    se::CommandBuffer::State expected_state) const {
  if (command_buffer->state() != expected_state) {
    return Internal("Command buffer must be in %v state, got %v",
                    expected_state, command_buffer->state());
  }
  return absl::OkStatus();
}

bool CommandBufferCmdExecutor::IsSource(CommandId id) const {
  return execution_graph_ ? execution_graph_->is_source(id) : id == 0;
}

bool CommandBufferCmdExecutor::IsSink(CommandId id) const {
  return execution_graph_ ? execution_graph_->is_sink(id)
                          : id + 1 == static_cast<CommandId>(commands_.size());
}

std::vector<const se::CommandBuffer::Command*>
CommandBufferCmdExecutor::SinkCommands(
    const CommandBufferParams& record_params) const {
  std::vector<CommandId> sink_ids;
  if (execution_graph_) {
    auto sink_span = execution_graph_->sink();
    sink_ids.assign(sink_span.begin(), sink_span.end());
  } else {
    sink_ids.push_back(commands_.size() - 1);
  }

  std::vector<const se::CommandBuffer::Command*> sink_commands;
  for (CommandId id : sink_ids) {
    auto* record_state = record_params.state.GetOrNull<RecordState>(
        commands_[id].get(), record_params.command_buffer,
        record_params.unroll_iteration);
    sink_commands.push_back(record_state->command);
  }
  return sink_commands;
}

std::vector<const se::CommandBuffer::Command*>
CommandBufferCmdExecutor::SourceCommands(
    const CommandBufferParams& record_params) const {
  std::vector<CommandId> source_ids;
  if (execution_graph_) {
    auto source_span = execution_graph_->source();
    source_ids.assign(source_span.begin(), source_span.end());
  } else {
    source_ids.push_back(0);
  }

  std::vector<const se::CommandBuffer::Command*> source_commands;
  for (CommandId id : source_ids) {
    auto* record_state = record_params.state.GetOrNull<RecordState>(
        commands_[id].get(), record_params.command_buffer,
        record_params.unroll_iteration);
    source_commands.push_back(record_state->command);
  }
  return source_commands;
}

std::vector<const se::CommandBuffer::Command*>
CommandBufferCmdExecutor::Dependencies(const CommandBufferParams& record_params,
                                       CommandId id) const {
  // Collect commands that are dependencies of the command `id`.
  absl::InlinedVector<CommandId, 4> dependencies_ids;

  if (IsSource(id)) {
    VLOG(2) << "Command ID " << id
            << " is a source command, empty dependencies";
    return {};
  }

  if (execution_graph_) {
    for (const ExecutionGraph::NodeEdge& in_edge :
         execution_graph_->in_edges(id)) {
      dependencies_ids.push_back(in_edge.id);
    }
  } else {
    dependencies_ids.push_back(id - 1);
  }

  // Collect dependencies from the recorded command state.
  std::vector<const se::CommandBuffer::Command*> dependencies;
  for (CommandId dependency_id : dependencies_ids) {
    auto* record_state = record_params.state.GetOrNull<RecordState>(
        commands_[dependency_id].get(), record_params.command_buffer,
        record_params.unroll_iteration);

    // If record state doesn't exist yet or command is null, we need to
    // recursively follow dependencies to find the real command dependencies.
    if (record_state == nullptr || record_state->command == nullptr) {
      // Some commands might end up not recording anything into the command
      // buffer, e.g. memcpy commands where source and destination are the
      // same. We have to follow dependencies of such commands to find the
      // real dependencies, so we don't record a command that is immediately
      // ready to execute, as it will create data races.
      auto deps = Dependencies(record_params, dependency_id);
      dependencies.insert(dependencies.end(), deps.begin(), deps.end());
    } else {
      dependencies.push_back(record_state->command);
    }
  }

  return dependencies;
}

bool CommandBufferCmdExecutor::IsSource(const CommandThunk* cmd) const {
  for (CommandId id = 0; id < static_cast<CommandId>(commands_.size()); ++id) {
    if (cmd == commands_[id].get()) {
      return IsSource(id);
    }
  }
  return false;
}

std::vector<const se::CommandBuffer::Command*>
CommandBufferCmdExecutor::Dependencies(const CommandBufferParams& record_params,
                                       const CommandThunk* cmd) const {
  for (CommandId id = 0; id < static_cast<CommandId>(commands_.size()); ++id) {
    if (cmd == commands_[id].get()) {
      return Dependencies(record_params, id);
    }
  }
  return {};
}

absl::Status CommandBufferCmdExecutor::HandleCmdCreateOrUpdate(
    CommandBufferParams& record_params, const CommandThunk* cmd,
    CreateCommand create_command, UpdateCommand update_command) const {
  CommandStateManager& state = record_params.state;
  se::CommandBuffer* command_buffer = record_params.command_buffer;

  // Check if record state already exists for this command.
  auto* record_state = state.GetOrNull<RecordState>(
      cmd, command_buffer, record_params.unroll_iteration);

  if (record_state == nullptr) {
    // Create new record state and call create_command to record the command.
    record_state = state.GetOrCreate<RecordState>(
        cmd, command_buffer, record_params.unroll_iteration);
    TF_ASSIGN_OR_RETURN(record_state->command, create_command());
  } else {
    // Update existing command using the stored command handle.
    TF_RETURN_IF_ERROR(update_command(record_state->command));
  }

  return absl::OkStatus();
}

const absl::flat_hash_set<BufferUse>& CommandBufferCmdExecutor::buffer_uses()
    const {
  return buffers_;
}

absl::Span<const BufferAllocation::Index>
CommandBufferCmdExecutor::allocs_indices() const {
  return allocs_indices_;
}

bool CommandBufferCmdExecutor::command_buffer_requires_initialization() const {
  return absl::c_any_of(commands_, [](const auto& cmd) {
    return cmd->command_buffer_requires_initialization();
  });
}

bool CommandBufferCmdExecutor::command_buffer_force_update() const {
  return absl::c_any_of(commands_, [](const auto& cmd) {
    return cmd->command_buffer_force_update();
  });
}

bool CommandBufferCmdExecutor::command_buffer_support_loop_unroll() const {
  return absl::c_all_of(commands_, [](const auto& cmd) {
    return cmd->command_buffer_support_loop_unroll();
  });
}

absl::StatusOr<std::string> CommandBufferCmdExecutor::RenderExecutionGraph() {
  ExecutionGraph::Renderer* renderer = ExecutionGraph::GetRenderer();
  if (renderer == nullptr) {
    return Unimplemented("No execution graph renderer registered");
  }

  if (synchronization_mode_ == SynchronizationMode::kSerialize) {
    return Unimplemented(
        "Execution graph rendering is only supported for "
        "concurrent/LHS synchronization mode");
  }

  auto operations = CreateCommandOperations(commands_, synchronization_mode_);
  absl::InlinedVector<const ExecutionGraph::Operation*, 32> operations_ptrs;
  operations_ptrs.reserve(operations.size());
  for (const auto& operation : operations) {
    operations_ptrs.push_back(&operation);
  }

  std::string graph = renderer->GenerateGraphAsString(operations_ptrs);
  return renderer->PublishGraph(graph);
}

}  // namespace xla::gpu
