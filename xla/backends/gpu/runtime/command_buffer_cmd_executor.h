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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_EXECUTOR_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Forward declaration.
class CommandBufferCmd;

//===----------------------------------------------------------------------===//
// CmdOrThunk
//===----------------------------------------------------------------------===//

// A variant that can hold either a CommandBufferCmd or a Thunk*.
// This allows polymorphic dispatch between the two types.
class CmdOrThunk {
 public:
  using BufferUseVector = absl::InlinedVector<BufferUse, 4>;

  explicit CmdOrThunk(std::unique_ptr<CommandBufferCmd> cmd);
  explicit CmdOrThunk(Thunk* thunk);

  // Delete copy operations (unique_ptr member makes this move-only).
  CmdOrThunk(const CmdOrThunk&) = delete;
  CmdOrThunk& operator=(const CmdOrThunk&) = delete;

  // Default move operations.
  CmdOrThunk(CmdOrThunk&&) = default;
  CmdOrThunk& operator=(CmdOrThunk&&) = default;

  // Destructor must be declared here and defined in .cc to handle unique_ptr.
  ~CmdOrThunk();

  CommandBufferCmd* cmd() const;
  Thunk* thunk() const;

  std::variant<const CommandBufferCmd*, const Thunk*> cmd_or_thunk() const;

  absl::Status Prepare(const Thunk::PrepareParams& params);
  absl::Status Initialize(const Thunk::InitializeParams& params);
  absl::Status Record(const Thunk::ExecuteParams& execute_params) const;

  // Returns async events if the command/thunk is an async collective.
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const;

  // Returns true if command/thunk is an async start command.
  bool IsAsyncStart() const;

  // Returns true if command/thunk is an async done command.
  bool IsAsyncDone() const;

  bool IsCollective() const;

  bool command_buffer_requires_initialization() const;
  bool command_buffer_support_loop_unroll() const;
  bool command_buffer_force_update() const;

  BufferUseVector buffer_uses() const;

  std::shared_ptr<Resource> token() const;

  void add_resouce_use(ResourceUse resource_use);

  ResourceUseVector resources() const;

  absl::string_view profile_annotation() const;

  se::StreamPriority command_buffer_priority() const;

  void set_command_buffer_priority(se::StreamPriority priority);

  std::string ToString(int indent = 0) const;

  // Return the dependencies of the command from within the executor, if the
  // command is a source command, it will return the executor dependencies
  // specified in record_params.
  std::vector<const se::CommandBuffer::Command*> CommandBufferDependencies(
      const CommandBufferParams& record_params) const;

 private:
  std::variant<std::unique_ptr<CommandBufferCmd>, Thunk*> cmd_or_thunk_;

  ResourceUseVector resources_;

  // The token resource is used to specify additional dependency across
  // commands, like control dependency across HLO operators, and LHS
  // scheduling dependency.
  std::shared_ptr<Resource> token_;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

// A sequence of commands (corresponds to a ThunkSequence from the Thunk API).
class CommandBufferCmdSequence : public std::vector<CmdOrThunk> {
 public:
  template <typename CmdT, typename... Args>
  void Emplace(Args&&... args) {
    this->emplace_back(
        CmdOrThunk(std::make_unique<CmdT>(std::forward<Args>(args)...)));
  }

  std::string ToString(int indent = 0) const;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdExecutor
//===----------------------------------------------------------------------===//

// Command executor is responsible for recording commands sequence into the
// underlying command buffer and setting up dependencies between commands.
class CommandBufferCmdExecutor {
 public:
  CommandBufferCmdExecutor() = default;
  CommandBufferCmdExecutor(CommandBufferCmdExecutor&&) = default;
  CommandBufferCmdExecutor& operator=(CommandBufferCmdExecutor&&) = default;

  // Synchronization mode defines how much concurrency is allowed between
  // commands in the sequence.
  enum class SynchronizationMode {
    // Serializes execution of all commands recorded into the command buffer
    // by adding a dependency between them.
    kSerialize,

    // Relies on execution graph to insert dependencies between commands
    // that have buffer of resource conflicts, and building a DAG of commands.
    kConcurrent,

    // Uses the same latency hidden scheduling results used in the thunk
    // scheduling.
    kLHS,
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, SynchronizationMode mode) {
    switch (mode) {
      case SynchronizationMode::kSerialize:
        sink.Append("serialize");
        break;
      case SynchronizationMode::kConcurrent:
        sink.Append("concurrent");
        break;
      case SynchronizationMode::kLHS:
        sink.Append("lhs");
        break;
    }
  }

  // Creates a command executor from a sequence of commands using given
  // synchronization mode.
  static absl::StatusOr<CommandBufferCmdExecutor> Create(
      CommandBufferCmdSequence commands,
      SynchronizationMode synchronization_mode);

  // Prepares all commands added to a sequence.
  absl::Status Prepare(const Thunk::PrepareParams& params);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params);

  // Records commands into the command buffer. This method automatically
  // switches between `RecordCreate` or `RecordUpdate` depending on the
  // command buffer state.

  // This Record function allows multiple CommandbufferCmdEXecutor to be
  // recorded into a single command buffer. e.g. we can have Executor A, B, C
  // to be recorded into the same command buffer in the order of A -> B -> C.
  // In this pattern, B's source commands will depend on A's sink commands,
  // and C's source commands will also depend on B's sink commands.

  // If record_action is `RecordCreate`, it will set up initial
  // dependencies for recorded commands by the `dependencies` parameter.
  // If record_action is `RecordUpdate`, it will only update previously
  // recorded commands' dependencies, no other actions.

  // Records commands into the command buffer. Uses record_params.is_finalize
  // to determine whether to finalize the command buffer after recording.
  // The CommandBufferParams is accessed via execute_params.record_params.
  absl::Status Record(const Thunk::ExecuteParams& execute_params) const;

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<BufferUse>& buffer_uses() const;

  // Returns buffer allocations indices referenced by commands in this
  // sequence.
  absl::Span<const BufferAllocation::Index> allocs_indices() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool command_buffer_requires_initialization() const;

  bool command_buffer_force_update() const;

  bool command_buffer_support_loop_unroll() const;

  // Returns all commands associated with the given ids on a certain unroll
  // iteration.
  std::vector<const se::CommandBuffer::Command*> SourceCommands(
      const CommandBufferParams& record_params) const;

  std::vector<const se::CommandBuffer::Command*> SinkCommands(
      const CommandBufferParams& record_params) const;

  // Renders the execution graph using default renderer. Returns url of the
  // rendered graph, or an error if rendering failed.
  absl::StatusOr<std::string> RenderExecutionGraph();

  // Returns true if the given command is a source command (has no
  // dependencies within this executor).
  bool IsSource(std::variant<const CommandBufferCmd*, const Thunk*> cmd) const;

  // Returns dependencies of the given command.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const CommandBufferParams& record_params,
      std::variant<const CommandBufferCmd*, const Thunk*> cmd) const;

  using CreateCommand =
      absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>()>;

  using UpdateCommand = absl::FunctionRef<absl::Status(
      const se::CommandBuffer::Command* absl_nonnull command)>;

  absl::Status HandleCmdCreateOrUpdate(
      CommandBufferParams& record_params,
      std::variant<const CommandBufferCmd*, const Thunk*> cmd,
      CreateCommand create_command, UpdateCommand update_command) const;

 private:
  // We use index into the `commands_` vector as a command id.
  using CommandId = int64_t;

  // A state associated with commands in the sequence. We rely on this state
  // to efficiently update command recorded into the command buffer.
  struct RecordState : public CommandBufferState {
    const se::CommandBuffer::Command* command;
  };

  CommandBufferCmdExecutor(SynchronizationMode synchronization_mode,
                           CommandBufferCmdSequence commands,
                           std::optional<ExecutionGraph> execution_graph);

  absl::Status CheckCommandBufferState(
      se::CommandBuffer* absl_nonnull command_buffer,
      se::CommandBuffer::State expected_state) const;

  // Returns true if command has no dependencies.
  bool IsSource(CommandId id) const;

  // Returns true if command is not a dependency of any other commands.
  bool IsSink(CommandId id) const;

  // Returns dependencies of the command with the given id.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const CommandBufferParams& record_params, CommandId id) const;

  SynchronizationMode synchronization_mode_;
  CommandBufferCmdSequence commands_;

  // In automatic synchronization mode we build an execution graph for the
  // sequence of commands and use it to set up dependencies between commands.
  std::optional<ExecutionGraph> execution_graph_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<BufferUse> buffers_;

  // Unique buffer allocations indices referenced by all commands in this
  // sequence (sorted by the buffer allocation index).
  std::vector<BufferAllocation::Index> allocs_indices_;

  // A mapping from command id to unique buffer allocations indices referenced
  // by the command (sorted by the buffer allocation index).
  std::vector<std::vector<BufferAllocation::Index>> cmd_allocs_indices_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_EXECUTOR_H_

