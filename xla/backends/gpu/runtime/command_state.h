/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_

#include "xla/tsl/lib/gtl/int_type.h"
#include "absl/functional/function_ref.h"
#include "absl/container/flat_hash_map.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

// Forward declare.
class CommandBufferCmd;

// A base class for externally managed command state.
//
// Commands can be executed concurrently on many stream executors (underlying
// devices) and command buffers. Managing per-executor state can become
// expensive as it requires synchronization. Furthermore the number of command
// buffers command is recorded into is unbounded as they come and go (command
// buffers evicted and reconstructed) which makes it hard to manage the
// lifetime of resources attached to command buffers.
//
// Externally managed state (owned and synchronized by CommandBufferThunk)
// allows commands to attach a piece of information to command buffer in a
// safe and performant way.
//
// To make a command stateful, it needs a `CommandStateManager` indirection:
//
//   class MyCommand : public CommandBufferCmd {
//     public:
//
//     // Container for mutable state required for command execution.
//     struct MyState : CommandState {
//       ...
//     };
//
//     absl::StatusOr<Command*> Record(...) override {
//       // Attach a new instance of `MyState` to the active command buffer.
//       // When a command buffer will be destroyed, the state will be destroyed
//       // as well automatically by XLA runtime. If this command will be
//       // recorded into another command buffer, the state will be re-created
//       // automatically using the provided callback.
//       //
//       // CommandBufferThunk guarantees that the state manger passed to a
//       // command recording function is tied to exactly the same command/
//       // buffer that command is recording into.
//       MyState* my_state = record_params.state.GetOrCreate<MyState>(this,
//         [&] { // create a new instnace of `MyState` });
//       ...
//     }
//
//   };
//
class CommandState {
 public:
  virtual ~CommandState() = default;
};

// Command state manager owns command state recorded into the `command_buffer`
// by commands in a command sequence. State is created lazily the first time
// command is recorded using a given state manager (into a given command
// buffer). State manager is owned by a command buffer thunk together with
// the command buffer itself and they are destroyed together, which ties state
// lifetime to the command buffer.
//
// Note that the same command can be recorded as a part of multiple iterations
// of unrolled loop, and for this reason the state can be attached to a
// concreate iteration index.
class CommandStateManager {
 public:
  explicit CommandStateManager(
      const stream_executor::CommandBuffer* command_buffer);

  template <typename S>
  S* GetOrNull(const CommandBufferCmd* cmd, int64_t unroll_iteration = 0);

  template <typename S>
  S* GetOrCreate(const CommandBufferCmd* cmd,
                 absl::FunctionRef<std::unique_ptr<S>()> create,
                 int64_t unroll_iteration = 0);

  template <typename S>
  S* GetOrCreate(const CommandBufferCmd* cmd, int64_t unroll_iteration = 0);

  const stream_executor::CommandBuffer* command_buffer() const {
    return command_buffer_;
  }

 private:
  // We use TypeId to distinguish between different state types.
  TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);

  static TypeId GetNextTypeId();

  template <typename T>
  static TypeId GetTypeId() {
    static const TypeId id = GetNextTypeId();
    return id;
  }

  CommandState* GetOrNull(const CommandBufferCmd* cmd, TypeId type_id,
                          int64_t unroll_iteration);

  CommandState* GetOrCreate(
      const CommandBufferCmd* cmd, TypeId type_id, int64_t unroll_iteration,
      absl::FunctionRef<std::unique_ptr<CommandState>()> create);

  using Key = std::tuple<const CommandBufferCmd*, TypeId, int64_t>;
  absl::flat_hash_map<Key, std::unique_ptr<CommandState>> state_;

  const stream_executor::CommandBuffer* command_buffer_;
};

//===----------------------------------------------------------------------===//
// CommandStateManager templates implementation
//===----------------------------------------------------------------------===//

template <typename S>
S* CommandStateManager::GetOrNull(const CommandBufferCmd* cmd,
                                  int64_t unroll_iteration) {
  static_assert(std::is_base_of_v<CommandState, S>);
  return static_cast<S*>(GetOrNull(cmd, GetTypeId<S>(), unroll_iteration));
}

template <typename S>
S* CommandStateManager::GetOrCreate(
    const CommandBufferCmd* cmd, absl::FunctionRef<std::unique_ptr<S>()> create,
    int64_t unroll_iteration) {
  static_assert(std::is_base_of_v<CommandState, S>);
  return static_cast<S*>(GetOrCreate(cmd, GetTypeId<S>(), unroll_iteration,
                                     [&] { return create(); }));
}

template <typename S>
S* CommandStateManager::GetOrCreate(const CommandBufferCmd* cmd,
                                    int64_t unroll_iteration) {
  return GetOrCreate<S>(
      cmd, [] { return std::make_unique<S>(); }, unroll_iteration);
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_
