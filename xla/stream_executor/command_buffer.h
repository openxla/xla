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

#ifndef XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "tsl/platform/errors.h"

namespace stream_executor {

class Stream;

//===----------------------------------------------------------------------===//
// CommandBuffer
//===----------------------------------------------------------------------===//

// Command buffer represent a "bundle of work items" for StreamExecutor device
// that can be submitted with one API call, e.g. command buffer might have
// multiple device kernels and synchronization barriers between them. Command
// buffers allow to amortize the cost of launching "work" on device by building
// it on the host ahead of time without expensive interaction with underlying
// device.

// CommandBuffer lowers CommandBufferCmdSequence to a GPU graph, commands in the
// sequence have dependencies that can specified by command index in the
// sequence. CommandBuffer implementation will convert dependencies through
// command index to node dependencies in the implementation.
class CommandBuffer {
 public:
  using Index = int64_t;
  using CmdIndexSet = absl::flat_hash_set<Index>;
  static std::string CmdIndexSetToString(const CmdIndexSet& index_set) {
    std::ostringstream oss;
    oss << "CmdIndexSet: {";
    bool first = true;
    for (const auto& index : index_set) {
      if (!first) {
        oss << ", ";
      }
      oss << index;
      first = false;
    }
    oss << "}";
    return oss.str();
  }

  // Builder constructs nested command buffers owned by a parent command buffer.
  using Builder = std::function<absl::Status(CommandBuffer*)>;

  CommandBuffer() = default;
  virtual ~CommandBuffer() = default;

  CommandBuffer(const CommandBuffer&) = delete;
  void operator=(const CommandBuffer&) = delete;

  // Command buffer state:
  //
  //   (1) kCreate:    a new command buffer under construction
  //   (2) kUpdate:    updating a previously finalized command buffer
  //   (3) kFinalized: command buffer ready for execution
  //
  // Supported state transitions:
  //
  //   (1) Finalize: (kCreate|kUpdate) -> kFinalized
  //   (2) Update:   kFinalized -> kUpdate
  //
  enum class State { kCreate, kUpdate, kFinalized };

  // Command buffers have two modes of execution:
  //
  //   (1) kPrimary: command buffer can be submitted for execution via
  //                 StreamExecutor APIs
  //   (2) kNested:  command buffer can be executed only within a primary
  //                 command buffer
  //
  enum class Mode { kPrimary, kNested };

  friend absl::string_view ModeToString(Mode mode) {
    switch (mode) {
      case CommandBuffer::Mode::kPrimary:
        return "primary";
      case CommandBuffer::Mode::kNested:
        return "nested";
    }
  }

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  // Adds an execution barrier that depends on the commands in dep_idxes.
  virtual absl::Status EmptyOp(Index cmd_idx, const CmdIndexSet& dep_idxes) = 0;

  // Adds an execution barrier that depends on all tail nodes (nodes that no
  // other nodes depend on them) in current graph.
  virtual absl::Status Barrier() = 0;

  // Adds a kernel launch command that depends on the commands in
  // dep_idxes.
  virtual absl::Status Launch(Index cmd_idx, const CmdIndexSet& dep_idxes,
                              const ThreadDim& threads, const BlockDim& blocks,
                              const Kernel& kernel, const KernelArgs& args) = 0;

  // Adds a kernel launch command that depends on all tail nodes (nodes that no
  // other nodes depend on them) in current graph.
  virtual absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                              const Kernel& kernel, const KernelArgs& args) = 0;

  // Type-safe wrapper for launching typed kernels that depends on the commands
  // in dep_idxes. Notice that the order of arguments is different do
  // disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  absl::Status Launch(Index cmd_idx, const CmdIndexSet& dep_idxes,
                      const ThreadDim& threads, const BlockDim& blocks,
                      const TypedKernel<Params...>& kernel, Args... args);

  // Type-safe wrapper for launching typed kernels that depends on all tail
  // nodes in current graph.
  template <typename... Params, typename... Args>
  absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                      const TypedKernel<Params...>& kernel, Args... args);

  // Adds a nested command buffer that depends on the commands in
  // dep_idxes.
  virtual absl::Status AddNestedCommandBuffer(Index cmd_idx,
                                              const CmdIndexSet& dep_idxes,
                                              const CommandBuffer& nested) = 0;

  // Adds a nested command buffer that depends on all tail nodes (nodes that no
  // other nodes depend on them) in current graph.
  virtual absl::Status AddNestedCommandBuffer(const CommandBuffer& nested) = 0;

  // Adds a device-to-device memory copy that depends on the commands in
  // dep_idxes.
  virtual absl::Status MemcpyDeviceToDevice(Index cmd_idx,
                                            const CmdIndexSet& dep_idxes,
                                            DeviceMemoryBase* dst,
                                            const DeviceMemoryBase& src,
                                            uint64_t size) = 0;

  // Adds a device-to-device memory copy that depends on all tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  virtual absl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                            const DeviceMemoryBase& src,
                                            uint64_t size) = 0;

  // Adds a memset command that depends on the commands in dep_idxes.
  virtual absl::Status Memset(Index cmd_idx, const CmdIndexSet& dep_idxes,
                              DeviceMemoryBase* dst, BitPattern bit_pattern,
                              size_t num_elements) = 0;

  // Adds a memset command that depends on all tail nodes (nodes that no other
  // nodes depend on them) in current graph.
  virtual absl::Status Memset(DeviceMemoryBase* dst, BitPattern bit_pattern,
                              size_t num_elements) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if `pred` value is `true`.
  virtual absl::Status If(Index cmd_idx, const CmdIndexSet& dep_idxes,
                          DeviceMemory<bool> pred, Builder then_builder) = 0;

  // Adds a conditional If operation that depends on all tail nodes (nodes that
  // no other nodes depend on them) in current graph.
  virtual absl::Status If(DeviceMemory<bool> pred, Builder then_builder) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if `pred` value is `true`, or a command buffer
  // constructed by `else_builder` if `pred` is `false`.
  virtual absl::Status IfElse(Index cmd_idx, const CmdIndexSet& dep_idxes,
                              DeviceMemory<bool> pred, Builder then_builder,
                              Builder else_builder) = 0;

  // Adds a conditional IfElse operation that depends on all tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  virtual absl::Status IfElse(DeviceMemory<bool> pred, Builder then_builder,
                              Builder else_builder) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `branches` builder at `index`. If `index` is out of range, then it
  // will run a conditional command buffer constructed by the last builder.
  //
  // See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
  virtual absl::Status Case(Index cmd_idx, const CmdIndexSet& dep_idxes,
                            DeviceMemory<int32_t> index,
                            std::vector<Builder> branches) = 0;

  // Adds a conditional Case operation that depends on all tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  virtual absl::Status Case(DeviceMemory<int32_t> index,
                            std::vector<Builder> branches) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `body_builder` exactly `num_iteration` times. This means the
  // condition is known at compile time (`num_iteration` < `loop_counter`), and
  // does not require a `cond_builder`.
  virtual absl::Status For(Index cmd_idx, const CmdIndexSet& dep_idxes,
                           int32_t num_iteration,
                           DeviceMemory<int32_t> loop_counter,

                           Builder body_builder) = 0;

  // Adds a conditional For operation that depends on all tail nodes (nodes that
  // no other nodes depend on them) in current graph.
  virtual absl::Status For(int32_t num_iteration,
                           DeviceMemory<int32_t> loop_counter,
                           Builder body_builder) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `cond_builder` that must update `pred` value, and then depending on
  // the value might execute command buffer constructed by `body_builder` and
  // `cond_builder`. Will continue while `pred` value (which is continuously
  // updated by `cond_builder`) is `true`.
  //
  // In pseudocode:
  //
  //   cond_builder()
  //   while(pred):
  //     body_builder()
  //     cond_builder()
  //
  // We use execution scope builder for the condition because we have to build
  // condition twice: (1) before the conditional node in the scope defined by
  // `execution_scope_id` (2) inside the loop body with default execution scope.
  virtual absl::Status While(Index cmd_idx, const CmdIndexSet& dep_idxes,
                             DeviceMemory<bool> pred, Builder cond_builder,
                             Builder body_builder) = 0;

  // Adds a conditional While operation that depends on all tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  virtual absl::Status While(DeviceMemory<bool> pred, Builder cond_builder,
                             Builder body_builder) = 0;

  // Submits the command buffer for execution.
  virtual absl::Status Submit(Stream* stream) {
    return absl::UnimplementedError("Not implemented for this command buffer.");
  }

  //--------------------------------------------------------------------------//
  // Command buffer state management API
  //--------------------------------------------------------------------------//

  // Finalizes command buffer and makes it executable. Once command buffer is
  // finalized no commands can be added to it.
  virtual absl::Status Finalize() = 0;

  // Begins command buffer update. Command buffer update should be finalized
  // before it can be executed.
  virtual absl::Status Update() = 0;

  // Returns command buffer execution mode.
  virtual Mode mode() const = 0;

  // Returns command buffer state.
  virtual State state() const = 0;

  //--------------------------------------------------------------------------//
  // Command buffer tracing API
  //--------------------------------------------------------------------------//
 private:
  friend class TraceCommandBufferFactory;
  // Tracing APIs are private because they do not compose with command buffer
  // updates. Instead of tracing directly into the command buffer users should
  // create traced command buffers using factory methods and add them to primary
  // command buffers as nested operations.

  // Traces `function` invocation by recording all operations on the `stream`
  // into the command buffer. Command buffer must be empty.
  virtual absl::Status Trace(Stream* stream,
                             absl::AnyInvocable<absl::Status()> function) = 0;
};

//===----------------------------------------------------------------------===//
// CommandBuffer templates implementation below
//===----------------------------------------------------------------------===//

template <typename... Params, typename... Args>
inline absl::Status CommandBuffer::Launch(Index cmd_idx,
                                          const CmdIndexSet& dep_idxes,
                                          const ThreadDim& threads,
                                          const BlockDim& blocks,
                                          const TypedKernel<Params...>& kernel,
                                          Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(
      Launch(cmd_idx, dep_idxes, threads, blocks, *kernel, *kernel_args));
  return absl::OkStatus();
}

template <typename... Params, typename... Args>
inline absl::Status CommandBuffer::Launch(const ThreadDim& threads,
                                          const BlockDim& blocks,
                                          const TypedKernel<Params...>& kernel,

                                          Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(Launch(threads, blocks, *kernel, *kernel_args));
  return absl::OkStatus();
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
