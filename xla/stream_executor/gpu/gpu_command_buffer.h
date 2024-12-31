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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// GpuCommandBuffer provides platform-specific CommandBuffer implementation
// (it's backed by CUDA or HIP graphs on NVIDIA and AMD devices).
// This class is used to record and execute a sequence of GPU commands
// efficiently by leveraging GPU graphs (CUDA or HIP).

// GpuCommandBuffer lowers CommandBufferCmdSequence to a GPU graph, a
// CommandBufferCmd may lower into one or more graph nodes (command nodes), the
// HEAD nodes for a CommandBufferCmd is the nodes that have no dependencies
// within the command nodes, and the TAIL nodes are the nodes that no other
// nodes within the command nodes depend on them.

// In CommandBufferCmdSequence, dependency is calculated by the data flow across
// CommandBufferCmds, if CmdA depends on CmdB, then when lowering to GPU graph,
// GpuCommandBuffer will create a dependency from the TAIL nodes of CmdB to HEAD
// nodes of CmdA.
class GpuCommandBuffer : public CommandBuffer {
  // GraphNodeHandleOpaque is an opaque type that won't be ODR used, hence
  // doesn't need to fully defined. It's an implementation detail of the
  // GraphNodeHandle defined below.
  struct GraphNodeHandleOpaque;

  // GraphConditionalOpaque is an opaque type that won't be ODR used, hence
  // doesn't need to fully defined. It's an implementation detail of the
  // GraphConditionalHandle defined below.
  struct GraphConditionalOpaque;

 public:
  // A graph node handle is an opaque handle that identifies a graph node in the
  // graph associated with a command buffer. GraphNodeHandles are created by
  // node factory functions and can be referenced in node update functions.
  // The handle has the same properties as a pointer (can be constructed from a
  // nullptr, trivial copyable, POD, etc.), that's why we use a pointer to
  // define it.
  using GraphNodeHandle = GraphNodeHandleOpaque*;
  using GraphNodeHandles = absl::InlinedVector<GraphNodeHandle, 1>;
  using CmdIdxSetOrNodeHandles =
      std::variant<GraphNodeHandles, const CmdIndexSet*>;

  static std::string GraphNodeHandlesToString(const GraphNodeHandles& handles) {
    std::ostringstream oss;
    oss << "GraphNodeHandles: [";
    for (size_t i = 0; i < handles.size(); ++i) {
      oss << handles[i];
      if (i != handles.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    return oss.str();
  }

  static std::string DependenciesToString(
      const CmdIdxSetOrNodeHandles& dependencies) {
    if (std::holds_alternative<GraphNodeHandles>(dependencies)) {
      return GraphNodeHandlesToString(std::get<GraphNodeHandles>(dependencies));
    } else {
      return CmdIndexSetToString(*std::get<const CmdIndexSet*>(dependencies));
    }
  }

  // A graph conditional handle is an opaque handle that is tied to a nested
  // command buffer. Its value determines whether the nested command buffer
  // is executed or not. Set condition functions will update the conditional
  // handles values. The handle has the same properties as a pointer (can be
  // constructed from a nullptr, trivially copyable, POD, etc.), that's why
  // we use a pointer to define it.
  using GraphConditionalHandle = GraphConditionalOpaque*;
  using GraphConditionalHandles = absl::Span<const GraphConditionalHandle>;

  // A handle to a Gpu graph node and a metadata describing its properties. Each
  // command (launch, memcpy, etc.) creates one or more graph nodes.
  struct GpuGraphNodeInfo {
    // A handle to the gpu graph node corresponding to a command.
    GraphNodeHandle handle{};
  };

  // A handle to Gpu graph barrier and metadata describing its properties. Each
  // call to `Barrier` creates a new barrier record.
  struct GpuGraphBarrierInfo {
    // A handle to graph node acting as a barrier that defines execution order.
    // It can be a handle to a `GpuGraphNodeInfo` node or a handle to an empty
    // node created to be a barrier. We try to reuse existing nodes as barriers
    // if possible to reduce the size of constructed gpu graphs.
    GraphNodeHandle handle{};

    // If `true` it means `handle` corresponds to an empty node specifically
    // created to act as an execution barrier, otherwise `handle` points to one
    // of the nodes created for recorded commands.
    bool is_barrier_node = true;

    // Nodes with index smaller than `nodes_offset` are synchronized with this
    // barrier. We use this offset to find nodes added after the last barrier
    // that should be added as dependencies to the next barrier.
    size_t nodes_offset = 0;
  };

  GpuCommandBuffer(Mode mode, StreamExecutor* parent);

  using CommandBuffer::EmptyOp;
  absl::Status EmptyOp(Index cmd_idx, const CmdIndexSet& dep_idxes) {
    return EmptyOp(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes});
  }

  // Adds an execution barrier that depends on all the tail nodes (nodes that no
  // other nodes depend on them) in current graph.
  using CommandBuffer::Barrier;
  absl::Status Barrier() {
    return EmptyOp(cmd_tail_nodes_.size(), graph_tail_nodes());
  }

  using CommandBuffer::Launch;
  absl::Status Launch(Index cmd_idx, const CmdIndexSet& dep_idxes,
                      const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args) {
    return Launch(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, threads, blocks,
                  kernel, args);
  }

  // Adds a kernel launch command that depends on all the tail nodes (nodes that
  // no other nodes depend on them) in current graph.
  absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args) {
    return Launch(cmd_tail_nodes_.size(), graph_tail_nodes(), threads, blocks,
                  kernel, args);
  }

  absl::Status AddNestedCommandBuffer(Index cmd_idx,
                                      const CmdIndexSet& dep_idxes,
                                      const CommandBuffer& nested) {
    return AddNestedCommandBuffer(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes},
                                  nested);
  }

  // Adds a nested command buffer that depends on all the tail nodes (nodes that
  // no other nodes depend on them) in current graph.
  absl::Status AddNestedCommandBuffer(const CommandBuffer& nested) {
    return AddNestedCommandBuffer(cmd_tail_nodes_.size(), graph_tail_nodes(),
                                  nested);
  }

  absl::Status MemcpyDeviceToDevice(Index cmd_idx, const CmdIndexSet& dep_idxes,
                                    DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) {
    return MemcpyDeviceToDevice(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes},
                                dst, src, size);
  }

  // Adds a device-to-device memory copy that depends on all the tail nodes
  // (nodes that no other nodes depend on them) in current graph.
  absl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) {
    return MemcpyDeviceToDevice(cmd_tail_nodes_.size(), graph_tail_nodes(), dst,
                                src, size);
  }

  absl::Status Memset(Index cmd_idx, const CmdIndexSet& dep_idxes,
                      DeviceMemoryBase* dst, BitPattern bit_pattern,
                      size_t num_elements) {
    return Memset(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, dst, bit_pattern,
                  num_elements);
  }

  // Adds a memset command into command buffer that depends on all the tail
  // nodes (nodes that no other nodes depend on them) in current graph.
  absl::Status Memset(DeviceMemoryBase* dst, BitPattern bit_pattern,
                      size_t num_elements) {
    return Memset(cmd_tail_nodes_.size(), graph_tail_nodes(), dst, bit_pattern,
                  num_elements);
  }

  absl::Status If(Index cmd_idx, const CmdIndexSet& dep_idxes,
                  DeviceMemory<bool> predicate, Builder then_builder) {
    return If(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, predicate,
              then_builder);
  }

  // Adds a conditional If operation that depends on all the tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  absl::Status If(DeviceMemory<bool> pred, Builder then_builder) {
    return If(cmd_tail_nodes_.size(), graph_tail_nodes(), pred, then_builder);
  }

  absl::Status IfElse(Index cmd_idx, const CmdIndexSet& dep_idxes,
                      DeviceMemory<bool> predicate, Builder then_builder,
                      Builder else_builder) {
    return IfElse(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, predicate,
                  then_builder, else_builder);
  }

  // Adds a conditional IfElse operation that depends on all the tail nodes
  // (nodes that no other nodes depend on them) in current graph.
  absl::Status IfElse(DeviceMemory<bool> pred, Builder then_builder,
                      Builder else_builder) {
    return IfElse(cmd_tail_nodes_.size(), graph_tail_nodes(), pred,
                  then_builder, else_builder);
  }

  absl::Status Case(Index cmd_idx, const CmdIndexSet& dep_idxes,
                    DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) {
    return Case(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, index, branches);
  }

  // Adds a conditional Case operation that depends on all tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  absl::Status Case(DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) {
    return Case(cmd_tail_nodes_.size(), graph_tail_nodes(), index, branches);
  }

  absl::Status For(Index cmd_idx, const CmdIndexSet& dep_idxes,
                   int32_t num_iteration, DeviceMemory<int32_t> loop_counter,
                   Builder body_builder) {
    return For(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, num_iteration,
               loop_counter, body_builder);
  }

  // Adds a conditional For operation that depends on all tail nodes (nodes that
  // no other nodes depend on them) in current graph.
  absl::Status For(int32_t num_iteration, DeviceMemory<int32_t> loop_counter,
                   Builder body_builder) {
    return For(cmd_tail_nodes_.size(), graph_tail_nodes(), num_iteration,
               loop_counter, body_builder);
  }

  absl::Status While(Index cmd_idx, const CmdIndexSet& dep_idxes,
                     DeviceMemory<bool> pred, Builder cond_builder,
                     Builder body_builder) {
    return While(cmd_idx, CmdIdxSetOrNodeHandles{&dep_idxes}, pred,
                 cond_builder, body_builder);
  }

  // Adds a conditional While operation that depends on all tail nodes (nodes
  // that no other nodes depend on them) in current graph.
  absl::Status While(DeviceMemory<bool> pred, Builder cond_builder,
                     Builder body_builder) {
    return While(cmd_tail_nodes_.size(), graph_tail_nodes(), pred, cond_builder,
                 body_builder);
  }

  absl::Status Finalize() override;
  absl::Status Update() override;
  absl::Status Submit(Stream* stream) override;

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }

  int64_t command_size() { return cmd_tail_nodes_.size(); }

  absl::Span<const GpuGraphNodeInfo> nodes() const { return nodes_; }

  // Returns the list of dependencies for a given node.
  // `node` must be a node added to the current command
  // buffer. The returned node pointer's lifetimes are bound
  // to the current command buffer.
  virtual absl::StatusOr<std::vector<GraphNodeHandle>> GetNodeDependencies(
      GraphNodeHandle node) = 0;

 protected:
  // We track the total number of allocated and alive
  // executable graphs in the process to track the command
  // buffers resource usage. Executable graph allocates
  // resources on a GPU devices (rule of thumb is ~8kb per
  // node), so we have to be careful not to keep too many of
  // them alive for too long, or we have a higher risk of
  // OOM errors.
  static int64_t AliveExecs();
  static int64_t NotifyExecCreated();
  static int64_t NotifyExecDestroyed();
  using NoOpKernel = TypedKernel<>;

  GraphNodeHandles graph_tail_nodes() {
    GraphNodeHandles nodes;
    nodes.reserve(graph_tail_nodes_.size());
    for (auto& node : graph_tail_nodes_) {
      nodes.push_back(node);
    }
    return nodes;
  }

  absl::Status EmptyOp(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies);

  absl::Status Launch(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                      const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args);

  template <typename... Params, typename... Args>
  absl::Status Launch(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                      const ThreadDim& threads, const BlockDim& blocks,
                      const TypedKernel<Params...>& kernel, Args... args);

  absl::Status AddNestedCommandBuffer(Index cmd_idx,
                                      CmdIdxSetOrNodeHandles dependencies,
                                      const CommandBuffer& nested);

  absl::Status MemcpyDeviceToDevice(Index cmd_idx,
                                    CmdIdxSetOrNodeHandles dependencies,
                                    DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src, uint64_t size);

  absl::Status Memset(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                      DeviceMemoryBase* dst, BitPattern bit_pattern,
                      size_t num_elements);

  absl::Status If(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                  DeviceMemory<bool> predicate, Builder then_builder);

  absl::Status IfElse(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                      DeviceMemory<bool> predicate, Builder then_builder,
                      Builder else_builder);

  absl::Status Case(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                    DeviceMemory<int32_t> index, std::vector<Builder> branches);

  absl::Status For(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                   int32_t num_iteration, DeviceMemory<int32_t> loop_counter,
                   Builder body_builder);

  absl::Status While(Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
                     DeviceMemory<bool> pred, Builder cond_builder,
                     Builder body_builder);

 private:
  // A callback to launch a kernel that updates conditional
  // handles state.
  using SetConditionFn = std::function<absl::Status(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      absl::Span<const GraphConditionalHandle>)>;

  // An extension of `Builder` for building conditional
  // command buffers tied to conditional handles.
  using ConditionBuilder =
      std::function<absl::Status(GpuCommandBuffer*, GraphConditionalHandle)>;

  // Wraps a regular command buffer builder into condition
  // builder.
  static ConditionBuilder ToConditionBuilder(Builder builder);

 public:
  enum class ConditionType { kIf, kWhile };

 private:
  // Prepares a nested command buffer for an update of the
  // graph. It's a prerequisite to a call to `Update` on a
  // nested command buffer. The return value needs to be
  // kept alive until the update is finished. An update
  // finishes by a call to `Finalize`.
  virtual std::unique_ptr<ScopedUpdateMode> ActivateUpdateMode(
      GpuCommandBuffer* nested_cmd_buffer) = 0;

  // For each conditional node in the Gpu graph we keep a
  // record of conditional command buffers attached to a
  // node, so we can apply updates to them.
  struct ConditionalCommandBuffers {
    std::vector<GraphConditionalHandle> conditionals;
    std::vector<std::unique_ptr<GpuCommandBuffer>> command_buffers;
  };

  absl::StatusOr<std::vector<GraphConditionalHandle>> CreateConditionalHandles(
      size_t num_handles);

  absl::StatusOr<std::vector<std::unique_ptr<GpuCommandBuffer>>>
  CreateConditionalCommandBuffers(
      GraphNodeHandles dependencies, ConditionType type,
      absl::Span<const GraphConditionalHandle> conditionals,
      absl::Span<const ConditionBuilder> builders);

  absl::Status UpdateConditionalCommandBuffers(
      absl::Span<const GraphConditionalHandle> handles,
      absl::Span<const std::unique_ptr<GpuCommandBuffer>> command_buffers,
      absl::Span<const ConditionBuilder> builders);

  absl::StatusOr<std::unique_ptr<GpuCommandBuffer>>
  CreateConditionalCommandBuffer(GraphNodeHandles dependencies,
                                 ConditionType type,
                                 GraphConditionalHandle conditional);

  // Adds a new conditional command (If, IfElse, Case,
  // While, For) to the command buffer.
  absl::Status AddConditionalCommandNode(
      ConditionType type, Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      SetConditionFn set_condition,
      absl::Span<const ConditionBuilder> builders);

  // Launches a kernels that updates the state of the given
  // graph conditional based on the predicate. If the
  // predicate is true, `if_conditional` is set to 1,
  // otherwise to 0.
  virtual absl::Status LaunchSetIfConditionKernel(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      GraphConditionalHandle if_conditional, DeviceMemory<bool> predicate) = 0;

  // Launches a kernels that updates the state of the given
  // graph conditionals based on the predicate. If the
  // predicate is true, `if_conditional` is set to 1 and
  // `else_conditional` to 0. If the predicate is false,
  // `if_conditional` is set to 0 and `else_conditional`
  // to 1.
  virtual absl::Status LaunchSetIfElseConditionKernel(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      GraphConditionalHandle if_conditional,
      GraphConditionalHandle else_conditional,
      DeviceMemory<bool> predicate) = 0;

  // Launches a kernel that updates the state of the given
  // graph conditionals based on the index and batch_offset.
  // conditional[x] is set to 1 if index
  // == x + batch_offset and 0 otherwise. `conditionals` may
  // contain up to 8 conditionals
  virtual absl::Status LaunchSetCaseConditionKernel(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      GraphConditionalHandles conditionals, DeviceMemory<int32_t> index,
      int32_t batch_offset, bool enable_conditional_default) = 0;

  // Launches a kernel that updates the state of the given
  // graph conditional based on the loop counter and the
  // total number of iterations. If the loop counter is less
  // than the number of iterations, `conditional` is set to
  // 1, otherwise to 0. The loop counter is also incremented
  // by 1.
  virtual absl::Status LaunchSetForConditionKernel(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      GraphConditionalHandle conditional, DeviceMemory<int32_t> loop_counter,
      int32_t iterations) = 0;

  virtual absl::Status LaunchSetForConditionKernel(
      GraphConditionalHandle conditional, DeviceMemory<int32_t> loop_counter,
      int32_t iterations) = 0;

  // Launches a kernel that updates the state of the given
  // graph conditional based on the predicate. If the
  // predicate is true, `conditional` is set to 1, otherwise
  // to 0.
  virtual absl::Status LaunchSetWhileConditionKernel(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      GraphConditionalHandle conditional, DeviceMemory<bool> predicate) = 0;

  virtual absl::Status LaunchSetWhileConditionKernel(
      GraphConditionalHandle conditional, DeviceMemory<bool> predicate) = 0;

  // Launches CUDA kernels with packed arguments.
  absl::Status LaunchWithPackedArgs(
      Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& packed_args);

 protected:
  using GraphNodeHandleSet = absl::flat_hash_set<GraphNodeHandle>;
  static std::string GraphNodeHandleSetToString(
      const GraphNodeHandleSet& handle_set) {
    std::ostringstream oss;
    oss << "GraphNodeHandleSet: {";
    bool first = true;
    for (const auto& index : handle_set) {
      if (!first) {
        oss << ", ";
      }
      oss << index;
      first = false;
    }
    oss << "}";
    return oss.str();
  }

  // Returns OK status if command buffer is not finalized
  // and it is still possible to add new commands to it,
  // otherwise returns internal error.
  absl::Status CheckNotFinalized();

  // Returns OK status if the command buffer can be updated.
  virtual absl::Status CheckCanBeUpdated() = 0;

  absl::StatusOr<GraphNodeHandleSet> GetCmdTailNodes(Index cmd_idx) {
    auto it = cmd_tail_nodes_.find(cmd_idx);
    if (it == cmd_tail_nodes_.end()) {
      return absl::InternalError(absl::StrCat("Command index ", cmd_idx,
                                              " not found in the tail "
                                              "nodes map."));
    }
    return it->second;
  }

 private:
  // Returns OK status if the number of command buffers is
  // equal to the expected one, otherwise returns internal
  // error.
  absl::Status CheckNumCommandBuffers(
      const ConditionalCommandBuffers& cmd_buffers, size_t num_cmd_buffers);

  Mode mode_;
  State state_ = State::kCreate;

  StreamExecutor* parent_;  // not owned, must outlive *this

 private:
  // ExecutionScope holds the state of an underlying CUDA
  // graph (nodes an barriers added to a graph) for a single
  // execution scope. Tracks indices into data structures
  // during command buffer updates.
  struct UpdateState {
    // Index points to the graph node inside `nodes` that
    // will be updated next.
    int64_t node_idx = 0;

    // Index points to the barrier node inside `barriers`
    // that will be updated on a next call to
    // `Barrier(...)`.
    int64_t barrier_idx = 0;

    // Index points to the conditional command buffers that
    // will be updated next when we'll be updating next
    // conditional command (If, Case, While).
    int64_t conditional_idx = 0;
  };

  // Gpu graph nodes corresponding to recorded commands
  // (launch, memcpy, etc.).
  std::vector<GpuGraphNodeInfo> nodes_;

  // Gpu graph barriers that define recorded commands
  // execution order.
  std::vector<GpuGraphBarrierInfo> barriers_;

  // Command buffers for conditional nodes in the Gpu graph.
  // Underlying Gpu graphs owned by the `graph_` instance.
  std::vector<ConditionalCommandBuffers> conditional_command_buffers_;

  // Tracks execution scope update state.
  UpdateState update_state_;

  // GpuCommandBuffer lowers CommandBufferCmdSequence to a GPU graph, a
  // CommandBufferCmd may lower into one or more graph nodes (command nodes),
  // the HEAD nodes for a CommandBufferCmd is the nodes that have no
  // dependencies within the command nodes, and the TAIL nodes are the nodes
  // that no other nodes within the command nodes depend on them.

  // This map tracks each command's TAIL node sets.
  absl::flat_hash_map<int64_t, GraphNodeHandleSet> cmd_tail_nodes_;

  // Graph tail nodes is a set of node handles in the whole graph that does not
  // have other nodes dependent on them.

  // When lowering a new CommandBufferCmd without specifying the dependency
  // commands, will assume it depends on all the tail nodes in current graph.
  GraphNodeHandleSet graph_tail_nodes_;

  // When lowering new command, update the graph tail node set and current
  // command tail nodes.
  absl::Status UpdateGraphAndCmdTailNodes(Index cmd_idx,
                                          GraphNodeHandles new_nodes,
                                          CmdIdxSetOrNodeHandles dependencies);

  // Track the number of command buffer updates for
  // debugging.
  int64_t num_updates_ = 0;

  // Creates a nested command buffer, associated with the
  // same executor. The given graph will not be owned by the
  // created command buffer.
 protected:
  struct ConditionalNodeResult {
    GraphNodeHandle node_handle;
    std::unique_ptr<GpuCommandBuffer> command_buffer;
  };

 private:
  // Adds a new conditional node to the graph and creates a
  // corresponding nested command buffer.
  virtual absl::StatusOr<ConditionalNodeResult> CreateConditionalNode(
      CmdIdxSetOrNodeHandles dependencies, GraphConditionalHandle conditional,
      ConditionType type) = 0;

  // Adds a new memset node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateMemsetNode(
      CmdIdxSetOrNodeHandles dependencies, DeviceMemoryBase destination,
      BitPattern bit_pattern, size_t num_elements) = 0;

  // Updates an existing memset node. Note that
  // `node_handle` needs to be refer to a node created by
  // `CreateMemsetNode`.
  virtual absl::Status UpdateMemsetNode(GraphNodeHandle node_handle,
                                        DeviceMemoryBase destination,
                                        BitPattern bit_pattern,
                                        size_t num_elements) = 0;

  // Adds a new memcpy node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateMemcpyD2DNode(
      CmdIdxSetOrNodeHandles dependencies, DeviceMemoryBase destination,
      DeviceMemoryBase source, uint64_t size) = 0;

  virtual absl::Status UpdateMemcpyD2DNode(GraphNodeHandle node_handle,
                                           DeviceMemoryBase destination,
                                           DeviceMemoryBase source,
                                           uint64_t size) = 0;

  // Adds a new nested command buffer node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateChildNode(
      CmdIdxSetOrNodeHandles dependencies, const CommandBuffer& nested) = 0;

  // Associate another command buffer with this child node.
  // Will return an error if the given node has not been
  // created as a child node.
  virtual absl::Status UpdateChildNode(GraphNodeHandle node_handle,
                                       const CommandBuffer& nested) = 0;

  // Adds a new kernel launch node to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateKernelNode(
      CmdIdxSetOrNodeHandles dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) = 0;

  // Updates the kernel launch node with the given
  // parameters. Will return an error if the given node has
  // not been created as a kernel launch node.
  virtual absl::Status UpdateKernelNode(
      GraphNodeHandle node_handle, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) = 0;

  // Creates a new no-op node acting as a barrier and adds
  // it to the graph.
  virtual absl::StatusOr<GraphNodeHandle> CreateEmptyNode(
      CmdIdxSetOrNodeHandles dependencies) = 0;

  // Enables or disables the execution of the given node in
  // the graph.
  virtual absl::Status SetNodeExecutionEnabled(GraphNodeHandle node_handle,
                                               bool enabled) = 0;

  // Launches an instantiated graph. Only supported on
  // primary command buffers.
  virtual absl::Status LaunchGraph(Stream* stream) = 0;

  // Returns the number of nodes in the graph associated
  // with this command buffer.
  virtual absl::StatusOr<size_t> GetNodeCount() const = 0;

  // This gets called at the beginning of `Finalize` and
  // allows subclasses to perform any necessary preparation
  // before the graph is finalized.
  virtual absl::Status PrepareFinalization() = 0;

  // Create a new conditional handle in the underlying
  // graph.
  virtual absl::StatusOr<GraphConditionalHandle> CreateConditionalHandle() = 0;

  // Writes the underlying graph to a file in graphviz DOT
  // format.
  virtual absl::Status WriteGraphToDotFile(absl::string_view path) = 0;

  // Instantiates the executable graph from the underlying
  // graph.
  virtual absl::Status InstantiateGraph() = 0;
};

template <typename... Params, typename... Args>
inline absl::Status GpuCommandBuffer::Launch(
    Index cmd_idx, CmdIdxSetOrNodeHandles dependencies,
    const ThreadDim& threads, const BlockDim& blocks,
    const TypedKernel<Params...>& kernel, Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(
      Launch(cmd_idx, dependencies, threads, blocks, *kernel, *kernel_args));
  return absl::OkStatus();
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
