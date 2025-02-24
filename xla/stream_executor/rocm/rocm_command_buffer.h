/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/scoped_gpu_graph_exec.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Implements GpuCommandBuffer for AMD GPUs.
class RocmCommandBuffer : public GpuCommandBuffer {
 public:
  // Creates a new ROCm command buffer and the underlying HIP graph.
  static absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> Create(
      Mode mode, StreamExecutor* parent);

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  absl::StatusOr<GraphNodeHandle> CreateEmptyNode(
      GraphNodeHandles dependencies) override;

  // Adds a kernel launch command that depends on the commands in
  // deps.
  absl::StatusOr<GraphNodeHandle> CreateLaunchNode(
      GraphNodeHandles deps, const ThreadDim& threads, const BlockDim& blocks,
      const Kernel& kernel, const KernelArgs& args) override {
    return absl::UnimplementedError("CreateLaunchNode");
  }

  absl::Status UpdateLaunchNode(GraphNodeHandle node, const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgs& args) override {
    return absl::UnimplementedError("UpdateLaunchNode");
  }

  absl::StatusOr<GraphNodeHandle> CreateChildNode(
      GraphNodeHandles deps, const CommandBuffer& child) override;

  absl::Status UpdateChildNode(GraphNodeHandle node,
                               const CommandBuffer& child) override;

  // Adds a device-to-device memory copy that depends on the commands in
  // deps.
  absl::StatusOr<GraphNodeHandle> CreateMemcpyD2DNode(GraphNodeHandles deps,
                                                      DeviceMemoryBase dst,
                                                      DeviceMemoryBase src,
                                                      uint64_t size) override;

  absl::Status UpdateMemcpyD2DNode(GraphNodeHandle node, DeviceMemoryBase dst,
                                   DeviceMemoryBase src,
                                   uint64_t size) override;

  // Adds a memset command that depends on the commands in deps.
  absl::StatusOr<GraphNodeHandle> CreateMemsetNode(
      GraphNodeHandles deps, DeviceMemoryBase dst, BitPattern bit_pattern,
      size_t num_elements) override;

  absl::Status UpdateMemsetNode(GraphNodeHandle node, DeviceMemoryBase dst,
                                BitPattern bit_pattern,
                                size_t num_elements) override;

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  absl::StatusOr<GraphConditionalHandle> CreateConditionalHandle() override;

  // Adds a new conditional node to the graph and creates a
  // corresponding nested command buffer.
  absl::StatusOr<ConditionalNodeResult> CreateConditionalNode(
      GraphNodeHandles dependencies, GraphConditionalHandle conditional,
      ConditionType type) override {
    return absl::UnimplementedError("CreateConditionalNode");
  }

  absl::StatusOr<GraphNodeHandle> CreateSetIfElseConditionKernelNode(
      GraphNodeHandles dependencies, GraphConditionalHandle then_condition,
      GraphConditionalHandle else_condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("CreateSetIfElseConditionKernelNode");
  }

  absl::Status UpdateSetIfElseConditionKernelNode(
      GraphNodeHandle node, GraphConditionalHandle then_condition,
      GraphConditionalHandle else_condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("UpdateSetIfElseConditionKernelNode");
  }

  absl::StatusOr<GraphNodeHandle> CreateSetIfConditionKernelNode(
      GraphNodeHandles dependencies, GraphConditionalHandle then_condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("CreateSetIfConditionKernelNode");
  }

  absl::Status UpdateSetIfConditionKernelNode(
      GraphNodeHandle node, GraphConditionalHandle then_condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("UpdateSetIfConditionKernelNode");
  }

  absl::StatusOr<GraphNodeHandle> CreateSetForConditionKernelNode(
      GraphNodeHandles dependencies, GraphConditionalHandle condition,
      DeviceMemory<int32_t> loop_counter, int32_t iterations) override {
    return absl::UnimplementedError("CreateSetForConditionKernelNode");
  }

  absl::Status UpdateSetForConditionKernelNode(
      GraphNodeHandle node, GraphConditionalHandle condition,
      DeviceMemory<int32_t> loop_counter, int32_t iterations) override {
    return absl::UnimplementedError("UpdateSetForConditionKernelNode");
  }

  absl::StatusOr<GraphNodeHandle> CreateSetWhileConditionKernelNode(
      GraphNodeHandles dependencies, GraphConditionalHandle condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("CreateSetWhileConditionKernelNode");
  }

  absl::Status UpdateSetWhileConditionKernelNode(
      GraphNodeHandle node, GraphConditionalHandle condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("UpdateSetWhileConditionKernelNode");
  }

  absl::StatusOr<GraphNodeHandle> CreateSetCaseConditionKernelNode(
      GraphNodeHandles dependencies, GraphConditionalHandle handle0,
      GraphConditionalHandle handle1, GraphConditionalHandle handle2,
      GraphConditionalHandle handle3, GraphConditionalHandle handle4,
      GraphConditionalHandle handle5, GraphConditionalHandle handle6,
      GraphConditionalHandle handle7, DeviceMemory<int32_t> index,
      int32_t batch_offset, int32_t num_branches,
      bool enable_conditional_default) override {
    return absl::UnimplementedError("CreateSetCaseConditionKernelNode");
  }

  absl::Status UpdateSetCaseConditionKernelNode(
      GraphNodeHandle node, GraphConditionalHandle handle0,
      GraphConditionalHandle handle1, GraphConditionalHandle handle2,
      GraphConditionalHandle handle3, GraphConditionalHandle handle4,
      GraphConditionalHandle handle5, GraphConditionalHandle handle6,
      GraphConditionalHandle handle7, DeviceMemory<int32_t> index,
      int32_t batch_offset, int32_t num_branches,
      bool enable_conditional_default) override {
    return absl::UnimplementedError("UpdateSetCaseConditionKernelNode");
  }

  absl::Status Submit(Stream* stream) override {
    return absl::UnimplementedError("Submit");
  }

  absl::Status Finalize() override {
    return absl::UnimplementedError("Finalize");
  }

  ~RocmCommandBuffer() override {
    return absl::UnimplementedError("~RocmCommandBuffer");
  }

 private:
  RocmCommandBuffer(Mode mode, StreamExecutor* parent, hipGraph_t graph,
                    bool is_owned_graph, CommandBuffer* parent = nullptr)
      : GpuCommandBuffer(mode, parent),
        graph_(graph),
        is_owned_graph_(is_owned_graph),
        parent_(parent) {}

  // Converts a list of platform independent GraphNodeHandles into a list of
  // Rocm specific hipGraphNode_t.
  absl::StatusOr<std::vector<hipGraphNode_t>> ToHipGraphHandles(
      GraphNodeHandles dependencies);

  CUgraphExec GetPrimaryExec() const {
    if (mode() == Mode::kPrimary) {
      return exec_;
    }
    return static_cast<CudaCommandBuffer*>(parent())->GetPrimaryExec();
  }

  absl::Status PrepareFinalization() override;

  absl::StatusOr<GraphNodeHandle> CreateKernelNode(
      GraphNodeHandles dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args);

  absl::Status UpdateKernelNode(GraphNodeHandle node_handle,
                                const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgsPackedArrayBase& args);

  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status()> function) override;

  absl::Status SetNodeExecutionEnabled(GraphNodeHandle node_handle,
                                       bool enabled);

  absl::Status LaunchGraph(Stream* stream);

  absl::StatusOr<size_t> GetNodeCount() const override;

  absl::Status PrepareFinalization() override;

  absl::Status WriteGraphToDotFile(absl::string_view path) override;

  absl::Status InstantiateGraph() override;

  using ScopedRocmGraphExec = ScopedGraphExec<hipGraphExec_t>;
  absl::Status CheckCanBeUpdated() override;

  StreamExecutor* stream_executor_;

  static_assert(std::is_pointer_v<hipGraph_t>, "hipGraph_t must be a pointer");
  static_assert(std::is_pointer_v<hipGraphExec_t>,
                "hipGraphExec_t must be a pointer");

  hipGraph_t graph_ = nullptr;  // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;  // ownership of `graph_`

  hipGraphExec_t exec_ = nullptr;    // owned if `is_owned_graph_exec_`
  bool is_owned_graph_exec_ = true;  // ownership of `is_owned_graph_exec_`
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
