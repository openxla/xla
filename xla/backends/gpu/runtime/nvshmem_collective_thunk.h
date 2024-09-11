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

#ifndef XLA_SERVICE_GPU_RUNTIME_NVSHMEM_COLLECTIVE_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NVSHMEM_COLLECTIVE_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

struct NvshmemCollectiveConfig {
  int64_t operand_count;
  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64_t op_id;
  CollectiveOpGroupMode group_mode;

  template <typename OpT>
  void SetCollectiveOpKindAndID(OpT op);
  void SetCollectiveOpKindAndID(const HloCollectivePermuteInstruction* instr);
  void SetCollectiveOpKindAndID(const HloSendRecvInstruction* instr);
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;
};

template <typename OpT>
void NvshmemCollectiveConfig::SetCollectiveOpKindAndID(OpT op) {
  if (op.getChannelId()) {
    collective_op_kind = RendezvousKey::kCrossModule;
    op_id = static_cast<int64_t>(op.getChannelId()->getHandle());
  } else {
    collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    op_id = static_cast<int64_t>(unique_id.getInt());
  }
}

NvshmemCollectiveConfig GetNvshmemCollectiveConfig(
    const HloInstruction* hlo, std::optional<bool> use_global_device_ids);

template <typename OpT>
NvshmemCollectiveConfig GetNvshmemCollectiveConfigForMlir(
    OpT op, std::optional<bool> use_global_device_ids) {
  NvshmemCollectiveConfig config;
  config.operand_count = op.getInputs().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    const Shape shape = GetShape(op.getInputs()[i]);
    config.operand_element_type.push_back(shape.element_type());
  }
  config.replica_groups = ConvertReplicaGroups(op.getReplicaGroups()).value();
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetCollectiveOpGroupMode(op.getChannelId().has_value(),
                                               use_global_device_ids)
                          .value();
  return config;
}

//===----------------------------------------------------------------------===//
// NvshmemCollectiveThunk
//===----------------------------------------------------------------------===//

// Forward declare.
class NvshmemCollectiveDoneThunk;

// Thunk base class for NVSHMEM collective operations.
class NvshmemCollectiveThunk : public Thunk {
 public:
  NvshmemCollectiveThunk(Kind kind, ThunkInfo thunk_info, bool is_sync);

  struct Buffer {
    int64_t element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
    int64_t source_memory_space;
    int64_t destination_memory_space;
    mlir::Value source_value;
    mlir::Value destination_value;
  };

  // Logging support.
  static std::string GetDeviceString(
      const Thunk::CollectiveExecuteParams& params);

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;

  absl::Status Initialize(const InitializeParams& params) override;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }
  void set_async_events(
      std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events) {
    async_events_ = async_events;
  }

  CollectiveStreamId nvshmem_stream_id() const {
    return xla::gpu::GetCollectiveStreamId(IsAsync(), GetAsyncStreamKind());
  }

  ExecutionStreamId nvshmem_execution_stream_id() const {
    return ExecutionStreamId(execution_stream_id().value() +
                             nvshmem_stream_id().value());
  }

 protected:
  virtual absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                            se::Stream& stream,
                                            CommunicatorHandle comm) = 0;
  virtual const NvshmemCollectiveConfig& config() const = 0;
  virtual AsyncStreamKind GetAsyncStreamKind() const {
    return AsyncStreamKind::kCollective;
  }

  // A collective thunk is normally an independent operation in a sense that
  // different instances of the same collective thunk communicate each other.
  // The only exception are SendThunk and RecvThunk. Assume two devices are
  // executing a program contains the following instructions, the Recv from
  // device 1 will release the Send from device 0. Adding first call
  // rendezvous on the SendThunk would cause a runtime deadlock.
  //  Send(src_target={0,1})
  //  Recv(src_target={0,1})
  virtual bool NeedFirstCallRendzevous() const { return true; }

 private:
  bool IsAsync() const { return async_events_ != nullptr; }
  std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events_;

  // After a first call to this particular instance of a NCCL collective thunk
  // we do a round of rendezvous to make sure that all participants successfully
  // allocated on-device state required for executing collective operation. This
  // is required to avoid deadlocks when one device goes too far ahead and
  // causes a deadlock in CUDA driver (root cause is mysterious).
  //
  // TODO(ezhulenev): Try to move this flag to NCCL clique as we need to make
  // sure that all NCCL resources are allocated just once.
  RendezvousSingleFlag first_call_rendezvous_flag_;
};

//===----------------------------------------------------------------------===//
// NvshmemCollectiveDoneThunk
//===----------------------------------------------------------------------===//

class NvshmemCollectiveDoneThunk : public Thunk {
 public:
  NvshmemCollectiveDoneThunk(
      Thunk::Kind kind, ThunkInfo thunk_info,
      std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events,
      AsyncStreamKind async_stream_kind);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  // return the execution stream id wheer previous async operator was launched
  // to.
  ExecutionStreamId nvshmem_execution_stream_id() const {
    return ExecutionStreamId(
        execution_stream_id().value() +
        xla::gpu::GetCollectiveStreamId(true, async_stream_kind_).value());
  }

 private:
  std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events_;
  AsyncStreamKind async_stream_kind_ = AsyncStreamKind::kCollective;
};

//===----------------------------------------------------------------------===//

absl::Status IsValidNvshmemOperand(mlir::Value operand,
                                   Thunk::Kind reduction_op);

absl::Status IsValidNvshmemOperand(Shape shape, Thunk::Kind reduction_op);

template <typename NvshmemThunkType, typename OpT>
absl::Status AddNvshmemOpDescription(absl::Status status, OpT op,
                                     int64_t replica_count,
                                     int64_t partition_count) {
  if (status.ok()) {
    return status;
  }
  CollectiveOpGroupMode group_mode = NvshmemThunkType::GetGroupMode(op);

  int64_t operand_count = 0;
  std::string str;

  if constexpr (std::is_base_of_v<HloInstruction, std::remove_pointer_t<OpT>>) {
    operand_count = op->operand_count();
    str = op->ToString();
  } else {
    operand_count = op->getNumOperands() / 2;
    str = llvm_ir::DumpToString(op.getOperation());
  }

  return absl::Status(
      status.code(),
      absl::StrFormat(
          "%s\n"
          "%s with replica_count: %d, partition_count: %d, group_mode: %s, "
          "operand_count: %d\n%s",
          status.message(), NvshmemThunkType::GetHloOpName(), replica_count,
          partition_count, CollectiveOpGroupModeToString(group_mode),
          operand_count, str));
}

//===----------------------------------------------------------------------===//

absl::StatusOr<GpuCliqueKey> GetGpuNvshmemCliqueKey(
    GpuCollectives* collectives, const Thunk::CollectiveExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, CollectiveStreamId stream_id,
    AsyncStreamKind stream_kind);

// Returns a nccl comm and a flag indicating if it's a local communicator.
absl::StatusOr<CommunicatorHandle> GetNvshmemComm(
    GpuCollectives* collectives, const Thunk::CollectiveExecuteParams& params,
    const Thunk::CollectiveCliques& collective_cliques,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, CollectiveStreamId stream_id,
    AsyncStreamKind stream_kind);

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const Thunk::ExecuteParams& params,
    const std::vector<NvshmemCollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const BufferAllocations* buffer_allocations,
    const std::vector<NvshmemCollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NVSHMEM_COLLECTIVE_THUNK_H_
