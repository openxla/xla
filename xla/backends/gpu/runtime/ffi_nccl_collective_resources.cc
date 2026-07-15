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

#include "xla/backends/gpu/runtime/ffi_nccl_collective_resources.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_kernel_api.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_nccl_collective_resources.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

static_assert(sizeof(uintptr_t) <= sizeof(uint64_t));

struct OwnerToken {};

struct Region {
  BufferAllocation::Index allocation;
  uint64_t offset;
  uint64_t size;
  uint64_t required_alignment;
  XLA_FFI_NcclCollectiveMemoryKind memory_kind;
};

struct BarrierResources {
  std::unique_ptr<se::MemoryAllocation> signal_buffer;
  std::unique_ptr<se::MemoryAllocation> signal_value;
  tsl::TiedRef<SymmetricMemory> symmetric_memory_ref;
  std::shared_ptr<SymmetricMemory> symmetric_memory;
};

struct ValidatedRequest {
  GpuCliqueKey clique_key;
  RankId rank;
  int32_t clique_size;
  se::StreamExecutor* executor;
  std::vector<std::vector<GlobalDeviceId>> device_groups;
  std::vector<Region> regions;
  std::vector<BufferAllocation::Index> allocations;
  bool barrier_before_launch;
};

enum class ResourcePhase { kPending, kCommitted, kInitialized };

absl::Status CheckStructSize(absl::string_view name, size_t expected,
                             size_t actual) {
  if (actual < expected) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s has size %d; expected at least %d", name, actual, expected));
  }
  return absl::OkStatus();
}

absl::Status ValidateByteRange(uint64_t offset, uint64_t size,
                               uint64_t containing_size,
                               absl::string_view description) {
  if (offset > containing_size || size > containing_size - offset) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s byte range [%d, %d) exceeds containing size %d",
                        description, offset, offset + size, containing_size));
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> AddAddressOffset(void* base, uint64_t offset,
                                          absl::string_view description) {
  if (base == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s has a null base address", description));
  }
  uint64_t address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
  if (offset > std::numeric_limits<uint64_t>::max() - address) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Address overflow while resolving %s", description));
  }
  return address + offset;
}

absl::Status ValidateMultimemCliqueCoverage(SymmetricMemory& symmetric_memory,
                                            RankId rank, int32_t clique_size,
                                            uint64_t offset, uint64_t size,
                                            size_t region_index) {
  for (int32_t peer = 0; peer < clique_size; ++peer) {
    if (peer == rank.value()) continue;

    absl::StatusOr<se::DeviceAddressBase> peer_memory =
        symmetric_memory.peer_addr(RankId(peer));
    if (!peer_memory.ok()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Collective multimem region %d requires the complete clique to be "
          "one load/store-accessible team; rank %d is unavailable: %s",
          region_index, peer, peer_memory.status().message()));
    }
    if (peer_memory->is_null()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Collective multimem region %d requires the complete clique to be "
          "one load/store-accessible team; rank %d has no peer address",
          region_index, peer));
    }
    RETURN_IF_ERROR(ValidateByteRange(
        offset, size, peer_memory->size(),
        absl::StrFormat("Collective multimem region %d rank %d", region_index,
                        peer)));
  }
  return absl::OkStatus();
}

absl::StatusOr<CollectiveOpGroupMode> DecodeGroupMode(
    XLA_FFI_NcclCollectiveGroupMode mode) {
  switch (mode) {
    case XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
    case XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_PARTITION:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION;
    case XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA_AND_PARTITION:
      return CollectiveOpGroupMode::
          COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
    case XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_FLATTENED_ID:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported collective group mode %d", mode));
}

absl::StatusOr<int64_t> LogicalGroupDomainSize(
    CollectiveOpGroupMode mode, const DeviceAssignment& device_assignment) {
  int64_t replica_count = device_assignment.replica_count();
  int64_t partition_count = device_assignment.computation_count();
  if (replica_count <= 0 || partition_count <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective device assignment must have positive dimensions; got "
        "%d replicas and %d partitions",
        replica_count, partition_count));
  }

  switch (mode) {
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA:
    case CollectiveOpGroupMode::
        COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION:
      return replica_count;
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION:
      return partition_count;
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID:
      if (replica_count >
          std::numeric_limits<int64_t>::max() / partition_count) {
        return absl::InvalidArgumentError(
            "Flattened collective logical-ID domain overflows int64");
      }
      return replica_count * partition_count;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported collective group mode %d", mode));
  }
}

absl::StatusOr<std::vector<ReplicaGroup>> ValidateAndCopyGroups(
    const XLA_FFI_NcclCollectiveGroup& group,
    const DeviceAssignment& device_assignment,
    CollectiveOpGroupMode group_mode) {
  RETURN_IF_ERROR(CheckStructSize("XLA_FFI_NcclCollectiveGroup",
                                  XLA_FFI_NcclCollectiveGroup_STRUCT_SIZE,
                                  group.struct_size));
  if (group.expected_clique_size <= 0) {
    return absl::InvalidArgumentError(
        "Expected collective clique size must be positive");
  }
  if (group.num_groups == 0 || group.group_offsets == nullptr) {
    return absl::InvalidArgumentError(
        "Collective groups must contain at least one group");
  }
  if (group.num_groups == std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError("Collective group count overflows");
  }
  if (group.num_members == 0 || group.members == nullptr) {
    return absl::InvalidArgumentError(
        "Collective groups must contain at least one member");
  }
  if (group.group_offsets[0] != 0 ||
      group.group_offsets[group.num_groups] != group.num_members) {
    return absl::InvalidArgumentError(
        "Collective group offsets must span the complete members array");
  }

  ASSIGN_OR_RETURN(int64_t domain_size,
                   LogicalGroupDomainSize(group_mode, device_assignment));
  if (static_cast<uint64_t>(domain_size) > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError(
        "Collective logical-ID domain is too large for this host");
  }

  std::vector<bool> present(static_cast<size_t>(domain_size), false);
  std::vector<ReplicaGroup> replica_groups;
  replica_groups.reserve(group.num_groups);
  size_t expected_group_size = 0;

  for (size_t group_index = 0; group_index < group.num_groups; ++group_index) {
    size_t begin = group.group_offsets[group_index];
    size_t end = group.group_offsets[group_index + 1];
    if (begin > end || end > group.num_members || begin == end) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Collective group %d has invalid member offsets [%d, %d)",
          group_index, begin, end));
    }
    size_t group_size = end - begin;
    if (group_index == 0) {
      expected_group_size = group_size;
    } else if (group_size != expected_group_size) {
      return absl::InvalidArgumentError(
          "All collective groups must have equal cardinality");
    }

    ReplicaGroup replica_group;
    for (size_t member_index = begin; member_index < end; ++member_index) {
      int64_t member = group.members[member_index];
      if (member < 0 || member >= domain_size) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Collective group member %d is outside logical-ID domain [0, %d)",
            member, domain_size));
      }
      if (present[member]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Collective group member %d appears more than once", member));
      }
      present[member] = true;
      replica_group.add_replica_ids(member);
    }
    replica_groups.push_back(std::move(replica_group));
  }

  if (group.num_members != static_cast<size_t>(domain_size)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective groups cover %d logical IDs but group mode requires the "
        "complete domain of %d IDs",
        group.num_members, domain_size));
  }
  return replica_groups;
}

}  // namespace

struct FfiNcclCollectiveResources::State {
  State() : owner(std::make_shared<const OwnerToken>()) {}

  std::shared_ptr<const OwnerToken> owner;
  bool requested = false;

  XLA_FFI_ExecutionStage stage = XLA_FFI_ExecutionStage_INSTANTIATE;
  se::Stream* stream = nullptr;
  const BufferAllocations* buffer_allocations = nullptr;
  const CollectiveParams* collective_params = nullptr;
  CollectiveCliqueRequests* collective_clique_requests = nullptr;
  CollectiveMemoryRequests* collective_memory_requests = nullptr;
  CollectiveCliques* collective_cliques = nullptr;
  const CollectiveMemory* collective_memory = nullptr;
};

class FfiNcclCollectiveResources::Resource final
    : public ffi::NcclCollectiveResourceHandle {
 public:
  Resource(std::shared_ptr<const OwnerToken> owner, ValidatedRequest request)
      : owner_(std::move(owner)),
        clique_key_(std::move(request.clique_key)),
        rank_(request.rank),
        clique_size_(request.clique_size),
        executor_(request.executor),
        device_groups_(std::move(request.device_groups)),
        regions_(std::move(request.regions)),
        allocations_(std::move(request.allocations)),
        barrier_before_launch_(request.barrier_before_launch) {}

  std::shared_ptr<const OwnerToken> owner_;
  GpuCliqueKey clique_key_;
  RankId rank_;
  int32_t clique_size_;
  se::StreamExecutor* executor_;
  std::vector<std::vector<GlobalDeviceId>> device_groups_;
  std::vector<Region> regions_;
  std::vector<BufferAllocation::Index> allocations_;
  bool barrier_before_launch_;
  ResourcePhase phase_ = ResourcePhase::kPending;
  bool barrier_enqueued_ = false;
  std::unique_ptr<BarrierResources> barrier_;
};

namespace {

absl::StatusOr<Region> ValidateAndMapRegion(
    const XLA_FFI_NcclCollectiveRegion& region, size_t region_index,
    const BufferAllocations& buffer_allocations) {
  RETURN_IF_ERROR(CheckStructSize("XLA_FFI_NcclCollectiveRegion",
                                  XLA_FFI_NcclCollectiveRegion_STRUCT_SIZE,
                                  region.struct_size));
  if (region.data == nullptr || region.containing_buffer_size == 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective region %d must identify a non-empty FFI buffer",
        region_index));
  }
  if (region.byte_size == 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective region %d must have a positive byte size", region_index));
  }
  if (region.required_alignment == 0 ||
      (region.required_alignment & (region.required_alignment - 1)) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective region %d alignment must be a positive power of two",
        region_index));
  }
  switch (region.memory_kind) {
    case XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC:
    case XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM:
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Collective region %d has unsupported memory kind %d",
                          region_index, region.memory_kind));
  }
  RETURN_IF_ERROR(ValidateByteRange(
      region.byte_offset, region.byte_size, region.containing_buffer_size,
      absl::StrFormat("Collective region %d", region_index)));
  ASSIGN_OR_RETURN(
      uint64_t local_address,
      AddAddressOffset(region.data, region.byte_offset,
                       absl::StrFormat("collective region %d", region_index)));
  if (local_address % region.required_alignment != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective region %d local address 0x%x does not meet alignment %d",
        region_index, local_address, region.required_alignment));
  }

  se::DeviceAddressBase buffer(region.data, region.containing_buffer_size);
  std::optional<BufferAllocation::Index> allocation_index =
      buffer_allocations.FindAllocationIndex(buffer);
  if (!allocation_index.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective region %d does not belong to an XLA buffer allocation",
        region_index));
  }
  se::DeviceAddressBase allocation =
      buffer_allocations.GetDeviceAddress(*allocation_index);
  uintptr_t allocation_address =
      reinterpret_cast<uintptr_t>(allocation.opaque());
  uintptr_t buffer_address = reinterpret_cast<uintptr_t>(region.data);
  if (buffer_address < allocation_address) {
    return absl::InternalError(
        "XLA buffer allocation lookup returned an invalid allocation");
  }
  uint64_t buffer_offset = buffer_address - allocation_address;
  RETURN_IF_ERROR(ValidateByteRange(
      buffer_offset, region.containing_buffer_size, allocation.size(),
      absl::StrFormat("Collective region %d FFI buffer", region_index)));
  if (region.byte_offset >
      std::numeric_limits<uint64_t>::max() - buffer_offset) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective region %d allocation offset overflows", region_index));
  }
  return Region{*allocation_index, buffer_offset + region.byte_offset,
                region.byte_size, region.required_alignment,
                region.memory_kind};
}

absl::StatusOr<std::unique_ptr<BarrierResources>> InitializeBarrier(
    se::Stream* stream, const GpuCliqueKey& clique_key, RankId rank,
    int32_t clique_size, CollectiveCliques& collective_cliques) {
  if (clique_size > se::gpu::MultiGpuBarrierKernel::kMaxPeers) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective clique size %d exceeds the prefix barrier limit of %d",
        clique_size, se::gpu::MultiGpuBarrierKernel::kMaxPeers));
  }

  ASSIGN_OR_RETURN(GpuCommunicator * communicator,
                   collective_cliques.GetComm(clique_key, rank));
  ASSIGN_OR_RETURN(GpuCommunicatorTopology topology,
                   communicator->GetTopology());
  if (topology.lsa_size != clique_size || topology.lsa_team_count != 1) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Collective entry synchronization requires the complete clique to be "
        "one LSA; got LSA size %d and %d teams for clique size %d",
        topology.lsa_size, topology.lsa_team_count, clique_size));
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocator> allocator,
      stream->parent()->CreateMemoryAllocator(se::MemorySpace::kCollective));
  auto resources = std::make_unique<BarrierResources>();
  ASSIGN_OR_RETURN(resources->signal_buffer,
                   allocator->Allocate(GetMultiGpuBarrierSignalBufferSize()));
  ASSIGN_OR_RETURN(resources->signal_value,
                   allocator->Allocate(GetMultiGpuBarrierSignalValueSize()));

  if (resources->signal_buffer == nullptr ||
      resources->signal_buffer->address().is_null() ||
      resources->signal_buffer->address().size() <
          GetMultiGpuBarrierSignalBufferSize() ||
      resources->signal_value == nullptr ||
      resources->signal_value->address().is_null() ||
      resources->signal_value->address().size() <
          GetMultiGpuBarrierSignalValueSize()) {
    return absl::ResourceExhaustedError(
        "Failed to allocate collective prefix barrier control memory");
  }

  se::DeviceAddressBase signal_buffer =
      resources->signal_buffer->address().GetByteSlice(
          0, GetMultiGpuBarrierSignalBufferSize());
  se::DeviceAddressBase signal_value =
      resources->signal_value->address().GetByteSlice(
          0, GetMultiGpuBarrierSignalValueSize());
  RETURN_IF_ERROR(stream->MemZero(&signal_buffer, signal_buffer.size()));
  RETURN_IF_ERROR(stream->MemZero(&signal_value, signal_value.size()));
  RETURN_IF_ERROR(stream->BlockHostUntilDone());

  ASSIGN_OR_RETURN(std::unique_ptr<SymmetricMemory> symmetric_memory,
                   communicator->CreateSymmetricMemory(signal_buffer));
  ASSIGN_OR_RETURN(
      resources->symmetric_memory_ref,
      collective_cliques.Tie(clique_key, std::move(symmetric_memory)));
  resources->symmetric_memory = resources->symmetric_memory_ref.Lock();
  if (resources->symmetric_memory == nullptr) {
    return absl::InternalError(
        "Collective prefix barrier registration expired during Initialize");
  }

  se::DeviceAddressBase backing = resources->symmetric_memory->addr();
  if (backing.opaque() != signal_buffer.opaque() ||
      backing.size() < signal_buffer.size()) {
    return absl::FailedPreconditionError(
        "Collective prefix barrier registration does not match its backing "
        "allocation");
  }

  return resources;
}

}  // namespace

FfiNcclCollectiveResources::FfiNcclCollectiveResources()
    : state_(std::make_unique<State>()) {}

FfiNcclCollectiveResources::~FfiNcclCollectiveResources() = default;

absl::Status FfiNcclCollectiveResources::BeginInvocation(
    XLA_FFI_ExecutionStage stage, se::Stream* stream,
    const BufferAllocations* buffer_allocations,
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* collective_clique_requests,
    CollectiveMemoryRequests* collective_memory_requests,
    CollectiveCliques* collective_cliques,
    const CollectiveMemory* collective_memory) {
  switch (stage) {
    case XLA_FFI_ExecutionStage_PREPARE:
    case XLA_FFI_ExecutionStage_INITIALIZE:
    case XLA_FFI_ExecutionStage_EXECUTE:
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported collective resource execution stage %d", stage));
  }
  if (stream != nullptr && collective_params != nullptr &&
      collective_params->executor != nullptr &&
      stream->parent() != collective_params->executor) {
    return absl::FailedPreconditionError(
        "Collective resource stream and runtime executor do not match");
  }

  state_->stage = stage;
  state_->stream = stream;
  state_->buffer_allocations = buffer_allocations;
  state_->collective_params = collective_params;
  state_->collective_clique_requests = collective_clique_requests;
  state_->collective_memory_requests = collective_memory_requests;
  state_->collective_cliques = collective_cliques;
  state_->collective_memory = collective_memory;
  return absl::OkStatus();
}

absl::Status FfiNcclCollectiveResources::Request(
    XLA_FFI_NcclCollectiveResources_Request_Args* args) {
  if (args == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Request args must not be null");
  }
  RETURN_IF_ERROR(
      CheckStructSize("XLA_FFI_NcclCollectiveResources_Request_Args",
                      XLA_FFI_NcclCollectiveResources_Request_Args_STRUCT_SIZE,
                      args->struct_size));
  if (args->ctx == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Request context must not be null");
  }
  if (state_->stage != XLA_FFI_ExecutionStage_PREPARE) {
    return absl::FailedPreconditionError(
        "NCCL collective resources can only be requested during FFI Prepare");
  }
  if (state_->requested) {
    return absl::FailedPreconditionError(
        "Only one NCCL collective resource request is allowed per execution");
  }
  if (args->group == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Request requires a group");
  }
  if (args->region_count != 0 && args->regions == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Request regions must not be null");
  }
  if (args->barrier_before_launch > 1) {
    return absl::InvalidArgumentError(
        "NCCL collective resource barrier flag must be zero or one");
  }
  if (state_->buffer_allocations == nullptr ||
      state_->collective_params == nullptr) {
    return absl::FailedPreconditionError(
        "NCCL collective resource Request requires NCCL collective Prepare "
        "contexts");
  }

  const CollectiveParams& params = *state_->collective_params;
  if (params.executor == nullptr || params.collectives == nullptr ||
      params.device_assn == nullptr) {
    return absl::FailedPreconditionError(
        "NCCL collective resource Request requires an executor, collectives "
        "API, and device assignment");
  }
  if (params.executor->GetPlatform()->Name() != "CUDA") {
    return absl::UnimplementedError(
        "NCCL collective resources require the CUDA platform");
  }

  ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                   DecodeGroupMode(args->group->group_mode));
  ASSIGN_OR_RETURN(
      std::vector<ReplicaGroup> replica_groups,
      ValidateAndCopyGroups(*args->group, *params.device_assn, group_mode));
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(params, replica_groups, group_mode,
                      CommunicationId(args->group->communication_id)));
  if (clique_key.num_devices() !=
      static_cast<size_t>(args->group->expected_clique_size)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ABI clique size %d does not match runtime clique size %d",
        args->group->expected_clique_size, clique_key.num_devices()));
  }
  if (clique_key.num_devices() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return absl::InvalidArgumentError(
        "NCCL collective clique size exceeds the public ABI limit");
  }
  bool barrier_before_launch = args->barrier_before_launch != 0;
  if (barrier_before_launch &&
      clique_key.num_devices() >
          static_cast<size_t>(se::gpu::MultiGpuBarrierKernel::kMaxPeers)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective clique size %d exceeds the prefix barrier limit of %d",
        clique_key.num_devices(), se::gpu::MultiGpuBarrierKernel::kMaxPeers));
  }
  std::optional<RankId> rank = clique_key.rank(params.global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Global device %d is not a member of collective clique %s",
        params.global_device_id.value(), clique_key.ToString()));
  }

  ASSIGN_OR_RETURN(std::vector<std::vector<GlobalDeviceId>> device_groups,
                   GetParticipatingDevicesGroups(*params.device_assn,
                                                 replica_groups, group_mode));
  for (std::vector<GlobalDeviceId>& device_group : device_groups) {
    std::sort(device_group.begin(), device_group.end());
  }
  std::sort(device_groups.begin(), device_groups.end());

  std::vector<Region> regions;
  regions.reserve(args->region_count);
  std::set<BufferAllocation::Index> unique_allocations;
  for (size_t i = 0; i < args->region_count; ++i) {
    ASSIGN_OR_RETURN(
        Region region,
        ValidateAndMapRegion(args->regions[i], i, *state_->buffer_allocations));
    unique_allocations.insert(region.allocation);
    regions.push_back(region);
  }

  if (clique_key.num_devices() != 0 &&
      args->region_count >
          std::numeric_limits<size_t>::max() / clique_key.num_devices()) {
    return absl::InvalidArgumentError(
        "NCCL collective address table size overflows");
  }

  std::vector<BufferAllocation::Index> allocations(unique_allocations.begin(),
                                                   unique_allocations.end());
  ValidatedRequest request{
      std::move(clique_key),
      *rank,
      static_cast<int32_t>(args->group->expected_clique_size),
      params.executor,
      std::move(device_groups),
      std::move(regions),
      std::move(allocations),
      barrier_before_launch};
  auto resource = std::make_unique<Resource>(state_->owner, std::move(request));

  args->rank = resource->rank_.value();
  args->clique_size = resource->clique_size_;
  args->resource = reinterpret_cast<XLA_FFI_NcclCollectiveResource*>(
      static_cast<ffi::NcclCollectiveResourceHandle*>(resource.release()));
  state_->requested = true;
  return absl::OkStatus();
}

absl::Status FfiNcclCollectiveResources::Commit(
    XLA_FFI_NcclCollectiveResources_Commit_Args* args) {
  if (args == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Commit args must not be null");
  }
  RETURN_IF_ERROR(
      CheckStructSize("XLA_FFI_NcclCollectiveResources_Commit_Args",
                      XLA_FFI_NcclCollectiveResources_Commit_Args_STRUCT_SIZE,
                      args->struct_size));
  if (args->ctx == nullptr || args->resource == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Commit requires context and resource");
  }
  if (state_->stage != XLA_FFI_ExecutionStage_PREPARE) {
    return absl::FailedPreconditionError(
        "NCCL collective resources can only be committed during FFI Prepare");
  }

  auto* handle =
      reinterpret_cast<ffi::NcclCollectiveResourceHandle*>(args->resource);
  auto* resource = static_cast<Resource*>(handle);
  if (resource->owner_.get() != state_->owner.get()) {
    return absl::InvalidArgumentError(
        "NCCL collective resource belongs to a different execution");
  }
  if (resource->phase_ != ResourcePhase::kPending) {
    return absl::FailedPreconditionError(
        "NCCL collective resource can only be committed once");
  }
  if (state_->collective_clique_requests == nullptr ||
      (!resource->allocations_.empty() &&
       state_->collective_memory_requests == nullptr)) {
    return absl::FailedPreconditionError(
        "NCCL collective resource Commit requires collective resource "
        "collectors");
  }

  RETURN_IF_ERROR(state_->collective_clique_requests->RequestClique(
      resource->clique_key_, resource->device_groups_));
  for (BufferAllocation::Index allocation : resource->allocations_) {
    RETURN_IF_ERROR(
        state_->collective_memory_requests->RequestSymmetricAllocation(
            resource->clique_key_, allocation));
  }
  resource->phase_ = ResourcePhase::kCommitted;
  return absl::OkStatus();
}

absl::Status FfiNcclCollectiveResources::Initialize(
    XLA_FFI_NcclCollectiveResources_Initialize_Args* args) {
  if (args == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Initialize args must not be null");
  }
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_NcclCollectiveResources_Initialize_Args",
      XLA_FFI_NcclCollectiveResources_Initialize_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->ctx == nullptr || args->resource == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Initialize requires context and resource");
  }
  if (state_->stage != XLA_FFI_ExecutionStage_INITIALIZE) {
    return absl::FailedPreconditionError(
        "NCCL collective resources can only be initialized during FFI "
        "Initialize");
  }

  auto* handle =
      reinterpret_cast<ffi::NcclCollectiveResourceHandle*>(args->resource);
  auto* resource = static_cast<Resource*>(handle);
  if (resource->owner_.get() != state_->owner.get()) {
    return absl::InvalidArgumentError(
        "NCCL collective resource belongs to a different execution");
  }
  if (resource->phase_ != ResourcePhase::kCommitted) {
    return absl::FailedPreconditionError(
        "NCCL collective resource must be committed exactly once before "
        "Initialize");
  }
  if (state_->stream == nullptr || state_->buffer_allocations == nullptr ||
      state_->collective_params == nullptr ||
      (resource->barrier_before_launch_ &&
       state_->collective_cliques == nullptr) ||
      (!resource->regions_.empty() && state_->collective_memory == nullptr)) {
    return absl::FailedPreconditionError(
        "NCCL collective resource Initialize requires acquired collective "
        "contexts");
  }
  if (state_->stream->parent() != resource->executor_ ||
      state_->collective_params->executor != resource->executor_) {
    return absl::FailedPreconditionError(
        "NCCL collective resource executor changed after Prepare");
  }
  std::optional<RankId> rank =
      resource->clique_key_.rank(state_->collective_params->global_device_id);
  if (!rank.has_value() || *rank != resource->rank_) {
    return absl::FailedPreconditionError(
        "NCCL collective resource rank changed after Prepare");
  }

  std::unique_ptr<BarrierResources> barrier;
  if (resource->barrier_before_launch_) {
    ASSIGN_OR_RETURN(barrier,
                     InitializeBarrier(state_->stream, resource->clique_key_,
                                       resource->rank_, resource->clique_size_,
                                       *state_->collective_cliques));
  }
  resource->barrier_ = std::move(barrier);
  resource->phase_ = ResourcePhase::kInitialized;
  return absl::OkStatus();
}

absl::Status FfiNcclCollectiveResources::Resolve(
    XLA_FFI_NcclCollectiveResources_Resolve_Args* args) {
  if (args == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Resolve args must not be null");
  }
  RETURN_IF_ERROR(
      CheckStructSize("XLA_FFI_NcclCollectiveResources_Resolve_Args",
                      XLA_FFI_NcclCollectiveResources_Resolve_Args_STRUCT_SIZE,
                      args->struct_size));
  if (args->ctx == nullptr || args->resource == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective resource Resolve requires context and resource");
  }
  if (state_->stage != XLA_FFI_ExecutionStage_INITIALIZE) {
    return absl::FailedPreconditionError(
        "NCCL collective addresses can only be resolved during FFI "
        "Initialize");
  }

  auto* handle =
      reinterpret_cast<ffi::NcclCollectiveResourceHandle*>(args->resource);
  auto* resource = static_cast<Resource*>(handle);
  if (resource->owner_.get() != state_->owner.get()) {
    return absl::InvalidArgumentError(
        "NCCL collective resource belongs to a different execution");
  }
  if (resource->phase_ != ResourcePhase::kInitialized) {
    return absl::FailedPreconditionError(
        "NCCL collective resource must be initialized before address "
        "resolution");
  }
  if (state_->buffer_allocations == nullptr ||
      (!resource->regions_.empty() && state_->collective_memory == nullptr)) {
    return absl::FailedPreconditionError(
        "NCCL collective address resolution requires acquired collective "
        "memory");
  }
  size_t expected_count =
      resource->regions_.size() * static_cast<size_t>(resource->clique_size_);
  if (args->address_count != expected_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "NCCL collective address table has %d entries; expected %d",
        args->address_count, expected_count));
  }
  if (expected_count != 0 && args->addresses == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective address table must not be null");
  }

  std::vector<uint64_t> resolved;
  resolved.reserve(expected_count);
  for (size_t region_index = 0; region_index < resource->regions_.size();
       ++region_index) {
    const Region& region = resource->regions_[region_index];
    if (region.allocation < 0 || static_cast<size_t>(region.allocation) >=
                                     state_->buffer_allocations->size()) {
      return absl::InternalError(absl::StrFormat(
          "Collective region %d has an invalid XLA allocation", region_index));
    }
    se::DeviceAddressBase allocation =
        state_->buffer_allocations->GetDeviceAddress(region.allocation);
    RETURN_IF_ERROR(ValidateByteRange(
        region.offset, region.size, allocation.size(),
        absl::StrFormat("Collective region %d XLA allocation", region_index)));

    auto [symmetric_memory, symmetric_offset] =
        state_->collective_memory->FindSymmetricMemory(resource->clique_key_,
                                                       region.allocation);
    if (symmetric_memory == nullptr) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "No symmetric memory was acquired for collective region %d",
          region_index));
    }
    se::DeviceAddressBase local_symmetric = symmetric_memory->addr();
    RETURN_IF_ERROR(ValidateByteRange(
        symmetric_offset, allocation.size(), local_symmetric.size(),
        absl::StrFormat("Collective region %d symmetric allocation",
                        region_index)));
    ASSIGN_OR_RETURN(
        uint64_t local_symmetric_address,
        AddAddressOffset(
            local_symmetric.opaque(), symmetric_offset,
            absl::StrFormat("collective region %d symmetric allocation",
                            region_index)));
    if (local_symmetric_address !=
        reinterpret_cast<uintptr_t>(allocation.opaque())) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Collective region %d symmetric-memory backing does not match its "
          "XLA allocation",
          region_index));
    }
    if (region.offset >
        std::numeric_limits<uint64_t>::max() - symmetric_offset) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Collective region %d symmetric offset overflows", region_index));
    }
    uint64_t offset = symmetric_offset + region.offset;

    if (region.memory_kind == XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM) {
      RETURN_IF_ERROR(ValidateMultimemCliqueCoverage(
          *symmetric_memory, resource->rank_, resource->clique_size_, offset,
          region.size, region_index));
      ASSIGN_OR_RETURN(se::DeviceAddressBase multimem,
                       symmetric_memory->multimem_addr());
      if (multimem.is_null()) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "Multimem address is unavailable for collective region %d",
            region_index));
      }
      RETURN_IF_ERROR(ValidateByteRange(
          offset, region.size, multimem.size(),
          absl::StrFormat("Collective multimem region %d", region_index)));
      ASSIGN_OR_RETURN(
          uint64_t address,
          AddAddressOffset(
              multimem.opaque(), offset,
              absl::StrFormat("collective multimem region %d", region_index)));
      if (address % region.required_alignment != 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Collective multimem region %d address 0x%x does not meet "
            "alignment %d",
            region_index, address, region.required_alignment));
      }
      resolved.insert(resolved.end(), resource->clique_size_, address);
      continue;
    }

    for (int32_t peer = 0; peer < resource->clique_size_; ++peer) {
      uint64_t address;
      if (peer == resource->rank_.value()) {
        ASSIGN_OR_RETURN(
            address, AddAddressOffset(
                         allocation.opaque(), region.offset,
                         absl::StrFormat("collective region %d local address",
                                         region_index)));
      } else {
        ASSIGN_OR_RETURN(se::DeviceAddressBase peer_memory,
                         symmetric_memory->peer_addr(RankId(peer)));
        RETURN_IF_ERROR(
            ValidateByteRange(offset, region.size, peer_memory.size(),
                              absl::StrFormat("Collective region %d rank %d",
                                              region_index, peer)));
        ASSIGN_OR_RETURN(
            address,
            AddAddressOffset(peer_memory.opaque(), offset,
                             absl::StrFormat("collective region %d rank %d",
                                             region_index, peer)));
      }
      if (address % region.required_alignment != 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Collective region %d rank %d address 0x%x does not meet "
            "alignment %d",
            region_index, peer, address, region.required_alignment));
      }
      resolved.push_back(address);
    }
  }

  std::copy(resolved.begin(), resolved.end(), args->addresses);
  return absl::OkStatus();
}

absl::Status FfiNcclCollectiveResources::QueryTopology(
    XLA_FFI_NcclCollectiveResources_QueryTopology_Args* args) {
  if (args == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective topology args must not be null");
  }
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_NcclCollectiveResources_QueryTopology_Args",
      XLA_FFI_NcclCollectiveResources_QueryTopology_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->ctx == nullptr || args->resource == nullptr ||
      args->topology == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective topology requires context, resource, and output");
  }
  RETURN_IF_ERROR(CheckStructSize("XLA_FFI_NcclCollectiveTopology",
                                  XLA_FFI_NcclCollectiveTopology_STRUCT_SIZE,
                                  args->topology->struct_size));
  if (state_->stage != XLA_FFI_ExecutionStage_INITIALIZE) {
    return absl::FailedPreconditionError(
        "NCCL collective topology can only be queried during FFI Initialize");
  }

  auto* handle =
      reinterpret_cast<ffi::NcclCollectiveResourceHandle*>(args->resource);
  auto* resource = static_cast<Resource*>(handle);
  if (resource->owner_.get() != state_->owner.get()) {
    return absl::InvalidArgumentError(
        "NCCL collective resource belongs to a different execution");
  }
  if (resource->phase_ != ResourcePhase::kInitialized) {
    return absl::FailedPreconditionError(
        "NCCL collective resource must be initialized before querying "
        "topology");
  }
  if (state_->collective_cliques == nullptr) {
    return absl::FailedPreconditionError(
        "NCCL collective topology requires acquired collective cliques");
  }

  ASSIGN_OR_RETURN(GpuCommunicator * communicator,
                   state_->collective_cliques->GetComm(resource->clique_key_,
                                                       resource->rank_));
  ASSIGN_OR_RETURN(GpuCommunicatorTopology topology,
                   communicator->GetTopology());
  if (topology.lsa_size <= 0 || topology.lsa_size > resource->clique_size_ ||
      topology.lsa_team_count <= 0 ||
      topology.lsa_size > std::numeric_limits<int32_t>::max() ||
      topology.lsa_team_count > std::numeric_limits<int32_t>::max() ||
      topology.lsa_size * topology.lsa_team_count != resource->clique_size_) {
    return absl::InternalError(
        "Collective backend returned invalid load/store topology");
  }

  args->topology->clique_size = resource->clique_size_;
  args->topology->lsa_size = static_cast<int32_t>(topology.lsa_size);
  args->topology->lsa_team_count =
      static_cast<int32_t>(topology.lsa_team_count);
  args->topology->world_is_lsa = topology.lsa_size == resource->clique_size_ &&
                                         topology.lsa_team_count == 1
                                     ? uint8_t{1}
                                     : uint8_t{0};
  args->topology->multimem_supported =
      topology.multimem_supported ? uint8_t{1} : uint8_t{0};
  return absl::OkStatus();
}

absl::Status FfiNcclCollectiveResources::BeginCollective(
    XLA_FFI_NcclCollectiveResources_BeginCollective_Args* args) {
  if (args == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective BeginCollective args must not be null");
  }
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_NcclCollectiveResources_BeginCollective_Args",
      XLA_FFI_NcclCollectiveResources_BeginCollective_Args_STRUCT_SIZE,
      args->struct_size));
  if (args->ctx == nullptr || args->resource == nullptr) {
    return absl::InvalidArgumentError(
        "NCCL collective BeginCollective requires context and resource");
  }
  if (state_->stage != XLA_FFI_ExecutionStage_EXECUTE) {
    return absl::FailedPreconditionError(
        "NCCL collective BeginCollective is valid only during FFI Execute");
  }

  auto* handle =
      reinterpret_cast<ffi::NcclCollectiveResourceHandle*>(args->resource);
  auto* resource = static_cast<Resource*>(handle);
  if (resource->owner_.get() != state_->owner.get()) {
    return absl::InvalidArgumentError(
        "NCCL collective resource belongs to a different execution");
  }
  if (resource->phase_ != ResourcePhase::kInitialized) {
    return absl::FailedPreconditionError(
        "NCCL collective resource must be initialized before Execute");
  }
  if (!resource->barrier_before_launch_ || resource->barrier_ == nullptr) {
    return absl::FailedPreconditionError(
        "NCCL collective resource did not request entry synchronization");
  }
  if (resource->barrier_enqueued_) {
    return absl::FailedPreconditionError(
        "NCCL collective execution was already begun");
  }
  if (state_->stream == nullptr || state_->collective_params == nullptr ||
      state_->stream->parent() != resource->executor_ ||
      state_->collective_params->executor != resource->executor_) {
    return absl::FailedPreconditionError(
        "NCCL collective resource executor changed after Prepare");
  }
  std::optional<RankId> rank =
      resource->clique_key_.rank(state_->collective_params->global_device_id);
  if (!rank.has_value() || *rank != resource->rank_) {
    return absl::FailedPreconditionError(
        "NCCL collective resource rank changed after Prepare");
  }

  se::DeviceAddressBase signal_value =
      resource->barrier_->signal_value->address().GetByteSlice(
          0, GetMultiGpuBarrierSignalValueSize());
  resource->barrier_enqueued_ = true;
  return LaunchNcclLsaBarrier(
      state_->stream, resource->clique_size_, resource->rank_,
      resource->barrier_->symmetric_memory.get(), signal_value);
}

}  // namespace xla::gpu
