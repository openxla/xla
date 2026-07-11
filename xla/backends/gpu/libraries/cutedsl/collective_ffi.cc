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

#include "xla/backends/gpu/libraries/cutedsl/collective_ffi.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/libraries/cutedsl/collective_config.h"
#include "xla/backends/gpu/libraries/cutedsl/module.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"
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
#include "xla/ffi/ffi.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::cutedsl {
namespace {

constexpr absl::string_view kCollectiveCallTarget =
    "__xla_gpu_cutedsl_collective_v3";
constexpr absl::string_view kFunctionPrefix = "cutlass_call";
constexpr size_t kInlineBufferCount = 8;
constexpr size_t kInlinePeerAddressCount = 32;

static_assert(sizeof(uintptr_t) <= sizeof(uint64_t));

// A POD descriptor matching cutlass.jax.types.JaxArray.
struct CuteXlaFfiBuffer {
  void* buffer;
  const int64_t* shape;
};

class CollectiveCallStateV3 {
 public:
  explicit CollectiveCallStateV3(CollectiveCallConfigV3 config)
      : config_(std::move(config)) {}

  const CollectiveCallConfigV3& config() const { return config_; }

 private:
  CollectiveCallConfigV3 config_;
};

struct CollectiveCallPreparedStateV3 {
  GpuCliqueKey clique_key;
  RankId rank;
  int32_t clique_size;
  se::StreamExecutor* executor;
  std::shared_ptr<LoadedModule> module;
  absl::flat_hash_map<int64_t, CuteDSLRT_Function_t*> functions;
};

// Fields are declared so that reverse-order destruction releases the symmetric
// registration before either backing allocation.
struct BarrierResourcesV3 {
  std::unique_ptr<se::MemoryAllocation> signal_buffer;
  std::unique_ptr<se::MemoryAllocation> signal_value;
  tsl::TiedRef<SymmetricMemory> symmetric_memory_ref;
  std::shared_ptr<SymmetricMemory> symmetric_memory;
  std::vector<se::DeviceAddressBase> peer_addresses;
};

struct CollectiveCallInitializedStateV3 {
  se::StreamExecutor* executor;
  std::vector<uint64_t> peer_addresses;
  std::shared_ptr<BarrierResourcesV3> barrier;
};

struct RetainedExecutionResourcesV3 {
  std::shared_ptr<LoadedModule> module;
  std::shared_ptr<BarrierResourcesV3> barrier;
};

absl::StatusOr<uint64_t> AddAddressOffset(void* base, uint64_t offset,
                                          absl::string_view description) {
  if (base == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s has a null base address", description));
  }

  uint64_t base_value =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
  if (offset > std::numeric_limits<uint64_t>::max() - base_value) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Address overflow while resolving %s", description));
  }
  return base_value + offset;
}

absl::Status ValidateByteRange(uint64_t offset, uint64_t size,
                               uint64_t containing_size,
                               absl::string_view description) {
  if (offset > containing_size || size > containing_size - offset) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s byte range [%d, %d) exceeds containing buffer size %d", description,
        offset, offset + size, containing_size));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> LogicalGroupDomainSize(
    const CollectiveCallConfigV3& config,
    const DeviceAssignment& device_assignment) {
  int64_t replica_count = device_assignment.replica_count();
  int64_t partition_count = device_assignment.computation_count();
  if (replica_count <= 0 || partition_count <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Collective device assignment must have positive dimensions; got "
        "%d replicas and %d partitions",
        replica_count, partition_count));
  }

  switch (config.group_mode) {
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
          absl::StrFormat("Unsupported collective group mode %d",
                          static_cast<int64_t>(config.group_mode)));
  }
}

absl::Status ValidateReplicaGroupDomain(
    const CollectiveCallConfigV3& config,
    const DeviceAssignment& device_assignment) {
  ASSIGN_OR_RETURN(int64_t domain_size,
                   LogicalGroupDomainSize(config, device_assignment));
  if (static_cast<uint64_t>(domain_size) > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError(
        "Collective logical-ID domain is too large for this host");
  }

  std::vector<bool> present(static_cast<size_t>(domain_size), false);
  size_t member_count = 0;
  for (size_t group_index = 0; group_index < config.replica_groups.size();
       ++group_index) {
    const ReplicaGroup& group = config.replica_groups[group_index];
    for (int64_t member : group.replica_ids()) {
      if (member < 0 || member >= domain_size) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member %d in group %d is outside logical-ID "
            "domain [0, %d)",
            member, group_index, domain_size));
      }
      if (present[member]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member %d appears more than once", member));
      }
      present[member] = true;
      ++member_count;
    }
  }

  if (member_count != static_cast<size_t>(domain_size)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Replica groups cover %d logical IDs but group mode requires the "
        "complete domain of %d IDs",
        member_count, domain_size));
  }
  return absl::OkStatus();
}

absl::StatusOr<se::DeviceAddressBase> GetPeerRegionBuffer(
    const PeerRegionV3& region, size_t region_index,
    ffi::RemainingArgs arguments, ffi::RemainingRets results) {
  if (region.buffer_index < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Peer region %d has negative buffer index %d",
                        region_index, region.buffer_index));
  }

  size_t buffer_index = static_cast<size_t>(region.buffer_index);
  switch (region.endpoint) {
    case PeerRegionEndpointV3::kArgument: {
      if (buffer_index >= arguments.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Peer region %d references argument %d but the call has %d "
            "arguments",
            region_index, buffer_index, arguments.size()));
      }
      ASSIGN_OR_RETURN(ffi::AnyBuffer buffer,
                       arguments.get<ffi::AnyBuffer>(buffer_index));
      return buffer.device_memory();
    }
    case PeerRegionEndpointV3::kResult: {
      if (buffer_index >= results.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Peer region %d references result %d but the call has %d results",
            region_index, buffer_index, results.size()));
      }
      ASSIGN_OR_RETURN(ffi::Result<ffi::AnyBuffer> result,
                       results.get<ffi::AnyBuffer>(buffer_index));
      return result->device_memory();
    }
  }

  return absl::InvalidArgumentError(absl::StrFormat(
      "Peer region %d has an unsupported endpoint", region_index));
}

absl::StatusOr<std::vector<se::DeviceAddressBase>> GetPeerRegionBuffers(
    const CollectiveCallConfigV3& config, ffi::RemainingArgs arguments,
    ffi::RemainingRets results, bool require_addresses) {
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(config.peer_regions.size());

  for (size_t region_index = 0; region_index < config.peer_regions.size();
       ++region_index) {
    const PeerRegionV3& region = config.peer_regions[region_index];
    ASSIGN_OR_RETURN(
        se::DeviceAddressBase buffer,
        GetPeerRegionBuffer(region, region_index, arguments, results));

    uint64_t byte_offset = static_cast<uint64_t>(region.byte_offset);
    uint64_t byte_size = static_cast<uint64_t>(region.byte_size);
    RETURN_IF_ERROR(
        ValidateByteRange(byte_offset, byte_size, buffer.size(),
                          absl::StrFormat("Peer region %d", region_index)));

    if (require_addresses) {
      ASSIGN_OR_RETURN(
          uint64_t address,
          AddAddressOffset(buffer.opaque(), byte_offset,
                           absl::StrFormat("peer region %d", region_index)));
      uint64_t alignment = static_cast<uint64_t>(region.required_alignment);
      if (address % alignment != 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Peer region %d local address 0x%x does not meet required "
            "alignment %d",
            region_index, address, alignment));
      }
    }

    buffers.push_back(buffer);
  }
  return buffers;
}

bool HasBarrier(const CollectiveCallConfigV3& config) {
  return absl::c_any_of(config.steps, [](const CollectiveStepV3& step) {
    return step.kind == CollectiveStepKindV3::kBarrier;
  });
}

std::string FunctionPrefix(int64_t ordinal) {
  return ordinal == 0 ? std::string(kFunctionPrefix)
                      : absl::StrCat(kFunctionPrefix, "_", ordinal);
}

}  // namespace

namespace internal {

absl::StatusOr<std::vector<uint64_t>> ResolvePeerAddressesV3(
    const GpuCliqueKey& clique_key, RankId rank,
    absl::Span<const PeerRegionV3> peer_regions,
    absl::Span<const se::DeviceAddressBase> buffers,
    const CollectiveMemory& collective_memory) {
  if (peer_regions.size() != buffers.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Peer-region count %d does not match buffer count %d",
                        peer_regions.size(), buffers.size()));
  }
  if (rank.value() < 0 ||
      static_cast<size_t>(rank.value()) >= clique_key.num_devices()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Clique rank %d is outside clique size %d",
                        rank.value(), clique_key.num_devices()));
  }
  if (clique_key.num_devices() > std::numeric_limits<size_t>::max() /
                                     std::max<size_t>(peer_regions.size(), 1)) {
    return absl::InvalidArgumentError("Peer-address table size overflows");
  }

  std::vector<uint64_t> peer_addresses;
  peer_addresses.reserve(peer_regions.size() * clique_key.num_devices());

  for (size_t region_index = 0; region_index < peer_regions.size();
       ++region_index) {
    const PeerRegionV3& region = peer_regions[region_index];
    const se::DeviceAddressBase& buffer = buffers[region_index];
    if (region.byte_offset < 0 || region.byte_size <= 0 ||
        region.required_alignment <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Peer region %d has invalid offset, size, or alignment",
          region_index));
    }

    uint64_t region_offset = static_cast<uint64_t>(region.byte_offset);
    uint64_t region_size = static_cast<uint64_t>(region.byte_size);
    uint64_t alignment = static_cast<uint64_t>(region.required_alignment);
    RETURN_IF_ERROR(
        ValidateByteRange(region_offset, region_size, buffer.size(),
                          absl::StrFormat("Peer region %d", region_index)));

    auto [symmetric_memory, buffer_offset] =
        collective_memory.FindSymmetricMemory(clique_key, buffer);
    if (symmetric_memory == nullptr) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "No symmetric memory was acquired for peer region %d", region_index));
    }

    se::DeviceAddressBase local_symmetric_address = symmetric_memory->addr();
    RETURN_IF_ERROR(ValidateByteRange(
        buffer_offset, buffer.size(), local_symmetric_address.size(),
        absl::StrFormat("Peer region %d containing XLA buffer", region_index)));
    ASSIGN_OR_RETURN(
        uint64_t local_symmetric_buffer_address,
        AddAddressOffset(local_symmetric_address.opaque(), buffer_offset,
                         absl::StrFormat("peer region %d XLA symmetric buffer",
                                         region_index)));
    if (local_symmetric_buffer_address !=
        reinterpret_cast<uintptr_t>(buffer.opaque())) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Peer region %d symmetric-memory backing address 0x%x does not "
          "match the FFI buffer 0x%x",
          region_index, local_symmetric_buffer_address,
          reinterpret_cast<uintptr_t>(buffer.opaque())));
    }
    if (region_offset > std::numeric_limits<uint64_t>::max() - buffer_offset) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Offset overflow while resolving peer region %d", region_index));
    }
    uint64_t offset_in_symmetric_memory = buffer_offset + region_offset;

    for (size_t peer = 0; peer < clique_key.num_devices(); ++peer) {
      se::DeviceAddressBase peer_base;
      if (peer == static_cast<size_t>(rank.value())) {
        // NCCL can return a distinct virtual alias for the local rank from
        // ncclGetPeerDevicePointer. Preserve the actual FFI buffer address in
        // the local slot; it is the address XLA uses for ordinary dataflow.
        peer_base = local_symmetric_address;
      } else {
        ASSIGN_OR_RETURN(peer_base, symmetric_memory->peer_addr(RankId(peer)));
      }
      RETURN_IF_ERROR(ValidateByteRange(
          offset_in_symmetric_memory, region_size, peer_base.size(),
          absl::StrFormat("Peer region %d rank %d", region_index, peer)));
      ASSIGN_OR_RETURN(
          uint64_t peer_address,
          AddAddressOffset(
              peer_base.opaque(), offset_in_symmetric_memory,
              absl::StrFormat("peer region %d rank %d", region_index, peer)));
      if (peer_address % alignment != 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Peer region %d rank %d address 0x%x does not meet required "
            "alignment %d",
            region_index, peer, peer_address, alignment));
      }
      peer_addresses.push_back(peer_address);
    }
  }

  return peer_addresses;
}

}  // namespace internal

namespace {

absl::StatusOr<std::unique_ptr<CollectiveCallStateV3>> Instantiate(
    ffi::RemainingArgs arguments, ffi::RemainingRets results,
    ffi::Dictionary attributes) {
  ASSIGN_OR_RETURN(CollectiveCallConfigV3 config,
                   ParseCollectiveCallConfigV3(attributes));

  // Instantiate receives prototype buffers with null data pointers but exact
  // types and shapes. Validate all configuration-to-buffer mappings here and
  // repeat address-dependent validation during Prepare.
  RETURN_IF_ERROR(GetPeerRegionBuffers(config, arguments, results,
                                       /*require_addresses=*/false)
                      .status());
  return std::make_unique<CollectiveCallStateV3>(std::move(config));
}

absl::StatusOr<std::unique_ptr<CollectiveCallPreparedStateV3>> Prepare(
    CollectiveCallStateV3* state, ffi::RemainingArgs arguments,
    ffi::RemainingRets results, const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests,
    CollectiveMemoryRequests* memory_requests) {
  if (state == nullptr || collective_params == nullptr ||
      clique_requests == nullptr || memory_requests == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 Prepare requires state and collective "
        "resource contexts");
  }
  if (collective_params->executor == nullptr ||
      collective_params->collectives == nullptr ||
      collective_params->device_assn == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 requires an executor, collectives API, and "
        "runtime device assignment");
  }

  const CollectiveCallConfigV3& config = state->config();
  // A partial group can cause only some ranks to request the clique and leave
  // the others deadlocked. Validate the complete logical domain before making
  // any resource request.
  RETURN_IF_ERROR(
      ValidateReplicaGroupDomain(config, *collective_params->device_assn));

  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, config.replica_groups, config.group_mode,
          CommunicationId(static_cast<uint64_t>(config.communication_id))));
  if (clique_key.num_devices() == 0 ||
      clique_key.num_devices() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CuTeDSL collective clique size %d does not fit the v3 int32 call "
        "frame",
        clique_key.num_devices()));
  }
  if (HasBarrier(config) &&
      clique_key.num_devices() >
          static_cast<size_t>(se::gpu::MultiGpuBarrierKernel::kMaxPeers)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CuTeDSL collective clique size %d exceeds the generic barrier limit "
        "of %d peers",
        clique_key.num_devices(), se::gpu::MultiGpuBarrierKernel::kMaxPeers));
  }
  std::optional<RankId> rank =
      clique_key.rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Global device %d is not a member of CuTeDSL collective clique %s",
        collective_params->global_device_id.value(), clique_key.ToString()));
  }

  ASSIGN_OR_RETURN(
      std::vector<std::vector<GlobalDeviceId>> device_groups,
      GetParticipatingDevicesGroups(*collective_params->device_assn,
                                    config.replica_groups, config.group_mode));
  absl::c_for_each(device_groups, [](std::vector<GlobalDeviceId>& group) {
    absl::c_sort(group);
  });
  absl::c_sort(device_groups);

  ASSIGN_OR_RETURN(std::vector<se::DeviceAddressBase> peer_region_buffers,
                   GetPeerRegionBuffers(config, arguments, results,
                                        /*require_addresses=*/true));

  absl::btree_set<int64_t> ordinals;
  for (const CollectiveStepV3& step : config.steps) {
    if (step.kind == CollectiveStepKindV3::kLaunch) {
      ordinals.insert(step.operand);
    }
  }
  const CollectiveModuleImageV3& image = config.module;
  absl::string_view digest(reinterpret_cast<const char*>(image.sha256.data()),
                           image.sha256.size());
  ASSIGN_OR_RETURN(
      std::shared_ptr<LoadedModule> module,
      GetOrLoadModule(image.bytes, digest, collective_params->executor));

  absl::flat_hash_map<int64_t, CuteDSLRT_Function_t*> functions;
  functions.reserve(ordinals.size());
  // Every rank loads the same module and validates the same function ordinals
  // in a deterministic order before any clique request.
  for (int64_t ordinal : ordinals) {
    absl::StatusOr<CuteDSLRT_Function_t*> function =
        module->GetFunction(FunctionPrefix(ordinal));
    if (!function.ok()) {
      return absl::Status(
          function.status().code(),
          absl::StrFormat("Function ordinal %d is unavailable: %s", ordinal,
                          function.status().message()));
    }
    functions.emplace(ordinal, *function);
  }

  // Resource requests happen only after all configuration, topology, buffer,
  // module, and function checks that do not themselves require acquisition.
  RETURN_IF_ERROR(clique_requests->RequestClique(clique_key, device_groups));
  for (const se::DeviceAddressBase& buffer : peer_region_buffers) {
    RETURN_IF_ERROR(
        memory_requests->RequestSymmetricAddress(clique_key, buffer));
  }

  int32_t clique_size = static_cast<int32_t>(clique_key.num_devices());
  return std::make_unique<CollectiveCallPreparedStateV3>(
      CollectiveCallPreparedStateV3{std::move(clique_key), *rank, clique_size,
                                    collective_params->executor,
                                    std::move(module), std::move(functions)});
}

absl::StatusOr<std::shared_ptr<BarrierResourcesV3>> InitializeBarrier(
    se::Stream* stream, const CollectiveCallPreparedStateV3& prepared,
    CollectiveCliques& collective_cliques) {
  if (prepared.clique_size > se::gpu::MultiGpuBarrierKernel::kMaxPeers) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CuTeDSL collective clique size %d exceeds the generic barrier limit "
        "of %d peers",
        prepared.clique_size, se::gpu::MultiGpuBarrierKernel::kMaxPeers));
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocator> collective_allocator,
      stream->parent()->CreateMemoryAllocator(se::MemorySpace::kCollective));
  auto resources = std::make_shared<BarrierResourcesV3>();
  ASSIGN_OR_RETURN(
      resources->signal_buffer,
      collective_allocator->Allocate(GetMultiGpuBarrierSignalBufferSize()));
  ASSIGN_OR_RETURN(
      resources->signal_value,
      collective_allocator->Allocate(GetMultiGpuBarrierSignalValueSize()));

  if (resources->signal_buffer == nullptr ||
      resources->signal_buffer->address().is_null() ||
      resources->signal_buffer->address().size() <
          GetMultiGpuBarrierSignalBufferSize() ||
      resources->signal_value == nullptr ||
      resources->signal_value->address().is_null() ||
      resources->signal_value->address().size() <
          GetMultiGpuBarrierSignalValueSize()) {
    return absl::ResourceExhaustedError(
        "Failed to allocate CuTeDSL collective barrier control memory");
  }

  se::DeviceAddressBase signal_buffer =
      resources->signal_buffer->address().GetByteSlice(
          0, GetMultiGpuBarrierSignalBufferSize());
  se::DeviceAddressBase signal_value =
      resources->signal_value->address().GetByteSlice(
          0, GetMultiGpuBarrierSignalValueSize());
  RETURN_IF_ERROR(stream->MemZero(&signal_buffer, signal_buffer.size()));
  RETURN_IF_ERROR(stream->MemZero(&signal_value, signal_value.size()));
  // Initialization must complete before any rank registers and launches the
  // monotonic barrier state.
  RETURN_IF_ERROR(stream->BlockHostUntilDone());

  ASSIGN_OR_RETURN(
      GpuCommunicator * communicator,
      collective_cliques.GetComm(prepared.clique_key, prepared.rank));
  ASSIGN_OR_RETURN(std::unique_ptr<SymmetricMemory> symmetric_memory,
                   communicator->CreateSymmetricMemory(signal_buffer));
  ASSIGN_OR_RETURN(
      resources->symmetric_memory_ref,
      collective_cliques.Tie(prepared.clique_key, std::move(symmetric_memory)));
  resources->symmetric_memory = resources->symmetric_memory_ref.Lock();
  if (resources->symmetric_memory == nullptr) {
    return absl::InternalError(
        "CuTeDSL collective barrier registration expired during Initialize");
  }

  // The tied reference keeps the registration associated with the cached XLA
  // clique, while the locked shared pointer keeps it alive through deferred
  // stream completion cleanup. NCCL's implementation also retains its shared
  // communicator state. Both handles are released before backing allocations.
  se::DeviceAddressBase barrier_backing = resources->symmetric_memory->addr();
  if (barrier_backing.opaque() != signal_buffer.opaque() ||
      barrier_backing.size() < signal_buffer.size()) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective barrier symmetric-memory backing address does not "
        "match its XLA allocation");
  }

  resources->peer_addresses.reserve(prepared.clique_size);
  for (int32_t peer = 0; peer < prepared.clique_size; ++peer) {
    se::DeviceAddressBase peer_address;
    if (peer == prepared.rank.value()) {
      peer_address = signal_buffer;
    } else {
      ASSIGN_OR_RETURN(peer_address,
                       resources->symmetric_memory->peer_addr(RankId(peer)));
    }
    if (peer_address.is_null() || peer_address.size() < signal_buffer.size()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "CuTeDSL collective barrier peer address for rank %d is unavailable "
          "or too small",
          peer));
    }
    resources->peer_addresses.push_back(
        peer_address.GetByteSlice(0, signal_buffer.size()));
  }
  return resources;
}

absl::StatusOr<std::unique_ptr<CollectiveCallInitializedStateV3>> Initialize(
    se::Stream* stream, CollectiveCallStateV3* state,
    CollectiveCallPreparedStateV3* prepared, ffi::RemainingArgs arguments,
    ffi::RemainingRets results, const CollectiveParams* collective_params,
    CollectiveCliques* collective_cliques,
    const CollectiveMemory* collective_memory) {
  if (stream == nullptr || state == nullptr || prepared == nullptr ||
      collective_params == nullptr || collective_cliques == nullptr ||
      collective_memory == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 Initialize requires stream, state, and "
        "acquired collective contexts");
  }
  if (stream->parent() != prepared->executor ||
      collective_params->executor != prepared->executor) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 executor changed between Prepare and "
        "Initialize");
  }
  std::optional<RankId> runtime_rank =
      prepared->clique_key.rank(collective_params->global_device_id);
  if (!runtime_rank.has_value() || *runtime_rank != prepared->rank) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 clique rank changed between Prepare and "
        "Initialize");
  }

  const CollectiveCallConfigV3& config = state->config();
  ASSIGN_OR_RETURN(std::vector<se::DeviceAddressBase> peer_region_buffers,
                   GetPeerRegionBuffers(config, arguments, results,
                                        /*require_addresses=*/true));
  ASSIGN_OR_RETURN(
      std::vector<uint64_t> peer_addresses,
      internal::ResolvePeerAddressesV3(prepared->clique_key, prepared->rank,
                                       config.peer_regions, peer_region_buffers,
                                       *collective_memory));

  std::shared_ptr<BarrierResourcesV3> barrier;
  if (HasBarrier(config)) {
    ASSIGN_OR_RETURN(barrier,
                     InitializeBarrier(stream, *prepared, *collective_cliques));
  }

  return std::make_unique<CollectiveCallInitializedStateV3>(
      CollectiveCallInitializedStateV3{
          prepared->executor, std::move(peer_addresses), std::move(barrier)});
}

absl::Status ExecuteConfiguredSteps(
    se::Stream* stream, const CollectiveCallConfigV3& config,
    const CollectiveCallPreparedStateV3& prepared,
    CollectiveCallInitializedStateV3& initialized, ffi::RemainingArgs arguments,
    ffi::RemainingRets results) {
  absl::InlinedVector<CuteXlaFfiBuffer, kInlineBufferCount> buffers;
  buffers.reserve(arguments.size() + results.size());

  for (size_t i = 0; i < arguments.size(); ++i) {
    ASSIGN_OR_RETURN(ffi::AnyBuffer argument, arguments.get<ffi::AnyBuffer>(i));
    ffi::AnyBuffer::Dimensions dimensions = argument.dimensions();
    buffers.push_back({argument.untyped_data(),
                       dimensions.empty() ? nullptr : dimensions.data()});
  }
  for (size_t i = 0; i < results.size(); ++i) {
    ASSIGN_OR_RETURN(ffi::Result<ffi::AnyBuffer> result,
                     results.get<ffi::AnyBuffer>(i));
    ffi::AnyBuffer::Dimensions dimensions = result->dimensions();
    buffers.push_back({result->untyped_data(),
                       dimensions.empty() ? nullptr : dimensions.data()});
  }

  // Pointer-valued parameters use the MLIR packed C interface's extra level
  // of indirection. Scalar parameters below point directly at scalar storage.
  absl::InlinedVector<void*, kInlineBufferCount + 1> pointer_values;
  pointer_values.reserve(buffers.size() + 1);
  void* platform_stream = stream->platform_specific_handle().stream;
  if (platform_stream == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 requires a CUDA platform stream");
  }
  pointer_values.push_back(platform_stream);
  for (CuteXlaFfiBuffer& buffer : buffers) {
    pointer_values.push_back(&buffer);
  }

  absl::InlinedVector<void*, kInlineBufferCount + kInlinePeerAddressCount + 4>
      packed_arguments;
  packed_arguments.reserve(pointer_values.size() +
                           initialized.peer_addresses.size() + 3);
  for (void*& pointer_value : pointer_values) {
    packed_arguments.push_back(&pointer_value);
  }
  for (uint64_t& peer_address : initialized.peer_addresses) {
    packed_arguments.push_back(&peer_address);
  }

  int32_t rank = static_cast<int32_t>(prepared.rank.value());
  int32_t size = prepared.clique_size;
  int32_t cuda_error = 0;
  packed_arguments.push_back(&rank);
  packed_arguments.push_back(&size);
  packed_arguments.push_back(&cuda_error);

  const RuntimeFunctions& functions = prepared.module->functions();
  for (size_t step_index = 0; step_index < config.steps.size(); ++step_index) {
    const CollectiveStepV3& step = config.steps[step_index];
    switch (step.kind) {
      case CollectiveStepKindV3::kBarrier: {
        if (initialized.barrier == nullptr) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "CuTeDSL collective step %d requires uninitialized barrier "
              "state",
              step_index));
        }
        se::DeviceAddressBase signal_value =
            initialized.barrier->signal_value->address().GetByteSlice(
                0, GetMultiGpuBarrierSignalValueSize());
        RETURN_IF_ERROR(LaunchMultiGpuBarrier(
            stream, prepared.clique_size, prepared.rank,
            initialized.barrier->peer_addresses, signal_value));
        break;
      }
      case CollectiveStepKindV3::kLaunch: {
        auto function = prepared.functions.find(step.operand);
        if (function == prepared.functions.end()) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "CuTeDSL collective function ordinal %d was not prepared",
              step.operand));
        }

        cuda_error = 0;
        CuteDSLRT_Error_t error = functions.function_run(
            function->second, packed_arguments.data(), packed_arguments.size());
        if (error != kCuteDslRtSuccess) {
          return absl::InternalError(absl::StrFormat(
              "CuTeDSL collective launch step %d failed: %s; CUDA error %d",
              step_index, FormatRuntimeError(functions, error), cuda_error));
        }
        if (cuda_error != 0) {
          return absl::InternalError(absl::StrFormat(
              "CuTeDSL collective launch step %d returned CUDA error %d",
              step_index, cuda_error));
        }
        break;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status RetainResourcesUntilStreamComplete(
    se::Stream* stream, std::shared_ptr<LoadedModule> module,
    std::shared_ptr<BarrierResourcesV3> barrier) {
  auto retained = std::make_unique<RetainedExecutionResourcesV3>(
      RetainedExecutionResourcesV3{std::move(module), std::move(barrier)});

  absl::StatusOr<std::unique_ptr<se::Event>> created_event =
      stream->parent()->CreateEvent();
  if (!created_event.ok()) {
    LOG(WARNING) << "Could not create a completion event for CuTeDSL "
                    "collective resources; synchronizing the stream: "
                 << created_event.status();
    absl::Status synchronized = stream->BlockHostUntilDone();
    if (synchronized.ok()) return absl::OkStatus();

    // Work may still reference these objects. Keeping them alive is safer
    // than releasing module or registered barrier resources after a failed
    // synchronization.
    retained.release();
    return absl::InternalError(absl::StrFormat(
        "Could not retain CuTeDSL collective resources: event creation "
        "failed (%s) and stream synchronization failed (%s)",
        created_event.status().ToString(), synchronized.ToString()));
  }

  std::unique_ptr<se::Event> event = std::move(*created_event);
  absl::Status recorded = stream->RecordEvent(event.get());
  if (!recorded.ok()) {
    LOG(WARNING) << "Could not record a completion event for CuTeDSL "
                    "collective resources; synchronizing the stream: "
                 << recorded;
    absl::Status synchronized = stream->BlockHostUntilDone();
    if (synchronized.ok()) return absl::OkStatus();

    event.release();
    retained.release();
    return absl::InternalError(absl::StrFormat(
        "Could not retain CuTeDSL collective resources: event recording "
        "failed (%s) and stream synchronization failed (%s)",
        recorded.ToString(), synchronized.ToString()));
  }

  // Waiting on a host worker avoids destroying CUDA/NCCL-backed objects from
  // a stream host callback. If synchronization reports an error, preserve the
  // resources because the device may still reference them.
  tsl::Env::Default()->SchedClosure([event = std::move(event),
                                     retained = std::move(retained)]() mutable {
    absl::Status synchronized = event->Synchronize();
    if (!synchronized.ok()) {
      LOG(ERROR) << "CuTeDSL collective completion event failed: "
                 << synchronized << "; retaining module and barrier resources";
      event.release();
      retained.release();
    }
  });
  return absl::OkStatus();
}

absl::Status Execute(se::Stream* stream, CollectiveCallStateV3* state,
                     CollectiveCallPreparedStateV3* prepared,
                     CollectiveCallInitializedStateV3* initialized,
                     ffi::RemainingArgs arguments, ffi::RemainingRets results) {
  if (stream == nullptr || state == nullptr || prepared == nullptr ||
      initialized == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 Execute requires stream and all lifecycle "
        "state");
  }
  if (stream->parent() != prepared->executor ||
      stream->parent() != initialized->executor) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 executor changed before Execute");
  }
  if (prepared->module == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective v3 module is unavailable during Execute");
  }

  const CollectiveCallConfigV3& config = state->config();
  if (config.peer_regions.size() >
      std::numeric_limits<size_t>::max() /
          static_cast<size_t>(prepared->clique_size)) {
    return absl::FailedPreconditionError(
        "CuTeDSL collective peer-address table size overflows");
  }
  size_t expected_peer_addresses =
      config.peer_regions.size() * static_cast<size_t>(prepared->clique_size);
  if (initialized->peer_addresses.size() != expected_peer_addresses) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "CuTeDSL collective peer-address table has %d entries; expected %d",
        initialized->peer_addresses.size(), expected_peer_addresses));
  }

  absl::Status execution = ExecuteConfiguredSteps(
      stream, config, *prepared, *initialized, arguments, results);
  // A failed launch can still leave earlier work in flight, so arrange safe
  // teardown for every Execute attempt after entering the step executor.
  absl::Status retention = RetainResourcesUntilStreamComplete(
      stream, prepared->module, initialized->barrier);
  if (!execution.ok()) {
    if (!retention.ok()) {
      LOG(ERROR) << "CuTeDSL collective execution failed and resource "
                    "retention also failed: "
                 << retention;
    }
    return execution;
  }
  return retention;
}

XLA_FFI_DEFINE_HANDLER(kInstantiate, Instantiate,
                       ffi::Ffi::BindInstantiate()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attrs<ffi::Dictionary>());

XLA_FFI_DEFINE_HANDLER(kPrepare, Prepare,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::State<CollectiveCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>()
                           .Ctx<ffi::CollectiveMemoryRequests>());

XLA_FFI_DEFINE_HANDLER(kInitialize, Initialize,
                       ffi::Ffi::BindInitialize()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::State<CollectiveCallStateV3>>()
                           .Ctx<ffi::Prepared<CollectiveCallPreparedStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliques>()
                           .Ctx<ffi::CollectiveMemory>());

XLA_FFI_DEFINE_HANDLER(
    kExecute, Execute,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::State<CollectiveCallStateV3>>()
        .Ctx<ffi::Prepared<CollectiveCallPreparedStateV3>>()
        .Ctx<ffi::Initialized<CollectiveCallInitializedStateV3>>()
        .RemainingArgs()
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kCollectiveCallTarget.data(),
                         "CUDA",
                         (XLA_FFI_Handler_Bundle{/*instantiate=*/kInstantiate,
                                                 /*prepare=*/kPrepare,
                                                 /*initialize=*/kInitialize,
                                                 /*execute=*/kExecute}));

}  // namespace
}  // namespace xla::gpu::cutedsl
