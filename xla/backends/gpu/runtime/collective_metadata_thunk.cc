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

#include "xla/backends/gpu/runtime/collective_metadata_thunk.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"
#include "xla/backends/gpu/runtime/collective_kernel_api.h"
#include "xla/backends/gpu/runtime/collective_multimem.h"
#include "xla/backends/gpu/runtime/collective_multimem_registry.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// TODO(460077850): Support global device ids and channel id.
CollectiveConfig CollectiveMetadataThunk::GetCollectiveConfig(
    const HloInstruction& hlo) {
  CollectiveConfig config;
  config.operand_element_type.reserve(hlo.operands().size());
  for (const HloInstruction* operand : hlo.operands()) {
    config.operand_element_type.push_back(operand->shape().element_type());
  }

  if (hlo.has_backend_config()) {
    xla::gpu::GpuBackendConfig backend_config =
        hlo.backend_config<GpuBackendConfig>().value_or(GpuBackendConfig());
    if (backend_config.has_collective_metadata_backend_config()) {
      ::google::protobuf::RepeatedPtrField<ReplicaGroup> replica_groups =
          backend_config.collective_metadata_backend_config()
              .collective_devices()
              .replica_groups();
      config.replica_groups = std::vector<ReplicaGroup>(replica_groups.begin(),
                                                        replica_groups.end());
    }
  }

  config.group_mode = CollectiveOpGroupMode::
      COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;

  return config;
}

absl::StatusOr<std::vector<void*>> CollectiveMetadataThunk::CollectParamToPeers(
    const GpuCliqueKey& clique_key, RankId rank, se::Stream* stream,
    std::vector<se::DeviceAddressBase> parameters) {
  return xla::gpu::CollectParamToPeers(clique_key, rank, stream,
                                       std::move(parameters));
}

absl::Status CollectiveMetadataThunk::CopyCollectiveMetadataToDevice(
    se::Stream* stream, CollectiveKernelMetadata metadata,
    const std::vector<void*>& param_to_peers_ptrs,
    const std::vector<void*>& multimem_addresses,
    se::DeviceAddressBase destination) {
  return xla::gpu::CopyCollectiveMetadataToDevice(
      stream, metadata, param_to_peers_ptrs, multimem_addresses, destination);
}

absl::Status CollectiveMetadataThunk::Prepare(const PrepareParams& params) {
  // We currently support only a single memory space for multimem parameters.
  // So we just pick the first one here.
  auto fast_memory_parameter =
      absl::c_find_if(parameters_, [](const Buffer& parameter) {
        return parameter.memory_space == xla::Layout::kGenericFastMemorySpace;
      });
  if (fast_memory_parameter == parameters_.end()) {
    return absl::OkStatus();
  }

  se::DeviceAddressBase memory_range;
  TF_ASSIGN_OR_RETURN(memory_range,
                      params.executor->GetMemoryRange(
                          params.buffer_allocations->GetDeviceAddress(
                              fast_memory_parameter->slice)));

  // Since there is no parameter in the collective memory space, we don't need
  // to set up the collective multimem.
  if (memory_range.is_null()) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*is_p2p=*/false));
  params.multimem_registry->Request({clique_key, /*map_to=*/memory_range});
  return absl::OkStatus();
}

absl::Status CollectiveMetadataThunk::Initialize(
    const InitializeParams& params) {
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*is_p2p=*/false));
  const int64_t num_ranks = clique_key.num_devices();
  TF_RET_CHECK(result_.size() ==
               sizeof(CollectiveKernelMetadata) +
                   num_ranks * parameters_.size() * sizeof(uint64_t) +
                   parameters_.size() * sizeof(uint64_t));

  std::vector<se::DeviceAddressBase> parameters;
  parameters.reserve(parameters_.size());
  for (const Buffer& parameter : parameters_) {
    parameters.push_back(
        params.buffer_allocations->GetDeviceAddress(parameter.slice));
  }
  se::DeviceAddressBase result_ptr =
      params.buffer_allocations->GetDeviceAddress(result_);

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(auto multimem, GetCollectiveMultimem(clique_key, params));

  std::optional<RankId> rank = clique_key.rank(global_device_id);
  TF_RET_CHECK(rank.has_value());

  TF_ASSIGN_OR_RETURN(std::vector<void*> param_to_peers_ptrs,
                      CollectParamToPeers(clique_key, *rank, params.stream,
                                          std::move(parameters)));
  CollectiveKernelMetadata metadata;
  metadata.rank = rank->value();

  std::vector<void*> multimem_addresses(parameters_.size(), nullptr);
  TF_RETURN_IF_ERROR(CopyCollectiveMetadataToDevice(
      params.stream, metadata, param_to_peers_ptrs, multimem_addresses,
      result_ptr));
  TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
  return absl::OkStatus();
}

absl::Status CollectiveMetadataThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CollectiveMultimem>>
CollectiveMetadataThunk::GetCollectiveMultimem(const GpuCliqueKey& clique_key,
                                               const InitializeParams& params) {
  se::DeviceAddressBase memory_range;
  for (const Buffer& parameter : parameters_) {
    if (parameter.memory_space == xla::Layout::kGenericFastMemorySpace) {
      TF_ASSIGN_OR_RETURN(
          memory_range,
          params.executor->GetMemoryRange(
              params.buffer_allocations->GetDeviceAddress(parameter.slice)));
      break;
    }
  }

  // Since there is no parameter in the collective memory space, we don't need
  // to set up the collective multimem.
  if (memory_range.is_null()) {
    return nullptr;
  }

  const MultimemRequest request{clique_key, memory_range};
  TF_ASSIGN_OR_RETURN(std::shared_ptr<CollectiveMultimem> collective_multimem,
                      params.multicast_memory_registry->Get(request));
  absl::MutexLock lock(mutex_);
  return (collective_multimem_[params.executor] =
              std::move(collective_multimem));
}

}  // namespace gpu
}  // namespace xla
