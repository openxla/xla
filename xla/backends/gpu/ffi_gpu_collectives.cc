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

#include "xla/backends/gpu/ffi_gpu_collectives.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi_gpu_collectives.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::ffi {
const XLA_FFI_Api* GetXlaFfiApi();
}  // namespace xla::ffi

namespace xla::gpu {
namespace {

absl::Status ActualStructSizeIsGreaterOrEqual(absl::string_view struct_name,
                                              size_t expected, size_t actual) {
  if (actual < expected) {
    return InvalidArgument("Unexpected %s size: expected %zu, got %zu",
                           struct_name, expected, actual);
  }
  if (actual > expected) {
    VLOG(2) << "Unexpected " << struct_name << " size: expected " << expected
            << ", got " << actual << ". Check installed software versions.";
  }
  return absl::OkStatus();
}

XLA_FFI_Error* MakeError(const absl::Status& status) {
  if (status.ok()) {
    return nullptr;
  }
  const std::string message(status.message());
  XLA_FFI_Error_Create_Args args;
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.message = message.c_str();
  args.errc = static_cast<XLA_FFI_Error_Code>(status.code());
  return ffi::GetXlaFfiApi()->XLA_FFI_Error_Create(&args);
}

#define XLA_FFI_RETURN_IF_ERROR(expr) \
  do {                                \
    absl::Status _status = (expr);    \
    if (!_status.ok()) {              \
      return MakeError(_status);      \
    }                                 \
  } while (false)

#define XLA_FFI_STATUS_MACROS_CONCAT_INNER(x, y) x##y
#define XLA_FFI_STATUS_MACROS_CONCAT(x, y) \
  XLA_FFI_STATUS_MACROS_CONCAT_INNER(x, y)

#define XLA_FFI_ASSIGN_OR_RETURN(lhs, rexpr)                             \
  XLA_FFI_ASSIGN_OR_RETURN_IMPL(                                         \
      XLA_FFI_STATUS_MACROS_CONCAT(_ffi_status_or_value, __LINE__), lhs, \
      rexpr)

#define XLA_FFI_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                  \
  if (!statusor.ok()) {                                     \
    return MakeError(std::move(statusor).status());         \
  }                                                         \
  lhs = std::move(statusor).value()

absl::StatusOr<CollectiveOpGroupMode> ToCollectiveOpGroupMode(
    XLA_FFI_Gpu_GroupMode group_mode) {
  switch (group_mode) {
    case XLA_FFI_GPU_GROUP_CROSS_REPLICA:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
    case XLA_FFI_GPU_GROUP_CROSS_PARTITION:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION;
    case XLA_FFI_GPU_GROUP_CROSS_REPLICA_AND_PARTITION:
      return CollectiveOpGroupMode::
          COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
    case XLA_FFI_GPU_GROUP_FLATTENED_ID:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID;
    default:
      return InvalidArgument("Invalid GPU collective group mode: %d",
                             static_cast<int>(group_mode));
  }
}

absl::StatusOr<std::vector<ReplicaGroup>> ToReplicaGroups(
    const XLA_FFI_Gpu_ReplicaGroup* groups, size_t num_groups) {
  if (groups == nullptr && num_groups != 0) {
    return InvalidArgument("groups must be set when num_groups is non-zero");
  }

  std::vector<ReplicaGroup> replica_groups;
  replica_groups.reserve(num_groups);
  for (size_t i = 0; i < num_groups; ++i) {
    if (groups[i].ids == nullptr && groups[i].size != 0) {
      return InvalidArgument(
          "group ids must be set when group size is non-zero");
    }
    ReplicaGroup replica_group;
    for (size_t j = 0; j < groups[i].size; ++j) {
      replica_group.add_replica_ids(groups[i].ids[j]);
    }
    replica_groups.push_back(std::move(replica_group));
  }
  return replica_groups;
}

absl::StatusOr<GpuCliqueKey> GetCliqueKey(
    const CollectiveParams& params, XLA_FFI_Gpu_GroupMode group_mode,
    const std::vector<ReplicaGroup>& replica_groups, int64_t communication_id) {
  if (communication_id < 0) {
    return InvalidArgument("communication_id must be non-negative");
  }
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode mode,
                      ToCollectiveOpGroupMode(group_mode));
  return GetGpuCliqueKey(params, replica_groups, mode,
                         CommunicationId(communication_id));
}

absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>> GetDeviceGroups(
    const CollectiveParams& params, XLA_FFI_Gpu_GroupMode group_mode,
    const std::vector<ReplicaGroup>& replica_groups) {
  TF_RET_CHECK(params.device_assn != nullptr)
      << "Device assignment is required for GPU communicator FFI calls";

  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode mode,
                      ToCollectiveOpGroupMode(group_mode));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<GlobalDeviceId>> device_groups,
      GetParticipatingDevicesGroups(*params.device_assn, replica_groups, mode));

  for (auto& group : device_groups) {
    absl::c_sort(group);
  }
  absl::c_sort(device_groups);
  return device_groups;
}

XLA_FFI_Error* CommunicatorRequest(
    const XLA_FFI_Gpu_Collectives_Extension* self,
    XLA_FFI_Gpu_Communicator_Request_Args* args) {
  if (self == nullptr) {
    return MakeError(
        InvalidArgument("GPU collectives extension is not available"));
  }
  if (args == nullptr) {
    return MakeError(
        InvalidArgument("XLA_FFI_Gpu_Communicator_Request_Args is null"));
  }
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Gpu_Communicator_Request_Args",
      XLA_FFI_Gpu_Communicator_Request_Args_STRUCT_SIZE, args->struct_size));

  const auto* params =
      reinterpret_cast<const CollectiveParams*>(self->collective_params);
  if (params == nullptr) {
    return MakeError(InvalidArgument("Collective params are not available"));
  }

  auto* clique_requests = reinterpret_cast<CollectiveCliqueRequests*>(
      self->collective_clique_requests);
  if (clique_requests == nullptr) {
    return MakeError(FailedPrecondition(
        "GPU communicator request is only available during the prepare stage"));
  }

  XLA_FFI_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                           ToReplicaGroups(args->groups, args->num_groups));
  XLA_FFI_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCliqueKey(*params, args->group_mode, replica_groups,
                   args->communication_id));
  XLA_FFI_ASSIGN_OR_RETURN(
      std::vector<std::vector<GlobalDeviceId>> device_groups,
      GetDeviceGroups(*params, args->group_mode, replica_groups));
  XLA_FFI_RETURN_IF_ERROR(
      clique_requests->RequestClique(clique_key, device_groups));
  return nullptr;
}

XLA_FFI_Error* CommunicatorGet(const XLA_FFI_Gpu_Collectives_Extension* self,
                               XLA_FFI_Gpu_Communicator_Get_Args* args) {
  if (self == nullptr) {
    return MakeError(
        InvalidArgument("GPU collectives extension is not available"));
  }
  if (args == nullptr) {
    return MakeError(
        InvalidArgument("XLA_FFI_Gpu_Communicator_Get_Args is null"));
  }
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Gpu_Communicator_Get_Args",
      XLA_FFI_Gpu_Communicator_Get_Args_STRUCT_SIZE, args->struct_size));

  const auto* params =
      reinterpret_cast<const CollectiveParams*>(self->collective_params);
  if (params == nullptr) {
    return MakeError(InvalidArgument("Collective params are not available"));
  }

  const auto* cliques =
      reinterpret_cast<const CollectiveCliques*>(self->collective_cliques);
  if (cliques == nullptr) {
    return MakeError(FailedPrecondition(
        "GPU communicator get is only available after cliques are acquired"));
  }

  XLA_FFI_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                           ToReplicaGroups(args->groups, args->num_groups));
  XLA_FFI_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCliqueKey(*params, args->group_mode, replica_groups,
                   args->communication_id));
  XLA_FFI_ASSIGN_OR_RETURN(
      GpuCommunicator * comm,
      cliques->GetComm(clique_key, params->global_device_id));

  PlatformCommunicatorHandle platform_comm = comm->platform_comm();
  if (platform_comm.handle == nullptr) {
    return MakeError(
        Unimplemented("Platform communicator handle is not available"));
  }

  args->communicator = platform_comm.handle;
  return nullptr;
}

#undef XLA_FFI_ASSIGN_OR_RETURN_IMPL
#undef XLA_FFI_ASSIGN_OR_RETURN
#undef XLA_FFI_STATUS_MACROS_CONCAT
#undef XLA_FFI_STATUS_MACROS_CONCAT_INNER
#undef XLA_FFI_RETURN_IF_ERROR

}  // namespace

XLA_FFI_Gpu_Collectives_Extension MakeGpuCollectivesExtension(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* collective_clique_requests,
    const CollectiveCliques* collective_cliques) {
  XLA_FFI_Gpu_Collectives_Extension extension;
  extension.extension_base = XLA_FFI_Extension{
      /*struct_size=*/sizeof(XLA_FFI_Gpu_Collectives_Extension),
      /*id=*/
      XLA_FFI_ExtensionId{
          /*extension_type=*/XLA_FFI_Gpu_Collectives_Extension_Type,
          /*major_version=*/XLA_FFI_Gpu_Collectives_Extension_Major,
          /*minor_version=*/XLA_FFI_Gpu_Collectives_Extension_Minor},
      /*next=*/nullptr};
  extension.collective_params =
      const_cast<void*>(reinterpret_cast<const void*>(collective_params));
  extension.collective_clique_requests =
      reinterpret_cast<void*>(collective_clique_requests);
  extension.collective_cliques =
      const_cast<void*>(reinterpret_cast<const void*>(collective_cliques));
  extension.request_communicator = CommunicatorRequest;
  extension.get_communicator = CommunicatorGet;
  return extension;
}

}  // namespace xla::gpu
