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

#ifndef XLA_FFI_API_NCCL_COLLECTIVE_RESOURCES_H_
#define XLA_FFI_API_NCCL_COLLECTIVE_RESOURCES_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_nccl_collective_resources.h"
#include "xla/ffi/api/ffi.h"

namespace xla::ffi {

// Selects how NcclCollectiveGroup::members is interpreted against XLA's
// [replica, partition] device assignment.
enum class NcclCollectiveGroupMode : uint8_t {
  // Members are replica IDs; XLA forms a separate clique per partition.
  kCrossReplica = XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA,
  // Members are partition IDs; XLA forms a separate clique per replica.
  kCrossPartition = XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_PARTITION,
  // Members are replica IDs; each clique includes every partition.
  kCrossReplicaAndPartition =
      XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA_AND_PARTITION,
  // Members are replica-major flattened IDs.
  kFlattenedId = XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_FLATTENED_ID,
};

// Describes a complete, equal-sized partition of the logical ID domain.
// group_offsets begins with zero, ends with members.size(), and delimits the
// groups concatenated in members. Request selects the group containing the
// current device and copies both spans before returning.
struct NcclCollectiveGroup {
  NcclCollectiveGroupMode group_mode;
  // Distinguishes independent communication channels with identical members.
  uint64_t communication_id;
  Span<const size_t> group_offsets;
  Span<const int64_t> members;
};

// Selects the device address exposed for a collective region.
enum class NcclCollectiveMemoryKind : uint8_t {
  // One directly load/store-accessible pointer per clique rank. These pointers
  // can cross physical hosts within one NCCL MNNVL LSA.
  kSymmetric = XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC,
  // One hardware multicast alias repeated across the region's address-table
  // row. Resolution verifies that all clique ranks are in one
  // multicast-capable LSA team.
  kMultimem = XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM,
};

// Identifies a byte range within one complete FFI argument or result buffer.
// XLA registers the buffer's backing allocation and later resolves the range.
struct NcclCollectiveRegion {
  // Device address and size of the complete FFI buffer.
  void* data;
  size_t containing_buffer_size;
  // Range [byte_offset, byte_offset + byte_size) within the FFI buffer.
  size_t byte_offset;
  size_t byte_size;
  // Required power-of-two alignment of every resolved address.
  size_t required_alignment;
  NcclCollectiveMemoryKind memory_kind;
};

// Identifies the current device within the clique selected by Request.
struct NcclCollectiveInfo {
  int32_t rank;
  int32_t clique_size;
};

// Device-resident region-major peer-address table materialized in
// caller-provided storage. Its contents become ready in Initialize stream
// order.
struct NcclCollectiveDeviceAddressTable {
  const uint64_t* device_data;
  size_t address_count;
};

// Describes direct load/store reachability within an acquired clique. LSA
// membership follows NCCL's logical topology and can span processes and
// physical hosts within one properly configured MNNVL/IMEX fabric.
struct NcclCollectiveTopology {
  // Number of communicator ranks in the complete XLA clique.
  int32_t clique_size;
  // Number of contiguous ranks in each equal-sized LSA team.
  int32_t lsa_size;
  // Number of LSA teams; lsa_size * lsa_team_count == clique_size.
  int32_t lsa_team_count;
  // Derived convenience value: lsa_size == clique_size &&
  // lsa_team_count == 1. This is not an independent health probe.
  bool world_is_lsa;
  // Hardware multicast support for an LSA team, not necessarily the world.
  bool multimem_supported;
};

class NcclCollectiveResources;

// Move-only owner for one execution-scoped collective resource request. It must
// outlive all device work that uses the resource. Resolve staging remains
// retained independently until its stream-ordered copy completes; destruction
// does not synchronize the GPU stream.
class NcclCollectiveResource {
 public:
  NcclCollectiveResource(const NcclCollectiveResource&) = delete;
  NcclCollectiveResource& operator=(const NcclCollectiveResource&) = delete;

  NcclCollectiveResource(NcclCollectiveResource&& other) noexcept
      : extension_(std::exchange(other.extension_, nullptr)),
        resource_(std::exchange(other.resource_, nullptr)),
        info_(other.info_),
        address_count_(other.address_count_) {}

  NcclCollectiveResource& operator=(NcclCollectiveResource&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    extension_ = std::exchange(other.extension_, nullptr);
    resource_ = std::exchange(other.resource_, nullptr);
    info_ = other.info_;
    address_count_ = other.address_count_;
    return *this;
  }

  ~NcclCollectiveResource() { Reset(); }

  const NcclCollectiveInfo& info() const { return info_; }

 private:
  friend class NcclCollectiveResources;

  NcclCollectiveResource(
      const XLA_FFI_NcclCollectiveResources_Extension* extension,
      XLA_FFI_NcclCollectiveResource* resource, NcclCollectiveInfo info,
      size_t address_count)
      : extension_(extension),
        resource_(resource),
        info_(info),
        address_count_(address_count) {}

  void Reset() {
    if (resource_ == nullptr) return;
    XLA_FFI_NcclCollectiveResources_Destroy_Args args = {};
    args.struct_size = XLA_FFI_NcclCollectiveResources_Destroy_Args_STRUCT_SIZE;
    args.resource = resource_;
    extension_->destroy(&args);
    resource_ = nullptr;
  }

  const XLA_FFI_NcclCollectiveResources_Extension* extension_ = nullptr;
  XLA_FFI_NcclCollectiveResource* resource_ = nullptr;
  NcclCollectiveInfo info_ = {};
  size_t address_count_ = 0;
};

// Header-only wrapper for the NCCL collective-resources C extension. XLA owns
// clique acquisition, symmetric registration, and optional entry
// synchronization; the handler owns only the opaque request token.
//
// The required lifecycle is:
//
//   Prepare:    Request, then Commit.
//   Initialize: Initialize, then ResolveDeviceAddresses.
//   Execute:    EnqueueBarrierBeforeLaunch, if requested.
//
// The resource token must remain alive until all enqueued device work using its
// addresses or barrier has completed.
class NcclCollectiveResources {
 public:
  // Creates a stage-scoped view over the extension in api. ctx must be the
  // execution context passed to the currently running FFI handler stage.
  NcclCollectiveResources(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx)
      : api_(api), ctx_(ctx), extension_(FindExtension(api)) {}

  // Returns true when api publishes a compatible, complete extension table.
  bool available() const { return CheckExtension().has_value(); }

  // Validates and plans a clique plus symmetric registrations during Prepare.
  // This does not publish the requests; call Commit before Prepare returns.
  // Only one request may succeed in an FFI execution. The optional barrier
  // requires every clique rank to be directly load/store accessible and does
  // not span LSA teams. It uses execution-scoped control buffers and is not
  // command-buffer compatible.
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> Request(
      const NcclCollectiveGroup& group,
      Span<const NcclCollectiveRegion> regions,
      bool barrier_before_launch) const {
    ErrorOr<const XLA_FFI_NcclCollectiveResources_Extension*> extension =
        CheckExtension();
    if (!extension.has_value()) return Unexpected(extension.error());
    if ((*extension)->request == nullptr) {
      return Unexpected(Unimplemented(
          "NCCL collective resource Request operation is unavailable"));
    }

    XLA_FFI_NcclCollectiveGroup c_group = {};
    c_group.struct_size = XLA_FFI_NcclCollectiveGroup_STRUCT_SIZE;
    c_group.group_mode =
        static_cast<XLA_FFI_NcclCollectiveGroupMode>(group.group_mode);
    c_group.communication_id = group.communication_id;
    c_group.num_groups =
        group.group_offsets.size() == 0 ? 0 : group.group_offsets.size() - 1;
    c_group.group_offsets = group.group_offsets.begin();
    c_group.num_members = group.members.size();
    c_group.members = group.members.begin();

    std::vector<XLA_FFI_NcclCollectiveRegion> c_regions;
    c_regions.reserve(regions.size());
    for (const NcclCollectiveRegion& region : regions) {
      c_regions.push_back(XLA_FFI_NcclCollectiveRegion{
          XLA_FFI_NcclCollectiveRegion_STRUCT_SIZE,
          /*extension_start=*/nullptr,
          region.data,
          region.containing_buffer_size,
          region.byte_offset,
          region.byte_size,
          region.required_alignment,
          static_cast<XLA_FFI_NcclCollectiveMemoryKind>(region.memory_kind),
      });
    }

    XLA_FFI_NcclCollectiveResources_Request_Args args = {};
    args.struct_size = XLA_FFI_NcclCollectiveResources_Request_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.group = &c_group;
    args.regions = c_regions.empty() ? nullptr : c_regions.data();
    args.region_count = c_regions.size();
    args.barrier_before_launch =
        barrier_before_launch ? uint8_t{1} : uint8_t{0};

    if (XLA_FFI_Error* error = (*extension)->request(&args)) {
      if (args.resource != nullptr) Destroy(*extension, args.resource);
      return Unexpected(TakeError(error));
    }
    if (args.resource == nullptr) {
      return Unexpected(Error::Internal(
          "NCCL collective resource Request returned a null resource"));
    }
    if (args.rank < 0 || args.clique_size <= 0 ||
        args.rank >= args.clique_size) {
      Destroy(*extension, args.resource);
      return Unexpected(Error::Internal(
          "NCCL collective resource Request returned invalid clique metadata"));
    }
    if (regions.size() > std::numeric_limits<size_t>::max() /
                             static_cast<size_t>(args.clique_size)) {
      Destroy(*extension, args.resource);
      return Unexpected(Error::Internal(
          "NCCL collective resource Request returned a clique size that "
          "overflows the address table"));
    }

    NcclCollectiveInfo info = {args.rank, args.clique_size};
    size_t address_count =
        regions.size() * static_cast<size_t>(args.clique_size);
    return std::unique_ptr<NcclCollectiveResource>(new NcclCollectiveResource(
        *extension, args.resource, std::move(info), address_count));
  }

  // Publishes a planned resource during Prepare so XLA can acquire its clique
  // and memory registrations before Initialize. May be called exactly once.
  Error Commit(const NcclCollectiveResource& resource) const {
    Error association = CheckResource(resource);
    if (association.failure()) return association;
    if (extension_->commit == nullptr) {
      return Unimplemented(
          "NCCL collective resource Commit operation is unavailable");
    }

    XLA_FFI_NcclCollectiveResources_Commit_Args args = {};
    args.struct_size = XLA_FFI_NcclCollectiveResources_Commit_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.resource = resource.resource_;
    XLA_FFI_Error* error = extension_->commit(&args);
    return error == nullptr ? Error::Success() : TakeError(error);
  }

  // Associates a committed token with resources acquired by XLA. Call during
  // Initialize before ResolveDeviceAddresses.
  Error Initialize(const NcclCollectiveResource& resource) const {
    Error association = CheckResource(resource);
    if (association.failure()) return association;
    if (extension_->initialize == nullptr) {
      return Unimplemented(
          "NCCL collective resource Initialize operation is unavailable");
    }

    XLA_FFI_NcclCollectiveResources_Initialize_Args args = {};
    args.struct_size =
        XLA_FFI_NcclCollectiveResources_Initialize_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.resource = resource.resource_;
    XLA_FFI_Error* error = extension_->initialize(&args);
    return error == nullptr ? Error::Success() : TakeError(error);
  }

  // Returns direct-access topology metadata during Initialize. Call after
  // Initialize succeeds. A healthy single-clique MNNVL communicator can report
  // the complete cross-host clique as one LSA; ordinary network connectivity
  // or healthy IMEX alone does not imply that result.
  ErrorOr<NcclCollectiveTopology> QueryTopology(
      const NcclCollectiveResource& resource) const {
    Error association = CheckResource(resource);
    if (association.failure()) return Unexpected(std::move(association));
    if (extension_->query_topology == nullptr) {
      return Unexpected(
          Unimplemented("NCCL collective topology operation is unavailable"));
    }

    XLA_FFI_NcclCollectiveTopology topology = {};
    topology.struct_size = XLA_FFI_NcclCollectiveTopology_STRUCT_SIZE;
    XLA_FFI_NcclCollectiveResources_QueryTopology_Args args = {};
    args.struct_size =
        XLA_FFI_NcclCollectiveResources_QueryTopology_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.resource = resource.resource_;
    args.topology = &topology;
    if (XLA_FFI_Error* error = extension_->query_topology(&args)) {
      return Unexpected(TakeError(error));
    }
    bool world_is_lsa = topology.lsa_size == topology.clique_size &&
                        topology.lsa_team_count == 1;
    if (topology.clique_size != resource.info_.clique_size ||
        topology.lsa_size <= 0 || topology.lsa_size > topology.clique_size ||
        topology.lsa_team_count <= 0 ||
        static_cast<int64_t>(topology.lsa_size) * topology.lsa_team_count !=
            topology.clique_size ||
        (topology.world_is_lsa != 0) != world_is_lsa) {
      return Unexpected(Error::Internal(
          "NCCL collective topology operation returned invalid metadata"));
    }
    return NcclCollectiveTopology{
        topology.clique_size,
        topology.lsa_size,
        topology.lsa_team_count,
        world_is_lsa,
        topology.multimem_supported != 0,
    };
  }

  // Materializes a region-major address table in caller-provided device storage
  // during Initialize. Storage may be larger than the logical table but must
  // contain at least one entry per requested region and clique rank, fit in one
  // XLA buffer allocation, and not overlap a requested collective region.
  // Symmetric rows have one address per rank; multimem rows repeat one
  // multicast alias after verifying that the complete clique is directly
  // accessible. Request a byte range twice when the kernel needs both
  // representations. A transient result is insufficient because its allocation
  // can be reused before Execute; a dedicated entry operand aliased to a result
  // can reserve storage for the complete execution. Storage must be a declared
  // FFI argument or result. For command-buffer-compatible handlers, that also
  // lets XLA track its address across graph replay. This does not make other
  // handler operations command-buffer compatible.
  ErrorOr<NcclCollectiveDeviceAddressTable> ResolveDeviceAddresses(
      const NcclCollectiveResource& resource,
      BufferR1<DataType::U64> device_storage) const {
    Error association = CheckResource(resource);
    if (association.failure()) return Unexpected(std::move(association));
    if (extension_->resolve == nullptr) {
      return Unexpected(Unimplemented(
          "NCCL collective resource Resolve operation is unavailable"));
    }
    size_t address_capacity = device_storage.element_count();
    uint64_t* device_data = device_storage.typed_data();
    if (address_capacity < resource.address_count_) {
      return Unexpected(Error::InvalidArgument(
          "NCCL collective device address storage is too small"));
    }
    if (address_capacity != 0 && device_data == nullptr) {
      return Unexpected(Error::InvalidArgument(
          "NCCL collective device address storage must not be null"));
    }

    XLA_FFI_NcclCollectiveDeviceAddressTable table = {};
    table.struct_size = XLA_FFI_NcclCollectiveDeviceAddressTable_STRUCT_SIZE;
    table.device_data = device_data;
    table.address_capacity = address_capacity;
    XLA_FFI_NcclCollectiveResources_Resolve_Args args = {};
    args.struct_size = XLA_FFI_NcclCollectiveResources_Resolve_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.resource = resource.resource_;
    args.table = &table;
    if (XLA_FFI_Error* error = extension_->resolve(&args)) {
      return Unexpected(TakeError(error));
    }
    if (table.device_data != device_data ||
        table.address_capacity != address_capacity ||
        table.address_count != resource.address_count_) {
      return Unexpected(Error::Internal(
          "NCCL collective resource Resolve returned an invalid device "
          "address table"));
    }
    return NcclCollectiveDeviceAddressTable{table.device_data,
                                            table.address_count};
  }

  // Begins collective execution by enqueueing the entry synchronization
  // requested during Prepare. This may be called at most once and does not
  // synchronize multiple LSA teams.
  Error EnqueueBarrierBeforeLaunch(
      const NcclCollectiveResource& resource) const {
    Error association = CheckResource(resource);
    if (association.failure()) return association;
    if (extension_->enqueue_barrier_before_launch == nullptr) {
      return Unimplemented(
          "NCCL collective pre-launch barrier operation is unavailable");
    }

    XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args args = {};
    args.struct_size =
        XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.resource = resource.resource_;
    XLA_FFI_Error* error = extension_->enqueue_barrier_before_launch(&args);
    return error == nullptr ? Error::Success() : TakeError(error);
  }

 private:
  static const XLA_FFI_NcclCollectiveResources_Extension* FindExtension(
      const XLA_FFI_Api* api) {
    if (api == nullptr) return nullptr;
    XLA_FFI_Extension_Base* extension = api->extension_start;
    while (extension != nullptr) {
      if (extension->struct_size < XLA_FFI_Extension_Base_STRUCT_SIZE) {
        return nullptr;
      }
      if (extension->type == XLA_FFI_Extension_NcclCollectiveResources) {
        return reinterpret_cast<
            const XLA_FFI_NcclCollectiveResources_Extension*>(extension);
      }
      extension = extension->next;
    }
    return nullptr;
  }

  ErrorOr<const XLA_FFI_NcclCollectiveResources_Extension*> CheckExtension()
      const {
    if (extension_ == nullptr) {
      return Unexpected(
          Unimplemented("NCCL collective-resources extension not found"));
    }
    if (extension_->extension_base.struct_size <
        XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_0_1_STRUCT_SIZE) {
      return Unexpected(Unimplemented(
          "NCCL collective-resources extension table is incomplete"));
    }
    if (extension_->abi_major_version !=
        XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_MAJOR) {
      return Unexpected(
          Unimplemented("Incompatible NCCL collective-resources extension ABI "
                        "major version"));
    }
    if (extension_->abi_minor_version <
        XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_MINOR) {
      return Unexpected(Unimplemented(
          "NCCL collective-resources extension ABI minor version is older "
          "than the client requires"));
    }
    if (extension_->destroy == nullptr) {
      return Unexpected(Unimplemented(
          "NCCL collective resource Destroy operation is unavailable"));
    }
    return extension_;
  }

  Error CheckResource(const NcclCollectiveResource& resource) const {
    ErrorOr<const XLA_FFI_NcclCollectiveResources_Extension*> extension =
        CheckExtension();
    if (!extension.has_value()) return extension.error();
    if (resource.resource_ == nullptr || resource.extension_ != *extension) {
      return Error::InvalidArgument(
          "NCCL collective resource does not belong to this context");
    }
    return Error::Success();
  }

  Error TakeError(XLA_FFI_Error* error) const {
    XLA_FFI_Error_Code errc = XLA_FFI_Error_Code_UNKNOWN;
    if (api_->struct_size >=
            XLA_FFI_STRUCT_SIZE(XLA_FFI_Api, XLA_FFI_Error_GetCode) &&
        api_->XLA_FFI_Error_GetCode != nullptr) {
      XLA_FFI_Error_GetCode_Args args = {};
      args.struct_size = XLA_FFI_Error_GetCode_Args_STRUCT_SIZE;
      args.error = error;
      api_->XLA_FFI_Error_GetCode(&args);
      errc = args.errc;
    }

    const char* message = internal::GetErrorMessage(api_, error);
    std::string owned_message = message == nullptr ? std::string() : message;
    internal::DestroyError(api_, error);
    return Error(errc, std::move(owned_message));
  }

  static void Destroy(
      const XLA_FFI_NcclCollectiveResources_Extension* extension,
      XLA_FFI_NcclCollectiveResource* resource) {
    XLA_FFI_NcclCollectiveResources_Destroy_Args args = {};
    args.struct_size = XLA_FFI_NcclCollectiveResources_Destroy_Args_STRUCT_SIZE;
    args.resource = resource;
    extension->destroy(&args);
  }

  static Error Unimplemented(std::string message) {
    return Error(ErrorCode::kUnimplemented, std::move(message));
  }

  const XLA_FFI_Api* api_;
  XLA_FFI_ExecutionContext* ctx_;
  const XLA_FFI_NcclCollectiveResources_Extension* extension_;
};

template <>
struct CtxDecoding<NcclCollectiveResources> {
  using Type = NcclCollectiveResources;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    return NcclCollectiveResources(api, ctx);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_NCCL_COLLECTIVE_RESOURCES_H_
