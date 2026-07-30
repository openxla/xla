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

#ifndef XLA_FFI_API_C_API_NCCL_COLLECTIVE_RESOURCES_H_
#define XLA_FFI_API_C_API_NCCL_COLLECTIVE_RESOURCES_H_

#include <stddef.h>
#include <stdint.h>
#include "xla/ffi/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provides execution-scoped NCCL resources for FFI handlers that implement
// collectives in NVIDIA GPU device code. XLA derives and acquires the requested
// NCCL communicator, registers FFI buffers as symmetric memory, resolves device
// addresses, and can provide a stream-ordered barrier before the handler's
// kernel launch.
//
// This is the same resource model used by XLA's Mosaic GPU collective-metadata
// path: the handler receives a clique rank, a device-resident table of peer
// pointers for symmetric buffers and optional multimem aliases, and an optional
// clique-wide prefix barrier.
//
// A handler uses the extension in the following FFI stages:
//
//   Prepare:    Request, then Commit.
//   Initialize: Initialize, then Resolve.
//   Execute:    EnqueueBarrierBeforeLaunch, if requested.
//   Any stage:  Destroy after all device work using the resource has finished.
//
// The extension ABI is versioned independently from the root XLA FFI API.
// Existing structs are frozen. A major version change may break compatibility.
// A minor version may only append fields to the extension table or add new
// structs. Clients must require an exact major version, a runtime minor version
// greater than or equal to the client minor version, and sufficient struct_size
// before reading the table.
#define XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_MAJOR 0
#define XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_MINOR 1

typedef struct XLA_FFI_NcclCollectiveResource XLA_FFI_NcclCollectiveResource;

// Selects how members in XLA_FFI_NcclCollectiveGroup are interpreted against
// XLA's [replica, partition] device assignment.
typedef enum {
  // Members are replica IDs. A separate clique is formed for each partition.
  XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA = 0,

  // Members are partition IDs. A separate clique is formed for each replica.
  XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_PARTITION = 1,

  // Members are replica IDs. Each clique contains the listed replicas across
  // every partition.
  XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA_AND_PARTITION = 2,

  // Members are flattened replica-major IDs: replica * partition_count +
  // partition.
  XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_FLATTENED_ID = 3,
} XLA_FFI_NcclCollectiveGroupMode;

// Describes a complete, equal-sized partition of the logical ID domain selected
// by group_mode. group_offsets has num_groups + 1 elements, begins with zero,
// ends with num_members, and indexes members. Every logical ID must occur
// exactly once. Request selects the group containing the current device.
//
// Callers retain ownership of both arrays; Request copies everything it needs.
struct XLA_FFI_NcclCollectiveGroup {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  // Determines whether members contains replica, partition, or flattened IDs.
  XLA_FFI_NcclCollectiveGroupMode group_mode;

  // Distinguishes independent communication channels with identical members.
  // Every rank participating in an operation must use the same value.
  uint64_t communication_id;

  // Number of groups represented by group_offsets and members.
  size_t num_groups;

  // Offsets delimiting each group in members. Contains num_groups + 1 entries.
  const size_t* group_offsets;

  // Number of logical IDs in members.
  size_t num_members;

  // Concatenated logical IDs for all groups.
  const int64_t* members;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveGroup, members);

typedef enum {
  // Resolves one load/store-accessible device pointer per clique rank. Address
  // resolution calls the backend's peer-address API for every non-local rank
  // and fails unless the complete clique is directly accessible. The returned
  // pointers may span processes and physical hosts when NCCL exposes one
  // MNNVL LSA through CUDA fabric handles and IMEX.
  XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC = 0,

  // Resolves an LSA multimem alias for hardware multicast operations. The same
  // alias is repeated for every rank in the region's address-table row.
  // Resolve verifies peer access to every clique rank before returning the
  // alias, so this kind succeeds only when the complete clique is one
  // multicast-capable LSA team. It can span hosts within one MNNVL partition;
  // it cannot span separate NVLink cliques through a network transport.
  XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM = 1,
} XLA_FFI_NcclCollectiveMemoryKind;

// Describes one byte range in an FFI argument or result. data and
// containing_buffer_size identify the complete logical FFI buffer; the
// requested range is [byte_offset, byte_offset + byte_size). XLA registers the
// complete backing allocation and resolves the requested range after
// collective acquisition.
struct XLA_FFI_NcclCollectiveRegion {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  // Device address of the complete FFI argument or result buffer.
  void* data;
  // Size in bytes of the complete FFI buffer beginning at data.
  size_t containing_buffer_size;
  // Offset in bytes from data to the first byte exposed to the handler.
  size_t byte_offset;
  // Number of bytes exposed to the handler.
  size_t byte_size;
  // Required power-of-two alignment of every resolved address.
  size_t required_alignment;
  // Address kind to resolve for this region.
  XLA_FFI_NcclCollectiveMemoryKind memory_kind;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveRegion, memory_kind);

// Validates and plans one execution-scoped clique and the symmetric
// registrations needed by regions. This operation is valid only during FFI
// Prepare and does not publish resource requests; Commit publishes the plan.
// Only one Request may succeed per FFI execution.
//
// On success, resource is non-null and owned by the caller. rank is the current
// device's zero-based rank in the derived clique, and clique_size is the number
// of ranks. The caller must eventually destroy resource.
//
// The caller must keep `resource` alive until all device work that uses its
// resolved addresses or prefix barrier has completed. Resolve retains its host
// staging independently until the stream-ordered copy completes. Destroy does
// not synchronize device work or make early destruction safe.
struct XLA_FFI_NcclCollectiveResources_Request_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  const XLA_FFI_NcclCollectiveGroup* group;
  const XLA_FFI_NcclCollectiveRegion* regions;
  size_t region_count;
  // When nonzero, asks XLA to create a one-shot barrier that the handler must
  // enqueue before its kernel. Every clique rank must be directly accessible
  // from every other rank; cliques spanning multiple LSA teams are not
  // supported. This mode uses execution-scoped control buffers and is not
  // command-buffer compatible.
  uint8_t barrier_before_launch;
  XLA_FFI_NcclCollectiveResource* resource;  // out
  int32_t rank;                              // out
  int32_t clique_size;                       // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Request_Args,
                             clique_size);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_Request(
    XLA_FFI_NcclCollectiveResources_Request_Args* args);

// Publishes a validated Request to XLA's clique and symmetric-memory resource
// collectors. XLA acquires the published resources between Prepare and
// Initialize. This operation is valid only during FFI Prepare and exactly once
// per resource. Initialize rejects resources that were not committed.
struct XLA_FFI_NcclCollectiveResources_Commit_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Commit_Args,
                             resource);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_Commit(
    XLA_FFI_NcclCollectiveResources_Commit_Args* args);

// Associates a committed token with the clique and symmetric-memory resources
// acquired by XLA after Prepare. If requested, this also creates control memory
// for the prefix barrier. This operation is valid only during FFI Initialize
// and must precede Resolve.
struct XLA_FFI_NcclCollectiveResources_Initialize_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Initialize_Args,
                             resource);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_Initialize(
    XLA_FFI_NcclCollectiveResources_Initialize_Args* args);

// Describes the load/store-accessible topology of an acquired clique. An LSA
// (load/store accessible) team is a set of communicator ranks whose registered
// symmetric memory can be addressed directly from GPU device code. Teams are
// determined by NCCL's logical topology, not by process or physical-host
// boundaries.
//
// With NCCL 2.29, a communicator contained in one healthy MNNVL partition can
// appear as one logical topology node across multiple hosts. When CUDA fabric
// handles can be exchanged through IMEX, this commonly produces
// lsa_size == clique_size and lsa_team_count == 1. IMEX health alone is not
// sufficient: every communicator rank must also belong to that MNNVL clique,
// and NCCL must enable and discover the fabric topology.
struct XLA_FFI_NcclCollectiveTopology {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  // Number of ranks in the complete clique.
  int32_t clique_size;
  // Number of contiguous communicator ranks in the current rank's LSA team,
  // including the current rank. NCCL uses equal-sized teams; a rank's team is
  // rank / lsa_size and its rank within that team is rank % lsa_size.
  int32_t lsa_size;
  // Number of equal-sized LSA teams partitioning the complete clique. The
  // invariant lsa_size * lsa_team_count == clique_size always holds.
  int32_t lsa_team_count;
  // Convenience value derived from lsa_size and lsa_team_count. It is nonzero
  // exactly when lsa_size == clique_size and lsa_team_count == 1; it is not an
  // independent fabric-health probe.
  uint8_t world_is_lsa;
  // Nonzero when NCCL reports hardware multicast support for an LSA team. This
  // does not imply the complete clique is one LSA, and an individual multimem
  // address can still fail when its symmetric window is ineligible.
  uint8_t multimem_supported;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveTopology,
                             multimem_supported);

// Returns topology metadata for an initialized resource. This operation is
// valid only during FFI Initialize and after Initialize has succeeded. The
// caller owns topology and must initialize its struct_size.
struct XLA_FFI_NcclCollectiveResources_QueryTopology_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
  XLA_FFI_NcclCollectiveTopology* topology;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_QueryTopology_Args,
                             topology);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_QueryTopology(
    XLA_FFI_NcclCollectiveResources_QueryTopology_Args* args);

// Caller-provided device storage for a region-major address table. The caller
// must reserve the storage for the table before Initialize and keep it alive
// through all device work that reads it. Its contents become ready in
// Initialize stream order and are visible to later Execute work under XLA's
// normal FFI execution ordering.
struct XLA_FFI_NcclCollectiveDeviceAddressTable {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  // Caller-provided writable device storage. Null is valid only when
  // address_capacity is zero.
  uint64_t* device_data;
  // Number of uint64_t entries available at device_data.
  size_t address_capacity;
  // Number of entries materialized by Resolve. The caller initializes this to
  // zero; Resolve sets it to region_count * clique_size on success. Resolve
  // does not change device_data or address_capacity.
  size_t address_count;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveDeviceAddressTable,
                             address_count);

// Resolves and materializes the resource's region-major address table in the
// caller-provided device storage. address_capacity must be at least
// region_count * clique_size. Resolve writes only that logical prefix; any
// remaining entries are unchanged. The complete capacity must fit in one XLA
// buffer allocation, and the logical prefix must not overlap a requested
// collective region. Entry
// device_data[region * clique_size + rank] describes that region for that rank.
// For MULTIMEM regions, every entry in a row contains the same multicast alias,
// after Resolve verifies that all clique ranks are load/store accessible from
// the current rank. Request the same byte range once as SYMMETRIC and once as
// MULTIMEM when a kernel needs both per-rank pointers and a multicast alias.
// Zero regions materialize no entries and permit null, zero-capacity storage.
// This operation is valid only during FFI Initialize and may be called once.
// Storage must not be a transient result whose allocation can be reused before
// the handler executes. A dedicated entry operand aliased to a result is one
// way to reserve it for the complete execution. The storage must be a declared
// FFI argument or result. For command-buffer-compatible handlers, that also
// lets XLA track its address across graph replay. Tracked storage is necessary
// but does not by itself make every handler operation command-buffer
// compatible.
struct XLA_FFI_NcclCollectiveResources_Resolve_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
  XLA_FFI_NcclCollectiveDeviceAddressTable* table;  // in/out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Resolve_Args,
                             table);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_Resolve(
    XLA_FFI_NcclCollectiveResources_Resolve_Args* args);

// Writes the resource's region-major address table to caller-provided host
// storage. address_count must equal region_count * clique_size. Entry
// addresses[region * clique_size + rank] has the same meaning as the
// corresponding entry materialized by Resolve. Zero regions permit a null
// addresses pointer. This operation is valid only during FFI Initialize.
// Resolve and ResolveHost share one resolution attempt: exactly one of them may
// succeed for a resource.
struct XLA_FFI_NcclCollectiveResources_ResolveHost_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
  uint64_t* addresses;
  size_t address_count;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_ResolveHost_Args,
                             address_count);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_ResolveHost(
    XLA_FFI_NcclCollectiveResources_ResolveHost_Args* args);

// Begins collective execution by enqueueing the entry synchronization
// requested during Prepare. Work enqueued after this call does not begin until
// every rank in the NCCL LSA reaches the same boundary. This operation is
// valid only during FFI Execute, only when Request set barrier_before_launch,
// and at most once per resource. It does not synchronize multiple LSA teams.
struct XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(
    XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args, resource);

typedef XLA_FFI_Error*
XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch(
    XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args* args);

// Releases the host-side token and resources owned specifically by it. Resolve
// staging remains retained independently until its stream-ordered copy
// completes. Destroy does not synchronize a GPU stream or invalidate XLA's
// underlying buffer allocations. A null resource is not valid.
struct XLA_FFI_NcclCollectiveResources_Destroy_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_NcclCollectiveResource* resource;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Destroy_Args,
                             resource);

typedef void XLA_FFI_NcclCollectiveResources_Destroy(
    XLA_FFI_NcclCollectiveResources_Destroy_Args* args);

// Extension table published through XLA_FFI_Api::extension_start. Minor
// versions preserve this table as a prefix and append new fields at the end.
// Callers must apply the version and struct-size rules documented above and
// check optional callbacks for null. destroy is required for every returned
// resource.
struct XLA_FFI_NcclCollectiveResources_Extension {
  XLA_FFI_Extension_Base extension_base;
  int32_t abi_major_version;
  int32_t abi_minor_version;

  XLA_FFI_NcclCollectiveResources_Request* request;
  XLA_FFI_NcclCollectiveResources_Commit* commit;
  XLA_FFI_NcclCollectiveResources_Initialize* initialize;
  XLA_FFI_NcclCollectiveResources_Resolve* resolve;
  XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch*
      enqueue_barrier_before_launch;
  XLA_FFI_NcclCollectiveResources_Destroy* destroy;
  XLA_FFI_NcclCollectiveResources_QueryTopology* query_topology;
  XLA_FFI_NcclCollectiveResources_ResolveHost* resolve_host;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Extension,
                             resolve_host);

// Minimum table size for ABI 0.1. Later minor versions preserve this prefix.
#define XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_0_1_STRUCT_SIZE \
  XLA_FFI_STRUCT_SIZE(XLA_FFI_NcclCollectiveResources_Extension, resolve_host)

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_C_API_NCCL_COLLECTIVE_RESOURCES_H_
