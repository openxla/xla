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
// A handler uses the extension in the following FFI stages:
//
//   Prepare:    Request, then Commit.
//   Initialize: Initialize, then Resolve.
//   Execute:    EnqueuePrefixBarrier, if requested.
//   Any stage:  Destroy after all device work using the resource has finished.
//
// The extension is versioned independently from the root XLA FFI API. Existing
// structs are frozen. Minor versions may append callbacks to the extension
// table or add new structs, but must not change an existing field's meaning.
#define XLA_FFI_NCCL_COLLECTIVE_RESOURCES_API_MAJOR 0
#define XLA_FFI_NCCL_COLLECTIVE_RESOURCES_API_MINOR 1

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

  // Number of devices expected in the current device's derived clique. Request
  // rejects a mismatch with XLA's device assignment.
  int32_t expected_clique_size;

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
  // resolution fails unless every rank is directly accessible from the current
  // device, even when the underlying symmetric registration spans hosts.
  XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC = 0,

  // Resolves an LSA multimem alias for hardware multicast operations. The same
  // alias is repeated for every rank in the region's address-table row. All
  // clique ranks must belong to one multicast-capable load/store-accessible
  // team; this is not a network-spanning address.
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
// Destroy does not synchronize the GPU stream. The caller must keep `resource`
// alive until all device work that uses its resolved addresses or prefix
// barrier has completed.
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
  // supported.
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

// Describes the load/store-accessible topology of an acquired clique. LSA
// teams are determined by the collective backend from direct GPU load/store
// reachability, not process or host boundaries. For example, an IMEX-backed
// multi-node NVLink partition can form one LSA team across physical hosts.
struct XLA_FFI_NcclCollectiveTopology {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  // Number of ranks in the complete clique.
  int32_t clique_size;
  // Number of ranks directly load/store accessible from the current rank,
  // including the current rank.
  int32_t lsa_size;
  // Number of LSA teams partitioning the complete clique.
  int32_t lsa_team_count;
  // Nonzero exactly when every clique rank belongs to one LSA team.
  uint8_t world_is_lsa;
  // Nonzero when the backend reports hardware multicast support for the
  // current LSA team. Individual multimem address resolution can still fail
  // if its symmetric window is ineligible.
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

// Writes the region-major address table resolved during Initialize. The table
// contains region_count * clique_size entries. Entry
// addresses[region * clique_size + rank] describes that region for that rank.
// For MULTIMEM regions, every entry in a row contains the same multicast alias.
// address_count must equal the complete table size; zero regions permit a null
// addresses pointer. This operation is valid only during FFI Initialize.
struct XLA_FFI_NcclCollectiveResources_Resolve_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
  uint64_t* addresses;
  size_t address_count;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Resolve_Args,
                             address_count);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_Resolve(
    XLA_FFI_NcclCollectiveResources_Resolve_Args* args);

// Enqueues the requested prefix barrier on the FFI execution stream. Work
// enqueued after this call does not begin until every clique rank reaches its
// corresponding barrier. This operation is valid only during FFI Execute, only
// when Request set barrier_before_launch, and at most once per resource. The
// barrier uses direct peer stores and therefore does not span LSA teams. It
// can span physical hosts when cross-host LSA is provided by an NVLink fabric.
struct XLA_FFI_NcclCollectiveResources_EnqueuePrefixBarrier_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_NcclCollectiveResource* resource;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(
    XLA_FFI_NcclCollectiveResources_EnqueuePrefixBarrier_Args, resource);

typedef XLA_FFI_Error* XLA_FFI_NcclCollectiveResources_EnqueuePrefixBarrier(
    XLA_FFI_NcclCollectiveResources_EnqueuePrefixBarrier_Args* args);

// Releases the host-side token and resources owned specifically by it. This
// does not synchronize a GPU stream or invalidate XLA's underlying buffer
// allocations. A null resource is not valid.
struct XLA_FFI_NcclCollectiveResources_Destroy_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_NcclCollectiveResource* resource;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Destroy_Args,
                             resource);

typedef void XLA_FFI_NcclCollectiveResources_Destroy(
    XLA_FFI_NcclCollectiveResources_Destroy_Args* args);

// Extension table published through XLA_FFI_Api::extension_start. Callers must
// validate the version and struct size, and must check an optional callback for
// null before calling it. destroy is required for every returned resource.
struct XLA_FFI_NcclCollectiveResources_Extension {
  XLA_FFI_Extension_Base extension_base;
  int32_t api_major_version;
  int32_t api_minor_version;

  XLA_FFI_NcclCollectiveResources_Request* request;
  XLA_FFI_NcclCollectiveResources_Commit* commit;
  XLA_FFI_NcclCollectiveResources_Initialize* initialize;
  XLA_FFI_NcclCollectiveResources_Resolve* resolve;
  XLA_FFI_NcclCollectiveResources_EnqueuePrefixBarrier* enqueue_prefix_barrier;
  XLA_FFI_NcclCollectiveResources_Destroy* destroy;
  // Added in API minor version 1. Appended to preserve existing field offsets.
  XLA_FFI_NcclCollectiveResources_QueryTopology* query_topology;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_NcclCollectiveResources_Extension,
                             query_topology);

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_C_API_NCCL_COLLECTIVE_RESOURCES_H_
