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

#ifndef XLA_FFI_API_FFI_GPU_COLLECTIVES_H_
#define XLA_FFI_API_FFI_GPU_COLLECTIVES_H_

#include <stddef.h>
#include <stdint.h>
#include "xla/ffi/api/c_api.h"

// XLA:GPU collectives FFI extension.
//
// Exposes the XLA-owned host collective communicator (e.g. `ncclComm_t`) to FFI
// handlers: `request_communicator` requests a clique in the Prepare stage and
// `get_communicator` returns the handle once cliques are acquired. Retrieved
// via `Ctx<Extension<xla::ffi::GpuCollectives>>()`; header-only, no
// registration.

#ifdef __cplusplus
extern "C" {
#endif

// Unique extension id ("gpu_coll" in ASCII).
#define XLA_FFI_Gpu_Collectives_Extension_Type INT64_C(0x6770755f636f6c6c)
#define XLA_FFI_Gpu_Collectives_Extension_Major 1
#define XLA_FFI_Gpu_Collectives_Extension_Minor 0

// Mirrors `xla::CollectiveOpGroupMode`.
typedef enum {
  XLA_FFI_GPU_GROUP_CROSS_REPLICA = 0,
  XLA_FFI_GPU_GROUP_CROSS_PARTITION = 1,
  XLA_FFI_GPU_GROUP_CROSS_REPLICA_AND_PARTITION = 2,
  XLA_FFI_GPU_GROUP_FLATTENED_ID = 3,
} XLA_FFI_Gpu_GroupMode;

typedef struct {
  const int64_t* ids;
  size_t size;
} XLA_FFI_Gpu_ReplicaGroup;

typedef struct XLA_FFI_Gpu_Collectives_Extension
    XLA_FFI_Gpu_Collectives_Extension;

struct XLA_FFI_Gpu_Communicator_Request_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_Gpu_GroupMode group_mode;
  const XLA_FFI_Gpu_ReplicaGroup* groups;
  size_t num_groups;
  int64_t communication_id;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Gpu_Communicator_Request_Args,
                             communication_id);

// Requests the collective clique so it is acquired before execution. Prepare
// stage only.
typedef XLA_FFI_Error* XLA_FFI_Gpu_Communicator_Request(
    const XLA_FFI_Gpu_Collectives_Extension* self,
    XLA_FFI_Gpu_Communicator_Request_Args* args);

struct XLA_FFI_Gpu_Communicator_Get_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_Gpu_GroupMode group_mode;
  const XLA_FFI_Gpu_ReplicaGroup* groups;
  size_t num_groups;
  int64_t communication_id;
  void* communicator;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Gpu_Communicator_Get_Args, communicator);

// Returns the non-owning communicator handle for the clique. Valid once cliques
// are acquired (Initialize/Execute stages).
typedef XLA_FFI_Error* XLA_FFI_Gpu_Communicator_Get(
    const XLA_FFI_Gpu_Collectives_Extension* self,
    XLA_FFI_Gpu_Communicator_Get_Args* args);

struct XLA_FFI_Gpu_Collectives_Extension {
  XLA_FFI_Extension extension_base;

  // Opaque per-invocation backend state, set by the GPU runtime.
  // `collective_clique_requests` is non-null only in Prepare;
  // `collective_cliques` only once cliques are acquired.
  void* collective_params;
  void* collective_clique_requests;
  void* collective_cliques;

  XLA_FFI_Gpu_Communicator_Request* request_communicator;
  XLA_FFI_Gpu_Communicator_Get* get_communicator;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Gpu_Collectives_Extension,
                             get_communicator);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
namespace xla::ffi {

// Traits for `Ctx<Extension<GpuCollectives>>()`; yields the extension pointer.
struct GpuCollectives {
  using CExtension = XLA_FFI_Gpu_Collectives_Extension;
  using Type = const XLA_FFI_Gpu_Collectives_Extension*;

  static constexpr auto kName = "GpuCollectives";
  static constexpr int64_t kExtensionType =
      XLA_FFI_Gpu_Collectives_Extension_Type;
  static constexpr int32_t kMajorVersion =
      XLA_FFI_Gpu_Collectives_Extension_Major;
  static constexpr int32_t kMinorVersion =
      XLA_FFI_Gpu_Collectives_Extension_Minor;

  static Type Create(const CExtension* ext) { return ext; }
  static bool Support(int32_t major_version, int32_t minor_version) {
    return major_version == kMajorVersion && minor_version >= kMinorVersion;
  }
};

}  // namespace xla::ffi
#endif  // __cplusplus

#endif  // XLA_FFI_API_FFI_GPU_COLLECTIVES_H_
