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

#ifndef XLA_FFI_NCCL_COLLECTIVE_RESOURCES_API_H_
#define XLA_FFI_NCCL_COLLECTIVE_RESOURCES_API_H_

#include "absl/status/status.h"
#include "xla/ffi/api/c_api_nccl_collective_resources.h"

namespace xla::ffi {

class NcclCollectiveResourceHandle {
 public:
  virtual ~NcclCollectiveResourceHandle() = default;
};

class NcclCollectiveResourcesApi {
 public:
  virtual ~NcclCollectiveResourcesApi() = default;

  virtual absl::Status Request(
      XLA_FFI_NcclCollectiveResources_Request_Args* args) = 0;
  virtual absl::Status Commit(
      XLA_FFI_NcclCollectiveResources_Commit_Args* args) = 0;
  virtual absl::Status Initialize(
      XLA_FFI_NcclCollectiveResources_Initialize_Args* args) = 0;
  virtual absl::Status Resolve(
      XLA_FFI_NcclCollectiveResources_Resolve_Args* args) = 0;
  virtual absl::Status QueryTopology(
      XLA_FFI_NcclCollectiveResources_QueryTopology_Args* args) = 0;
  virtual absl::Status BeginCollective(
      XLA_FFI_NcclCollectiveResources_BeginCollective_Args* args) = 0;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_NCCL_COLLECTIVE_RESOURCES_API_H_
