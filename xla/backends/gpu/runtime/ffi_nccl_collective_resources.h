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

#ifndef XLA_BACKENDS_GPU_RUNTIME_FFI_NCCL_COLLECTIVE_RESOURCES_H_
#define XLA_BACKENDS_GPU_RUNTIME_FFI_NCCL_COLLECTIVE_RESOURCES_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/nccl_collective_resources_api.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace xla::gpu {

class BufferAllocations;
class CollectiveCliqueRequests;
class CollectiveCliques;
class CollectiveMemory;
class CollectiveMemoryRequests;
struct CollectiveParams;

// Execution-scoped adapter between the public FFI resource contract and XLA's
// NCCL lifecycle.
class FfiNcclCollectiveResources final
    : public ffi::NcclCollectiveResourcesApi {
 public:
  FfiNcclCollectiveResources();
  ~FfiNcclCollectiveResources() override;

  FfiNcclCollectiveResources(const FfiNcclCollectiveResources&) = delete;
  FfiNcclCollectiveResources& operator=(const FfiNcclCollectiveResources&) =
      delete;

  absl::Status BeginInvocation(
      XLA_FFI_ExecutionStage stage, stream_executor::Stream* stream,
      const BufferAllocations* buffer_allocations,
      const CollectiveParams* collective_params,
      CollectiveCliqueRequests* collective_clique_requests,
      CollectiveMemoryRequests* collective_memory_requests,
      CollectiveCliques* collective_cliques,
      const CollectiveMemory* collective_memory);

  absl::Status Request(
      XLA_FFI_NcclCollectiveResources_Request_Args* args) override;
  absl::Status Commit(
      XLA_FFI_NcclCollectiveResources_Commit_Args* args) override;
  absl::Status Initialize(
      XLA_FFI_NcclCollectiveResources_Initialize_Args* args) override;
  absl::Status Resolve(
      XLA_FFI_NcclCollectiveResources_Resolve_Args* args) override;
  absl::Status ResolveHost(
      XLA_FFI_NcclCollectiveResources_ResolveHost_Args* args) override;
  absl::Status QueryTopology(
      XLA_FFI_NcclCollectiveResources_QueryTopology_Args* args) override;
  absl::Status EnqueueBarrierBeforeLaunch(
      XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args* args)
      override;

 private:
  class Resource;
  struct State;

  absl::StatusOr<size_t> ValidateAddressResolution(Resource* resource);
  absl::StatusOr<std::vector<uint64_t>> BuildResolvedAddresses(
      Resource* resource, size_t expected_count);

  std::unique_ptr<State> state_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_FFI_NCCL_COLLECTIVE_RESOURCES_H_
