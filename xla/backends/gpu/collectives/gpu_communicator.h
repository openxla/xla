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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_

#include <cstddef>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/util.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

// GpuCommunicator extends Communicator with synchronous versions of the
// collective methods.
//
// For example, the Communicator::AllReduce method (which is asynchronous and
// returns an AsyncValueRef<Event>) has a corresponding
// GpuCommunicator::LaunchAllReduce method (which is synchronous and returns an
// absl::Status).
class GpuCommunicator : public Communicator {
 public:
  ~GpuCommunicator() override = default;

  // Platform-specific handle to the underlying communicator implementation. It
  // allows exporting collective communication primitives created and owned by
  // the XLA runtime to external libraries, for example via FFI calls.
  struct PlatformCommunicatorHandle {
    void* handle = nullptr;  // will be nullptr if not supported
  };

  // Returns a platform-spcific handle to the unerdlying communicator object for
  // host initiated collectives.
  virtual PlatformCommunicatorHandle platform_host_comm() const {
    return PlatformCommunicatorHandle{nullptr};
  }

  // Returns a platform-spcific handle to the unerdlying communicator object for
  // device initiated collectives.
  virtual PlatformCommunicatorHandle platform_device_comm() const {
    return PlatformCommunicatorHandle{nullptr};
  }

  // Creates a symmetric memory from the existing device address range. This is
  // a collective operation, and all ranks in a clique must call this operation
  // in order to make a progress.
  virtual absl::StatusOr<std::unique_ptr<SymmetricMemory>>
  CreateSymmetricMemory(se::DeviceAddressBase addr) {
    return Unimplemented("Symmetric memory is not implemented");
  }

  // Executes f in a group. f should invoke synchronous collective methods like
  // LaunchAllReduce and not asynchronous collective methods like AllReduce.
  virtual Future<> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) = 0;

  virtual absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                           se::DeviceAddressBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) = 0;

  virtual absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) = 0;

  virtual absl::Status LaunchCollectivePermute(
      se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) = 0;

  virtual absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;

  virtual absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
