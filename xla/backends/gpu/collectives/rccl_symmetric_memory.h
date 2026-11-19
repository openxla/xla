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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/rccl/rccl.h"
#include "rocm/rocm_config.h"  // IWYU pragma: keep  (defines TF_ROCM_VERSION)
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

// An RCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration. Analogous to
// NcclSymmetricMemory for the ROCm/RCCL platform.
class RcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~RcclSymmetricMemory() final;

  // `executor` is the communicator's single-threaded executor (may be null for
  // blocking/synchronous communicators). It is used to serialize the
  // window-lifecycle operations that mutate communicator state - symmetric
  // window registration (in Create) and deregistration (in the destructor) -
  // onto the communicator's owning thread.
  //
  // The read-only device-pointer lookups (peer_addr(), i.e.
  // ncclGetPeerDevicePointer) are NOT dispatched onto `executor`: they only
  // read already-populated window/comm metadata and do not touch RCCL's
  // per-thread enqueue/group state.
  //
  // `symmetric_window` selects the window registration flags: when true the
  // window is registered with NCCL_WIN_COLL_SYMMETRIC.
  static absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>> Create(
      ncclComm_t comm, stream_executor::DeviceAddressBase addr,
      std::shared_ptr<tsl::Executor> executor, bool symmetric_window = false);

  stream_executor::DeviceAddressBase addr() const final;

  // multimem_addr() is deliberately not overridden: no AMD GPU provides
  // multicast memory, so RCCL always reports comm->nvlsSupport == 0. The base
  // class returns Unimplemented.

  // For RCCL versions that expose ncclGetPeerDevicePointer (ROCm 10.0+)
  // returns the address of the symmetric memory on a peer device.
  absl::StatusOr<stream_executor::DeviceAddressBase> peer_addr(
      RankId peer) const final;

  ncclWindow_t win() const { return win_; }

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  RcclSymmetricMemory(ncclComm_t comm, ncclWindow_t win,
                      stream_executor::DeviceAddressBase addr,
                      std::shared_ptr<tsl::Executor> executor);

  ncclComm_t comm_;
  ncclWindow_t win_;
  stream_executor::DeviceAddressBase addr_;
  std::shared_ptr<tsl::Executor> executor_;
};

namespace internal {

// Returns OK when `nccl_cumem_enable` (the raw NCCL_CUMEM_ENABLE value, or
// nullptr when unset) enables RCCL's cuMem allocator, and therefore symmetric
// window support. Exposed for testing; Create() reads the environment once.
absl::Status NcclCuMemEnabledStatus(const char* nccl_cumem_enable);

}  // namespace internal

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
