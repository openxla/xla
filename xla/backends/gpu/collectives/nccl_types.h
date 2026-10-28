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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_TYPES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_TYPES_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/nccl/nccl.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

struct NcclCommState {
  explicit NcclCommState(ncclComm_t comm) : comm(comm) {}
  ~NcclCommState();

  // Call only after the parent has been successfully aborted or destroyed.
  // Then abort any children produced by interrupted splits and free the memory
  // that held their handles. A child whose abort fails stays in
  // pending_split_comms (its handle is never freed, see the destructor) and the
  // first failure is returned. `abort_fn` exists so tests can fake NCCL.
  absl::Status AbortPendingSplits(
      ncclResult_t (*abort_fn)(ncclComm_t) = ncclCommAbort)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex);

  // Keeps the memory holding child communicator handles alive after a split
  // is interrupted, because NCCL may still write to it.
  // AbortPendingSplits() then cleans up the children and frees this memory.
  std::vector<std::unique_ptr<ncclComm_t>> pending_split_comms
      ABSL_GUARDED_BY(mutex);

  bool has_symmetric_window() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    return num_symmetric_windows > 0;
  }

  ncclComm_t comm ABSL_GUARDED_BY(mutex);

  // Number of active NCCL_WIN_COLL_SYMMETRIC windows registered on `comm`.
  // Neither symmetric kernels (`symk`) nor Copy Engine collectives can activate
  // unless at least one symmetric window is registered on `comm`.
  int num_symmetric_windows ABSL_GUARDED_BY(mutex) = 0;

  // NCCL API restricts concurrent usage with the same communicator.
  // See:
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
  // Therefore all NCCL API calls are guarded by this mutex.
  absl::Mutex mutex;
};

//===----------------------------------------------------------------------===//
// Conversions between XLA and NCCL data types
//===----------------------------------------------------------------------===//

size_t ToNcclCount(PrimitiveType dtype, size_t count);

absl::StatusOr<ncclDataType_t> ToNcclDataType(
    PrimitiveType dtype, bool is_reduction_op,
    stream_executor::CudaComputeCapability cc);

ncclRedOp_t ToNcclReduction(ReductionKind kind);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_TYPES_H_
