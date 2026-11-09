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

#include "xla/backends/gpu/collectives/nccl_types.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/nccl/nccl.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

NcclCommState::~NcclCommState() {
  // After successful parent teardown, abort pending children and free their
  // output storage.
  if (parent_torn_down) {
    if (absl::Status s = AbortPendingSplits(abort_fn_); !s.ok()) {
      LOG(ERROR) << s;
    }
  }
  // If parent teardown failed, leave output storage unread and allocated:
  // NCCL may still write to it. Also leak storage for children whose abort
  // failed.
  if (!pending_split_comms.empty()) {
    LOG(WARNING) << "Leaking " << pending_split_comms.size()
                 << " NCCL split output(s): parent teardown did not complete "
                    "or a child abort failed";
  }
  for (auto& output : pending_split_comms) {
    (void)output.release();
  }
}

absl::Status NcclCommState::AbortPendingSplits(
    ncclResult_t (*abort_fn)(ncclComm_t)) {
  absl::Status first_error;
  std::vector<std::unique_ptr<ncclComm_t>> failed;
  for (auto& output : pending_split_comms) {
    // A null handle means NCCL never published a child, nothing to abort.
    if (*output == nullptr) {
      continue;
    }
    ncclResult_t result = abort_fn(*output);
    if (result != ncclSuccess) {
      first_error.Update(
          Internal("Failed to abort interrupted NCCL split child: %s",
                   ncclGetErrorString(result)));
      // Keep the handle. The destructor leaks it on purpose, as for a failed
      // parent teardown.
      failed.push_back(std::move(output));
    }
  }
  pending_split_comms = std::move(failed);
  return first_error;
}

size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

absl::StatusOr<ncclDataType_t> ToNcclDataType(
    PrimitiveType dtype, bool is_reduction_op,
    stream_executor::CudaComputeCapability cc) {
  switch (dtype) {
    case S8:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return ncclInt8;
    // For pre-Hopper FP8 reductions, let NCCL throw appropriate errors.
    case F8E5M2:
      return (cc.IsAtLeastHopper() || is_reduction_op) ? ncclFloat8e5m2
                                                       : ncclInt8;
    case F8E4M3FN:
      return (cc.IsAtLeastHopper() || is_reduction_op) ? ncclFloat8e4m3
                                                       : ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to
      // 32-bit.
      if (is_reduction_op) {
        return InvalidArgument(
            "Unsupported data type for reduction operation: %s",
            primitive_util::LowercasePrimitiveTypeName(dtype));
      }
      // For collectives that just move data around, we can use ncclFloat16
      // for 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return InvalidArgument("Unsupported data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

ncclRedOp_t ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

}  // namespace xla::gpu
