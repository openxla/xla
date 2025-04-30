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

#include "xla/backends/gpu/collectives/nvshmem_communicator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "third_party/nvshmem/nvshmemx.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// NVSHMEM Utility Functions
//==-----------------------------------------------------------------------===//

static size_t ToRealCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<nvshmem_team_t> ToNvshmemTeam(CommAffinity kind) {
  switch (kind) {
    case CommAffinity::kNODE:
      return NVSHMEMX_TEAM_NODE;
    case CommAffinity::kSHARED:
      return NVSHMEM_TEAM_SHARED;
    case CommAffinity::kWORLD:
      return NVSHMEM_TEAM_WORLD;
    default:
      return absl::InvalidArgumentError("Invalid Nvshmem team.");
  }
}

static absl::StatusOr<std::string> ToNvshmemTeamString(CommAffinity kind) {
  switch (kind) {
    case CommAffinity::kNODE:
      return "Node";
    case CommAffinity::kSHARED:
      return "Shared";
    case CommAffinity::kWORLD:
      return "World";
    default:
      return absl::InvalidArgumentError("Invalid Nvshmem team.");
  }
}

//==-----------------------------------------------------------------------===//
// NVSHMEM Templated APIs
//==-----------------------------------------------------------------------===//

#define CALL_NVSHMEM_COLL(coll, TYPENAME, TYPE, OP, team, source_ptr,         \
                          dest_ptr, stream)                                   \
  do {                                                                        \
    if (nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                      \
            team, (TYPE*)dest_ptr, (const TYPE*)source_ptr, count, stream) != \
        0) {                                                                  \
      return absl::InternalError("Nvshmem collective failed");                \
    }                                                                         \
  } while (0)

#define NVSHMEM_BITWISE_REDUCTION_BITWISE_DATATYPE(                     \
    coll, TYPENAME, TYPE, team, source_ptr, dest_ptr, count, stream,    \
    reduction_kind)                                                     \
  switch (reduction_kind) {                                             \
    case ReductionKind::SUM:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, sum, team, source_ptr,  \
                        dest_ptr, stream);                              \
      break;                                                            \
    case ReductionKind::MIN:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, min, team, source_ptr,  \
                        dest_ptr, stream);                              \
      break;                                                            \
    case ReductionKind::MAX:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, max, team, source_ptr,  \
                        dest_ptr, stream);                              \
      break;                                                            \
    case ReductionKind::PRODUCT:                                        \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, prod, team, source_ptr, \
                        dest_ptr, stream);                              \
      break;                                                            \
    default:                                                            \
      return absl::InternalError("Invalid NVSHMEM reduction kind.");    \
  }

#define NVSHMEM_REDUCTION_DATATYPE(coll, TYPENAME, TYPE, team, source_ptr, \
                                   dest_ptr, num_elements, gpu_stream,     \
                                   reduction_kind)                         \
  switch (reduction_kind) {                                                \
    case ReductionKind::SUM:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, sum, team, source_ptr,     \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    case ReductionKind::MIN:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, min, team, source_ptr,     \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    case ReductionKind::MAX:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, max, team, source_ptr,     \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    case ReductionKind::PRODUCT:                                           \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, prod, team, source_ptr,    \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    default:                                                               \
      return absl::InternalError("Invalid NVSHMEM reduction kind.");       \
  }

#define CALL_NVSHMEM_REDUCTION_DATATYPE(TYPENAME, TYPE, team, gpu_stream,     \
                                        reduction_kind, dest_ptr, source_ptr, \
                                        count)                                \
  NVSHMEM_REDUCTION_DATATYPE(reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD,      \
                             (TYPE*)source_ptr, (TYPE*)dest_ptr, count,       \
                             gpu_stream, reduction_kind);
#define CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(TYPENAME, TYPE, team,        \
                                                gpu_stream, reduction_kind,  \
                                                dest_ptr, source_ptr, count) \
  NVSHMEM_BITWISE_REDUCTION_BITWISE_DATATYPE(                                \
      reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, (TYPE*)source_ptr,         \
      (TYPE*)dest_ptr, count, gpu_stream, reduction_kind);

//==-----------------------------------------------------------------------===//
// NVSHMEM Communicator
//==-----------------------------------------------------------------------===//

NvshmemCommunicator::NvshmemCommunicator(NvshmemCollectives* collectives,
                                         CommAffinity comm)
    : collectives_(collectives), comm_(comm) {
  VLOG(1) << "Created " << *this;
}

NvshmemCommunicator::~NvshmemCommunicator() {
  if (comm_ == CommAffinity::kINVALID) {
    VLOG(1) << "Skipping destruction; invalid comm_ " << *this;
    return;
  }

  if (aborted_) {
    VLOG(1) << "Skipping destruction; already aborted " << *this;
    return;
  }

  VLOG(1) << "Destroy " << *this;

  auto team = ToNvshmemTeam(comm_);
  // Cannot destroy node or world since they are pre-defined comms.
  if (team.ok() && (team.value() != NVSHMEMX_TEAM_NODE) &&
      (team.value() != NVSHMEM_TEAM_WORLD)) {
    nvshmem_team_destroy(team.value());
  }
}

absl::Status NvshmemCommunicator::Abort() {
  VLOG(1) << "Abort NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("NvshmemCommunicator aborted");
  }
  aborted_ = true;
  // Call nvshmem_global_exit with a non-zero return code
  // to abort the program.
  nvshmem_global_exit(1);
  return absl::OkStatus();
}

absl::Status NvshmemCommunicator::Barrier(
    const Communicator::Executor& executor) {
  VLOG(1) << "Barrier NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("NvshmemCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto gpu_stream = se::gpu::AsGpuStreamValue(stream);
  TF_ASSIGN_OR_RETURN(nvshmemx_team_t team, ToNvshmemTeam(comm_));

  if (nvshmemx_barrier_on_stream(team, gpu_stream) != 0) {
    return absl::InternalError("Nvshmem team barrier failed.");
  }
  return absl::OkStatus();
}
absl::StatusOr<size_t> NvshmemCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }

  int32_t count = 0;
  TF_ASSIGN_OR_RETURN(nvshmem_team_t team, ToNvshmemTeam(comm_));
  count = nvshmem_team_n_pes(team);
  if (count < 0) {
    return absl::InvalidArgumentError(
        "NvshmemCommunicator::NumRanks invalid team.");
  }
  return count;
}

tsl::AsyncValueRef<NvshmemCommunicator::Event> NvshmemCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  TF_ASSIGN_OR_RETURN(nvshmemx_team_t team, ToNvshmemTeam(comm_));
  void* dest_ptr = send_buffer.opaque();
  void* source_ptr = recv_buffer.opaque();
  count = ToRealCount(dtype, count);
  VLOG(3) << absl::StreamFormat(
      "Launch NVSHMEM AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%s; team=%d;"
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind),
      ToNvshmemTeamString(comm_).value(), team, stream);

  switch (dtype) {
    case PrimitiveType::F64: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          double, double, team, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::F16: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          half, __half, team, se::gpu::AsGpuStreamValue(stream), reduction_kind,
          dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::F32: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          float, float, team, se::gpu::AsGpuStreamValue(stream), reduction_kind,
          dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::BF16: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          bfloat16, __nv_bfloat16, team, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::S32: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          int32, int32_t, team, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::S64: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          int64, int64_t, team, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::U32: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          uint32, uint32_t, team, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::U64: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          uint64, uint64_t, team, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    default:
      return absl::InternalError("Invalid Nvshmem reduction type.");
  }
  return OkEvent();
}

std::string NvshmemCommunicator::ToString() const {
  return absl::StrFormat("NvshmemCommunicator(nvshmem_team_t=%d)", comm_);
}

absl::StatusOr<se::Stream*> NvshmemCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

}  // namespace xla::gpu
