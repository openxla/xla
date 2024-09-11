/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/nvshmem_api.h"

#include "absl/strings/str_format.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"
#include "third_party/nvshmem/nvshmem.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// Macros to return or warn on NVSHMEM errors.
//==-----------------------------------------------------------------------===//

static absl::Status NvshmemToStatus(int s, const char* file, int64_t line,
                                    const char* expr) {
  if (s == 0) return absl::OkStatus();

  return absl::InternalError(
      absl::StrFormat("%s:%d: NVSHMEM operation %s failed."
                      " For extra logging, rerun with 'NVSHMEM_DEBUG=INFO'.",
                      file, line, expr));
}

#define XLA_NVSHMEM_STATUS(expr) \
  xla::gpu::NvshmemToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_NVSHMEM_RETURN_IF_ERROR(expr)      \
  do {                                         \
    absl::Status s = XLA_NVSHMEM_STATUS(expr); \
    if (!s.ok()) {                             \
      return s;                                \
    }                                          \
  } while (0)

#define XLA_NVSHMEM_LOG_IF_ERROR(expr)         \
  do {                                         \
    absl::Status s = XLA_NVSHMEM_STATUS(expr); \
    if (!s.ok()) {                             \
      LOG(ERROR) << s.ToString();              \
    }                                          \
  } while (0)

#define XLA_NVSHMEM_CHECK(expr) CHECK(XLA_NVSHMEM_STATUS(expr).ok())

#define CALL_NVSHMEM_COLL(coll, TYPENAME, TYPE, OP, team, source_ptr,     \
                          dest_ptr, stream)                               \
  do {                                                                    \
    XLA_NVSHMEM_RETURN_IF_ERROR(                                          \
        nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                  \
            team, (TYPE*)dest_ptr, (const TYPE*)source_ptr, num_elements, \
            stream));                                                     \
  } while (0)

#define NVSHMEM_BITWISE_REDUCTION_BITWISE_DATATYPE(                         \
    coll, TYPENAME, TYPE, team, source_ptr, dest_ptr, num_elements, stream, \
    reduction_kind)                                                         \
  switch (reduction_kind) {                                                 \
    case ReductionKind::SUM:                                                \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, sum, team, source_ptr,      \
                        dest_ptr, stream);                                  \
      break;                                                                \
    case ReductionKind::MIN:                                                \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, min, team, source_ptr,      \
                        dest_ptr, stream);                                  \
      break;                                                                \
    case ReductionKind::MAX:                                                \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, max, team, source_ptr,      \
                        dest_ptr, stream);                                  \
      break;                                                                \
    case ReductionKind::PRODUCT:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, prod, team, source_ptr,     \
                        dest_ptr, stream);                                  \
      break;                                                                \
    default:                                                                \
      return absl::InternalError("Invalid NVSHMEM reduction kind.");        \
  }

#define NVSHMEM_REDUCTION_DATATYPE(coll, TYPENAME, TYPE, team, source_ptr, \
                                   dest_ptr, num_elements, stream,         \
                                   reduction_kind)                         \
  switch (reduction_kind) {                                                \
    case ReductionKind::SUM:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, sum, team, source_ptr,     \
                        dest_ptr, stream);                                 \
      break;                                                               \
    case ReductionKind::MIN:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, min, team, source_ptr,     \
                        dest_ptr, stream);                                 \
      break;                                                               \
    case ReductionKind::MAX:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, max, team, source_ptr,     \
                        dest_ptr, stream);                                 \
      break;                                                               \
    case ReductionKind::PRODUCT:                                           \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, prod, team, source_ptr,    \
                        dest_ptr, stream);                                 \
      break;                                                               \
    default:                                                               \
      return absl::InternalError("Invalid NVSHMEM reduction kind.");       \
  }

#define CALL_NVSHMEM_REDUCTION_DATATYPE(TYPENAME, TYPE, team, stream,          \
                                        reduction_kind, dest_ptr, source_ptr,  \
                                        num_elements)                          \
  NVSHMEM_REDUCTION_DATATYPE(reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD,       \
                             (TYPE*)source_ptr, (TYPE*)dest_ptr, num_elements, \
                             stream, reduction_kind);
#define CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(TYPENAME, TYPE, team, stream, \
                                                reduction_kind, dest_ptr,     \
                                                source_ptr, num_elements)     \
  NVSHMEM_BITWISE_REDUCTION_BITWISE_DATATYPE(                                 \
      reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, (TYPE*)source_ptr,          \
      (TYPE*)dest_ptr, num_elements, stream, reduction_kind);

int NvshmemApi::process_id_ = -1;
size_t NvshmemApi::num_processes_ = 0;
size_t NvshmemApi::device_count_per_process_ = 0;
std::function<absl::StatusOr<std::string>(std::string_view)>
    NvshmemApi::kv_store_get_ = nullptr;
std::function<absl::Status(std::string_view, std::string_view)>
    NvshmemApi::kv_store_set_ = nullptr;
bool NvshmemApi::initialized_ = false;

NvshmemApi& NvshmemApi::Default() {
  static NvshmemApi instance;
  return instance;
}

void NvshmemApi::SetEnvInfo(
    int process_id, size_t num_processes, size_t device_count_per_process,
    std::function<absl::StatusOr<std::string>(std::string_view)> kv_store_get,
    std::function<absl::Status(std::string_view, std::string_view)>
        kv_store_set) {
  process_id_ = process_id;
  num_processes_ = num_processes;
  device_count_per_process_ = device_count_per_process;
  kv_store_get_ = kv_store_get;
  kv_store_set_ = kv_store_set;
}

NvshmemApi::NvshmemApi() {
  // Initialize NVSHMEM here since code path
  // is already protected by singleton pattern
  if (process_id_ == -1) {
    LOG(FATAL)
        << "NvshmemApi::SetEnvInfo was not called before using NVSHMEM API";
  }
  if (device_count_per_process_ != 1) {
    LOG(FATAL) << "NVSHMEM API is only supported with one device per process";
  }
  CHECK(Initialize().ok());
}

NvshmemApi::~NvshmemApi() {
  VLOG(3) << absl::StreamFormat(
      "Finilizing NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  nvshmemx_hostlib_finalize();
}
bool NvshmemApi::IsInitialized() { return initialized_; }
absl::Status NvshmemApi::Initialize() {
  if (initialized_) {
    return absl::OkStatus();
  }
  nvshmemx_init_attr_t nvshmem_init_attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_uniqueid_t nvshmem_id = NVSHMEMX_UNIQUEID_INITIALIZER;

  // Initialize NVSHMEM
  if (process_id_ == 0) {
    XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_get_uniqueid(&nvshmem_id));
    std::string_view nvshmem_id_str(reinterpret_cast<char*>(&nvshmem_id),
                                    sizeof(nvshmemx_uniqueid_t));
    TF_RETURN_IF_ERROR(kv_store_set_(kv_store_key_, nvshmem_id_str));
  } else {
    TF_ASSIGN_OR_RETURN(std::string id_str, kv_store_get_(kv_store_key_));
    std::copy(id_str.data(), id_str.data() + sizeof(nvshmemx_uniqueid_t),
              reinterpret_cast<char*>(&nvshmem_id));
  }
  XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_set_attr_uniqueid_args(
      process_id_, num_processes_, &nvshmem_id, &nvshmem_init_attr));
  XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_hostlib_init_attr(
      NVSHMEMX_INIT_WITH_UNIQUEID, &nvshmem_init_attr));

  VLOG(3) << absl::StreamFormat(
      "Initialized NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  all_teams.resize((int64_t)NvshmemApi::TEAMSKIND::kTOTAL_TEAMS_KIND);
  all_teams[(int64_t)NvshmemApi::TEAMSKIND::kWORLD] = NVSHMEM_TEAM_WORLD;
  all_teams[(int64_t)NvshmemApi::TEAMSKIND::kSHARED] = NVSHMEM_TEAM_SHARED;
  all_teams[(int64_t)NvshmemApi::TEAMSKIND::kNODE] = NVSHMEMX_TEAM_NODE;

  initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<void*> NvshmemApi::Allocate(uint64_t bytes) {
  VLOG(3) << absl::StreamFormat(
      "Start allocation of %s (%llu bytes) for NVSHMEM",
      tsl::strings::HumanReadableNumBytes(bytes), bytes);
  void* buffer = nvshmem_malloc(bytes);
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from NVSHMEM memory",
        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  return buffer;
}

absl::Status NvshmemApi::Deallocate(void* buffer) {
  VLOG(3) << absl::StreamFormat("Start de-allocation for NVSHMEM buffer: %p",
                                buffer);
  nvshmem_free(buffer);
  return absl::OkStatus();
}

absl::Status NvshmemApi::DoAllreduce(NvshmemApi::TEAMSKIND team_kind,
                                     PrimitiveType type,
                                     se::DeviceMemoryBase dest,
                                     se::DeviceMemoryBase source,
                                     int64_t num_elements, se::Stream& stream,
                                     ReductionKind reduction_kind) {
  // we only allow intra-node reduction with nvshmem for now.
  CHECK(team_kind == NvshmemApi::TEAMSKIND::kNODE);
  // nvshmemx_barrier_all_on_stream(stream);
  auto gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  nvshmemx_team_t& team = all_teams[(int64_t)team_kind];

  XLA_NVSHMEM_RETURN_IF_ERROR(nvshmemx_barrier_on_stream(team, gpu_stream));

  void* dest_ptr = dest.opaque();
  void* source_ptr = source.opaque();

  switch (type) {
    case PrimitiveType::F64: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(double, double, team, gpu_stream,
                                      reduction_kind, dest_ptr, source_ptr,
                                      num_elements);
      break;
    }
    case PrimitiveType::F16: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(half, __half, team, gpu_stream,
                                      reduction_kind, dest_ptr, source_ptr,
                                      num_elements);
      break;
    }
    case PrimitiveType::F32: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(float, float, team, gpu_stream,
                                      reduction_kind, dest_ptr, source_ptr,
                                      num_elements);
      break;
    }
    case PrimitiveType::BF16: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(bfloat16, __nv_bfloat16, team, gpu_stream,
                                      reduction_kind, dest_ptr, source_ptr,
                                      num_elements);
      break;
    }
    case PrimitiveType::S32: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(int32, int32_t, team, gpu_stream,
                                              reduction_kind, dest_ptr,
                                              source_ptr, num_elements);
      break;
    }
    case PrimitiveType::S64: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(int64, int64_t, team, gpu_stream,
                                              reduction_kind, dest_ptr,
                                              source_ptr, num_elements);
      break;
    }
    case PrimitiveType::U32: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          uint32, uint32_t, team, gpu_stream, reduction_kind, dest_ptr,
          source_ptr, num_elements);
      break;
    }
    case PrimitiveType::U64: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          uint64, uint64_t, team, gpu_stream, reduction_kind, dest_ptr,
          source_ptr, num_elements);
      break;
    }
    default:
      return absl::InternalError("Invalid NVShmem reduction type.");
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
