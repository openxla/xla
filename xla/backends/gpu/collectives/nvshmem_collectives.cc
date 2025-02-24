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

#include "xla/backends/gpu/collectives/nvshmem_collectives.h"

#include "absl/strings/str_format.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"
#include "third_party/nvshmem/nvshmem.h"
#include "third_party/nvshmem/nvshmemx.h"
#include "xla/core/collectives/collectives_registry.h"

#include <cuda.h>

namespace xla::gpu {

NvshmemCollectives::~NvshmemCollectives() {
  if (initialized_) Finalize();
}

NvshmemCollectives* NvshmemCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Get("gpu", "nvshmem");
  CHECK_OK(collectives) << "Failed to get NVSHMEM collectives";  // Crash OK

  if (auto* nvshmem_collectives =
          tsl::down_cast<NvshmemCollectives*>(*collectives)) {
    return nvshmem_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for NVSHMEM";
}

void NvshmemCollectives::SetEnvInfo(
    int process_id, size_t num_processes, size_t device_count_per_process,
    std::weak_ptr<KeyValueStoreInterface> kv_store) {
  process_id_ = process_id;
  num_processes_ = num_processes;
  device_count_per_process_ = device_count_per_process;
  kv_store_ = kv_store;
}

absl::Status NvshmemCollectives::Initialize() {
  if (process_id_ == -1) {
    LOG(FATAL) << "NvshmemCollectives::SetEnvInfo was not called before using "
                  "NVSHMEM API";
  }
  if (device_count_per_process_ != 1) {
    LOG(FATAL) << "NVSHMEM API is only supported with one device per process";
  }
  nvshmemx_init_attr_t nvshmem_init_attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_uniqueid_t nvshmem_id = NVSHMEMX_UNIQUEID_INITIALIZER;

  // Initialize NVSHMEM
  if (std::shared_ptr<KeyValueStoreInterface> kv_store = kv_store_.lock()) {
    if (process_id_ == 0) {
      if (nvshmemx_get_uniqueid(&nvshmem_id) != 0) {
        return absl::InternalError("nvshmemx_get_uniqueid failed.");
      }
      absl::string_view nvshmem_id_str(reinterpret_cast<char*>(&nvshmem_id),
                                       sizeof(nvshmemx_uniqueid_t));
      TF_RETURN_IF_ERROR(kv_store->Set(kv_store_key_, nvshmem_id_str));
    } else {
      TF_ASSIGN_OR_RETURN(std::string id_str,
                          kv_store->Get(kv_store_key_, absl::Minutes(10)));
      std::copy(id_str.data(), id_str.data() + sizeof(nvshmemx_uniqueid_t),
                reinterpret_cast<char*>(&nvshmem_id));
    }
  } else {
    return absl::InternalError(
        "KV store is not available for nvshmem initialization.");
  }

  if (nvshmemx_set_attr_uniqueid_args(process_id_, num_processes_, &nvshmem_id,
                                      &nvshmem_init_attr) != 0) {
    return absl::InternalError("nvshmemx_set_attr_uniqueid_args failed.");
  }
  if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID,
                                 &nvshmem_init_attr) != 0) {
    return absl::InternalError("nvshmemx_hostlib_init_attr failed.");
  }

  VLOG(3) << absl::StreamFormat(
      "Initialized NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  return absl::OkStatus();
}

absl::Status NvshmemCollectives::InitializeOnce() {
  static absl::once_flag once_flag;
  absl::Status status = absl::OkStatus();
  absl::call_once(once_flag, [&]() {
    status = Initialize();
    initialized_ = true;
  });
  return status;
}

void NvshmemCollectives::Finalize() {
  VLOG(3) << absl::StreamFormat(
      "Finilizing NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  nvshmemx_hostlib_finalize();
}

absl::StatusOr<void*> NvshmemCollectives::Allocate(uint64_t bytes) {
  TF_RETURN_IF_ERROR(InitializeOnce());
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

absl::Status NvshmemCollectives::Deallocate(void* buffer) {
  TF_RETURN_IF_ERROR(InitializeOnce());
  VLOG(3) << absl::StreamFormat("Start de-allocation for NVSHMEM buffer: %p",
                                buffer);
  nvshmem_free(buffer);
  return absl::OkStatus();
}

}  // namespace xla::gpu

// NvshmemCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("gpu", "nvshmem", -100,
                         std::make_unique<xla::gpu::NvshmemCollectives>());
