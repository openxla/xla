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

#include "xla/stream_executor/collectives.h"

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif

namespace stream_executor::gpu {

/* static */ absl::StatusOr<void*> Collectives::CollectiveMemoryAllocate(
    StreamExecutor* executor, uint64_t bytes) {
  if (bytes == 0) return nullptr;

  std::unique_ptr<ActivateContext> activation = executor->Activate();

  void* ptr = nullptr;
// TODO: remove when rocm v6.1 is not supported anymore
// See comment below (ncclMemFree)
#if defined(ncclMemAlloc)
  ncclResult_t res = ncclMemAlloc(&ptr, bytes);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to allocate %s (%llu bytes) from device collective memory: %s, "
        "Last NCCL warning(error) log entry (may be unrelated): %s",
        tsl::strings::HumanReadableNumBytes(bytes), bytes,
        ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }
#elif TENSORFLOW_USE_ROCM
  hipError_t res = wrap::hipMalloc(&ptr, bytes);
  if (res != hipSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to allocate %s (%llu bytes) from device collective memory: %s, "
        "Last NCCL warning(error)",
        tsl::strings::HumanReadableNumBytes(bytes), bytes,
        hipGetErrorString(res)));
  }
#endif
  VLOG(2) << "Allocated collective memory " << ptr << " for executor "
          << executor << " of " << bytes << " bytes";
  return ptr;
}

/* static */ absl::Status Collectives::CollectiveMemoryDeallocate(
    StreamExecutor* executor, void* location) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

// TODO: remove when rocm v6.1 is not supported anymore
// In case there is ncclMemAlloc defined use
// then use it, otherwise we might be running
// with the rocm version < 6.2 where ncclMemFree
// is not supported, so fall back to wrap::hipFree
#if defined(ncclMemFree)
  ncclResult_t res = ncclMemFree(location);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to free device collective memory at %p; result: %s, Last NCCL "
        "warning(error) log entry (may be unrelated): %s",
        location, ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }
#elif TENSORFLOW_USE_ROCM
  hipError_t res = wrap::hipFree(location);
  if (res != hipSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to free device collective memory at %p; result: %s, Last NCCL "
        "warning(error)",
        location, hipGetErrorString(res)));
  }
#endif

  VLOG(2) << "Deallocated collective memory " << location << " for executor "
          << executor;
  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
