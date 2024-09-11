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

#ifndef XLA_SERVICE_GPU_RUNTIME_NVSHMEM_API_H_
#define XLA_SERVICE_GPU_RUNTIME_NVSHMEM_API_H_

#include <functional>
#include <string_view>

#include <cuda.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/service/collective_ops_utils.h"

#include "third_party/nvshmem/nvshmemx.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NvshmemApi
//===----------------------------------------------------------------------===//

class NvshmemApi {
 public:
  enum class TEAMSKIND {
    kWORLD = 0,
    kSHARED = 1,
    kNODE = 2,
    kTOTAL_TEAMS_KIND = 3,
  };

  constexpr static int64_t kMaxNumTeams = 10;
  // Returns a default NvshmemApi for a current process.
  // NvshmemApi follows the Singleton design pattern
  static NvshmemApi& Default();

  static bool IsInitialized();
  static void SetEnvInfo(
      int process_id, size_t num_processes, size_t device_count_per_process,
      std::function<absl::StatusOr<std::string>(std::string_view)> kv_store_get,
      std::function<absl::Status(std::string_view, std::string_view)>
          kv_store_set);
  NvshmemApi(NvshmemApi const&) = delete;
  void operator=(NvshmemApi const&) = delete;
  absl::Status DoAllreduce(TEAMSKIND team_kind, PrimitiveType type,
                           se::DeviceMemoryBase dest,
                           se::DeviceMemoryBase source, int64_t num_elements,
                           se::Stream& stream, ReductionKind reduction_kind);
  absl::StatusOr<void*> Allocate(uint64_t bytes);
  absl::Status Deallocate(void* buffer);

 private:
  NvshmemApi();
  ~NvshmemApi();

  absl::Status Initialize();

  // Env variable
  static int process_id_;
  static size_t num_processes_;
  static size_t device_count_per_process_;
  static std::function<absl::StatusOr<std::string>(std::string_view)>
      kv_store_get_;
  static std::function<absl::Status(std::string_view, std::string_view)>
      kv_store_set_;
  static constexpr char kv_store_key_[] = "nvshmem_global_init";
  std::vector<nvshmemx_team_t> all_teams;

  // Internal state
  static bool initialized_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NVSHMEM_API_H_
