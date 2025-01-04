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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_

#include <functional>
#include <string_view>


#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/core/collectives/collectives.h"

namespace xla::gpu {

// NVIDIA NVSHMEM library
class NvshmemCollectives : public Collectives {
 public:
  NvshmemCollectives* Default();

  absl::Status Initialize(int process_id, size_t num_processes, size_t device_count_per_process,
      std::function<absl::StatusOr<std::string>(std::string_view)> kv_store_get,
      std::function<absl::Status(std::string_view, std::string_view)> kv_store_set);

  void Finalize();

  absl::StatusOr<void*> Allocate(uint64_t bytes);

  absl::Status Deallocate(void* buffer);

 private:
  static constexpr char kv_store_key_[] = "nvshmem_global_init";
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_
