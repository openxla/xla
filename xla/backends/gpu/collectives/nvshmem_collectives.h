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
  ~NvshmemCollectives() override;

  static NvshmemCollectives* Default();

  void SetEnvInfo(
      int process_id, size_t num_processes, size_t device_count_per_process,
      std::function<absl::StatusOr<std::string>(std::string_view)> kv_store_get,
      std::function<absl::Status(std::string_view, std::string_view)>
          kv_store_set);

  absl::StatusOr<void*> Allocate(uint64_t bytes);

  absl::Status Deallocate(void* buffer);

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(int32_t, const CliqueKey&, const std::optional<CliqueId>&,
                      absl::Span<const DeviceRank>,
                      const Collectives::Config&) final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const>, int32_t, absl::Span<const RankId>,
      const Collectives::Config&) final {
    return absl::UnimplementedError("Not implemented.");
  }

 private:
  absl::Status Initialize();
  absl::Status InitializeOnce();

  void Finalize();

  int process_id_ = -1;
  size_t num_processes_ = 0;
  size_t device_count_per_process_ = 0;
  std::function<absl::StatusOr<std::string>(std::string_view)> kv_store_get_ =
      nullptr;
  std::function<absl::Status(std::string_view, std::string_view)>
      kv_store_set_ = nullptr;
  bool initialized_ = false;

  static constexpr char kv_store_key_[] = "nvshmem_global_init";
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_
