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

#ifndef XLA_BACKENDS_GPU_HLO_TRANSFORMS_ESTIMATE_CUB_SORT_SCRATCH_SIZE_H_
#define XLA_BACKENDS_GPU_HLO_TRANSFORMS_ESTIMATE_CUB_SORT_SCRATCH_SIZE_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Updates the scratch size of CUB sort custom calls to match the actual
// scratch size and changes the custom call target to the execute target
// (xla.gpu.ext.cub_sort_keys or xla.gpu.ext.cub_sort_pairs).
//
// On CUDA, scratch size is computed purely in C++ from the target compute
// capability (supporting both live-GPU and deviceless compilation). On other
// GPU platforms (ROCm, SYCL), invokes the platform FFI instantiate handler.
class EstimateCubSortScratchSize : public HloModulePass {
 public:
  explicit EstimateCubSortScratchSize(
      std::string platform_name,
      stream_executor::GpuComputeCapability gpu_compute_capability)
      : platform_name_(std::move(platform_name)),
        gpu_compute_capability_(std::move(gpu_compute_capability)) {}

  absl::string_view name() const override {
    return "estimate-cub-sort-scratch-size";
  }

 protected:
  absl::Status RunOnSortInstruction(HloCustomCallInstruction* custom_call);
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::string platform_name_;
  stream_executor::GpuComputeCapability gpu_compute_capability_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_HLO_TRANSFORMS_ESTIMATE_CUB_SORT_SCRATCH_SIZE_H_
