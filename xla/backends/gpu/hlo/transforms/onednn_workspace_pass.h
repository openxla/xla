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

#ifndef XLA_BACKENDS_GPU_HLO_TRANSFORMS_ONEDNN_WORKSPACE_PASS_H_
#define XLA_BACKENDS_GPU_HLO_TRANSFORMS_ONEDNN_WORKSPACE_PASS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Resizes the workspace of cuBLASLt matmul and DNN convolution custom calls to
// what the oneDNN primitive behind them needs at run time. For matmul that is
// the scratchpad; for convolution it is the scratchpad plus the pre-packed
// filter. Added only by IntelGpuCompiler, after layout assignment. The sizes
// are computed for the device with the given ordinal.
class OneDnnWorkspacePass : public HloModulePass {
 public:
  OneDnnWorkspacePass(se::GpuComputeCapability gpu_version, int device_ordinal)
      : gpu_version_(gpu_version), device_ordinal_(device_ordinal) {}

  absl::string_view name() const override { return "onednn-workspace"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
  int device_ordinal_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_HLO_TRANSFORMS_ONEDNN_WORKSPACE_PASS_H_
