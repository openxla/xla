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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_CUDNN_NON_GEMM_FUSION_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CUDNN_NON_GEMM_FUSION_REWRITER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// This class marks cuDNN-supported non-GEMM fusions as kCuDnnFusionKind.
class CudnnNonGemmFusionRewriter : public HloModulePass {
 public:
  // stream_exec may be null for deviceless compilation (requires cuDNN >= 9.8).
  explicit CudnnNonGemmFusionRewriter(
      se::StreamExecutor* stream_exec,
      const se::DeviceDescription& gpu_device_info)
      : stream_exec_(stream_exec), gpu_device_info_(gpu_device_info) {}

  absl::string_view name() const override {
    return "cudnn-non-gemm-fusion-rewriter";
  }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::StreamExecutor* stream_exec_;
  const se::DeviceDescription& gpu_device_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CUDNN_NON_GEMM_FUSION_REWRITER_H_
