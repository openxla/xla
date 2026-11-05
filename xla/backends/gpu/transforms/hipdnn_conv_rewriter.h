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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_HIPDNN_CONV_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_HIPDNN_CONV_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Promotes forward convolution custom calls from the MIOpen-backed
// __cudnn$convForward target to the hipDNN-backed __cudnn$convForwardGraph
// target on ROCm.
//
// This runs only when --xla_gpu_enable_hipdnn is set. Without the flag every
// convolution keeps the legacy MIOpen path, which is why the pass is a
// separate opt-in step rather than part of ConvRewriter.
//
// Only plain forward convolutions are promoted. hipDNN 0.4 resolves its
// convolution graphs through MIOpen and has no engine for the fused patterns
// cuDNN handles on this target -- bias add, scaling and amax reduction all
// fail to find an engine configuration -- so promoting them would turn a
// working MIOpen convolution into a compilation failure.
class HipdnnConvRewriter : public HloModulePass {
 public:
  HipdnnConvRewriter() = default;

  absl::string_view name() const override { return "hipdnn-conv-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_HIPDNN_CONV_REWRITER_H_
