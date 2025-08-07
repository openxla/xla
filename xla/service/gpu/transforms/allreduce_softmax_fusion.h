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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_ALLREDUCE_SOFTMAX_FUSION_H_
#define XLA_SERVICE_GPU_TRANSFORMS_ALLREDUCE_SOFTMAX_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// HLO pass that fuses AllReduce operations with Triton Softmax kernels.
// This pass runs after triton-softmax-rewriter and identifies patterns where
// AllReduce → TritonSoftmax can be combined into a single fused kernel.
//
// The fusion reduces kernel launch overhead and memory bandwidth by:
// 1. Eliminating intermediate memory storage between AllReduce and Softmax
// 2. Reducing the number of kernel launches
// 3. Improving cache locality within the fused kernel
class AllReduceSoftmaxFusion : public HloModulePass {
 public:
  AllReduceSoftmaxFusion() {}

  absl::string_view name() const override { return "allreduce-softmax-fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_ALLREDUCE_SOFTMAX_FUSION_H_ 