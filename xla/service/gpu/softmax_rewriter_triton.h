/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_SOFTMAX_REWRITER_TRITON_H_
#define XLA_SERVICE_GPU_SOFTMAX_REWRITER_TRITON_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/gpu_types.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrite compatible Softmax into a custom fusion region to be code-generated
// with the Triton-based Softmax emitter.
class SoftmaxRewriterTriton : public HloModulePass {
 public:
  explicit SoftmaxRewriterTriton(GpuVersion gpu_version)
      : gpu_version_(gpu_version) {}
  absl::string_view name() const override { return "triton-softmax-rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  GpuVersion gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_SOFTMAX_REWRITER_TRITON_H_
