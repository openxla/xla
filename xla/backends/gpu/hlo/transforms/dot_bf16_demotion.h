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

#ifndef XLA_BACKENDS_GPU_HLO_TRANSFORMS_DOT_BF16_DEMOTION_H_
#define XLA_BACKENDS_GPU_HLO_TRANSFORMS_DOT_BF16_DEMOTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Losslessly demotes TF32-eligible F32 dots to BF16 when both operands are
// provably BF16-exact and narrowing their producers will not duplicate
// materialized F32 buffers.
//
// Runs after simplification (HloCSE/HloDCE/ConditionalSimplifier) and before
// DotMerger using HloDataflowAnalysis:
//  1. Forward BF16-exactness analysis over F32 HloValues (across data movement,
//     tuples, conditionals, calls, and while loops).
//  2. Single-BFS F32 constraint propagation ensuring every materialized F32
//     value has a single storage type (either kept F32 everywhere or narrowed
//     in-place to BF16), while allowing direct zero-copy bypass of already-
//     materialized BF16 values (`convert<f32>(bf16)`) and scalar BF16-exact
//     constants.
//  3. In-place narrowing of F32 positions feeding surviving demoted dots.
class DotBf16Demotion : public HloModulePass {
 public:
  explicit DotBf16Demotion(const se::GpuComputeCapability& gpu_version =
                               se::CudaComputeCapability::Ampere())
      : gpu_version_(gpu_version) {}

  absl::string_view name() const override { return "dot-bf16-demotion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_HLO_TRANSFORMS_DOT_BF16_DEMOTION_H_
