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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_BLOCK_SCALING_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_BLOCK_SCALING_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Handles composites representing a dot operation on quantized inputs
// (block scaled dot).
// This operation has hardware support on Blackwell, and is available starting
// from cuDNN v9.7 and cuDNN frontend v1.10.
//
// The block scaled dot composite takes four inputs (lhs, lhs_scale, rhs,
// rhs_scale), where each side is represented by two tensors: (a) quantized
// input, and (b) scaling factor, which gets broadcasted to the input size.
//
// Simplified operation graph:
//   1. lhs_dequantized = multiply(lhs, broadcast(lhs_scale))
//   2. rhs_dequantized = multiply(rhs, broadcast(rhs_scale))
//   3. block_scaled_dot = dot(lhs_dequantized, rhs_dequantized)
//
// The cuDNN kernel supports the following formats:
//   - MXFP8: input type E4M3FN or E5M2, scale type E8M0FNU, block size 32;
//   - NVFP4: input type E2M1FN, scale type E4M3FN (positive), block size 16;
//
// Additionally, the cuDNN kernel imposes some restrictions on the format of
// the scaling factor tensor (minimum tile size 128x4, must be swizzled in a
// specific way). The passes add the necessary transformations (padding,
// transposition) to satisfy these constraints.
//
// Dot dimension numbers must be normalized (one contracting dimension, one
// non-contracting dimension, at most one batch dimension) in order to be
// matched by this pass.

// The first pass rewrites composite calls or custom calls representing a block
// scaled dot operation into a dot fusion that will be picked up by the
// autotuner.
class BlockScalingRewriter : public HloModulePass {
 public:
  explicit BlockScalingRewriter(bool allow_cudnn) : allow_cudnn_(allow_cudnn) {}

  absl::string_view name() const override { return "block-scaling-rewriter"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Custom call target.
  static constexpr absl::string_view kBlockScaledDotCustomCallTarget =
      "__op$block_scaled_dot";

 private:
  bool allow_cudnn_;
};

// The second pass prepends swizzling of the scaling factors before the cuDNN
// fusion containing block scaled dot operation (required by the kernel). The
// added computation may be fused with other ops by the following fusion passes.
class CudnnBlockScalingRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "cudnn-block-scaling-rewriter";
  }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Verify that the block scaled dot operation is supported by cuDNN.
  static bool IsCudnnSupported(const HloInstruction* root);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_BLOCK_SCALING_REWRITER_H_
