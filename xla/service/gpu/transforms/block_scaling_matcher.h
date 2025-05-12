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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_BLOCK_SCALING_MATCHER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_BLOCK_SCALING_MATCHER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {
namespace block_scaling {

// Result of matching a block scaled dequantize operation graph.
// Example:
//   %input = f32[16,128] convert(parameter(0))
//   %scale = f32[16,4] convert(parameter(1))
//   %broadcast = f32[16,4,32] broadcast(%scale), dimensions={0,1}
//   %reshape = f32[16,128] reshape(%broadcast)
//   %result = f32[16,128] multiply(%input, %reshape)
struct BlockScaledDequantizeOps {
  static std::optional<BlockScaledDequantizeOps> Match(
      const HloInstruction* instruction);

  // Matched instruction references.
  const HloInstruction* input;
  const HloInstruction* scale;
  const HloInstruction* broadcast;
  const HloInstruction* reshape;
  const HloInstruction* result;

  // Get the block size from the reshape instruction.
  int64_t GetBlockSize(int64_t dimension) const;

  // Get scale parameter reference (may return nullptr).
  const HloInstruction* GetScaleParameter() const;
};

// Result of matching a block scaled dot operation graph.
struct BlockScaledDotOps {
  static std::optional<BlockScaledDotOps> Match(
      const HloInstruction* instruction);

  // Matched instruction references.
  BlockScaledDequantizeOps lhs;
  BlockScaledDequantizeOps rhs;
  const HloInstruction* result;

  // Dimension number helpers.
  int64_t lhs_contracting_dim() const;
  int64_t lhs_noncontracting_dim() const;
  std::optional<int64_t> lhs_batch_dim() const;

  int64_t rhs_contracting_dim() const;
  int64_t rhs_noncontracting_dim() const;
  std::optional<int64_t> rhs_batch_dim() const;

  // Checks that the block scaled dot operation is supported.
  // cuDNN kernel currently supports MXFP8 and NVFP4.
  bool IsSupported() const;
  bool IsMXFP8() const;
  bool IsNVFP4() const;
};

// cuDNN scale swizzling constants.
static constexpr int64_t kScaleContractingTileSize = 4;
static constexpr int64_t kScaleNoncontractingTileSize = 128;
static constexpr int64_t kSwizzleHorizontalSize = 4;
static constexpr int64_t kSwizzleVerticalSize = 32;
static_assert(kSwizzleHorizontalSize * kSwizzleVerticalSize ==
              kScaleNoncontractingTileSize);

// Common block size constants.
static constexpr int kBlockSizeMXFP8 = 32;
static constexpr int kBlockSizeNVFP4 = 16;

// Composite name for the block scaled dot operation.
static constexpr char kBlockScaledDotCompositeName[] = "mx.block_scaled_dot";

}  // namespace block_scaling
}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_BLOCK_SCALING_MATCHER_H_
