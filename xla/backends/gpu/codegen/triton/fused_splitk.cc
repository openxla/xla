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

#include "xla/backends/gpu/codegen/triton/fused_splitk.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

#include "absl/numeric/bits.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// A dot qualifies as "narrow" if each non-contracting side fits into (a
// power-of-two slice of) a single output tile...
constexpr int64_t kMaxNarrowNonContractingSize = 16;
// ...and the contraction is long enough that splitting it is what provides
// the parallelism (it also guarantees at least a handful of contraction
// tiles for any reasonable tile size).
constexpr int64_t kMinContractionSize = 4096;
// Upper bound on the split purely as a safety backstop for tiny outputs
// (e.g. an unbatched narrow dot, where num_output_tiles == 1).
constexpr int64_t kMaxSplitK = 1024;
// The number of warps the emitted programs use; only feeds the occupancy
// estimate below, so it does not need to match the tuned config exactly.
constexpr int64_t kAssumedWarpsPerProgram = 4;

}  // namespace

std::optional<NarrowDotSizes> FusedSplitKQualifyingSizes(
    const HloDotInstruction* dot) {
  // The atomic fadd accumulates in the output buffer, so the output type
  // must be the f32 accumulator type.
  if (dot->shape().element_type() != F32 ||
      GetGemmAccumulatorType(dot) != F32) {
    return std::nullopt;
  }
  if (dot->dot_dimension_numbers().lhs_contracting_dimensions_size() != 1) {
    return std::nullopt;
  }
  absl::StatusOr<std::array<DotOperandDims, 2>> dims =
      DotOperandDims::FromDot(dot);
  if (!dims.ok()) {
    return std::nullopt;
  }
  NarrowDotSizes sizes{
      (*dims)[0].TotalSize(DotOperandDims::kNonContracting),
      (*dims)[1].TotalSize(DotOperandDims::kNonContracting),
      (*dims)[0].TotalSize(DotOperandDims::kContracting)};
  if (sizes.m <= 0 || sizes.n <= 0 ||
      sizes.m > kMaxNarrowNonContractingSize ||
      sizes.n > kMaxNarrowNonContractingSize ||
      sizes.k < kMinContractionSize) {
    return std::nullopt;
  }
  return sizes;
}

bool FusedSplitKEnabled(const HloModuleConfig& config) {
  const DebugOptions& debug_options = config.debug_options();
  return debug_options.xla_gpu_enable_fused_split_k() &&
         // The cross-split atomic float additions are order-nondeterministic.
         !RequireDeterminism(config) &&
         // The experimental tiling emitter does not support split-K.
         !debug_options.xla_gpu_experimental_enable_tiling_propagation();
}

int64_t ChooseFusedSplitK(const HloDotInstruction* dot, int64_t block_k,
                          const se::DeviceDescription& device_info) {
  std::optional<NarrowDotSizes> sizes = FusedSplitKQualifyingSizes(dot);
  if (block_k <= 0 || !sizes.has_value() ||
      dot != dot->parent()->root_instruction()) {
    return 1;
  }
  // For qualifying narrow dots the output tile covers all of M and N, so the
  // number of output tiles is the batch product (m/n are the full
  // non-contracting products, making the division exact).
  int64_t num_output_tiles = std::max<int64_t>(
      1, ShapeUtil::ElementsIn(dot->shape()) / (sizes->m * sizes->n));
  int64_t k_tiles = CeilOfRatio<int64_t>(sizes->k, block_k);

  // Enough programs to fully occupy every SM. Same computation as
  // CalculateSmOccupancy in gpu_dot_fusion_cost_model.h (unreachable from
  // here without a dependency cycle through the performance model): blocks
  // per SM limited by the thread slots and the block slots.
  int64_t threads_per_program =
      kAssumedWarpsPerProgram * device_info.threads_per_warp();
  int64_t blocks_per_sm =
      device_info.threads_per_core_limit() / threads_per_program;
  if (device_info.max_blocks_per_multiprocessor() > 0) {
    blocks_per_sm = std::min<int64_t>(
        blocks_per_sm, device_info.max_blocks_per_multiprocessor());
  }
  blocks_per_sm = std::max<int64_t>(1, blocks_per_sm);
  int64_t target = CeilOfRatio<int64_t>(
      blocks_per_sm * device_info.core_count(), num_output_tiles);
  int64_t split = static_cast<int64_t>(
      absl::bit_ceil(static_cast<uint64_t>(std::max<int64_t>(1, target))));
  split = std::min(split, kMaxSplitK);
  // Round down to a proper divisor of the contraction tile count, keeping at
  // least one full tile per program.
  while (split > 1 && (split >= k_tiles || k_tiles % split != 0)) {
    split /= 2;
  }
  return split;
}

int64_t ChooseFusedSplitKForFusionRoot(
    const HloInstruction& root, const se::DeviceDescription& device_info) {
  if (root.opcode() != HloOpcode::kDot) {
    return 1;
  }
  absl::StatusOr<Tile> tile = root.backend_config<Tile>();
  if (!tile.ok() || tile->sizes().empty()) {
    return 1;
  }
  return ChooseFusedSplitK(Cast<HloDotInstruction>(&root),
                           tile->sizes(tile->sizes_size() - 1), device_info);
}

}  // namespace gpu
}  // namespace xla
