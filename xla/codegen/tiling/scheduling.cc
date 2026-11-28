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

#include "xla/codegen/tiling/scheduling.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tile.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

absl::StatusOr<int64_t> CheckedMul(int64_t a, int64_t b) {
  int64_t result = 0;
  if (llvm::MulOverflow(a, b, result)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Tile count overflows int64: %d * %d.", a, b));
  }
  return result;
}

absl::StatusOr<int64_t> CheckedProduct(absl::Span<const int64_t> xs) {
  for (int64_t x : xs) {
    if (x < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Negative tile count: %d.", x));
    }
    if (x == 0) {
      return 0;
    }
  }
  int64_t product = 1;
  for (int64_t x : xs) {
    ABSL_ASSIGN_OR_RETURN(product, CheckedMul(product, x));
  }
  return product;
}

absl::StatusOr<int64_t> GetCheckedNumTiles(
    const TilingSpace& tiling_space, const TilingSpace::DimensionInfo& dim) {
  TF_RET_CHECK(!tiling_space.IsSymbolic() && dim.tile_size.has_value());
  if (*dim.tile_size <= 0 || dim.dimension_size < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid dimension size %d or tile size %d.",
                        dim.dimension_size, *dim.tile_size));
  }
  int64_t step = *dim.tile_size;
  if (dim.peer_dim.has_value()) {
    llvm::SmallVector<TilingSpace::DimensionInfo, 4> dims =
        tiling_space.dimensions();
    int64_t peer_id = dim.peer_dim->dim_id.value();
    TF_RET_CHECK(peer_id >= 0 && peer_id < static_cast<int64_t>(dims.size()));
    int64_t peers = dims[peer_id].dimension_size;
    if (peers <= 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid peer count: %d.", peers));
    }
    ABSL_ASSIGN_OR_RETURN(step, CheckedMul(step, peers));
  }
  return CeilOfRatio(dim.dimension_size, step);
}

// Uses effective stride CeilOfRatio(num_tiles, num_pids) to match
// Schedule::GetNumTilesPerPid() and cover [0, num_tiles) without gaps or OOB.
SymbolicExpr BuildGlobalTileId(int64_t num_tiles, int64_t num_tiles_per_pid,
                               mlir::MLIRContext* ctx, int64_t* num_pids) {
  *num_pids = CeilOfRatio(num_tiles, num_tiles_per_pid);
  int64_t effective_tiles_per_pid = CeilOfRatio(num_tiles, *num_pids);
  SymbolicExpr pid = CreateDimExpr(0, ctx);
  return effective_tiles_per_pid > 1
             ? pid * effective_tiles_per_pid + CreateDimExpr(1, ctx)
             : pid;
}

std::optional<int64_t> GetParallelDimensionPosition(
    const TilingSpace& tiling_space, TiledDimId target_id) {
  int64_t pos = 0;
  for (const TilingSpace::DimensionInfo& dim : tiling_space.dimensions()) {
    if (dim.type == TilingSpace::DimensionSemantics::kParallel) {
      if (dim.id == target_id) {
        return pos;
      }
      ++pos;
    }
  }
  return std::nullopt;
}

llvm::SmallVector<int64_t> SwapParallelDimensions(
    const HloInstruction& root, const TilingSpace& tiling_space, int64_t dim_a,
    int64_t dim_b, int64_t num_parallel_dims) {
  const int64_t rank = root.shape().dimensions().size();
  if (dim_a < 0 || dim_b < 0 || dim_a >= rank || dim_b >= rank) {
    return {};
  }
  const TilingSpace::DimensionInfo& a_info =
      tiling_space.GetDimensionInfo(root, dim_a);
  const TilingSpace::DimensionInfo& b_info =
      tiling_space.GetDimensionInfo(root, dim_b);
  if (a_info.type != TilingSpace::DimensionSemantics::kParallel ||
      b_info.type != TilingSpace::DimensionSemantics::kParallel) {
    return {};
  }
  std::optional<int64_t> a_pos =
      GetParallelDimensionPosition(tiling_space, a_info.id);
  std::optional<int64_t> b_pos =
      GetParallelDimensionPosition(tiling_space, b_info.id);
  if (!a_pos.has_value() || !b_pos.has_value() || *a_pos == *b_pos ||
      *a_pos >= num_parallel_dims || *b_pos >= num_parallel_dims) {
    return {};
  }
  llvm::SmallVector<int64_t> permutation(num_parallel_dims);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[*a_pos], permutation[*b_pos]);
  return permutation;
}

// Computes the parallel-dimensions permutation for the ragged non-contracting
// case.
llvm::SmallVector<int64_t> GetRaggedNonContractingPermutation(
    const HloInstruction& root, const TilingSpace& tiling_space,
    const DotDimensionNumbers& dot_dims, int64_t num_parallel_dims,
    int64_t M_avg) {
  const int64_t m_output_dim = dot_dims.lhs_batch_dimensions_size();
  const int64_t n_output_dim = m_output_dim + 1;
  if (n_output_dim >= static_cast<int64_t>(root.shape().dimensions().size()) ||
      M_avg >= root.shape().dimensions(n_output_dim)) {
    return {};
  }
  return SwapParallelDimensions(root, tiling_space, m_output_dim, n_output_dim,
                                num_parallel_dims);
}

// Computes the parallel-dimensions permutation for the ragged contracting case.
llvm::SmallVector<int64_t> GetRaggedContractingPermutation(
    const HloInstruction& root, const TilingSpace& tiling_space,
    const DotDimensionNumbers& dot_dims, int64_t num_parallel_dims,
    int64_t M_avg) {
  // Output ordering: [G, batch..., K (lhs_nc), N (rhs_nc)].
  const int64_t k_output_dim = 1 + dot_dims.lhs_batch_dimensions_size();
  const int64_t n_output_dim = k_output_dim + 1;
  if (n_output_dim >= static_cast<int64_t>(root.shape().dimensions().size()) ||
      M_avg >= root.shape().dimensions(n_output_dim)) {
    return {};
  }
  return SwapParallelDimensions(root, tiling_space, k_output_dim, n_output_dim,
                                num_parallel_dims);
}

// Computes the parallel-dimensions permutation for a ragged dot.
llvm::SmallVector<int64_t> GetRaggedDotPermutation(
    const HloInstruction& root, const TilingSpace& tiling_space,
    int64_t num_parallel_dims) {
  const auto* ragged_dot = Cast<HloRaggedDotInstruction>(&root);
  const RaggedDotDimensionNumbers& ragged_dims =
      ragged_dot->ragged_dot_dimension_numbers();
  const DotDimensionNumbers& dot_dims = ragged_dims.dot_dimension_numbers();
  if (ragged_dims.lhs_ragged_dimensions_size() != 1) {
    return {};
  }
  const Shape& lhs_shape = root.operand(0)->shape();
  const Shape& rhs_shape = root.operand(1)->shape();
  const Shape& group_sizes_shape = root.operand(2)->shape();
  const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  if (lhs_ragged_dim < 0 ||
      lhs_ragged_dim >= static_cast<int64_t>(lhs_shape.dimensions().size()) ||
      group_sizes_shape.dimensions().empty() ||
      absl::c_contains(dot_dims.lhs_batch_dimensions(), lhs_ragged_dim)) {
    return {};
  }
  const int64_t num_lhs_non_contracting =
      lhs_shape.dimensions().size() -
      dot_dims.lhs_contracting_dimensions_size() -
      dot_dims.lhs_batch_dimensions_size();
  const int64_t num_rhs_non_contracting =
      rhs_shape.dimensions().size() -
      dot_dims.rhs_contracting_dimensions_size() -
      dot_dims.rhs_batch_dimensions_size() -
      ragged_dims.rhs_group_dimensions_size();
  if (num_lhs_non_contracting != 1 || num_rhs_non_contracting != 1) {
    return {};
  }

  const bool is_contracting =
      absl::c_contains(dot_dims.lhs_contracting_dimensions(), lhs_ragged_dim);
  const int64_t M_total = lhs_shape.dimensions(lhs_ragged_dim);
  const int64_t G = group_sizes_shape.dimensions().back();
  const int64_t M_avg = M_total / std::max<int64_t>(G, 1);

  if (!is_contracting) {
    return GetRaggedNonContractingPermutation(root, tiling_space, dot_dims,
                                              num_parallel_dims, M_avg);
  }
  return GetRaggedContractingPermutation(root, tiling_space, dot_dims,
                                         num_parallel_dims, M_avg);
}

llvm::SmallVector<int64_t> GetParallelDimensionsPermutation(
    const TiledHloComputation& tiled_computation) {
  if (tiled_computation.roots().size() != 1) {
    return {};
  }

  const HloInstruction* root = tiled_computation.roots().front()->hlo();
  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  int64_t num_parallel_dims = tiling_space.num_parallel_dimensions();
  if (num_parallel_dims < 2) {
    return {};
  }

  // ---- Regular dot / scaled dot ----
  if (root->opcode() == HloOpcode::kDot ||
      root->opcode() == HloOpcode::kScaledDot) {
    const Shape& lhs_shape = root->operand(0)->shape();
    const Shape& rhs_shape = root->operand(1)->shape();
    const DotDimensionNumbers& dimension_numbers =
        root->dot_dimension_numbers();

    // We only support transposing standard [batch..., m, k] * [batch..., k, n]
    // shapes where m and n are exactly rank 1.
    int64_t num_lhs_non_contracting_dims =
        lhs_shape.dimensions().size() -
        dimension_numbers.lhs_contracting_dimensions_size() -
        dimension_numbers.lhs_batch_dimensions_size();
    int64_t num_rhs_non_contracting_dims =
        rhs_shape.dimensions().size() -
        dimension_numbers.rhs_contracting_dimensions_size() -
        dimension_numbers.rhs_batch_dimensions_size();
    if (num_lhs_non_contracting_dims != 1 ||
        num_rhs_non_contracting_dims != 1) {
      return {};
    }

    // Heuristic: if the LHS operand is smaller than the RHS operand, it is more
    // beneficial to traverse the RHS non-contracting dimensions ('n') first
    // (more slowly) while keeping the LHS tile in the L2 cache.
    if (ShapeUtil::ByteSizeOf(lhs_shape) >= ShapeUtil::ByteSizeOf(rhs_shape)) {
      return {};
    }

    const int64_t m_output_dim = dimension_numbers.lhs_batch_dimensions_size();
    return SwapParallelDimensions(*root, tiling_space, m_output_dim,
                                  m_output_dim + 1, num_parallel_dims);
  }

  // ---- Ragged dot ----
  if (root->opcode() == HloOpcode::kRaggedDot) {
    return GetRaggedDotPermutation(*root, tiling_space, num_parallel_dims);
  }

  return {};
}

// What the scheduler needs to know about the index-wise variadic instructions
// of a computation.
struct IndexWiseVariadicInfo {
  // The index-wise variadic instructions, in any region of the computation.
  llvm::SmallVector<const TiledHloInstruction*, 2> instructions;
  // Maps each all-gather dimension ID to how many tiles each peer owns along
  // that dimension.
  //
  // Example: `f32[4, 16] -> f32[16, 16] all-gather` across 4 GPUs on dim 0
  // (`d0`) with tile size [2, 16]:
  //   - `d0` has 16 / 2 = 8 tiles in total across all 4 GPUs.
  //   - Each GPU owns a 4-row slice, which is 4 / 2 = 2 tiles along `d0`.
  //   - So `tiles_per_peer_by_dim[d0] = 2`.
  llvm::DenseMap<int64_t, int64_t> tiles_per_peer_by_dim;
  // Number of peers across `tiles_per_peer_by_dim` and `kReplica` dimensions.
  int64_t num_peers = 1;
};

// Collects the index-wise variadic instructions of `region` and its nested
// regions.
void CollectIndexWiseVariadic(
    const TiledHloRegion& region,
    llvm::SmallVectorImpl<const TiledHloInstruction*>& result) {
  for (const TiledHloInstruction* instruction : region.instructions()) {
    if (IsIndexWiseVariadic(*instruction->hlo())) {
      result.push_back(instruction);
    }
    for (const TiledHloRegion& nested : instruction->hlo_regions()) {
      CollectIndexWiseVariadic(nested, result);
    }
  }
}

// Finds the index-wise variadic instructions of the computation and, for the
// operand tiles that carry a replica id, derives which parallel tile dimension
// carries the peer index and how many tiles per peer it spans.
absl::StatusOr<IndexWiseVariadicInfo> BuildIndexWiseVariadicInfo(
    const TiledHloComputation& tiled_computation) {
  IndexWiseVariadicInfo info;
  CollectIndexWiseVariadic(tiled_computation.tiled_root_region(),
                           info.instructions);
  if (info.instructions.empty()) {
    return info;
  }

  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  TF_RET_CHECK(!tiling_space.IsSymbolic())
      << "Expected tile sizes to be assigned at this point.";
  llvm::SmallVector<TilingSpace::DimensionInfo, 4> dims =
      tiling_space.dimensions();
  const int64_t num_dims = dims.size();

  llvm::SmallVector<int64_t, 4> tile_sizes;
  tile_sizes.reserve(num_dims);
  for (const TilingSpace::DimensionInfo& dim : dims) {
    if (!dim.tile_size.has_value() || *dim.tile_size <= 0 ||
        dim.dimension_size < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid dimension %d: size=%d, tile_size=%d.", dim.id.value(),
          dim.dimension_size, dim.tile_size.value_or(-1)));
    }
    tile_sizes.push_back(*dim.tile_size);
  }
  // Variables are laid out as [tile ids, tile sizes, runtime variables].
  const int64_t num_vars = 2 * num_dims + tiling_space.num_rt_vars();

  std::optional<int64_t> num_peers;
  for (const TiledHloInstruction* instruction : info.instructions) {
    const HloInstruction& hlo = *instruction->hlo();
    for (const auto& [output_index, operand] :
         llvm::enumerate(instruction->operands())) {
      const Tile& tile = operand->tile();
      if (tile.num_replica_ids() == 0) {
        // The output tile is not associated with a single peer; nothing to
        // factor out of its tile ids.
        continue;
      }
      TF_RET_CHECK(tile.num_replica_ids() == 1)
          << "Only one replica dimension is supported.";
      const DimTile& replica_dim_tile = tile.replica_ids().back();
      if (!replica_dim_tile.upper_bound ||
          replica_dim_tile.upper_bound.GetType() !=
              SymbolicExprType::kConstant) {
        return absl::UnimplementedError(absl::StrFormat(
            "Replica upper bound of output %d of %s must be constant.",
            output_index, hlo.name()));
      }
      const int64_t num_replicas = replica_dim_tile.upper_bound.GetValue();
      if (num_replicas <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica count of output %d of %s must be positive, got %d.",
            output_index, hlo.name(), num_replicas));
      }
      SymbolicExpr peer = replica_dim_tile.offset;
      llvm::DenseSet<VariableID> used_vars;
      peer.GetUsedVariables(used_vars);
      if (used_vars.size() != 1 || *used_vars.begin() < 0 ||
          *used_vars.begin() >= num_dims ||
          dims[*used_vars.begin()].type !=
              TilingSpace::DimensionSemantics::kParallel) {
        return absl::UnimplementedError(absl::StrFormat(
            "Peer index %s of output %d of %s must depend on exactly one "
            "parallel tile id.",
            peer.ToString(num_dims), output_index, hlo.name()));
      }
      const int64_t dim_id = *used_vars.begin();
      // Number of blocks across all replicas.
      const int64_t num_blocks =
          CeilOfRatio(dims[dim_id].dimension_size, tile_sizes[dim_id]);
      if (num_blocks % num_replicas != 0) {
        return absl::UnimplementedError(absl::StrFormat(
            "Dimension %d of output %d of %s has %d blocks, which is not a "
            "multiple of the number of peers %d.",
            dim_id, output_index, hlo.name(), num_blocks, num_replicas));
      }
      const int64_t tiles_per_peer = num_blocks / num_replicas;
      std::vector<int64_t> values(num_vars, 0);
      auto peer_at = [&](int64_t tile_id) {
        values[dim_id] = tile_id;
        return peer.Evaluate(values);
      };
      if (!llvm::all_of(llvm::seq<int64_t>(0, num_blocks), [&](int64_t t) {
            return peer_at(t) == t / tiles_per_peer;
          })) {
        return absl::UnimplementedError(absl::StrFormat(
            "Peer index %s of output %d of %s must be tile id %d divided by "
            "%d.",
            peer.ToString(num_dims), output_index, hlo.name(), dim_id,
            tiles_per_peer));
      }

      if (num_peers.has_value() && *num_peers != num_replicas) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output %d of %s reads from %d peers, but other outputs read "
            "from %d peers.",
            output_index, hlo.name(), num_replicas, *num_peers));
      }
      num_peers = num_replicas;
      info.tiles_per_peer_by_dim[dim_id] = tiles_per_peer;
    }
  }
  for (const TilingSpace::DimensionInfo& dim : dims) {
    if (dim.type != TilingSpace::DimensionSemantics::kReplica) {
      continue;
    }
    if (dim.dimension_size <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Replica dimension %d of %s must be positive, got %d.",
          dim.id.value(), dim.hlo->name(), dim.dimension_size));
    }
    if (num_peers.has_value() && *num_peers != dim.dimension_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Replica dimension %d of %s has %d replicas, but other dimensions "
          "have %d peers.",
          dim.id.value(), dim.hlo->name(), dim.dimension_size, *num_peers));
    }
    num_peers = dim.dimension_size;
  }
  info.num_peers = num_peers.value_or(1);
  return info;
}

// Internal struct for holding the parallel dimensions of a root.
struct RootDims {
  // Parallel output dimension partitioned across peers (gather dimension for
  // all-gather, scatter dimension for reduce-scatter).
  std::optional<int64_t> peer_dim_id;
  int64_t tiles_per_peer = 1;
  llvm::SmallVector<int64_t, 4> non_peer_dim_ids;
  llvm::SmallVector<int64_t, 4> non_peer_block_counts;
  int64_t num_tiles = 1;
};

absl::StatusOr<llvm::SmallVector<RootDims, 2>> BuildRootsForIndexWiseVariadic(
    const TiledHloComputation& tiled_computation,
    const IndexWiseVariadicInfo& info) {
  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  // Group the parallel dimensions by the root (output) that defines them.
  llvm::SmallVector<RootDims, 2> roots;
  int64_t num_grouped_dims = 0;
  for (const auto& [root_index, tiled_root] :
       llvm::enumerate(tiled_computation.roots())) {
    const HloInstruction& hlo = *tiled_root->hlo();
    RootDims& root = roots.emplace_back();
    for (int64_t i = 0; i < GetFirstShape(&hlo).dimensions().size(); ++i) {
      const TilingSpace::DimensionInfo* dim =
          &tiling_space.GetDimensionInfo(hlo, i);
      if (dim->type != TilingSpace::DimensionSemantics::kParallel) {
        continue;
      }
      ++num_grouped_dims;
      const int64_t dim_id = dim->id.value();
      auto it = info.tiles_per_peer_by_dim.find(dim_id);
      if (dim->peer_dim.has_value() || it != info.tiles_per_peer_by_dim.end()) {
        if (root.peer_dim_id.has_value() ||
            (dim->peer_dim.has_value() &&
             it != info.tiles_per_peer_by_dim.end())) {
          return absl::UnimplementedError(absl::StrFormat(
              "Output %d of %s has more than one dimension carrying the peer "
              "index.",
              root_index, hlo.name()));
        }
        root.peer_dim_id = dim_id;
        if (!dim->peer_dim.has_value()) {
          root.tiles_per_peer = it->second;
          continue;
        }
      }
      ABSL_ASSIGN_OR_RETURN(int64_t num_blocks,
                            GetCheckedNumTiles(tiling_space, *dim));
      root.non_peer_dim_ids.push_back(dim_id);
      root.non_peer_block_counts.push_back(num_blocks);
    }
    if (info.num_peers > 1 && !root.peer_dim_id.has_value()) {
      return absl::UnimplementedError(
          absl::StrFormat("Output %d of %s does not depend on the peer index.",
                          root_index, hlo.name()));
    }
    ABSL_ASSIGN_OR_RETURN(int64_t non_peer_tiles,
                          CheckedProduct(root.non_peer_block_counts));
    ABSL_ASSIGN_OR_RETURN(root.num_tiles,
                          CheckedMul(non_peer_tiles, root.tiles_per_peer));
  }
  if (num_grouped_dims != tiling_space.num_parallel_dimensions()) {
    return absl::UnimplementedError(
        "Not all parallel dimensions are defined by the roots.");
  }
  return roots;
}

absl::StatusOr<Schedule> BuildIndexWiseVariadicSchedule(
    mlir::MLIRContext* ctx, const TilingSpace& tiling_space,
    const IndexWiseVariadicInfo& info, llvm::ArrayRef<RootDims> roots,
    int64_t num_tiles_per_pid, int64_t max_tiles) {
  ABSL_ASSIGN_OR_RETURN(int64_t total_tiles,
                        CheckedMul(info.num_peers, max_tiles));
  Schedule schedule;
  llvm::SmallVector<TilingSpace::DimensionInfo, 4> dims =
      tiling_space.dimensions();
  SymbolicExpr zero = CreateSymbolicConstant(0, ctx);
  auto set_root_zero = [&](const RootDims& root) {
    for (int64_t dim_id : root.non_peer_dim_ids) {
      schedule.dim_id_to_pid_expr[dim_id] = zero;
    }
    if (root.peer_dim_id.has_value()) {
      const TilingSpace::DimensionInfo& peer_dim = dims[*root.peer_dim_id];
      int64_t id = peer_dim.peer_dim.has_value()
                       ? peer_dim.peer_dim->dim_id.value()
                       : *root.peer_dim_id;
      schedule.dim_id_to_pid_expr.try_emplace(id, zero);
    }
  };
  if (total_tiles == 0) {
    for (const RootDims& root : roots) {
      set_root_zero(root);
    }
    schedule.num_tiles = 0;
    schedule.num_pids = 0;
    return schedule;
  }

  int64_t num_pids = 0;
  SymbolicExpr global_tile_id =
      BuildGlobalTileId(total_tiles, num_tiles_per_pid, ctx, &num_pids);
  SymbolicExpr peer = global_tile_id.floorDiv(max_tiles);
  SymbolicExpr intra_peer_tile_id =
      info.num_peers > 1 ? global_tile_id % max_tiles : global_tile_id;
  for (const RootDims& root : roots) {
    if (root.num_tiles == 0) {
      set_root_zero(root);
      continue;
    }
    llvm::SmallVector<int64_t, 4> block_counts = root.non_peer_block_counts;
    if (root.tiles_per_peer > 1) {
      block_counts.push_back(root.tiles_per_peer);
    }
    // `DelinearizeIndex` does not wrap its outermost component, so for a padded
    // tile (`intra_peer_tile_id >= num_tiles`) that component exceeds its
    // block count and the tile is masked by its upper bound.
    llvm::SmallVector<SymbolicExpr, 4> delinearized =
        DelinearizeIndex(block_counts, intra_peer_tile_id, ctx);
    for (int64_t i = 0; i < root.non_peer_dim_ids.size(); ++i) {
      schedule.dim_id_to_pid_expr[root.non_peer_dim_ids[i]] = delinearized[i];
    }
    if (!root.peer_dim_id.has_value()) {
      continue;
    }
    const TilingSpace::DimensionInfo& peer_dim = dims[*root.peer_dim_id];
    if (peer_dim.peer_dim.has_value()) {
      schedule.dim_id_to_pid_expr[peer_dim.peer_dim->dim_id.value()] = peer;
    } else {
      SymbolicExpr peer_tile_id = peer * root.tiles_per_peer;
      if (root.tiles_per_peer > 1) {
        peer_tile_id = peer_tile_id + delinearized.back();
      }
      schedule.dim_id_to_pid_expr[*root.peer_dim_id] = peer_tile_id;
    }
    if (root.num_tiles < max_tiles) {
      // Push the padded dimension past the last peer (`d_replica` when peers
      // stride across the root dimension, or the spatial dimension when
      // `kReplica` selects a peer output buffer so it stays in `[0,
      // num_peers)`).
      int64_t padded_dim_id =
          peer_dim.peer_dim.has_value() &&
                  peer_dim.peer_dim->mode !=
                      TilingSpace::PeerIndexing::kPerPeerBuffer
              ? peer_dim.peer_dim->dim_id.value()
              : *root.peer_dim_id;
      ABSL_ASSIGN_OR_RETURN(int64_t pad_stride,
                            CheckedMul(info.num_peers, root.num_tiles));
      SymbolicExpr& padded_pid_expr =
          schedule.dim_id_to_pid_expr[padded_dim_id];
      padded_pid_expr =
          padded_pid_expr +
          intra_peer_tile_id.floorDiv(root.num_tiles) * pad_stride;
    }
  }
  schedule.num_tiles = total_tiles;
  schedule.num_pids = num_pids;
  return schedule;
}

// Schedules a computation that contains index-wise variadic instructions.
//
// All roots are "zipped": one program handles tile `i` of every root, so
// that each variadic instruction is executed once per program for all of
// its outputs. If the outputs are associated with a peer the peer index is
// the outermost tile index and is shared by all roots; below it every root
// gets its own intra-peer tile index over its remaining ("non-peer")
// parallel dimensions and any intra-peer tiles along the peer dimension
// itself. Roots with fewer tiles than the largest root are padded: their
// padded tiles are mapped out of bounds so that the emitter masks them.
absl::StatusOr<Schedule> GetIndexWiseVariadicSchedule(
    const TiledHloComputation& tiled_computation,
    const IndexWiseVariadicInfo& info, int64_t num_tiles_per_pid) {
  mlir::MLIRContext* ctx = tiled_computation.GetMLIRContext();
  using Roots = llvm::SmallVector<RootDims, 2>;
  ABSL_ASSIGN_OR_RETURN(
      Roots roots, BuildRootsForIndexWiseVariadic(tiled_computation, info));

  // Find the maximum number of tiles across all roots.
  int64_t max_tiles = roots.empty() ? 1 : 0;
  for (const RootDims& root : roots) {
    max_tiles = std::max(max_tiles, root.num_tiles);
  }
  return BuildIndexWiseVariadicSchedule(ctx, tiled_computation.tiling_space(),
                                        info, roots, num_tiles_per_pid,
                                        max_tiles);
}

}  // namespace

std::string Schedule::ToString() const {
  std::set<int64_t> dim_ids;
  for (const auto& [dim_id, expr] : dim_id_to_pid_expr) {
    dim_ids.insert(dim_id);
  }
  std::vector<std::string> expr_strs;
  expr_strs.reserve(dim_ids.size());
  for (int64_t dim_id : dim_ids) {
    expr_strs.push_back(absl::StrFormat(
        "d%d -> %s", dim_id,
        dim_id_to_pid_expr.at(dim_id).ToString({"pid", "tid"}, {})));
  }
  return absl::StrFormat("%s, num_pids=%d, num_tiles=%d",
                         absl::StrJoin(expr_strs, ", "), num_pids, num_tiles);
}

absl::StatusOr<Schedule> GetSchedule(
    const TiledHloComputation& tiled_computation, int64_t num_tiles_per_pid) {
  if (num_tiles_per_pid <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "num_tiles_per_pid must be positive, got %d.", num_tiles_per_pid));
  }
  ABSL_ASSIGN_OR_RETURN(IndexWiseVariadicInfo variadic_info,
                        BuildIndexWiseVariadicInfo(tiled_computation));
  if (!variadic_info.instructions.empty()) {
    return GetIndexWiseVariadicSchedule(tiled_computation, variadic_info,
                                        num_tiles_per_pid);
  }

  // Compute the block counts for each replica and parallel dimension.
  // Replica dimensions are placed outermost so each contiguous group of
  // parallel blocks targets the same peer.
  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  TF_RET_CHECK(!tiling_space.IsSymbolic())
      << "Expected tile sizes to be assigned at this point.";
  llvm::SmallVector<int64_t, 4> block_counts;
  llvm::SmallVector<int64_t, 4> dim_ids;
  llvm::SmallVector<int64_t, 4> parallel_dim_block_counts;
  llvm::SmallVector<int64_t, 4> parallel_dim_ids;
  for (const auto& [dim_id, dimension] :
       llvm::enumerate(tiling_space.dimensions())) {
    if (dimension.type == TilingSpace::DimensionSemantics::kReplica) {
      ABSL_ASSIGN_OR_RETURN(int64_t count,
                            GetCheckedNumTiles(tiling_space, dimension));
      block_counts.push_back(count);
      dim_ids.push_back(dim_id);
    } else if (dimension.type == TilingSpace::DimensionSemantics::kParallel) {
      ABSL_ASSIGN_OR_RETURN(int64_t count,
                            GetCheckedNumTiles(tiling_space, dimension));
      parallel_dim_block_counts.push_back(count);
      parallel_dim_ids.push_back(dim_id);
    }
  }

  llvm::SmallVector<int64_t> permutation =
      GetParallelDimensionsPermutation(tiled_computation);
  if (!permutation.empty()) {
    parallel_dim_block_counts = llvm::to_vector<4>(
        xla::Permute(parallel_dim_block_counts, permutation));
    parallel_dim_ids =
        llvm::to_vector<4>(xla::Permute(parallel_dim_ids, permutation));
  }
  block_counts.append(parallel_dim_block_counts);
  dim_ids.append(parallel_dim_ids);

  mlir::MLIRContext* ctx = tiled_computation.GetMLIRContext();
  ABSL_ASSIGN_OR_RETURN(int64_t num_tiles, CheckedProduct(block_counts));
  Schedule schedule;
  if (num_tiles == 0) {
    SymbolicExpr zero = CreateSymbolicConstant(0, ctx);
    for (int64_t dim_id : dim_ids) {
      schedule.dim_id_to_pid_expr[dim_id] = zero;
    }
    schedule.num_tiles = 0;
    schedule.num_pids = 0;
    return schedule;
  }

  int64_t num_pids = 0;
  SymbolicExpr global_tile_id =
      BuildGlobalTileId(num_tiles, num_tiles_per_pid, ctx, &num_pids);
  llvm::SmallVector<SymbolicExpr, 4> delinearized_pid =
      DelinearizeIndex(block_counts, global_tile_id, ctx);
  for (const auto& [dim_id, expr] : llvm::zip(dim_ids, delinearized_pid)) {
    schedule.dim_id_to_pid_expr[dim_id] = expr;
  }
  schedule.num_tiles = num_tiles;
  schedule.num_pids = num_pids;
  return schedule;
}

int64_t Schedule::GetNumTilesPerPid() const {
  if (num_tiles <= 0 || num_pids <= 0) {
    return 0;
  }
  return CeilOfRatio(num_tiles, num_pids);
}

}  // namespace xla::gpu
