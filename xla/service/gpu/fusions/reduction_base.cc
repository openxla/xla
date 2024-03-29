/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/reduction_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/union_find.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

const Shape& FirstShape(const Shape& in) {
  return in.IsTuple() ? in.tuple_shapes(0) : in;
}

int RowReductionGetRowsPerWarp(int reduced_dimension_size) {
  if (WarpSize() % reduced_dimension_size != 0 ||
      reduced_dimension_size >= WarpSize()) {
    return 1;
  }
  return WarpSize() / reduced_dimension_size;
}

int64_t ComputeColReductionActiveCore(Vector3 reduction_dimensions,
                                      int64_t tile_y, int64_t num_threads_y,
                                      int64_t num_threads_x, int vector_size) {
  constexpr int kColMajorKept = ReductionDimensions::kColMajorKeptDimension;
  constexpr int kColReduced = ReductionDimensions::kColReducedDimension;
  constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;
  // Compute active core number by assuming each block occupy one sm core. It is
  // a conservative but easy approach otherwise we should consider shared memory
  // size. Due to column reduction always use 1024 as block size, the computed
  // active core num has not much difference from the actual situation.
  int64_t blocks_x = CeilOfRatio(reduction_dimensions[kColMinorKept],
                                 num_threads_x * vector_size);
  int64_t block_tile_y = num_threads_y * tile_y;
  int64_t blocks_y =
      CeilOfRatio(reduction_dimensions[kColReduced], block_tile_y);
  int64_t blocks = reduction_dimensions[kColMajorKept] * blocks_x * blocks_y;
  return blocks;
}

int GetVectorSize(const HloFusionAnalysis& analysis,
                  const ReductionDimensions& reduction_dimensions,
                  int64_t num_threads_y, int64_t num_threads_x,
                  Vector3 reduction_tiling, bool column_vectorization,
                  bool adjust_tiling) {
  if (MayPreventVectorization(analysis.fusion())) {
    return 1;
  }

  constexpr int kColReduced = ReductionDimensions::kColReducedDimension;
  constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;
  if (!reduction_dimensions.is_row_reduction) {
    if (!column_vectorization) return 1;
    int vector_size = 2;
    auto core_num_if_vectorize = ComputeColReductionActiveCore(
        reduction_dimensions.dimensions, reduction_tiling[kColReduced],
        num_threads_y, num_threads_x, vector_size);
    auto num_kept_minor = reduction_dimensions.dimensions[kColMinorKept];
    // Check if the last dimension is divisible by (vector_size *
    // num_threads_x) and active core is enough or allow to adjust tiling.
    if (num_kept_minor % (vector_size * num_threads_x) == 0 &&
        (core_num_if_vectorize >= analysis.device_info().core_count() ||
         adjust_tiling)) {
      return vector_size;
    }
    return 1;
  }

  constexpr int kRowMinorReduced =
      ReductionDimensions::kRowMinorReducedDimension;
  if (reduction_dimensions.dimensions[kRowMinorReduced] % 2 != 0) {
    return 1;
  }

  // Enabling vectorization if number of threads is <= warpsize leads to half or
  // more of the threads not doing any work.
  if (num_threads_x <= WarpSize()) {
    return 1;
  }

  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &analysis.device_info().gpu_compute_capability());
  if (cuda_cc == nullptr) return 1;
  if (cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) return 2;
  if (cuda_cc->IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
    return analysis.input_output_info().smallest_input_dtype_bits <= 32 &&
                   reduction_dimensions.dimensions[kRowMinorReduced] %
                           (reduction_tiling[kRowMinorReduced] *
                            num_threads_x) ==
                       0
               ? 2
               : 1;
  }
  return 1;
}

std::tuple<Vector3, int, bool> AdjustColReductionTilingConfig(
    const HloFusionAnalysis& analysis, Vector3 reduction_dimensions,
    Vector3 reduction_tiling, int64_t num_threads_y, int64_t num_threads_x,
    int vector_size) {
  constexpr int kColMajorKept = ReductionDimensions::kColMajorKeptDimension;
  constexpr int kColReduced = ReductionDimensions::kColReducedDimension;
  constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;

  auto core_count = analysis.device_info().core_count();
  constexpr int minimum_tile_size = 8;

  // Early return if device occupancy is already high.
  if (ComputeColReductionActiveCore(
          reduction_dimensions, reduction_tiling[kColReduced], num_threads_y,
          num_threads_x, vector_size) >= core_count) {
    return {reduction_tiling, vector_size, false};
  }

  auto roots = analysis.fusion().GetRoots();
  for (auto [root, hero] : llvm::zip(roots, analysis.fusion_heroes())) {
    // Only adjust tile_y if hero is reduction and output element type is F32.
    // F32 atomic is fast so that we can ignore the extra atomic overhead
    // by adjusting tile_y to increase the parallelism of the kernel.
    if (hero->opcode() == HloOpcode::kReduce) {
      if (hero != (&root.instruction()) ||
          hero->shape().element_type() != F32) {
        // If we can not adjust tile_y but active core number is small, reset
        // vector size as 1.
        return {reduction_tiling, 1, false};
      }
    }
  }

  auto actual_tile_size =
      CeilOfRatio(reduction_dimensions[kColReduced], num_threads_y);
  auto current_tile_size = actual_tile_size;
  while (current_tile_size >= minimum_tile_size * 2) {
    if (ComputeColReductionActiveCore(reduction_dimensions, current_tile_size,
                                      num_threads_y, num_threads_x,
                                      vector_size) >= core_count)
      break;
    current_tile_size = current_tile_size / 2;
  }
  bool tile_size_decreased = current_tile_size != actual_tile_size;
  reduction_tiling[kColReduced] = current_tile_size;
  return {reduction_tiling, tile_size_decreased ? vector_size : 1,
          tile_size_decreased};
}

ReductionGroups GroupDisjointReductions(const HloFusionAnalysis& analysis) {
  const int num_fusion_outputs = analysis.fusion_roots().size();

  CHECK_NE(0, num_fusion_outputs);
  if (num_fusion_outputs == 1) {
    return {{{analysis.fusion_roots()[0]}}, {0}, {true}};
  }

  absl::node_hash_map<HloInstructionAdaptor,
                      tensorflow::UnionFind<HloInstructionAdaptor>>
      disjoint_sets;

  // TODO(b/249976438): we currently do not treat properly
  // aliasing between inputs and outputs of the fusion, so for now put all
  // non-reduction roots into one group to avoid read-after-write conflicts.
  std::optional<HloInstructionAdaptor> first_non_reduction_root = std::nullopt;

  absl::node_hash_map<HloInstructionAdaptor,
                      absl::flat_hash_set<HloInstructionAdaptor>>
      reachable_outputs;
  absl::flat_hash_set<HloInstructionAdaptor> roots_with_reduction;
  absl::flat_hash_map<const HloInstruction*, int> root_indices;
  const auto& roots = analysis.fusion().GetRoots();
  ReductionGroups result;
  result.group_id_per_root.resize(roots.size());
  result.is_reduction_root.reserve(roots.size());
  for (auto [root, hero] : llvm::zip(roots, analysis.fusion_heroes())) {
    int index = root_indices.size();
    root_indices[&root.instruction()] = index;
    disjoint_sets[root].Get() = root;
    reachable_outputs[root].insert(root);
    result.is_reduction_root.push_back(
        IsRealReductionHero(root.instruction(), *hero));
    if (result.is_reduction_root.back()) {
      roots_with_reduction.insert(root);
    } else if (first_non_reduction_root) {
      disjoint_sets[*first_non_reduction_root].Merge(&disjoint_sets[root]);
    } else {
      first_non_reduction_root = root;
    }
  }

  std::vector<HloInstructionAdaptor> instructions;
  HloBfsConsumersFirstTraversal(
      roots, analysis.fusion(),
      [&](HloInstructionAdaptor consumer) {
        auto& consumer_reachable = reachable_outputs[consumer];
        for (auto producer : consumer.GetOperands()) {
          reachable_outputs[producer].insert(consumer_reachable.begin(),
                                             consumer_reachable.end());
        }
        instructions.push_back(consumer);
        return TraversalResult::kAdvance;
      },
      [&](HloInstructionAdaptor argument) {
        instructions.push_back(argument);
      });

  for (auto instr : instructions) {
    const auto& reachable = reachable_outputs[instr];
    std::vector<HloInstructionAdaptor> reached_output_ids;
    bool added_to_reduce = false;
    for (auto output : roots) {
      bool has_real_hero = roots_with_reduction.contains(output);
      if (has_real_hero &&
          (hlo_query::IsBroadcastedConstantOrScalar(instr.instruction()))) {
        if (added_to_reduce) {
          // Do not group more than one output reduce instructions through
          // broadcasted constants or scalars, as the recomputation should be
          // acceptable.
          VLOG(3) << "Skip broadcasted constant or scalar " << instr.ToString();
          continue;
        }
      }
      // Now group output instructions if they have common predecessors.
      if (reachable.contains(output)) {
        VLOG(3) << "Reaching " << output.ToString() << " from "
                << instr.ToString();
        reached_output_ids.push_back(output);
        if (has_real_hero) {
          added_to_reduce = true;
        }
      }
    }
    for (size_t j = 1; j < reached_output_ids.size(); ++j) {
      disjoint_sets[reached_output_ids[0]].Merge(
          &disjoint_sets[reached_output_ids[j]]);
    }
  }

  // Place output instructions in the same set into the same group.
  ConstHloInstructionMap<std::vector<const HloInstruction*>> group_map;
  for (auto root : roots) {
    group_map[&disjoint_sets[root].Get().instruction()].push_back(
        &root.instruction());
  }

  result.grouped_roots.reserve(group_map.size());
  absl::c_for_each(group_map, [&](auto& it) {
    for (auto* root : it.second) {
      result.group_id_per_root[root_indices[root]] =
          result.grouped_roots.size();
    }
    result.grouped_roots.emplace_back(std::move(it.second));
  });
  return result;
}

}  // namespace

int ReductionInfo::GetRowsPerWarp() const {
  if (!is_row_reduction_) return 1;
  return RowReductionGetRowsPerWarp(
      tiling_.GetShape()[ReductionDimensions::kRowMinorReducedDimension]);
}

LaunchDimensions ReductionInfo::launch_dimensions() const {
  size_t blocks_y = groups_.grouped_roots.size();
  return {se::BlockDim(/*x=*/tiling_.GetNumBlocks(),
                       /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1),
          se::ThreadDim(/*x=*/tiling_.GetNumThreadsPerBlock(),
                        /*y=*/1, /*z=*/1)};
}

ReductionInfo ReductionInfo::Create(const HloFusionAnalysis& analysis,
                                    bool column_vectorization,
                                    bool adjust_tiling) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  Shape input_shape = hero_reduction->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  auto shape = reduction_dimensions.dimensions;
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << shape[0] << " " << shape[1] << " " << shape[2];
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);

  int64_t num_threads_y =
      reduction_dimensions.is_row_reduction ? 1 : WarpSize();
  int64_t rows_per_warp =
      reduction_dimensions.is_row_reduction
          ? RowReductionGetRowsPerWarp(
                shape[ReductionDimensions::kRowMinorReducedDimension])
          : 1;
  int64_t num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      if (rows_per_warp > 1) {
        return shape[ReductionDimensions::kRowMinorReducedDimension];
      }
      int64_t max_block_size =
          MinThreadsXRowReduction(hero_reduction->GetModule()->config());
      return std::min(
          max_block_size,
          RoundUpTo(
              CeilOfRatio(shape[ReductionDimensions::kRowMinorReducedDimension],
                          reduction_tiling
                              [ReductionDimensions::kRowMinorReducedDimension]),
              WarpSize()));
    }
    return WarpSize();
  }();

  // If we're limited by the size of the x dimension, add additional parallelism
  // in the y dimension. The code generator doesn't currently support
  // parallelizing the z dimension (major reduced dimensions). The general
  // recommendation is to use between 128 and 512 threads, so we just go for
  // 256. See https://forums.developer.nvidia.com/t/55529
  constexpr int64_t kThreadsPerBlockTarget = 256;
  if (reduction_dimensions.is_row_reduction &&
      num_threads_x * 2 <= kThreadsPerBlockTarget) {
    int64_t kept_size =
        reduction_dimensions.dimensions[ReductionDimensions::kRowKeptDimension];
    // Increase the size of the y dimension as long as there's remaining
    // parallelism.
    if (kept_size * num_threads_x <= kThreadsPerBlockTarget) {
      num_threads_y = kept_size;
      // num_threads_x is a power of two, but it may be less than 32. If dim_y
      // is also small, we may have to increase the bound so the total number of
      // threads is a multiple of 32.
      while ((num_threads_x * num_threads_y) % 32) ++num_threads_y;
    } else {
      num_threads_y = kThreadsPerBlockTarget / num_threads_x;
    }
  }

  int vector_size = GetVectorSize(analysis, reduction_dimensions, num_threads_y,
                                  num_threads_x, reduction_tiling,
                                  column_vectorization, adjust_tiling);
  bool tile_size_decreased = false;
  if (!reduction_dimensions.is_row_reduction && adjust_tiling) {
    // Adjust tile_y and vector size for column reduction if device occupancy is
    // low.
    std::tie(reduction_tiling, vector_size, tile_size_decreased) =
        AdjustColReductionTilingConfig(analysis, shape, reduction_tiling,
                                       num_threads_y, num_threads_x,
                                       vector_size);
  }

  absl::InlinedVector<int64_t, 4> num_threads{1, num_threads_y, num_threads_x};
  absl::InlinedVector<int64_t, 4> tiled_shape{shape[0], shape[1],
                                              shape[2] / vector_size};
  absl::InlinedVector<int64_t, 4> tile_per_thread{
      reduction_tiling[0], reduction_tiling[1],
      reduction_dimensions.is_row_reduction ? reduction_tiling[2] / vector_size
                                            : reduction_tiling[2]};
  if (rows_per_warp > 1) {
    // If we produce more than one element per thread, that means the reduced
    // dimension is small and it can't be tiled - we already have more threads
    // in a warp than the size of the reduced dimension. The code generator
    // doesn't currently support tiling the kept dimension, because it just
    // uses the thread ID as the coordinate.
    tile_per_thread[2] = 1;
  }
  if ((!reduction_dimensions.is_row_reduction && column_vectorization) ||
      vector_size != 1) {
    num_threads.push_back(1);  // The vector dimension is a loop.
    tiled_shape.push_back(vector_size);
    tile_per_thread.push_back(vector_size);
  }

  Tiling tiling(tiled_shape, tile_per_thread, num_threads,
                /*loops_to_unroll=*/{false, false, true, false});
  bool reduction_is_race_free = ReductionIsRaceFree(
      hero_reduction->GetModule()->config(), reduction_dimensions);
  // If tile_y is decreased, reduction is not race free.
  reduction_is_race_free = reduction_is_race_free && !tile_size_decreased;
  return ReductionInfo(analysis, tiling, reduction_dimensions.is_row_reduction,
                       reduction_is_race_free,
                       GroupDisjointReductions(analysis), hero_reduction);
}

std::optional<IndexingMap> ReductionInfo::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  if (!groups_.is_reduction_root[root_index]) {
    // Non-transpose roots are elementwise by definition.
    return ComputeThreadIdToInputIndexing(root_index, 0, ctx);
  }
  auto* root = analysis_.fusion_roots()[root_index];
  auto* hero = analysis_.fusion_heroes()[root_index];

  auto block_offsets = GetBlockOffsetsForTiling(tiling_, ctx);
  auto thread_ids = DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx),
                                             tiling_.GetThreadsPerBlock(),
                                             tiling_.GetThreadStrides());

  auto physical_shape = ShapeUtil::DeleteDimensions(hero->dimensions(),
                                                    hero->operand(0)->shape());
  std::vector<DimVar> dimension_ranges{
      {{0, tiling_.GetNumThreadsPerBlock() - 1}},
      {},
      {},
      {{0, tiling_.GetNumBlocks() - 1}},
      {{0, static_cast<int64_t>(groups_.grouped_roots.size() - 1)}},
      {},
  };

  constexpr int kRowKept = ReductionDimensions::kRowKeptDimension;
  constexpr int kRowMinorReduced =
      ReductionDimensions::kRowMinorReducedDimension;

  constexpr int kColMajorKept = ReductionDimensions::kColMajorKeptDimension;
  constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;
  constexpr int kColReduced = ReductionDimensions::kColReducedDimension;

  auto physical_index = [&]() {
    if (is_row_reduction_) {
      IndexingMap linear_index(
          mlir::AffineMap::get(
              6, 0, block_offsets.getResult(kRowKept) + thread_ids[kRowKept],
              ctx),
          dimension_ranges, /*range_vars=*/{}, /*rt_vars=*/{});
      int rows_per_warp = GetRowsPerWarp();
      if (rows_per_warp > 1) {
        linear_index.AddConstraint(
            thread_ids[kRowMinorReduced] % (WarpSize() / rows_per_warp),
            {0, 0});
      } else {
        linear_index.AddConstraint(thread_ids[kRowMinorReduced], {0, 0});
      }
      return ComposeIndexingMaps(
          linear_index, GetBitcastMap(ShapeUtil::MakeShape(
                                          PRED, {tiling_.GetShape()[kRowKept]}),
                                      physical_shape, ctx));
    }

    IndexingMap projected_index(
        mlir::AffineMap::get(
            6, 0,
            {block_offsets.getResult(kColMajorKept),
             block_offsets.getResult(kColMinorKept) + thread_ids[kColReduced]},
            ctx),
        dimension_ranges, /*range_vars=*/{}, /*rt_vars=*/{});

    projected_index.AddConstraint(
        mlir::getAffineDimExpr(
            KernelFusionInterface::kIndexingMapThreadIdxDims[0], ctx) %
            WarpSize(),
        {0, 0});
    if (!is_row_reduction_) {
      projected_index.AddConstraint(
          projected_index.GetAffineMap().getResult(1),
          {0, tiling_.GetShape()[ReductionDimensions::kColMinorKeptDimension] -
                  1});
    }

    return ComposeIndexingMaps(
        projected_index,
        GetBitcastMap(ShapeUtil::DeleteDimension(
                          ReductionDimensions::kColReducedDimension,
                          tiling_.GetXlaShape()),
                      physical_shape, ctx));
  }();

  auto map = ComposeIndexingMaps(
      physical_index,
      GetBitcastMap(FirstShape(hero->shape()), FirstShape(root->shape()), ctx));

  int group_index = groups_.group_id_per_root[root_index];
  map.AddConstraint(
      mlir::getAffineDimExpr(KernelFusionInterface::kIndexingMapBlockIdxDims[1],
                             ctx),
      {group_index, group_index});
  return map;
}

std::optional<IndexingMap> ReductionInfo::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  auto* hero = analysis_.fusion_heroes()[root_index];
  if (groups_.is_reduction_root[root_index] &&
      hero_operand_index >= hero->operand_count() / 2) {
    // We don't have indexing for the init values.
    return std::nullopt;
  }

  auto map = ComposeIndexingMaps(
      GetIndexingMapForTiling(tiling_, ctx),
      GetBitcastMap(tiling_.GetXlaShape(),
                    hero->operand(hero_operand_index)->shape(), ctx));
  // Only threads with the right y block index actually do anything for this
  // root.
  int group_index = groups_.group_id_per_root[root_index];
  map.AddConstraint(
      mlir::getAffineDimExpr(KernelFusionInterface::kIndexingMapBlockIdxDims[1],
                             ctx),
      {group_index, group_index});
  return map;
}

}  // namespace gpu
}  // namespace xla
