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
#include "xla/service/gpu/fusions/reduction_mlir.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputation;
using mlir_converter::PartitionedComputations;

struct MlirReductionFusion::EmitterState {
  // Uses the given indexing map to reduce a subset of the inputs in a single
  // thread. The subset may be a single element.
  absl::StatusOr<SmallVector<Value>> EmitPerThreadReducedElements(
      const IndexingMap& input_indexing, const HloInstruction* hero,
      ValueRange inits, const ReductionInfo& reduction_info);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  SmallVector<Value> AllocateSharedTiles(const HloInstruction* hero,
                                         absl::Span<const int64_t> shape);

  SmallVector<Value> FusionParams() {
    return ValueRange(entry_function.getArguments().take_front(
        fusion.fused_parameters().size()));
  }

  const MlirReductionFusion& owner;
  mlir::func::FuncOp entry_function;
  const HloFusionInstruction& fusion;
  const PartitionedComputations& computations;
  const mlir_converter::CallTargetProvider& call_target;
  mlir::ImplicitLocOpBuilder builder;
};

MlirReductionFusion::MlirReductionFusion(const HloFusionAnalysis& analysis)
    : ReductionFusionBase(analysis) {
  for (auto [index, hero] : llvm::enumerate(analysis.fusion_heroes())) {
    if (reduction_info().GetGroups().is_reduction_root[index]) {
      reduction_roots_[hero].push_back(index);
    }
  }

  for (const auto& [hero, _] : reduction_roots_) {
    reduction_heroes_.push_back(hero);
  }
}

int MlirReductionFusion::elements_store_per_thread() const {
  if (!reduction_info().IsRowReduction()) {
    return reduction_info()
        .GetTiling()
        .GetThreadTileSize()[ReductionDimensions::kVectorizedDimension];
  }
  return 1;
}

bool MlirReductionFusion::IsSupported(const HloFusionAnalysis& analysis) {
  auto info = ReductionInfo::Create(analysis);
  return info.GetGroups().grouped_roots.size() == 1 &&
         !absl::c_linear_search(info.GetGroups().is_reduction_root, false) &&
         info.IsRaceFree();
}

std::vector<const HloInstruction*>
MlirReductionFusion::GetInstructionsWithCustomCodegen(
    const HloFusionInstruction& fusion) const {
  return reduction_heroes_;
}

absl::Status MlirReductionFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  // Reduction groups will probably be implemented in a separate pass, since
  // they share nothing by definition.
  TF_RET_CHECK(reduction_info().GetGroups().grouped_roots.size() == 1)
      << "Only one reduction group is supported.";
  EmitterState state{*this,        entry_function,
                     fusion,       computations,
                     call_targets, {entry_function.getLoc(), entry_function}};
  state.builder.setInsertionPointToStart(entry_function.addEntryBlock());
  return EmitReduction(state);
}

absl::Status MlirReductionFusion::EmitReduction(EmitterState& state) const {
  CHECK(IsSupported(analysis()))
      << "Attempting to output code for an unsupported reduction";
  auto& builder = state.builder;
  const auto& tiling = reduction_info().GetTiling();

  // The number of warps working on one element in a row reduction.
  int num_warps_row = tiling.GetThreadsPerBlock()
                          [ReductionDimensions::kRowMinorReducedDimension] /
                      WarpSize();
  int col_vec_size = 1;
  if (!reduction_info().IsRowReduction()) {
    col_vec_size =
        reduction_info()
            .GetTiling()
            .GetThreadTileSize()[ReductionDimensions::kVectorizedDimension];
  }
  auto ctx = state.entry_function.getContext();

  auto zero = builder.create<mlir::arith::ConstantIndexOp>(0);
  auto lane_id = builder.create<mlir::gpu::LaneIdOp>();
  auto is_first_lane = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, lane_id, zero);
  auto thread_id = EmitThreadId(builder, 0);
  auto block_id = EmitBlockId(builder, 0);
  Value cstTrue = builder.create<mlir::arith::ConstantOp>(
      builder.getIntegerAttr(builder.getI1Type(), 1));

  auto thread_ids = mlir_converter::ApplyAffineMap(
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0,
          DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx),
                                   tiling.GetThreadsPerBlock(),
                                   tiling.GetThreadStrides()),
          ctx),
      {thread_id}, {}, builder);
  SmallVector<Value> thread_and_block_indices{thread_id, zero, zero,
                                              block_id,  zero, zero};

  auto warp_id = builder.create<mlir::arith::DivUIOp>(
      reduction_info().IsRowReduction()
          ? thread_ids[ReductionDimensions::kRowMinorReducedDimension]
          : thread_id,
      builder.create<mlir::arith::ConstantIndexOp>(WarpSize()));

  auto output_args = state.entry_function.getArguments().drop_front(
      state.fusion.fused_parameters().size());

  std::vector<int64_t> shared_tile_size;
  SmallVector<Value> shared_write_indices;
  SmallVector<Value> shared_read_indices;
  Value shared_write_condition = cstTrue;
  Value shared_read_condition = cstTrue;
  if (!reduction_info().IsRowReduction()) {
    shared_tile_size = {WarpSize(), WarpSize() + 1};
    shared_write_indices = {lane_id, warp_id};
    shared_read_indices = {warp_id, lane_id};
  } else if (reduction_info().GetRowsPerWarp() == 1 && num_warps_row > 1) {
    auto kKept = ReductionDimensions::kRowKeptDimension;
    shared_tile_size = {tiling.GetThreadsPerBlock()[kKept], num_warps_row};
    shared_write_condition = is_first_lane;
    shared_read_condition = builder.create<mlir::arith::CmpIOp>(
        mlir::arith::CmpIPredicate::ult,
        thread_ids[ReductionDimensions::kRowMinorReducedDimension],
        builder.create<mlir::arith::ConstantIndexOp>(num_warps_row));
    shared_write_indices = {thread_ids[kKept], warp_id};
    shared_read_indices = {thread_ids[kKept], lane_id};
  }
  bool use_shared = !shared_tile_size.empty();

  auto output_indexing = ComputeThreadIdToOutputIndexing(0, ctx);
  Value thread_has_output;
  if (reduction_info().IsRowReduction()) {
    thread_has_output = mlir_converter::CheckConstraints(
        *output_indexing, thread_and_block_indices, {}, builder);
  } else {
    // Has checked (dim % (vec_size * num_threads_x)) == 0.
    thread_has_output = mlir_converter::CheckConstraints(
        *output_indexing, thread_and_block_indices, {zero}, builder);
  }

  llvm::DenseMap<const HloInstruction*, SmallVector<Value>> inits;
  for (auto [index, hero] : llvm::enumerate(reduction_heroes_)) {
    int num_inputs = hero->operand_count() / 2;
    const auto& computation =
        state.computations.FindPartitionedComputation(hero->parent());
    inits[hero] = ProvideParameterRange(
        computation.FindSubgraph(hero), hero, num_inputs, num_inputs, {},
        state.call_target, state.entry_function, builder);
    if (!reduction_info().IsRowReduction()) {
      for (int i = 1; i < col_vec_size; i++) {
        auto init_values = ProvideParameterRange(
            computation.FindSubgraph(hero), hero, num_inputs, num_inputs, {},
            state.call_target, state.entry_function, builder);
        inits[hero].append(init_values.begin(), init_values.end());
      }
    }
  }

  auto evaluate_epilogue =
      [&](SmallVector<SmallVector<Value>> results,
          SmallVector<Value> output_indices) -> mlir::ValueRange {
    if (!state.computations.epilogue()) {
      return results.front();
    }

    llvm::SmallVector<Value> hero_values;
    for (const auto& result : results) {
      CHECK(result.size() == 1)
          << "Epilogue fusions are not supported with variadic reduce.";
      hero_values.push_back(result.front());
    }
    return EmitEpilogue(state.computations, state.entry_function, hero_values,
                        output_indices, builder);
  };

  SmallVector<Value> updated_outputs;
  SmallVector<llvm::SmallVector<Value>> results;
  for (auto* hero : reduction_heroes_) {
    auto input_indexing = ComputeThreadIdToInputIndexing(
        reduction_roots_.at(hero).front(), 0, ctx);
    TF_ASSIGN_OR_RETURN(auto accumulated, state.EmitPerThreadReducedElements(
                                              *input_indexing, hero,
                                              inits[hero], reduction_info()));

    // In row reductions, we can do a warp shuffle before writing to shared
    // memory. In column reductions, the members of the warp process different
    // output elements, so we need to transpose first.
    if (reduction_info().IsRowReduction()) {
      auto reducer = state.GetReducer(hero);
      int max_dist = WarpSize() / 2 / reduction_info().GetRowsPerWarp();
      accumulated =
          builder.create<ShuffleReduceOp>(reducer, accumulated, max_dist)
              .getResults();
    }

    results.push_back(accumulated);
  }

  if (use_shared) {
    // Write results to shared memory.
    for (auto [hero, result] : llvm::zip(reduction_heroes_, results)) {
      SmallVector<Value> dest;
      int reduced_number = hero->operand_count() / 2;
      for (auto [index, value] : llvm::enumerate(result)) {
        if (index == 0 || (!reduction_info().IsRowReduction() &&
                           index % reduced_number == 0)) {
          dest = state.AllocateSharedTiles(hero, shared_tile_size);
        }
        updated_outputs.push_back(builder.create<PredicatedInsertOp>(
            shared_write_condition, value, dest[index % reduced_number],
            shared_write_indices));
      }
    }
  } else {
    // Evaluate the epilogue, if there is one.
    auto output_indices = mlir_converter::ApplyAffineMap(
        output_indexing->GetAffineMap(), thread_and_block_indices, {}, builder);
    auto result_scalars = evaluate_epilogue(results, output_indices);
    for (auto [value, output] : llvm::zip(result_scalars, output_args)) {
      updated_outputs.push_back(builder.create<PredicatedInsertOp>(
          thread_has_output, value, output, output_indices));
    }
    builder.create<mlir::func::ReturnOp>(updated_outputs);
    return absl::OkStatus();
  }

  // Wait for the entire tile to be written.
  auto shared_tiles = builder
                          .create<SyncThreadsOp>(
                              mlir::TypeRange(updated_outputs), updated_outputs)
                          .getResults();
  auto write_outputs = [&](mlir::OpBuilder then_builder, mlir::Location loc) {
    SmallVector<SmallVector<SmallVector<Value>>> current_results(1);
    llvm::SmallVector<llvm::SmallVector<Value>> outputs(1);
    if (!reduction_info().IsRowReduction()) {
      current_results.resize(col_vec_size);
      outputs.resize(col_vec_size);
    }
    mlir::ImplicitLocOpBuilder b(loc, then_builder);
    int tile_index = 0;
    llvm::SmallVector<Value> updated_outputs;
    for (auto* hero : reduction_heroes_) {
      // Load from shared memory.
      SmallVector<SmallVector<Value>> reduced(1);
      if (!reduction_info().IsRowReduction()) {
        reduced.resize(col_vec_size);
      }
      int reduced_number = hero->operand_count() / 2;
      for (auto [index, init] : llvm::enumerate(inits[hero])) {
        // If a warp didn't write anything, use the init values instead.
        reduced[index / reduced_number].push_back(
            b.create<PredicatedExtractOp>(shared_read_condition, init,
                                          shared_tiles[tile_index++],
                                          shared_read_indices)
                .getResult());
      }

      for (auto [index, elems_in_reduced] : llvm::enumerate(reduced)) {
        auto reduced_result =
            builder
                .create<ShuffleReduceOp>(state.GetReducer(hero),
                                         elems_in_reduced, WarpSize() / 2)
                .getResults();
        current_results[index].push_back(reduced_result);
      }
    }

    SmallVector<SmallVector<Value>> result_scalars;
    SmallVector<SmallVector<Value>> output_offsets;
    if (reduction_info().IsRowReduction()) {
      output_offsets.push_back(mlir_converter::ApplyAffineMap(
          output_indexing->GetAffineMap(), thread_and_block_indices, {},
          builder));
      result_scalars.push_back(
          evaluate_epilogue(current_results.front(), output_offsets.front()));
    } else {
      for (int vec_dim = 0; vec_dim < col_vec_size; ++vec_dim) {
        output_offsets.push_back(mlir_converter::ApplyAffineMap(
            output_indexing->GetAffineMap(), thread_and_block_indices,
            {builder.create<mlir::arith::ConstantIndexOp>(vec_dim)}, builder));
        result_scalars.push_back(evaluate_epilogue(current_results[vec_dim],
                                                   output_offsets[vec_dim]));
      }
    }

    for (auto [elems_in_results, output_offset] :
         llvm::zip(result_scalars, output_offsets)) {
      for (auto [output_value, dest] :
           llvm::zip(elems_in_results, output_args)) {
        updated_outputs.push_back(b.create<PredicatedInsertOp>(
            thread_has_output, output_value, dest, output_offset));
      }
    }
    b.create<mlir::scf::YieldOp>(loc, updated_outputs);
  };

  auto warp_writes = reduction_info().IsRowReduction()
                         ? builder.create<mlir::arith::CmpIOp>(
                               mlir::arith::CmpIPredicate::eq, warp_id, zero)
                         : cstTrue;
  auto written = builder.create<mlir::scf::IfOp>(
      warp_writes, write_outputs, [&](mlir::OpBuilder b, mlir::Location loc) {
        SmallVector<Value> return_results = ValueRange(output_args);
        if (!reduction_info().IsRowReduction()) {
          for (int i = 1; i < col_vec_size; ++i) {
            return_results.append(output_args.begin(), output_args.end());
          }
        }
        b.create<mlir::scf::YieldOp>(loc, return_results);
      });
  builder.create<mlir::func::ReturnOp>(written.getResults());

  return absl::OkStatus();
}

absl::StatusOr<SmallVector<Value>>
MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    const IndexingMap& input_indexing, const HloInstruction* hero,
    ValueRange inits, const ReductionInfo& reduction_info) {
  int nested_level = reduction_info.IsRowReduction() ? 0 : 1;
  auto body_builder = [&](ValueRange outputs, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto indices = mlir_converter::ApplyAffineMap(
        input_indexing.GetAffineMap(), dim_values, symbol_values, builder);
    auto operands = FusionParams();
    absl::c_copy(indices, std::back_inserter(operands));
    auto values = ProvideParameterRange(computations.FindSubgraph(hero), hero,
                                        0, hero->operand_count() / 2, indices,
                                        call_target, entry_function, builder);

    SmallVector<Value> reduce_args = outputs;
    reduce_args.append(values.begin(), values.end());
    auto results =
        builder.create<PureCallOp>(GetReducer(hero), reduce_args).getResults();
    return results;
  };

  if (reduction_info.IsRowReduction()) {
    return owner.EmitThreadLoopNest(builder, inits, input_indexing,
                                    body_builder);
  } else {
    auto column_reduction_body =
        [&](ValueRange outputs, ValueRange dim_values,
            ValueRange symbol_values) -> SmallVector<Value> {
      SmallVector<Value> results, symbols = symbol_values;
      int reduced_number = hero->operand_count() / 2;
      int vec_size =
          reduction_info.GetTiling()
              .GetThreadTileSize()[ReductionDimensions::kVectorizedDimension];
      for (int dim = 0; dim < vec_size; dim++) {
        symbols.push_back(builder.create<mlir::arith::ConstantIndexOp>(dim));
        auto result =
            body_builder(outputs.slice(dim * reduced_number, reduced_number),
                         dim_values, symbols);
        results.append(result.begin(), result.end());
        symbols.pop_back();
      }
      return results;
    };
    return owner.EmitThreadLoopNest(builder, inits, input_indexing,
                                    column_reduction_body, /*nested_level=*/1);
  }
}

SmallVector<Value> MlirReductionFusion::EmitterState::AllocateSharedTiles(
    const HloInstruction* hero, absl::Span<const int64_t> shape) {
  SmallVector<Value> tiles;
  for (int i = 0; i < hero->operand_count() / 2; ++i) {
    tiles.push_back(
        builder.create<AllocateSharedOp>(mlir_converter::TensorShapeToMlirType(
            ShapeUtil::MakeShapeWithDescendingLayout(
                hero->operand(i)->shape().element_type(), shape),
            builder)));
  }
  return tiles;
}

}  // namespace gpu
}  // namespace xla
