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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
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

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputations;

using HloValueMap =
    llvm::DenseMap<const HloInstruction*, llvm::SmallVector<Value>>;

struct MlirReductionFusion::EmitterState {
  // Uses the given indexing map to reduce a subset of the inputs in a single
  // thread. The subset may be a single element.

  HloValueMap EmitPerThreadReducedElements(const HloValueMap& inits);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  SmallVector<Value> AllocateSharedTiles(const HloInstruction* hero,
                                         absl::Span<const int64_t> shape);

  SmallVector<Value> FusionParams() {
    return ValueRange(entry_function.getArguments().take_front(
        fusion.fused_parameters().size()));
  }

  int OutputIndex(const HloInstruction* root, int result_index) {
    if (root->shape().IsTuple()) {
      // If the root is a tuple, that means we're dealing with a variadic
      // reduction. Variadic reductions have no epilogues or side outputs.
      return result_index;
    }

    CHECK_EQ(result_index, 0);
    return absl::c_find(owner.analysis().fusion_roots(), root) -
           owner.analysis().fusion_roots().begin();
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
  absl::flat_hash_set<const HloInstruction*> seen_heroes;
  for (auto [root, hero, is_reduction] :
       llvm::zip(analysis.fusion_roots(), analysis.fusion_heroes(),
                 reduction_info().GetGroups().is_reduction_root)) {
    (is_reduction ? reduction_roots_ : side_output_roots_).push_back(root);
    if (is_reduction && seen_heroes.insert(hero).second) {
      reduction_heroes_.push_back(hero);
    }
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
  return info.GetGroups().grouped_roots.size() == 1 && info.IsRaceFree();
}

std::optional<mlir_converter::EpilogueSpecification>
MlirReductionFusion::GetEpilogue(const HloFusionInstruction& fusion,
                                 mlir::MLIRContext* mlir_context) const {
  return mlir_converter::EpilogueSpecification::FromOutputIndexing(
      analysis(), reduction_heroes_, reduction_roots_, *this, mlir_context);
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
  Value cst_true = builder.create<mlir::arith::ConstantOp>(
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

  std::vector<int64_t> shared_tile_size;
  SmallVector<Value> shared_write_indices;
  SmallVector<Value> shared_read_indices;
  Value shared_write_condition = cst_true;
  Value shared_read_condition = cst_true;
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

  Value thread_has_output;
  if (reduction_info().IsRowReduction()) {
    thread_has_output = mlir_converter::CheckConstraints(
        *ComputeThreadIdToOutputIndexing(0, ctx), thread_and_block_indices, {},
        builder);
  } else {
    // Has checked (dim % (vec_size * num_threads_x)) == 0.
    thread_has_output = mlir_converter::CheckConstraints(
        *ComputeThreadIdToOutputIndexing(0, ctx), thread_and_block_indices,
        {zero}, builder);
  }

  llvm::DenseMap<const HloInstruction*, SmallVector<Value>> inits;
  for (auto [index, hero] : llvm::enumerate(reduction_heroes_)) {
    int num_inputs = hero->operand_count() / 2;
    const auto& computation =
        state.computations.FindPartitionedComputation(hero->parent());
    inits[hero] =
        ProvideParameterRange(computation, hero, num_inputs, num_inputs, {},
                              state.call_target, state.entry_function, builder);
  }

  auto evaluate_epilogue =
      [&](const HloValueMap& results, llvm::SmallVector<Value> outputs,
          SmallVector<Value> symbols = {}) -> llvm::SmallVector<Value> {
    llvm::SmallVector<Value> indices = EmitThreadAndBlockIds(builder);
    if (state.computations.epilogue()) {
      llvm::SmallVector<Value> hero_values(reduction_heroes_.size());
      const auto& epilogue = *state.computations.epilogue();
      for (auto hero : reduction_heroes_) {
        const auto& result = results.at(hero);
        CHECK(result.size() == 1)
            << "Epilogue fusions are not supported with variadic reduce.";
        hero_values[epilogue.injected_values.at(hero)] = result.front();
      }

      int num_symbols =
          state.computations.epilogue()->root_indexing.front().getNumSymbols();
      CHECK(symbols.empty() || symbols.size() == num_symbols)
          << "symbols should be empty or match epilogue symbos number.";
      auto epilogue_indices = indices;
      if (symbols.empty()) {
        for (int i = 0; i < num_symbols; ++i) {
          epilogue_indices.push_back(zero);
        }
      } else {
        epilogue_indices.append(symbols.begin(), symbols.end());
      }
      llvm::SmallVector<Value> epilogue_values =
          EmitEpilogue(state.computations, state.entry_function, hero_values,
                       epilogue_indices, builder);
      for (auto [index, root] : llvm::enumerate(epilogue.roots)) {
        auto& output = outputs[state.OutputIndex(root, 0)];
        auto output_indices = mlir_converter::ApplyAffineMap(
            epilogue.root_indexing[index], indices, symbols, builder);
        output = builder.create<PredicatedInsertOp>(
            thread_has_output, epilogue_values[index], output, output_indices);
      }
    } else {
      CHECK_EQ(reduction_roots_.size(), 1);
      auto* reduction = reduction_roots_.front();
      int root_index = absl::c_find(analysis().fusion_roots(), reduction) -
                       analysis().fusion_roots().begin();
      auto output_indices = mlir_converter::ApplyAffineMap(
          ComputeThreadIdToOutputIndexing(root_index, builder.getContext())
              ->GetAffineMap(),
          indices, symbols, builder);

      for (auto [result_index, result] :
           llvm::enumerate(results.at(reduction))) {
        auto& output = outputs[state.OutputIndex(reduction, result_index)];
        output = builder.create<PredicatedInsertOp>(thread_has_output, result,
                                                    output, output_indices);
      }
    }
    return outputs;
  };

  auto accumulated = state.EmitPerThreadReducedElements(inits);
  llvm::SmallVector<Value> outputs =
      mlir::ValueRange(state.entry_function.getArguments().drop_front(
          state.fusion.fused_parameters().size()));
  int outputs_size = static_cast<int>(outputs.size());
  SmallVector<Value> return_outputs = outputs;
  if (!reduction_info().IsRowReduction()) {
    for (int i = 1; i < col_vec_size; ++i) {
      return_outputs.append(outputs);
    }
  }
  for (auto root : side_output_roots_) {
    for (auto [index, out] : llvm::enumerate(accumulated[root])) {
      return_outputs[index * outputs_size + state.OutputIndex(root, 0)] = out;
    }
  }

  // In row reductions, we can do a warp shuffle before writing to shared
  // memory. In column reductions, the members of the warp process different
  // output elements, so we need to transpose first.
  if (reduction_info().IsRowReduction()) {
    for (auto* hero : reduction_heroes_) {
      auto reducer = state.GetReducer(hero);
      int max_dist = WarpSize() / 2 / reduction_info().GetRowsPerWarp();
      accumulated[hero] =
          builder.create<ShuffleReduceOp>(reducer, accumulated[hero], max_dist)
              .getResults();
    }
  }

  if (!use_shared) {
    builder.create<mlir::func::ReturnOp>(
        evaluate_epilogue(accumulated, return_outputs, /*symbols=*/{}));
    return absl::OkStatus();
  }

  SmallVector<Value> shared_tiles;
  // Write results to shared memory.
  for (auto hero : reduction_heroes_) {
    const auto& result = accumulated[hero];
    int reduced_number = hero->operand_count() / 2;
    SmallVector<Value> dest;
    for (auto [index, value] : llvm::enumerate(result)) {
      if (index == 0 ||
          (!reduction_info().IsRowReduction() && index % reduced_number == 0)) {
        dest = state.AllocateSharedTiles(hero, shared_tile_size);
      }
      shared_tiles.push_back(builder.create<PredicatedInsertOp>(
          shared_write_condition, value, dest[index % reduced_number],
          shared_write_indices));
    }
  }

  // Wait for the entire tile to be written.
  auto synced_tiles =
      builder.create<SyncThreadsOp>(mlir::TypeRange(shared_tiles), shared_tiles)
          .getResults();
  auto write_outputs = [&](mlir::OpBuilder then_builder, mlir::Location loc) {
    SmallVector<HloValueMap> accumulated(/*Size=*/1);
    if (!reduction_info().IsRowReduction()) {
      accumulated.resize(col_vec_size);
    }
    mlir::ImplicitLocOpBuilder b(loc, then_builder);
    int tile_index = 0;
    for (auto* hero : reduction_heroes_) {
      // Load from shared memory.
      SmallVector<SmallVector<Value>> reduced(/*Size=*/1);
      if (!reduction_info().IsRowReduction()) {
        reduced.resize(col_vec_size);
      }
      int reduced_number = hero->operand_count() / 2;
      int total_size = reduction_info().IsRowReduction()
                           ? reduced_number
                           : reduced_number * col_vec_size;
      for (int id = 0; id < total_size; id++) {
        // If a warp didn't write anything, use the init values instead.
        reduced[id / reduced_number].push_back(
            b.create<PredicatedExtractOp>(
                 shared_read_condition, inits[hero][id % reduced_number],
                 synced_tiles[tile_index++], shared_read_indices)
                .getResult());
      }

      for (auto [index, elems_in_reduced] : llvm::enumerate(reduced)) {
        accumulated[index][hero] =
            builder
                .create<ShuffleReduceOp>(state.GetReducer(hero),
                                         elems_in_reduced, WarpSize() / 2)
                .getResults();
      }
    }

    if (reduction_info().IsRowReduction()) {
      b.create<mlir::scf::YieldOp>(
          loc, evaluate_epilogue(accumulated.front(), return_outputs));
    } else {
      SmallVector<Value> final_outputs;
      for (int vec_dim = 0; vec_dim < col_vec_size; ++vec_dim) {
        auto vec_symbol = builder.create<mlir::arith::ConstantIndexOp>(vec_dim);
        auto outputs_begin = return_outputs.begin() + vec_dim * outputs_size;
        auto outputs_end =
            return_outputs.begin() + (vec_dim + 1) * outputs_size;
        final_outputs.append(evaluate_epilogue(accumulated[vec_dim],
                                               {outputs_begin, outputs_end},
                                               /*symbols=*/{vec_symbol}));
      }
      b.create<mlir::scf::YieldOp>(loc, final_outputs);
    }
  };

  auto warp_writes = reduction_info().IsRowReduction()
                         ? builder.create<mlir::arith::CmpIOp>(
                               mlir::arith::CmpIPredicate::eq, warp_id, zero)
                         : cst_true;
  auto written = builder.create<mlir::scf::IfOp>(
      warp_writes, write_outputs, [&](mlir::OpBuilder b, mlir::Location loc) {
        b.create<mlir::scf::YieldOp>(loc, return_outputs);
      });
  builder.create<mlir::func::ReturnOp>(written.getResults());

  return absl::OkStatus();
}

HloValueMap MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    const HloValueMap& inits) {
  const auto& reduction_info = owner.reduction_info();
  const auto& tiling = owner.reduction_info().GetTiling();
  auto tile_indexing = GetIndexingMapForTiling(tiling, builder.getContext());

  SmallVector<Value> iter_arg_inits;
  ValueRange output_args = entry_function.getArguments().drop_front(
      fusion.fused_parameters().size());
  int repeats = 1;
  if (!owner.reduction_info().IsRowReduction()) {
    repeats =
        reduction_info.GetTiling()
            .GetThreadTileSize()[ReductionDimensions::kVectorizedDimension];
  }
  for (int cur = 0; cur < repeats; cur++) {
    for (auto [is_reduction, hero, output] :
         llvm::zip(owner.reduction_info().GetGroups().is_reduction_root,
                   owner.analysis().fusion_heroes(), output_args)) {
      if (is_reduction) {
        iter_arg_inits.append(inits.at(hero));
      } else {
        iter_arg_inits.push_back(output);
      }
    }
  }

  const auto& computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());

  auto body_builder = [&](ValueRange iter_args, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto tile_indices = mlir_converter::ApplyAffineMap(
        tile_indexing.GetAffineMap(), dim_values, symbol_values, builder);

    llvm::SmallVector<Value> results;
    int start = 0;
    for (auto [is_reduction, hero] :
         llvm::zip(owner.reduction_info().GetGroups().is_reduction_root,
                   owner.analysis().fusion_heroes())) {
      const xla::Shape& input_shape =
          is_reduction ? hero->operand(0)->shape() : hero->shape();
      llvm::SmallVector<Value> input_indices = mlir_converter::ApplyAffineMap(
          GetBitcastMap(tiling.GetXlaShape(), input_shape, builder.getContext())
              .GetAffineMap(),
          tile_indices, {}, builder);
      if (is_reduction) {
        int num_outs = hero->operand_count() / 2;
        auto values = ProvideParameterRange(
            computations.FindPartitionedComputation(hero->parent()), hero, 0,
            num_outs, input_indices, call_target, entry_function, builder);
        SmallVector<Value> reduce_args = iter_args.slice(start, num_outs);
        reduce_args.append(values);
        absl::c_copy(builder.create<PureCallOp>(GetReducer(hero), reduce_args)
                         .getResults(),
                     std::back_inserter(results));
        start += num_outs;
      } else {
        auto* root_tuple = fusion.fused_expression_root();
        Value value = mlir_converter::ProvideParameter(
            computation, root_tuple, root_tuple->operand_index(hero),
            input_indices, call_target, entry_function, builder);
        results.push_back(builder.create<mlir::tensor::InsertOp>(
            value, iter_args[start], input_indices));
        ++start;
      }
    }
    return results;
  };

  SmallVector<Value> results;
  if (owner.reduction_info().IsRowReduction()) {
    results = owner.EmitThreadLoopNest(builder, iter_arg_inits, tile_indexing,
                                       body_builder);
  } else {
    auto column_reduction_body =
        [&](ValueRange iter_args, ValueRange dim_values,
            ValueRange symbol_values) -> SmallVector<Value> {
      SmallVector<Value> results, symbols = symbol_values;
      int vec_size = repeats;
      int args_per_iter = static_cast<int>(iter_arg_inits.size()) / repeats;
      for (int dim = 0; dim < vec_size; dim++) {
        symbols.push_back(builder.create<mlir::arith::ConstantIndexOp>(dim));
        auto result =
            body_builder(iter_args.slice(dim * args_per_iter, args_per_iter),
                         dim_values, symbols);
        results.append(result.begin(), result.end());
        symbols.pop_back();
      }
      return results;
    };
    results =
        owner.EmitThreadLoopNest(builder, iter_arg_inits, tile_indexing,
                                 column_reduction_body, /*nested_level=*/1);
  }
  mlir::ValueRange result_range = results;
  HloValueMap results_per_hero;
  for (int cur = 0; cur < repeats; cur++) {
    absl::flat_hash_set<const HloInstruction*> heros;
    for (auto [is_reduction, hero] :
         llvm::zip(owner.reduction_info().GetGroups().is_reduction_root,
                   owner.analysis().fusion_heroes())) {
      if (heros.find(hero) != heros.end()) {
        continue;
      }
      int num_outs =
          hero->shape().IsTuple() ? hero->shape().tuple_shapes_size() : 1;
      auto current_range = result_range.take_front(num_outs);
      results_per_hero[hero].append(current_range.begin(), current_range.end());
      result_range = result_range.drop_front(num_outs);
      heros.insert(hero);
    }
  }
  return results_per_hero;
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
