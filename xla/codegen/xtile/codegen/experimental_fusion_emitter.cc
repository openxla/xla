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

#include "xla/codegen/xtile/codegen/experimental_fusion_emitter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/tiling/experimental/reshape_analysis.h"
#include "xla/codegen/tiling/experimental/scheduling.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/xtile/codegen/dot_algorithms.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"  // IWYU pragma: keep
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::xtile {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::stream_executor::GpuComputeCapability;

namespace arith = ::mlir::arith;
namespace stablehlo = ::mlir::stablehlo;
namespace ge = ::xla::gpu::experimental;

absl::StatusOr<std::vector<TensorValue>> EmitTiledComputation(
    EmitterContext& emitter_ctx, const ge::TiledHloRegion& region,
    absl::Span<const ge::TiledHloInstruction* const> roots);

absl::StatusOr<TensorValue> EmitTiledHloInstruction(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_hlo);

Value MakeIndex(mlir::ImplicitLocOpBuilder& b, int64_t value) {
  return arith::ConstantIndexOp::create(b, value);
}

TensorValue Iota(mlir::ImplicitLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return stablehlo::IotaOp::create(b, type, /*iota_dimension=*/0);
}

template <typename T>
ArrayRef<T> MakeArrayRef(const absl::Span<const T> span) {
  return ArrayRef(span.data(), span.size());
}

absl::StatusOr<TensorValue> EmitAllReduce(
    EmitterContext& emitter_ctx, const HloAllReduceInstruction* all_reduce,
    const ge::TiledHloInstruction& tiled_all_reduce, ValueRange operands) {
  if (all_reduce->device_list()->replica_groups().empty()) {
    return Internal(
        "Triton emitting AllReduce without replica groups is not supported.");
  }

  llvm::SmallVector<int64_t> flattened_replica_group_ids;
  for (const auto& replica_group : all_reduce->replica_groups()) {
    for (const auto& replica_id : replica_group.replica_ids()) {
      flattened_replica_group_ids.push_back(replica_id);
    }
  }

  std::optional<int64_t> channel_handle = all_reduce->channel_id();
  bool use_global_device_ids = all_reduce->use_global_device_ids();

  ImplicitLocOpBuilder& b = emitter_ctx.b();
  ASSIGN_OR_RETURN(
      auto output_element_type,
      xtile::PrimitiveTypeToMlirType(b, all_reduce->shape().element_type()));
  ASSIGN_OR_RETURN(SmallVector<int64_t> tile_sizes,
                   tiled_all_reduce.tile().GetStaticTileSizes());
  auto output_type =
      mlir::RankedTensorType::get(tile_sizes, output_element_type);

  auto replica_groups_type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(all_reduce->replica_groups().size()),
       static_cast<int64_t>(
           all_reduce->replica_groups()[0].replica_ids_size())},
      b.getI64Type());
  auto replica_groups_attr = mlir::DenseIntElementsAttr::get(
      replica_groups_type, flattened_replica_group_ids);
  auto channel_handle_attr =
      channel_handle ? mlir::stablehlo::ChannelHandleAttr::get(b.getContext(),
                                                               *channel_handle,
                                                               /*type=*/0)
                     : nullptr;
  auto all_reduce_op = mlir::stablehlo::AllReduceOp::create(
      b, output_type, operands, replica_groups_attr, channel_handle_attr,
      use_global_device_ids);

  RETURN_IF_ERROR(EmitReduceComputation(b, all_reduce, all_reduce->to_apply(),
                                        all_reduce_op));
  return mlir::cast<TensorValue>(all_reduce_op.getResult(0));
}

absl::StatusOr<TensorValue> EmitBroadcast(
    mlir::ImplicitLocOpBuilder& b,
    const ge::TiledHloInstruction& tiled_broadcast, TensorValue input) {
  ASSIGN_OR_RETURN(SmallVector<int64_t> input_tile_shape,
                   tiled_broadcast.operand(0)->tile().GetStaticTileSizes());
  ASSIGN_OR_RETURN(SmallVector<int64_t> output_tile_shape,
                   tiled_broadcast.tile().GetStaticTileSizes());
  if (input_tile_shape.empty() && output_tile_shape.empty()) {
    return input;
  }
  TF_RET_CHECK(!output_tile_shape.empty());

  return xtile::BroadcastInDims(
      b, input, output_tile_shape,
      MakeArrayRef(tiled_broadcast.hlo()->dimensions()));
}

absl::StatusOr<TensorValue> EmitConcatenate(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_concat) {
  auto& b = emitter_ctx.b();
  const HloConcatenateInstruction* hlo_concat =
      ::xla::Cast<HloConcatenateInstruction>(tiled_concat.hlo());
  const int64_t concatenate_dimension = hlo_concat->concatenate_dimension();

  TF_RET_CHECK(tiled_concat.operands().size() ==
               tiled_concat.hlo_regions().size())
      << "Concatenate must have the same number of operands and regions";

  ASSIGN_OR_RETURN(SmallVector<int64_t> tile_sizes,
                   tiled_concat.tile().GetStaticTileSizes());
  int64_t concat_dim_tile_size = tile_sizes[concatenate_dimension];

  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(emitter_ctx, tiled_concat));
  RETURN_IF_ERROR(CheckConcatenateOperands(*hlo_concat, concat_dim_tile_size));
  ASSIGN_OR_RETURN(
      auto element_type,
      xtile::PrimitiveTypeToMlirType(b, hlo_concat->shape().element_type()));
  Type result_type = mlir::RankedTensorType::get(tile_sizes, element_type);
  // We will load and compute from a single operand, so we need to figure out
  // which one by looking at the offset within the concatenation dimension.
  Value concatenate_dimension_offset =
      tile_info.offsets()[concatenate_dimension];

  // It would have been nice to be able to use `scf::IndexSwitchOp`, but Triton
  // does not want to deal with the `Index` type, and does not support the op.
  // Instead, we generate a sequence of nested `scf::IfOp`s.
  SmallVector<mlir::scf::IfOp, 4> if_ops;
  int64_t limit = 0;
  for (const auto& [i, operand] : llvm::enumerate(tiled_concat.operands())) {
    // Write in the else branch of the previous if op if one exists.
    if (!if_ops.empty()) {
      b.setInsertionPointToStart(if_ops.back().elseBlock());
    }
    // Add an `if_op` if we have not reached the last operand. The last operand
    // directly populates the `else` block of the previous `if_op`.
    if (if_ops.size() < tiled_concat.operands().size() - 1) {
      limit += operand->hlo()->shape().dimensions()[concatenate_dimension];
      Value offset_limit = CreateConst(b, b.getIndexType(), limit);

      auto cond =
          arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                concatenate_dimension_offset, offset_limit);
      auto if_op =
          mlir::scf::IfOp::create(b, mlir::TypeRange(result_type), cond,
                                  /*withElseRegion=*/true);

      // Propagate the result from the nested `if_op` if we were already within
      // an `if_op`.
      if (!if_ops.empty()) {
        mlir::scf::YieldOp::create(b, if_op.getResult(0));
      }
      b.setInsertionPointToStart(if_op.thenBlock());
      if_ops.push_back(if_op);
    }
    const auto& region = tiled_concat.hlo_regions()[i];
    ASSIGN_OR_RETURN(std::vector<TensorValue> results,
                     EmitTiledComputation(emitter_ctx, region, region.roots()));
    TF_RET_CHECK(results.size() == 1)
        << "Concatenation region must have exactly one result"
        << results.size();
    TF_RET_CHECK(results[0].getType() == result_type)
        << "Region result type must match the concatenate result type";
    mlir::scf::YieldOp::create(b, results.back());
  }
  b.setInsertionPointAfter(if_ops.front());
  return mlir::cast<TensorValue>(if_ops.front().getResult(0));
}

// Computes and applies a mask to the reduction dimension of the dot operand
// passed as a parameter.
//
// Note: we currently assume that contracting_dimension_tile_index is an i32
// scalar.
absl::StatusOr<TensorValue> MaskDotOperand(
    mlir::ImplicitLocOpBuilder& b, const ge::TiledHloInstruction& dot_operand,
    TensorValue dot_operand_value, Value contracting_dimension_tile_index,
    int contraction_dimension_index) {
  llvm::ArrayRef<int64_t> tile_shape = dot_operand_value.getType().getShape();

  int64_t contracting_dimension_size =
      dot_operand.hlo()->shape().dimensions(contraction_dimension_index);
  int64_t tile_size = tile_shape[contraction_dimension_index];

  if (contracting_dimension_size % tile_size == 0) {
    return dot_operand_value;
  }

  // Only mask out tiles that we know to go beyond boundaries of the
  // contracting dimension---i.e. tiles whose index exceeds the number of
  // full tiles (tiles without padding).
  Type result_type = dot_operand_value.getType();
  Value tile_size_value = CreateConst(b, b.getI32Type(), tile_size);
  Value num_full_tiles = arith::DivSIOp::create(
      b, CreateConst(b, b.getI32Type(), contracting_dimension_size),
      tile_size_value);
  // if tile_index >= num_full_tiles...
  auto cond =
      arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                            contracting_dimension_tile_index, num_full_tiles);
  auto if_op = mlir::scf::IfOp::create(b, mlir::TypeRange(result_type), cond,
                                       /*withElseRegion=*/true);
  // then ...
  {
    b.setInsertionPointToStart(if_op.thenBlock());
    // indices =
    //   contracting_dimension_tile_index * tile_size + range(0, tile_size)
    // mask = indices < contracting_dimension_size
    // operand = select(broadcast(mask, operand.shape), operand, 0)
    Value tile_offset = arith::MulIOp::create(
        b, contracting_dimension_tile_index, tile_size_value);
    TensorValue range = Iota(b, tile_size);
    TensorValue broadcasted_tile_offset =
        xtile::Splat(b, tile_offset, {tile_size});
    Value indices = arith::AddIOp::create(b, range, broadcasted_tile_offset);

    Value boundary =
        CreateConst(b, b.getI32Type(), contracting_dimension_size, {tile_size});

    Value mask =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, indices, boundary);

    mask = xtile::BroadcastInDims(b, mlir::cast<TensorValue>(mask), tile_shape,
                                  {contraction_dimension_index});
    ASSIGN_OR_RETURN(
        auto element_type,
        PrimitiveTypeToMlirType(b, dot_operand.hlo()->shape().element_type()));

    TensorValue zero = CreateConst(b, element_type, 0.0f, tile_shape);

    Value masked_dot_operand =
        arith::SelectOp::create(b, mask, dot_operand_value, zero);
    mlir::scf::YieldOp::create(b, masked_dot_operand);
  }
  // else ...
  {
    b.setInsertionPointToStart(if_op.elseBlock());
    mlir::scf::YieldOp::create(b, dot_operand_value);
  }
  b.setInsertionPointAfter(if_op);
  return mlir::cast<TensorValue>(if_op.getResult(0));
}

// Returns the number of sequential dimensions in the HLO.
int64_t GetNumSequentialDimIds(const HloInstruction& hlo) {
  if (HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kScaledDot>(&hlo)) {
    const DotDimensionNumbers& dim_numbers = hlo.dot_dimension_numbers();
    return dim_numbers.lhs_contracting_dimensions().size();
  }
  if (HloPredicateIsOp<HloOpcode::kReduce>(&hlo)) {
    const HloReduceInstruction& reduce =
        *::xla::Cast<HloReduceInstruction>(&hlo);
    return reduce.dimensions().size();
  }
  return 0;
}

// Returns the positions of the sequential dimensions in the HLO.
SmallVector<int64_t> GetSequentialDimIds(const HloInstruction& hlo) {
  int64_t num_sequential_dims = GetNumSequentialDimIds(hlo);
  SmallVector<int64_t> sequential_dim_ids;
  int64_t output_rank = hlo.shape().dimensions().size();
  for (int64_t dim_id = output_rank, e = output_rank + num_sequential_dims;
       dim_id < e; ++dim_id) {
    sequential_dim_ids.push_back(dim_id);
  }
  return sequential_dim_ids;
}

// Returns the number of iterations of the loop over the contraction/reduction
// dimensions.
absl::StatusOr<SmallVector<int64_t>> GetSequentialLoopIterationCounts(
    const ge::TiledHloInstruction& tiled_hlo,
    ArrayRef<int64_t> sequential_dim_ids) {
  const HloInstruction& hlo = *tiled_hlo.hlo();
  if (sequential_dim_ids.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "No sequential dimensions found for the HLO", hlo.ToString()));
  }
  const ge::TilingSpace& tiling_space = tiled_hlo.tile().tiling_space();

  int64_t output_rank = hlo.shape().dimensions().size();
  SmallVector<int64_t> loop_iteration_counts;
  for (int64_t dim_id : sequential_dim_ids) {
    const ge::TilingSpace::DimensionInfo& dim_info =
        tiling_space.GetDimensionInfo(hlo, output_rank++);
    TF_RET_CHECK(dim_info.type ==
                 ge::TilingSpace::DimensionSemantics::kSequential)
        << "Expected a sequential dimension info for contracting dimension "
        << dim_id << " in op " << hlo.ToString();
    TF_RET_CHECK(dim_info.tile_size.has_value())
        << "Tile size is not set for contracting dimension ";
    loop_iteration_counts.push_back(
        CeilOfRatio(dim_info.dimension_size, *dim_info.tile_size));
  }
  return loop_iteration_counts;
}

// Emits dot instruction that has LHS and RHS as part of its region.
// Tiling analysis identifies instructions that belong to the dot and puts them
// inside of the dot's regions.
//
// To emit it we create a loop over the contracting dimension and emit the
// region of the dot inside:
//
// acc = [tile_m, tile_n] 0.0f
// for (k = 0 .. size_k / tile_k) {
//   <contents of the region, including lhs and rhs>
//   acc += dot(lhs, rhs)
// }
// c = acc
absl::StatusOr<TensorValue> EmitDot(EmitterContext& emitter_ctx,
                                    const ge::TiledHloInstruction& tiled_dot) {
  TF_RET_CHECK(tiled_dot.hlo_regions().size() == 1);
  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_dot.tile().GetStaticTileSizes());

  auto& b = emitter_ctx.b();
  const auto& dot = *::xla::Cast<HloDotInstruction>(tiled_dot.hlo());
  // The specific accumulator type to use may not correspond to the output type
  // of the dot. In particular, that is the case when an algorithm is specified
  // and the dot's output type does not match its expectations.
  ASSIGN_OR_RETURN(Type accumulator_type, xtile::GetDotAccumulatorType(b, dot));
  TensorValue accumulator =
      CreateConst(b, accumulator_type, 0.0f, padded_tile_sizes);

  SmallVector<int64_t> sequential_dim_ids =
      GetSequentialDimIds(*tiled_dot.hlo());
  ASSIGN_OR_RETURN(
      SmallVector<int64_t> loop_iteration_count,
      GetSequentialLoopIterationCounts(tiled_dot, sequential_dim_ids));
  TF_RET_CHECK(loop_iteration_count.size() == 1)
      << "Expected exactly one loop iteration count for dot";

  auto for_op = mlir::scf::ForOp::create(
      b,
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_count.front()),
      /*step=*/MakeIndex(b, 1), accumulator);

  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value iv = for_op.getInductionVar();
    Value iv_i32 = Cast(b, for_op.getInductionVar(), b.getI32Type());
    const ge::TilingSpace::DimensionInfo& dim_info =
        tiled_dot.tile().tiling_space().GetDimensionInfo(
            *tiled_dot.hlo(), sequential_dim_ids.front());
    TF_RET_CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(
        dim_info.id, iv, Interval{0, loop_iteration_count.front() - 1}));

    // Emit the dot region.
    const ge::TiledHloInstruction* lhs_operand = tiled_dot.operand(0);
    const ge::TiledHloInstruction* rhs_operand = tiled_dot.operand(1);
    ASSIGN_OR_RETURN(
        auto results,
        EmitTiledComputation(emitter_ctx, tiled_dot.hlo_regions().front(),
                             {lhs_operand, rhs_operand}));

    // Canonicalize LHS to match Triton's expectations.
    TensorValue lhs_tensor = results[0];
    int64_t lhs_contracting_dim_idx =
        dot.dot_dimension_numbers().lhs_contracting_dimensions(0);
    ASSIGN_OR_RETURN(lhs_tensor,
                     MaskDotOperand(b, *lhs_operand, lhs_tensor, iv_i32,
                                    lhs_contracting_dim_idx));

    // Canonicalize RHS to match Triton's expectations.
    TensorValue rhs_tensor = results[1];
    int64_t rhs_contracting_dim_idx =
        dot.dot_dimension_numbers().rhs_contracting_dimensions(0);
    ASSIGN_OR_RETURN(rhs_tensor,
                     MaskDotOperand(b, *rhs_operand, rhs_tensor, iv_i32,
                                    rhs_contracting_dim_idx));

    // Emit the partial dot.
    Value acc = for_op.getRegionIterArgs().front();
    ASSIGN_OR_RETURN(
        Value acc_next,
        xtile::EmitSingleTileDot(
            b, dot, xtile::DotOperands{lhs_tensor, rhs_tensor, acc}));
    mlir::scf::YieldOp::create(b, acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  ASSIGN_OR_RETURN(Type dot_output_type,
                   PrimitiveTypeToMlirType(b, dot.shape().element_type()));

  Value result = for_op.getResult(0);
  if (dot_output_type != accumulator_type) {
    result = Cast(b, result, dot_output_type);
  }
  return mlir::cast<TensorValue>(result);
}

// Emits scaled dot instruction that is not nested into the fusion.
absl::StatusOr<TensorValue> EmitScaledDot(
    EmitterContext& emitter_ctx,
    const ge::TiledHloInstruction& tiled_scaled_dot) {
  TF_RET_CHECK(tiled_scaled_dot.hlo_regions().size() == 1);
  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_scaled_dot.tile().GetStaticTileSizes());

  auto& b = emitter_ctx.b();
  const auto& scaled_dot =
      *::xla::Cast<HloScaledDotInstruction>(tiled_scaled_dot.hlo());
  // The specific accumulator type to use may not correspond to the output type
  // of the dot. In particular, that is the case when an algorithm is specified
  // and the dot's output type does not match its expectations.
  Type accumulator_type = b.getF32Type();
  TensorValue accumulator =
      CreateConst(b, accumulator_type, 0.0f, padded_tile_sizes);

  SmallVector<int64_t> sequential_dim_ids =
      GetSequentialDimIds(*tiled_scaled_dot.hlo());
  ASSIGN_OR_RETURN(
      SmallVector<int64_t> loop_iteration_counts,
      GetSequentialLoopIterationCounts(tiled_scaled_dot, sequential_dim_ids));
  TF_RET_CHECK(loop_iteration_counts.size() == 1)
      << "Expected exactly one loop iteration count for scaled dot";

  auto for_op = mlir::scf::ForOp::create(
      b,
      /*lowerBound=*/MakeIndex(b, 0),
      /*upperBound=*/MakeIndex(b, loop_iteration_counts.front()),
      /*step=*/MakeIndex(b, 1), accumulator);

  {  // Loop body.
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(for_op.getBody());
    Value iv = for_op.getInductionVar();
    const ge::TilingSpace::DimensionInfo& dim_info =
        tiled_scaled_dot.tile().tiling_space().GetDimensionInfo(
            *tiled_scaled_dot.hlo(), sequential_dim_ids.front());
    TF_RET_CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(
        dim_info.id, iv, Interval{0, loop_iteration_counts.front() - 1}));

    // Emit the dot region.
    const ge::TiledHloRegion& region = tiled_scaled_dot.hlo_regions().front();
    ASSIGN_OR_RETURN(auto results,
                     EmitTiledComputation(emitter_ctx, region, region.roots()));

    // Emit the partial dot.
    Value acc = for_op.getRegionIterArgs().front();
    ASSIGN_OR_RETURN(
        Value acc_next,
        xtile::EmitSingleTileScaledDot(
            b, scaled_dot,
            xtile::ScaledDotOperands{
                results[0], results[1], results[2], results[3], acc,
                ::xla::stablehlo::ConvertDotDimensionNumbers(
                    scaled_dot.dot_dimension_numbers(), &b)}));
    mlir::scf::YieldOp::create(b, acc_next);
  }

  // The output of the loop may not match the expected output type of the dot.
  // We make sure to issue a conversion if necessary.
  ASSIGN_OR_RETURN(
      Type scaled_dot_output_type,
      PrimitiveTypeToMlirType(b, scaled_dot.shape().element_type()));

  Value result = for_op.getResult(0);
  if (scaled_dot_output_type != accumulator_type) {
    result = Cast(b, result, scaled_dot_output_type);
  }
  return mlir::cast<TensorValue>(result);
}

// Emits a kRaggedDot instruction.
//
// kRaggedNonContracting (G is kSequential outer loop):
//   Output (M_total, N).  For each group g in [0, G):
//     - Load group_size_g from group_sizes[g] (1-element tile at G-loop IV).
//     - Check if this M tile (pid_M * BLOCK_M) belongs to group g.
//     - If yes, emit K-loop and accumulate partial dot into output.
//   Loop-carried: accumulator (M, N) and last_m (prefix sum of group sizes).
//
// kRaggedContracting: Grid = G × K × N. Prefix-sum per program to find start_m.
absl::StatusOr<TensorValue> EmitRaggedDot(
    EmitterContext& emitter_ctx,
    const ge::TiledHloInstruction& tiled_ragged_dot) {
  auto& b = emitter_ctx.b();
  const auto* ragged_dot_instr =
      ::xla::Cast<HloRaggedDotInstruction>(tiled_ragged_dot.hlo());
  const RaggedDotDimensionNumbers& ragged_dims =
      ragged_dot_instr->ragged_dot_dimension_numbers();
  const DotDimensionNumbers& dot_dims = ragged_dims.dot_dimension_numbers();

  const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  const bool is_contracting =
      absl::c_count(dot_dims.lhs_contracting_dimensions(), lhs_ragged_dim) > 0;
  const bool is_batch =
      absl::c_count(dot_dims.lhs_batch_dimensions(), lhs_ragged_dim) > 0;

  TF_RET_CHECK(tiled_ragged_dot.hlo_regions().size() == 1);

  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_ragged_dot.tile().GetStaticTileSizes());

  const ge::TilingSpace& tiling_space = tiled_ragged_dot.tile().tiling_space();
  const int64_t output_rank =
      static_cast<int64_t>(tiled_ragged_dot.hlo()->shape().dimensions().size());

  // Accumulator type: derived from the ragged dot's precision config so that
  // user-specified algorithms (e.g. ALG_DOT_BF16_BF16_F32) are respected,
  // mirroring the regular EmitDot path.
  ASSIGN_OR_RETURN(const Type acc_type,
                   xtile::GetDotAccumulatorType(b, *ragged_dot_instr));
  TensorValue accumulator = CreateConst(b, acc_type, 0.0f, padded_tile_sizes);

  // Helper to emit a tiled operand and its transitive deps within the region.
  auto emit_operand = [&](const ge::TiledHloInstruction* operand_t)
      -> absl::StatusOr<TensorValue> {
    absl::flat_hash_set<const ge::TiledHloInstruction*> deps;
    std::function<void(const ge::TiledHloInstruction*)> collect;
    collect = [&](const ge::TiledHloInstruction* t) {
      if (!deps.insert(t).second) return;
      for (const ge::TiledHloInstruction* op : t->operands()) {
        for (const auto& ri :
             tiled_ragged_dot.hlo_regions().front().instructions()) {
          if (ri.get() == op) {
            collect(op);
            break;
          }
        }
      }
    };
    collect(operand_t);
    TensorValue result;
    for (const auto& region_instr :
         tiled_ragged_dot.hlo_regions().front().instructions()) {
      if (!deps.count(region_instr.get())) continue;
      ASSIGN_OR_RETURN(TensorValue v,
                       EmitTiledHloInstruction(emitter_ctx, *region_instr));
      emitter_ctx.MapTiledHloToTensorValue(region_instr.get(), v);
      if (region_instr.get() == operand_t) result = v;
    }
    TF_RET_CHECK(result) << "operand_t not found in its own dep set";
    return result;
  };

  if (is_batch) {
    // --- kRaggedBatch ---
    //
    // kRaggedBatch = regular batched GEMM: output[b,:,:] = LHS[b,:,:] @
    // RHS[b,:,:] for all b in [0, B_total). group_sizes control which batch
    // elements belong to which group (for scheduling only, not computation).
    //
    // TilingSpace: B_total, M, N kParallel; K kSequential.
    // Output tile: [1, BLOCK_M, BLOCK_N] (one batch element per program).
    // The group_sizes operand (op 2) is NOT loaded — it's irrelevant here.

    const ge::TilingSpace::DimensionInfo& k_dim_info =
        tiling_space.GetDimensionInfo(*ragged_dot_instr, output_rank);
    CHECK(k_dim_info.tile_size.has_value())
        << "K tile size not set for kRaggedBatch.";
    const int64_t K_tiles =
        CeilOfRatio(k_dim_info.dimension_size, *k_dim_info.tile_size);

    const int64_t num_batch = dot_dims.lhs_batch_dimensions_size();

    const ge::TiledHloInstruction* lhs_tiled = tiled_ragged_dot.operand(0);
    const ge::TiledHloInstruction* rhs_tiled = tiled_ragged_dot.operand(1);
    // operand(2) = group_sizes — NOT emitted.

    // K sequential accumulation loop (same structure as EmitDot).
    auto k_for =
        mlir::scf::ForOp::create(b, MakeIndex(b, 0), MakeIndex(b, K_tiles),
                                 MakeIndex(b, 1), accumulator);
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(k_for.getBody());
      Value k_iv = k_for.getInductionVar();
      Value k_iv_i32 = Cast(b, k_iv, b.getI32Type());

      CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(
          k_dim_info.id, k_iv, Interval{0, K_tiles - 1}));

      ASSIGN_OR_RETURN(TensorValue lhs_tensor, emit_operand(lhs_tiled));
      ASSIGN_OR_RETURN(TensorValue rhs_tensor, emit_operand(rhs_tiled));

      // K-boundary masking.
      ASSIGN_OR_RETURN(lhs_tensor,
                       MaskDotOperand(b, *lhs_tiled, lhs_tensor, k_iv_i32,
                                      dot_dims.lhs_contracting_dimensions(0)));
      ASSIGN_OR_RETURN(rhs_tensor,
                       MaskDotOperand(b, *rhs_tiled, rhs_tensor, k_iv_i32,
                                      dot_dims.rhs_contracting_dimensions(0)));

      // Squeeze batch dims before inner 2-D dot (same as batched
      // kRaggedNonContracting pattern).
      auto lhs_type = lhs_tensor.getType();
      auto acc_rc = mlir::cast<mlir::RankedTensorType>(
          k_for.getRegionIterArgs()[0].getType());
      auto acc_ref = acc_rc.getShape();

      llvm::SmallVector<int64_t> lhs_2d(lhs_type.getShape().begin() + num_batch,
                                        lhs_type.getShape().end());
      llvm::SmallVector<int64_t> rhs_2d(
          rhs_tensor.getType().getShape().begin() + num_batch,
          rhs_tensor.getType().getShape().end());
      llvm::SmallVector<int64_t> acc_2d(acc_ref.begin() + num_batch,
                                        acc_ref.end());

      TensorValue dot_lhs = lhs_tensor, dot_rhs = rhs_tensor;
      Value dot_acc = k_for.getRegionIterArgs()[0];
      if (num_batch > 0) {
        dot_lhs = mlir::cast<TensorValue>(
            stablehlo::ReshapeOp::create(
                b,
                mlir::RankedTensorType::get(lhs_2d, lhs_type.getElementType()),
                lhs_tensor)
                .getResult());
        dot_rhs = mlir::cast<TensorValue>(
            stablehlo::ReshapeOp::create(
                b,
                mlir::RankedTensorType::get(
                    rhs_2d, rhs_tensor.getType().getElementType()),
                rhs_tensor)
                .getResult());
        dot_acc =
            stablehlo::ReshapeOp::create(
                b, mlir::RankedTensorType::get(acc_2d, acc_rc.getElementType()),
                dot_acc)
                .getResult();
      }

      DotDimensionNumbers inner_dot_dims;
      inner_dot_dims.add_lhs_contracting_dimensions(
          dot_dims.lhs_contracting_dimensions(0) - num_batch);
      inner_dot_dims.add_rhs_contracting_dimensions(
          dot_dims.rhs_contracting_dimensions(0) - num_batch);

      ASSIGN_OR_RETURN(
          Value acc_next,
          xtile::EmitSingleTileDot(
              b, *ragged_dot_instr, inner_dot_dims,
              xtile::DotOperands{dot_lhs, dot_rhs,
                                 mlir::cast<TensorValue>(dot_acc)}));

      if (num_batch > 0) {
        acc_next =
            stablehlo::ReshapeOp::create(
                b,
                mlir::RankedTensorType::get(
                    llvm::SmallVector<int64_t>(acc_ref.begin(), acc_ref.end()),
                    acc_rc.getElementType()),
                acc_next)
                .getResult();
      }
      mlir::scf::YieldOp::create(b, acc_next);
    }

    Value result = k_for.getResult(0);
    ASSIGN_OR_RETURN(Type out_type,
                     PrimitiveTypeToMlirType(
                         b, tiled_ragged_dot.hlo()->shape().element_type()));
    if (out_type != acc_type) result = Cast(b, result, out_type);
    return mlir::cast<TensorValue>(result);
  }

  // Helper to emit group_sizes and any transitively required instructions
  // (e.g. a scalar constant that gets broadcast-simplified) using only the
  // dependency chain of gs_tiled within the region.
  auto emit_gs =
      [&](const ge::TiledHloInstruction* gs_t) -> absl::StatusOr<TensorValue> {
    // Collect gs_tiled's transitive dependencies that live in the region.
    absl::flat_hash_set<const ge::TiledHloInstruction*> gs_deps;
    std::function<void(const ge::TiledHloInstruction*)> collect;
    collect = [&](const ge::TiledHloInstruction* t) {
      if (!gs_deps.insert(t).second) return;
      for (const ge::TiledHloInstruction* op : t->operands()) {
        for (const auto& ri :
             tiled_ragged_dot.hlo_regions().front().instructions()) {
          if (ri.get() == op) {
            collect(op);
            break;
          }
        }
      }
    };
    collect(gs_t);

    // Emit each dep in def-before-use (region) order.
    TensorValue gs_tile;
    for (const auto& region_instr :
         tiled_ragged_dot.hlo_regions().front().instructions()) {
      if (!gs_deps.count(region_instr.get())) continue;
      ASSIGN_OR_RETURN(TensorValue result,
                       EmitTiledHloInstruction(emitter_ctx, *region_instr));
      emitter_ctx.MapTiledHloToTensorValue(region_instr.get(), result);
      if (region_instr.get() == gs_t) gs_tile = result;
    }
    TF_RET_CHECK(gs_tile) << "gs_tiled not found in its own dep set";
    return gs_tile;
  };

  if (!is_contracting) {
    // --- kRaggedNonContracting ---
    // TilingSpace layout:
    //   dim[output_rank+0]: G  kSequential (tile_size=1, outer group loop)
    //   dim[output_rank+1]: K  kSequential (tile_size=BLOCK_K, inner K-loop)
    // Operands in the region: [lhs, rhs, group_sizes]

    const ge::TilingSpace::DimensionInfo& g_dim_info =
        tiling_space.GetDimensionInfo(*ragged_dot_instr, output_rank);
    const ge::TilingSpace::DimensionInfo& k_dim_info =
        tiling_space.GetDimensionInfo(*ragged_dot_instr, output_rank + 1);

    const int64_t G = g_dim_info.dimension_size;
    CHECK(k_dim_info.tile_size.has_value())
        << "K tile size must be set before emitting ragged dot.";
    const int64_t K_tiles =
        CeilOfRatio(k_dim_info.dimension_size, *k_dim_info.tile_size);

    const ge::TiledHloInstruction* lhs_tiled = tiled_ragged_dot.operand(0);
    const ge::TiledHloInstruction* rhs_tiled = tiled_ragged_dot.operand(1);
    const ge::TiledHloInstruction* gs_tiled = tiled_ragged_dot.operand(2);

    // M is at output dim `num_batch_dims` (after any batch dims).
    // `lhs_ragged_dim` is also the M dim index in the LHS tile shape.
    const int64_t num_batch_dims = dot_dims.lhs_batch_dimensions_size();

    // Compute tile_m_abs BEFORE registering G so M's parallel dim is
    // evaluated cleanly without any sequential dim interference
    // (especially important when the M/N schedule swap is active).
    const int64_t BLOCK_M = padded_tile_sizes[num_batch_dims];
    const ge::DimTile& m_output_dim_pre =
        tiled_ragged_dot.tile().dim_tiles()[num_batch_dims];
    ASSIGN_OR_RETURN(
        SmallVector<Value> m_abs_pre,
        emitter_ctx.EvaluateTilingParameters({m_output_dim_pre.offset}));
    Value tile_m_abs = m_abs_pre[0];  // index, computed before G IV registered
    Value tile_m_i32_cmp = Cast(b, tile_m_abs, b.getI32Type());
    Value tile_end_i32 = arith::AddIOp::create(
        b, tile_m_i32_cmp, CreateConst(b, b.getI32Type(), BLOCK_M));

    // Outer G-loop: iter_args = (accumulator, last_m=0).
    auto g_for_op = mlir::scf::ForOp::create(
        b, MakeIndex(b, 0), MakeIndex(b, G), MakeIndex(b, 1),
        mlir::ValueRange{accumulator, MakeIndex(b, 0)});

    {
      mlir::OpBuilder::InsertionGuard g_guard(b);
      b.setInsertionPointToStart(g_for_op.getBody());
      Value g_iv = g_for_op.getInductionVar();
      Value last_m = g_for_op.getRegionIterArgs()[1];
      TensorValue acc_in =
          mlir::cast<TensorValue>(g_for_op.getRegionIterArgs()[0]);

      // Register G sequential dim so EvaluateTilingParameters can resolve
      // RTVar offsets that depend on the G loop IV.
      CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(g_dim_info.id, g_iv,
                                                        Interval{0, G - 1}));

      // Emit group_sizes tile (G-scoped).
      // Uses emit_gs which handles any HLO form:  parameter, inlined constant,
      // broadcast of scalar constant, or any other op-defined group_sizes.
      ASSIGN_OR_RETURN(TensorValue gs_tile, emit_gs(gs_tiled));

      // Extract group_size_g from gs_tile.
      // gs_tile is tensor<1xi{32|64}>; XTile/Triton's tensor.extract requires
      // a rank-0 tensor — reshape to tensor<elem_type> (rank-0) first.
      auto gs_elem_type = gs_tile.getType().getElementType();
      TensorValue gs_tile_scalar = mlir::cast<TensorValue>(
          stablehlo::ReshapeOp::create(
              b, mlir::RankedTensorType::get({}, gs_elem_type), gs_tile)
              .getResult());
      Value gs_raw = mlir::tensor::ExtractOp::create(b, gs_tile_scalar);
      Value group_size_g =
          mlir::arith::IndexCastOp::create(b, b.getIndexType(), gs_raw);

      // Determine whether this program's M tile OVERLAPS with group g.
      // tile_m_abs was computed before the G-loop to avoid interference
      // from the G sequential dim registration (see comment above).
      Value last_m_plus_gs =
          mlir::arith::AddIOp::create(b, last_m, group_size_g);
      // Signed overlap check in i32 (works for all schedule permutations):
      //   tile_m_abs_i32 < last_m_plus_gs_i32  AND
      //   tile_end_i32   > last_m_i32
      Value last_m_i32_cmp = Cast(b, last_m, b.getI32Type());
      Value last_m_plus_i32_cmp = Cast(b, last_m_plus_gs, b.getI32Type());
      Value overlaps = mlir::arith::AndIOp::create(
          b,
          mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::slt,
                                      tile_m_i32_cmp, last_m_plus_i32_cmp),
          mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::sgt,
                                      tile_end_i32, last_m_i32_cmp));

      // Conditional: run K-loop if this M tile overlaps group g at all.
      auto if_op = mlir::scf::IfOp::create(b, acc_in.getType(), overlaps,
                                           /*hasElse=*/true);

      {  // Then block: K-loop accumulation.
        mlir::OpBuilder::InsertionGuard if_guard(b);
        b.setInsertionPointToStart(if_op.thenBlock());

        auto k_for_op =
            mlir::scf::ForOp::create(b, MakeIndex(b, 0), MakeIndex(b, K_tiles),
                                     MakeIndex(b, 1), mlir::ValueRange{acc_in});

        {  // K-loop body.
          mlir::OpBuilder::InsertionGuard k_guard(b);
          b.setInsertionPointToStart(k_for_op.getBody());
          Value k_iv = k_for_op.getInductionVar();
          TensorValue acc_k_in =
              mlir::cast<TensorValue>(k_for_op.getRegionIterArgs()[0]);

          // Register K sequential dim IV.
          CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(
              k_dim_info.id, k_iv, Interval{0, K_tiles - 1}));

          // Emit LHS and RHS tiles (K-scoped).
          // emit_operand handles the full instruction chain (parameter,
          // bitcast, transpose, etc.) that layout normalization may insert
          // inside the fusion for non-standard memory layouts.
          ASSIGN_OR_RETURN(TensorValue lhs_tensor, emit_operand(lhs_tiled));
          ASSIGN_OR_RETURN(TensorValue rhs_tensor, emit_operand(rhs_tiled));

          // Mask LHS rows that are outside group g.
          // Row r (0..BLOCK_M-1) is valid iff:
          //   tile_m_abs + r >= last_m  AND  tile_m_abs + r < last_m_plus_gs
          // Build an i32 mask over [BLOCK_M] rows, then broadcast to [M, K].
          {
            auto lhs_shape = lhs_tensor.getType().getShape();
            // lhs_ragged_dim is the M dimension in the LHS tile shape.
            // For no-batch: lhs_ragged_dim=0 → lhs_shape[0]=BLOCK_M.
            // For batched:  lhs_ragged_dim=1 → lhs_shape[1]=BLOCK_M.
            int64_t lhs_m_tile_size = lhs_shape[lhs_ragged_dim];  // == BLOCK_M
            int64_t lhs_m_dim = lhs_m_tile_size;
            // iota over M rows: 0..BLOCK_M-1
            TensorValue row_iota = Iota(b, static_cast<int32_t>(lhs_m_dim));
            // abs_m[r] = tile_m_abs_i32 + r
            // Use pre-computed tile_m_i32_cmp (avoids re-evaluating M offset
            // inside G-loop where sequential dims are registered).
            TensorValue tile_m_splat =
                xtile::Splat(b, tile_m_i32_cmp, {lhs_m_dim});
            Value abs_m = arith::AddIOp::create(b, row_iota, tile_m_splat);
            // last_m and last_m_plus_gs in i32 (already computed above).
            TensorValue lm_splat = xtile::Splat(b, last_m_i32_cmp, {lhs_m_dim});
            TensorValue lmp_splat =
                xtile::Splat(b, last_m_plus_i32_cmp, {lhs_m_dim});
            Value mask_lo = arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                                                  abs_m, lm_splat);
            Value mask_hi = arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                                  abs_m, lmp_splat);
            Value row_mask = arith::AndIOp::create(b, mask_lo, mask_hi);
            // Broadcast [BLOCK_M] mask along the M dim of the LHS tile.
            row_mask =
                xtile::BroadcastInDims(b, mlir::cast<TensorValue>(row_mask),
                                       lhs_shape, {lhs_ragged_dim});
            ASSIGN_OR_RETURN(Type lhs_elem_ty,
                             PrimitiveTypeToMlirType(
                                 b, lhs_tiled->hlo()->shape().element_type()));
            TensorValue lhs_zero = CreateConst(b, lhs_elem_ty, 0.0f, lhs_shape);
            lhs_tensor = mlir::cast<TensorValue>(
                arith::SelectOp::create(b, row_mask, lhs_tensor, lhs_zero)
                    .getResult());
          }

          // The RHS tile has shape [...batch..., G_tile=1, K_tile, N].
          // Squeeze out the G dimension (at rhs_group_dim) so the inner
          // dot sees [...batch..., K_tile, N].
          const int64_t rhs_group_dim_orig =
              ragged_dims.rhs_group_dimensions(0);
          auto rhs_full_shape = rhs_tensor.getType().getShape();
          llvm::SmallVector<int64_t> rhs_squeezed_shape;
          for (int64_t i = 0; i < static_cast<int64_t>(rhs_full_shape.size());
               ++i) {
            if (i != rhs_group_dim_orig)
              rhs_squeezed_shape.push_back(rhs_full_shape[i]);
          }
          rhs_tensor = mlir::cast<TensorValue>(
              stablehlo::ReshapeOp::create(
                  b,
                  mlir::RankedTensorType::get(
                      rhs_squeezed_shape,
                      rhs_tensor.getType().getElementType()),
                  rhs_tensor)
                  .getResult());

          // After squeezing G, the K contracting dim shifts:
          //   If K_orig > rhs_group_dim → k_idx decreases by 1
          //   Otherwise stays the same.
          const int64_t rhs_k_orig = dot_dims.rhs_contracting_dimensions(0);
          const int64_t rhs_k_dim_after_squeeze =
              (rhs_k_orig > rhs_group_dim_orig) ? rhs_k_orig - 1 : rhs_k_orig;

          // Mask LHS at K-boundary (same as EmitDot).
          Value k_iv_i32 = Cast(b, k_iv, b.getI32Type());
          ASSIGN_OR_RETURN(
              lhs_tensor,
              MaskDotOperand(b, *lhs_tiled, lhs_tensor, k_iv_i32,
                             dot_dims.lhs_contracting_dimensions(0)));

          // Mask squeezed RHS at K-boundary inline (MaskDotOperand reads the
          // pre-squeeze HLO shape, so we replicate its logic here with the
          // squeezed shape).
          {
            int64_t K = rhs_tiled->hlo()->shape().dimensions(rhs_k_orig);
            int64_t tile_k = rhs_squeezed_shape[rhs_k_dim_after_squeeze];
            if (K % tile_k != 0) {
              Value tile_size_value = CreateConst(b, b.getI32Type(), tile_k);
              Value num_full_tiles = arith::DivSIOp::create(
                  b, CreateConst(b, b.getI32Type(), K), tile_size_value);
              auto cond = arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                                                k_iv_i32, num_full_tiles);
              auto if_mask = mlir::scf::IfOp::create(b, rhs_tensor.getType(),
                                                     cond, /*withElse=*/true);
              {
                mlir::OpBuilder::InsertionGuard mg(b);
                b.setInsertionPointToStart(if_mask.thenBlock());
                Value tile_offset =
                    arith::MulIOp::create(b, k_iv_i32, tile_size_value);
                TensorValue range = Iota(b, tile_k);
                TensorValue bcast_off = xtile::Splat(b, tile_offset, {tile_k});
                Value indices = arith::AddIOp::create(b, range, bcast_off);
                Value boundary = CreateConst(b, b.getI32Type(), K, {tile_k});
                Value mask = arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                                   indices, boundary);
                llvm::ArrayRef<int64_t> tile_shape =
                    rhs_tensor.getType().getShape();
                mask = xtile::BroadcastInDims(b, mlir::cast<TensorValue>(mask),
                                              tile_shape,
                                              {rhs_k_dim_after_squeeze});
                ASSIGN_OR_RETURN(
                    Type elem_ty,
                    PrimitiveTypeToMlirType(
                        b, rhs_tiled->hlo()->shape().element_type()));
                TensorValue zero = CreateConst(b, elem_ty, 0.0f, tile_shape);
                Value masked =
                    arith::SelectOp::create(b, mask, rhs_tensor, zero);
                mlir::scf::YieldOp::create(b, masked);
              }
              {
                mlir::OpBuilder::InsertionGuard mg(b);
                b.setInsertionPointToStart(if_mask.elseBlock());
                mlir::scf::YieldOp::create(b, rhs_tensor);
              }
              b.setInsertionPointAfter(if_mask);
              rhs_tensor = mlir::cast<TensorValue>(if_mask.getResult(0));
            }
          }

          // For batched ragged dot (num_batch_dims > 0), the lhs/rhs/acc have
          // leading batch dims of size 1 (one batch element per tile). Triton's
          // tt.dot operates on 2D tensors. Squeeze those size-1 batch dims
          // before the dot, then unsqueeze the result.
          //
          // Non-batched path (num_batch_dims == 0) is unchanged.
          auto lhs_type = lhs_tensor.getType();
          auto acc_type_cast =
              mlir::cast<mlir::RankedTensorType>(acc_k_in.getType());
          auto acc_shape_ref = acc_type_cast.getShape();

          // Compute 2-D (squeezed) shapes by dropping batch dims.
          llvm::SmallVector<int64_t> lhs_2d_shape(
              lhs_type.getShape().begin() + num_batch_dims,
              lhs_type.getShape().end());
          llvm::SmallVector<int64_t> rhs_2d_shape(
              rhs_squeezed_shape.begin() + num_batch_dims,
              rhs_squeezed_shape.end());
          llvm::SmallVector<int64_t> acc_2d_shape(
              acc_shape_ref.begin() + num_batch_dims, acc_shape_ref.end());

          TensorValue dot_lhs = lhs_tensor;
          TensorValue dot_rhs = rhs_tensor;
          Value dot_acc = acc_k_in;
          if (num_batch_dims > 0) {
            dot_lhs = mlir::cast<TensorValue>(
                stablehlo::ReshapeOp::create(
                    b,
                    mlir::RankedTensorType::get(lhs_2d_shape,
                                                lhs_type.getElementType()),
                    lhs_tensor)
                    .getResult());
            dot_rhs = mlir::cast<TensorValue>(
                stablehlo::ReshapeOp::create(
                    b,
                    mlir::RankedTensorType::get(
                        rhs_2d_shape, rhs_tensor.getType().getElementType()),
                    rhs_tensor)
                    .getResult());
            dot_acc = stablehlo::ReshapeOp::create(
                          b,
                          mlir::RankedTensorType::get(
                              acc_2d_shape, acc_type_cast.getElementType()),
                          acc_k_in)
                          .getResult();
          }

          // Build inner dot dimension numbers.
          // When batch dims are squeezed out, contracting dim indices decrease
          // by num_batch_dims (the batch dims occupied the leading positions).
          DotDimensionNumbers inner_dot_dims;
          inner_dot_dims.add_lhs_contracting_dimensions(
              dot_dims.lhs_contracting_dimensions(0) - num_batch_dims);
          inner_dot_dims.add_rhs_contracting_dimensions(
              rhs_k_dim_after_squeeze - num_batch_dims);

          // Partial dot: LHS [M, K] × RHS [K, N] → acc [M, N].
          ASSIGN_OR_RETURN(
              Value acc_next_2d,
              xtile::EmitSingleTileDot(
                  b, *ragged_dot_instr, inner_dot_dims,
                  xtile::DotOperands{dot_lhs, dot_rhs,
                                     mlir::cast<TensorValue>(dot_acc)}));

          // Unsqueeze batch dims back if they were squeezed.
          Value acc_next = acc_next_2d;
          if (num_batch_dims > 0) {
            acc_next = stablehlo::ReshapeOp::create(
                           b,
                           mlir::RankedTensorType::get(
                               llvm::SmallVector<int64_t>(acc_shape_ref.begin(),
                                                          acc_shape_ref.end()),
                               acc_type_cast.getElementType()),
                           acc_next_2d)
                           .getResult();
          }
          mlir::scf::YieldOp::create(b, acc_next);
        }
        b.setInsertionPointAfter(k_for_op);
        mlir::scf::YieldOp::create(b, k_for_op.getResult(0));
      }

      {  // Else block: accumulator unchanged.
        mlir::OpBuilder::InsertionGuard else_guard(b);
        b.setInsertionPointToStart(if_op.elseBlock());
        mlir::scf::YieldOp::create(b, acc_in);
      }
      b.setInsertionPointAfter(if_op);

      // Update last_m for next G iteration.
      Value last_m_next = mlir::arith::AddIOp::create(b, last_m, group_size_g);
      mlir::scf::YieldOp::create(
          b, mlir::ValueRange{if_op.getResult(0), last_m_next});
    }
    b.setInsertionPointAfter(g_for_op);

    TensorValue result = mlir::cast<TensorValue>(g_for_op.getResult(0));

    // Cast accumulator to the declared output element type if needed.
    ASSIGN_OR_RETURN(Type out_type,
                     PrimitiveTypeToMlirType(
                         b, tiled_ragged_dot.hlo()->shape().element_type()));
    if (out_type != acc_type) {
      result = mlir::cast<TensorValue>(Cast(b, result, out_type));
    }
    return result;

  } else {
    // --- kRaggedContracting ---
    // Grid = G × K_tiles × N_tiles. Prefix-sum per program to find start_m.

    // M sequential dim: output_rank + 0.
    const ge::TilingSpace::DimensionInfo& m_dim_info =
        tiling_space.GetDimensionInfo(*ragged_dot_instr, output_rank);
    CHECK(m_dim_info.tile_size.has_value())
        << "M tile size not set for kRaggedContracting.";
    const int64_t BLOCK_M = *m_dim_info.tile_size;
    const int64_t M_total = m_dim_info.dimension_size;
    auto gs_rtvar_or = tiling_space.GetRTVarInfo(*ragged_dot_instr, 2);
    auto sm_rtvar_or = tiling_space.GetRTVarInfo(*ragged_dot_instr, -1);
    CHECK(gs_rtvar_or.has_value()) << "Missing group_size RTVar.";
    CHECK(sm_rtvar_or.has_value()) << "Missing start_m RTVar.";

    const int64_t num_batch_dims = dot_dims.lhs_batch_dimensions_size();

    // g_abs: the group index for this tile (kParallel dim 0 of output [G,...]).
    // tile_propagation layout: [G, batch..., K, N]  → G at output dim 0.
    const ge::DimTile& g_out_dim = tiled_ragged_dot.tile().dim_tiles()[0];
    ASSIGN_OR_RETURN(SmallVector<Value> g_abs_vec,
                     emitter_ctx.EvaluateTilingParameters({g_out_dim.offset}));
    Value g_abs = g_abs_vec[0];  // index type, = g * G_tile_size(=1) = g

    // Resolve group_sizes to its fusion argument buffer (needed for the
    // O(G) prefix sum: we must load gs[0..g-1] at arbitrary offsets).
    const ge::TiledHloInstruction* gs_tiled = tiled_ragged_dot.operand(2);
    const HloFusionInstruction& fusion = emitter_ctx.fusion();
    const HloInstruction* gs_hlo_param = gs_tiled->hlo();
    if (gs_hlo_param->opcode() == HloOpcode::kParameter &&
        !fusion.IsUserOf(gs_hlo_param)) {
      gs_hlo_param = gs_hlo_param->parent()->FusionInstruction()->operand(
          gs_hlo_param->parameter_number());
    }
    int64_t gs_arg_idx = fusion.operand_index(gs_hlo_param);
    mlir::Value gs_buf = emitter_ctx.entry_func().getArgument(gs_arg_idx);
    ASSIGN_OR_RETURN(
        Type gs_elem_ty,
        PrimitiveTypeToMlirType(b, gs_tiled->hlo()->shape().element_type()));

    // Helper: load a single group_sizes element from the buffer at index `idx`.
    // For batched group_sizes [B, G], this is a follow-up; for now we assert
    // no batch dims to keep the prefix sum simple.
    // TODO(ragged_dot): support batched kRaggedContracting.
    CHECK_EQ(num_batch_dims, 0)
        << "Batched kRaggedContracting not yet supported.";

    // Compute start_m = sum(group_sizes[0..g-1]) via a scalar prefix-sum
    // loop.  This is O(G) per program; acceptable for small G (~8–128 experts).
    // A vectorized approach (load all G elements, masked tl.sum) can be added
    // as a follow-up optimization.
    auto load_gs_scalar = [&](Value idx) -> Value {
      SmallVector<int64_t> sizes_1{1};
      auto gs_1_type = mlir::RankedTensorType::get(sizes_1, gs_elem_ty);
      TensorValue g1 = mlir::cast<TensorValue>(
          xtile::ExtractTileOp::create(b, gs_1_type, gs_buf,
                                       SmallVector<Value>{idx}, sizes_1,
                                       SmallVector<int64_t>{1})
              .getResult());
      TensorValue g0 = mlir::cast<TensorValue>(
          stablehlo::ReshapeOp::create(
              b, mlir::RankedTensorType::get({}, gs_elem_ty), g1)
              .getResult());
      Value gv = mlir::tensor::ExtractOp::create(b, g0);
      return mlir::arith::IndexCastOp::create(b, b.getIndexType(), gv);
    };

    Value start_m = MakeIndex(b, 0);
    {
      auto pfx = mlir::scf::ForOp::create(b, MakeIndex(b, 0), g_abs,
                                          MakeIndex(b, 1), start_m);
      mlir::OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(pfx.getBody());
      Value new_sum = arith::AddIOp::create(
          b, pfx.getRegionIterArgs()[0], load_gs_scalar(pfx.getInductionVar()));
      mlir::scf::YieldOp::create(b, new_sum);
      b.setInsertionPointAfter(pfx);
      start_m = pfx.getResult(0);
    }

    // Load group_size_g = group_sizes[g_abs] using the tiled group_sizes
    // operand (which already has the correct kParallel G offset from tile_id).
    // emit_gs resolves any HLO form (param, constant, broadcast, etc.).
    ASSIGN_OR_RETURN(TensorValue gs_tile, emit_gs(gs_tiled));
    Value gs_raw;
    {
      TensorValue gs_0d = mlir::cast<TensorValue>(
          stablehlo::ReshapeOp::create(
              b,
              mlir::RankedTensorType::get({},
                                          gs_tile.getType().getElementType()),
              gs_tile)
              .getResult());
      gs_raw = mlir::tensor::ExtractOp::create(b, gs_0d);
    }
    Value group_size_g =
        mlir::arith::IndexCastOp::create(b, b.getIndexType(), gs_raw);

    // M loop count = cdiv(group_size_g, BLOCK_M).
    Value bm_val = MakeIndex(b, BLOCK_M);
    Value bm_m1 = MakeIndex(b, BLOCK_M - 1);
    Value m_loop_count = arith::DivSIOp::create(
        b, arith::AddIOp::create(b, group_size_g, bm_m1), bm_val);

    // Resolve K and N parallel dim offsets from the tile_id.
    // Output dim ordering: [G=0, batch..., K, N]
    // out_k_start = 1 + num_batch_dims, out_n_start = out_k_start + 1
    const int64_t out_k_start = 1 + num_batch_dims;
    const ge::DimTile& k_out_dim =
        tiled_ragged_dot.tile().dim_tiles()[out_k_start];
    ASSIGN_OR_RETURN(SmallVector<Value> k_abs_vec,
                     emitter_ctx.EvaluateTilingParameters({k_out_dim.offset}));
    Value k_abs = k_abs_vec[0];

    const int64_t out_n_start = out_k_start + 1;
    const ge::DimTile& n_out_dim =
        tiled_ragged_dot.tile().dim_tiles()[out_n_start];
    ASSIGN_OR_RETURN(SmallVector<Value> n_abs_vec,
                     emitter_ctx.EvaluateTilingParameters({n_out_dim.offset}));
    Value n_abs = n_abs_vec[0];

    // LHS and RHS buffer arguments for direct ExtractTileOp loads.
    // For FP8 inputs, the operand of the ragged-dot inside the fusion may be a
    // convert(parameter) instruction (e.g. f8→f32 widening). Traverse through
    // any kConvert to reach the underlying buffer parameter.
    const HloInstruction* lhs_hlo = tiled_ragged_dot.operand(0)->hlo();
    if (lhs_hlo->opcode() == HloOpcode::kConvert) {
      lhs_hlo = lhs_hlo->operand(0);
    }
    const PrimitiveType lhs_buf_elem_type = lhs_hlo->shape().element_type();
    if (lhs_hlo->opcode() == HloOpcode::kParameter &&
        !fusion.IsUserOf(lhs_hlo)) {
      lhs_hlo = lhs_hlo->parent()->FusionInstruction()->operand(
          lhs_hlo->parameter_number());
    }
    int64_t lhs_arg_idx = fusion.operand_index(lhs_hlo);
    mlir::Value lhs_buf = emitter_ctx.entry_func().getArgument(lhs_arg_idx);

    const HloInstruction* rhs_hlo = tiled_ragged_dot.operand(1)->hlo();
    if (rhs_hlo->opcode() == HloOpcode::kConvert) {
      rhs_hlo = rhs_hlo->operand(0);
    }
    const PrimitiveType rhs_buf_elem_type = rhs_hlo->shape().element_type();
    if (rhs_hlo->opcode() == HloOpcode::kParameter &&
        !fusion.IsUserOf(rhs_hlo)) {
      rhs_hlo = rhs_hlo->parent()->FusionInstruction()->operand(
          rhs_hlo->parameter_number());
    }
    int64_t rhs_arg_idx = fusion.operand_index(rhs_hlo);
    mlir::Value rhs_buf = emitter_ctx.entry_func().getArgument(rhs_arg_idx);

    // Determine BLOCK_K and BLOCK_N from the output tile padded sizes.
    const int64_t BLOCK_K = padded_tile_sizes[out_k_start];
    const int64_t BLOCK_N = padded_tile_sizes[out_n_start];

    // Element types for LHS and RHS — use the underlying buffer element type
    // (before any convert, e.g. f8e4m3fnuz for FP8 inputs). EmitSingleTileDot
    // will cast to the required accumulation type (f32) if needed.
    ASSIGN_OR_RETURN(Type lhs_elem_mlir_ty,
                     PrimitiveTypeToMlirType(b, lhs_buf_elem_type));
    ASSIGN_OR_RETURN(Type rhs_elem_mlir_ty,
                     PrimitiveTypeToMlirType(b, rhs_buf_elem_type));

    // Output tile shape for this program: padded_tile_sizes = [1, K, N] (or
    // [1, B..., K, N] for batched).  The leading 1 is the G tile size.
    // Inner accumulator: 2-D [K_tile, N_tile] (drop the G=1 leading dim).
    // After the M loop we reshape back to padded_tile_sizes.
    llvm::SmallVector<int64_t> acc_2d_shape(padded_tile_sizes.begin() + 1,
                                            padded_tile_sizes.end());
    TensorValue accumulator_2d = CreateConst(b, acc_type, 0.0f, acc_2d_shape);

    // lhs_contracting_dim_in_tile: position of M in the LHS tile
    // rhs_contracting_dim_in_tile: position of M in the RHS tile
    const int64_t lhs_m_tile_dim =
        dot_dims.lhs_contracting_dimensions(0) - num_batch_dims;
    const int64_t rhs_m_tile_dim =
        dot_dims.rhs_contracting_dimensions(0) - num_batch_dims;

    // M sequential accumulation loop.
    auto m_for = mlir::scf::ForOp::create(b, MakeIndex(b, 0), m_loop_count,
                                          MakeIndex(b, 1), accumulator_2d);
    {
      mlir::OpBuilder::InsertionGuard m_guard(b);
      b.setInsertionPointToStart(m_for.getBody());
      Value m_iv = m_for.getInductionVar();
      TensorValue acc_in =
          mlir::cast<TensorValue>(m_for.getRegionIterArgs()[0]);

      // Register the M sequential dim IV so that the relative tile offset
      // (m_iv * BLOCK_M) can be evaluated by EvaluateTilingParameters if
      // needed elsewhere. The actual buffer address uses start_m + m_iv*BLOCK_M
      // computed below.
      //
      // Use the per-group maximum loop count as the interval upper bound.
      // M_total/BLOCK_M is G times too large: Triton uses this interval for
      // speculative prefetch analysis and generates loads at m_abs = start_m +
      // (M_total/BLOCK_M)*BLOCK_M which is far beyond the LHS/RHS buffer →
      // GPU memory access fault when those speculative addresses hit unmapped
      // pages. The correct bound is cdiv(cdiv(M_total, G), BLOCK_M) - 1.
      {
        const int64_t G_nc_iv =
            ragged_dot_instr->operand(2)->shape().dimensions(0);
        const int64_t q_max_nc_iv =
            (G_nc_iv > 0) ? (M_total + G_nc_iv - 1) / G_nc_iv : M_total;
        const int64_t m_loop_max_nc_iv = (q_max_nc_iv + BLOCK_M - 1) / BLOCK_M;
        CHECK(emitter_ctx.MapSymbolIdToSequentialDimValue(
            m_dim_info.id, m_iv,
            Interval{0, m_loop_max_nc_iv > 0 ? m_loop_max_nc_iv - 1 : 0}));
      }

      // Absolute M offset = start_m + m_iv * BLOCK_M.
      // Clamp to prevent Triton speculative loads beyond the group's slice.
      // The clamp: m_abs_upper = start_m + (m_loop_count - 1) * BLOCK_M
      // = start of the LAST VALID tile. This is correct for all group sizes:
      //   - group_size_g=10, BLOCK_M=32: m_loop_count=1, upper=start_m ✓
      //   - group_size_g=56, BLOCK_M=32: m_loop_count=2, upper=start_m+32 ✓
      //   - group_size_g=128, BLOCK_M=64: m_loop_count=2, upper=start_m+64 ✓
      // The clamp only activates for OOB-speculation (m_iv ≥ m_loop_count).
      // Those iterations are masked to zero by M-boundary masking below.
      Value m_abs_raw = arith::AddIOp::create(
          b, start_m, arith::MulIOp::create(b, m_iv, MakeIndex(b, BLOCK_M)));
      {
        // m_loop_count was computed above: cdiv(group_size_g, BLOCK_M).
        Value last_valid_idx =
            arith::SubIOp::create(b, m_loop_count, MakeIndex(b, 1));
        Value m_abs_upper = arith::AddIOp::create(
            b, start_m,
            arith::MulIOp::create(b, last_valid_idx, MakeIndex(b, BLOCK_M)));
        m_abs_raw = arith::MinSIOp::create(b, m_abs_raw, m_abs_upper);
      }
      Value m_abs = m_abs_raw;

      // Load LHS tile directly with absolute M offset (no emit_operand).
      // The tile propagation uses relative M offset (no sm_sym) since
      // start_m is a synthetic prefix-sum RTVar with no backing tiled HLO.
      // We add start_m here manually to get the correct absolute buffer offset.
      //
      // LHS offset array: for each dim, use m_abs at the contracting dim and
      // k_abs at the non-contracting dim.
      // For LHS [M, K] (contracting_dim=0): offsets=[m_abs, k_abs]
      // For LHS [K, M] (contracting_dim=1): offsets=[k_abs, m_abs]
      TensorValue lhs_tensor;
      {
        SmallVector<Value> lhs_offsets(2);
        SmallVector<int64_t> lhs_sizes(2);
        lhs_offsets[lhs_m_tile_dim] = m_abs;
        lhs_sizes[lhs_m_tile_dim] = BLOCK_M;
        lhs_offsets[1 - lhs_m_tile_dim] = k_abs;
        lhs_sizes[1 - lhs_m_tile_dim] = BLOCK_K;
        auto lhs_tile_type =
            mlir::RankedTensorType::get(lhs_sizes, lhs_elem_mlir_ty);
        lhs_tensor = mlir::cast<TensorValue>(
            xtile::ExtractTileOp::create(b, lhs_tile_type, lhs_buf, lhs_offsets,
                                         lhs_sizes, SmallVector<int64_t>{1, 1})
                .getResult());
      }

      // Load RHS tile directly with absolute M offset.
      // RHS offset array: for each dim, use m_abs at the contracting dim and
      // n_abs at the non-contracting dim.
      TensorValue rhs_tensor;
      {
        SmallVector<Value> rhs_offsets(2);
        SmallVector<int64_t> rhs_sizes(2);
        rhs_offsets[rhs_m_tile_dim] = m_abs;
        rhs_sizes[rhs_m_tile_dim] = BLOCK_M;
        rhs_offsets[1 - rhs_m_tile_dim] = n_abs;
        rhs_sizes[1 - rhs_m_tile_dim] = BLOCK_N;
        auto rhs_tile_type =
            mlir::RankedTensorType::get(rhs_sizes, rhs_elem_mlir_ty);
        rhs_tensor = mlir::cast<TensorValue>(
            xtile::ExtractTileOp::create(b, rhs_tile_type, rhs_buf, rhs_offsets,
                                         rhs_sizes, SmallVector<int64_t>{1, 1})
                .getResult());
      }

      // M-boundary masking: zero elements where m_iv*BLOCK_M + idx >=
      // group_size_g. Applied to both LHS and RHS since they share the same M
      // contracting dim.
      {
        Value m_iv_i32 = Cast(b, m_iv, b.getI32Type());
        Value gs_g_i32 = Cast(b, group_size_g, b.getI32Type());
        Value bm_i32 = CreateConst(b, b.getI32Type(), BLOCK_M);
        Value m_base = arith::MulIOp::create(b, m_iv_i32, bm_i32);
        TensorValue row_iota = Iota(b, static_cast<int32_t>(BLOCK_M));
        Value abs_row = arith::AddIOp::create(
            b, row_iota, xtile::Splat(b, m_base, {BLOCK_M}));
        Value m_mask_1d =
            arith::CmpIOp::create(b, arith::CmpIPredicate::slt, abs_row,
                                  xtile::Splat(b, gs_g_i32, {BLOCK_M}));

        // Broadcast along the M (contracting) dim of the LHS tile.
        {
          auto lhs_shape = lhs_tensor.getType().getShape();
          Value lhs_mask =
              xtile::BroadcastInDims(b, mlir::cast<TensorValue>(m_mask_1d),
                                     lhs_shape, {lhs_m_tile_dim});
          // lhs_elem_mlir_ty was computed before the M loop.
          lhs_tensor = mlir::cast<TensorValue>(
              arith::SelectOp::create(
                  b, lhs_mask, lhs_tensor,
                  CreateConst(b, lhs_elem_mlir_ty, 0.0f, lhs_shape))
                  .getResult());
        }
        // Same for RHS.
        {
          auto rhs_shape = rhs_tensor.getType().getShape();
          Value rhs_mask =
              xtile::BroadcastInDims(b, mlir::cast<TensorValue>(m_mask_1d),
                                     rhs_shape, {rhs_m_tile_dim});
          rhs_tensor = mlir::cast<TensorValue>(
              arith::SelectOp::create(
                  b, rhs_mask, rhs_tensor,
                  CreateConst(b, rhs_elem_mlir_ty, 0.0f, rhs_shape))
                  .getResult());
        }
      }

      // Inner dot: LHS[..., BLOCK_M, ...] contracted at lhs_m_tile_dim
      //            with RHS[..., BLOCK_M, ...] → acc[BLOCK_K, BLOCK_N].
      // The dot accumulates into the 2-D acc_in [K_tile, N_tile].
      DotDimensionNumbers inner_dot_dims;
      inner_dot_dims.add_lhs_contracting_dimensions(lhs_m_tile_dim);
      inner_dot_dims.add_rhs_contracting_dimensions(rhs_m_tile_dim);

      ASSIGN_OR_RETURN(Value acc_next,
                       xtile::EmitSingleTileDot(
                           b, *ragged_dot_instr, inner_dot_dims,
                           xtile::DotOperands{lhs_tensor, rhs_tensor, acc_in}));
      mlir::scf::YieldOp::create(b, acc_next);
    }
    b.setInsertionPointAfter(m_for);

    // Reshape 2-D result [K_tile, N_tile] → padded_tile_sizes [1, K, N]
    // to match what EmitGeneric expects for the InsertTileOp.
    Value result_2d = m_for.getResult(0);
    ASSIGN_OR_RETURN(Type out_type,
                     PrimitiveTypeToMlirType(
                         b, tiled_ragged_dot.hlo()->shape().element_type()));
    if (out_type != acc_type) {
      result_2d = Cast(b, result_2d, out_type);
    }
    auto result_full = mlir::cast<TensorValue>(
        stablehlo::ReshapeOp::create(
            b,
            mlir::RankedTensorType::get(
                llvm::SmallVector<int64_t>(padded_tile_sizes.begin(),
                                           padded_tile_sizes.end()),
                mlir::cast<mlir::RankedTensorType>(result_2d.getType())
                    .getElementType()),
            result_2d)
            .getResult());
    return result_full;
  }
}

absl::StatusOr<TensorValue> EmitIota(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_iota) {
  auto& b = emitter_ctx.b();
  const HloIotaInstruction* hlo_iota =
      ::xla::Cast<HloIotaInstruction>(tiled_iota.hlo());
  int64_t iota_dim = hlo_iota->iota_dimension();

  ASSIGN_OR_RETURN(SmallVector<int64_t> padded_tile_sizes,
                   tiled_iota.tile().GetStaticTileSizes());

  // We can treat iota more or less as a parameter load, except that we need to
  // generate the right values in the right place as opposed to loading them.
  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(emitter_ctx, tiled_iota));

  // First, stride as needed between the iota components.
  Value range = arith::MulIOp::create(
      b, Iota(b, padded_tile_sizes[iota_dim]),
      xtile::Splat(
          b, CreateConst(b, b.getI32Type(), tile_info.tile_strides()[iota_dim]),
          padded_tile_sizes[iota_dim]));

  // Cast the offset to the iota dimension to i32, because
  // stable_hlo.broadcast_in_dims does not support index type.
  auto iota_dim_offset = Cast(b, tile_info.offsets()[iota_dim], b.getI32Type());
  // Then, add the base offset to the iota components.
  range = arith::AddIOp::create(
      b, range, xtile::Splat(b, iota_dim_offset, padded_tile_sizes[iota_dim]));
  ASSIGN_OR_RETURN(
      Type iota_element_type,
      PrimitiveTypeToMlirType(b, hlo_iota->shape().element_type()));
  range = Cast(b, range, iota_element_type);

  // And finally, produce a broadcast along the non-iota dimensions in order to
  // produce the whole iota tile.
  return xtile::BroadcastInDims(b, mlir::cast<TensorValue>(range),
                                padded_tile_sizes,
                                /*dims=*/{iota_dim});
}

TensorValue EmitTranspose(mlir::ImplicitLocOpBuilder& b,
                          ArrayRef<int64_t> tile_sizes,
                          ArrayRef<int64_t> dimensions, TensorValue input) {
  SmallVector<int64_t> padded_tile_sizes = GetPaddedTileSizes(tile_sizes);

  Type input_element_type = input.getType().getElementType();
  Type output_tensor_type =
      mlir::RankedTensorType::get(padded_tile_sizes, input_element_type);

  mlir::DenseI64ArrayAttr order = b.getDenseI64ArrayAttr(dimensions);
  return ::mlir::stablehlo::TransposeOp::create(b, output_tensor_type, input,
                                                order);
}

absl::StatusOr<TensorValue> EmitPad(EmitterContext& emitter_ctx,
                                    const ge::TiledHloInstruction& tiled_pad) {
  auto& b = emitter_ctx.b();
  ASSIGN_OR_RETURN(SmallVector<int64_t> tile_sizes,
                   tiled_pad.tile().GetStaticTileSizes());

  const ge::TiledHloInstruction* tiled_operand = tiled_pad.operand(0);
  const auto& pad_input_shape = tiled_operand->hlo()->shape().dimensions();

  // Compute tile offsets.
  ASSIGN_OR_RETURN(TileInfo tile_info,
                   TileInfo::Construct(emitter_ctx, tiled_pad));
  SmallVector<Value, 3> tile_offsets = tile_info.offsets();

  // Compute mask.
  Type i32_type = b.getI32Type();
  Value mask;
  for (auto [dim_index, sizes] : llvm::enumerate(
           llvm::zip(pad_input_shape, tile_sizes, tile_offsets,
                     tiled_pad.hlo()->padding_config().dimensions()))) {
    auto [pad_input_dim_size, pad_output_dim_size, tile_offset, dim_config] =
        sizes;
    if (dim_config.edge_padding_low() != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Low padding is not supported but got edge_padding_low: ",
          dim_config.edge_padding_low()));
    }
    if (dim_config.interior_padding() != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Interior padding is not supported but got interior_padding: ",
          dim_config.interior_padding()));
    }

    if (pad_input_dim_size == pad_output_dim_size) {
      continue;
    }

    // LHS for the compare is an iota broadcasted to the output shape.
    TensorValue range = Iota(b, pad_output_dim_size);
    TensorValue bcast = xtile::BroadcastInDims(
        b, range, tile_sizes, {static_cast<int64_t>(dim_index)});

    // RHS for the compare is splat(pad_input_dim_size - tile_offset).
    Value tile_offset_i32 = Cast(b, tile_offset, i32_type);
    Value threshold = arith::SubIOp::create(
        b, CreateConst(b, i32_type, pad_input_dim_size), tile_offset_i32);
    TensorValue threshold_splat = xtile::Splat(b, threshold, tile_sizes);
    Value cmp = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, bcast,
                                      threshold_splat);
    mask = mask ? stablehlo::AndOp::create(b, mask, cmp) : cmp;
  }
  if (!mask) {
    return emitter_ctx.TiledHloToTensorValue(*tiled_operand);
  }
  const ge::TiledHloInstruction* padding_value = tiled_pad.operand(1);

  TensorValue pad_value_splat = xtile::Splat(
      b, emitter_ctx.TiledHloToTensorValue(*padding_value), tile_sizes);
  return mlir::cast<TensorValue>(
      arith::SelectOp::create(b, mask,
                              emitter_ctx.TiledHloToTensorValue(*tiled_operand),
                              pad_value_splat)
          .getResult());
}

// Trivial dimensions in output might be tiled with tile size > 1 and a
// simple reshape op will fail as tile size of input and output are
// different. For example:
// f32[1,8] result = reshape(f32[2,4] operand)
// where `result` has tile sizes [2,8]. Simple reshape will fail as we go from
// 8 to 16 elements in a tile.
// But if we represent this as a reshape followed by a broadcast
//   [2,4] - reshape -> [8] - broadcast -> [1,8]
// Broadcast handles the expansion of the tile size.
absl::StatusOr<TensorValue> EmitTiledBroadcastedReshape(
    mlir::ImplicitLocOpBuilder& b, const Shape& output_shape,
    llvm::ArrayRef<int64_t> output_tile_sizes, TensorValue input) {
  SmallVector<int64_t> dim_positions =
      ge::PositionsOfNonTrivialDims(output_shape.dimensions());
  SmallVector<int64_t> reshape_tile_sizes;
  reshape_tile_sizes.reserve(dim_positions.size());
  for (int64_t dim : dim_positions) {
    reshape_tile_sizes.push_back(output_tile_sizes[dim]);
  }
  ASSIGN_OR_RETURN(TensorValue re,
                   EmitTiledReshape(b, reshape_tile_sizes, input));
  // Instead of expand we create broadcast as some tile sizes might be > 1.
  return xtile::BroadcastInDims(b, re, output_tile_sizes, dim_positions);
}

absl::StatusOr<TensorValue> EmitBitcast(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_bitcast,
    TensorValue input) {
  Shape input_shape = tiled_bitcast.hlo()->operand(0)->shape();
  const Shape& output_shape = tiled_bitcast.hlo()->shape();
  const PrimitiveType input_primitive_type = input_shape.element_type();
  const PrimitiveType output_primitive_type = output_shape.element_type();

  auto& b = emitter_ctx.b();
  ASSIGN_OR_RETURN(Type output_element_type,
                   PrimitiveTypeToMlirType(b, output_primitive_type));
  ASSIGN_OR_RETURN(SmallVector<int64_t> operand_tile_sizes,
                   tiled_bitcast.operand(0)->tile().GetStaticTileSizes());
  ASSIGN_OR_RETURN(SmallVector<int64_t> output_tile_sizes,
                   tiled_bitcast.tile().GetStaticTileSizes());

  // If the bitcast changes the element type to an element type of the same
  // bitwidth, we need to emit a ttir::BitcastOp.
  if (input_primitive_type != output_primitive_type) {
    if (primitive_util::BitWidth(input_primitive_type) !=
        primitive_util::BitWidth(output_primitive_type)) {
      return absl::InvalidArgumentError(
          "Bitcast with different bitwidth for operand and output shape "
          "element type is not yet supported.");
    }
    auto output_type =
        mlir::RankedTensorType::get(operand_tile_sizes, output_element_type);
    input = mlir::cast<TensorValue>(
        mlir::tensor::BitcastOp::create(b, output_type, input).getResult());
    input_shape.set_element_type(output_shape.element_type());
  }

  // Bitcast is transpose.
  if (!input_shape.dimensions().empty()) {
    if (std::optional<std::vector<int64_t>> transpose_dims =
            ShapeUtil::DeduceTransposeDimensionsForBitcast(input_shape,
                                                           output_shape)) {
      return EmitTiledTranspose(b, output_tile_sizes,
                                llvm::to_vector(*transpose_dims), input);
    }
  }

  // Bitcast is reshape.
  if (ShapeUtil::ReshapeIsBitcast(input_shape, output_shape,
                                  /*ignore_element_type=*/true)) {
    return EmitTiledBroadcastedReshape(b, output_shape, output_tile_sizes,
                                       input);
  }

  // Bitcast is decomposable to a transpose+reshape+transpose.
  auto trt = ShapeUtil::DecomposeBitcastToTrt(input_shape, output_shape);
  TF_RET_CHECK(trt.has_value());

  // When replacing the `bitcast` with `transpose` + `reshape` + `transpose` we
  // need to provide the tile sizes at output of each op. We already have the
  // tiling of the `input` (before the first transpose) and the tiling of the
  // final output (after the second transpose), so what's missing are the two
  // tilings in between - after the first transpose and after the reshape. In
  // the case of arbitrary ops, we would need to run the tiling analysis to
  // compute this, but in the case of bitcast we can trivially compute the
  // needed tile sizes from the input and output.

  // The tiles sizes we need to use for the output of the first transpose
  // are the permuted tiles sizes of the input. Note that these are
  // different, even in rank, compared to the tile sizes of the final shape of
  // the bitcast, so it's not possible to easily propagate them from the output.
  std::vector<int64_t> transpose1_tile_sizes =
      Permute(operand_tile_sizes, trt->transpose1_dims);
  TensorValue normalized_input =
      trt->IsTranspose1Identity()
          ? input
          : EmitTiledTranspose(b, transpose1_tile_sizes,
                               llvm::to_vector(trt->transpose1_dims), input);

  // Like the first transpose above, the tile sizes after the second transpose
  // are a permutation (according to transpose2_dims) of the tile sizes of
  // the reshape. Since we know the tile sizes of the final transpose and need
  // the tile sizes of the reshape, we compute the tile sizes backwards, taking
  // the inverse permutation.
  std::vector<int64_t> reshape_tile_sizes =
      PermuteInverse(output_tile_sizes, trt->transpose2_dims);
  TensorValue normalized_reshape;
  if (ShapeUtil::Equal(trt->transpose1_shape, trt->reshape_shape)) {
    normalized_reshape = normalized_input;
  } else {
    ASSIGN_OR_RETURN(
        normalized_reshape,
        EmitTiledBroadcastedReshape(b, trt->reshape_shape, reshape_tile_sizes,
                                    normalized_input));
  }

  // The final transpose simply uses the tile sizes computed for the original
  // bitcast by the tiling analysis.
  return trt->IsTranspose2Identity()
             ? normalized_reshape
             : EmitTiledTranspose(b, output_tile_sizes,
                                  llvm::to_vector(trt->transpose2_dims),
                                  normalized_reshape);
}

absl::Status EmitScanComputation(mlir::ImplicitLocOpBuilder& b,
                                 const HloInstruction* hlo_scan,
                                 const HloComputation* scan_computation,
                                 mlir::Operation* scan) {
  const auto* scan_instr = ::xla::Cast<HloScanInstruction>(hlo_scan);
  int num_operands = scan_instr->inputs().size();
  SmallVector<Type> result_tys;
  SmallVector<mlir::Location> locs;

  // The arguments to the scan combiner are (in_1, ..., in_N, acc_1, ...,
  // acc_N). First, add types for the inputs.
  for (int i = 0; i < num_operands; ++i) {
    const HloInstruction* input_hlo = hlo_scan->operand(i);
    ASSIGN_OR_RETURN(
        Type input_elem_type,
        PrimitiveTypeToMlirType(b, input_hlo->shape().element_type()));
    Type input_tensor_type = mlir::RankedTensorType::get({}, input_elem_type);
    result_tys.push_back(input_tensor_type);
    locs.push_back(b.getLoc());
  }

  // Next, add types for the accumulators (initial values).
  for (int i = 0; i < num_operands; ++i) {
    const HloInstruction* init_hlo = hlo_scan->operand(num_operands + i);
    ASSIGN_OR_RETURN(
        Type init_elem_type,
        PrimitiveTypeToMlirType(b, init_hlo->shape().element_type()));
    Type init_tensor_type = mlir::RankedTensorType::get({}, init_elem_type);
    result_tys.push_back(init_tensor_type);
    locs.push_back(b.getLoc());
  }

  mlir::Block* scanner =
      b.createBlock(&scan->getRegion(0), {}, result_tys, locs);
  b.setInsertionPointToStart(scanner);

  std::vector<const HloInstruction*> to_emit;
  absl::flat_hash_map<const HloInstruction*, TensorValue> region_values;
  for (const HloInstruction* instr :
       scan_computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      int parameter_number = instr->parameter_number();
      TF_RET_CHECK(parameter_number < num_operands * 2);
      auto argument =
          mlir::cast<TensorValue>(scanner->getArgument(parameter_number));

      if (!argument) {
        return Internal("Expected scanner argument to be a tensor.");
      }
      TF_RET_CHECK(region_values.insert({instr, argument}).second);
    } else {
      to_emit.push_back(instr);
    }
  }
  TF_RET_CHECK(!to_emit.empty());

  const HloInstruction* root_instr = scan_computation->root_instruction();
  if (root_instr->opcode() == HloOpcode::kTuple) {
    TF_RET_CHECK(to_emit.back() == root_instr);
    to_emit.pop_back();
  }

  auto status_or_result = EmitScope(b, to_emit, region_values);
  if (!status_or_result.ok()) {
    return status_or_result.status();
  }
  mlir::Value result = *status_or_result;

  SmallVector<Value> yielded_results;
  if (root_instr->opcode() == HloOpcode::kTuple) {
    for (const HloInstruction* operand : root_instr->operands()) {
      yielded_results.push_back(region_values[operand]);
    }
  } else {
    yielded_results.push_back(result);
    yielded_results.push_back(result);
  }

  stablehlo::ReturnOp::create(b, yielded_results);
  b.setInsertionPointAfter(scan);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<TensorValue>> EmitScan(
    EmitterContext& emitter_ctx,
    const ge::TiledHloInstruction& tiled_hlo_scan) {
  auto& b = emitter_ctx.b();
  const HloScanInstruction& hlo_scan =
      *::xla::Cast<HloScanInstruction>(tiled_hlo_scan.hlo());

  int num_operands = hlo_scan.inputs().size();
  SmallVector<Value> inputs;
  SmallVector<Value> inits;
  SmallVector<Type> carry_types;
  SmallVector<Type> output_types;

  ASSIGN_OR_RETURN(SmallVector<int64_t> unpadded_tile_sizes,
                   tiled_hlo_scan.operand(0)->tile().GetStaticTileSizes());

  for (int i = 0; i < num_operands; ++i) {
    const ge::TiledHloInstruction* input_operand = tiled_hlo_scan.operand(i);

    TensorValue input = emitter_ctx.TiledHloToTensorValue(*input_operand);

    llvm::SmallVector<int64_t> mask_dim_bounds;
    mask_dim_bounds.reserve(unpadded_tile_sizes.size());
    for (auto [idx, dim_size] : llvm::enumerate(unpadded_tile_sizes)) {
      if (idx == hlo_scan.scan_dimension()) {
        mask_dim_bounds.push_back(dim_size);
      } else {
        mask_dim_bounds.push_back(input.getType().getDimSize(idx));
      }
    }
    mlir::Value neutral_value = mlir::tensor::ExtractOp::create(
        b, emitter_ctx.TiledHloToTensorValue(
               *tiled_hlo_scan.operand(num_operands + i)));

    input = mlir::cast<TensorValue>(
        b.createOrFold<xtile::MaskOp>(input, mask_dim_bounds, neutral_value));

    TensorValue init = emitter_ctx.TiledHloToTensorValue(
        *tiled_hlo_scan.operand(num_operands + i));

    inputs.push_back(input);
    inits.push_back(init);
    carry_types.push_back(init.getType());
    output_types.push_back(input.getType());
  }

  auto scan = xtile::ScanOp::create(
      b, output_types, carry_types, inputs, inits, hlo_scan.scan_dimension(),
      unpadded_tile_sizes[hlo_scan.scan_dimension()], hlo_scan.is_reverse());

  RETURN_IF_ERROR(EmitScanComputation(b, &hlo_scan, hlo_scan.to_apply(), scan));

  std::vector<TensorValue> results;
  for (auto output : scan.getOutputs()) {
    results.push_back(mlir::cast<TensorValue>(output));
  }
  return results;
}

absl::StatusOr<TensorValue> EmitReduce(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_hlo) {
  if (tiled_hlo.hlo()->dimensions().size() != 1 ||
      tiled_hlo.hlo()->operand_count() != 2) {
    // Triton does support variadic reduce and reductions over multiple
    // dimensions but we don't support it here yet. For example, xtile.mask
    // only supports masking of at most one dimension. To support
    // multi-dimensional we should use a different method or update xtile.mask.
    return absl::InvalidArgumentError(absl::StrCat(
        "Only reduce with one dimension and two operands is supported. Got ",
        tiled_hlo.hlo()->dimensions().size(), " dimensions and ",
        tiled_hlo.hlo()->operand_count(), " operands."));
  }
  ImplicitLocOpBuilder& b = emitter_ctx.b();
  const HloReduceInstruction& reduce_hlo =
      *::xla::Cast<HloReduceInstruction>(tiled_hlo.hlo());
  const ge::TiledHloInstruction* tiled_input = tiled_hlo.operand(0);
  TensorValue input_value = emitter_ctx.TiledHloToTensorValue(*tiled_input);
  ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> mask_dim_bounds,
                   tiled_input->tile().GetStaticTileSizes());
  int64_t reduce_dim = reduce_hlo.dimensions()[0];
  mask_dim_bounds[reduce_dim] =
      tiled_input->hlo()->shape().dimensions(reduce_dim);
  TensorValue init_value =
      emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(1));
  // N.B.: while that mostly works in practice, there are valid HLOs, for
  // example `reduce(p0, init=1), to_apply=add`, that will produce the wrong
  // result with this implementation.
  mlir::Value neutral_value = mlir::tensor::ExtractOp::create(b, init_value);
  input_value = mlir::cast<TensorValue>(b.createOrFold<xtile::MaskOp>(
      input_value, mask_dim_bounds, neutral_value));
  stablehlo::ReduceOp reduction = stablehlo::ReduceOp::create(
      b, input_value, init_value, reduce_hlo.dimensions());
  RETURN_IF_ERROR(EmitReduceComputation(
      b, &reduce_hlo, tiled_hlo.hlo()->to_apply(), reduction));
  return mlir::cast<TensorValue>(reduction.getResult(0));
}

absl::StatusOr<TensorValue> EmitTiledHloInstruction(
    EmitterContext& emitter_ctx, const ge::TiledHloInstruction& tiled_hlo) {
  auto& b = emitter_ctx.b();
  const HloInstruction* hlo = tiled_hlo.hlo();
  VLOG(4) << "EmitTiledHloInstruction: " << hlo->ToString();

  const HloFusionInstruction& fusion = emitter_ctx.fusion();
  if (hlo->opcode() == HloOpcode::kParameter && !fusion.IsUserOf(hlo)) {
    hlo = hlo->parent()->FusionInstruction()->operand(hlo->parameter_number());
  }

  if (fusion.IsUserOf(hlo)) {
    int64_t arg_index = fusion.operand_index(hlo);
    // Walk up the parameter chain to find the outermost operand index.
    while (auto* instr = hlo->parent()->FusionInstruction()) {
      arg_index = hlo->parameter_number();  // Nested operands are parameters.
      hlo = instr->operand(arg_index);
    }
    ASSIGN_OR_RETURN(TileInfo tile_info,
                     TileInfo::Construct(emitter_ctx, tiled_hlo));
    ASSIGN_OR_RETURN(
        TensorValue parameter,
        EmitParameterExtract(b, tile_info,
                             emitter_ctx.entry_func().getArgument(arg_index)));

    // Workaround(i1_to_i8_workaround)
    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to type checking that we perform a conversion after
    // loading if the type of the loaded parameter does not match what is
    // expected.
    Type loaded_element_type = getElementTypeOrSelf(parameter.getType());
    ASSIGN_OR_RETURN(Type expected_element_type,
                     PrimitiveTypeToMlirType(b, hlo->shape().element_type()));

    if (expected_element_type != loaded_element_type) {
      // Ensure that we didn't mess up somewhere else by checking that we
      // indeed loaded the expected storage type for the expected element type.
      if (loaded_element_type != StorageType(expected_element_type)) {
        return absl::InternalError(absl::StrCat(
            "Parameters were loaded with an unexpected element type "
            "while lowering ",
            fusion.called_computation()->ToString()));
      }
      parameter =
          mlir::cast<TensorValue>(Cast(b, parameter, expected_element_type));
    }
    return parameter;
  }
  if (hlo->opcode() == HloOpcode::kDot) {
    return EmitDot(emitter_ctx, tiled_hlo);
  }
  if (hlo->opcode() == HloOpcode::kScaledDot) {
    return EmitScaledDot(emitter_ctx, tiled_hlo);
  }
  if (hlo->opcode() == HloOpcode::kRaggedDot) {
    return EmitRaggedDot(emitter_ctx, tiled_hlo);
  }
  if (hlo->opcode() == HloOpcode::kConcatenate) {
    return EmitConcatenate(emitter_ctx, tiled_hlo);
  }
  if (hlo->opcode() == HloOpcode::kGetTupleElement) {
    int64_t index = hlo->tuple_index();
    if (index == 0) {
      return emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0));
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported get-tuple-element index ", index));
  }
  std::vector<Value> operands;
  operands.reserve(hlo->operands().size());
  for (const ge::TiledHloInstruction* operand : tiled_hlo.operands()) {
    operands.push_back(emitter_ctx.TiledHloToTensorValue(*operand));
  }
  // Please keep the cases in alphabetical order.
  switch (hlo->opcode()) {
    case HloOpcode::kAllGather: {
      // AllGather is a no-op. Tile extraction handles the data movement.
      return emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0));
    }
    case HloOpcode::kAllReduce: {
      const HloComputation* computation =
          fusion.fused_instructions_computation();
      const HloInstruction* root_instruction = computation->root_instruction();
      return EmitAllReduce(emitter_ctx,
                           xla::Cast<HloAllReduceInstruction>(root_instruction),
                           tiled_hlo, operands);
    }
    case HloOpcode::kBitcast: {
      return EmitBitcast(emitter_ctx, tiled_hlo,
                         mlir::cast<TensorValue>(operands[0]));
    }
    case HloOpcode::kBroadcast: {
      return EmitBroadcast(b, tiled_hlo, mlir::cast<TensorValue>(operands[0]));
    }
    case HloOpcode::kConstant: {
      if (ShapeUtil::IsEffectiveScalar(hlo->shape())) {
        ASSIGN_OR_RETURN(auto tile_sizes,
                         tiled_hlo.tile().GetStaticTileSizes());
        return EmitConstant(b, *hlo, GetPaddedTileSizes(tile_sizes));
      }
      return absl::UnimplementedError(
          absl::StrCat("Unsupported non-scalar constant ", hlo->ToString()));
    }
    case HloOpcode::kDynamicSlice: {
      return emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0));
    }
    case HloOpcode::kIota: {
      return EmitIota(emitter_ctx, tiled_hlo);
    }
    case HloOpcode::kPad: {
      return EmitPad(emitter_ctx, tiled_hlo);
    }
    case HloOpcode::kReshape: {
      ASSIGN_OR_RETURN(auto tile_sizes, tiled_hlo.tile().GetStaticTileSizes());
      return EmitTiledReshape(
          emitter_ctx.b(), tile_sizes,
          emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0)));
    }
    case HloOpcode::kSlice: {
      return emitter_ctx.TiledHloToTensorValue(*tiled_hlo.operand(0));
    }
    case HloOpcode::kTranspose: {
      ASSIGN_OR_RETURN(auto tile_sizes, tiled_hlo.tile().GetStaticTileSizes());
      return EmitTranspose(b, tile_sizes, hlo->dimensions(),
                           mlir::cast<TensorValue>(operands[0]));
    }
    case HloOpcode::kReduce: {
      return EmitReduce(emitter_ctx, tiled_hlo);
    }
    case HloOpcode::kScan: {
      ASSIGN_OR_RETURN(auto result, EmitScan(emitter_ctx, tiled_hlo));
      return result.front();
    }
    default:
      break;
  }
  if (hlo->IsElementwise()) {
    ASSIGN_OR_RETURN(Value result, EmitElementwise(b, *hlo, operands));
    return mlir::cast<TensorValue>(result);
  }
  return absl::UnimplementedError(
      absl::StrCat("Unsupported operation ", hlo->ToString()));
}

absl::StatusOr<std::vector<TensorValue>> EmitTiledComputation(
    EmitterContext& emitter_ctx, const ge::TiledHloRegion& region,
    absl::Span<const ge::TiledHloInstruction* const> roots) {
  for (const auto& tiled_hlo : region.instructions()) {
    const HloInstruction* hlo = tiled_hlo->hlo();
    VLOG(8) << "Emitting " << hlo->ToString(HloPrintOptions::ShortParsable());
    ASSIGN_OR_RETURN(TensorValue result,
                     EmitTiledHloInstruction(emitter_ctx, *tiled_hlo));
    TF_RET_CHECK(emitter_ctx.MapTiledHloToTensorValue(tiled_hlo.get(), result))
        << hlo->ToString();
  }
  std::vector<TensorValue> results;
  results.reserve(roots.size());
  for (const auto* root : roots) {
    results.push_back(emitter_ctx.TiledHloToTensorValue(*root));
  }
  VLOG(8) << "Emitted computation";
  return std::move(results);
}

// Emit values for trivial sequential dimensions, i.e. dimensions with tile size
// greater than or equal to the dimension size.
// Tiling analysis still creates a dimension for such contracting dimensions but
// their parent instructions will not have regions and thus we don't emit their
// operands as part of them. As a concrete example, consider the following
// reduction:
//
// fusion {
//   p = f32[5,3] parameter(0)
//   c = f32[] constant(10)
//   ROOT reduce = f32[3] reduce(p, c), dimensions={0}, to_apply=maximum
// }
//
// If reduction tile covers the entire dimension then we will not have a
// computation of [reduce {region=[p, c]}] but rather a list of
// [p, c, reduce], where p has a symbol dimension that is created by reduce.
// To emit p we have to have a value for the symbol dimension.
// Thus we emit sequential dimensions at the start as we know they will be
// trivially 0.
void EmitFullyTiledSequentialDimensions(
    ImplicitLocOpBuilder& b, EmitterContext& emitter_ctx,
    const ge::TiledHloComputation& tiled_computation) {
  const auto& tiling_space = tiled_computation.tiling_space();
  for (const auto& [dim_id, dim_info] :
       llvm::enumerate(tiling_space.dimensions())) {
    if (dim_info.type != ge::TilingSpace::DimensionSemantics::kSequential) {
      continue;
    }
    QCHECK(dim_info.hlo != nullptr) << "Sequential dimension " << dim_id
                                    << " does not have a corresponding "
                                       "HLO.";
    QCHECK(dim_info.tile_size.has_value()) << "Sequential dimension " << dim_id
                                           << " does not have a tile size set.";
    if (dim_info.hlo->opcode() == HloOpcode::kReduce &&
        *dim_info.tile_size >= dim_info.dimension_size) {
      VLOG(2) << "Mapping reduce sequential dimension " << dim_id << " of size "
              << dim_info.dimension_size << " with tile size "
              << *dim_info.tile_size << " for hlo " << dim_info.hlo->name()
              << " to a new value 0";
      emitter_ctx.MapSymbolIdToSequentialDimValue(
          ge::TiledDimId(dim_id), MakeIndex(b, 0), Interval{0, 0});
    }
  }
}

// Applies L2 tile reordering to the flat tile_id for kRaggedNonContracting and
// kRaggedContracting ragged-dot fusions that set
// BlockLevelFusionConfig.group_size > 1.
//
// kRaggedNonContracting: reorders the (M, N) tile enumeration so that
// `group_size` consecutive M-tiles share the same N-tile before the next
// N-tile group begins, keeping the RHS column block hot in L2:
//
//   num_pid_in_group = group_size * num_pid_n
//   group_id         = pid // num_pid_in_group
//   first_pid_m      = group_id * group_size
//   group_size_m     = min(num_pid_m - first_pid_m, group_size)
//   pid_m            = first_pid_m + (pid % num_pid_in_group) % group_size_m
//   pid_n            = (pid % num_pid_in_group) // group_size_m
//   remapped_pid     = pid_m * num_pid_n + pid_n
//
// kRaggedContracting: applies the same algorithm over the 3-D grid
// (G × K_tiles × N_tiles), grouping G-slices so that `group_size` consecutive
// G programs share the same (K_tile, N_tile) pair, keeping the RHS block hot
// in L2 across groups.  G plays the role of M and
// KN_tiles = K_tiles * N_tiles plays the role of N:
//
//   num_pid_in_group = group_size * KN_tiles
//   group_id         = pid // num_pid_in_group
//   first_pid_g      = group_id * group_size
//   group_size_g     = min(num_pid_g - first_pid_g, group_size)
//   pid_g            = first_pid_g + (pid % num_pid_in_group) % group_size_g
//   pid_kn           = (pid % num_pid_in_group) // group_size_g
//   remapped_pid     = pid_g * KN_tiles + pid_kn
//
// Returns `raw_tile_id` unchanged when the fusion does not qualify (batch dims
// present, group_size <= 1, or non-3D kRaggedContracting output tile).
Value ApplyGroupSizeTileIdRemapping(ImplicitLocOpBuilder& b,
                                    const HloFusionInstruction& fusion,
                                    Value raw_tile_id) {
  // Read group_size from the fusion backend config.
  auto gpu_config_or = fusion.backend_config<xla::gpu::GpuBackendConfig>();
  if (!gpu_config_or.ok()) return raw_tile_id;
  const xla::gpu::GpuBackendConfig& gpu_config = *gpu_config_or;
  if (!gpu_config.fusion_backend_config().has_block_level_fusion_config()) {
    return raw_tile_id;
  }
  const xla::gpu::BlockLevelFusionConfig& blk_cfg =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  const int gs = std::max(1, blk_cfg.group_size());
  if (gs <= 1) return raw_tile_id;

  // Find the kRaggedDot inside the fusion.
  const HloComputation* comp = fusion.fused_instructions_computation();
  const HloRaggedDotInstruction* rd = nullptr;
  for (const HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() == HloOpcode::kRaggedDot) {
      rd = ::xla::Cast<HloRaggedDotInstruction>(instr);
      break;
    }
  }
  if (rd == nullptr) return raw_tile_id;

  const auto& rdims = rd->ragged_dot_dimension_numbers();
  const auto& ddims = rdims.dot_dimension_numbers();
  const int64_t ragged_lhs = rdims.lhs_ragged_dimensions(0);
  const bool is_contracting =
      absl::c_count(ddims.lhs_contracting_dimensions(), ragged_lhs) > 0;
  const bool is_batch =
      absl::c_count(ddims.lhs_batch_dimensions(), ragged_lhs) > 0;

  // Batch variants are not supported for either remapping.
  if (is_batch) return raw_tile_id;

  if (!is_contracting) {
    // ---- kRaggedNonContracting: (M, N) 2-D grid grouping ----
    // Only handle the no-batch, 2-D grid case ([M, N] output tile).
    if (blk_cfg.output_tiles_size() < 1 ||
        blk_cfg.output_tiles(0).sizes_size() != 2) {
      return raw_tile_id;
    }

    // Grid dimensions: num_pid_m × num_pid_n programs.
    const int64_t BLOCK_M = blk_cfg.output_tiles(0).sizes(0);
    const int64_t BLOCK_N = blk_cfg.output_tiles(0).sizes(1);
    const int64_t M_total = rd->shape().dimensions(0);
    const int64_t N_total = rd->shape().dimensions(1);
    const int64_t num_pid_m = (M_total + BLOCK_M - 1) / BLOCK_M;
    const int64_t num_pid_n = (N_total + BLOCK_N - 1) / BLOCK_N;

    if (num_pid_m <= 1 || num_pid_n <= 0)
      return raw_tile_id;  // Nothing to reorder.

    // Cast pid to i32 for arithmetic (Triton program IDs are 32-bit).
    Value pid = Cast(b, raw_tile_id, b.getI32Type());
    auto ci = [&](int64_t v) { return CreateConst(b, b.getI32Type(), v); };
    Value G = ci(gs);
    Value npm = ci(num_pid_m);
    Value npn = ci(num_pid_n);
    // num_pid_in_group = group_size * num_pid_n
    Value g_npn = arith::MulIOp::create(b, G, npn);
    // group_id = pid / (group_size * num_pid_n)
    Value group_id = arith::DivSIOp::create(b, pid, g_npn);
    // first_pid_m = group_id * group_size
    Value first_pid_m = arith::MulIOp::create(b, group_id, G);
    // rem_in_group = pid % (group_size * num_pid_n)
    Value rem = arith::RemSIOp::create(b, pid, g_npn);
    // group_size_m = min(num_pid_m - first_pid_m, group_size)
    Value npm_minus_first = arith::SubIOp::create(b, npm, first_pid_m);
    Value group_size_m = arith::MinSIOp::create(b, npm_minus_first, G);
    // pid_m = first_pid_m + (rem % group_size_m)
    Value pid_m = arith::AddIOp::create(
        b, first_pid_m, arith::RemSIOp::create(b, rem, group_size_m));
    // pid_n = rem / group_size_m
    Value pid_n = arith::DivSIOp::create(b, rem, group_size_m);
    // remapped flat index = pid_m * num_pid_n + pid_n
    Value remapped =
        arith::AddIOp::create(b, arith::MulIOp::create(b, pid_m, npn), pid_n);
    // Cast back to index type for EmitterContext.
    return arith::IndexCastOp::create(b, b.getIndexType(), remapped);
  }

  // ---- kRaggedContracting: (G, KN) 3-D grid grouping.
  // Output tile = [G=1, BLOCK_K, BLOCK_N] → sizes_size() == 3.
  if (blk_cfg.output_tiles_size() < 1 ||
      blk_cfg.output_tiles(0).sizes_size() != 3) {
    return raw_tile_id;
  }

  // Grid: G × K_tiles × N_tiles programs.
  // G plays the role of M, KN_tiles = K_tiles * N_tiles plays the role of N.
  // rd->shape() = (G, K_output, N_output).
  const int64_t BLOCK_K = blk_cfg.output_tiles(0).sizes(1);
  const int64_t BLOCK_N = blk_cfg.output_tiles(0).sizes(2);
  const int64_t num_pid_g = rd->shape().dimensions(0);
  const int64_t K_output = rd->shape().dimensions(1);
  const int64_t N_output = rd->shape().dimensions(2);
  const int64_t K_tiles = (K_output + BLOCK_K - 1) / BLOCK_K;
  const int64_t N_tiles = (N_output + BLOCK_N - 1) / BLOCK_N;
  const int64_t num_pid_kn = K_tiles * N_tiles;

  if (num_pid_g <= 1 || num_pid_kn <= 0)
    return raw_tile_id;  // Nothing to reorder.

  // Cast pid to i32 for arithmetic (Triton program IDs are 32-bit).
  Value pid = Cast(b, raw_tile_id, b.getI32Type());
  auto ci = [&](int64_t v) { return CreateConst(b, b.getI32Type(), v); };
  Value GS = ci(gs);
  Value npg = ci(num_pid_g);
  Value npkn = ci(num_pid_kn);
  // num_pid_in_group = group_size * num_pid_kn
  Value gs_npkn = arith::MulIOp::create(b, GS, npkn);
  // group_id = pid / (group_size * num_pid_kn)
  Value group_id = arith::DivSIOp::create(b, pid, gs_npkn);
  // first_pid_g = group_id * group_size
  Value first_pid_g = arith::MulIOp::create(b, group_id, GS);
  // rem_in_group = pid % (group_size * num_pid_kn)
  Value rem = arith::RemSIOp::create(b, pid, gs_npkn);
  // group_size_g = min(num_pid_g - first_pid_g, group_size)
  Value npg_minus_first = arith::SubIOp::create(b, npg, first_pid_g);
  Value group_size_g = arith::MinSIOp::create(b, npg_minus_first, GS);
  // pid_g = first_pid_g + (rem % group_size_g)
  Value pid_g = arith::AddIOp::create(
      b, first_pid_g, arith::RemSIOp::create(b, rem, group_size_g));
  // pid_kn = rem / group_size_g
  Value pid_kn = arith::DivSIOp::create(b, rem, group_size_g);
  // remapped flat index = pid_g * num_pid_kn + pid_kn
  Value remapped =
      arith::AddIOp::create(b, arith::MulIOp::create(b, pid_g, npkn), pid_kn);
  // Cast back to index type for EmitterContext.
  return arith::IndexCastOp::create(b, b.getIndexType(), remapped);
}

absl::Status EmitGeneric(ImplicitLocOpBuilder& b,
                         const HloFusionInstruction& fusion,
                         const ge::TiledHloComputation& tiled_computation,
                         const ge::Schedule& schedule, xtile::EntryFuncOp fn,
                         MLIRContext* mlir_context) {
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Emitting XTile IR for fusion\n"
            << ExtractInstructionIntoNewModule(fusion)->ToString();
    VLOG(6) << "Tiled computation: \n" << tiled_computation.ToString();
  }
  Value program_id = fn.getProgramId();
  Value tile_id = program_id;

  // If there are more than one tile per pid, we need to add a scf.for loop to
  // iterate through the tiles.
  int64_t num_tiles_per_pid = schedule.GetNumTilesPerPid();
  if (num_tiles_per_pid > 1) {
    Value zero = arith::ConstantIndexOp::create(b, 0);
    Value one = arith::ConstantIndexOp::create(b, 1);
    Value num_tiles_per_pid_val =
        arith::ConstantIndexOp::create(b, num_tiles_per_pid);

    // Loop ub = min(num_tiles_per_pid, num_tiles - pid * num_tiles_per_pid).
    Value upper_bound = arith::SubIOp::create(
        b, arith::ConstantIndexOp::create(b, schedule.num_tiles),
        arith::MulIOp::create(b, program_id, num_tiles_per_pid_val));
    upper_bound = arith::MinUIOp::create(b, upper_bound, num_tiles_per_pid_val);
    auto for_op = mlir::scf::ForOp::create(b, zero, num_tiles_per_pid_val, one);

    tile_id = arith::AddIOp::create(
        b, arith::MulIOp::create(b, program_id, num_tiles_per_pid_val),
        for_op.getInductionVar());
    b.setInsertionPointToStart(for_op.getBody());
  }
  // Apply GROUP_SIZE L2 tile reordering for kRaggedNonContracting and
  // kRaggedContracting ragged-dot fusions where
  // BlockLevelFusionConfig.group_size > 1.  The remapping transforms the flat
  // program_id to an L2-friendly tile coordinate before EmitterContext is
  // constructed, so that all EvaluateTilingParameters calls automatically yield
  // the reordered tile coordinates.
  tile_id = ApplyGroupSizeTileIdRemapping(b, fusion, fn.getProgramId());
  EmitterContext emitter_ctx{b,        &fusion, program_id,       tile_id,
                             schedule, fn,      tiled_computation};

  VLOG(2) << "EmitTiledComputation: " << tiled_computation.ToString();
  EmitFullyTiledSequentialDimensions(b, emitter_ctx, tiled_computation);
  ASSIGN_OR_RETURN(
      auto results,
      EmitTiledComputation(emitter_ctx, tiled_computation.tiled_root_region(),
                           tiled_computation.roots()));
  const HloComputation* computation = fusion.fused_instructions_computation();
  for (const auto& [root, result, arg] :
       llvm::zip(tiled_computation.roots(), results,
                 fn.getArguments().drop_front(computation->num_parameters()))) {
    // Workaround(i1_to_i8_workaround)
    // Some types are stored using different types, e.g. i1 is stored in memory
    // as i8. It's important to check converted types before storing if the type
    // of the result does not match the type of the output pointer.
    Type result_element_type = getElementTypeOrSelf(result.getType());
    Type result_storage_type = StorageType(result_element_type);

    if (result_element_type != result_storage_type) {
      result = mlir::cast<TensorValue>(Cast(b, result, result_storage_type));
    }

    ASSIGN_OR_RETURN(auto tile_info, TileInfo::Construct(emitter_ctx, *root));

    xtile::InsertTileOp::create(b, result, arg, tile_info.offsets(),
                                tile_info.padded_tile_sizes(),
                                tile_info.tile_strides());
  }

  return absl::OkStatus();
}

// Implementation for the experimental tiling space.
class TileRequirementsVisitor : public DefaultTileRequirementsVisitor {
 public:
  explicit TileRequirementsVisitor(const ge::TiledHloComputation& computation) {
    for (const ge::TiledHloInstruction* tiled_hlo :
         computation.instructions()) {
      PopulateMap(tiled_hlo);
    }
  }

  absl::StatusOr<llvm::SmallVector<int64_t>> RequiredReplicaIdBounds(
      const HloInstruction& instr) const override {
    ASSIGN_OR_RETURN(auto tiled_hlo, LookupTiledHlo(&instr));
    llvm::SmallVector<int64_t> bounds;
    bounds.reserve(tiled_hlo->tile().replica_ids().size());
    for (const auto& replica_id : tiled_hlo->tile().replica_ids()) {
      SymbolicExpr upper_bound = replica_id.upper_bound.Canonicalize();
      if (upper_bound.GetType() != SymbolicExprType::kConstant) {
        return absl::InternalError(
            absl::StrCat("Replica ID bound expression is not a constant: ",
                         upper_bound.ToString()));
      }
      bounds.push_back(upper_bound.GetValue());
    }
    return bounds;
  }

 private:
  // Look up the instruction in the tiled HLO map.
  // For parameters to nested fusions, we walk up the parameter chain to find
  // the outermost operand index.
  absl::StatusOr<const ge::TiledHloInstruction*> LookupTiledHlo(
      const HloInstruction* original_instr) const {
    auto it = hlo_to_tiled_.find(original_instr);
    if (it != hlo_to_tiled_.end()) {
      return it->second;
    }
    if (original_instr->opcode() == HloOpcode::kParameter) {
      if (auto* fusion = original_instr->parent()->FusionInstruction()) {
        const HloInstruction* resolved_instr =
            fusion->operand(original_instr->parameter_number());
        return LookupTiledHlo(resolved_instr);
      }
    }
    return absl::InternalError(absl::StrCat(
        "InternalError: HLO instruction not found in tiled HLO map: ",
        original_instr->ToString()));
  }

  void PopulateMap(const ge::TiledHloInstruction* tiled_hlo) {
    hlo_to_tiled_[tiled_hlo->hlo()] = tiled_hlo;
    for (const auto& region : tiled_hlo->hlo_regions()) {
      for (const auto& region_instruction : region.instructions()) {
        PopulateMap(region_instruction.get());
      }
    }
  }

  absl::flat_hash_map<const HloInstruction*,
                      const xla::gpu::experimental::TiledHloInstruction*>
      hlo_to_tiled_;
};

}  // namespace

// TODO(b/447133106): Contrary to the name, this function still does a lot of
// triton specific things. It should be migrated to use non-triton specific
// utilities.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const ::xla::gpu::experimental::TiledHloComputation& tiled_computation,
    MLIRContext& mlir_context, absl::Span<mlir::Type> opaque_args_types,
    const std::optional<GpuComputeCapability>& gpu_cc) {
  const HloComputation* hlo_computation =
      fusion.fused_instructions_computation();

  Location loc = mlir::NameLoc::get(
      mlir::StringAttr::get(&mlir_context, hlo_computation->name()));
  ImplicitLocOpBuilder b(loc, &mlir_context);

  mlir::OwningOpRef<mlir::ModuleOp> xtile_module =
      llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(xtile_module->getBody());

  // Compute function argument types.
  ASSIGN_OR_RETURN(SmallVector<Type> fn_arg_types,
                   GetFnArgTypes(b, fusion, opaque_args_types, gpu_cc,
                                 TileRequirementsVisitor(tiled_computation)));
  // Metadata arguments are opaque to the tiling infra.
  llvm::SmallVector<mlir::NamedAttribute> named_attributes{b.getNamedAttr(
      "num_opaque_args", b.getI32IntegerAttr(opaque_args_types.size()))};

  auto fn = xtile::EntryFuncOp::create(b, fn_name, fn_arg_types,
                                       named_attributes, {});
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  ASSIGN_OR_RETURN(auto schedule, GetSchedule(tiled_computation));
  RETURN_IF_ERROR(
      EmitGeneric(b, fusion, tiled_computation, schedule, fn, &mlir_context));

  b.create<xtile::EntryFuncReturnOp>();
  if (VLOG_IS_ON(8)) {
    std::string s;
    llvm::raw_string_ostream os(s);
    xtile_module->print(os);
    XLA_VLOG_LINES(8, s);
  }
  // This should be enabled only in debug mode probably.
  {
    // Verify that the emitted module contains only ops from dialects that can
    // be shared between backends.
    mlir::PassManager pm(&mlir_context);
    pm.addPass(xtile::createVerifyLegalXTileOpsPass());
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    RETURN_IF_ERROR(diagnostic_handler.consumeStatus(pm.run(*xtile_module)));
  }
  return xtile_module;
}

}  // namespace xla::xtile
