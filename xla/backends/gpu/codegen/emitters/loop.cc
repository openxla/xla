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
#include "xla/backends/gpu/codegen/emitters/loop.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;

const Shape& GetIndexShape(const Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes(0) : shape;
}

}  // namespace

std::optional<IndexingMap> LoopFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  auto launch_dims = launch_dimensions();
  return GetDefaultThreadIdIndexingMap(
      launch_dims, config_.unroll_factor,
      GetIndexShape(analysis_.fusion_root(root_index).shape()), ctx);
}

std::optional<IndexingMap> LoopFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  std::optional<IndexingMap> thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, ctx);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* fusion_root =
      &analysis_.fusion_root(root_index).instruction();
  auto output_to_input_indexing =
      ComputeOutputToInputIndexing(fusion_root, /*output_id=*/0, ctx);
  IndexingMapSet output_to_input_indexing_set =
      output_to_input_indexing.indexing_maps[hero_operand_index];
  // Since we are computing the indexing for a non-fusion op, there is only one
  // indexing map per operand.
  CHECK_EQ(output_to_input_indexing_set.size(), 1);
  IndexingMap thread_id_to_input_indexing_map = ComposeIndexingMaps(
      *thread_id_to_output_indexing, *output_to_input_indexing_set.begin());
  thread_id_to_input_indexing_map.Simplify();
  return thread_id_to_input_indexing_map;
}

LaunchDimensions LoopFusion::launch_dimensions() const {
  return CalculateLaunchDimensions(
      GetIndexShape(analysis_.fusion_root(0).shape()), analysis_.device_info(),
      config_);
}

absl::Status LoopFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  mlir::MLIRContext* context = builder.getContext();

  auto workgroup_ids = EmitWorkGroupIds(builder);

  auto indexing =
      ComputeThreadIdToOutputIndexing(0, entry_function.getContext());
  TF_RET_CHECK(indexing) << "Indexing is never nullopt";

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);
  llvm::SmallVector<const Shape*> result_shapes;
  for (const HloInstructionAdaptor& root : analysis_.fusion_roots()) {
    if (root.shape().IsTuple()) {
      for (const auto& shape : root.shape().tuple_shapes()) {
        result_shapes.push_back(&shape);
      }
    } else {
      result_shapes.push_back(&root.shape());
    }
  }

  auto body_builder = [&](ImplicitLocOpBuilder& nested_b,
                          ValueRange symbol_values, ValueRange map_results,
                          ValueRange output_tensors) -> SmallVector<Value> {
    auto root_fn = call_targets(
        fusion.fused_instructions_computation()->root_instruction());
    // Generate the operands for the root function: input tensors +
    // output indices.
    SmallVector<Value> operands(
        entry_function.getArguments().take_front(num_inputs));
    absl::c_copy(map_results, std::back_inserter(operands));
    auto result_scalars =
        nested_b.create<PureCallOp>(root_fn, operands).getResults();

    SmallVector<Value> result_tensors;
    result_tensors.reserve(output_tensor_args.size());
    for (auto [root_shape, tensor, value] :
         llvm::zip(result_shapes, output_tensors, result_scalars)) {
      llvm::SmallVector<Value> output_indices = emitters::ApplyIndexing(
          GetBitcastMap(*result_shapes.front(), *root_shape, context),
          map_results, {}, nested_b);
      result_tensors.push_back(nested_b.create<mlir::tensor::InsertOp>(
          value, tensor, output_indices));
    }
    return result_tensors;
  };

  const auto loop_builder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::ValueRange workitem_ids_and_outputs) {
    ImplicitLocOpBuilder nested_b(loc, builder);

    mlir::ValueRange workitem_ids = workitem_ids_and_outputs.take_front(3);
    mlir::ValueRange outputs = workitem_ids_and_outputs.drop_front(3);

    llvm::SmallVector<Value, 6> work_dims;
    work_dims.insert(work_dims.end(), workitem_ids.begin(), workitem_ids.end());
    work_dims.insert(work_dims.end(), workgroup_ids.begin(),
                     workgroup_ids.end());

    auto loop_results =
        emitters::EmitXlaLoopOp(nested_b, mlir::ValueRange(work_dims), outputs,
                                *indexing, body_builder);
    auto terminator = nested_b.create<mlir::scf::InParallelOp>();
    nested_b.setInsertionPointToStart(terminator.getBody());
    for (auto [result, output] : llvm::zip(loop_results, outputs)) {
      auto output_tensor = mlir::cast<mlir::RankedTensorType>(output.getType());
      llvm::SmallVector<mlir::OpFoldResult> offsets(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(0));
      llvm::SmallVector<mlir::OpFoldResult> sizes =
          mlir::getAsIndexOpFoldResult(context, output_tensor.getShape());
      llvm::SmallVector<mlir::OpFoldResult> strides(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(1));
      nested_b.create<mlir::tensor::ParallelInsertSliceOp>(
          result, output, offsets, sizes, strides);
    }
  };

  const auto& counts = launch_dimensions().thread_counts_per_block();
  SmallVector<mlir::OpFoldResult> upper_bounds = mlir::getAsIndexOpFoldResult(
      context, {static_cast<int64_t>(counts.x), static_cast<int64_t>(counts.y),
                static_cast<int64_t>(counts.z)});
  builder.create<mlir::func::ReturnOp>(
      builder
          .create<mlir::scf::ForallOp>(upper_bounds, output_tensor_args,
                                       std::nullopt, loop_builder)
          .getResults());

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
