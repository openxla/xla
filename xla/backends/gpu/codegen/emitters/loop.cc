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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/loop_kernel_emitter.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::MLIRContext;

const Shape& GetIndexShape(const Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes(0) : shape;
}

// On gfx1151, a transpose loop fusion that keeps the minor dimension reads
// operand rows blocked_size * row_bytes apart. When that stride has a large
// power-of-two factor, the reads hit few memory channels. Visiting
// ~kContiguousReadBytes of adjacent rows together spreads them out.
// Both constants were measured on gfx1151.
constexpr int64_t kContiguousReadBytes = 4096;
constexpr int64_t kMinStrideAlignmentBytes = 512;

// Returns the traversal order for a loop fusion with such a transpose, or
// nullopt to keep the memory order.
std::optional<emitters::BlockedDimensionOrder> GetTransposeFriendlyOrder(
    const HloFusionAnalysis& analysis, int unroll_factor) {
  const stream_executor::RocmComputeCapability* rocm =
      analysis.device_info().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || rocm->gfx_version() != "gfx1151") {
    return std::nullopt;
  }

  const Shape indexing_shape =
      emitters::LoopFusionKernelEmitter::GetIndexingShape(
          analysis.fusion_spec());
  // The order is defined on the output of the transpose. Only look at
  // transposes whose output has the shape of the elements the work items
  // enumerate, i.e. that are only followed by elementwise operations.
  std::optional<HloInstructionAdaptor> hero = FindHero(
      analysis.fusion_root(0), [](const HloInstruction& instruction) {
        return instruction.opcode() == HloOpcode::kTranspose &&
               !ShapeUtil::TransposeIsBitcast(instruction.operand(0)->shape(),
                                              instruction.shape(),
                                              instruction.dimensions());
      });
  if (!hero.has_value() ||
      !ShapeUtil::EqualIgnoringElementType(hero->shape(), indexing_shape)) {
    return std::nullopt;
  }

  const HloInstruction& transpose = hero->instruction();
  absl::InlinedVector<int64_t, 3> permutation;
  absl::StatusOr<absl::InlinedVector<int64_t, 3>> dims =
      ShapeUtil::GetNormalizedLogicalTransposeShape(
          transpose.operand(0)->shape(), transpose.shape(),
          transpose.dimensions(), permutation);
  if (!dims.ok()) {
    return std::nullopt;
  }

  // Only handle the case where two adjacent dimensions are swapped and the
  // minor dimension stays in place: [..., J, I, K] <- [..., I, J, K].
  const int64_t rank = dims->size();
  if (rank < 3 || permutation.back() != rank - 1 ||
      permutation[rank - 3] != rank - 2 || permutation[rank - 2] != rank - 3) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < rank - 3; ++i) {
    if (permutation[i] != i) {
      return std::nullopt;
    }
  }

  PrimitiveType element_type = transpose.shape().element_type();
  if (!primitive_util::IsArrayType(element_type) ||
      primitive_util::BitWidth(element_type) % 8 != 0) {
    return std::nullopt;
  }
  const int64_t blocked_size = (*dims)[rank - 3];
  const int64_t middle_size = (*dims)[rank - 2];
  const int64_t minor_size = (*dims)[rank - 1];
  // Every work item has to process consecutive elements of one row.
  if (minor_size % unroll_factor != 0) {
    return std::nullopt;
  }

  const int64_t row_bytes =
      minor_size * primitive_util::ByteWidth(element_type);
  const int64_t stride_bytes = blocked_size * row_bytes;
  // Rows that are not a multiple of kMinStrideAlignmentBytes apart do not
  // collide.
  if (stride_bytes % kMinStrideAlignmentBytes != 0) {
    return std::nullopt;
  }

  // Number of rows that make up `kContiguousReadBytes`, rounded down to a power
  // of two, limited to a power of two that divides the size of the dimension.
  int64_t block_size = absl::bit_floor(
      static_cast<uint64_t>(std::max<int64_t>(kContiguousReadBytes / row_bytes,
                                              1)));
  block_size = std::min<int64_t>(
      block_size, static_cast<int64_t>(blocked_size & -blocked_size));
  if (block_size < 2) {
    return std::nullopt;
  }
  return emitters::BlockedDimensionOrder{blocked_size, block_size, middle_size,
                                         minor_size};
}

}  // namespace

LoopFusion::LoopFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      unroll_factor_(ComputeLoopFusionConfig(analysis)),
      order_(GetTransposeFriendlyOrder(analysis, unroll_factor_)) {}

std::optional<IndexingMap> LoopFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  return emitters::LoopFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
      GetWorkDimensions(),
      GetIndexShape(analysis_.fusion_root(root_index).shape()), mlir_context,
      order_);
}

std::optional<std::vector<IndexingMap>>
LoopFusion::ComputeThreadIdToInputIndexing(int64_t root_index,
                                           MLIRContext* mlir_context) const {
  std::optional<IndexingMap> thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, mlir_context);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* fusion_root =
      &analysis_.fusion_root(root_index).instruction();
  auto output_to_input_indexing =
      ComputeOutputToInputIndexing(fusion_root, /*output_id=*/0, mlir_context);
  std::vector<IndexingMap> result;
  result.reserve(fusion_root->operand_count());
  for (int64_t operand_index = 0; operand_index < fusion_root->operand_count();
       ++operand_index) {
    auto output_to_input_indexing_maps =
        output_to_input_indexing.indexing_maps[operand_index];
    // Since we are computing the indexing for a non-fusion op, there is only
    // one indexing map per operand.
    CHECK_EQ(output_to_input_indexing_maps.size(), 1);
    IndexingMap thread_id_to_input_indexing_map =
        ComposeIndexingMaps(*thread_id_to_output_indexing,
                            output_to_input_indexing_maps.begin()->map());
    thread_id_to_input_indexing_map.Simplify();
    result.push_back(thread_id_to_input_indexing_map);
  }
  return result;
}

LaunchDimensions LoopFusion::launch_dimensions() const {
  Shape indexing_shape = emitters::LoopFusionKernelEmitter::GetIndexingShape(
      analysis_.fusion_spec());
  auto dims = CalculateLaunchDimensions(indexing_shape, analysis_.device_info(),
                                        unroll_factor_);
  const auto& blocks = dims.block_counts();
  auto split_x = MaybeSplitGridDimensionX(dims.thread_counts_per_block().x,
                                          blocks.x, analysis_.device_info());
  if (split_x[0] != blocks.x) {  // dim X has been split
    if (blocks.z != 1) {
      LOG(FATAL) << "Unable to split launch dimensions since block.z ("
                 << blocks.z << ") != 1";
    }
    // split blocks.x into x and y, move blocks.y -> blocks.z
    return LaunchDimensions(se::BlockDim(split_x[0], split_x[1], blocks.y),
                            dims.thread_counts_per_block());
  }
  return dims;
}

WorkDimensions LoopFusion::GetWorkDimensions() const {
  WorkDimensions work_dimensions = launch_dimensions().AsWorkDimensions();
  work_dimensions.work_tile_size.dimensions.push_back(unroll_factor_);
  return work_dimensions;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoopFusion::CreateMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  emitters::LoopFusionKernelEmitter emitter(
      mlir_context, fusion, analysis_.fusion_spec(), buffer_assignment,
      GetDefaultBufferAlignment(), GetWorkDimensions(), entry_function_name,
      BackendKind::kGpu, order_);

  ABSL_ASSIGN_OR_RETURN(auto kernel_definition, emitter.EmitKernelDefinition());
  return std::move(kernel_definition).TakeSource().TakeModule();
}

absl::Status LoopFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError("Not implemented");
}

}  // namespace gpu
}  // namespace xla
