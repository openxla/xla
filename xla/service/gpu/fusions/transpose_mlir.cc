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
#include "xla/service/gpu/fusions/transpose_mlir.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::MLIRContext;
using mlir::RankedTensorType;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::InsertOp;
using mlir_converter::ApplyIndexing;

constexpr int kBaseBlockSize = WarpSize();
constexpr int kMinThreadsPerBlock = 128;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kMaxVectorizedBytes = 4;

}  // namespace

MlirTransposeFusion::MlirTransposeFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      transpose_(analysis.tiled_transpose()),
      permutation_(transpose_.permutation),
      input_shape_(Permute(transpose_.dimensions, permutation_)) {
  ConstHloInstructionSet transposes_to_tile;
  int index = 0;
  int64_t shmem_usage = 0;
  int max_element_bytes = 0;
  for (auto [root, hero] :
       llvm::zip(analysis_.fusion_roots(), analysis_.fusion_heroes())) {
    if (auto transpose = GetDescriptionForTiledTransposeEmitter(
            root.instruction(), hero.instruction())) {
      transposes_to_tile.insert(&hero.instruction());
      shmem_transpose_roots_.push_back(&root.instruction());
      int size = primitive_util::ByteWidth(hero.shape().element_type());
      max_element_bytes = std::max(max_element_bytes, size);
      shmem_usage += kBaseBlockSize * (kBaseBlockSize + 1) * size;
      shmem_transpose_root_indices_.push_back(index);
    } else {
      side_output_roots_.push_back(&root.instruction());
      side_output_root_indices_.push_back(index);
    }
    ++index;
  }
  shmem_transposes_ = {transposes_to_tile.begin(), transposes_to_tile.end()};

  auto compute_block_sizes = [this](int vector_size) {
    vector_size_ = vector_size;
    block_size_ = kBaseBlockSize * vector_size_;
    block_sizes_ = {1, 1, block_size_};
    block_sizes_[permutation_[2]] = block_size_;
    block_counts_ = {CeilOfRatio(input_shape_[0], block_sizes_[0]),
                     CeilOfRatio(input_shape_[1], block_sizes_[1]),
                     CeilOfRatio(input_shape_[2], block_sizes_[2])};
  };
  // Compute initial block sizes without vectorization. We use the result to
  // determine whether we can vectorize.
  compute_block_sizes(1);

  // Enable vectorization if we have enough work, enough shared memory and
  // the input dimensions are divisible by the vector size. Vectorizing loads
  // for large data types does not help (there's already enough parallelism).
  const auto& device = analysis_.device_info();
  for (int vec_size = kMaxVectorizedBytes / max_element_bytes; vec_size > 1;
       vec_size /= 2) {
    int elems_per_thread = vec_size * vec_size;
    bool enough_work = Product(block_counts_) * kMinThreadsPerBlock >=
                       elems_per_thread * device.core_count() *
                           device.threads_per_core_limit();
    bool enough_shmem =
        shmem_usage * elems_per_thread <= device.shared_memory_per_block();
    bool aligned_dims = (input_shape_[2] % vec_size == 0) &&
                        (input_shape_[permutation_[2]] % vec_size == 0);
    if (enough_work && enough_shmem && aligned_dims) {
      compute_block_sizes(vec_size);
      break;
    }
  }

  shmem_usage_per_block_ = shmem_usage * vector_size_ * vector_size_;
  threads_per_block_ = ChooseSuitableThreadsNumber(device);
  CHECK(threads_per_block_ % kBaseBlockSize == 0);
  num_rows_ = threads_per_block_ / kBaseBlockSize;
}

std::optional<IndexingMap> MlirTransposeFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index);
  if (hero.opcode() != HloOpcode::kTranspose) {
    // The shape of non-transpose roots are bitcast compatible with the input
    // shape of transpose heroes.
    auto map = ComposeIndexingMaps(
        GetIndexing(/*input=*/true, hero.shape(), mlir_context),
        GetBitcastMap(hero.shape(), analysis_.fusion_root(root_index).shape(),
                      mlir_context));
    map.Simplify();
    return map;
  }
  return GetIndexing(/*input=*/false, hero.shape(), mlir_context);
}

std::optional<IndexingMap> MlirTransposeFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (hero.opcode() != HloOpcode::kTranspose) {
    auto map = ComposeIndexingMaps(
        *ComputeThreadIdToOutputIndexing(root_index, mlir_context),
        *ComputeOutputToInputIndexing(
             &analysis_.fusion_root(root_index).instruction(), 0, mlir_context)
             .indexing_maps[hero_operand_index]
             .begin());
    map.Simplify();
    return map;
  }
  return GetIndexing(/*input=*/true, hero.operand(hero_operand_index)->shape(),
                     mlir_context);
}

LaunchDimensions MlirTransposeFusion::launch_dimensions() const {
  return LaunchDimensions(Product(block_counts_), threads_per_block_);
}

IndexingMap MlirTransposeFusion::GetSharedMemoryIndexing(
    bool read, mlir::MLIRContext* ctx) const {
  auto thread_offsets =
      Permute(GetThreadOffsets(ctx), read ? Vector3{0, 1, 2} : permutation_);
  return {mlir::AffineMap::get(6, 2, thread_offsets, ctx),
          DimVarsFromTensorSizes({threads_per_block_, 1, 1, 1, 1, 1}),
          RangeVarsFromTensorSizes({block_size_ / num_rows_, vector_size_}),
          {}};
}

MlirTransposeFusion::WriteResult MlirTransposeFusion::EmitWriteToShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const mlir_converter::PartitionedComputation& root_computation,
    const mlir_converter::CallTargetProvider& call_target_provider,
    ValueRange output_args) const {
  MLIRContext* ctx = builder.getContext();
  auto shmem_tensor_size = block_sizes_;
  // Avoid bank conflicts.
  ++shmem_tensor_size.back();

  // Allocate shared memory.
  SmallVector<Value> inits;
  for (auto* transpose : shmem_transposes_) {
    auto elem_type = mlir_converter::PrimitiveTypeToMlirType(
        transpose->shape().element_type(), builder);
    inits.push_back(builder.create<AllocateSharedOp>(
        RankedTensorType::get(shmem_tensor_size, elem_type)));
  }

  // Add output arguments for side outputs.
  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  for (int index : side_output_root_indices_) {
    inits.push_back(entry_function.getArgument(num_inputs + index));
  }

  IndexingMap write_indexing = GetSharedMemoryIndexing(/*read=*/false, ctx);
  auto body_builder = [&](ValueRange output_tensors, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto input_indices = [&](const HloInstruction* instr) {
      return ApplyIndexing(GetIndexing(/*input=*/true, instr->shape(), ctx),
                           dim_values, symbol_values, builder);
    };
    SmallVector<Value> result_tensors;
    auto shmem_indices =
        ApplyIndexing(write_indexing, dim_values, symbol_values, builder);
    for (auto [transpose, output] :
         llvm::zip(shmem_transposes_, output_tensors)) {
      // Emit loop that writes subgraphs of transpose operands to shmem.
      auto result_scalar = mlir_converter::ProvideParameter(
          root_computation, transpose,
          /*operand_index=*/0, input_indices(transpose->operand(0)),
          call_target_provider, entry_function, builder)[0];
      result_tensors.push_back(
          builder.create<InsertOp>(result_scalar, output, shmem_indices));
    }

    // Produce all side outputs and then write them.
    SmallVector<Value> side_outputs;
    SmallVector<SmallVector<Value>> side_output_indices;
    auto* root_tuple = fusion.fused_expression_root();
    for (auto root : side_output_roots_) {
      side_output_indices.push_back(input_indices(root));
      ValueRange param_values = mlir_converter::ProvideParameter(
          root_computation, root_tuple, root_tuple->operand_index(root),
          side_output_indices.back(), call_target_provider, entry_function,
          builder);
      side_outputs.append(param_values.begin(), param_values.end());
    }

    for (const auto& [value, indices, output] :
         llvm::zip(side_outputs, side_output_indices,
                   output_tensors.take_back(side_output_roots_.size()))) {
      result_tensors.push_back(
          builder.create<InsertOp>(value, output, indices));
    }

    return result_tensors;
  };

  auto indexing = GetIndexing(
      /*input=*/true, shmem_transposes_.front()->operand(0)->shape(), ctx);
  auto written_vector =
      EmitThreadLoopNest(builder, inits, indexing, body_builder);
  ValueRange written = written_vector;
  auto shmem_tensors = written.take_front(shmem_transposes_.size());

  WriteResult result;
  result.shmem_tensors =
      builder
          .create<SyncThreadsOp>(mlir::TypeRange(shmem_tensors), shmem_tensors)
          .getResults();
  result.updated_outputs = output_args;
  for (auto [index, side_output_result] :
       llvm::zip(side_output_root_indices_,
                 written.take_back(side_output_roots_.size()))) {
    result.updated_outputs[index] = side_output_result;
  }
  return result;
}

void MlirTransposeFusion::EmitReadFromShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const mlir_converter::PartitionedComputations& computations,
    const WriteResult& written) const {
  auto* mlir_context = builder.getContext();
  auto output_indexing = *ComputeThreadIdToOutputIndexing(
      shmem_transpose_root_indices_[0], mlir_context);
  auto shmem_read_indexing =
      GetSharedMemoryIndexing(/*read=*/true, mlir_context);
  auto result_tensors = EmitThreadLoopNest(
      builder, written.updated_outputs, output_indexing,
      [&](ValueRange output_tensors, ValueRange dim_values,
          ValueRange symbol_values) -> SmallVector<Value> {
        auto shmem_indices = ApplyIndexing(shmem_read_indexing, dim_values,
                                           symbol_values, builder);
        absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>
            transpose_values;
        for (auto [transpose, shmem] :
             llvm::zip(shmem_transposes_, written.shmem_tensors)) {
          transpose_values[transpose].push_back(
              builder.create<ExtractOp>(shmem, shmem_indices));
        }
        llvm::SmallVector<Value> epilogue_indices = dim_values;
        absl::c_copy(symbol_values, std::back_inserter(epilogue_indices));
        auto result_scalars =
            EmitEpilogue(/*epilogue_index=*/0, computations, entry_function,
                         transpose_values, epilogue_indices, builder);
        SmallVector<Value> results = output_tensors;
        for (auto [root, indexing, root_index] :
             llvm::zip(shmem_transpose_roots_,
                       computations.epilogues().front().root_indexing,
                       shmem_transpose_root_indices_)) {
          llvm::SmallVector<Value> indices =
              ApplyIndexing(indexing, dim_values, symbol_values, builder);
          results[root_index] = builder.create<InsertOp>(
              result_scalars.at(root).front(), results[root_index], indices);
        }
        return results;
      });

  builder.create<ReturnOp>(result_tensors);
}

std::vector<mlir_converter::EpilogueSpecification>
MlirTransposeFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                  MLIRContext* mlir_context) const {
  std::vector<mlir_converter::EpilogueSpecification> epilogues{
      mlir_converter::EpilogueSpecification::FromOutputIndexing(
          analysis_, shmem_transposes_, shmem_transpose_roots_, *this,
          mlir_context)};
  // Add empty epilogues for the side outputs. This ensures their roots don't
  // get "fused" into the tuple function.
  for (const auto* root : side_output_roots_) {
    epilogues.push_back(
        mlir_converter::EpilogueSpecification::FromIdentityIndexing(
            root, root, mlir_context));
  }
  return epilogues;
}

absl::Status MlirTransposeFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  // Write intermediate results to shmem.
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  auto written = EmitWriteToShMemMlir(
      builder, entry_function, fusion, root_computation, call_targets,
      entry_function.getArguments().take_back(analysis_.fusion_roots().size()));
  // Read intermediate results from shmem and compute epilogues.
  EmitReadFromShMemMlir(builder, entry_function, fusion, computations, written);
  return absl::OkStatus();
}

llvm::SmallVector<mlir::AffineExpr, 4> MlirTransposeFusion::GetThreadOffsets(
    mlir::MLIRContext* ctx) const {
  auto thread = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], ctx);
  auto loop = mlir::getAffineSymbolExpr(0, ctx);
  auto vector = mlir::getAffineSymbolExpr(1, ctx);
  int loop_stride = block_size_ * num_rows_;
  auto linear_index = loop * loop_stride + thread * vector_size_ + vector;
  return DelinearizeInBoundsIndex(linear_index, block_sizes_);
}

IndexingMap MlirTransposeFusion::GetIndexing(bool input,
                                             const xla::Shape& shape,
                                             mlir::MLIRContext* ctx) const {
  auto raw_id = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapBlockIdxDims[0], ctx);
  auto block_ids = Permute(DelinearizeInBoundsIndex(raw_id, block_counts_),
                           input ? Vector3{0, 1, 2} : permutation_);
  auto thread_offsets = GetThreadOffsets(ctx);
  llvm::SmallVector<AffineExpr, 3> offsets;
  for (auto [block_id, block_size, thread] :
       llvm::zip(block_ids, block_sizes_, thread_offsets)) {
    offsets.push_back(block_id * block_size + thread);
  }
  IndexingMap result{
      mlir::AffineMap::get(6, 2, offsets, ctx),
      DimVarsFromTensorSizes(
          {threads_per_block_, 1, 1, Product(block_counts_), 1, 1}),
      RangeVarsFromTensorSizes({block_size_ / num_rows_, vector_size_}),
      {}};
  auto normalized_shape =
      input ? ShapeUtil::MakeShape(shape.element_type(), input_shape_)
            : ShapeUtil::MakeShape(shape.element_type(), transpose_.dimensions);
  for (auto [size, dim] : llvm::zip(normalized_shape.dimensions(),
                                    result.GetAffineMap().getResults())) {
    result.AddConstraint(dim, {0, size - 1});
  }
  result =
      ComposeIndexingMaps(result, GetBitcastMap(normalized_shape, shape, ctx));
  result.Simplify();
  return result;
}

namespace {

struct HloBytesUsageDesc {
  int tile_size;
  int elem_bytes;
  int ref_count;
  int ref_id;
  int unused_bytes;
};

std::optional<HloBytesUsageDesc> GetHloBytesUsageDesc(
    HloInstructionAdaptor& instr_adaptor,
    absl::InlinedVector<int, 2>& live_range,
    absl::InlinedVector<HloBytesUsageDesc, 2>* instr_bytes_desc,
    absl::flat_hash_map<const HloInstruction*, int>& instr_to_ids) {
  auto* instr = &instr_adaptor.instruction();
  // These Hlo which only involve index computation don't need to
  // allocate new registers.
  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kBitcast) {
    auto* input = &(instr_adaptor.GetOperand(0).instruction());
    int input_id = instr_to_ids[input];
    auto& input_desc = (*instr_bytes_desc)[input_id];
    if (instr_to_ids[instr] == live_range[input_id]) {
      (*instr_bytes_desc)[input_desc.ref_id].ref_count--;
    }
    (*instr_bytes_desc)[input_desc.ref_id].ref_count++;
    return HloBytesUsageDesc{0, input_desc.elem_bytes, 0, input_desc.ref_id, 0};
  }
  if (instr->IsElementwise()) {
    int output_bytes = primitive_util::ByteWidth(instr->shape().element_type());
    int max_tile_size = 0, unused_bytes = 0, reuse_desc_id = 0;
    bool can_reuse = false;
    for (auto operand_adaptor : instr_adaptor.GetOperands()) {
      auto* operand = &operand_adaptor.instruction();
      int operand_id = instr_to_ids[operand];
      auto& operand_desc = (*instr_bytes_desc)[operand_id];
      auto& ref_desc = (*instr_bytes_desc)[operand_desc.ref_id];
      if (instr_to_ids[instr] == live_range[operand_id]) {
        --ref_desc.ref_count;
        if (ref_desc.ref_count == 0) {
          unused_bytes += ref_desc.tile_size * ref_desc.elem_bytes;
        }
      }
      if (ref_desc.tile_size >= max_tile_size) {
        // Dest can reuse src registers iff src has no next user and
        // owned bytes is larger than needed bytes.
        bool reuse_operand =
            ref_desc.ref_count == 0 && output_bytes <= ref_desc.elem_bytes;
        can_reuse = ref_desc.tile_size > max_tile_size
                        ? reuse_operand
                        : can_reuse || reuse_operand;
        if (reuse_operand) {
          reuse_desc_id = operand_desc.ref_id;
        }
        max_tile_size = ref_desc.tile_size;
      }
    }
    if (can_reuse) {
      auto& reuse_desc = (*instr_bytes_desc)[reuse_desc_id];
      ++reuse_desc.ref_count;
      unused_bytes -= reuse_desc.tile_size * reuse_desc.elem_bytes;
      return HloBytesUsageDesc{0, reuse_desc.elem_bytes, 0, reuse_desc_id,
                               unused_bytes};
    }
    return HloBytesUsageDesc{max_tile_size, output_bytes, 1,
                             instr_to_ids[instr], unused_bytes};
  }
  // We can add more supported hlo if needed.
  return std::nullopt;
}

std::optional<int> ComputeMaxBytesCountFromInstrs(
    absl::Span<const HloInstructionAdaptor> post_order,
    absl::flat_hash_map<const HloInstruction*, int>& instr_to_ids,
    absl::InlinedVector<int, 2>& live_range,
    absl::InlinedVector<HloBytesUsageDesc, 2>* instr_bytes_desc,
    absl::InlinedVector<bool, 2>* visited,
    const std::function<bool(const HloInstruction*)>& should_visit,
    int instrs_bytes) {
  int max_bytes_count = instrs_bytes;
  for (auto instr_adaptor : post_order) {
    auto* instr = &instr_adaptor.instruction();
    int instr_id = instr_to_ids[instr];
    if (!should_visit(instr) || (*visited)[instr_id]) {
      continue;
    }
    bool operands_visited = true;
    for (auto operand_adaptor : instr_adaptor.GetOperands()) {
      auto* operand = &operand_adaptor.instruction();
      int operand_id = instr_to_ids[operand];
      operands_visited = operands_visited && (*visited)[operand_id];
    }
    if (operands_visited) {
      int instr_id = instr_to_ids[instr];
      auto bytes_desc = GetHloBytesUsageDesc(instr_adaptor, live_range,
                                             instr_bytes_desc, instr_to_ids);
      if (!bytes_desc.has_value()) {
        return std::nullopt;
      }
      (*instr_bytes_desc)[instr_id] = bytes_desc.value();
      (*visited)[instr_id] = true;
      instrs_bytes +=
          bytes_desc.value().tile_size * bytes_desc.value().elem_bytes;
      max_bytes_count = std::max(max_bytes_count, instrs_bytes);
      instrs_bytes -= bytes_desc.value().unused_bytes;
      // This means current hlo is the root of fusion. We release the occupied
      // registers immediately.
      if (live_range[instr_id] == instr_id) {
        auto& ref_desc = (*instr_bytes_desc)[bytes_desc.value().ref_id];
        --ref_desc.ref_count;
        if (ref_desc.ref_count == 0) {
          instrs_bytes -= ref_desc.tile_size * ref_desc.elem_bytes;
        }
      }
    }
  }
  return max_bytes_count;
}

std::optional<int> ComputeRegNumFromParamToTarget(
    absl::InlinedVector<HloInstructionAdaptor, 2>& post_order,
    absl::flat_hash_map<const HloInstruction*, int>& instr_to_ids,
    absl::InlinedVector<int, 2>& live_range,
    const HloFusionAdaptor& fusion_adaptor, const HloInstruction* target_instr,
    IndexingMap& initial_map, MLIRContext* mlir_context) {
  absl::flat_hash_map<const HloInstruction*, IndexingMap> instr_indexing;
  instr_indexing.emplace(target_instr, initial_map);
  for (int id = instr_to_ids[target_instr]; id >= 0; --id) {
    auto& instr_adaptor = post_order[id];
    auto it = instr_indexing.find(&instr_adaptor.instruction());
    if (it == instr_indexing.end()) {
      continue;
    }
    if (instr_adaptor.shape().IsTuple()) {
      return std::nullopt;
    }
    auto consumer_indexing = it->second;
    auto operands_indexing = ComputeOutputToInputIndexing(
        &instr_adaptor.instruction(), /*output_id=*/0, mlir_context);
    for (const auto& [producer_operand_id, producer_operand_indexing] :
         llvm::enumerate(operands_indexing.indexing_maps)) {
      if (producer_operand_indexing.size() != 1 ||
          producer_operand_indexing.begin()->IsUndefined()) {
        return std::nullopt;
      }
      auto* producer_instr =
          &(instr_adaptor.GetOperand(producer_operand_id).instruction());
      auto composed_map = ComposeIndexingMaps(
          consumer_indexing, *producer_operand_indexing.begin());
      composed_map.Simplify();
      auto producer_indexing = instr_indexing.find(producer_instr);
      if (producer_indexing != instr_indexing.end()) {
        if (producer_indexing->second != composed_map) {
          return std::nullopt;
        }
      } else {
        instr_indexing.emplace(producer_instr, std::move(composed_map));
      }
    }
  }

  int bytes_count = 0;
  absl::InlinedVector<HloBytesUsageDesc, 2> instr_bytes_desc(post_order.size());
  absl::InlinedVector<bool, 2> visited(post_order.size(), false);
  for (auto* param : fusion_adaptor.GetParameters()) {
    auto it = instr_indexing.find(param);
    if (it != instr_indexing.end()) {
      int param_id = instr_to_ids[param];
      visited[param_id] = true;
      auto& param_map = it->second;
      // Evaluate the block tiling sizes of the fusion parameters.
      auto range_evaluator = param_map.GetRangeEvaluator();
      int tile_size = 1;
      int elem_size = primitive_util::ByteWidth(param->shape().element_type());
      for (auto expr : param_map.GetAffineMap().getResults()) {
        Interval interval = range_evaluator.ComputeExpressionRange(expr);
        tile_size *= (interval.upper - interval.lower + 1);
      }
      instr_bytes_desc[param_id] = {tile_size, elem_size, 1, param_id};
      bytes_count += tile_size * elem_size;
    }
  }
  auto max_bytes = ComputeMaxBytesCountFromInstrs(
      absl::MakeSpan(post_order).subspan(0, instr_to_ids[target_instr] + 1),
      instr_to_ids, live_range, &instr_bytes_desc, &visited,
      [&](const HloInstruction* hlo) {
        return instr_indexing.find(hlo) != instr_indexing.end();
      },
      bytes_count);
  if (!max_bytes.has_value()) {
    return std::nullopt;
  }
  return max_bytes.value() / 4;
}

absl::InlinedVector<int, 2> AnalyzeRegLiveRange(
    absl::InlinedVector<HloInstructionAdaptor, 2>& post_order,
    absl::flat_hash_map<const HloInstruction*, int>& instr_to_ids,
    const HloFusionAdaptor& fusion_adaptor) {
  absl::InlinedVector<int, 2> live_range(post_order.size(), -1);
  for (int id = post_order.size() - 1; id >= 0; --id) {
    auto* instr = &post_order[id].instruction();
    if (!fusion_adaptor.ContainsInstruction(instr)) {
      continue;
    }
    for (auto operand_adaptor : post_order[id].GetOperands()) {
      int operand_id = instr_to_ids[&operand_adaptor.instruction()];
      live_range[operand_id] = std::max(live_range[operand_id], id);
    }
    live_range[id] = std::max(live_range[id], id);
  }
  return live_range;
}

}  // namespace

std::optional<int> MlirTransposeFusion::ComputeRegNumBeforeShmemWrite(
    absl::InlinedVector<HloInstructionAdaptor, 2>& post_order,
    absl::flat_hash_map<const HloInstruction*, int>& instr_to_ids,
    absl::InlinedVector<int, 2>& live_range,
    const HloFusionAdaptor& fusion_adaptor, MLIRContext* mlir_context) {
  llvm::SmallVector<AffineExpr, 3> offsets;
  llvm::SmallVector<AffineExpr, 3> dim_exprs(3);
  llvm::SmallVector<AffineExpr, 3> sym_exprs(3);
  mlir::bindDimsList(mlir_context, llvm::MutableArrayRef(dim_exprs));
  mlir::bindSymbolsList(mlir_context, llvm::MutableArrayRef(sym_exprs));
  for (auto [dim_expr, sym_expr] : llvm::zip(dim_exprs, sym_exprs)) {
    offsets.push_back(dim_expr * block_size_ + sym_expr);
  }
  absl::InlinedVector<int64_t, 3> sym_ranges(3, 1);
  sym_ranges[2] = std::min(static_cast<int64_t>(block_size_), input_shape_[2]);
  sym_ranges[permutation_[2]] = std::min(static_cast<int64_t>(block_size_),
                                         input_shape_[permutation_[2]]);
  IndexingMap block_indexing{mlir::AffineMap::get(3, 3, offsets, mlir_context),
                             DimVarsFromTensorSizes({1, 1, 1}),
                             RangeVarsFromTensorSizes(sym_ranges),
                             /*rt_vars=*/{}};
  auto normalized_shape = ShapeUtil::MakeShape(F32, input_shape_);
  // Compute registers count by iterating transpose heroes and side outputs. The
  // reason is that emitter generate their IR one by one in the first stage.
  int max_registers_count = 0;
  for (auto* transpose : shmem_transposes_) {
    auto operand_adaptor =
        HloInstructionAdaptor(*transpose, &fusion_adaptor).GetOperand(0);
    auto initial_map = ComposeIndexingMaps(
        block_indexing,
        GetBitcastMap(normalized_shape, operand_adaptor.shape(), mlir_context));
    std::optional<int> registers_count = ComputeRegNumFromParamToTarget(
        post_order, instr_to_ids, live_range, fusion_adaptor,
        &operand_adaptor.instruction(), initial_map, mlir_context);
    if (!registers_count.has_value()) {
      return std::nullopt;
    }
    max_registers_count =
        std::max(max_registers_count, registers_count.value());
  }
  for (auto* side_output : side_output_roots_) {
    auto initial_map = ComposeIndexingMaps(
        block_indexing,
        GetBitcastMap(normalized_shape, side_output->shape(), mlir_context));
    std::optional<int> registers_count = ComputeRegNumFromParamToTarget(
        post_order, instr_to_ids, live_range, fusion_adaptor, side_output,
        initial_map, mlir_context);
    if (!registers_count.has_value()) {
      return std::nullopt;
    }
    max_registers_count =
        std::max(max_registers_count, registers_count.value());
  }
  return max_registers_count;
}

std::optional<int> MlirTransposeFusion::ComputeRegNumAfterShmemRead(
    absl::InlinedVector<HloInstructionAdaptor, 2>& post_order,
    absl::flat_hash_map<const HloInstruction*, int>& instr_to_ids,
    absl::InlinedVector<int, 2>& live_range,
    const HloFusionAdaptor& fusion_adaptor) {
  // Due to the second stage of emitter read all of the elements in shmem, we
  // aggregate the bytes usage of transpose heroes firstly.
  int instrs_size = post_order.size();
  int bytes = 0, begin_id = instrs_size - 1;
  absl::InlinedVector<HloBytesUsageDesc, 2> instr_bytes_desc(post_order.size());
  absl::InlinedVector<bool, 2> visited(post_order.size(), false);
  int tile_size = 1;
  tile_size *= input_shape_[2] < block_size_ ? input_shape_[2] : block_size_;
  tile_size *= input_shape_[permutation_[2]] < block_size_
                   ? input_shape_[permutation_[2]]
                   : block_size_;
  for (auto* transpose : shmem_transposes_) {
    int id = instr_to_ids[transpose];
    begin_id = std::min(begin_id, id);
    int elem_size =
        primitive_util::ByteWidth(transpose->shape().element_type());
    instr_bytes_desc[id] = HloBytesUsageDesc{tile_size, elem_size, 1, id, 0};
    bytes += tile_size * elem_size;
    visited[id] = true;
  }
  std::optional<int> max_bytes = ComputeMaxBytesCountFromInstrs(
      absl::MakeSpan(post_order).subspan(begin_id, instrs_size - begin_id),
      instr_to_ids, live_range, &instr_bytes_desc, &visited,
      [&](const HloInstruction* hlo) {
        return fusion_adaptor.ContainsInstruction(hlo);
      },
      bytes);
  if (!max_bytes.has_value()) {
    return std::nullopt;
  }
  return max_bytes.value() / 4;
}

int MlirTransposeFusion::ChooseSuitableThreadsNumber(
    const stream_executor::DeviceDescription& device_info) {
  auto& fusion_adaptor = analysis_.fusion();
  // Visit hlo with post order to ensure the bytes usage of hlo's operands has
  // been computed yet.
  auto post_order = fusion_adaptor.MakeInstructionPostOrder();
  absl::InlinedVector<HloInstructionAdaptor, 2> post_order_with_params;
  for (auto* param : fusion_adaptor.GetParameters()) {
    post_order_with_params.emplace_back(*param, &fusion_adaptor);
  }
  absl::c_copy(post_order, std::back_inserter(post_order_with_params));
  absl::flat_hash_map<const HloInstruction*, int> instr_to_ids;
  for (int id = 0; id < post_order_with_params.size(); ++id) {
    instr_to_ids[&post_order_with_params[id].instruction()] = id;
  }
  auto live_range =
      AnalyzeRegLiveRange(post_order_with_params, instr_to_ids, fusion_adaptor);
  MLIRContext mlir_context;
  // loop can always be fully unrolled in transpose due to its trip count is
  // small. We compute the register usage based on this assumption. And we
  // should also consider there has two emitter stages in transpose: (1) each of
  // the heroes write to shmem. (2) Read from shmem to compute epilogue outputs.
  int64_t registers_per_block = 0;
  std::optional<int> regs_before_write =
      ComputeRegNumBeforeShmemWrite(post_order_with_params, instr_to_ids,
                                    live_range, fusion_adaptor, &mlir_context);
  std::optional<int> regs_after_read = ComputeRegNumAfterShmemRead(
      post_order_with_params, instr_to_ids, live_range, fusion_adaptor);
  if (regs_before_write.has_value() && regs_after_read.has_value()) {
    registers_per_block =
        std::max(regs_before_write.value(), regs_after_read.value());
  }
  int64_t max_registers_per_core = device_info.registers_per_core_limit();
  int64_t max_threads_per_core = device_info.threads_per_core_limit();
  int64_t shared_memory_per_core = device_info.shared_memory_per_core();
  // Blocks which reside in same core is limited by shared memory size. Maybe we
  // need to also check if exceed max resident blocks per core.
  int64_t max_blocks_per_core = shared_memory_per_core / shmem_usage_per_block_;
  int best_threads_per_block =
      std::min(static_cast<int64_t>(kMaxThreadsPerBlock),
               device_info.threads_per_block_limit());
  int max_resident_threads = 0;
  for (int64_t threads = best_threads_per_block; threads >= kMinThreadsPerBlock;
       threads /= 2) {
    int64_t blocks_per_core =
        std::min(max_blocks_per_core, max_threads_per_core / threads);
    // The number of registers calculated does not take into account temporarily
    // results and tensor indices, so we should leave enough registers for them.
    if (blocks_per_core * registers_per_block <=
        max_registers_per_core * 0.6f) {
      int resident_threads = blocks_per_core * threads;
      if (resident_threads >= max_resident_threads) {
        max_resident_threads = resident_threads;
        best_threads_per_block = threads;
      }
    }
  }
  return best_threads_per_block;
}

}  // namespace gpu
}  // namespace xla
