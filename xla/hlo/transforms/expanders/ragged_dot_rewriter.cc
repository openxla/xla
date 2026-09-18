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

#include "xla/hlo/transforms/expanders/ragged_dot_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace se = ::stream_executor;

namespace {

std::unique_ptr<HloComputation> CreateScalarAddComputation(PrimitiveType type) {
  auto embedded_builder = HloComputation::Builder("add");
  auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(type, {}), "lhs"));
  auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(type, {}), "rhs"));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
  return embedded_builder.Build();
}

std::unique_ptr<HloInstruction> Zero(PrimitiveType type) {
  return HloInstruction::CreateConstant(LiteralUtil::Zero(type));
}

// Takes an array of shape [batch_dims..., num_groups] and returns an array of
// the same shape with the elements of the array along the last dimension
// now representing the cumulative sum of all elements in the input array up to
// the current group.
std::unique_ptr<HloInstruction> CreateCumulativeSum(
    HloInstruction* group_sizes) {
  int64_t batch_dims = group_sizes->shape().dimensions().size() - 1;
  int64_t num_groups = group_sizes->shape().dimensions(batch_dims);

  Window cumsum_window;
  // Add batch dimensions.
  for (int i = 0; i < batch_dims; ++i) {
    WindowDimension* dim = cumsum_window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  // Add group dimension.
  WindowDimension* dim = cumsum_window.add_dimensions();
  dim->set_size(num_groups);
  dim->set_padding_low(num_groups - 1);
  dim->set_padding_high(0);
  dim->set_stride(1);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);

  auto type = group_sizes->shape().element_type();
  HloComputation* add = group_sizes->GetModule()->AddEmbeddedComputation(
      CreateScalarAddComputation(type));
  auto zero = group_sizes->parent()->AddInstruction(Zero(type));
  return HloInstruction::CreateReduceWindow(group_sizes->shape(), group_sizes,
                                            zero, cumsum_window, add);
}

// Expands ragged_op by one dimension with each row in the new dimension
// representing each group. It then zeros out the elements that don't belong to
// that group.
HloInstruction* RaggedToDense(HloInstruction* ragged_operand,
                              HloInstruction* group_sizes, int new_dim_index,
                              int ragged_dim) {
  auto computation = ragged_operand->parent();
  HloInstruction* cumulative_sum =
      computation->AddInstruction(CreateCumulativeSum(group_sizes));
  // Group dimension is always the last dimension.
  int group_dim = group_sizes->shape().dimensions().size() - 1;
  int num_groups = group_sizes->shape().dimensions(group_dim);

  // Create the mask to zero out the top half. First slice off the last element
  // of the cumulative sum array and append 0 to the beginning.
  auto slice_shape = cumulative_sum->shape();
  slice_shape.set_dimensions(group_dim, num_groups - 1);
  llvm::SmallVector<int64_t> slice_starts(slice_shape.dimensions().size(), 0);
  llvm::SmallVector<int64_t> slice_limits(slice_shape.dimensions().size());
  for (int i = 0; i < slice_shape.dimensions().size(); ++i) {
    slice_limits[i] = slice_shape.dimensions(i);
  }
  llvm::SmallVector<int64_t> slice_strides(slice_shape.dimensions().size(), 1);
  auto slice = computation->AddInstruction(HloInstruction::CreateSlice(
      slice_shape, cumulative_sum, slice_starts, slice_limits, slice_strides));

  // Concat a zero to the beginning of the cumulative sum array (represents how
  // many elements to zero out on top for the first group (zero)).
  auto zero_slice_shape = slice_shape;
  zero_slice_shape.set_dimensions(group_dim, 1);
  auto group_type = group_sizes->shape().element_type();
  auto zero_group_type = computation->AddInstruction(Zero(group_type));
  auto zero_arr = computation->AddInstruction(
      HloInstruction::CreateBroadcast(zero_slice_shape, zero_group_type, {}));
  auto concat = computation->AddInstruction(HloInstruction::CreateConcatenate(
      group_sizes->shape(), {zero_arr, slice}, group_dim));

  // Broadcast cumulative sum array to operand shape + group dimension.
  auto old_shape = ragged_operand->shape();
  llvm::SmallVector<int64_t> new_shape_dims;
  new_shape_dims.reserve(old_shape.dimensions().size() + 1);
  new_shape_dims.append(old_shape.dimensions().begin(),
                        old_shape.dimensions().end());
  new_shape_dims.insert(new_shape_dims.begin() + new_dim_index, num_groups);
  auto new_shape_gs = ShapeUtil::MakeShape(group_type, new_shape_dims);
  llvm::SmallVector<int64_t> broadcast_dims(
      concat->shape().dimensions().size());
  for (int i = 0; i < concat->shape().dimensions().size(); ++i) {
    broadcast_dims[i] = i;
  }
  auto broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape_gs, concat, broadcast_dims));

  auto iota = computation->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(group_type, {old_shape.dimensions(ragged_dim)}), 0));
  auto broadcast_iota = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape_gs, iota, {ragged_dim + 1}));

  // Zero out the top if row < cum_group_size.
  auto new_shape_pred = ShapeUtil::MakeShape(PRED, new_shape_gs.dimensions());
  auto compare_top = computation->AddInstruction(HloInstruction::CreateCompare(
      new_shape_pred, broadcast, broadcast_iota, ComparisonDirection::kLe));

  // Put zeros on the bottom if row >= cum_group_size.
  auto broadcast_groups =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          new_shape_gs, cumulative_sum, broadcast_dims));
  auto compare_bottom =
      computation->AddInstruction(HloInstruction::CreateCompare(
          new_shape_pred, broadcast_iota, broadcast_groups,
          ComparisonDirection::kLt));
  // Combine top+bottom half masks.
  auto mask = computation->AddInstruction(HloInstruction::CreateBinary(
      new_shape_pred, HloOpcode::kAnd, compare_top, compare_bottom));

  // Load LHS & broadcast it for each group.
  llvm::SmallVector<int64_t> old_dims_new_shape(old_shape.dimensions().size());
  for (int i = 0; i < new_dim_index; ++i) {
    old_dims_new_shape[i] = i;
  }
  for (int i = new_dim_index; i < old_shape.dimensions().size(); ++i) {
    old_dims_new_shape[i] = i + 1;
  }
  auto element_type = old_shape.element_type();
  auto new_shape = ShapeUtil::MakeShape(element_type, new_shape_dims);
  auto broadcast_ragged =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          new_shape, ragged_operand, old_dims_new_shape));

  auto zero_operand_type = computation->AddInstruction(Zero(element_type));
  auto zero_new_shape = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape, zero_operand_type, {}));
  // Apply mask to operand.
  return computation->AddInstruction(HloInstruction::CreateTernary(
      new_shape, HloOpcode::kSelect, mask, broadcast_ragged, zero_new_shape));
}

HloInstruction* TransposeIndexToFront(HloInstruction* instr, int index) {
  if (index == 0) {
    return instr;
  }
  auto new_shape = instr->shape();
  int64_t dim0 = new_shape.dimensions(0);
  int64_t dimIndex = new_shape.dimensions(index);
  new_shape.set_dimensions(0, dimIndex);
  new_shape.set_dimensions(index, dim0);
  llvm::SmallVector<int64_t> transpose_dims;
  transpose_dims.reserve(new_shape.dimensions().size());
  transpose_dims.push_back(index);
  for (int i = 1; i < new_shape.dimensions().size(); ++i) {
    if (i == index) {
      transpose_dims.push_back(0);
    } else {
      transpose_dims.push_back(i);
    }
  }
  auto computation = instr->parent();
  return computation->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, instr, transpose_dims));
}

enum class RaggedDotMode {
  kRaggedNonContracting,
  kRaggedContracting,
  kRaggedBatch,
};

RaggedDotMode GetRaggedDotMode(int lhs_ragged_dim,
                               const DotDimensionNumbers& dnums) {
  if (std::find(dnums.lhs_contracting_dimensions().begin(),
                dnums.lhs_contracting_dimensions().end(),
                lhs_ragged_dim) != dnums.lhs_contracting_dimensions().end()) {
    return RaggedDotMode::kRaggedContracting;
  }
  if (std::find(dnums.lhs_batch_dimensions().begin(),
                dnums.lhs_batch_dimensions().end(),
                lhs_ragged_dim) != dnums.lhs_batch_dimensions().end()) {
    return RaggedDotMode::kRaggedBatch;
  }
  return RaggedDotMode::kRaggedNonContracting;
}

int FindRhsRaggedDim(const DotDimensionNumbers& dot_dims, int lhs_ragged_dim) {
  const auto& lhs_contracting_dims = dot_dims.lhs_contracting_dimensions();
  int ragged_contracting_index =
      std::distance(std::find(lhs_contracting_dims.begin(),
                              lhs_contracting_dims.end(), lhs_ragged_dim),
                    lhs_contracting_dims.begin());
  return dot_dims.rhs_contracting_dimensions(ragged_contracting_index);
}

DotDimensionNumbers CreateRaggedNonContractingDotDims(
    const DotDimensionNumbers& old_dims, int new_dim_index, int rhs_group_dim) {
  DotDimensionNumbers new_dims;
  new_dims.add_lhs_contracting_dimensions(new_dim_index);
  for (auto dim : old_dims.lhs_contracting_dimensions()) {
    new_dims.add_lhs_contracting_dimensions(dim + 1);
  }
  for (auto dim : old_dims.lhs_batch_dimensions()) {
    new_dims.add_lhs_batch_dimensions(dim);
  }
  new_dims.add_rhs_contracting_dimensions(rhs_group_dim);
  for (auto dim : old_dims.rhs_contracting_dimensions()) {
    new_dims.add_rhs_contracting_dimensions(dim);
  }
  for (auto dim : old_dims.rhs_batch_dimensions()) {
    new_dims.add_rhs_batch_dimensions(dim);
  }
  return new_dims;
}

DotDimensionNumbers CreateRaggedContractingDotDims(
    const DotDimensionNumbers& old_dims) {
  DotDimensionNumbers new_dims;
  // Add group dimension to the beginning of the batch dimensions.
  new_dims.add_lhs_batch_dimensions(0);
  for (auto dim : old_dims.lhs_batch_dimensions()) {
    new_dims.add_lhs_batch_dimensions(dim + 1);
  }
  new_dims.add_rhs_batch_dimensions(0);
  for (auto dim : old_dims.rhs_batch_dimensions()) {
    new_dims.add_rhs_batch_dimensions(dim + 1);
  }
  for (auto dim : old_dims.rhs_contracting_dimensions()) {
    new_dims.add_rhs_contracting_dimensions(dim + 1);
  }
  for (auto dim : old_dims.lhs_contracting_dimensions()) {
    new_dims.add_lhs_contracting_dimensions(dim + 1);
  }
  return new_dims;
}

absl::StatusOr<std::unique_ptr<HloInstruction>> RaggedToGeneral(
    HloRaggedDotInstruction* ragged_dot) {
  const auto& ragged_dims = ragged_dot->ragged_dot_dimension_numbers();
  const auto& dot_dims = ragged_dims.dot_dimension_numbers();
  if (ragged_dims.lhs_ragged_dimensions().size() != 1) {
    return absl::UnimplementedError("lhs_ragged_dimensions must have size 1");
  }
  int lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);

  auto lhs = ragged_dot->mutable_operand(0);
  auto rhs = ragged_dot->mutable_operand(1);
  auto group_sizes = ragged_dot->mutable_operand(2);
  int new_dim_index = group_sizes->shape().dimensions().size() - 1;
  DotDimensionNumbers new_dot_dims;

  RaggedDotMode mode =
      GetRaggedDotMode(lhs_ragged_dim, ragged_dims.dot_dimension_numbers());
  switch (mode) {
    case RaggedDotMode::kRaggedNonContracting: {
      if (ragged_dims.rhs_group_dimensions().size() != 1) {
        return absl::UnimplementedError(
            "rhs_group_dimensions must have size equal to 1 when lhs ragged "
            "dimension is a non-contracting dimension");
      }
      int rhs_group_dim = ragged_dims.rhs_group_dimensions(0);
      lhs = RaggedToDense(lhs, group_sizes, new_dim_index, lhs_ragged_dim);
      new_dot_dims = CreateRaggedNonContractingDotDims(dot_dims, new_dim_index,
                                                       rhs_group_dim);
      break;
    }
    case RaggedDotMode::kRaggedContracting: {
      lhs = RaggedToDense(lhs, group_sizes, new_dim_index, lhs_ragged_dim);
      lhs = TransposeIndexToFront(lhs, new_dim_index);
      int rhs_ragged_dim = FindRhsRaggedDim(dot_dims, lhs_ragged_dim);
      rhs = RaggedToDense(rhs, group_sizes, new_dim_index, rhs_ragged_dim);
      rhs = TransposeIndexToFront(rhs, new_dim_index);
      new_dot_dims = CreateRaggedContractingDotDims(dot_dims);
      break;
    }
    case RaggedDotMode::kRaggedBatch: {
      new_dot_dims = dot_dims;
      break;
    }
  }

  return HloInstruction::CreateDot(ragged_dot->shape(), lhs, rhs, new_dot_dims,
                                   ragged_dot->precision_config());
}

bool IsFP16Operation(const HloInstruction* ragged_dot) {
  return (ragged_dot->shape().element_type() == F16) &&
         (ragged_dot->operand(0)->shape().element_type() == F16) &&
         (ragged_dot->operand(1)->shape().element_type() == F16);
}

bool IsBF16Operation(const HloInstruction* ragged_dot) {
  return (ragged_dot->shape().element_type() == BF16) &&
         (ragged_dot->operand(0)->shape().element_type() == BF16) &&
         (ragged_dot->operand(1)->shape().element_type() == BF16);
}

bool CanBeHandledByCuDNNFusion(
    const HloInstruction* instruction,
    stream_executor::dnn::VersionInfo cudnn_version) {
  const HloRaggedDotInstruction* ragged_dot =
      DynCast<HloRaggedDotInstruction>(instruction);
  const auto& ragged_dims = ragged_dot->ragged_dot_dimension_numbers();
  if (ragged_dims.lhs_ragged_dimensions().size() != 1 ||
      (ragged_dot->shape().element_type() != F16 &&
       ragged_dot->shape().element_type() != BF16)) {
    return false;
  }
  int lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  RaggedDotMode mode =
      GetRaggedDotMode(lhs_ragged_dim, ragged_dims.dot_dimension_numbers());
  if (mode == RaggedDotMode::kRaggedContracting) {
    // Wgrad: needs cuDNN's moe_grouped_matmul_bwd, gated separately since it
    // requires a newer cuDNN than the forward ragged-dot fusion.
    return cudnn_version >= kMinCudnnVersionForRaggedDotWgradFusion;
  }
  return mode == RaggedDotMode::kRaggedNonContracting;
}

// Pads the given dimension of `operand` up to `new_size` with zeros. Returns
// `operand` unchanged if it is already `new_size`.
HloInstruction* PadDimTo(HloInstruction* operand, int dim, int64_t new_size) {
  int64_t old_size = operand->shape().dimensions(dim);
  if (old_size == new_size) {
    return operand;
  }
  auto computation = operand->parent();
  Shape new_shape = operand->shape();
  new_shape.set_dimensions(dim, new_size);
  PaddingConfig padding_config;
  for (int i = 0; i < operand->shape().dimensions().size(); ++i) {
    auto* padding_dim = padding_config.add_dimensions();
    padding_dim->set_edge_padding_low(0);
    padding_dim->set_edge_padding_high(i == dim ? new_size - old_size : 0);
    padding_dim->set_interior_padding(0);
  }
  auto* zero =
      computation->AddInstruction(Zero(operand->shape().element_type()));
  return computation->AddInstruction(
      HloInstruction::CreatePad(new_shape, operand, zero, padding_config));
}

// Slices the given dimension of `operand` down to `new_size`, starting at 0.
// Returns `operand` unchanged if it is already `new_size`.
HloInstruction* SliceDimTo(HloInstruction* operand, int dim, int64_t new_size) {
  int64_t old_size = operand->shape().dimensions(dim);
  if (old_size == new_size) {
    return operand;
  }
  auto computation = operand->parent();
  Shape new_shape = operand->shape();
  new_shape.set_dimensions(dim, new_size);
  llvm::SmallVector<int64_t> starts(operand->shape().dimensions().size(), 0);
  llvm::SmallVector<int64_t> limits(operand->shape().dimensions().begin(),
                                    operand->shape().dimensions().end());
  limits[dim] = new_size;
  llvm::SmallVector<int64_t> strides(operand->shape().dimensions().size(), 1);
  return computation->AddInstruction(
      HloInstruction::CreateSlice(new_shape, operand, starts, limits, strides));
}

// Returns true if `dim` is `shape`'s fastest-moving (minor-most) dimension,
// using its explicit layout if one is set, or XLA's default layout
// (descending dimension order, i.e. the last dimension is minor-most)
// otherwise.
bool IsFastestMovingDimension(const Shape& shape, int dim) {
  if (shape.has_layout()) {
    return shape.layout().minor_to_major(0) == dim;
  }
  return dim == shape.dimensions().size() - 1;
}

// Swaps the two dimensions of a rank-2 `operand`.
HloInstruction* SwapDims2D(HloInstruction* operand) {
  auto computation = operand->parent();
  Shape new_shape = ShapeUtil::MakeShape(
      operand->shape().element_type(),
      {operand->shape().dimensions(1), operand->shape().dimensions(0)});
  return computation->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, operand, {1, 0}));
}

// cuDNN's ragged-dot wgrad path (kRaggedContracting mode) lowers to a
// cuBLASLt grouped GEMM that relies on TMA and requires 16-byte alignment on
// the fastest-moving (minor-most) dimension of the lhs/rhs operands. If that
// dimension is the ragged M dimension, the alignment requirement falls on
// the per-group sizes -- which are only known at runtime and can't be
// padded at compile time. To avoid that, operands with M as the
// fastest-moving dimension are first transposed so that K/N becomes the
// fastest-moving dimension instead; K and N are static, so they can then be
// padded up to the required alignment ahead of time. The ragged-dot is kept
// (so it is still recognized and handled by the cuDNN fusion compiler), and
// the [G, K, N] result is sliced back down to the original K, N.
//
// Only the simple, non-batched wgrad shapes (2D lhs/rhs) produced for the
// cuDNN fusion path are supported; other shapes are returned unchanged.
absl::StatusOr<HloInstruction*> PadWgradForCuDNNAlignment(
    HloRaggedDotInstruction* ragged_dot) {
  const auto& ragged_dims = ragged_dot->ragged_dot_dimension_numbers();
  const auto& dot_dims = ragged_dims.dot_dimension_numbers();
  int lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  if (GetRaggedDotMode(lhs_ragged_dim, dot_dims) !=
      RaggedDotMode::kRaggedContracting) {
    return nullptr;
  }

  HloInstruction* lhs = ragged_dot->mutable_operand(0);
  HloInstruction* rhs = ragged_dot->mutable_operand(1);
  if (lhs->shape().dimensions().size() != 2 ||
      rhs->shape().dimensions().size() != 2) {
    return nullptr;
  }
  int rhs_ragged_dim = FindRhsRaggedDim(dot_dims, lhs_ragged_dim);

  // Only swap an operand's dimensions if M is actually the fastest-moving
  // one; if K/N is already the fastest-moving dimension, it is already
  // safe to pad at compile time and no transpose is needed.
  bool transposed = false;
  if (IsFastestMovingDimension(lhs->shape(), lhs_ragged_dim)) {
    lhs = SwapDims2D(lhs);
    lhs_ragged_dim = 1 - lhs_ragged_dim;
    transposed = true;
  }
  if (IsFastestMovingDimension(rhs->shape(), rhs_ragged_dim)) {
    rhs = SwapDims2D(rhs);
    rhs_ragged_dim = 1 - rhs_ragged_dim;
    transposed = true;
  }
  int lhs_k_dim = 1 - lhs_ragged_dim;
  int rhs_n_dim = 1 - rhs_ragged_dim;

  int64_t alignment_elements = std::max<int64_t>(
      16 / primitive_util::ByteWidth(lhs->shape().element_type()), 1);
  int64_t k = lhs->shape().dimensions(lhs_k_dim);
  int64_t n = rhs->shape().dimensions(rhs_n_dim);
  int64_t padded_k = RoundUpTo(k, alignment_elements);
  int64_t padded_n = RoundUpTo(n, alignment_elements);
  if (!transposed && padded_k == k && padded_n == n) {
    return nullptr;
  }

  HloInstruction* padded_lhs = PadDimTo(lhs, lhs_k_dim, padded_k);
  HloInstruction* padded_rhs = PadDimTo(rhs, rhs_n_dim, padded_n);

  // Reflect wherever M ended up (0 or 1) for each (possibly swapped)
  // operand; downstream consumers key off dnums rather than assuming a
  // fixed position.
  RaggedDotDimensionNumbers new_ragged_dims = ragged_dims;
  new_ragged_dims.set_lhs_ragged_dimensions(0, lhs_ragged_dim);
  new_ragged_dims.mutable_dot_dimension_numbers()
      ->set_lhs_contracting_dimensions(0, lhs_ragged_dim);
  new_ragged_dims.mutable_dot_dimension_numbers()
      ->set_rhs_contracting_dimensions(0, rhs_ragged_dim);

  Shape padded_shape = ragged_dot->shape();
  padded_shape.set_dimensions(1, padded_k);
  padded_shape.set_dimensions(2, padded_n);

  HloComputation* computation = ragged_dot->parent();
  HloInstruction* padded_ragged_dot =
      computation->AddInstruction(HloInstruction::CreateRaggedDot(
          padded_shape, padded_lhs, padded_rhs, ragged_dot->mutable_operand(2),
          new_ragged_dims, ragged_dot->precision_config()));
  padded_ragged_dot->set_metadata(ragged_dot->metadata());

  HloInstruction* result = SliceDimTo(padded_ragged_dot, 1, k);
  result = SliceDimTo(result, 2, n);
  return result;
}

bool CanBeHandledByGpublasltGroupGemm(
    const se::GpuComputeCapability& gpu_compute_capability,
    const HloInstruction* instruction) {
  // Currently only Hipblaslt supports GroupGemm.
  // The current status of Hipblaslt support for GroupGemm is as follows:
  // For MI300 targets (gfx942) : datatype supported FP16 and BF16
  // For MI350/355 targets (gfx950) : datatype supported FP16 only

  if (const auto* rocm_cc = gpu_compute_capability.rocm_compute_capability()) {
    const std::string& gfx_version = rocm_cc->gfx_version();
    VLOG(2) << "RaggedDotRewriter running on ROCm device: " << gfx_version;

    if (gfx_version == "gfx942" &&
        (IsFP16Operation(instruction) || IsBF16Operation(instruction))) {
      return true;
    }

    if (gfx_version == "gfx950" && IsFP16Operation(instruction)) {
      return true;
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> RaggedDotRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const bool has_grouped_gemm =
      module->config()
          .debug_options()
          .xla_gpu_experimental_use_ragged_dot_grouped_gemm() &&
      module->config().debug_options().xla_gpu_enable_cublaslt();
  const se::CudaComputeCapability* cuda_cc =
      gpu_compute_capability_.cuda_compute_capability();
  const bool ragged_dot_fusion_enabled =
      module->config()
          .debug_options()
          .xla_gpu_experimental_use_ragged_dot_fusion() &&
      cudnn_version_ >= kMinCudnnVersionForRaggedDotFusion &&
      cuda_cc != nullptr && cuda_cc->IsAtLeastAmpere();

  // Gather all Ragged Dot operations.
  std::vector<HloRaggedDotInstruction*> ragged_dots;
  std::vector<HloRaggedDotInstruction*> cudnn_fusion_dots;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kRaggedDot) {
        // Only ragged-dot that cannot be lowered through Gpublaslt
        // GroupGemm or cuDNN fusion are added to the list of operations to
        // rewrite in regular dot.
        if (ragged_dot_fusion_enabled &&
            CanBeHandledByCuDNNFusion(instruction, cudnn_version_)) {
          cudnn_fusion_dots.push_back(
              Cast<HloRaggedDotInstruction>(instruction));
          continue;
        }
        if (has_grouped_gemm && CanBeHandledByGpublasltGroupGemm(
                                    gpu_compute_capability_, instruction)) {
          continue;
        }
        ragged_dots.push_back(Cast<HloRaggedDotInstruction>(instruction));
      }
    }
  }

  bool changed = !ragged_dots.empty();

  for (auto* ragged_dot : ragged_dots) {
    ABSL_ASSIGN_OR_RETURN(auto general_dot, RaggedToGeneral(ragged_dot));
    general_dot->set_metadata(ragged_dot->metadata());
    ABSL_RETURN_IF_ERROR(ragged_dot->parent()->ReplaceWithNewInstruction(
        ragged_dot, std::move(general_dot)));
  }

  for (auto* ragged_dot : cudnn_fusion_dots) {
    ABSL_ASSIGN_OR_RETURN(HloInstruction * replacement,
                          PadWgradForCuDNNAlignment(ragged_dot));
    if (replacement == nullptr) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(
        ragged_dot->parent()->ReplaceInstruction(ragged_dot, replacement));
    changed = true;
  }

  return changed;
}

}  // namespace xla
