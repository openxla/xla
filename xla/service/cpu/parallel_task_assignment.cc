/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/parallel_task_assignment.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/intrinsic_declarations.h"
#include "xla/codegen/intrinsic/erf.h"
#include "xla/codegen/intrinsic/exp.h"
#include "xla/codegen/intrinsic/log1p.h"
#include "xla/codegen/intrinsic/rsqrt.h"
#include "xla/codegen/intrinsic/tanh.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_partition.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

const HloInstruction* StripBitcasts(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast) {
    instr = instr->operand(0);
  }
  return instr;
}

bool IsInPlaceDynamicUpdateSliceRoot(const HloInstruction* root) {
  root = StripBitcasts(root);
  if (root->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return false;
  }
  const HloInstruction* base = StripBitcasts(root->operand(0));
  return base->LatestNonGteAncestor()->opcode() == HloOpcode::kParameter;
}

bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr) {
  absl::StatusOr<BackendConfig> cpu_backend_config_or =
      instr->backend_config<BackendConfig>();
  if (cpu_backend_config_or.ok() &&
      !cpu_backend_config_or->outer_dimension_partitions().empty()) {
    return false;
  }

  if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return true;
  }

  if (instr->IsLoopFusion()) {
    const HloInstruction* fused_root = instr->fused_expression_root();
    if (fused_root->opcode() == HloOpcode::kTuple) {
      return fused_root->operand_count() > 0 &&
             absl::c_all_of(fused_root->operands(),
                            IsInPlaceDynamicUpdateSliceRoot);
    }
    return IsInPlaceDynamicUpdateSliceRoot(fused_root);
  }

  return false;
}

bool IsScatterFusion(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion &&
         instr.fusion_kind() != HloInstruction::FusionKind::kCustom &&
         instr.fused_expression_root()->opcode() == HloOpcode::kScatter;
}

using RootIndexList = absl::InlinedVector<size_t, 2>;

void InsertSortedUnique(RootIndexList& dst, size_t index) {
  const auto it = absl::c_lower_bound(dst, index);
  if (it == dst.end() || *it != index) {
    dst.insert(it, index);
  }
}

void MergeSortedUnique(RootIndexList& dst, absl::Span<const size_t> src) {
  if (src.empty()) {
    return;
  }
  if (dst.empty()) {
    dst.assign(src.begin(), src.end());
    return;
  }
  if (src.size() == 1) {
    InsertSortedUnique(dst, src[0]);
    return;
  }
  RootIndexList merged;
  merged.reserve(dst.size() + src.size());
  size_t i = 0;
  size_t j = 0;
  while (i < dst.size() && j < src.size()) {
    if (dst[i] < src[j]) {
      merged.push_back(dst[i++]);
    } else if (src[j] < dst[i]) {
      merged.push_back(src[j++]);
    } else {
      merged.push_back(dst[i++]);
      ++j;
    }
  }
  merged.insert(merged.end(), dst.begin() + i, dst.end());
  merged.insert(merged.end(), src.begin() + j, src.end());
  dst = std::move(merged);
}

double SafeArrayElements(const Shape& shape) {
  if (!shape.IsArray() || shape.is_unbounded_dynamic()) {
    return 0.0;
  }
  double count = 1.0;
  for (int64_t dim : shape.dimensions()) {
    if (dim <= 0) {
      return 0.0;
    }
    count *= static_cast<double>(dim);
  }
  return count;
}

double ElementStorageBytes(const Shape& shape) {
  if (!shape.IsArray() || !primitive_util::IsArrayType(shape.element_type())) {
    return 1.0;
  }
  if (shape.has_layout() && shape.layout().element_size_in_bits() > 0) {
    return static_cast<double>(shape.layout().element_size_in_bits()) /
           static_cast<double>(CHAR_BIT);
  }
  return static_cast<double>(CeilOfRatio<int64_t>(
      primitive_util::StorageBitWidth(shape.element_type()), CHAR_BIT));
}

template <typename Fn>
void ForEachDimensionMinorToMajor(const Shape& shape, Fn&& fn) {
  if (!shape.IsArray()) {
    return;
  }
  if (shape.has_layout() && !shape.layout().minor_to_major().empty()) {
    for (int64_t dim : shape.layout().minor_to_major()) {
      if (!fn(dim)) {
        break;
      }
    }
    return;
  }
  for (int64_t dim = static_cast<int64_t>(shape.dimensions().size()) - 1;
       dim >= 0; --dim) {
    if (!fn(dim)) {
      break;
    }
  }
}

int64_t EffectiveMinorDimension(const Shape& shape) {
  int64_t result = -1;
  ForEachDimensionMinorToMajor(shape, [&](int64_t dim) {
    if (shape.dimensions(dim) > 1) {
      result = dim;
      return false;
    }
    return true;
  });
  return result;
}

struct StridedTransposeInfo {
  double stride_bytes = 0.0;
  double sweep_bytes = 0.0;
};

// Detects physical transposes expressed either as a kTranspose or as a layout
// changing kCopy where the innermost output loop reads the source with non unit
// stride.
std::optional<StridedTransposeInfo> GetStridedTransposeInfo(
    const HloInstruction& instr) {
  const HloOpcode opcode = instr.opcode();
  if (opcode != HloOpcode::kTranspose && opcode != HloOpcode::kCopy) {
    return std::nullopt;
  }
  if (opcode == HloOpcode::kTranspose && instr.IsEffectiveBitcast()) {
    return std::nullopt;
  }
  const Shape& src = instr.operand(0)->shape();
  const Shape& dst = instr.shape();
  if (!src.IsArray() || !dst.IsArray() ||
      src.dimensions().size() != dst.dimensions().size()) {
    return std::nullopt;
  }
  const int64_t src_minor = EffectiveMinorDimension(src);
  const int64_t dst_minor = EffectiveMinorDimension(dst);
  if (src_minor < 0 || dst_minor < 0) {
    return std::nullopt;
  }
  const int64_t mapped_src_dim =
      opcode == HloOpcode::kTranspose ? instr.dimensions(dst_minor) : dst_minor;
  if (mapped_src_dim == src_minor) {
    return std::nullopt;
  }
  double stride_elems = 1.0;
  ForEachDimensionMinorToMajor(src, [&](int64_t dim) {
    if (dim == mapped_src_dim) {
      return false;
    }
    stride_elems *=
        static_cast<double>(std::max<int64_t>(1, src.dimensions(dim)));
    return true;
  });
  const double stride_bytes = stride_elems * ElementStorageBytes(src);
  const double inner_steps =
      static_cast<double>(std::max<int64_t>(1, dst.dimensions(dst_minor)));
  return StridedTransposeInfo{stride_bytes, stride_bytes * inner_steps};
}

bool CanEmitFastConcatenateCopy(const HloInstruction& concat,
                                const HloFusionInstruction* enclosing_fusion) {
  if (enclosing_fusion != nullptr &&
      enclosing_fusion->fused_expression_root() != &concat) {
    return false;
  }
  const Shape& output_shape = concat.shape();
  if (!output_shape.IsArray()) {
    return false;
  }
  if (enclosing_fusion != nullptr) {
    for (const HloInstruction* operand : concat.operands()) {
      if (operand->opcode() != HloOpcode::kParameter) {
        return false;
      }
      if (output_shape.has_layout() && operand->shape().has_layout() &&
          !LayoutUtil::Equal(operand->shape().layout(),
                             output_shape.layout())) {
        return false;
      }
    }
  }
  return true;
}

// Returns true when `concat` is the fusion hero routed to
// ConcatenateFusionKernelEmitter, which emits one loop per operand rather than
// a per element binary search branch tree.
bool IsConcatenateFusionHero(const HloInstruction& concat,
                             const HloFusionInstruction* enclosing_fusion) {
  if (enclosing_fusion == nullptr) {
    return true;
  }
  const HloInstruction* curr = enclosing_fusion->fused_expression_root();
  while (curr != &concat) {
    if (curr->operand_count() == 1 &&
        (curr->opcode() == HloOpcode::kBitcast ||
         curr->opcode() == HloOpcode::kReshape ||
         (curr->IsElementwise() && curr->opcode() != HloOpcode::kCopy))) {
      curr = curr->operand(0);
    } else {
      return false;
    }
  }
  return true;
}

double ConcatenateOuterElements(const HloInstruction& concat) {
  const Shape& output_shape = concat.shape();
  const int64_t concat_dim = concat.concatenate_dimension();
  bool past_concat_dim = false;
  double outer_elems = 1.0;
  ForEachDimensionMinorToMajor(output_shape, [&](int64_t dim) {
    if (past_concat_dim) {
      outer_elems *= static_cast<double>(
          std::max<int64_t>(1, output_shape.dimensions(dim)));
    } else if (dim == concat_dim) {
      past_concat_dim = true;
    }
    return true;
  });
  return outer_elems;
}

// Partitioning a concatenate with many narrow slices (such as 216 scalar
// columns) replaces constant loop bounds with dynamic workgroup bounds around
// hundreds of scalar memory ops, triggering superlinear LLVM compile times for
// little or no runtime gain.
bool IsManyOperandNarrowConcatenate(const HloInstruction& instr) {
  constexpr int64_t kMaxParallelConcatOperands = 64;
  constexpr double kMaxNarrowSliceBytes = 8.0;
  const HloInstruction* concat = &instr;
  if (instr.IsLoopFusion()) {
    concat = StripBitcasts(instr.fused_expression_root());
  }
  if (concat->opcode() != HloOpcode::kConcatenate ||
      concat->operand_count() < kMaxParallelConcatOperands) {
    return false;
  }
  const double total_bytes =
      SafeArrayElements(concat->shape()) * ElementStorageBytes(concat->shape());
  const double slice_copies = ConcatenateOuterElements(*concat) *
                              static_cast<double>(concat->operand_count());
  if (slice_copies <= 0.0) {
    return false;
  }
  const double avg_slice_bytes = total_bytes / slice_copies;
  return avg_slice_bytes <= kMaxNarrowSliceBytes;
}

// Returns the single outermost dimension size for a fast copy concatenate, or 0
// when concatenate_dimension is the outermost physical dimension (where
// ConcatenateFusionKernelEmitter always emits one workgroup). Returns nullopt
// for non fast copy instructions.
std::optional<int64_t> FastConcatenateOuterDimSize(
    const HloInstruction& instr) {
  const HloInstruction* concat = nullptr;
  const HloFusionInstruction* enclosing = nullptr;
  if (instr.opcode() == HloOpcode::kConcatenate) {
    concat = &instr;
  } else if (instr.IsLoopFusion()) {
    enclosing = Cast<HloFusionInstruction>(&instr);
    if (enclosing->fused_expression_root()->opcode() ==
        HloOpcode::kConcatenate) {
      concat = enclosing->fused_expression_root();
    }
  }
  if (concat == nullptr || !CanEmitFastConcatenateCopy(*concat, enclosing)) {
    return std::nullopt;
  }
  const Shape& output_shape = concat->shape();
  const int64_t concat_dim = concat->concatenate_dimension();
  int64_t outer_dim = -1;
  bool past_concat_dim = false;
  ForEachDimensionMinorToMajor(output_shape, [&](int64_t dim) {
    if (past_concat_dim) {
      outer_dim = dim;
    } else if (dim == concat_dim) {
      past_concat_dim = true;
    }
    return true;
  });
  if (outer_dim < 0) {
    return int64_t{0};
  }
  return output_shape.dimensions(outer_dim);
}

// Returns the product of contiguous innermost output dimensions that are not
// sourced from the broadcast operand. LLVM LICM hoists operand evaluation out
// of those inner loops.
double BroadcastInvariantInnerElements(const HloInstruction& broadcast) {
  const Shape& out_shape = broadcast.shape();
  if (!out_shape.IsArray() || out_shape.dimensions().empty()) {
    return 1.0;
  }
  absl::Span<const int64_t> bcast_dims = broadcast.dimensions();
  double invariant = 1.0;
  ForEachDimensionMinorToMajor(out_shape, [&](int64_t dim) {
    if (absl::c_linear_search(bcast_dims, dim)) {
      return false;
    }
    int64_t size = out_shape.dimensions(dim);
    if (size > 0) {
      invariant *= static_cast<double>(size);
    }
    return true;
  });
  return std::max(1.0, invariant);
}

bool PreservesElementwiseIndexing(const HloInstruction& instr) {
  if (instr.IsElementwise() || instr.opcode() == HloOpcode::kTuple ||
      instr.opcode() == HloOpcode::kGetTupleElement) {
    return true;
  }
  switch (instr.opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return instr.operand_count() == 1 && instr.shape().IsArray() &&
             instr.operand(0)->shape().IsArray() &&
             SafeArrayElements(instr.shape()) ==
                 SafeArrayElements(instr.operand(0)->shape());
    default:
      return false;
  }
}

const HloInstruction* UnwrapConstantOperand(
    const HloInstruction* operand,
    const HloFusionInstruction* enclosing_fusion) {
  while (operand != nullptr) {
    switch (operand->opcode()) {
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy:
        operand = operand->operand(0);
        break;
      case HloOpcode::kParameter:
        if (enclosing_fusion == nullptr) {
          return nullptr;
        }
        operand = enclosing_fusion->operand(operand->parameter_number());
        enclosing_fusion = nullptr;
        break;
      case HloOpcode::kConstant:
        return operand;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

bool IsEffectiveScalarConstant(const HloInstruction* operand,
                               const HloFusionInstruction* enclosing_fusion) {
  const HloInstruction* constant =
      UnwrapConstantOperand(operand, enclosing_fusion);
  return constant != nullptr && constant->shape().IsArray() &&
         ShapeUtil::IsEffectiveScalar(constant->shape());
}

bool IsScalarFloatOne(const HloInstruction* operand,
                      const HloFusionInstruction* enclosing_fusion) {
  const HloInstruction* constant =
      UnwrapConstantOperand(operand, enclosing_fusion);
  if (constant == nullptr || !constant->shape().IsArray() ||
      !ShapeUtil::IsEffectiveScalar(constant->shape()) ||
      !primitive_util::IsFloatingPointType(constant->shape().element_type())) {
    return false;
  }
  DimensionVector zero_index(constant->shape().dimensions().size(), 0);
  std::optional<double> value = constant->literal().GetAsDouble(zero_index);
  return value.has_value() && *value == 1.0;
}

double GatherContiguousSliceBytes(const HloInstruction& gather) {
  const Shape& operand_shape = gather.operand(0)->shape();
  if (!operand_shape.IsArray() || SafeArrayElements(gather.shape()) == 0.0) {
    return 0.0;
  }
  const double elem_bytes = ElementStorageBytes(operand_shape);
  absl::Span<const int64_t> slice_sizes = gather.gather_slice_sizes();
  double contiguous_elems = 1.0;
  ForEachDimensionMinorToMajor(operand_shape, [&](int64_t dim) {
    const int64_t slice_dim = slice_sizes[dim];
    const int64_t full_dim = operand_shape.dimensions(dim);
    contiguous_elems *= static_cast<double>(std::max<int64_t>(1, slice_dim));
    return slice_dim >= full_dim;
  });
  return std::max(elem_bytes, contiguous_elems * elem_bytes);
}

bool IsAssociativeIntegerReducer(const HloComputation& comp) {
  const HloInstruction* root = comp.root_instruction();
  if (!root->shape().IsArray()) {
    return false;
  }
  const PrimitiveType root_ty = root->shape().element_type();
  if (!primitive_util::IsIntegralType(root_ty) && root_ty != PRED) {
    return false;
  }
  for (const HloInstruction* instr : comp.instructions()) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kAdd:
      case HloOpcode::kMultiply:
      case HloOpcode::kMinimum:
      case HloOpcode::kMaximum:
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
        break;
      default:
        return false;
    }
  }
  return true;
}

bool ReducesOnlyContiguousMinorDimension(const Shape& input_shape,
                                         absl::Span<const int64_t> dims) {
  if (!input_shape.IsArray() || dims.empty()) {
    return false;
  }
  int64_t active_reduced = 0;
  int64_t sole_dim = -1;
  for (int64_t dim : dims) {
    if (input_shape.dimensions(dim) > 1) {
      ++active_reduced;
      sole_dim = dim;
    }
  }
  return active_reduced == 1 &&
         sole_dim == EffectiveMinorDimension(input_shape);
}

bool WindowReducesOnlyContiguousMinorDimension(const Shape& input_shape,
                                               const Window& window) {
  if (!input_shape.IsArray()) {
    return false;
  }
  int64_t active_reduced = 0;
  int64_t sole_dim = -1;
  for (int64_t dim = 0; dim < window.dimensions_size(); ++dim) {
    if (window.dimensions(dim).size() > 1) {
      ++active_reduced;
      sole_dim = dim;
    }
  }
  return active_reduced == 1 &&
         sole_dim == EffectiveMinorDimension(input_shape);
}

const HloInstruction* DirectMemoryParameterOrConstant(
    const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape ||
         instr->opcode() == HloOpcode::kCopy ||
         (instr->opcode() == HloOpcode::kTranspose &&
          instr->IsEffectiveBitcast())) {
    instr = instr->operand(0);
  }
  if (instr->opcode() == HloOpcode::kParameter ||
      instr->opcode() == HloOpcode::kConstant) {
    return instr;
  }
  if (instr->opcode() == HloOpcode::kGetTupleElement &&
      instr->LatestNonGteAncestor()->opcode() == HloOpcode::kParameter) {
    return instr;
  }
  return nullptr;
}

// Returns true when a strided transpose or layout changing copy actually
// performs strided memory loads rather than reading a fused register
// intermediate or feeding only a fused dot contraction.
bool IsDirectMemoryStridedTranspose(
    const HloInstruction& op, const HloFusionInstruction* enclosing_fusion) {
  if (enclosing_fusion == nullptr) {
    return true;
  }
  if (op.operand_count() == 0 ||
      DirectMemoryParameterOrConstant(op.operand(0)) == nullptr) {
    return false;
  }
  if (op.user_count() > 0 &&
      absl::c_all_of(op.users(), [](const HloInstruction* user) {
        return user->opcode() == HloOpcode::kDot;
      })) {
    return false;
  }
  return true;
}

bool IsTiledEmitterSupportedType(PrimitiveType type) {
  if (type == PRED) {
    return true;
  }
  if (primitive_util::BitWidth(type) < 8) {
    return false;
  }
  return !primitive_util::IsF8Type(type) && !primitive_util::IsF6Type(type) &&
         !primitive_util::IsMXType(type);
}

// Mirrors IsSupportedTiledDot in tiled_fusion_emitter.cc for pure HLO queries.
bool IsTiledGemvDot(const HloInstruction& dot) {
  constexpr int kTiledMaxRank = 8;
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& out_shape = dot.shape();
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  if (out_shape.dimensions().size() > kTiledMaxRank ||
      lhs_shape.dimensions().size() > kTiledMaxRank ||
      rhs_shape.dimensions().size() > kTiledMaxRank) {
    return false;
  }
  const PrimitiveType dtype = out_shape.element_type();
  if (lhs_shape.element_type() != dtype || rhs_shape.element_type() != dtype ||
      dtype == PRED || primitive_util::IsComplexType(dtype) ||
      !IsTiledEmitterSupportedType(dtype)) {
    return false;
  }
  const DotDimensionNumbers& dnums = dot.dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1) {
    return false;
  }
  const int64_t num_batch = dnums.lhs_batch_dimensions_size();
  const int64_t lhs_free_count =
      static_cast<int64_t>(lhs_shape.dimensions().size()) - num_batch - 1;
  const int64_t rhs_free_count =
      static_cast<int64_t>(rhs_shape.dimensions().size()) - num_batch - 1;
  if (lhs_free_count > 1 || rhs_free_count > 1) {
    return false;
  }
  auto free_dim_size = [](const Shape& shape, absl::Span<const int64_t> batch,
                          absl::Span<const int64_t> contracting) {
    int64_t size = 1;
    for (int64_t d = 0; d < static_cast<int64_t>(shape.dimensions().size());
         ++d) {
      if (!absl::c_linear_search(batch, d) &&
          !absl::c_linear_search(contracting, d)) {
        size *= shape.dimensions(d);
      }
    }
    return size;
  };
  const int64_t lhs_free =
      free_dim_size(lhs_shape, dnums.lhs_batch_dimensions(),
                    dnums.lhs_contracting_dimensions());
  const int64_t rhs_free =
      free_dim_size(rhs_shape, dnums.rhs_batch_dimensions(),
                    dnums.rhs_contracting_dimensions());
  if (lhs_free != 1 && rhs_free != 1) {
    return false;
  }
  for (int operand_precision : dot.precision_config().operand_precision()) {
    if (operand_precision != PrecisionConfig::DEFAULT) {
      return false;
    }
  }
  const PrecisionConfig::Algorithm alg = dot.precision_config().algorithm();
  return alg == PrecisionConfig::ALG_UNSET ||
         (alg == PrecisionConfig::ALG_DOT_F32_F32_F32 && dtype == F32) ||
         (alg == PrecisionConfig::ALG_DOT_F64_F64_F64 && dtype == F64);
}

// Mirrors IsSupportedTiledFusion in tiled_fusion_emitter.cc so cost estimation
// recognizes fusions lowered via vectorized cache tiles instead of scalar
// loops.
bool IsTiledFusionSupported(const HloFusionInstruction& fusion,
                            bool enable_tiled_emitter) {
  constexpr int kTiledMaxRank = 8;
  if (!enable_tiled_emitter || !fusion.shape().IsArray()) {
    return false;
  }
  for (const HloInstruction* operand : fusion.operands()) {
    if (!operand->shape().IsArray()) {
      return false;
    }
  }
  int dot_count = 0;
  for (const HloInstruction* inst : fusion.fused_instructions()) {
    if (!inst->shape().IsArray() ||
        inst->shape().dimensions().size() > kTiledMaxRank ||
        !IsTiledEmitterSupportedType(inst->shape().element_type())) {
      return false;
    }
    const HloOpcode opcode = inst->opcode();
    switch (opcode) {
      case HloOpcode::kParameter:
      case HloOpcode::kIota:
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
      case HloOpcode::kBitcast:
      case HloOpcode::kConvert:
        break;
      case HloOpcode::kConstant:
        if (!ShapeUtil::IsEffectiveScalar(inst->shape())) {
          return false;
        }
        break;
      case HloOpcode::kDot:
        if (++dot_count > 1 || !IsTiledGemvDot(*inst)) {
          return false;
        }
        break;
      case HloOpcode::kBroadcast:
      case HloOpcode::kReduce:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kMap:
      case HloOpcode::kPopulationCount:
      case HloOpcode::kSign:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kRoundNearestEven:
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightArithmetic:
      case HloOpcode::kShiftRightLogical:
      case HloOpcode::kClz:
      case HloOpcode::kMulhi:
        return false;
      default:
        if (!inst->IsElementwise()) {
          return false;
        }
        break;
    }
  }
  return true;
}

const Shape* IndexingShapeForInstruction(const HloInstruction& instruction) {
  const Shape& root_0_shape = instruction.shape().IsTuple()
                                  ? instruction.shape().tuple_shapes(0)
                                  : instruction.shape();
  const Shape& indexing_shape =
      root_0_shape.IsTuple() ? root_0_shape.tuple_shapes(0) : root_0_shape;
  return indexing_shape.IsArray() ? &indexing_shape : nullptr;
}

// Mirrors GetWorkDimensions in fusion_emitter.cc: folds leading physical
// dimensions from major to minor until the product reaches `target_tasks`.
int64_t FoldedPartitionDimension(const Shape& shape, int64_t target_tasks) {
  if (!shape.IsArray() || shape.dimensions().empty()) {
    return 1;
  }
  int64_t folded = 1;
  if (shape.has_layout() && !shape.layout().minor_to_major().empty()) {
    absl::Span<const int64_t> min2maj = shape.layout().minor_to_major();
    for (int i = static_cast<int>(min2maj.size()) - 1; i >= 0; --i) {
      folded *= std::max<int64_t>(1, shape.dimensions(min2maj[i]));
      if (folded >= target_tasks) {
        return folded;
      }
    }
    return folded;
  }
  for (int64_t dim : shape.dimensions()) {
    folded *= std::max<int64_t>(1, dim);
    if (folded >= target_tasks) {
      return folded;
    }
  }
  return folded;
}

template <typename Intrinsic>
int64_t MaxIntrinsicVectorWidth(absl::string_view features, PrimitiveType type,
                                int64_t max_lanes) {
  int64_t best = 0;
  for (const std::vector<codegen::intrinsics::Type>& signature :
       Intrinsic::SupportedVectorTypes(features)) {
    if (signature.size() != 1 || signature.front().element_type() != type) {
      continue;
    }
    int64_t width = signature.front().vector_width().value_or(1);
    if (width <= max_lanes) {
      best = std::max(best, width);
    }
  }
  return best;
}

int64_t EffectiveVectorWidthBytes(
    const HloModule& module,
    const TargetMachineFeatures& target_machine_features) {
  int64_t hw_bytes =
      std::max<int64_t>(0, target_machine_features.vector_register_byte_size());
  int64_t prefer_bits =
      module.config().debug_options().xla_cpu_prefer_vector_width();
  if (hw_bytes > 0 && prefer_bits > 0) {
    return std::min<int64_t>(hw_bytes, std::max<int64_t>(1, prefer_bits / 8));
  }
  return hw_bytes;
}

struct ExecutionCostNs {
  // Per core execution work (ALU, SIMD intrinsics, scalar libm, scalar reducer
  // loops, random cache miss MLP stalls, and large stride DTLB miss stalls)
  // that scales linearly with core count.
  double core_ns = 0.0;
  // Shared cache and DRAM bandwidth time that saturates at max_memory_tasks_.
  double memory_ns = 0.0;
};

// Caps memory bound parallelism at ceil(sqrt(max_parallelism)), matching the
// legacy empirically fitted I/O bound scaling on 8 to 64 core hosts, with an
// upper bound of 12 so 144+ thread servers do not oversubscribe DRAM channels.
int64_t ComputeMaxMemoryBandwidthTasks(int64_t max_parallelism) {
  if (max_parallelism <= 1) {
    return 1;
  }
  const int64_t sqrt_cap = static_cast<int64_t>(
      std::ceil(std::sqrt(static_cast<double>(max_parallelism))));
  return std::clamp<int64_t>(sqrt_cap, 1, 12);
}

// Execution time model for CPU loop fusions and parallelizable standalone ops.
// Estimates per core execution time (accounting for target SIMD width, element
// width, vectorized intrinsics vs scalar libm calls, scalar reduction loops,
// and per core miss latency) and cache tiered memory traffic (including write
// allocate traffic), then picks the task count in [1, max_parallelism] that
// minimizes the geometric mean of predicted wall time and predicted CPU time
// (wall * sqrt(tasks)) including Worker::Parallelize dispatch cost, shared
// memory bandwidth saturation, and workgroup load balance.
class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(int64_t max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   int64_t effective_vector_bytes, std::string enabled_features,
                   bool has_avx512bf16, bool enable_tiled_emitter,
                   CpuDeviceSpec cpu_device_spec)
      : max_parallelism_(max_parallelism),
        max_memory_tasks_(ComputeMaxMemoryBandwidthTasks(max_parallelism)),
        shape_size_(shape_size),
        effective_vector_bytes_(effective_vector_bytes),
        enabled_features_(std::move(enabled_features)),
        has_avx512bf16_(has_avx512bf16),
        enable_tiled_emitter_(enable_tiled_emitter),
        cpu_device_spec_(cpu_device_spec),
        has_cpp_gen_intrinsics_(codegen::AreCppGenIntrinsicsAvailable()) {}
  ~DefaultCostModel() override = default;

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    if (max_parallelism_ <= 1 || instruction->shape().is_unbounded_dynamic()) {
      return 1;
    }
    for (const HloInstruction* operand : instruction->operands()) {
      if (operand->shape().is_unbounded_dynamic()) {
        return 1;
      }
    }

    if (IsScatterFusion(*instruction)) {
      return ChooseScatterFusionTaskCount(
          *Cast<HloFusionInstruction>(instruction));
    }

    const ExecutionCostNs cost =
        instruction->opcode() == HloOpcode::kFusion
            ? EstimateFusionCost(*Cast<HloFusionInstruction>(instruction))
            : EstimateStandaloneInstructionCost(*instruction);
    const int64_t tasks = ChooseTaskCount(cost, *instruction);
    VLOG(3) << "ParallelCostModel " << instruction->name()
            << " core_us=" << (cost.core_ns / 1e3)
            << " memory_us=" << (cost.memory_ns / 1e3)
            << " vec_bytes=" << effective_vector_bytes_
            << " -> tasks=" << tasks;
    return tasks;
  }

 private:
  // Nominal all core turbo frequency (2.5 GHz) for server x86 CPUs under SIMD
  // load, used to convert cycle estimates to nanoseconds for direct comparison
  // with memory bandwidth and dispatch latency.
  static constexpr double kCyclesPerNs = 2.5;
  static constexpr double kCacheLineBytes = 64.0;
  // Written cache lines incur read for ownership plus writeback (R + 2W).
  static constexpr double kWriteAllocateFactor = 2.0;

  // Piecewise single core streaming bandwidth model calibrated to x86 server
  // cache hierarchies: private L2 (first 512 KiB at 72 GB/s), active shared L3
  // working set (up to 16 MiB at 28 GB/s), and single core DRAM streaming
  // (16 GB/s, bounded by per core line fill buffers).
  static constexpr double kL2TierBytes = static_cast<double>(512LL << 10);
  static constexpr double kL3TotalBytes = static_cast<double>(16LL << 20);
  static constexpr double kL3TierBytes = kL3TotalBytes - kL2TierBytes;
  static constexpr double kL2BytesPerNs = 72.0;
  static constexpr double kL3BytesPerNs = 28.0;
  static constexpr double kDramBytesPerNs = 16.0;

  // Per core miss latency for random gather fetches beyond L2 or L3, and for
  // untiled strided transpose rows whose span exceeds hugepage L1 DTLB cover.
  static constexpr double kRandomL3FetchNs = 2.5;
  static constexpr double kRandomDramFetchNs = 8.0;
  static constexpr double kDtlbSweepThresholdBytes =
      static_cast<double>(128LL << 20);
  static constexpr double kDtlbMissStrideBytes = 2048.0;
  static constexpr double kTransposeDtlbMissCycles = 10.0;

  // Worker::Parallelize Eigen thread pool fork/join latency calibrated from
  // Perflab benchmarks (~8 us fixed wake/join barrier plus ~1.5 us per spawned
  // task) and a 16 us per task floor (~40K cycles or ~256 KiB L2/L3 traffic)
  // so small kernels stay single threaded.
  static constexpr double kDispatchBaseNs = 8'000.0;
  static constexpr double kDispatchPerTaskNs = 1'500.0;
  static constexpr double kMinWorkPerTaskNs = 16'000.0;

  // LoopFusionKernelEmitter wraps the loop body in an scf.if bounds guard when
  // the workgroup count does not evenly divide the partitioned dimension, which
  // impedes vectorization and adds compile overhead, steering choices toward
  // exact divisors when available.
  static constexpr double kInBoundsGuardFactor = 1.25;

  static double SequentialMemoryTimeNs(double read_bytes, double write_bytes) {
    const double physical_bytes =
        std::max(0.0, read_bytes) +
        kWriteAllocateFactor * std::max(0.0, write_bytes);
    if (physical_bytes <= 0.0) {
      return 0.0;
    }
    const double l2 = std::min(physical_bytes, kL2TierBytes);
    const double rem = physical_bytes - l2;
    const double l3 = std::min(rem, kL3TierBytes);
    const double dram = std::max(0.0, rem - l3);
    return l2 / kL2BytesPerNs + l3 / kL3BytesPerNs + dram / kDramBytesPerNs;
  }

  double ArrayBytes(const Shape& shape) const {
    if (!shape.IsArray() || shape.is_unbounded_dynamic()) {
      return 0.0;
    }
    if (LayoutUtil::HasLayout(shape)) {
      return static_cast<double>(std::max<int64_t>(0, shape_size_(shape)));
    }
    return static_cast<double>(
        std::max<int64_t>(0, ShapeUtil::ByteSizeOf(shape, sizeof(void*))));
  }

  double RecursiveArrayBytes(const Shape& shape) const {
    double total = 0.0;
    ShapeUtil::ForEachSubshape(shape,
                               [&](const Shape& subshape, const ShapeIndex&) {
                                 if (subshape.IsArray()) {
                                   total += ArrayBytes(subshape);
                                 }
                               });
    return total;
  }

  int64_t VectorLanes(PrimitiveType type) const {
    if (effective_vector_bytes_ <= 0 || !primitive_util::IsArrayType(type)) {
      return 1;
    }
    if (primitive_util::IsComplexType(type)) {
      type = primitive_util::ComplexComponentType(type);
    }
    const int64_t elem_bytes =
        CeilOfRatio<int64_t>(primitive_util::StorageBitWidth(type), CHAR_BIT);
    return std::max<int64_t>(1, effective_vector_bytes_ / elem_bytes);
  }

  static PrimitiveType FloatComputeType(PrimitiveType type) {
    if (primitive_util::IsComplexType(type)) {
      type = primitive_util::ComplexComponentType(type);
    }
    return type == F64 ? F64 : F32;
  }

  struct IntrinsicCycleCosts {
    double f32_vec;
    double f64_vec;
    double f32_libm;
    double f64_libm;
  };

  template <typename Intrinsic>
  double IntrinsicCycles(PrimitiveType type,
                         const IntrinsicCycleCosts& costs) const {
    const PrimitiveType cty = FloatComputeType(type);
    const int64_t max_lanes = VectorLanes(cty);
    const int64_t width =
        MaxIntrinsicVectorWidth<Intrinsic>(enabled_features_, cty, max_lanes);
    if (width > 0) {
      const double vec_cycles = cty == F64 ? costs.f64_vec : costs.f32_vec;
      return vec_cycles / static_cast<double>(width);
    }
    return cty == F64 ? costs.f64_libm : costs.f32_libm;
  }

  double ExpCycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    const PrimitiveType cty = FloatComputeType(type);
    if (cty == F64) {
      return IntrinsicCycles<ci::Exp>(
          F64, {/*f32_vec=*/15.0, /*f64_vec=*/26.0, /*f32_libm=*/35.0,
                /*f64_libm=*/65.0});
    }
    // PolynomialApproximations vectorizes F32 and F16 exp at widths up to 16;
    // degree 5 Estrin evaluation plus clamping pipelines at ~9 cycles/vector
    // across dual FMA and vector ALU ports.
    const int64_t width = std::min<int64_t>(16, VectorLanes(F32));
    return 9.0 / static_cast<double>(width) + SubF32ConvertOverhead(type);
  }

  double SinCosCycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    return IntrinsicCycles<ci::Sin>(
               type, {/*f32_vec=*/38.0, /*f64_vec=*/66.0, /*f32_libm=*/45.0,
                      /*f64_libm=*/75.0}) +
           SubF32ConvertOverhead(type);
  }

  double TanhCycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    return IntrinsicCycles<ci::Tanh>(
               type, {/*f32_vec=*/17.0, /*f64_vec=*/26.0, /*f32_libm=*/35.0,
                      /*f64_libm=*/60.0}) +
           SubF32ConvertOverhead(type);
  }

  double Expm1Cycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    if (has_cpp_gen_intrinsics_) {
      return IntrinsicCycles<ci::Expm1>(
                 type, {/*f32_vec=*/26.0, /*f64_vec=*/40.0, /*f32_libm=*/40.0,
                        /*f64_libm=*/65.0}) +
             SubF32ConvertOverhead(type);
    }
    // ExpandFloatOpsPass rewrites expm1 to exp + tanh + arithmetic when CppGen
    // intrinsics are unavailable.
    const double lanes =
        static_cast<double>(VectorLanes(FloatComputeType(type)));
    return ExpCycles(type) + TanhCycles(type) + 4.0 / lanes;
  }

  double ConvertF32ToBf16VecCycles() const {
    // Even with AVX512 BF16, XLA emits NaN canonicalization around truncation.
    return has_avx512bf16_ ? 1.5 : 2.5;
  }

  // Operations left directly on BF16 or F16 expand around an F32 computation;
  // when FloatNormalization has already inserted explicit kConvert ops, `type`
  // is F32 and this returns 0.
  double SubF32ConvertOverhead(PrimitiveType type) const {
    if (type == BF16) {
      return (1.2 + ConvertF32ToBf16VecCycles()) /
             static_cast<double>(VectorLanes(F32));
    }
    if (type == F16) {
      return 3.0 / static_cast<double>(VectorLanes(F32));
    }
    return 0.0;
  }

  double ElementwiseOpCyclesPerElement(
      const HloInstruction& op,
      const HloFusionInstruction* enclosing_fusion) const {
    namespace ci = codegen::intrinsics;
    const PrimitiveType ty = op.shape().element_type();
    const bool is_complex = primitive_util::IsComplexType(ty);
    const PrimitiveType fty = FloatComputeType(ty);

    switch (op.opcode()) {
      case HloOpcode::kBitcastConvert:
        return 0.0;

      case HloOpcode::kCopy:
        if (IsDirectMemoryStridedTranspose(op, enclosing_fusion)) {
          if (std::optional<StridedTransposeInfo> info =
                  GetStridedTransposeInfo(op);
              info.has_value()) {
            const bool is_tiled = enclosing_fusion != nullptr &&
                                  IsTiledFusionSupported(*enclosing_fusion,
                                                         enable_tiled_emitter_);
            return StridedTransposeCyclesPerElement(*info, is_tiled);
          }
        }
        return 0.0;

      case HloOpcode::kExp:
        if (is_complex) {
          return ExpCycles(fty) + 2.0 * SinCosCycles(fty);
        }
        return ExpCycles(ty);

      case HloOpcode::kLogistic: {
        const double lanes = static_cast<double>(VectorLanes(fty));
        const double div_cycles = (fty == F64 ? 14.0 : 8.0) / lanes;
        if (is_complex) {
          return ExpCycles(fty) + 2.0 * SinCosCycles(fty) + 24.0 / lanes;
        }
        return ExpCycles(ty) + div_cycles;
      }

      case HloOpcode::kExpm1:
        if (is_complex) {
          return ExpCycles(fty) + 2.0 * SinCosCycles(fty);
        }
        return Expm1Cycles(ty);

      case HloOpcode::kLog: {
        static constexpr IntrinsicCycleCosts kLogCosts = {
            /*f32_vec=*/17.0, /*f64_vec=*/26.0, /*f32_libm=*/35.0,
            /*f64_libm=*/60.0};
        if (is_complex) {
          // Complex log computes atan2(imag, real), which calls scalar libm.
          return (fty == F64 ? 70.0 : 40.0) +
                 IntrinsicCycles<ci::YnnLog>(fty, kLogCosts);
        }
        return IntrinsicCycles<ci::YnnLog>(ty, kLogCosts) +
               SubF32ConvertOverhead(ty);
      }

      case HloOpcode::kLog1p: {
        static constexpr IntrinsicCycleCosts kLog1pCosts = {
            /*f32_vec=*/24.0, /*f64_vec=*/34.0, /*f32_libm=*/40.0,
            /*f64_libm=*/65.0};
        if (is_complex) {
          return (fty == F64 ? 75.0 : 45.0) +
                 IntrinsicCycles<ci::Log1p>(fty, kLog1pCosts);
        }
        if (has_cpp_gen_intrinsics_) {
          return IntrinsicCycles<ci::YnnLog1p>(ty, kLog1pCosts) +
                 SubF32ConvertOverhead(ty);
        }
        return IntrinsicCycles<ci::Log1p>(ty, kLog1pCosts) +
               SubF32ConvertOverhead(ty);
      }

      case HloOpcode::kTanh:
        if (is_complex) {
          return 2.0 * (ExpCycles(fty) + SinCosCycles(fty)) +
                 24.0 / static_cast<double>(VectorLanes(fty));
        }
        return TanhCycles(ty);

      case HloOpcode::kErf:
        if (fty == F32) {
          // ExpandFloatOpsPass rewrites F32 erf to a vectorized rational
          // polynomial regardless of target features.
          return 16.0 / static_cast<double>(VectorLanes(F32)) +
                 SubF32ConvertOverhead(ty);
        }
        return IntrinsicCycles<ci::Erf>(
            F64, {/*f32_vec=*/16.0, /*f64_vec=*/40.0, /*f32_libm=*/35.0,
                  /*f64_libm=*/50.0});

      case HloOpcode::kSin:
      case HloOpcode::kCos:
        if (is_complex) {
          return 2.0 * ExpCycles(fty) + 2.0 * SinCosCycles(fty);
        }
        return SinCosCycles(ty);

      case HloOpcode::kSinh:
      case HloOpcode::kCosh:
        if (is_complex) {
          return 2.0 * (ExpCycles(fty) + 2.0 * SinCosCycles(fty));
        }
        if (fty == F64) {
          // ElementaryFunctionExpander expands F64 sinh/cosh to 2*exp + expm1 +
          // div.
          return 2.0 * ExpCycles(F64) + Expm1Cycles(F64) +
                 14.0 / static_cast<double>(VectorLanes(F64));
        }
        return IntrinsicCycles<ci::Sinh>(
                   ty, {/*f32_vec=*/32.0, /*f64_vec=*/60.0, /*f32_libm=*/45.0,
                        /*f64_libm=*/75.0}) +
               SubF32ConvertOverhead(ty);

      case HloOpcode::kAsin:
      case HloOpcode::kAcos:
        if (is_complex) {
          return fty == F64 ? 110.0 : 70.0;
        }
        if (fty == F64) {
          // ElementaryFunctionExpander expands F64 asin/acos to sqrt + scalar
          // libm atan2.
          return 70.0 + 8.0 / static_cast<double>(VectorLanes(F64));
        }
        return IntrinsicCycles<ci::Acos>(
                   ty, {/*f32_vec=*/32.0, /*f64_vec=*/80.0, /*f32_libm=*/45.0,
                        /*f64_libm=*/80.0}) +
               SubF32ConvertOverhead(ty);

      case HloOpcode::kAtan2:
        if (!is_complex && IsScalarFloatOne(op.operand(1), enclosing_fusion)) {
          // SimplifyArithPass rewrites atan2(y, 1.0) to vectorized math.atan.
          return IntrinsicCycles<ci::Atan>(
                     ty, {/*f32_vec=*/17.0, /*f64_vec=*/27.0, /*f32_libm=*/40.0,
                          /*f64_libm=*/70.0}) +
                 SubF32ConvertOverhead(ty);
        }
        // Lowered to unvectorized BuiltinAtan2 / BuiltinAtan2f.
        return fty == F64 ? 70.0 : 40.0;

      case HloOpcode::kSqrt:
        if (is_complex) {
          return 20.0 / static_cast<double>(VectorLanes(fty));
        }
        return (fty == F64 ? 8.0 : 5.0) /
                   static_cast<double>(VectorLanes(fty)) +
               SubF32ConvertOverhead(ty);

      case HloOpcode::kRsqrt: {
        if (is_complex) {
          return 30.0 / static_cast<double>(VectorLanes(fty));
        }
        const int64_t lanes = VectorLanes(fty);
        if (fty == F64) {
          const int64_t width =
              MaxIntrinsicVectorWidth<ci::Rsqrt>(enabled_features_, F64, lanes);
          // Without AVX512 rsqrt14pd, lowers to vectorized 1.0 / sqrt(x) on
          // the non pipelined FP64 divide/sqrt unit.
          return (width >= 2 ? 12.0 : 36.0) / static_cast<double>(lanes);
        }
        return 7.5 / static_cast<double>(lanes) + SubF32ConvertOverhead(ty);
      }

      // Transcendentals with no vectorized intrinsic in XLA:CPU; lowered to
      // scalar libm calls.
      case HloOpcode::kAcosh:
      case HloOpcode::kAsinh:
      case HloOpcode::kAtanh:
      case HloOpcode::kCbrt:
      case HloOpcode::kExp2:
      case HloOpcode::kLog2:
      case HloOpcode::kTan:
        if (is_complex) {
          return fty == F64 ? 120.0 : 75.0;
        }
        return (fty == F64 ? 65.0 : 30.0) + SubF32ConvertOverhead(ty);

      case HloOpcode::kPower:
        if (primitive_util::IsIntegralType(ty)) {
          // ExpandIntegerPowerPass emits a vectorizable exponentiation loop.
          return 10.0 / static_cast<double>(VectorLanes(ty));
        }
        if (is_complex) {
          return fty == F64 ? 120.0 : 75.0;
        }
        return (fty == F64 ? 65.0 : 30.0) + SubF32ConvertOverhead(ty);

      case HloOpcode::kDivide:
        if (is_complex) {
          return 24.0 / static_cast<double>(VectorLanes(fty));
        }
        if (primitive_util::IsFloatingPointType(ty)) {
          return (fty == F64 ? 14.0 : 6.0) /
                     static_cast<double>(VectorLanes(fty)) +
                 SubF32ConvertOverhead(ty);
        }
        if (IsEffectiveScalarConstant(op.operand(1), enclosing_fusion)) {
          return 2.5 / static_cast<double>(VectorLanes(ty));
        }
        // Variable integer division has no SIMD instruction on x86 or ARM.
        return primitive_util::BitWidth(ty) >= 64 ? 18.0 : 12.0;

      case HloOpcode::kRemainder:
        if (primitive_util::IsFloatingPointType(ty)) {
          // Lowered to scalar libm fmod / fmodf.
          return (fty == F64 ? 55.0 : 30.0) + SubF32ConvertOverhead(ty);
        }
        if (IsEffectiveScalarConstant(op.operand(1), enclosing_fusion)) {
          return 3.5 / static_cast<double>(VectorLanes(ty));
        }
        return primitive_util::BitWidth(ty) >= 64 ? 18.0 : 12.0;

      case HloOpcode::kConvert: {
        const PrimitiveType src_ty = op.operand(0)->shape().element_type();
        if (src_ty == ty) {
          return 0.0;
        }
        if (ty == BF16) {
          const int64_t lanes = VectorLanes(src_ty == F64 ? F64 : F32);
          return ConvertF32ToBf16VecCycles() / static_cast<double>(lanes);
        }
        if (src_ty == BF16) {
          const double ext =
              1.2 / static_cast<double>(VectorLanes(ty == F64 ? F64 : F32));
          return ty == F16 ? ext + 2.5 / static_cast<double>(VectorLanes(F32))
                           : ext;
        }
        if (src_ty == F16 || ty == F16) {
          const int64_t lanes =
              VectorLanes((src_ty == F64 || ty == F64) ? F64 : F32);
          return 2.5 / static_cast<double>(lanes);
        }
        const int64_t lanes = std::min(VectorLanes(src_ty), VectorLanes(ty));
        return 1.0 / static_cast<double>(lanes);
      }

      case HloOpcode::kCompare: {
        const PrimitiveType src_ty = op.operand(0)->shape().element_type();
        if (primitive_util::IsComplexType(src_ty)) {
          return 1.0 / static_cast<double>(VectorLanes(src_ty));
        }
        if (src_ty == BF16 || src_ty == F16) {
          return 2.5 / static_cast<double>(VectorLanes(F32));
        }
        return 0.5 / static_cast<double>(VectorLanes(src_ty));
      }

      case HloOpcode::kSelect:
        if (is_complex) {
          return 1.5 / static_cast<double>(VectorLanes(fty));
        }
        if (primitive_util::IsFloatingPointType(ty) &&
            primitive_util::BitWidth(ty) < 32) {
          return 0.75 / static_cast<double>(VectorLanes(F32));
        }
        return 0.75 / static_cast<double>(VectorLanes(ty));

      case HloOpcode::kMultiply:
        if (is_complex) {
          return 3.0 / static_cast<double>(VectorLanes(fty));
        }
        if (ty == BF16 || ty == F16) {
          return 0.5 / static_cast<double>(VectorLanes(F32)) +
                 SubF32ConvertOverhead(ty);
        }
        return 0.5 / static_cast<double>(VectorLanes(ty));

      default:
        if (is_complex) {
          return 1.0 / static_cast<double>(VectorLanes(fty));
        }
        if (ty == BF16 || ty == F16) {
          return 0.5 / static_cast<double>(VectorLanes(F32)) +
                 SubF32ConvertOverhead(ty);
        }
        return 0.5 / static_cast<double>(VectorLanes(ty));
    }
  }

  static double StridedTransposeCyclesPerElement(
      const StridedTransposeInfo& info, bool is_tiled) {
    // Non contiguous scalar loads plus L1D set aliasing on 1 KiB multiples, and
    // per core DTLB page walk stalls when an untiled row sweeps past L1
    // hugepage DTLB coverage.
    double cycles = 0.45;
    const int64_t stride_int = static_cast<int64_t>(info.stride_bytes);
    if (stride_int >= 1024 && (stride_int % 1024) == 0) {
      cycles += 0.35;
    }
    if (!is_tiled && info.stride_bytes >= kDtlbMissStrideBytes &&
        info.sweep_bytes >= kDtlbSweepThresholdBytes) {
      cycles += kTransposeDtlbMissCycles;
    }
    return cycles;
  }

  // Estimates scalar loop carried latency for a single invocation of a reducer
  // computation emitted by ElementalHloToMlir::EmitReduce / EmitReduceWindow.
  double ScalarReducerCyclesPerStep(const HloComputation& comp) const {
    double cycles = 0.0;
    for (const HloInstruction* instr : comp.instructions()) {
      const HloOpcode opcode = instr->opcode();
      if (opcode == HloOpcode::kParameter || opcode == HloOpcode::kConstant ||
          opcode == HloOpcode::kTuple ||
          opcode == HloOpcode::kGetTupleElement ||
          opcode == HloOpcode::kBitcast ||
          opcode == HloOpcode::kBitcastConvert) {
        continue;
      }
      const PrimitiveType ty = instr->shape().element_type();
      switch (opcode) {
        case HloOpcode::kConvert: {
          const PrimitiveType src_ty =
              instr->operand(0)->shape().element_type();
          if (src_ty == ty) {
            break;
          }
          if (ty == BF16 && !has_avx512bf16_) {
            cycles += 6.0;
          } else if (src_ty == BF16 || src_ty == F16 || ty == F16) {
            cycles += 1.5;
          } else {
            cycles += 1.0;
          }
          break;
        }
        case HloOpcode::kCompare: {
          const PrimitiveType src_ty =
              instr->operand(0)->shape().element_type();
          cycles += primitive_util::IsFloatingPointType(src_ty) ? 1.5 : 1.0;
          break;
        }
        case HloOpcode::kSelect:
          cycles += 1.5;
          break;
        case HloOpcode::kDivide:
        case HloOpcode::kRemainder:
        case HloOpcode::kSqrt:
        case HloOpcode::kRsqrt:
          cycles += 14.0;
          break;
        case HloOpcode::kAdd:
        case HloOpcode::kSubtract:
        case HloOpcode::kMultiply:
        case HloOpcode::kMinimum:
        case HloOpcode::kMaximum:
        case HloOpcode::kAbs:
        case HloOpcode::kNegate:
        case HloOpcode::kClamp:
          if (ty == BF16 || ty == F16) {
            // Scalar sub F32 float op expands to extend + F32 ALU + round to
            // nearest even truncate on every loop carried step.
            cycles += has_avx512bf16_ && ty == BF16 ? 5.0 : 12.0;
          } else if (primitive_util::IsFloatingPointType(ty) ||
                     primitive_util::IsComplexType(ty)) {
            cycles += 3.0;
          } else {
            cycles += opcode == HloOpcode::kMultiply ? 3.0 : 1.0;
          }
          break;
        default:
          if (instr->IsElementwise() &&
              (primitive_util::IsFloatingPointType(ty) ||
               primitive_util::IsComplexType(ty))) {
            cycles += std::max(
                3.0,
                ElementwiseOpCyclesPerElement(*instr, nullptr) *
                    static_cast<double>(VectorLanes(FloatComputeType(ty))));
          } else {
            cycles += 1.0;
          }
          break;
      }
    }
    return std::max(1.0, cycles);
  }

  double ReducerCyclesPerStep(const HloComputation& comp,
                              PrimitiveType element_type, bool single_output,
                              bool reduces_minor_dim,
                              bool contiguous_1d) const {
    if (single_output) {
      // When the contiguous minor dimension is preserved across outputs, LLVM
      // can vectorize F32/F64 and integer reductions across independent output
      // lanes without reassociating the accumulator.
      if (!reduces_minor_dim && element_type != BF16 && element_type != F16 &&
          !primitive_util::IsComplexType(element_type)) {
        return SubcomputationCyclesPerInvocation(comp);
      }
      if (contiguous_1d) {
        // LLVM auto vectorizes single loop associative integer reductions
        // across the contiguous minor dimension.
        if (IsAssociativeIntegerReducer(comp)) {
          return SubcomputationCyclesPerInvocation(comp);
        }
        // Contiguous 1D reductions stream linearly from cache and should not
        // over split a narrow output across every thread.
        if (element_type == BF16 || element_type == F16 ||
            element_type == F32) {
          return std::min(0.6, ScalarReducerCyclesPerStep(comp));
        }
      }
    }
    return ScalarReducerCyclesPerStep(comp);
  }

  double SubcomputationCyclesPerInvocation(const HloComputation& comp) const {
    double cycles = 0.0;
    for (const HloInstruction* instr : comp.instructions()) {
      cycles += InstructionComputeCycles(*instr, /*enclosing_fusion=*/nullptr);
    }
    return cycles;
  }

  double InstructionComputeCycles(
      const HloInstruction& op,
      const HloFusionInstruction* enclosing_fusion) const {
    if (op.IsElementwise()) {
      const double elems = SafeArrayElements(op.shape());
      if (elems == 0.0) {
        return 0.0;
      }
      return elems * ElementwiseOpCyclesPerElement(op, enclosing_fusion);
    }

    switch (op.opcode()) {
      case HloOpcode::kIota: {
        const double elems = SafeArrayElements(op.shape());
        return elems * (0.5 / static_cast<double>(
                                  VectorLanes(op.shape().element_type())));
      }
      case HloOpcode::kTranspose: {
        if (IsDirectMemoryStridedTranspose(op, enclosing_fusion)) {
          if (std::optional<StridedTransposeInfo> info =
                  GetStridedTransposeInfo(op);
              info.has_value()) {
            const bool is_tiled = enclosing_fusion != nullptr &&
                                  IsTiledFusionSupported(*enclosing_fusion,
                                                         enable_tiled_emitter_);
            return SafeArrayElements(op.shape()) *
                   StridedTransposeCyclesPerElement(*info, is_tiled);
          }
        }
        return 0.0;
      }
      case HloOpcode::kConcatenate: {
        const int64_t num_operands = op.operand_count();
        if (num_operands <= 1) {
          return 0.0;
        }
        if (CanEmitFastConcatenateCopy(op, enclosing_fusion) ||
            IsConcatenateFusionHero(op, enclosing_fusion)) {
          // ConcatenateFusionKernelEmitter emits either one CopySliceOp per
          // operand per outer index (fast copy) or one loop per operand without
          // per element binary search branches.
          const double slice_copies =
              ConcatenateOuterElements(op) * static_cast<double>(num_operands);
          return slice_copies;
        }
        // Non hero concatenates lower through ElementalHloToMlir::EmitConcat,
        // emitting a per element binary search tree of scf.if branches.
        const double tree_depth =
            std::ceil(std::log2(static_cast<double>(num_operands)));
        return SafeArrayElements(op.shape()) * 1.5 * tree_depth;
      }
      case HloOpcode::kPad: {
        const double elems = SafeArrayElements(op.shape());
        if (elems == 0.0) {
          return 0.0;
        }
        int64_t padded_dims = 0;
        for (const auto& dim : op.padding_config().dimensions()) {
          if (dim.edge_padding_low() != 0 || dim.edge_padding_high() != 0 ||
              dim.interior_padding() != 0) {
            ++padded_dims;
          }
        }
        if (padded_dims == 0) {
          return 0.0;
        }
        // ElementalHloToMlir::EmitPad evaluates per element range constraints
        // and branches between the source load and padding value.
        return elems * (0.6 + 0.2 * static_cast<double>(padded_dims));
      }
      case HloOpcode::kGather: {
        const double out_elems = SafeArrayElements(op.shape());
        if (out_elems == 0.0) {
          return 0.0;
        }
        const double elem_bytes = ElementStorageBytes(op.shape());
        const double slice_elems =
            std::max(1.0, GatherContiguousSliceBytes(op) / elem_bytes);
        const double num_slices = out_elems / slice_elems;
        // Bounds clamping and offset calculation happen once per contiguous
        // slice, while elements inside a slice copy at SIMD width.
        return num_slices * 2.0 +
               out_elems * (0.5 / static_cast<double>(
                                      VectorLanes(op.shape().element_type())));
      }

      case HloOpcode::kMap:
        return SafeArrayElements(op.shape()) *
               SubcomputationCyclesPerInvocation(*op.to_apply());

      case HloOpcode::kReduce: {
        const Shape& input_shape = op.operand(0)->shape();
        const Shape& out_shape =
            op.shape().IsArray() ? op.shape() : op.shape().tuple_shapes(0);
        const double reductions = std::max(
            0.0, SafeArrayElements(input_shape) - SafeArrayElements(out_shape));
        if (reductions == 0.0) {
          return 0.0;
        }
        const int64_t minor_dim = EffectiveMinorDimension(input_shape);
        const bool reduces_minor =
            minor_dim >= 0 && absl::c_linear_search(op.dimensions(), minor_dim);
        const bool contiguous_1d =
            ReducesOnlyContiguousMinorDimension(input_shape, op.dimensions());
        return reductions * ReducerCyclesPerStep(*op.to_apply(),
                                                 out_shape.element_type(),
                                                 op.shape().IsArray(),
                                                 reduces_minor, contiguous_1d);
      }

      case HloOpcode::kReduceWindow: {
        const Shape& input_shape = op.operand(0)->shape();
        const Shape& out_shape =
            op.shape().IsArray() ? op.shape() : op.shape().tuple_shapes(0);
        double window_elems = 1.0;
        for (const WindowDimension& dim : op.window().dimensions()) {
          window_elems *= static_cast<double>(std::max<int64_t>(1, dim.size()));
        }
        const double reductions =
            std::max(0.0, window_elems - 1.0) * SafeArrayElements(out_shape);
        if (reductions == 0.0) {
          return 0.0;
        }
        const int64_t minor_dim = EffectiveMinorDimension(input_shape);
        const bool reduces_minor = minor_dim >= 0 &&
                                   minor_dim < op.window().dimensions_size() &&
                                   op.window().dimensions(minor_dim).size() > 1;
        const bool contiguous_1d =
            WindowReducesOnlyContiguousMinorDimension(input_shape, op.window());
        return reductions * ReducerCyclesPerStep(*op.to_apply(),
                                                 out_shape.element_type(),
                                                 op.shape().IsArray(),
                                                 reduces_minor, contiguous_1d);
      }

      case HloOpcode::kDot: {
        const double flops = static_cast<double>(HloCostAnalysis::GetDotFlops(
            op.operand(0)->shape(), op.shape(), op.dot_dimension_numbers()));
        const double fmas = 0.5 * std::max(0.0, flops);
        if (enclosing_fusion != nullptr && IsTiledGemvDot(op) &&
            IsTiledFusionSupported(*enclosing_fusion, enable_tiled_emitter_)) {
          // EmitTiledFusionKernel lowers supported GEMV dot fusions to
          // vectorized tiles rather than a scalar contracting loop.
          const PrimitiveType ty = op.shape().element_type();
          const double lanes = static_cast<double>(VectorLanes(ty));
          return fmas * (1.0 / lanes + SubF32ConvertOverhead(ty));
        }
        // Untiled loop fused elemental dots emit a scalar contracting loop.
        return fmas * 1.5;
      }

      case HloOpcode::kConvolution: {
        const double flops =
            static_cast<double>(HloCostAnalysis::GetConvolutionFlops(
                &op, op.operand(0)->shape(), op.operand(1)->shape(),
                op.shape()));
        const double fmas = 0.5 * std::max(0.0, flops);
        return fmas /
               static_cast<double>(VectorLanes(op.shape().element_type()));
      }

      default:
        return 0.0;
    }
  }

  void AccumulateGatherMemoryCost(const HloInstruction& gather,
                                  double utilization, double& gather_read_bytes,
                                  double& random_miss_ns) const {
    if (utilization <= 0.0) {
      return;
    }
    const Shape& table_shape = gather.operand(0)->shape();
    const double out_bytes = ArrayBytes(gather.shape());
    const double table_bytes = ArrayBytes(table_shape);
    const double slice_bytes = GatherContiguousSliceBytes(gather);
    if (out_bytes <= 0.0 || table_bytes <= 0.0 || slice_bytes <= 0.0) {
      return;
    }
    // Batched gathers index within a per batch sub table rather than across the
    // entire operand tensor.
    double batch_product = 1.0;
    for (int64_t dim :
         gather.gather_dimension_numbers().operand_batching_dims()) {
      batch_product *= static_cast<double>(
          std::max<int64_t>(1, table_shape.dimensions(dim)));
    }
    const double indexed_table_bytes = table_bytes / batch_product;

    const double num_fetches = utilization * (out_bytes / slice_bytes);
    const double bytes_per_fetch =
        std::min(indexed_table_bytes, std::max(slice_bytes, kCacheLineBytes));
    double fetched_bytes = num_fetches * bytes_per_fetch;
    if (indexed_table_bytes <= kL2TierBytes) {
      // Sub tables that fit in private L1/L2 stay cache resident once loaded.
      fetched_bytes = std::min(fetched_bytes, table_bytes);
    }
    gather_read_bytes += fetched_bytes;

    // Random miss latency applies once per independent index lookup when the
    // indexed sub table exceeds private L2 cache.
    double slice_total_elems = 1.0;
    for (int64_t slice_dim : gather.gather_slice_sizes()) {
      slice_total_elems *= static_cast<double>(std::max<int64_t>(1, slice_dim));
    }
    const double num_lookups =
        utilization * (SafeArrayElements(gather.shape()) / slice_total_elems);
    if (indexed_table_bytes > kL3TotalBytes) {
      random_miss_ns += num_lookups * kRandomDramFetchNs;
    } else if (indexed_table_bytes > kL2TierBytes) {
      random_miss_ns += num_lookups * kRandomL3FetchNs;
    }
  }

  static double SliceOperandUtilization(const HloInstruction& slice,
                                        double util) {
    const Shape& in_shape = slice.operand(0)->shape();
    const Shape& out_shape = slice.shape();
    const double in_elems = SafeArrayElements(in_shape);
    const double out_elems = SafeArrayElements(out_shape);
    if (in_elems <= 0.0 || out_elems <= 0.0) {
      return 0.0;
    }
    const double elem_bytes = ElementStorageBytes(in_shape);
    double slice_contiguous_elems = 1.0;
    double row_stride_elems = 1.0;
    bool found_partial = false;
    ForEachDimensionMinorToMajor(in_shape, [&](int64_t dim) {
      const int64_t in_dim = std::max<int64_t>(1, in_shape.dimensions(dim));
      const int64_t out_dim = std::max<int64_t>(1, out_shape.dimensions(dim));
      const int64_t step = slice.opcode() == HloOpcode::kSlice
                               ? std::max<int64_t>(1, slice.slice_strides(dim))
                               : 1;
      if (out_dim < in_dim || step > 1) {
        slice_contiguous_elems *=
            step == 1 ? static_cast<double>(out_dim) : 1.0;
        row_stride_elems *= static_cast<double>(in_dim);
        found_partial = true;
        return false;
      }
      slice_contiguous_elems *= static_cast<double>(out_dim);
      row_stride_elems *= static_cast<double>(in_dim);
      return true;
    });
    const double num_slices = out_elems / std::max(1.0, slice_contiguous_elems);
    if (!found_partial || num_slices <= 1.0) {
      return util * (out_elems / in_elems);
    }
    const double slice_bytes = slice_contiguous_elems * elem_bytes;
    const double row_stride_bytes = row_stride_elems * elem_bytes;
    // Non contiguous slices fetch at least one cache line per row, and narrow
    // strides (<= 4 cache lines) trigger hardware adjacent line prefetching.
    double effective_slice_bytes = std::max(slice_bytes, kCacheLineBytes);
    if (row_stride_bytes <= 4.0 * kCacheLineBytes) {
      effective_slice_bytes =
          std::max(effective_slice_bytes,
                   std::min(row_stride_bytes, 2.0 * kCacheLineBytes));
    }
    effective_slice_bytes = std::min(effective_slice_bytes, row_stride_bytes);
    const double effective_read_bytes = num_slices * effective_slice_bytes;
    const double in_bytes = in_elems * elem_bytes;
    return util * std::min(1.0, effective_read_bytes / in_bytes);
  }

  ExecutionCostNs EstimateFusionCost(const HloFusionInstruction& fusion) const {
    const HloComputation* comp = fusion.fused_instructions_computation();
    std::vector<HloInstruction*> instructions =
        comp->MakeInstructionPostOrder();
    absl::c_reverse(instructions);

    const size_t num_instrs = instructions.size();
    absl::flat_hash_map<const HloInstruction*, size_t> instr_index;
    instr_index.reserve(num_instrs);
    for (size_t i = 0; i < num_instrs; ++i) {
      instr_index.emplace(instructions[i], i);
    }

    // Propagate utilization backward from root so dead branches and sliced or
    // gathered parameters are weighed by what the kernel actually evaluates,
    // while shared elementwise DAG nodes are counted once per root. Roots are
    // stored as sorted reverse post order indices so summation is deterministic
    // without copying hash sets.
    std::vector<RootIndexList> elementwise_roots(num_instrs);
    std::vector<double> root_utilization(num_instrs, 0.0);
    std::vector<double> utilization(num_instrs, 0.0);
    // Tracks parameter/constant leaves read only through gather so their
    // cache line aligned gather traffic replaces full table size.
    std::vector<double> direct_gather_bytes(num_instrs, 0.0);

    const size_t root_slot = instr_index.at(fusion.fused_expression_root());
    elementwise_roots[root_slot].push_back(root_slot);
    root_utilization[root_slot] = 1.0;

    double total_compute_cycles = 0.0;
    double extra_read_bytes = 0.0;
    double random_miss_ns = 0.0;

    for (size_t i = 0; i < num_instrs; ++i) {
      const HloInstruction* instr = instructions[i];
      double util = 0.0;
      for (size_t use_root : elementwise_roots[i]) {
        util += root_utilization[use_root];
      }
      utilization[i] = util;
      if (util <= 0.0) {
        continue;
      }

      total_compute_cycles += util * InstructionComputeCycles(*instr, &fusion);

      if (std::optional<StridedTransposeInfo> info =
              GetStridedTransposeInfo(*instr);
          info.has_value() && info->stride_bytes >= kCacheLineBytes &&
          IsDirectMemoryStridedTranspose(*instr, &fusion)) {
        extra_read_bytes += std::min(1.0, util) * ArrayBytes(instr->shape());
      }

      const HloOpcode opcode = instr->opcode();
      if (PreservesElementwiseIndexing(*instr)) {
        const RootIndexList& roots = elementwise_roots[i];
        for (const HloInstruction* operand : instr->operands()) {
          const size_t operand_slot = instr_index.at(operand);
          if (operand_slot != i) {
            MergeSortedUnique(elementwise_roots[operand_slot], roots);
          }
        }
        continue;
      }

      for (int64_t idx = 0; idx < instr->operand_count(); ++idx) {
        const HloInstruction* operand = instr->operand(idx);
        double op_util = util;

        switch (opcode) {
          case HloOpcode::kSlice:
          case HloOpcode::kDynamicSlice:
            if (idx == 0) {
              op_util = SliceOperandUtilization(*instr, util);
            } else {
              op_util = std::min(1.0, util);
            }
            break;

          case HloOpcode::kDynamicUpdateSlice:
            if (idx == 0) {
              const double out_elems = SafeArrayElements(instr->shape());
              const double upd_elems =
                  SafeArrayElements(instr->operand(1)->shape());
              op_util =
                  out_elems > 0.0
                      ? util * std::max(0.0, out_elems - upd_elems) / out_elems
                      : 0.0;
            } else if (idx >= 2) {
              op_util = std::min(1.0, util);
            }
            break;

          case HloOpcode::kGather:
            if (idx == 0) {
              if (const HloInstruction* mem_src =
                      DirectMemoryParameterOrConstant(operand);
                  mem_src != nullptr) {
                AccumulateGatherMemoryCost(
                    *instr, util, direct_gather_bytes[instr_index.at(mem_src)],
                    random_miss_ns);
                op_util = 0.0;
              } else {
                const double in_elems = SafeArrayElements(operand->shape());
                op_util =
                    in_elems > 0.0
                        ? util * SafeArrayElements(instr->shape()) / in_elems
                        : 0.0;
              }
            }
            break;

          case HloOpcode::kBroadcast: {
            const double in_elems = SafeArrayElements(operand->shape());
            if (in_elems <= 1.0) {
              op_util = std::min(1.0, util);
            } else {
              const double out_elems = SafeArrayElements(instr->shape());
              const double invariant = BroadcastInvariantInnerElements(*instr);
              op_util = util * (out_elems / invariant) / in_elems;
            }
            break;
          }

          case HloOpcode::kReduce:
            if (idx >= instr->operand_count() / 2) {
              op_util = std::min(1.0, util);
            }
            break;

          case HloOpcode::kReduceWindow:
            if (idx < instr->operand_count() / 2) {
              const double in_elems = SafeArrayElements(operand->shape());
              const Shape& out_shape = instr->shape().IsArray()
                                           ? instr->shape()
                                           : instr->shape().tuple_shapes(0);
              double window_elems = 1.0;
              for (const WindowDimension& dim : instr->window().dimensions()) {
                window_elems *=
                    static_cast<double>(std::max<int64_t>(1, dim.size()));
              }
              op_util = in_elems > 0.0 ? util * SafeArrayElements(out_shape) *
                                             window_elems / in_elems
                                       : 0.0;
            } else {
              op_util = std::min(1.0, util);
            }
            break;

          case HloOpcode::kPad:
            if (idx == 1) {
              op_util = std::min(1.0, util);
            }
            break;

          case HloOpcode::kScatter: {
            const auto* scatter = Cast<HloScatterInstruction>(instr);
            // Destination buffers are modified in place only at scattered
            // coordinates and are modeled directly by
            // ChooseScatterFusionTaskCount; propagate utilization only into
            // fused index and update producers.
            if (idx < scatter->scatter_operand_count()) {
              op_util = 0.0;
            }
            break;
          }

          default:
            break;
        }

        if (op_util > 0.0) {
          const size_t operand_slot = instr_index.at(operand);
          InsertSortedUnique(elementwise_roots[operand_slot], operand_slot);
          root_utilization[operand_slot] += op_util;
        }
      }
    }

    auto source_read_bytes = [&](size_t slot) -> double {
      const double full_bytes = ArrayBytes(instructions[slot]->shape());
      const double direct_util = std::min(1.0, utilization[slot]);
      // If the source is also scanned sequentially, its cache lines are already
      // counted up to full_bytes; otherwise count the gather fetches.
      return std::max(direct_util * full_bytes, direct_gather_bytes[slot]);
    };

    const double write_bytes =
        fusion.fused_expression_root()->opcode() == HloOpcode::kScatter
            ? 0.0
            : RecursiveArrayBytes(fusion.shape());
    double read_bytes = extra_read_bytes;

    for (size_t i = 0; i < num_instrs; ++i) {
      const HloInstruction* instr = instructions[i];
      if (instr->opcode() == HloOpcode::kParameter) {
        if (instr->shape().IsArray()) {
          read_bytes += source_read_bytes(i);
        }
      } else if (instr->opcode() == HloOpcode::kGetTupleElement &&
                 instr->shape().IsArray() &&
                 instr->LatestNonGteAncestor()->opcode() ==
                     HloOpcode::kParameter) {
        read_bytes += source_read_bytes(i);
      } else if (instr->opcode() == HloOpcode::kConstant &&
                 instr->shape().IsArray() &&
                 SafeArrayElements(instr->shape()) > 1.0) {
        read_bytes += source_read_bytes(i);
      }
    }

    return ExecutionCostNs{
        /*core_ns=*/total_compute_cycles / kCyclesPerNs + random_miss_ns,
        /*memory_ns=*/SequentialMemoryTimeNs(read_bytes, write_bytes),
    };
  }

  ExecutionCostNs EstimateStandaloneInstructionCost(
      const HloInstruction& instr) const {
    const double compute_ns =
        InstructionComputeCycles(instr, /*enclosing_fusion=*/nullptr) /
        kCyclesPerNs;

    const double write_bytes = RecursiveArrayBytes(instr.shape());
    double read_bytes = 0.0;
    double random_miss_ns = 0.0;

    switch (instr.opcode()) {
      case HloOpcode::kSlice:
        read_bytes += SliceOperandUtilization(instr, /*util=*/1.0) *
                      ArrayBytes(instr.operand(0)->shape());
        break;
      case HloOpcode::kDynamicSlice:
        read_bytes += SliceOperandUtilization(instr, /*util=*/1.0) *
                      ArrayBytes(instr.operand(0)->shape());
        for (int64_t i = 1; i < instr.operand_count(); ++i) {
          read_bytes += ArrayBytes(instr.operand(i)->shape());
        }
        break;
      case HloOpcode::kGather:
        AccumulateGatherMemoryCost(instr, /*utilization=*/1.0, read_bytes,
                                   random_miss_ns);
        read_bytes += ArrayBytes(instr.operand(1)->shape());
        break;
      case HloOpcode::kTranspose:
      case HloOpcode::kCopy:
        read_bytes += ArrayBytes(instr.operand(0)->shape());
        if (std::optional<StridedTransposeInfo> info =
                GetStridedTransposeInfo(instr);
            info.has_value() && info->stride_bytes >= kCacheLineBytes) {
          read_bytes += ArrayBytes(instr.shape());
        }
        break;
      default:
        for (const HloInstruction* operand : instr.operands()) {
          read_bytes += RecursiveArrayBytes(operand->shape());
        }
        break;
    }

    return ExecutionCostNs{
        /*core_ns=*/compute_ns + random_miss_ns,
        /*memory_ns=*/SequentialMemoryTimeNs(read_bytes, write_bytes),
    };
  }

  int64_t ChooseTaskCount(const ExecutionCostNs& cost,
                          const HloInstruction& instruction) const {
    const double single_thread_ns = std::max(cost.core_ns, cost.memory_ns);
    if (!std::isfinite(single_thread_ns) ||
        single_thread_ns < 2.0 * kMinWorkPerTaskNs) {
      return 1;
    }

    const Shape* indexing_shape = IndexingShapeForInstruction(instruction);
    if (indexing_shape == nullptr) {
      return 1;
    }
    const std::optional<int64_t> fast_concat_outer =
        FastConcatenateOuterDimSize(instruction);
    if (fast_concat_outer.has_value() && *fast_concat_outer < 2) {
      return 1;
    }

    int64_t max_tasks = std::min<int64_t>(
        max_parallelism_, static_cast<int64_t>(std::min<double>(
                              static_cast<double>(max_parallelism_),
                              single_thread_ns / kMinWorkPerTaskNs)));
    if (fast_concat_outer.has_value()) {
      max_tasks = std::min(max_tasks, *fast_concat_outer);
    }
    if (max_tasks < 2) {
      return 1;
    }

    // Minimize the geometric mean of predicted wall time and predicted CPU time
    // (sqrt(wall * (tasks * wall)) = wall * sqrt(tasks)), so doubling task
    // count requires at least a sqrt(2) (~1.41x) wall speedup.
    double best_objective = single_thread_ns;
    int64_t best_tasks = 1;
    absl::flat_hash_set<int64_t> evaluated_workgroups;
    evaluated_workgroups.reserve(static_cast<size_t>(max_tasks));

    for (int64_t tasks = 2; tasks <= max_tasks; ++tasks) {
      const int64_t partition_dim =
          fast_concat_outer.has_value()
              ? *fast_concat_outer
              : FoldedPartitionDimension(*indexing_shape, tasks);
      if (partition_dim < 2) {
        continue;
      }
      const int64_t split_tile = CeilOfRatio(partition_dim, tasks);
      const int64_t actual_wg = CeilOfRatio(partition_dim, split_tile);
      if (actual_wg < 2 || !evaluated_workgroups.insert(actual_wg).second) {
        continue;
      }

      const double work_frac =
          static_cast<double>(split_tile) / static_cast<double>(partition_dim);
      const double imbalance = work_frac * static_cast<double>(actual_wg);
      const bool has_in_bounds_guard =
          !fast_concat_outer.has_value() &&
          (split_tile * actual_wg != partition_dim);
      const double guard_factor =
          has_in_bounds_guard ? kInBoundsGuardFactor : 1.0;

      const double task_core_ns = cost.core_ns * work_frac * guard_factor;
      const double mem_parallelism =
          static_cast<double>(std::min(actual_wg, max_memory_tasks_));
      const double task_memory_ns =
          (cost.memory_ns / mem_parallelism) * imbalance * guard_factor;
      const double total_ns =
          kDispatchBaseNs +
          kDispatchPerTaskNs * static_cast<double>(actual_wg) +
          std::max(task_core_ns, task_memory_ns);
      const double objective =
          total_ns * std::sqrt(static_cast<double>(actual_wg));
      if (objective < best_objective) {
        best_objective = objective;
        best_tasks = actual_wg;
      }
    }

    return best_tasks;
  }

  // Computes the contiguous inner element count of one simplified scatter
  // window slice in `operand_shape`'s physical layout, together with the
  // minimum physical byte stride among scattered dimensions.
  static void ComputeScatterSliceGeometry(
      const Shape& operand_shape, absl::Span<const int64_t> slice_shape,
      absl::Span<const int64_t> scatter_dims_to_operand_dims,
      double element_bytes, int64_t& contiguous_slice_elems,
      double& min_scatter_stride_bytes) {
    const int64_t rank = operand_shape.dimensions().size();
    contiguous_slice_elems = 1;
    min_scatter_stride_bytes = std::numeric_limits<double>::infinity();
    if (rank == 0) {
      min_scatter_stride_bytes = element_bytes;
      return;
    }

    absl::InlinedVector<int64_t, 6> minor_to_major;
    if (LayoutUtil::HasLayout(operand_shape) &&
        operand_shape.layout().minor_to_major().size() == rank) {
      minor_to_major.assign(operand_shape.layout().minor_to_major().begin(),
                            operand_shape.layout().minor_to_major().end());
    } else {
      minor_to_major.reserve(rank);
      for (int64_t d = rank - 1; d >= 0; --d) {
        minor_to_major.push_back(d);
      }
    }

    absl::InlinedVector<double, 6> dim_stride_bytes(rank, element_bytes);
    double running_stride_bytes = element_bytes;
    bool still_contiguous = true;
    for (int64_t dim : minor_to_major) {
      dim_stride_bytes[dim] = running_stride_bytes;
      const int64_t operand_dim = operand_shape.dimensions(dim);
      const int64_t slice_dim = slice_shape[dim];
      if (still_contiguous && slice_dim > 0) {
        contiguous_slice_elems *= slice_dim;
        if (slice_dim < operand_dim) {
          still_contiguous = false;
        }
      }
      running_stride_bytes *=
          static_cast<double>(std::max<int64_t>(1, operand_dim));
    }

    for (int64_t scatter_dim : scatter_dims_to_operand_dims) {
      if (scatter_dim >= 0 && scatter_dim < rank) {
        min_scatter_stride_bytes =
            std::min(min_scatter_stride_bytes, dim_stride_bytes[scatter_dim]);
      }
    }
    if (!std::isfinite(min_scatter_stride_bytes)) {
      min_scatter_stride_bytes = element_bytes;
    }
  }

  // Upper bound on distinct non negative scalar index values representable by
  // `index_type`, used to catch narrow index types (such as S8 or U8)
  // scattering into a much wider destination buffer.
  static double MaxNonNegativeIndexCount(PrimitiveType index_type) {
    if (!primitive_util::IsIntegralType(index_type)) {
      return std::numeric_limits<double>::infinity();
    }
    const int64_t bits = primitive_util::BitWidth(index_type);
    const int64_t value_bits =
        primitive_util::IsSignedIntegralType(index_type) ? bits - 1 : bits;
    if (value_bits >= 53) {
      return std::numeric_limits<double>::infinity();
    }
    return std::ldexp(1.0, static_cast<int>(std::max<int64_t>(0, value_bits)));
  }

  // Chooses the workgroup count in [1, max_scatter_tasks] for a CPU scatter
  // fusion by comparing non atomic single threaded execution against multi
  // workgroup execution (including fused index and update producer work, atomic
  // store / atomicrmw / cmpxchg step up costs, cache tiered destination traffic
  // and random access latency, discrete num_updates load balance, and cross
  // core cache line contention on small destination footprints).
  int64_t ChooseScatterFusionTaskCount(
      const HloFusionInstruction& fusion) const {
    const int64_t max_scatter_tasks = GetCpuScatterEffectiveWorkGroups(
        fusion, max_parallelism_, cpu_device_spec_,
        /*buffer_assignment=*/nullptr, MinAlign());
    if (max_scatter_tasks <= 1) {
      return 1;
    }

    const auto& scatter =
        *Cast<HloScatterInstruction>(fusion.fused_expression_root());
    const Shape& first_update_shape =
        scatter.scatter_updates().front()->shape();
    if (first_update_shape.dimensions().empty() ||
        ShapeUtil::IsZeroElementArray(first_update_shape)) {
      return 1;
    }
    const int64_t num_updates = first_update_shape.dimensions(0);
    if (num_updates <= 1) {
      return 1;
    }
    const absl::Span<const int64_t> slice_shape =
        first_update_shape.dimensions().subspan(1);
    const int64_t slice_elems = Product(slice_shape);
    if (slice_elems <= 0) {
      return 1;
    }

    static constexpr double kL1TierBytes = static_cast<double>(32LL << 10);
    static constexpr double kNearL3TierBytes = static_cast<double>(2LL << 20);
    // Out of order single thread effective random fetch latencies and resident
    // cache tier bandwidths, plus multi workgroup L1 hit atomic latencies,
    // intra CCD and cross CCD RFO coherence penalties, and per cache line
    // contention serialization costs calibrated across AMD Rome, AMD Genoa, and
    // Intel Sapphire Rapids.
    static constexpr double kRandomL2FetchNs = 0.25;
    static constexpr double kRandomNearL3FetchNs = 0.65;
    static constexpr double kRandomFarL3FetchNs = 1.15;
    static constexpr double kScatterRandomDramFetchNs = 12.0;
    static constexpr double kScatterL2BytesPerNs = 200.0;
    static constexpr double kScatterNearL3BytesPerNs = 130.0;
    static constexpr double kScatterFarL3BytesPerNs = 110.0;
    static constexpr double kScatterDramBytesPerNs = 24.0;
    static constexpr double kLocalCacheDomainTasks = 8.0;
    static constexpr double kMaxExtraActiveDomains = 5.5;
    static constexpr double kMinScatterSingleThreadNs = 5.0 * kMinWorkPerTaskNs;

    static constexpr double kDirectAtomicStoreNs = 1.15;
    static constexpr double kDirectAtomicStore128Ns = 1.0;
    static constexpr double kDirectAtomicRmwNs = 4.2;
    static constexpr double kAtomicCas32Or64Ns = 7.2;
    static constexpr double kAtomicCasNarrowNs = 8.5;
    static constexpr double kAtomicCas128Ns = 14.0;

    // Intra domain RFO costs plus cross domain first hop and multi domain
    // directory/fabric congestion costs when the thread pool spans more than
    // one 8 core shared L3 domain (since Eigen distributes k workgroups across
    // the full pool without pinning them to a single domain).
    static constexpr double kUniqueSubLineLocalRfoNs = 2.5;
    static constexpr double kUniqueSubLineCrossHopNs = 7.5;
    static constexpr double kUniqueSubLineDomainCongNs = 4.2;
    static constexpr double kUniqueSliceLocalRfoNs = 1.5;
    static constexpr double kUniqueSliceCrossHopNs = 5.5;
    static constexpr double kUniqueSliceDomainCongNs = 3.0;
    static constexpr double kUniqueSeqLineCrossHopNs = 1.8;
    static constexpr double kUniqueSeqLineDomainCongNs = 0.9;

    static constexpr double kStoreFirstLineLocalRfoNs = 5.2;
    static constexpr double kStoreFirstLineCrossHopNs = 11.0;
    static constexpr double kStoreFirstLineDomainCongNs = 6.5;
    static constexpr double kStoreSeqLineLocalRfoNs = 2.0;
    static constexpr double kStoreSeqLineCrossHopNs = 1.5;
    static constexpr double kStoreSeqLineDomainCongNs = 0.8;

    static constexpr double kRmwFirstLineLocalRfoNs = 4.8;
    static constexpr double kRmwFirstLineCrossHopNs = 8.0;
    static constexpr double kRmwFirstLineDomainCongNs = 6.8;
    static constexpr double kRmwSeqLineLocalRfoNs = 2.5;
    static constexpr double kRmwSeqLineCrossHopNs = 2.0;
    static constexpr double kRmwSeqLineDomainCongNs = 1.5;

    static constexpr double kCasFirstLineLocalRfoNs = 8.5;
    static constexpr double kCasFirstLineCrossHopNs = 14.0;
    static constexpr double kCasFirstLineDomainCongNs = 11.0;
    static constexpr double kCasNarrowFirstLineLocalRfoNs = 8.5;
    static constexpr double kCasNarrowFirstLineCrossHopNs = 13.0;
    static constexpr double kCasNarrowFirstLineDomainCongNs = 10.0;
    static constexpr double kCas128FirstLineLocalRfoNs = 9.0;
    static constexpr double kCas128FirstLineCrossHopNs = 14.0;
    static constexpr double kCas128FirstLineDomainCongNs = 10.5;
    static constexpr double kCasSeqLineLocalRfoNs = 2.0;
    static constexpr double kCasSeqLineCrossHopNs = 1.0;
    static constexpr double kCasSeqLineDomainCongNs = 0.8;

    static constexpr double kStoreLineContentionNs = 80.0;
    static constexpr double kRmwLineContentionNs = 240.0;
    static constexpr double kCasLineContentionNs = 300.0;
    static constexpr double kCas128LineContentionNs = 340.0;

    // Fused index and update producer cost (excluding destination buffers).
    const ExecutionCostNs producer_cost = EstimateFusionCost(fusion);

    const ScatterDimensionNumbers& dnums = scatter.scatter_dimension_numbers();
    const int64_t index_vector_size = dnums.scatter_dims_to_operand_dims_size();
    const double updates_d = static_cast<double>(num_updates);
    const double total_slice_elems_d =
        updates_d * static_cast<double>(slice_elems);

    // Per update index fetch, unsigned range check, branch, and offset math.
    const double index_core_ns =
        updates_d * (1.0 + 0.35 * static_cast<double>(index_vector_size)) /
        kCyclesPerNs;

    const HloComputation& to_apply = *scatter.to_apply();
    const bool is_overwrite = IsCpuScatterOverwriteReducer(to_apply);
    const bool requires_atomic = CpuScatterRequiresAtomicUpdate(scatter);
    const double reducer_op_ns =
        is_overwrite ? 0.0
                     : ScalarReducerCyclesPerStep(to_apply) / kCyclesPerNs;

    // Count distinct addressable scatter slots across the scattered dimensions.
    const Shape& first_operand_shape =
        scatter.scatter_operands().front()->shape();
    double num_slots = 1.0;
    for (int64_t scatter_dim : dnums.scatter_dims_to_operand_dims()) {
      const int64_t valid_positions =
          std::max<int64_t>(1, first_operand_shape.dimensions(scatter_dim) -
                                   slice_shape[scatter_dim] + 1);
      num_slots *= static_cast<double>(valid_positions);
    }
    if (index_vector_size == 1) {
      num_slots = std::min(
          num_slots, MaxNonNegativeIndexCount(
                         scatter.scatter_indices()->shape().element_type()));
    }
    num_slots = std::max(1.0, num_slots);

    double write_1t_ns = 0.0;
    double write_mt_base_ns = 0.0;
    double local_coherence_ns = 0.0;
    double cross_domain_hop_ns = 0.0;
    double domain_congestion_per_domain_ns = 0.0;
    double contention_base_ns = 0.0;
    double dest_random_ns = 0.0;
    double dest_l2_l3_bw_ns = 0.0;
    double dest_dram_bw_ns = 0.0;
    bool uses_atomic_rmw_or_cas = false;

    for (const HloInstruction* operand : scatter.scatter_operands()) {
      const Shape& operand_shape = operand->shape();
      const PrimitiveType elem_type = operand_shape.element_type();
      const bool packed_sub_byte = IsCpuScatterPackedSubByteType(elem_type);
      const double elem_bytes = std::max(
          0.25,
          static_cast<double>(primitive_util::StorageBitWidth(elem_type)) /
              static_cast<double>(CHAR_BIT));
      const double dest_bytes = std::max(1.0, ArrayBytes(operand_shape));
      const double dest_cache_lines =
          std::max(1.0, std::ceil(dest_bytes / kCacheLineBytes));
      // Cross CCD RFO dominates when the scattered destination fits in cache;
      // once the destination spills to DRAM, DRAM miss latency dominates.
      const double fabric_scale = dest_bytes > kL3TotalBytes ? 0.25 : 1.0;

      int64_t contig_slice_elems = 1;
      double min_scatter_stride_bytes = elem_bytes;
      ComputeScatterSliceGeometry(
          operand_shape, slice_shape, dnums.scatter_dims_to_operand_dims(),
          elem_bytes, contig_slice_elems, min_scatter_stride_bytes);
      const double contig_slice_bytes =
          std::max(1.0, static_cast<double>(contig_slice_elems) * elem_bytes);
      const double fetches_per_update =
          static_cast<double>(slice_elems) /
          static_cast<double>(std::max<int64_t>(1, contig_slice_elems));
      const double first_fetches = updates_d * fetches_per_update;
      const double lines_per_fetch =
          std::max(1.0, std::ceil(contig_slice_bytes / kCacheLineBytes));
      const double extra_seq_lines =
          first_fetches * std::max(0.0, lines_per_fetch - 1.0);
      const double line_touches = first_fetches * lines_per_fetch;

      double active_cache_lines = 1.0;
      if (index_vector_size == 0) {
        active_cache_lines = std::clamp(fetches_per_update * lines_per_fetch,
                                        1.0, dest_cache_lines);
      } else {
        const double slot_span_bytes =
            std::max(contig_slice_bytes, min_scatter_stride_bytes);
        const double slots_per_line =
            slot_span_bytes >= kCacheLineBytes
                ? 1.0
                : std::max(1.0, std::floor(kCacheLineBytes / slot_span_bytes));
        active_cache_lines =
            std::clamp(std::ceil(num_slots / slots_per_line) *
                           fetches_per_update * lines_per_fetch,
                       1.0, dest_cache_lines);
      }

      // Single threaded write cost (always non atomic; vectorizes along
      // contiguous window slices for byte addressable element types).
      const int64_t vec_lanes =
          packed_sub_byte
              ? 1
              : std::max<int64_t>(
                    1, std::gcd(VectorLanes(elem_type), contig_slice_elems));
      const double vec_steps =
          total_slice_elems_d / static_cast<double>(vec_lanes);
      double single_write_ns = 0.0;
      if (is_overwrite) {
        single_write_ns =
            packed_sub_byte ? total_slice_elems_d * 0.9 : vec_steps * 0.35;
      } else {
        // Out of order single thread execution pipelines independent update
        // steps across iterations, whereas multi workgroup CAS loops serialize
        // the full reducer latency on each compare and swap attempt.
        single_write_ns =
            packed_sub_byte ? total_slice_elems_d * (1.2 + reducer_op_ns * 0.4)
                            : vec_steps * (0.45 + reducer_op_ns * 0.35);
      }
      write_1t_ns += single_write_ns;

      // Multi workgroup write, intra domain and cross domain RFO coherence, and
      // hot cache line contention serialization cost.
      if (!requires_atomic) {
        write_mt_base_ns += single_write_ns;
        if (contig_slice_bytes < kCacheLineBytes) {
          local_coherence_ns += first_fetches * kUniqueSubLineLocalRfoNs;
          cross_domain_hop_ns +=
              fabric_scale * first_fetches * kUniqueSubLineCrossHopNs;
          domain_congestion_per_domain_ns +=
              fabric_scale * first_fetches * kUniqueSubLineDomainCongNs;
        } else {
          local_coherence_ns += first_fetches * kUniqueSliceLocalRfoNs;
          cross_domain_hop_ns +=
              fabric_scale * (first_fetches * kUniqueSliceCrossHopNs +
                              extra_seq_lines * kUniqueSeqLineCrossHopNs);
          domain_congestion_per_domain_ns +=
              fabric_scale * (first_fetches * kUniqueSliceDomainCongNs +
                              extra_seq_lines * kUniqueSeqLineDomainCongNs);
        }
      } else {
        const bool direct_atomic_store =
            is_overwrite && !packed_sub_byte &&
            (elem_type != C128 || cpu_device_spec_.has_atomic_load_store_128);
        const double touches_per_line = line_touches / active_cache_lines;
        if (direct_atomic_store) {
          write_mt_base_ns +=
              total_slice_elems_d * (elem_type == C128 ? kDirectAtomicStore128Ns
                                                       : kDirectAtomicStoreNs);
          local_coherence_ns += first_fetches * kStoreFirstLineLocalRfoNs +
                                extra_seq_lines * kStoreSeqLineLocalRfoNs;
          cross_domain_hop_ns +=
              fabric_scale * (first_fetches * kStoreFirstLineCrossHopNs +
                              extra_seq_lines * kStoreSeqLineCrossHopNs);
          domain_congestion_per_domain_ns +=
              fabric_scale * (first_fetches * kStoreFirstLineDomainCongNs +
                              extra_seq_lines * kStoreSeqLineDomainCongNs);
          contention_base_ns = std::max(
              contention_base_ns, touches_per_line * kStoreLineContentionNs);
        } else if (!packed_sub_byte &&
                   IsCpuScatterDirectAtomicRmwReducer(to_apply, elem_type)) {
          uses_atomic_rmw_or_cas = true;
          write_mt_base_ns += total_slice_elems_d * kDirectAtomicRmwNs;
          local_coherence_ns += first_fetches * kRmwFirstLineLocalRfoNs +
                                extra_seq_lines * kRmwSeqLineLocalRfoNs;
          cross_domain_hop_ns +=
              fabric_scale * (first_fetches * kRmwFirstLineCrossHopNs +
                              extra_seq_lines * kRmwSeqLineCrossHopNs);
          domain_congestion_per_domain_ns +=
              fabric_scale * (first_fetches * kRmwFirstLineDomainCongNs +
                              extra_seq_lines * kRmwSeqLineDomainCongNs);
          contention_base_ns = std::max(
              contention_base_ns, touches_per_line * kRmwLineContentionNs);
        } else {
          uses_atomic_rmw_or_cas = true;
          double cas_ns = kAtomicCas32Or64Ns;
          double local_rfo_ns = kCasFirstLineLocalRfoNs;
          double cross_hop_ns = kCasFirstLineCrossHopNs;
          double domain_cong_ns = kCasFirstLineDomainCongNs;
          double line_contention_ns = kCasLineContentionNs;
          if (elem_type == C128) {
            cas_ns = kAtomicCas128Ns;
            local_rfo_ns = kCas128FirstLineLocalRfoNs;
            cross_hop_ns = kCas128FirstLineCrossHopNs;
            domain_cong_ns = kCas128FirstLineDomainCongNs;
            line_contention_ns = kCas128LineContentionNs;
          } else if (packed_sub_byte ||
                     primitive_util::StorageBitWidth(elem_type) <= 16) {
            cas_ns = kAtomicCasNarrowNs;
            local_rfo_ns = kCasNarrowFirstLineLocalRfoNs;
            cross_hop_ns = kCasNarrowFirstLineCrossHopNs;
            domain_cong_ns = kCasNarrowFirstLineDomainCongNs;
          }
          write_mt_base_ns += total_slice_elems_d * (cas_ns + reducer_op_ns);
          local_coherence_ns += first_fetches * local_rfo_ns +
                                extra_seq_lines * kCasSeqLineLocalRfoNs;
          cross_domain_hop_ns +=
              fabric_scale * (first_fetches * cross_hop_ns +
                              extra_seq_lines * kCasSeqLineCrossHopNs);
          domain_congestion_per_domain_ns +=
              fabric_scale * (first_fetches * domain_cong_ns +
                              extra_seq_lines * kCasSeqLineDomainCongNs);
          contention_base_ns = std::max(contention_base_ns,
                                        touches_per_line * line_contention_ns);
        }
      }

      // Cache tiered destination random fetch latency and read/write traffic
      // evaluated at the destination working set's resident cache tier.
      const double bytes_per_fetch =
          std::min(dest_bytes, std::max(contig_slice_bytes, kCacheLineBytes));
      const double rw_factor =
          (is_overwrite && contig_slice_bytes >= kCacheLineBytes)
              ? 1.5
              : kWriteAllocateFactor;
      const double physical_dest_bytes =
          rw_factor * first_fetches * bytes_per_fetch;
      if (dest_bytes > kL3TotalBytes) {
        dest_random_ns += first_fetches * kScatterRandomDramFetchNs;
        dest_dram_bw_ns += physical_dest_bytes / kScatterDramBytesPerNs;
      } else if (dest_bytes > kNearL3TierBytes) {
        dest_random_ns += first_fetches * kRandomFarL3FetchNs;
        dest_l2_l3_bw_ns += physical_dest_bytes / kScatterFarL3BytesPerNs;
      } else if (dest_bytes > kL2TierBytes) {
        dest_random_ns += first_fetches * kRandomNearL3FetchNs;
        dest_l2_l3_bw_ns += physical_dest_bytes / kScatterNearL3BytesPerNs;
      } else if (dest_bytes > kL1TierBytes) {
        dest_random_ns += first_fetches * kRandomL2FetchNs;
        dest_l2_l3_bw_ns += physical_dest_bytes / kScatterL2BytesPerNs;
      }
    }

    const double single_thread_core_ns =
        producer_cost.core_ns + index_core_ns + write_1t_ns + dest_random_ns;
    const double single_thread_mem_ns =
        producer_cost.memory_ns + dest_l2_l3_bw_ns + dest_dram_bw_ns;
    const double single_thread_ns =
        std::max(single_thread_core_ns, single_thread_mem_ns);
    if (!std::isfinite(single_thread_ns) ||
        single_thread_ns < kMinScatterSingleThreadNs) {
      return 1;
    }

    // Require a clear predicted wall time margin over non atomic single thread
    // execution before paying thread pool dispatch and cross core coherence.
    const double required_speedup_factor =
        uses_atomic_rmw_or_cas ? 0.80 : (requires_atomic ? 0.85 : 0.88);
    double best_time_ns = single_thread_ns * required_speedup_factor;
    int64_t best_tasks = 1;
    int64_t prev_updates_per_task = num_updates;

    const double pool_domains = std::max(
        1.0, static_cast<double>(max_parallelism_) / kLocalCacheDomainTasks);
    const double cross_domain_pair_prob = 1.0 - 1.0 / pool_domains;

    for (int64_t tasks = 2; tasks <= max_scatter_tasks; ++tasks) {
      const int64_t updates_per_task = CeilOfRatio(num_updates, tasks);
      if (updates_per_task == prev_updates_per_task) {
        // Increasing tasks without reducing the max updates per workgroup only
        // adds dispatch overhead without shrinking the critical path.
        continue;
      }
      prev_updates_per_task = updates_per_task;

      const double tasks_d = static_cast<double>(tasks);
      const double eff_parallelism =
          updates_d / static_cast<double>(updates_per_task);
      const double scatter_parallelism =
          eff_parallelism * (1.0 - 0.18 * (tasks_d - 1.0) / (tasks_d + 6.0));
      const double other_thread_frac = 1.0 - 1.0 / tasks_d;
      const double active_domains =
          pool_domains * (1.0 - std::pow(cross_domain_pair_prob, tasks_d));
      const double extra_active_domains =
          std::clamp(active_domains - 1.0, 0.0, kMaxExtraActiveDomains);
      const double cross_domain_coherence_ns =
          other_thread_frac * cross_domain_pair_prob * cross_domain_hop_ns +
          extra_active_domains * domain_congestion_per_domain_ns;
      const double coherence_ns =
          other_thread_frac * local_coherence_ns + cross_domain_coherence_ns;

      const double task_core_ns =
          producer_cost.core_ns / eff_parallelism +
          (index_core_ns + write_mt_base_ns + coherence_ns + dest_random_ns) /
              scatter_parallelism;

      const double shared_mem_parallelism =
          std::min(scatter_parallelism, static_cast<double>(max_memory_tasks_));
      const double task_mem_ns =
          (producer_cost.memory_ns + dest_dram_bw_ns) / shared_mem_parallelism +
          dest_l2_l3_bw_ns / scatter_parallelism;
      const double task_work_ns = std::max(task_core_ns, task_mem_ns);

      if (single_thread_ns / tasks_d < kMinWorkPerTaskNs &&
          task_work_ns < kMinWorkPerTaskNs) {
        break;
      }

      const double contention_ns = other_thread_frac * contention_base_ns;
      const double total_ns = kDispatchBaseNs + kDispatchPerTaskNs * tasks_d +
                              task_work_ns + contention_ns;
      if (total_ns < best_time_ns) {
        best_time_ns = total_ns;
        best_tasks = tasks;
      }
    }

    VLOG(3) << "ParallelCostModel scatter " << fusion.name()
            << " updates=" << num_updates << " slice_elems=" << slice_elems
            << " single_us=" << (single_thread_ns / 1e3)
            << " contention_us=" << (contention_base_ns / 1e3)
            << " best_us=" << (best_time_ns / 1e3)
            << " -> tasks=" << best_tasks;
    return best_tasks;
  }

  const int64_t max_parallelism_;
  const int64_t max_memory_tasks_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const int64_t effective_vector_bytes_;
  const std::string enabled_features_;
  const bool has_avx512bf16_;
  const bool enable_tiled_emitter_;
  const CpuDeviceSpec cpu_device_spec_;
  const bool has_cpp_gen_intrinsics_;
};

}  // namespace

ParallelTaskAssignment::ParallelTaskAssignment(
    const int64_t max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size, HloModule* module,
    const TargetMachineFeatures* target_machine_features)
    : cost_model_(std::make_unique<DefaultCostModel>(
          max_parallelism, shape_size,
          EffectiveVectorWidthBytes(*module, *target_machine_features),
          target_machine_features->get_enabled_target_features(),
          target_machine_features->has_avx512bf16(),
          options::EnableTiledEmitter(module->config()),
          target_machine_features->cpu_device_spec())),
      target_machine_features_(*target_machine_features) {
  VLOG(1) << "ParallelTaskAssignment max_parallelism: " << max_parallelism;
}

int64_t ParallelTaskAssignment::GetTargetParallelTaskCount(
    HloInstruction* instruction) {
  // Currently, we do not assign parallel tasks to instructions with at least
  // one of the following properties:
  // *) Internal threading (library calls to kConv, kDot, kFft, kCustomCall).
  // *) Emit custom loops (kSelectAndScatter).
  // *) Operations that are not thread safe (like infeed and rng).
  // *) Tuple-shaped.
  // *) Operations that might be implemented as an in-place
  //    dynamic-update-slice, because we can't know how many output elements
  //    they will write (out-of-place will touch the whole output buffer, while
  //    in-place will only touch the updated elements).
  // TODO(b/27458679) Parallelize instructions which are skipped here.
  auto opcode = instruction->opcode();
  if (MayBeImplementedAsInPlaceDynamicUpdateSlice(instruction) ||
      IsManyOperandNarrowConcatenate(*instruction) ||
      opcode == HloOpcode::kRng || opcode == HloOpcode::kConstant) {
    return 1;
  }

  // Skip custom fusions; evaluate eligible scatter fusions directly via their
  // update dimension and contention aware scatter cost model.
  if (opcode == HloOpcode::kFusion) {
    const HloFusionInstruction* fusion =
        Cast<HloFusionInstruction>(instruction);
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
      return 1;
    }
    if (IsScatterFusion(*fusion)) {
      return cost_model_->GetParallelTaskCount(instruction);
    }
  }

  if (instruction->shape().IsTuple() && !instruction->IsLoopFusion()) {
    return 1;
  }

  // Only allow instructions that can be trivially parallelized (where all
  // outputs can be computed independently of each other).
  if (instruction->IsElementwise() || instruction->IsLoopFusion() ||
      opcode == HloOpcode::kBroadcast || opcode == HloOpcode::kConcatenate ||
      opcode == HloOpcode::kDynamicSlice ||
      opcode == HloOpcode::kDynamicUpdateSlice ||
      opcode == HloOpcode::kGather || opcode == HloOpcode::kIota ||
      opcode == HloOpcode::kPad || opcode == HloOpcode::kReduce ||
      opcode == HloOpcode::kReduceWindow || opcode == HloOpcode::kReshape ||
      opcode == HloOpcode::kReverse || opcode == HloOpcode::kSlice ||
      opcode == HloOpcode::kTranspose ||
      (opcode == HloOpcode::kConvolution &&
       !PotentiallyImplementedAsEigenConvolution(*instruction,
                                                 target_machine_features_))) {
    // Consult 'cost_model_' to compute target parallel task count.
    return cost_model_->GetParallelTaskCount(instruction);
  }

  return 1;
}

absl::StatusOr<bool> ParallelTaskAssigner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "ParallelTaskAssigner ENTRY");
  XLA_VLOG_LINES(3, module->ToString());
  // Compute target parallel task counts for all instructions in 'module'.
  HloToParallelTasks hlo_to_parallel_tasks;
  ComputeTargetParallelTasks(module, &hlo_to_parallel_tasks);

  // Assign parallel tasks to target specific instructions in 'module'.
  // TODO(b/27458679) Support inter-op parallelism.
  bool changed = AssignParallelTasks(module, hlo_to_parallel_tasks);

  XLA_VLOG_LINES(2, "ParallelTaskAssigner EXIT");
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}

bool ParallelTaskAssigner::AssignParallelTasks(
    HloModule* module, const HloToParallelTasks& hlo_to_parallel_tasks) {
  return AssignParallelTasksHelper(module, module->entry_computation(),
                                   hlo_to_parallel_tasks);
}

bool ParallelTaskAssigner::AssignParallelTasksHelper(
    HloModule* module, HloComputation* computation,
    const HloToParallelTasks& hlo_to_parallel_tasks) {
  bool changed = false;
  // Snapshot set of instructions because outlining modifies the set below.
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  for (auto* instruction : instructions) {
    // Assign parallel tasks to sub-computations for While, Conditional and Call
    // HLOs.
    // TODO(b/27458679) Evaluate alternative intra-op parallelism placement,
    // and support other callable computations like reduce.
    bool control_flow_hlo = false;

    if (instruction->opcode() == HloOpcode::kWhile) {
      control_flow_hlo = true;
      changed |= AssignParallelTasksHelper(module, instruction->while_body(),
                                           hlo_to_parallel_tasks);

    } else if (instruction->opcode() == HloOpcode::kConditional) {
      control_flow_hlo = true;
      for (HloComputation* branch : instruction->branch_computations()) {
        changed |=
            AssignParallelTasksHelper(module, branch, hlo_to_parallel_tasks);
      }

    } else if (instruction->opcode() == HloOpcode::kCall) {
      control_flow_hlo = true;
      changed |= AssignParallelTasksHelper(module, instruction->to_apply(),
                                           hlo_to_parallel_tasks);
    }

    // Continue to the next instruction if we handled control flow above.
    if (control_flow_hlo) {
      continue;
    }

    // Skip if no parallel tasks were computed in first pass.
    auto it = hlo_to_parallel_tasks.find(instruction);
    if (it == hlo_to_parallel_tasks.end()) {
      continue;
    }
    // Get target parallel task count computed for 'instruction'.
    const int64_t target_parallel_task_count = (*it).second;
    std::vector<int64_t> dim_partition_counts;
    int64_t total_partition_count = 1;
    if (IsScatterFusion(*instruction)) {
      // Partition scatter fusions along num_updates rather than the destination
      // shape so rank 0 destinations, variadic tuples, and narrow histograms
      // with many updates partition cleanly while scatters with few updates
      // never create idle tasks.
      total_partition_count = GetCpuScatterEffectiveWorkGroups(
          *Cast<HloFusionInstruction>(instruction), target_parallel_task_count,
          target_machine_features_.cpu_device_spec(),
          /*buffer_assignment=*/nullptr, MinAlign());
      dim_partition_counts = {total_partition_count};
    } else {
      const Shape& root_0_shape = instruction->shape().IsTuple()
                                      ? instruction->shape().tuple_shapes(0)
                                      : instruction->shape();
      const Shape& indexing_shape =
          root_0_shape.IsTuple() ? root_0_shape.tuple_shapes(0) : root_0_shape;

      if (!indexing_shape.IsArray()) {
        continue;
      }

      // Assign feasible dimension partitions (based on actual dimension sizes).
      dim_partition_counts = ShapePartitionAssigner(indexing_shape)
                                 .Run(target_parallel_task_count);
      total_partition_count =
          ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
    }
    if (total_partition_count <= 1) {
      // Feasible partition calculation resulting in no partitioning, so skip.
      continue;
    }

    BackendConfig backend_config;
    absl::c_copy(dim_partition_counts,
                 tsl::protobuf::RepeatedFieldBackInserter(
                     backend_config.mutable_outer_dimension_partitions()));
    CHECK_OK(instruction->set_backend_config(backend_config));

    VLOG(2) << "Assigned parallel task count: " << total_partition_count
            << " to instruction: " << instruction->name();
    changed = true;
  }
  return changed;
}

void ParallelTaskAssigner::ComputeTargetParallelTasks(
    HloModule* module, HloToParallelTasks* hlo_to_parallel_tasks) {
  ParallelTaskAssignment parallel_task_assignment(max_parallelism_,
                                                  shape_size_function_, module,
                                                  &target_machine_features_);

  // Compute parallel task counts for all instructions in 'module'.
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      // Query ParallelTaskAssignment for target parallel task count.
      const int64_t target_parallel_task_count =
          parallel_task_assignment.GetTargetParallelTaskCount(instruction);
      if (target_parallel_task_count > 1) {
        hlo_to_parallel_tasks->insert(
            {instruction, target_parallel_task_count});
      }
    }
  }
}

}  // namespace xla::cpu
