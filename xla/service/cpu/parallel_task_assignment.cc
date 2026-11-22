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
#include <cstdint>
#include <limits>
#include <memory>
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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
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

// CpuScatterFusion always emits a single workgroup (num_threads_ = 1), so
// attaching outer_dimension_partitions to scatter fusions has no runtime effect
// and misleads downstream passes about whether the fusion runs in parallel.
bool IsSingleThreadedScatterFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  return instr.fused_expression_root()->opcode() == HloOpcode::kScatter;
}

bool HasUnboundedDynamicShape(const Shape& shape) {
  bool unbounded = false;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex&) {
        if (subshape.IsArray() && subshape.is_unbounded_dynamic()) {
          unbounded = true;
        }
      });
  return unbounded;
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
         instr->opcode() == HloOpcode::kCopy) {
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
// minimizes predicted roofline wall time including Worker::Parallelize dispatch
// cost and shared memory bandwidth saturation.
class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(int64_t max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   int64_t effective_vector_bytes, std::string enabled_features,
                   bool has_avx512bf16)
      : max_parallelism_(max_parallelism),
        max_memory_tasks_(ComputeMaxMemoryBandwidthTasks(max_parallelism)),
        shape_size_(shape_size),
        effective_vector_bytes_(effective_vector_bytes),
        enabled_features_(std::move(enabled_features)),
        has_avx512bf16_(has_avx512bf16),
        has_cpp_gen_intrinsics_(codegen::AreCppGenIntrinsicsAvailable()) {}
  ~DefaultCostModel() override = default;

  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    if (max_parallelism_ <= 1 ||
        HasUnboundedDynamicShape(instruction->shape())) {
      return 1;
    }
    for (const HloInstruction* operand : instruction->operands()) {
      if (HasUnboundedDynamicShape(operand->shape())) {
        return 1;
      }
    }

    const ExecutionCostNs cost =
        instruction->opcode() == HloOpcode::kFusion
            ? EstimateFusionCost(*Cast<HloFusionInstruction>(instruction))
            : EstimateStandaloneInstructionCost(*instruction);
    const int64_t tasks = ChooseTaskCount(cost);
    VLOG(3) << "ParallelCostModel " << instruction->name()
            << " core_us=" << (cost.core_ns / 1e3)
            << " memory_us=" << (cost.memory_ns / 1e3)
            << " vec_bytes=" << effective_vector_bytes_
            << " -> tasks=" << tasks;
    return tasks;
  }

 private:
  // Nominal CPU frequency for converting cycle estimates to nanoseconds.
  static constexpr double kCyclesPerNs = 2.5;
  static constexpr double kCacheLineBytes = 64.0;
  // Written cache lines incur read for ownership plus writeback (R + 2W).
  static constexpr double kWriteAllocateFactor = 2.0;

  // Piecewise single core physical bandwidth across private L2 (first 512 KiB
  // at 72 GB/s), shared L3 (up to 16 MiB at 28 GB/s), and DRAM (16 GB/s).
  static constexpr double kL2TierBytes = static_cast<double>(512LL << 10);
  static constexpr double kL3TotalBytes = static_cast<double>(16LL << 20);
  static constexpr double kL3TierBytes = kL3TotalBytes - kL2TierBytes;
  static constexpr double kL2BytesPerNs = 72.0;
  static constexpr double kL3BytesPerNs = 28.0;
  static constexpr double kDramBytesPerNs = 16.0;

  // Per core miss latency for random gather fetches beyond L2 or L3, and for
  // strided transpose rows whose address span exceeds hugepage L1 DTLB cover.
  static constexpr double kRandomL3FetchNs = 2.5;
  static constexpr double kRandomDramFetchNs = 8.0;
  static constexpr double kDtlbSweepThresholdBytes =
      static_cast<double>(128LL << 20);
  static constexpr double kDtlbMissStrideBytes = 2048.0;
  static constexpr double kTransposeDtlbMissCycles = 10.0;

  // Worker::Parallelize fixed launch cost, marginal per task fork and barrier
  // cost, and minimum single threaded work per spawned task.
  static constexpr double kDispatchBaseNs = 8'000.0;
  static constexpr double kDispatchPerTaskNs = 1'100.0;
  static constexpr double kMinWorkPerTaskNs = 17'000.0;

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

  template <typename Intrinsic>
  double IntrinsicCycles(PrimitiveType type, double f32_vec_cycles,
                         double f64_vec_cycles, double f32_libm_cycles,
                         double f64_libm_cycles) const {
    const PrimitiveType cty = FloatComputeType(type);
    const int64_t max_lanes = VectorLanes(cty);
    const int64_t width =
        MaxIntrinsicVectorWidth<Intrinsic>(enabled_features_, cty, max_lanes);
    if (width > 0) {
      const double vec_cycles = cty == F64 ? f64_vec_cycles : f32_vec_cycles;
      return vec_cycles / static_cast<double>(width);
    }
    return cty == F64 ? f64_libm_cycles : f32_libm_cycles;
  }

  double ExpCycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    const PrimitiveType cty = FloatComputeType(type);
    if (cty == F64) {
      return IntrinsicCycles<ci::Exp>(F64, 15.0, 26.0, 35.0, 65.0);
    }
    // PolynomialApproximations vectorizes F32 and F16 exp at widths up to 16.
    const int64_t width = std::min<int64_t>(16, VectorLanes(F32));
    return 15.0 / static_cast<double>(width) + SubF32ConvertOverhead(type);
  }

  double SinCosCycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    return IntrinsicCycles<ci::Sin>(type, 38.0, 66.0, 45.0, 75.0) +
           SubF32ConvertOverhead(type);
  }

  double TanhCycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    return IntrinsicCycles<ci::Tanh>(type, 17.0, 26.0, 35.0, 60.0) +
           SubF32ConvertOverhead(type);
  }

  double Expm1Cycles(PrimitiveType type) const {
    namespace ci = codegen::intrinsics;
    if (has_cpp_gen_intrinsics_) {
      return IntrinsicCycles<ci::Expm1>(type, 26.0, 40.0, 40.0, 65.0) +
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
    return has_avx512bf16_ ? 2.5 : 4.0;
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
        if (std::optional<StridedTransposeInfo> info =
                GetStridedTransposeInfo(op);
            info.has_value()) {
          return StridedTransposeCyclesPerElement(*info);
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

      case HloOpcode::kLog:
        if (is_complex) {
          // Complex log computes atan2(imag, real), which calls scalar libm.
          return (fty == F64 ? 70.0 : 40.0) +
                 IntrinsicCycles<ci::YnnLog>(fty, 17.0, 26.0, 35.0, 60.0);
        }
        return IntrinsicCycles<ci::YnnLog>(ty, 17.0, 26.0, 35.0, 60.0) +
               SubF32ConvertOverhead(ty);

      case HloOpcode::kLog1p:
        if (is_complex) {
          return (fty == F64 ? 75.0 : 45.0) +
                 IntrinsicCycles<ci::Log1p>(fty, 24.0, 34.0, 40.0, 65.0);
        }
        if (has_cpp_gen_intrinsics_) {
          return IntrinsicCycles<ci::YnnLog1p>(ty, 24.0, 34.0, 40.0, 65.0) +
                 SubF32ConvertOverhead(ty);
        }
        return IntrinsicCycles<ci::Log1p>(ty, 24.0, 34.0, 40.0, 65.0) +
               SubF32ConvertOverhead(ty);

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
        return IntrinsicCycles<ci::Erf>(F64, 16.0, 40.0, 35.0, 50.0);

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
        return IntrinsicCycles<ci::Sinh>(ty, 32.0, 60.0, 45.0, 75.0) +
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
        return IntrinsicCycles<ci::Acos>(ty, 32.0, 80.0, 45.0, 80.0) +
               SubF32ConvertOverhead(ty);

      case HloOpcode::kAtan2:
        if (!is_complex && IsScalarFloatOne(op.operand(1), enclosing_fusion)) {
          // SimplifyArithPass rewrites atan2(y, 1.0) to vectorized math.atan.
          return IntrinsicCycles<ci::Atan>(ty, 17.0, 27.0, 40.0, 70.0) +
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
      const StridedTransposeInfo& info) {
    // Non contiguous scalar loads plus L1D set aliasing on 1 KiB multiples, and
    // per core DTLB page walk stalls when a single row sweeps past L1 hugepage
    // DTLB coverage.
    double cycles = 0.45;
    const int64_t stride_int = static_cast<int64_t>(info.stride_bytes);
    if (stride_int >= 1024 && (stride_int % 1024) == 0) {
      cycles += 0.35;
    }
    if (info.stride_bytes >= kDtlbMissStrideBytes &&
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
        if (std::optional<StridedTransposeInfo> info =
                GetStridedTransposeInfo(op);
            info.has_value()) {
          return SafeArrayElements(op.shape()) *
                 StridedTransposeCyclesPerElement(*info);
        }
        return 0.0;
      }
      case HloOpcode::kConcatenate: {
        const int64_t num_operands = op.operand_count();
        if (num_operands <= 1) {
          return 0.0;
        }
        if (CanEmitFastConcatenateCopy(op, enclosing_fusion)) {
          // ConcatenateFusionKernelEmitter emits one CopySliceOp per operand
          // for every outer index, loading each operand pointer and copying a
          // potentially short slice.
          const double slice_copies =
              ConcatenateOuterElements(op) * static_cast<double>(num_operands);
          return slice_copies * 6.0;
        }
        // ElementalHloToMlir::EmitConcat emits a per element binary search tree
        // of scf.if branches across the concatenated operands.
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
      case HloOpcode::kGather:
        // Bounds clamping and multi dimensional offset calculation per element.
        return SafeArrayElements(op.shape()) * 2.0;

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
        // Loop fused elemental dots emit a scalar contracting loop per output.
        const double flops = static_cast<double>(HloCostAnalysis::GetDotFlops(
            op.operand(0)->shape(), op.shape(), op.dot_dimension_numbers()));
        const double fmas = 0.5 * std::max(0.0, flops);
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
    const double out_bytes = ArrayBytes(gather.shape());
    const double table_bytes = ArrayBytes(gather.operand(0)->shape());
    const double slice_bytes = GatherContiguousSliceBytes(gather);
    if (out_bytes <= 0.0 || table_bytes <= 0.0 || slice_bytes <= 0.0) {
      return;
    }
    const double num_fetches = utilization * (out_bytes / slice_bytes);
    const double bytes_per_fetch =
        std::min(table_bytes, std::max(slice_bytes, kCacheLineBytes));
    gather_read_bytes += num_fetches * bytes_per_fetch;
    if (table_bytes > kL3TotalBytes) {
      random_miss_ns += num_fetches * kRandomDramFetchNs;
    } else if (table_bytes > kL2TierBytes) {
      random_miss_ns += num_fetches * kRandomL3FetchNs;
    }
  }

  ExecutionCostNs EstimateFusionCost(const HloFusionInstruction& fusion) const {
    const HloComputation* comp = fusion.fused_instructions_computation();
    std::vector<HloInstruction*> instructions =
        comp->MakeInstructionPostOrder();
    absl::c_reverse(instructions);

    // Propagate utilization backward from root so dead branches and sliced or
    // gathered parameters are weighed by what the kernel actually evaluates,
    // while shared elementwise DAG nodes are counted once per root.
    absl::flat_hash_map<const HloInstruction*,
                        absl::flat_hash_set<const HloInstruction*>>
        elementwise_roots;
    absl::flat_hash_map<const HloInstruction*, double> root_utilization;
    absl::flat_hash_map<const HloInstruction*, double> utilization;
    // Tracks parameter/constant leaves read only through gather so their
    // cache line aligned gather traffic replaces full table size.
    absl::flat_hash_map<const HloInstruction*, double> direct_gather_bytes;

    const HloInstruction* root = fusion.fused_expression_root();
    elementwise_roots[root].insert(root);
    root_utilization[root] = 1.0;

    double total_compute_cycles = 0.0;
    double extra_read_bytes = 0.0;
    double random_miss_ns = 0.0;

    for (const HloInstruction* instr : instructions) {
      double util = 0.0;
      if (auto it = elementwise_roots.find(instr);
          it != elementwise_roots.end()) {
        for (const HloInstruction* use_root : it->second) {
          util += root_utilization[use_root];
        }
      }
      utilization[instr] = util;
      if (util <= 0.0) {
        continue;
      }

      total_compute_cycles += util * InstructionComputeCycles(*instr, &fusion);

      if (std::optional<StridedTransposeInfo> info =
              GetStridedTransposeInfo(*instr);
          info.has_value() && info->stride_bytes >= kCacheLineBytes) {
        extra_read_bytes += std::min(1.0, util) * ArrayBytes(instr->shape());
      }

      const HloOpcode opcode = instr->opcode();
      if (PreservesElementwiseIndexing(*instr)) {
        const absl::flat_hash_set<const HloInstruction*> roots =
            elementwise_roots.at(instr);
        for (const HloInstruction* operand : instr->operands()) {
          elementwise_roots[operand].insert(roots.begin(), roots.end());
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
              const double in_elems = SafeArrayElements(operand->shape());
              op_util =
                  in_elems > 0.0
                      ? util * SafeArrayElements(instr->shape()) / in_elems
                      : 0.0;
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
                    *instr, util, direct_gather_bytes[mem_src], random_miss_ns);
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

          default:
            break;
        }

        if (op_util > 0.0) {
          elementwise_roots[operand].insert(operand);
          root_utilization[operand] += op_util;
        }
      }
    }

    auto source_read_bytes = [&](const HloInstruction* src) -> double {
      const double full_bytes = ArrayBytes(src->shape());
      const double direct_util = std::min(1.0, utilization[src]);
      double bytes = direct_util * full_bytes;
      if (auto it = direct_gather_bytes.find(src);
          it != direct_gather_bytes.end()) {
        // If the source is also scanned sequentially, its cache lines are
        // already counted up to full_bytes; otherwise count the gather fetches.
        bytes = std::max(bytes, it->second);
      }
      return bytes;
    };

    const double write_bytes = RecursiveArrayBytes(fusion.shape());
    double read_bytes = extra_read_bytes;

    for (const HloInstruction* instr : instructions) {
      if (instr->opcode() == HloOpcode::kParameter) {
        if (instr->shape().IsArray()) {
          read_bytes += source_read_bytes(instr);
        }
      } else if (instr->opcode() == HloOpcode::kGetTupleElement &&
                 instr->shape().IsArray() &&
                 instr->LatestNonGteAncestor()->opcode() ==
                     HloOpcode::kParameter) {
        read_bytes += source_read_bytes(instr);
      } else if (instr->opcode() == HloOpcode::kConstant &&
                 instr->shape().IsArray() &&
                 SafeArrayElements(instr->shape()) > 1.0) {
        read_bytes += source_read_bytes(instr);
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
        read_bytes += ArrayBytes(instr.shape());
        break;
      case HloOpcode::kDynamicSlice:
        read_bytes += ArrayBytes(instr.shape());
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

  int64_t ChooseTaskCount(const ExecutionCostNs& cost) const {
    const double single_thread_ns = std::max(cost.core_ns, cost.memory_ns);
    if (!std::isfinite(single_thread_ns) ||
        single_thread_ns < 2.0 * kMinWorkPerTaskNs) {
      return 1;
    }

    const int64_t max_tasks = std::min<int64_t>(
        max_parallelism_, static_cast<int64_t>(std::min<double>(
                              static_cast<double>(max_parallelism_),
                              single_thread_ns / kMinWorkPerTaskNs)));
    if (max_tasks < 2) {
      return 1;
    }

    double best_time_ns = single_thread_ns;
    int64_t best_tasks = 1;
    double prev_parallel_ns = std::numeric_limits<double>::infinity();

    for (int64_t tasks = 2; tasks <= max_tasks; ++tasks) {
      const double task_core_ns = cost.core_ns / static_cast<double>(tasks);
      const double mem_parallelism =
          static_cast<double>(std::min(tasks, max_memory_tasks_));
      const double task_memory_ns = cost.memory_ns / mem_parallelism;
      const double total_ns = kDispatchBaseNs +
                              kDispatchPerTaskNs * static_cast<double>(tasks) +
                              std::max(task_core_ns, task_memory_ns);
      // total_ns is convex on tasks >= 2; stop once it starts rising.
      if (total_ns >= prev_parallel_ns) {
        break;
      }
      prev_parallel_ns = total_ns;
      if (total_ns < best_time_ns) {
        best_time_ns = total_ns;
        best_tasks = tasks;
      }
    }

    return best_tasks;
  }

  const int64_t max_parallelism_;
  const int64_t max_memory_tasks_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const int64_t effective_vector_bytes_;
  const std::string enabled_features_;
  const bool has_avx512bf16_;
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
          target_machine_features->has_avx512bf16())),
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
      opcode == HloOpcode::kRng || opcode == HloOpcode::kConstant) {
    return 1;
  }

  // Skip custom fusions and single threaded scatter fusions.
  if (opcode == HloOpcode::kFusion) {
    const HloFusionInstruction* fusion =
        Cast<HloFusionInstruction>(instruction);
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom ||
        IsSingleThreadedScatterFusion(*fusion)) {
      return 1;
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
    const Shape& root_0_shape = instruction->shape().IsTuple()
                                    ? instruction->shape().tuple_shapes(0)
                                    : instruction->shape();
    const Shape& indexing_shape =
        root_0_shape.IsTuple() ? root_0_shape.tuple_shapes(0) : root_0_shape;

    if (!indexing_shape.IsArray()) {
      continue;
    }

    // Assign feasible dimension partitions (based on actual dimension sizes).
    auto dim_partition_counts =
        ShapePartitionAssigner(indexing_shape).Run(target_parallel_task_count);
    const int64_t total_partition_count =
        ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
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
