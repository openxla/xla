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

#include "xla/service/cpu/ir_emission_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"
#include "xla/codegen/device_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/expanders/scatter_simplifier.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/util/determinism.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

int64_t GetMinimumAlignmentForArray(
    const Shape& shape, const TargetMachineFeatures& target_machine_features) {
  CHECK(shape.IsArray());

  // We don't require a layout to be set on `shape`.  This only works on CPU
  // because we don't pad our tensors or otherwise have complicated data tiling
  // schemes.

  int64_t allocation_size_bytes =
      ShapeUtil::ElementsIn(shape) *
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  return target_machine_features.minimum_alignment_for_allocation(
      allocation_size_bytes);
}

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features) {
  // The following conditions are necessary (but not sufficient) for
  // implementing `convolution` with Eigen convolution:
  // - the input and kernel have a non-zero number of elements.
  // - the input is in NHWC order.
  // - the kernel is in HWIO order.
  //
  // To be sufficient, certain layout constraints need to be satisfied as well.
  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(1)->shape();
  const Shape& output_shape = convolution.shape();

  auto is_aligned = [&](const Shape& shape) {
    return GetMinimumAlignmentForArray(shape, target_machine_features) >=
           TargetMachineFeatures::kEigenExpectedTensorAlignment;
  };

  if (!is_aligned(input_shape) || !is_aligned(kernel_shape) ||
      !is_aligned(output_shape)) {
    return false;
  }

  if (ShapeUtil::IsZeroElementArray(input_shape) ||
      ShapeUtil::IsZeroElementArray(kernel_shape)) {
    return false;
  }
  // Make sure input and kernel has the same data type.
  CHECK(
      ShapeUtil::SameElementTypeIgnoringFpPrecision(input_shape, kernel_shape));
  // TODO(b/65408531): Explore using Eigen dot for complex64 type.
  PrimitiveType primitive_type = input_shape.element_type();
  if (primitive_type != F16 && primitive_type != F32) {
    return false;
  }
  if (window_util::HasWindowReversal(convolution.window())) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  // Only 1D through 3D convolutions are supported at the moment.
  const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
  if (num_spatial_dims < 1 || num_spatial_dims > 3) {
    return false;
  }

  for (int64_t i = 0; i < num_spatial_dims; ++i) {
    if (dnums.input_spatial_dimensions(i) != i + 1) {
      return false;
    }
    if (dnums.kernel_spatial_dimensions(i) != i) {
      return false;
    }
    if (dnums.output_spatial_dimensions(i) != i + 1) {
      return false;
    }
  }

  return dnums.input_batch_dimension() == 0 &&
         dnums.input_feature_dimension() ==
             input_shape.dimensions().size() - 1 &&
         dnums.output_batch_dimension() == 0 &&
         dnums.output_feature_dimension() ==
             output_shape.dimensions().size() - 1 &&
         dnums.kernel_input_feature_dimension() ==
             kernel_shape.dimensions().size() - 2 &&
         dnums.kernel_output_feature_dimension() ==
             kernel_shape.dimensions().size() - 1;
}

bool CanUseEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features) {
  if (!PotentiallyImplementedAsEigenConvolution(convolution,
                                                target_machine_features)) {
    return false;
  }

  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(1)->shape();
  const Shape& output_shape = convolution.shape();

  return LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout()) &&
         LayoutUtil::IsMonotonicWithDim0Major(kernel_shape.layout()) &&
         LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout());
}

bool EmitsNoThunks(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    // Their semantics are encoded by buffer assignment.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    // They only constrain the dataflow graph or the schedule.
    case HloOpcode::kAfterAll:
    case HloOpcode::kAddDependency:
    case HloOpcode::kOptimizationBarrier:
    // Constants are owned by the executable and resolved through buffer
    // assignment at run time.
    case HloOpcode::kConstant:
      return true;
    default:
      return false;
  }
}

namespace {

bool CpuDeterministicOpsRequested(const HloModuleConfig& config) {
  const DebugOptions& debug_options = config.debug_options();
  return tsl::OpDeterminismRequired() ||
         debug_options.xla_gpu_deterministic_ops() ||
         debug_options.xla_gpu_exclude_nondeterministic_ops();
}

bool IsSupportedCpuScatterIndexType(PrimitiveType type) {
  if (!primitive_util::IsIntegralType(type)) {
    return false;
  }
  // S1 and U1 pack 8 elements per byte in XLA buffers, while MLIR elemental
  // lowering currently treats them as byte wide i8.
  return type != S1 && type != U1;
}

bool IsSupportedCpuScatterByteAddressableType(PrimitiveType type) {
  if (!primitive_util::IsArrayType(type) ||
      primitive_util::IsSubByteNonPredType(type)) {
    return false;
  }
  // 6 bit float types are not normalized by CpuCompiler and are unsupported in
  // ExpandFloatOpsPass for general comparisons and conversions.
  if (type == F6E2M3FN || type == F6E3M2FN) {
    return false;
  }
  return type == PRED || primitive_util::IsIntegralType(type) ||
         primitive_util::IsFloatingPointType(type) || type == C64 ||
         type == C128;
}

int64_t CpuScatterRequiredAtomicAlignmentBytes(PrimitiveType type) {
  if (IsCpuScatterPackedSubByteType(type)) {
    // Packed 2 bit and 4 bit atomics lower through byte wide i8 cmpxchg.
    return 1;
  }
  if (!IsSupportedCpuScatterByteAddressableType(type)) {
    return 0;
  }
  if (type == PRED) {
    return 1;
  }
  const int64_t bit_width = primitive_util::BitWidth(type);
  switch (bit_width) {
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      return bit_width / 8;
    default:
      return 0;
  }
}

int64_t GetEffectiveSliceAlignmentBytes(const BufferAllocation::Slice& slice,
                                        int64_t default_buffer_align_bytes) {
  if (default_buffer_align_bytes <= 0 || slice.allocation() == nullptr) {
    return 0;
  }
  return static_cast<int64_t>(
      llvm::MinAlign(static_cast<uint64_t>(default_buffer_align_bytes),
                     static_cast<uint64_t>(slice.offset())));
}

bool CanLowerCpuScatterAtomicUpdate(PrimitiveType type, bool is_overwrite,
                                    const CpuDeviceSpec& cpu_device_spec,
                                    int64_t guaranteed_align_bytes) {
  const int64_t required_align = CpuScatterRequiredAtomicAlignmentBytes(type);
  if (required_align <= 0 || guaranteed_align_bytes < required_align) {
    return false;
  }
  if (IsCpuScatterPackedSubByteType(type)) {
    // Both overwrite and read modify write on packed sub byte elements lower to
    // a byte wide i8 cmpxchg loop seeded by an i8 atomic load.
    return cpu_device_spec.has_atomic_rmw_64 &&
           cpu_device_spec.has_atomic_load_store_64;
  }
  if (type == C128) {
    if (is_overwrite) {
      return cpu_device_spec.has_atomic_load_store_128 ||
             cpu_device_spec.has_atomic_cmpxchg_128;
    }
    return cpu_device_spec.has_atomic_cmpxchg_128;
  }
  if (is_overwrite) {
    return cpu_device_spec.has_atomic_load_store_64;
  }
  return cpu_device_spec.has_atomic_rmw_64 &&
         cpu_device_spec.has_atomic_load_store_64;
}

// Returns true if `to_apply` contains arithmetic directly on an unnormalized
// F4 or F8 type that has no native LLVM scalar lowering (after CpuCompiler's
// FloatNormalization pass, narrow float reducers compute in F16 or F32 and only
// convert at the boundary).
bool HasUnnormalizedNarrowFloatReducerOp(const HloComputation& to_apply,
                                         PrimitiveType element_type) {
  if (element_type != F4E2M1FN && !primitive_util::IsF8Type(element_type)) {
    return false;
  }
  for (const HloInstruction* instr : to_apply.instructions()) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kConvert:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kCompare:
      case HloOpcode::kSelect:
      case HloOpcode::kAbs:
      case HloOpcode::kMinimum:
      case HloOpcode::kMaximum:
        continue;
      default:
        break;
    }
    if (instr->shape().element_type() == element_type) {
      return true;
    }
    for (const HloInstruction* operand : instr->operands()) {
      if (operand->shape().element_type() == element_type) {
        return true;
      }
    }
  }
  return false;
}

// Returns true if reordering concurrent scatter updates preserves exact
// bitwise output semantics.
bool IsCpuScatterOrderIndependent(const HloScatterInstruction& scatter) {
  if (scatter.unique_indices()) {
    return true;
  }
  if (scatter.scatter_operand_count() != 1) {
    return false;
  }
  const HloComputation* combiner = scatter.to_apply();
  if (combiner->num_parameters() != 2 || combiner->instruction_count() != 3) {
    return false;
  }
  const HloInstruction* root = combiner->root_instruction();
  if (root->operand_count() != 2) {
    return false;
  }
  const HloInstruction* p0 = combiner->parameter_instruction(0);
  const HloInstruction* p1 = combiner->parameter_instruction(1);
  const bool uses_both_params =
      (root->operand(0) == p0 && root->operand(1) == p1) ||
      (root->operand(0) == p1 && root->operand(1) == p0);
  if (!uses_both_params) {
    return false;
  }
  const PrimitiveType type = root->shape().element_type();
  if (primitive_util::IsIntegralType(type)) {
    switch (root->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kMultiply:
      case HloOpcode::kMinimum:
      case HloOpcode::kMaximum:
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
        return true;
      default:
        return false;
    }
  }
  if (type == PRED) {
    switch (root->opcode()) {
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
        return true;
      default:
        return false;
    }
  }
  // Floating point, complex, and overwrite reducers are order sensitive.
  return false;
}

struct OuterSubbuffer {
  const HloInstruction* root = nullptr;
  ShapeIndex index;
};

// Peels buffer forwarding wrappers and tuple/get-tuple-element chains so two
// fusion operands extracting the same or nested tuple subbuffer are recognized
// as aliasing even before buffer assignment runs.
OuterSubbuffer ResolveOuterSubbuffer(const HloInstruction* instr) {
  std::vector<int64_t> reversed_index;
  while (true) {
    if (instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kOptimizationBarrier ||
        instr->opcode() == HloOpcode::kAddDependency) {
      instr = instr->operand(0);
      continue;
    }
    if (instr->opcode() == HloOpcode::kGetTupleElement) {
      reversed_index.push_back(instr->tuple_index());
      instr = instr->operand(0);
      continue;
    }
    if (instr->opcode() == HloOpcode::kTuple && !reversed_index.empty()) {
      const int64_t idx = reversed_index.back();
      if (idx >= 0 && idx < instr->operand_count()) {
        reversed_index.pop_back();
        instr = instr->operand(idx);
        continue;
      }
    }
    break;
  }
  ShapeIndex index;
  for (auto it = reversed_index.rbegin(); it != reversed_index.rend(); ++it) {
    index.push_back(*it);
  }
  return OuterSubbuffer{instr, index};
}

bool OuterSubbuffersMayOverlap(const OuterSubbuffer& a,
                               const OuterSubbuffer& b) {
  if (a.root != b.root) {
    return false;
  }
  const int64_t prefix_len = std::min<int64_t>(a.index.size(), b.index.size());
  for (int64_t i = 0; i < prefix_len; ++i) {
    if (a.index[i] != b.index[i]) {
      return false;
    }
  }
  return true;
}

// Collects all fusion parameter numbers transitively feeding `scatter_indices`
// or any entry of `scatter_updates`.
absl::flat_hash_set<int64_t> CollectScatterProducerParameters(
    const HloScatterInstruction& scatter) {
  absl::flat_hash_set<int64_t> producer_params;
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist;
  worklist.push_back(scatter.scatter_indices());
  for (const HloInstruction* update : scatter.scatter_updates()) {
    worklist.push_back(update);
  }
  while (!worklist.empty()) {
    const HloInstruction* current = worklist.back();
    worklist.pop_back();
    if (!visited.insert(current).second) {
      continue;
    }
    if (current->opcode() == HloOpcode::kParameter) {
      producer_params.insert(current->parameter_number());
      continue;
    }
    for (const HloInstruction* operand : current->operands()) {
      worklist.push_back(operand);
    }
  }
  return producer_params;
}

bool HasCpuScatterFusionAliasingHazard(
    const HloFusionInstruction& fusion, const HloScatterInstruction& scatter,
    const BufferAssignment* buffer_assignment) {
  const int64_t operand_count = scatter.scatter_operand_count();
  std::vector<int64_t> operand_param_numbers;
  operand_param_numbers.reserve(operand_count);
  absl::flat_hash_set<int64_t> operand_param_set;
  std::vector<OuterSubbuffer> outer_operand_subbuffers;
  outer_operand_subbuffers.reserve(operand_count);

  for (const HloInstruction* operand : scatter.scatter_operands()) {
    // CpuScatterFusion emits in place updates and requires each scatter operand
    // (through any bitcasts) to be a distinct fusion parameter aliasing its
    // corresponding output.
    const HloInstruction* base = operand;
    while (base->opcode() == HloOpcode::kBitcast) {
      base = base->operand(0);
    }
    if (base->opcode() != HloOpcode::kParameter) {
      return true;
    }
    const int64_t param_num = base->parameter_number();
    if (!operand_param_set.insert(param_num).second) {
      return true;
    }
    operand_param_numbers.push_back(param_num);
    const OuterSubbuffer outer_subbuffer =
        ResolveOuterSubbuffer(fusion.operand(param_num));
    for (const OuterSubbuffer& prior : outer_operand_subbuffers) {
      if (OuterSubbuffersMayOverlap(outer_subbuffer, prior)) {
        return true;
      }
    }
    outer_operand_subbuffers.push_back(outer_subbuffer);
  }

  const absl::flat_hash_set<int64_t> producer_params =
      CollectScatterProducerParameters(scatter);
  for (int64_t producer_param : producer_params) {
    if (operand_param_set.contains(producer_param)) {
      return true;
    }
    const OuterSubbuffer producer_subbuffer =
        ResolveOuterSubbuffer(fusion.operand(producer_param));
    for (const OuterSubbuffer& operand_subbuffer : outer_operand_subbuffers) {
      if (OuterSubbuffersMayOverlap(producer_subbuffer, operand_subbuffer)) {
        return true;
      }
    }
  }

  for (const auto& [output_index, operand_alias] :
       fusion.output_to_operand_aliasing()) {
    const int64_t aliased_param = operand_alias.first;
    if (producer_params.contains(aliased_param)) {
      return true;
    }
    if (operand_count == 1) {
      if (!output_index.empty() || aliased_param != operand_param_numbers[0]) {
        return true;
      }
    } else {
      if (output_index.size() != 1 || output_index[0] < 0 ||
          output_index[0] >= operand_count ||
          aliased_param != operand_param_numbers[output_index[0]]) {
        return true;
      }
    }
  }

  if (buffer_assignment == nullptr) {
    return false;
  }

  std::vector<BufferAllocation::Slice> result_slices;
  result_slices.reserve(operand_count);
  for (int64_t i = 0; i < operand_count; ++i) {
    const ShapeIndex result_index =
        operand_count > 1 ? ShapeIndex{i} : ShapeIndex{};
    absl::StatusOr<BufferAllocation::Slice> result_slice =
        buffer_assignment->GetUniqueSlice(&fusion, result_index);
    absl::StatusOr<BufferAllocation::Slice> operand_slice =
        buffer_assignment->GetUniqueSlice(
            fusion.operand(operand_param_numbers[i]), {});
    if (!result_slice.ok() || !operand_slice.ok() ||
        *operand_slice != *result_slice) {
      return true;
    }
    for (const BufferAllocation::Slice& prior_slice : result_slices) {
      if (result_slice->OverlapsWith(prior_slice)) {
        return true;
      }
    }
    result_slices.push_back(*result_slice);
  }

  for (int64_t producer_param : producer_params) {
    const HloInstruction* outer_producer = fusion.operand(producer_param);
    bool overlaps = false;
    ShapeUtil::ForEachSubshape(
        outer_producer->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (overlaps || subshape.IsTuple()) {
            return;
          }
          for (const BufferAllocation::Slice& producer_slice :
               buffer_assignment->GetAllSlices(outer_producer, index)) {
            for (const BufferAllocation::Slice& result_slice : result_slices) {
              if (producer_slice.OverlapsWith(result_slice)) {
                overlaps = true;
                return;
              }
            }
          }
        });
    if (overlaps) {
      return true;
    }
  }

  return false;
}

}  // namespace

bool IsCpuScatterPackedSubByteType(PrimitiveType type) {
  if (!primitive_util::IsSubByteNonPredType(type)) {
    return false;
  }
  const int64_t bit_width = primitive_util::BitWidth(type);
  return bit_width == 2 || bit_width == 4;
}

bool IsCpuScatterOverwriteReducer(const HloComputation& to_apply) {
  if (to_apply.num_parameters() != 2) {
    return false;
  }
  const HloInstruction* p0 = to_apply.parameter_instruction(0);
  if (p0->IsDead()) {
    return true;
  }
  // Match EmitScatterAtomicUpdate: SubgraphToMlirFunction traverses only
  // instructions reachable from the root, so p0 with only dead users also
  // lowers to xla.atomic_store, whereas `ROOT p0 = parameter(0)` is reachable
  // and lowers to xla.atomic_rmw.
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist = {to_apply.root_instruction()};
  while (!worklist.empty()) {
    const HloInstruction* current = worklist.back();
    worklist.pop_back();
    if (current == p0) {
      return false;
    }
    if (!visited.insert(current).second) {
      continue;
    }
    for (const HloInstruction* operand : current->operands()) {
      worklist.push_back(operand);
    }
  }
  return true;
}

bool IsCpuScatterDirectAtomicRmwReducer(const HloComputation& to_apply,
                                        PrimitiveType element_type) {
  if (to_apply.num_parameters() != 2 || to_apply.instruction_count() != 3) {
    return false;
  }
  const bool is_byte_int = primitive_util::IsIntegralType(element_type) &&
                           !primitive_util::IsSubByteNonPredType(element_type);
  if (!is_byte_int && element_type != PRED) {
    return false;
  }
  const HloInstruction* root = to_apply.root_instruction();
  if (root->operand_count() != 2) {
    return false;
  }
  const HloInstruction* p0 = to_apply.parameter_instruction(0);
  const HloInstruction* p1 = to_apply.parameter_instruction(1);
  const bool forward_order = root->operand(0) == p0 && root->operand(1) == p1;
  const bool reverse_order = root->operand(0) == p1 && root->operand(1) == p0;
  if (!forward_order && !reverse_order) {
    return false;
  }
  if (element_type == PRED) {
    return root->opcode() == HloOpcode::kAnd ||
           root->opcode() == HloOpcode::kOr ||
           root->opcode() == HloOpcode::kXor;
  }
  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      return true;
    case HloOpcode::kSubtract:
      return forward_order;
    default:
      return false;
  }
}

bool CpuScatterRequiresAtomicUpdate(const HloScatterInstruction& scatter) {
  if (!scatter.unique_indices()) {
    return true;
  }
  return scatter.scatter_operand_count() == 1 &&
         IsCpuScatterPackedSubByteType(
             scatter.scatter_operands().front()->shape().element_type());
}

int64_t GetCpuScatterRequestedWorkGroups(const HloFusionInstruction& fusion) {
  absl::StatusOr<BackendConfig> backend_config =
      fusion.backend_config<BackendConfig>();
  if (!backend_config.ok()) {
    return 1;
  }
  int64_t requested = 1;
  for (int64_t dim_partitions : backend_config->outer_dimension_partitions()) {
    if (dim_partitions <= 1) {
      continue;
    }
    if (requested > std::numeric_limits<int64_t>::max() / dim_partitions) {
      return std::numeric_limits<int64_t>::max();
    }
    requested *= dim_partitions;
  }
  return requested;
}

int64_t GetCpuScatterEffectiveWorkGroups(
    const HloFusionInstruction& fusion, int64_t requested_workgroups,
    const CpuDeviceSpec& cpu_device_spec,
    const BufferAssignment* buffer_assignment,
    int64_t default_buffer_align_bytes) {
  if (requested_workgroups <= 1) {
    return 1;
  }
  const HloScatterInstruction* scatter =
      DynCast<HloScatterInstruction>(fusion.fused_expression_root());
  if (scatter == nullptr || scatter->scatter_operand_count() < 1) {
    return 1;
  }
  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
  if (!ScatterSimplifier::IsSimplifiedScatter(scatter) ||
      !dnums.input_batching_dims().empty() ||
      !dnums.scatter_indices_batching_dims().empty()) {
    return 1;
  }

  const Shape& indices_shape = scatter->scatter_indices()->shape();
  if (indices_shape.is_dynamic() ||
      !IsSupportedCpuScatterIndexType(indices_shape.element_type())) {
    return 1;
  }

  for (int64_t i = 0; i < scatter->scatter_operand_count(); ++i) {
    const Shape& operand_shape = scatter->scatter_operands()[i]->shape();
    const Shape& update_shape = scatter->scatter_updates()[i]->shape();
    if (operand_shape.is_dynamic() || update_shape.is_dynamic() ||
        ShapeUtil::IsZeroElementArray(operand_shape) ||
        operand_shape.element_type() != update_shape.element_type()) {
      return 1;
    }
    const PrimitiveType elem_type = operand_shape.element_type();
    if (!IsSupportedCpuScatterByteAddressableType(elem_type) &&
        !IsCpuScatterPackedSubByteType(elem_type)) {
      return 1;
    }
  }

  const Shape& first_update_shape = scatter->scatter_updates().front()->shape();
  if (first_update_shape.dimensions().empty() ||
      ShapeUtil::IsZeroElementArray(first_update_shape)) {
    return 1;
  }
  const int64_t num_updates = first_update_shape.dimensions(0);
  if (num_updates <= 1) {
    return 1;
  }
  int64_t slice_product = 1;
  for (int64_t dim : first_update_shape.dimensions().subspan(1)) {
    if (dim <= 0 || slice_product > std::numeric_limits<int64_t>::max() / dim) {
      return 1;
    }
    slice_product *= dim;
  }

  if (const HloModule* module = fusion.GetModule();
      module != nullptr && CpuDeterministicOpsRequested(module->config()) &&
      !IsCpuScatterOrderIndependent(*scatter)) {
    return 1;
  }

  if (scatter->scatter_operand_count() > 1) {
    // Variadic scatter only supports multi workgroup execution when indices are
    // guaranteed unique and every operand is byte addressable so distinct
    // indices never race on a shared storage byte.
    if (!scatter->unique_indices()) {
      return 1;
    }
    for (const HloInstruction* operand : scatter->scatter_operands()) {
      if (!IsSupportedCpuScatterByteAddressableType(
              operand->shape().element_type())) {
        return 1;
      }
    }
  } else if (CpuScatterRequiresAtomicUpdate(*scatter)) {
    const PrimitiveType elem_type =
        scatter->scatter_operands().front()->shape().element_type();
    int64_t guaranteed_align = default_buffer_align_bytes;
    if (buffer_assignment != nullptr) {
      absl::StatusOr<BufferAllocation::Slice> result_slice =
          buffer_assignment->GetUniqueSlice(&fusion, {});
      if (!result_slice.ok()) {
        return 1;
      }
      guaranteed_align = GetEffectiveSliceAlignmentBytes(
          *result_slice, default_buffer_align_bytes);
    }
    const bool is_overwrite =
        IsCpuScatterOverwriteReducer(*scatter->to_apply());
    if (!CanLowerCpuScatterAtomicUpdate(elem_type, is_overwrite,
                                        cpu_device_spec, guaranteed_align)) {
      return 1;
    }
    if (!is_overwrite &&
        HasUnnormalizedNarrowFloatReducerOp(*scatter->to_apply(), elem_type)) {
      return 1;
    }
  }

  if (HasCpuScatterFusionAliasingHazard(fusion, *scatter, buffer_assignment)) {
    return 1;
  }

  return std::min<int64_t>(requested_workgroups, num_updates);
}

}  // namespace cpu
}  // namespace xla
