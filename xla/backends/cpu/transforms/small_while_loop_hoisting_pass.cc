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

#include "xla/backends/cpu/transforms/small_while_loop_hoisting_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

// Upper bounds on hoisted loop body complexity. Hoisting links every nested
// kernel into a single LLVM module compiled on one thread and executes them
// serially in one workgroup. Loops with many non trivial array kernels gain
// negligible runtime benefit from skipping thunk dispatch while suffering
// superlinear LLVM optimization time. Scalar heavy loops (including tiny in
// place slice updates) deduplicate well and are allowed a higher total kernel
// budget.
constexpr int64_t kScalarKernelMaxElements = 16;
constexpr int64_t kMaxHoistedArrayLoopKernels = 64;
constexpr int64_t kMaxHoistedTotalKernels = 512;
constexpr int64_t kMaxHoistedFusedInstructions = 8192;

bool IsSupportedDotElementType(PrimitiveType type) {
  switch (type) {
    case PRED:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
      return true;
    default:
      return false;
  }
}

// Hoisted loops are compiled into a single kernel, which only works for
// instructions that ThunkEmitter lowers to kernels, copies, or control flow.
// Others (library calls, runtime state, I/O) would make ThunkEmitter emit the
// loop as an ordinary call, so hoisting them gains nothing.
bool InstructionIsUnavailable(
    const HloInstruction* instr,
    const TargetMachineFeatures& target_machine_features) {
  switch (instr->opcode()) {
    case HloOpcode::kCustomCall:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kSort:
    case HloOpcode::kFft:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRngGetAndUpdateState:
      return true;

    // Custom fusions call into oneDNN or YNNPACK.
    case HloOpcode::kFusion:
      return instr->fusion_kind() == HloInstruction::FusionKind::kCustom;

    case HloOpcode::kDot:
      return !IsSupportedDotElementType(instr->shape().element_type()) ||
             GetDotImplementationStrategy(instr->GetModule()->config(), *instr,
                                          target_machine_features,
                                          /*allow_runtime_calls=*/true) ==
                 DotImplementationStrategy::kEigen;

    // Convolutions that cannot use Eigen are wrapped into loop fusions.
    case HloOpcode::kConvolution:
      return CanUseEigenConvolution(*instr, target_machine_features);

    default:
      return IsCollective(instr);
  }
}

// Returns whether `computation`, or a fusion computation nested in it, has a
// dot that would call Eigen outside a fusion. Fusion emitters lower such dots
// inline, but inside a hoisted loop they would run on one thread.
bool HasEigenDot(const HloComputation& computation,
                 const TargetMachineFeatures& target_machine_features) {
  for (const HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() == HloOpcode::kDot &&
        GetDotImplementationStrategy(instr->GetModule()->config(), *instr,
                                     target_machine_features,
                                     /*allow_runtime_calls=*/true) ==
            DotImplementationStrategy::kEigen) {
      return true;
    }
    for (const HloComputation* called : instr->called_computations()) {
      if (HasEigenDot(*called, target_machine_features)) {
        return true;
      }
    }
  }
  return false;
}

// Returns whether `instr` or a computation it calls contains an instruction
// that cannot be part of a hoisted loop kernel.
bool ContainsUnavailableInstruction(
    const HloInstruction* instr,
    const TargetMachineFeatures& target_machine_features,
    absl::flat_hash_map<const HloInstruction*, bool>& has_unavailable_instr) {
  if (const auto itr = has_unavailable_instr.find(instr);
      itr != has_unavailable_instr.end()) {
    return itr->second;
  }

  if (InstructionIsUnavailable(instr, target_machine_features)) {
    return has_unavailable_instr.insert({instr, true}).first->second;
  }

  for (const HloComputation* called_comp : instr->called_computations()) {
    if (called_comp->IsFusionComputation()) {
      if (instr->fusion_kind() != HloInstruction::FusionKind::kOutput &&
          HasEigenDot(*called_comp, target_machine_features)) {
        return has_unavailable_instr.insert({instr, true}).first->second;
      }
      continue;
    }
    for (const HloInstruction* sub_instr : called_comp->instructions()) {
      if (ContainsUnavailableInstruction(sub_instr, target_machine_features,
                                         has_unavailable_instr)) {
        return has_unavailable_instr.insert({instr, true}).first->second;
      }
    }
  }

  return has_unavailable_instr.insert({instr, false}).first->second;
}

int64_t BoundedArrayElements(const Shape& shape) {
  if (!shape.IsArray()) {
    return 0;
  }
  if (shape.is_unbounded_dynamic()) {
    return kScalarKernelMaxElements + 1;
  }
  int64_t total = 1;
  for (int64_t dim : shape.dimensions()) {
    if (dim <= 0) {
      return 0;
    }
    if (dim > kScalarKernelMaxElements ||
        total > kScalarKernelMaxElements / dim) {
      return kScalarKernelMaxElements + 1;
    }
    total *= dim;
  }
  return total;
}

int64_t BoundedShapeElements(const Shape& shape) {
  if (shape.IsArray()) {
    return BoundedArrayElements(shape);
  }
  if (!shape.IsTuple()) {
    return 0;
  }
  int64_t total = 0;
  for (const Shape& sub : shape.tuple_shapes()) {
    total = std::min(kScalarKernelMaxElements + 1,
                     total + BoundedShapeElements(sub));
  }
  return total;
}

int64_t BoundedScatterUpdateElements(const HloScatterInstruction& scatter) {
  int64_t total = 0;
  for (const HloInstruction* update : scatter.scatter_updates()) {
    total = std::min(kScalarKernelMaxElements + 1,
                     total + BoundedArrayElements(update->shape()));
  }
  return total;
}

bool InstructionExceedsScalarWork(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kDynamicUpdateSlice:
      return BoundedArrayElements(instr.operand(1)->shape()) >
             kScalarKernelMaxElements;
    case HloOpcode::kScatter:
      return BoundedScatterUpdateElements(*Cast<HloScatterInstruction>(
                 &instr)) > kScalarKernelMaxElements;
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return BoundedArrayElements(instr.operand(0)->shape()) >
                 kScalarKernelMaxElements ||
             BoundedShapeElements(instr.shape()) > kScalarKernelMaxElements;
    case HloOpcode::kDot:
      return BoundedArrayElements(instr.operand(0)->shape()) >
                 kScalarKernelMaxElements ||
             BoundedArrayElements(instr.operand(1)->shape()) >
                 kScalarKernelMaxElements ||
             BoundedArrayElements(instr.shape()) > kScalarKernelMaxElements;
    default:
      return BoundedShapeElements(instr.shape()) > kScalarKernelMaxElements;
  }
}

bool IsDirectFusionParameter(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast) {
    instr = instr->operand(0);
  }
  if (instr->opcode() == HloOpcode::kGetTupleElement) {
    instr = instr->LatestNonGteAncestor();
  }
  return instr->opcode() == HloOpcode::kParameter;
}

bool IsRootInPlaceDus(const HloInstruction* sub, const HloInstruction* root) {
  if (sub->opcode() != HloOpcode::kDynamicUpdateSlice ||
      !IsDirectFusionParameter(sub->operand(0))) {
    return false;
  }
  if (sub == root) {
    return true;
  }
  return root->opcode() == HloOpcode::kTuple &&
         absl::c_linear_search(root->operands(), sub);
}

struct HoistedBodyComplexity {
  int64_t total_kernels = 0;
  int64_t array_loop_kernels = 0;
  int64_t fused_instructions = 0;

  bool ExceedsLimit() const {
    return total_kernels > kMaxHoistedTotalKernels ||
           array_loop_kernels > kMaxHoistedArrayLoopKernels ||
           fused_instructions > kMaxHoistedFusedInstructions;
  }
};

void AccumulateFusionBodyComplexity(const HloInstruction& fusion,
                                    HoistedBodyComplexity& complexity) {
  const HloComputation* body = fusion.fused_instructions_computation();
  const HloInstruction* root = body->root_instruction();

  bool is_array_loop = false;
  for (const HloInstruction* sub : body->instructions()) {
    if (EmitsNoThunks(*sub)) {
      continue;
    }
    ++complexity.fused_instructions;
    if (is_array_loop) {
      continue;
    }
    // An in place dynamic-update-slice root loops over only the update slice
    // rather than the full destination buffer.
    if (IsRootInPlaceDus(sub, root)) {
      if (BoundedArrayElements(sub->operand(1)->shape()) >
          kScalarKernelMaxElements) {
        is_array_loop = true;
      }
      continue;
    }
    if (InstructionExceedsScalarWork(*sub)) {
      is_array_loop = true;
    }
  }
  if (is_array_loop) {
    ++complexity.array_loop_kernels;
  }
}

bool AccumulateComputationComplexity(const HloComputation& comp,
                                     HoistedBodyComplexity& complexity) {
  for (const HloInstruction* instr : comp.instructions()) {
    const HloOpcode op = instr->opcode();
    if (EmitsNoThunks(*instr) || op == HloOpcode::kCopy) {
      continue;
    }
    if (op == HloOpcode::kWhile || op == HloOpcode::kConditional ||
        op == HloOpcode::kCall) {
      for (const HloComputation* called : instr->called_computations()) {
        if (AccumulateComputationComplexity(*called, complexity)) {
          return true;
        }
      }
      continue;
    }
    ++complexity.total_kernels;
    if (op == HloOpcode::kFusion) {
      AccumulateFusionBodyComplexity(*instr, complexity);
    } else {
      ++complexity.fused_instructions;
      if (InstructionExceedsScalarWork(*instr)) {
        ++complexity.array_loop_kernels;
      }
    }
    if (complexity.ExceedsLimit()) {
      return true;
    }
  }
  return false;
}

bool ExceedsHoistedBodySizeLimit(const HloInstruction& while_instr) {
  HoistedBodyComplexity complexity;
  for (const HloComputation* called : while_instr.called_computations()) {
    if (AccumulateComputationComplexity(*called, complexity)) {
      return true;
    }
  }
  return complexity.ExceedsLimit();
}

}  // namespace

SmallWhileLoopHoistingPass::SmallWhileLoopHoistingPass(
    int64_t small_buffer_access_size,
    const TargetMachineFeatures* target_machine_features)
    : small_buffer_access_size_(small_buffer_access_size),
      target_machine_features_(target_machine_features) {}

absl::StatusOr<bool> SmallWhileLoopHoistingPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }

  bool changed = false;
  absl::flat_hash_map<const HloInstruction*, bool> has_unavailable_instr;
  for (HloInstruction* while_instr : while_instrs) {
    if (ContainsUnavailableInstruction(while_instr, *target_machine_features_,
                                       has_unavailable_instr)) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(bool is_small_call_site, IsSmall(while_instr));
    if (!is_small_call_site) {
      continue;
    }

    HloComputation::Builder builder(
        absl::StrCat(while_instr->name(), "_computation"));
    std::vector<HloInstruction*> parameters;
    parameters.reserve(while_instr->operand_count());
    for (HloInstruction* operand : while_instr->operands()) {
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * parameter,
          builder.AddParameter(HloInstruction::CreateParameter(
              while_instr->operand_index(operand), operand->shape(),
              operand->name())));
      parameters.push_back(parameter);
    }
    builder.AddInstruction(
        while_instr->CloneWithNewOperands(while_instr->shape(), parameters));

    HloInstruction* call_instruction =
        while_instr->AddInstruction(HloInstruction::CreateCall(
            while_instr->shape(), while_instr->operands(),
            module->AddEmbeddedComputation(builder.Build())));
    call_instruction->add_frontend_attribute("xla_cpu_small_call", "true");

    ABSL_RETURN_IF_ERROR(call_instruction->CopyAllControlDepsFrom(while_instr));
    ABSL_RETURN_IF_ERROR(while_instr->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(while_instr->ReplaceAllUsesWith(call_instruction));
    ABSL_RETURN_IF_ERROR(while_instr->parent()->RemoveInstruction(while_instr));

    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> SmallWhileLoopHoistingPass::IsSmall(
    const HloInstruction* instr) {
  if (ExceedsHoistedBodySizeLimit(*instr)) {
    return false;
  }
  HloCostAnalysis cost_analysis(&CpuExecutable::ShapeSizeBytes);
  ABSL_RETURN_IF_ERROR(cost_analysis.RevisitInstruction(instr));
  return cost_analysis.bytes_accessed(*instr) < small_buffer_access_size_;
}

}  // namespace xla::cpu
