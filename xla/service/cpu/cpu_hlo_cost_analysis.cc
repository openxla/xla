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

#include "xla/service/cpu/cpu_hlo_cost_analysis.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::Status CpuHloCostAnalysis::HandleElementwiseOp(
    const HloInstruction* hlo) {
  current_properties_[kFlopsKey] = GetFlopsForElementwiseOp(hlo);
  return absl::OkStatus();
}

int64_t CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(
    PrimitiveType type, HloOpcode opcode) const {
  // Approximate cost relative to an add, for vectorized code on x86 and
  // AArch64. Transcendentals are expanded to polynomial approximations.
  int64_t flops;
  switch (opcode) {
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
      flops = 8;
      break;
    case HloOpcode::kCbrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
      flops = 10;
      break;
    case HloOpcode::kCos:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kSin:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
      flops = 15;
      break;
    case HloOpcode::kAtan2:
    case HloOpcode::kPower:
      flops = 20;
      break;
    default:
      flops = 1;
      break;
  }
  if (primitive_util::IsComplexType(type)) {
    flops *= 4;
  }
  return flops;
}

int64_t CpuHloCostAnalysis::GetFlopsForElementwiseOp(HloOpcode op_code,
                                                     const Shape& shape) const {
  int64_t flop_per_element =
      GetFlopsPerElementwiseOpElement(shape.element_type(), op_code);
  return flop_per_element * ShapeUtil::ElementsInRecursive(shape);
}

int64_t CpuHloCostAnalysis::GetFlopsForElementwiseOp(
    const HloInstruction* instr) const {
  return GetFlopsForElementwiseOp(instr->opcode(), instr->shape());
}

float CpuHloCostAnalysis::CommonElementwiseUtilization(
    const HloInstruction* a, const HloInstruction* b) const {
  float ret = 0;
  for (auto r : elementwise_use_roots_.at(a)) {
    if (elementwise_use_roots_.at(b).count(r)) {
      ret += root_utilizations_.at(r);
    }
  }
  return ret;
}

std::unique_ptr<HloCostAnalysis>
CpuHloCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<CpuHloCostAnalysis>(options_);
}

int64_t CpuHloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction* hlo) const {
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  float utilization = hlo_properties_.at(hlo)[kUtilizationKey];
  if (!options_.count_multiple_input_accesses) {
    utilization = fmin(utilization, 1.0);
  }
  return std::llround(GetShapeSize(hlo->shape()) * utilization);
}

// Same as GpuHloCostAnalysis::FusionCalculateUtilizations, without the IR size
// estimate.
absl::Status CpuHloCostAnalysis::FusionCalculateUtilizations(
    const HloInstruction* fusion) {
  const HloInstruction* root = fusion->fused_expression_root();
  // Traverse the fused computation from the root to the parameters,
  // propagating operand utilization. All users of an instruction are processed
  // before the instruction itself.
  std::vector<HloInstruction*> instructions =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  absl::c_reverse(instructions);

  for (const HloInstruction* instr : instructions) {
    hlo_properties_[instr][kUtilizationKey] = 0;
    elementwise_use_roots_[instr].clear();
    root_utilizations_[instr] = 0;
  }

  // Assume that the fusion always produces all of its outputs.
  root_utilizations_[root] = 1.0;
  elementwise_use_roots_[root].insert(root);

  current_properties_[kFlopsKey] = 0;

  for (const HloInstruction* instr : instructions) {
    Properties& instr_props = hlo_properties_[instr];
    for (const HloInstruction* r : elementwise_use_roots_[instr]) {
      instr_props[kUtilizationKey] += root_utilizations_[r];
    }

    float cur_instr_utilization = instr_props[kUtilizationKey];
    VLOG(8) << instr->name() << " utilization: " << cur_instr_utilization;

    current_properties_[kFlopsKey] +=
        cur_instr_utilization * instr_props[kFlopsKey];

    for (int operand_idx = 0; operand_idx < instr->operand_count();
         ++operand_idx) {
      const HloInstruction* operand = instr->operand(operand_idx);
      if ((instr->IsElementwise() ||
           (instr->opcode() == HloOpcode::kBitcast &&
            ShapeUtil::EqualIgnoringElementType(instr->operand(0)->shape(),
                                                instr->shape()))) ||
          instr->opcode() == HloOpcode::kTuple ||
          instr->opcode() == HloOpcode::kGetTupleElement) {
        for (const HloInstruction* r : elementwise_use_roots_[instr]) {
          elementwise_use_roots_[operand].insert(r);
        }
      } else {
        elementwise_use_roots_[operand].insert(operand);
        float cur_operand_utilization =
            cur_instr_utilization * operand_utilization(*instr, operand_idx);
        // Round up to a whole number of produced elements.
        int64_t operand_elements =
            ShapeUtil::ElementsInRecursive(operand->shape());
        if (operand_elements == 0) {
          cur_operand_utilization = 0;
        } else {
          cur_operand_utilization =
              ceil(cur_operand_utilization * operand_elements) /
              operand_elements;
        }
        root_utilizations_[operand] += cur_operand_utilization;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xla::cpu
