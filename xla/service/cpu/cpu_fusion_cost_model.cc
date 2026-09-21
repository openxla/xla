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

#include "xla/service/cpu/cpu_fusion_cost_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/window_util.h"

namespace xla::cpu {
namespace {

// Producers we never want to recompute, whatever the arithmetic says: their
// cost is not a per-element constant, or duplicating them is not legal/useful.
constexpr int64_t kProhibitive = std::numeric_limits<int64_t>::max() / 4;

// Flop-equivalent weight of one element of `opcode`.
//
// These are deliberately coarse. The decision the model makes is a comparison
// against a memory round trip worth tens to hundreds of flops, so what matters
// is the order of magnitude -- an `exp` costing 12 or 18 never flips a verdict,
// whereas classifying it the same as an `add` does.
int64_t OpcodeFlops(HloOpcode opcode) {
  switch (opcode) {
    // Pure data movement: the loop still runs, but no arithmetic is redone.
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kCopy:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kIota:
    case HloOpcode::kParameter:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return 0;

    // Single-cycle-ish arithmetic and logic.
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kCompare:
    case HloOpcode::kConvert:
    case HloOpcode::kFloor:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kSelect:
    case HloOpcode::kSign:
    case HloOpcode::kSubtract:
    case HloOpcode::kXor:
      return 1;

    // Indexed reads: an address computation plus a load, no real arithmetic,
    // but the load may miss.
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
      return 1;
    case HloOpcode::kGather:
      return 4;

    // Multi-cycle, non-pipelined.
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
      return 8;

    case HloOpcode::kCbrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
      return 10;

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
      return 15;

    case HloOpcode::kAtan2:
    case HloOpcode::kPower:
      return 20;

    // Not a per-element cost: recomputing a contraction per consumer element
    // is never what we want.
    case HloOpcode::kDot:
    case HloOpcode::kConvolution:
    case HloOpcode::kCustomCall:
    case HloOpcode::kSort:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      return kProhibitive;

    default:
      // Unknown opcode: charge a few flops. Being wrong here is bounded --
      // the comparison is against hundreds of flops of memory traffic.
      return 4;
  }
}

// Number of input elements that contribute to one output element of a
// reduction, i.e. how many times the reduction body runs per output element.
int64_t ReductionExtent(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kReduce) {
    if (!instr->shape().IsArray() || !instr->operand(0)->shape().IsArray()) {
      return 1;
    }
    const int64_t out = ShapeUtil::ElementsIn(instr->shape());
    if (out <= 0) return 1;
    return std::max<int64_t>(
        1, ShapeUtil::ElementsIn(instr->operand(0)->shape()) / out);
  }
  if (instr->opcode() == HloOpcode::kReduceWindow) {
    int64_t extent = 1;
    for (const WindowDimension& dim : instr->window().dimensions()) {
      extent *= std::max<int64_t>(1, dim.size());
    }
    return extent;
  }
  return 1;
}

}  // namespace

CpuFusionCostModel::Params CpuFusionCostModel::ParamsFromConfig(
    const HloModuleConfig& config) {
  Params params;
  const auto& debug_options = config.debug_options();
  if (debug_options.xla_cpu_fusion_machine_balance_flops_per_byte() > 0) {
    params.machine_balance_flops_per_byte =
        debug_options.xla_cpu_fusion_machine_balance_flops_per_byte();
  }
  if (debug_options.xla_cpu_fusion_cache_bytes() > 0) {
    params.cache_bytes = debug_options.xla_cpu_fusion_cache_bytes();
  }
  return params;
}

void CpuFusionCostModel::Clear() {
  per_element_flops_.clear();
  computation_flops_.clear();
  verdict_.clear();
}

int64_t CpuFusionCostModel::ComputationPerElementFlops(
    const HloComputation* computation) {
  const int64_t key = computation->unique_id();
  auto it = computation_flops_.find(key);
  if (it != computation_flops_.end()) return it->second;
  // Insert a placeholder first: reduction bodies are acyclic, but a fusion
  // computation can be re-entered through a nested call, and we must not
  // recurse forever.
  computation_flops_[key] = 0;
  int64_t total = 0;
  for (const HloInstruction* instr : computation->instructions()) {
    total += PerElementFlops(instr);
    if (total >= kProhibitive) {
      total = kProhibitive;
      break;
    }
  }
  computation_flops_[key] = total;
  return total;
}

int64_t CpuFusionCostModel::PerElementFlops(const HloInstruction* instr) {
  const int64_t key = instr->unique_id();
  auto it = per_element_flops_.find(key);
  if (it != per_element_flops_.end()) return it->second;

  int64_t flops = 0;
  switch (instr->opcode()) {
    case HloOpcode::kFusion:
      // Everything inside is recomputed together.
      flops = ComputationPerElementFlops(instr->fused_instructions_computation());
      break;
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow: {
      const int64_t body = std::max<int64_t>(
          1, ComputationPerElementFlops(instr->to_apply()));
      const int64_t extent = ReductionExtent(instr);
      flops = (body >= kProhibitive / std::max<int64_t>(1, extent))
                  ? kProhibitive
                  : body * extent;
      break;
    }
    case HloOpcode::kMap:
    case HloOpcode::kSelectAndScatter:
      flops = ComputationPerElementFlops(instr->to_apply());
      break;
    default:
      flops = OpcodeFlops(instr->opcode());
      break;
  }

  per_element_flops_[key] = flops;
  return flops;
}

int64_t CpuFusionCostModel::TotalElementReads(const HloInstruction* producer) {
  const int64_t producer_elements = ShapeUtil::ElementsIn(producer->shape());
  int64_t total = 0;
  for (const HloInstruction* user : producer->users()) {
    // A user reading each producer element once contributes
    // `producer_elements`; a user whose output is larger (broadcast-shaped
    // reuse) reads each element once per output element it appears in, which
    // the output size bounds. A user whose output is smaller -- a reduction,
    // a slice -- still reads each element at most once.
    int64_t reads = producer_elements;
    if (user->shape().IsArray()) {
      reads = std::max<int64_t>(reads, ShapeUtil::ElementsIn(user->shape()));
    }
    total += reads;
    if (total >= kProhibitive) return kProhibitive;
  }
  return std::max<int64_t>(total, producer_elements);
}

bool CpuFusionCostModel::RecomputeBeatsMaterializeForReads(
    const HloInstruction* producer, int64_t total_reads) {
  if (!producer->shape().IsArray()) return false;

  const int64_t producer_elements = ShapeUtil::ElementsIn(producer->shape());
  if (producer_elements <= 0) return false;

  // Below cache size the intermediate never reaches DRAM, so there is no
  // round trip to buy back and the duplicated arithmetic is pure loss.
  const int64_t producer_bytes = ShapeUtil::ByteSizeOfElements(producer->shape());
  if (producer_bytes < params_.cache_bytes) return false;

  const int64_t per_element = PerElementFlops(producer);
  if (per_element >= kProhibitive) return false;
  // Free to recompute: no arithmetic at all, only addressing.
  if (per_element == 0) return true;

  const int64_t element_bytes = producer_bytes / producer_elements;
  total_reads = std::max(total_reads, producer_elements);
  if (total_reads >= kProhibitive) return false;

  const int64_t balance = std::max<int64_t>(1, params_.machine_balance_flops_per_byte);

  // Recomputing costs the arithmetic we would not otherwise have redone.
  const int64_t extra_reads = total_reads - producer_elements;
  // Overflow in the recompute term can only mean "far too expensive".
  if (extra_reads > 0 && per_element > kProhibitive / extra_reads) return false;
  const int64_t extra_flops = extra_reads * per_element;

  // Not recomputing costs one write of the whole buffer plus every read of it.
  const int64_t saved_bytes = (producer_elements + total_reads) * element_bytes;
  // ...and overflow in the traffic term can only mean "well worth avoiding".
  if (saved_bytes > kProhibitive / balance) return true;

  return extra_flops <= balance * saved_bytes;
}

bool CpuFusionCostModel::RecomputeBeatsMaterialize(
    const HloInstruction* producer) {
  const int64_t key = producer->unique_id();
  auto it = verdict_.find(key);
  if (it != verdict_.end()) return it->second;
  const bool verdict =
      RecomputeBeatsMaterializeForReads(producer, TotalElementReads(producer));
  verdict_[key] = verdict;
  return verdict;
}

bool CpuFusionCostModel::RematerializationBeatsMaterialization(
    const HloInstruction* producer, int64_t emitted_copies) {
  if (!producer->shape().IsArray()) return false;
  emitted_copies = std::max<int64_t>(emitted_copies, 1);
  const int64_t producer_elements = ShapeUtil::ElementsIn(producer->shape());
  if (producer_elements <= 0) return false;
  if (emitted_copies >= kProhibitive / std::max<int64_t>(1, producer_elements)) {
    return false;
  }
  return RecomputeBeatsMaterializeForReads(producer,
                                           emitted_copies * producer_elements);
}

}  // namespace xla::cpu
