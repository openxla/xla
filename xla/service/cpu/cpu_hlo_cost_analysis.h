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

#ifndef XLA_SERVICE_CPU_CPU_HLO_COST_ANALYSIS_H_
#define XLA_SERVICE_CPU_CPU_HLO_COST_ANALYSIS_H_

#include <cstdint>
#include <memory>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Cost analysis for CPUs.
//
// Differs from HloCostAnalysis in two ways, both following GpuHloCostAnalysis:
// elementwise ops are charged a per-opcode number of flops instead of one, and
// operand utilization inside a fusion accounts for elements that are read
// more than once (e.g. through a broadcast).
class CpuHloCostAnalysis : public HloCostAnalysis {
 public:
  explicit CpuHloCostAnalysis(const Options& options)
      : HloCostAnalysis(options) {}

  absl::Status HandleElementwiseOp(const HloInstruction* hlo) override;

  // Returns the number of FLOPs needed to compute an element of the given
  // elementwise instruction.
  int64_t GetFlopsPerElementwiseOpElement(PrimitiveType type,
                                          HloOpcode opcode) const;

  // Returns the number of FLOPs needed to compute the output of the elementwise
  // instruction.
  int64_t GetFlopsForElementwiseOp(HloOpcode op_code, const Shape& shape) const;
  int64_t GetFlopsForElementwiseOp(const HloInstruction* instr) const;

  // Total common elementwise utilization of two instructions within a fusion.
  // See GpuHloCostAnalysis::CommonElementwiseUtilization.
  float CommonElementwiseUtilization(const HloInstruction* a,
                                     const HloInstruction* b) const;

 protected:
  std::unique_ptr<HloCostAnalysis> CreateNestedCostAnalysis() override;
  int64_t FusionParameterReadBytes(const HloInstruction* hlo) const override;
  absl::Status FusionCalculateUtilizations(
      const HloInstruction* fusion) override;

  // For each fused instruction, the roots from which it is reached through
  // elementwise ops only. See GpuHloCostAnalysis.
  absl::flat_hash_map<const HloInstruction*,
                      absl::btree_set<const HloInstruction*, HloPtrComparator>>
      elementwise_use_roots_;

  // Elementwise utilization of instruction's input subtree if it is a root.
  absl::flat_hash_map<const HloInstruction*, float> root_utilizations_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_HLO_COST_ANALYSIS_H_
