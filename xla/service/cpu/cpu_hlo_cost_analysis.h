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
// Follows the structure of GpuHloCostAnalysis:
// elementwise ops cost a per-opcode number of flops, and operand utilization
// inside a fusion accounts for elements that are read more than once (broadcast).
class CpuHloCostAnalysis : public HloCostAnalysis {
 public:
  explicit CpuHloCostAnalysis(const Options& options)
      : HloCostAnalysis(options) {}

  absl::Status HandleElementwiseOp(const HloInstruction* hlo) override;

  int64_t GetFlopsPerElementwiseOpElement(PrimitiveType type,
                                          HloOpcode opcode) const;

  int64_t GetFlopsForElementwiseOp(HloOpcode op_code, const Shape& shape) const;
  int64_t GetFlopsForElementwiseOp(const HloInstruction* instr) const;

  float CommonElementwiseUtilization(const HloInstruction* a,
                                     const HloInstruction* b) const;

 protected:
  std::unique_ptr<HloCostAnalysis> CreateNestedCostAnalysis() override;
  int64_t FusionParameterReadBytes(const HloInstruction* hlo) const override;
  absl::Status FusionCalculateUtilizations(
      const HloInstruction* fusion) override;

  absl::flat_hash_map<const HloInstruction*,
                      absl::btree_set<const HloInstruction*, HloPtrComparator>>
      elementwise_use_roots_;

  absl::flat_hash_map<const HloInstruction*, float> root_utilizations_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_HLO_COST_ANALYSIS_H_
