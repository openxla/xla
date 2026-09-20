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

#ifndef XLA_HLO_ANALYSIS_HLO_REACHABILITY_FROM_SOURCES_H_
#define XLA_HLO_ANALYSIS_HLO_REACHABILITY_FROM_SOURCES_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Reachability from a fixed set of source instructions to every instruction
// of one computation. Answers HloReachabilityMap::IsReachable(source,
// instruction) for those sources only. The table holds one bit per
// (instruction, source) pair: n * ceil(k / 64) words for n instructions and
// k sources. It is filled in one pass over the post order, not the n by n
// closure of HloReachabilityMap. Prefer HloDfsReachability for a few queries
// between instructions that are close in the post order: its per query
// search can visit most of the computation when they are far apart.
//
// The table is a snapshot of the computation when Build ran. Instructions
// and edges added or removed later are not reflected, and
// HloComputation::Cleanup renumbers the local ids the table is keyed by.
class HloReachabilityFromSources {
 public:
  // Every source must belong to `computation` (CHECKed). A source listed
  // twice is accepted.
  static std::unique_ptr<HloReachabilityFromSources> Build(
      const HloComputation* computation,
      absl::Span<const HloInstruction* const> sources);

  // True iff a directed path of operand or control edges leads from `source`
  // to `instruction`, or the two are the same instruction. `source` must be
  // one of the sources given to Build and `instruction` must have been in
  // the computation at Build; both are CHECKed. IsSource and IsPresent
  // answer those two preconditions.
  bool IsReachable(const HloInstruction* source,
                   const HloInstruction* instruction) const;

  bool IsSource(const HloInstruction* instruction) const {
    return SourceIndex(instruction) != kAbsent;
  }

  bool IsPresent(const HloInstruction* instruction) const {
    return Row(instruction) != kAbsent;
  }

 private:
  static constexpr int32_t kAbsent = -1;
  static constexpr int32_t kBitsPerWord = 64;

  HloReachabilityFromSources() = default;

  // Position of `instruction` among the sources, or kAbsent.
  int32_t SourceIndex(const HloInstruction* instruction) const {
    return Lookup(source_index_, instruction);
  }
  // Row of `instruction` in bits_, or kAbsent.
  int32_t Row(const HloInstruction* instruction) const {
    return Lookup(row_index_, instruction);
  }
  int32_t Lookup(const std::vector<int32_t>& by_local_id,
                 const HloInstruction* instruction) const;

  int64_t computation_id_ = -1;
  // Both keyed by HloInstruction::local_id().
  std::vector<int32_t> source_index_;
  std::vector<int32_t> row_index_;
  // Row r holds the sources that reach the instruction of row r, one bit per
  // source, in words_per_row_ words.
  size_t words_per_row_ = 0;
  std::vector<uint64_t> bits_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_REACHABILITY_FROM_SOURCES_H_
