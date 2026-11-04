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

#include "xla/hlo/analysis/hlo_reachability_from_sources.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

std::unique_ptr<HloReachabilityFromSources> HloReachabilityFromSources::Build(
    const HloComputation* computation,
    absl::Span<const HloInstruction* const> sources) {
  std::vector<HloInstruction*> post_order =
      computation->MakeInstructionPostOrder();
  auto result = absl::WrapUnique(new HloReachabilityFromSources());
  result->computation_id_ = computation->unique_id();

  int32_t max_local_id = -1;
  for (const HloInstruction* instruction : post_order) {
    max_local_id = std::max(max_local_id, instruction->local_id());
  }
  result->row_index_.assign(max_local_id + 1, kAbsent);
  result->source_index_.assign(max_local_id + 1, kAbsent);
  for (size_t row = 0; row < post_order.size(); ++row) {
    result->row_index_[post_order[row]->local_id()] = static_cast<int32_t>(row);
  }
  int32_t num_sources = 0;
  for (const HloInstruction* source : sources) {
    CHECK_EQ(source->parent(), computation)
        << source->name() << " is not in " << computation->name();
    int32_t& index = result->source_index_[source->local_id()];
    if (index == kAbsent) {
      index = num_sources++;
    }
  }
  result->words_per_row_ = (num_sources + kBitsPerWord - 1) / kBitsPerWord;
  const size_t words = result->words_per_row_;
  result->bits_.assign(post_order.size() * words, 0);
  if (words == 0) {
    return result;
  }

  // Post order lists every operand and control predecessor before its user,
  // so each row is complete when it is read.
  std::vector<uint64_t>& bits = result->bits_;
  for (size_t row = 0; row < post_order.size(); ++row) {
    const HloInstruction* instruction = post_order[row];
    uint64_t* own = &bits[row * words];
    const int32_t source = result->source_index_[instruction->local_id()];
    if (source != kAbsent) {
      own[source / kBitsPerWord] |= uint64_t{1} << (source % kBitsPerWord);
    }
    auto merge = [&](const HloInstruction* input) {
      const int32_t input_row = result->Row(input);
      DCHECK(input_row >= 0 && static_cast<size_t>(input_row) < row)
          << input->name() << " must precede " << instruction->name();
      const uint64_t* input_bits = &bits[input_row * words];
      for (size_t w = 0; w < words; ++w) {
        own[w] |= input_bits[w];
      }
    };
    for (const HloInstruction* operand : instruction->operands()) {
      merge(operand);
    }
    for (const HloInstruction* predecessor :
         instruction->control_predecessors()) {
      merge(predecessor);
    }
  }
  return result;
}

bool HloReachabilityFromSources::IsReachable(
    const HloInstruction* source, const HloInstruction* instruction) const {
  const int32_t index = SourceIndex(source);
  CHECK_NE(index, kAbsent) << source->name() << " is not a source";
  const int32_t row = Row(instruction);
  CHECK_NE(row, kAbsent) << instruction->name() << " is not in the analysis";
  const uint64_t word = bits_[row * words_per_row_ + index / kBitsPerWord];
  return (word & (uint64_t{1} << (index % kBitsPerWord))) != 0;
}

int32_t HloReachabilityFromSources::Lookup(
    const std::vector<int32_t>& by_local_id,
    const HloInstruction* instruction) const {
  if (instruction == nullptr || instruction->parent() == nullptr ||
      instruction->parent()->unique_id() != computation_id_) {
    return kAbsent;
  }
  const int32_t local_id = instruction->local_id();
  if (local_id < 0 || static_cast<size_t>(local_id) >= by_local_id.size()) {
    return kAbsent;
  }
  return by_local_id[local_id];
}

}  // namespace xla
