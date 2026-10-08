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

#ifndef XLA_SERVICE_HLO_EARLY_BUFFER_RELEASE_H_
#define XLA_SERVICE_HLO_EARLY_BUFFER_RELEASE_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/buffer_value.h"

// A schedule-only optimization: finish bounded groups of remaining consumers
// early to shorten the lifetimes of large intermediate buffers. The graph and
// the relative order of instructions outside each group are unchanged.
namespace xla {

class HloEarlyBufferRelease : public HloModulePass {
 public:
  struct Options {
    int64_t min_buffer_bytes = 1 << 20;
    int64_t max_group_size = 8;
    int64_t max_candidates = 32;
    int64_t max_iterations = 4;
    // Compared with the original schedule, not the previously accepted move.
    double max_relative_slowdown = 0.01;
  };

  // Estimate elapsed time, including stalls, in consistent units. A backend
  // should provide its scheduling cost model, not the sum of instruction costs
  // (which is invariant under reordering).
  using ScheduleCost = std::function<double(
      const HloComputation*, absl::Span<const HloInstruction* const>)>;

  HloEarlyBufferRelease(const AliasInfo* alias_info,
                        BufferValue::SizeFunction size_function,
                        ScheduleCost schedule_cost, Options options)
      : alias_info_(alias_info),
        size_function_(std::move(size_function)),
        schedule_cost_(std::move(schedule_cost)),
        options_(options) {}

  absl::string_view name() const override { return "hlo-early-buffer-release"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const AliasInfo* alias_info_;
  BufferValue::SizeFunction size_function_;
  ScheduleCost schedule_cost_;
  Options options_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_EARLY_BUFFER_RELEASE_H_
