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

#ifndef XLA_SERVICE_POST_LHS_MEMORY_SCHEDULER_H_
#define XLA_SERVICE_POST_LHS_MEMORY_SCHEDULER_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/latency_hiding_scheduler.h"

namespace xla {

// Reduces the peak of an LHS schedule by recursively scheduling siblings.
// Async regions retain their membership, but are movable atomic operators in
// their parent. Crossing intervals are kept as opaque composite regions.
// Changes must preserve the async event order and each instruction's enclosing
// async operations, and reduce whole-module modeled peak without increasing
// LHS's estimated cycles or exposed latency. This is a bounded heuristic, not
// an optimal scheduler or a guarantee about allocator fragmentation/runtime.
class PostLhsMemoryScheduler : public HloModulePass {
 public:
  explicit PostLhsMemoryScheduler(
      std::shared_ptr<const SchedulingContext> scheduling_context,
      int64_t max_candidate_evaluations = 64)
      : scheduling_context_(std::move(scheduling_context)),
        max_candidate_evaluations_(max_candidate_evaluations) {}

  absl::string_view name() const override {
    return "post-lhs-memory-scheduler";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::shared_ptr<const SchedulingContext> scheduling_context_;
  int64_t max_candidate_evaluations_;
};

}  // namespace xla

#endif  // XLA_SERVICE_POST_LHS_MEMORY_SCHEDULER_H_
