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

#ifndef XLA_SERVICE_ASYNC_CUSTOM_CALL_UTILS_H_
#define XLA_SERVICE_ASYNC_CUSTOM_CALL_UTILS_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Returns the corresponding "-start" custom-call target for a "-done" target,
// or std::nullopt if `done_target` does not end with "-done".
std::optional<std::string> MatchingStartTarget(absl::string_view done_target);

// Finds the minimal set of inlineable kCall instructions along interprocedural
// async intermediary paths from "-done" custom calls matching `is_done` to
// their corresponding "-start" custom calls within `execution_threads`.
absl::flat_hash_set<const HloInstruction*> FindCallsAlongAsyncPaths(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<bool(const HloInstruction*)> is_done);

// Inlines the minimal set of kCall instructions needed so that matched
// "-start"/"-done" custom-call pairs and their async intermediaries reside in
// the same computation. Returns true if any call instructions were inlined.
absl::StatusOr<bool> InlineCallsAlongAsyncPaths(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<bool(const HloInstruction*)> is_done);

// Completes rewriting a custom-call start/done pair into `async_start` and
// `async_done` (with `final_result` replacing uses of `done_call`), propagating
// dataflow along `forward_path`, transferring metadata and control
// dependencies, and removing the original custom-call instructions.
absl::Status FinishRewrite(
    HloComputation* computation, HloInstruction* start_call,
    HloInstruction* done_call, HloInstruction* async_start,
    HloInstruction* async_done,
    absl::Span<const hlo_instruction_utils::async::AsyncTraceStep> forward_path,
    HloInstruction* final_result, bool propagate_annotations = true);

}  // namespace xla

#endif  // XLA_SERVICE_ASYNC_CUSTOM_CALL_UTILS_H_
