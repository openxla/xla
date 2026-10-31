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

#ifndef XLA_SERVICE_ASYNC_COMPUTATION_UTILS_H_
#define XLA_SERVICE_ASYNC_COMPUTATION_UTILS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Returns whether the given computation should contain only parameter
// instructions and a single root.
bool AsyncComputationRequiresSingleInstruction(
    const HloComputation& computation);

// For computations meant to be async and that need to be single-instruction but
// are not, insert a new computation that wraps the invalid computation in a
// call instruction, and have async callers of the original computation call
// into the new computation. Returns whether any wrapper was added.
// An empty execution_threads set includes all threads; only bodies subject to
// AsyncComputationRequiresSingleInstruction are legalized. Intended for use
// before HLO scheduling; does not update existing schedules.
absl::StatusOr<bool> LegalizeAsyncComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads = {});

}  // namespace xla

#endif  // XLA_SERVICE_ASYNC_COMPUTATION_UTILS_H_
