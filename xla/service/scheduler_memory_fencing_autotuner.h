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

#ifndef XLA_SERVICE_SCHEDULER_MEMORY_FENCING_AUTOTUNER_H_
#define XLA_SERVICE_SCHEDULER_MEMORY_FENCING_AUTOTUNER_H_

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/scheduler_memory_fencing.h"

namespace xla {

struct SchedulerMemoryFencingCost {
  int64_t peak_memory_bytes;
  double estimated_cycles;
};

// Evaluates at most eight distinct fence sets on independent clones of the
// reference module, including unfenced LHS. Initial thresholds are 1% of the
// limit and 1 GiB / 512 MiB / 256 MiB / 128 MiB. Two further trials refine the
// best fenced candidate using actual buffer sizes. Selects lexicographically
// by peak memory, estimated cycles, and edge count.
//
// `schedule` must create fresh scheduling state and disable internal retries.
// The winning fences are applied to the original module and `schedule` is
// invoked once more, preserving original instruction ownership and all
// scheduler mutations. Thus there are at most nine scheduling invocations.
// If there are no possible fences, schedules the original module just once.
// Errors are propagated; a failed trial never changes the original module.
// The input fences must refer to the original scheduled module and use slack 1.
absl::Status AutotuneSchedulerMemoryFencing(
    HloModule* module,
    absl::Span<const SchedulerMemoryFencing::BufferFences> buffers,
    uint64_t memory_limit,
    absl::FunctionRef<absl::StatusOr<SchedulerMemoryFencingCost>(HloModule*)>
        schedule);

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULER_MEMORY_FENCING_AUTOTUNER_H_
