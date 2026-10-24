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

#ifndef XLA_SERVICE_GPU_SCHEDULER_MEMORY_FENCING_FALLBACK_H_
#define XLA_SERVICE_GPU_SCHEDULER_MEMORY_FENCING_FALLBACK_H_

#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {

// The assignment refers to instructions owned by the module. Keep them
// together, and destroy the assignment first.
struct BufferAssignedModule {
  std::unique_ptr<HloModule> module;
  std::unique_ptr<BufferAssignment> buffer_assignment;
};

// `prepare` must schedule the module, run all post-scheduling HLO
// transformations, and assign buffers, without emitting code. Autotuning must
// precede this call.
//
// When memory fencing is enabled, also prepares an unfenced clone of the input.
// Falls back to that clone if fencing increases temporary bytes, the largest
// temporary allocation, or total allocation bytes in any memory color. Ties
// keep the fenced candidate. The caller must use the returned assignment for
// code generation without rerunning HLO transformations or buffer assignment.
//
// A failed fenced preparation is propagated. If only the unfenced preparation
// fails, retains the successfully prepared fenced candidate.
absl::StatusOr<BufferAssignedModule> ScheduleWithMemoryFencingFallback(
    std::unique_ptr<HloModule> module,
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<BufferAssignment>>(HloModule*)>
        prepare);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_SCHEDULER_MEMORY_FENCING_FALLBACK_H_
