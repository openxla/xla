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

#include "xla/service/gpu/scheduler_memory_fencing_fallback.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/logical_buffer.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

struct AllocationSizes {
  int64_t temporary = 0;
  int64_t largest_temporary = 0;
  int64_t total = 0;
};

absl::flat_hash_map<LogicalBuffer::Color, AllocationSizes> GetAllocationSizes(
    const BufferAssignment& assignment) {
  absl::flat_hash_map<LogicalBuffer::Color, AllocationSizes> sizes;
  for (const BufferAllocation& allocation : assignment.Allocations()) {
    AllocationSizes& size = sizes[allocation.color()];
    size.total += allocation.size();
    if (allocation.IsPreallocatedTempBuffer()) {
      size.temporary += allocation.size();
      size.largest_temporary =
          std::max(size.largest_temporary, allocation.size());
    }
  }
  return sizes;
}

}  // namespace

absl::StatusOr<BufferAssignedModule> ScheduleWithMemoryFencingFallback(
    std::unique_ptr<HloModule> module,
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<BufferAssignment>>(HloModule*)>
        prepare) {
  const DebugOptions& options = module->config().debug_options();
  const int64_t threshold =
      options.has_xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes()
          ? options
                .xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes()
          : -1;
  std::unique_ptr<HloModule> unfenced;
  if (threshold >= 0 && !module->has_schedule() &&
      !(options.has_xla_gpu_enable_latency_hiding_scheduler() &&
        !options.xla_gpu_enable_latency_hiding_scheduler())) {
    // Clone before scheduling: removing control edges from the fenced result
    // would not undo LHS decisions, copies, or rematerialization.
    unfenced = module->Clone(/*suffix=*/"");
    unfenced->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes(-1);
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                        prepare(module.get()));
  if (unfenced == nullptr) {
    return BufferAssignedModule{std::move(module), std::move(assignment)};
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> unfenced_assignment =
      prepare(unfenced.get());
  if (!unfenced_assignment.ok()) {
    LOG(WARNING) << "Could not compare scheduler memory fencing against an "
                    "unfenced buffer assignment; retaining fenced module: "
                 << unfenced_assignment.status();
    return BufferAssignedModule{std::move(module), std::move(assignment)};
  }
  // Preserve the user's options in the selected executable and its dumps.
  unfenced->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes(
          threshold);

  auto unfenced_sizes = GetAllocationSizes(**unfenced_assignment);
  for (const auto& [color, fenced_size] : GetAllocationSizes(*assignment)) {
    const AllocationSizes& unfenced_size = unfenced_sizes[color];
    // Compare the final allocations, including packing/alignment, rather than
    // LHS's live-memory estimate. Keep colors separate so a reduction in host
    // memory cannot hide an increase in device memory (or vice versa).
    if (fenced_size.temporary > unfenced_size.temporary ||
        fenced_size.largest_temporary > unfenced_size.largest_temporary ||
        fenced_size.total > unfenced_size.total) {
      VLOG(1) << "Rejecting scheduler memory fencing for " << module->name()
              << " in memory color " << color
              << " (fenced/unfenced): temporary bytes " << fenced_size.temporary
              << "/" << unfenced_size.temporary
              << ", largest temporary allocation "
              << fenced_size.largest_temporary << "/"
              << unfenced_size.largest_temporary << ", total bytes "
              << fenced_size.total << "/" << unfenced_size.total;
      return BufferAssignedModule{std::move(unfenced),
                                  std::move(*unfenced_assignment)};
    }
  }
  VLOG(1) << "Retaining scheduler memory fencing for " << module->name()
          << ": final buffer assignments do not regress.";
  return BufferAssignedModule{std::move(module), std::move(assignment)};
}

}  // namespace xla::gpu
