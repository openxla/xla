/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_UTIL_H_
#define XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {
class BufferAssignment;
class HloModule;
}  // namespace xla

namespace xla::cpu {

// Creates and returns a list of `BufferAllocationInfo` instances containing
// relevant information from `buffer_assignment`.
template <typename HloModuleT = HloModule,
          typename BufferAssignmentT = BufferAssignment>
std::vector<BufferAllocationInfo> CreateBufferAllocationInfos(
    const HloModuleT& module, const BufferAssignmentT& buffer_assignment) {
  std::vector<BufferAllocationInfo> allocations;

  // A mapping from a buffer allocation index to the result parameter number.
  absl::flat_hash_map<int64_t, int64_t> result_allocations;
  const auto* root = module.entry_computation()->root_instruction();
  ShapeUtil::ForEachLeafShape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        int64_t allocation_index =
            buffer_assignment.GetUniqueSlice(root, index)->index();
        int64_t result_index = result_allocations.size();
        result_allocations[allocation_index] = result_index;
      });

  for (const auto& allocation : buffer_assignment.Allocations()) {
    // Check that the allocations index is contiguous in [0, num_allocations).
    DCHECK_EQ(allocation.index(), allocations.size());

    if (allocation.is_thread_local()) {
      allocations.push_back(
          BufferAllocationInfo::ThreadLocal(allocation.size()));

    } else if (allocation.is_constant()) {
      allocations.push_back(BufferAllocationInfo::Constant(allocation.size()));

    } else if (allocation.is_entry_computation_parameter() &&
               allocation.maybe_live_out()) {
      // Entry computation parameter that is aliased with one of the results.
      allocations.push_back(BufferAllocationInfo::InOutParameter(
          allocation.size(), allocation.parameter_number(),
          result_allocations.at(allocation.index())));

    } else if (allocation.is_entry_computation_parameter()) {
      // A read-only entry computation parameter.
      allocations.push_back(BufferAllocationInfo::EntryParameter(
          allocation.size(), allocation.parameter_number()));

    } else if (allocation.maybe_live_out() &&
               result_allocations.contains(allocation.index())) {
      // This is a result buffer that corresponds to a flatten result index.
      allocations.push_back(BufferAllocationInfo::Result(
          allocation.size(), result_allocations[allocation.index()]));

    } else if (allocation.maybe_live_out()) {
      // This is a result buffer that holds the tuple. It doesn't correspond to
      // a flatten result index, and it's never used by XLA:CPU at run time, but
      // we still record it as we want to know about all allocations.
      allocations.push_back(
          BufferAllocationInfo::Result(allocation.size(), -1));

    } else {
      // A temporary allocation that holds intermediate buffers.
      DCHECK(allocation.IsPreallocatedTempBuffer());
      allocations.push_back(BufferAllocationInfo::Temp(allocation.size()));
    }
  }

  return allocations;
}

std::vector<BufferAllocationInfo> CreateBufferAllocationInfos(
    const HloModuleProto& module,
    const BufferAssignmentProto& buffer_assignment);

absl::StatusOr<ShapeTree<int64_t>> CreateResultAllocationIndexTree(
    const HloModuleProto& module,
    const BufferAssignmentProto& buffer_assignment);

// Creates and returns a table containing the mapping from entry computation
// parameters to buffer allocation indices:
//
//   vector[parameter_number] == allocation.index()
//   vector[result_number]    == allocation.index()
//
std::vector<int32_t> CreateArgIndexTable(
    absl::Span<const BufferAllocationInfo> allocations);
std::vector<int32_t> CreateResultIndexTable(
    absl::Span<const BufferAllocationInfo> allocations);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_UTIL_H_
