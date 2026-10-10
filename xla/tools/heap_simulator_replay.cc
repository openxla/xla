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

// Replays the traced heap in the largest ordinary temporary allocation in an
// HloProto produced by --xla_dump_hlo_as_proto=true. An appended untraced
// suffix (e.g. tuple buffers) is reported separately. Does not compile or
// reschedule. Usage: heap_simulator_replay module.hlo.pb [alignment_bytes]

#include "xla/tools/heap_simulator_replay.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <map>
#include <memory>
#include <ostream>
#include <ratio>
#include <set>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::Status ReplayHeapSimulator(const HloProto& proto, int64_t alignment,
                                 std::ostream& output) {
  if (alignment <= 0) {
    return absl::InvalidArgumentError("Alignment must be positive");
  }
  const BufferAssignmentProto& assignment = proto.buffer_assignment();
  const BufferAllocationProto* allocation = nullptr;
  for (const BufferAllocationProto& candidate :
       assignment.buffer_allocations()) {
    if (candidate.color() != 0 || candidate.is_entry_computation_parameter() ||
        candidate.is_constant() || candidate.maybe_live_out() ||
        candidate.is_thread_local()) {
      continue;
    }
    if (allocation == nullptr || candidate.size() > allocation->size()) {
      allocation = &candidate;
    }
  }
  if (allocation == nullptr) {
    return absl::InvalidArgumentError("No ordinary temporary allocation found");
  }

  // Only stable IDs are used by the placement algorithms. A scalar placeholder
  // avoids constructing the module or materializing large constants. Sizes
  // come from the assignment, including the maximum size of each alias group.
  std::unique_ptr<HloInstruction> instruction = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U8, {}), "replay");
  std::map<int64_t, std::unique_ptr<HloValue>> values;
  std::map<int64_t, int64_t> sizes;
  std::map<int64_t, int64_t> offsets;
  for (const BufferAllocationProto::Assigned& assigned :
       allocation->assigned()) {
    if (assigned.size() < 0 || assigned.offset() < 0 ||
        assigned.offset() > allocation->size() ||
        assigned.size() > allocation->size() - assigned.offset()) {
      return absl::InvalidArgumentError("Buffer outside allocation");
    }
    const int64_t id = assigned.logical_buffer_id();
    if (values.find(id) != values.end()) {
      return absl::InvalidArgumentError("Duplicate assigned buffer");
    }
    values[id] =
        std::make_unique<HloValue>(id, instruction.get(), ShapeIndex{});
    sizes[id] = assigned.size();
    offsets[id] = assigned.offset();
  }
  // Match buffer IDs instead of buffer_allocation_index: older dumps can
  // retain the index from before CombineTempAllocations renumbered allocations.
  // Validate complete membership below, rejecting combined/constrained heaps.
  const HeapSimulatorTrace* trace = nullptr;
  for (const HeapSimulatorTrace& candidate :
       assignment.heap_simulator_traces()) {
    bool contains_buffer = false;
    for (const HeapSimulatorTrace::Event& event : candidate.events()) {
      if (values.find(event.buffer_id()) != values.end()) {
        contains_buffer = true;
        break;
      }
    }
    if (!contains_buffer) {
      continue;
    }
    if (trace != nullptr) {
      return absl::InvalidArgumentError(
          "Expected one trace; combined per-computation heaps are unsupported");
    }
    trace = &candidate;
  }
  if (trace == nullptr) {
    return absl::InvalidArgumentError("Allocation has no heap simulator trace");
  }
  std::set<int64_t> seen;
  std::set<int64_t> live;
  for (const HeapSimulatorTrace::Event& event : trace->events()) {
    const int64_t id = event.buffer_id();
    if (values.find(id) == values.end()) {
      return absl::InvalidArgumentError(
          "Trace crosses allocation boundaries; constrained heaps are "
          "unsupported");
    }
    if (event.kind() == HeapSimulatorTrace::Event::FREE) {
      if (live.erase(id) != 1) {
        return absl::InvalidArgumentError("FREE without a live buffer");
      }
    } else {
      if (event.kind() != HeapSimulatorTrace::Event::ALLOC &&
          event.kind() != HeapSimulatorTrace::Event::SHARE_WITH) {
        return absl::InvalidArgumentError("Unknown heap event");
      }
      if (event.kind() == HeapSimulatorTrace::Event::SHARE_WITH &&
          seen.find(event.share_with_canonical_id()) == seen.end()) {
        return absl::InvalidArgumentError("Unknown canonical buffer");
      }
      if (!seen.insert(id).second) {
        return absl::InvalidArgumentError("Buffer allocated more than once");
      }
      live.insert(id);
    }
  }
  if (!live.empty()) {
    return absl::InvalidArgumentError("Trace leaves live buffers");
  }
  int64_t traced_end = 0;
  for (int64_t id : seen) {
    traced_end = std::max(traced_end, offsets.at(id) + sizes.at(id));
  }
  // CombineTempAllocations can append allocations not handled by the heap
  // simulator, such as tuple buffers. Replay only the traced heap; do not
  // invent lifetimes for those buffers or include them in candidate sizes.
  for (const auto& [id, offset] : offsets) {
    if (seen.find(id) == seen.end() && sizes.at(id) > 0 &&
        offset < traced_end) {
      return absl::InvalidArgumentError(
          "Untraced buffers overlap the traced heap; unsupported allocation");
    }
  }

  output << "allocation=" << allocation->index()
         << " recorded_bytes=" << allocation->size()
         << " recorded_trace_bytes=" << traced_end
         << " untraced_suffix_bytes=" << allocation->size() - traced_end
         << " traced_buffers=" << seen.size()
         << " assigned_buffers=" << values.size() << " alignment=" << alignment
         << '\n';
  output
      << "| Order            | Placement     | Trace (bytes) | Time (ms) |\n"
         "| ---------------- | ------------- | ------------- | --------- |\n";
  using Heap = GlobalDecreasingSizeBestFitHeap<HloValue>;
  for (Heap::ChunkPlacement placement :
       {Heap::ChunkPlacement::kBestFit, Heap::ChunkPlacement::kLowestOffset}) {
    for (Heap::PackingStrategy strategy :
         {Heap::kSpatial, Heap::kTemporal, Heap::kSpatialTemporal}) {
      const auto start = std::chrono::steady_clock::now();
      Heap heap(alignment, strategy, nullptr,
                SliceTimePermutationIterator::Ty::kAll, placement);
      for (const HeapSimulatorTrace::Event& event : trace->events()) {
        const HloValue* value = values.at(event.buffer_id()).get();
        const int64_t size = sizes.at(event.buffer_id());
        switch (event.kind()) {
          case HeapSimulatorTrace::Event::ALLOC:
            heap.Alloc(value, size);
            break;
          case HeapSimulatorTrace::Event::FREE:
            heap.Free(value, size);
            break;
          case HeapSimulatorTrace::Event::SHARE_WITH:
            heap.ShareWith(
                value, values.at(event.share_with_canonical_id()).get(), size);
            break;
          default:
            return absl::InvalidArgumentError("Unknown heap event");
        }
      }
      ABSL_ASSIGN_OR_RETURN(const HeapSimulator::Result<HloValue> result,
                            heap.Finish());
      const double milliseconds = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
      const char* order = strategy == Heap::kSpatial    ? "size"
                          : strategy == Heap::kTemporal ? "lifetime"
                                                        : "size x duration";
      const char* policy = placement == Heap::ChunkPlacement::kBestFit
                               ? "best fit"
                               : "lowest offset";
      output << "| " << std::left << std::setw(16) << order << " | "
             << std::setw(13) << policy << " | " << std::right << std::setw(13)
             << result.heap_size << " | " << std::setw(9) << std::fixed
             << std::setprecision(2) << milliseconds << " |\n";
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
