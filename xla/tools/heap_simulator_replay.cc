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

// Replays the largest ordinary temporary allocation in an HloProto produced by
// --xla_dump_hlo_as_proto=true. Does not compile or reschedule the HLO.
// Usage: heap_simulator_replay module.hlo.pb [alignment_bytes]

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <ratio>
#include <set>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/numbers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

absl::Status Replay(const std::string& path, int64_t alignment) {
  HloProto proto;
  ABSL_RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(), path, &proto));
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
  for (const BufferAllocationProto::Assigned& assigned :
       allocation->assigned()) {
    if (assigned.size() < 0) {
      return absl::InvalidArgumentError("Negative buffer size");
    }
    const int64_t id = assigned.logical_buffer_id();
    values[id] =
        std::make_unique<HloValue>(id, instruction.get(), ShapeIndex{});
    sizes[id] = assigned.size();
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
  if (!live.empty() || seen.size() != values.size()) {
    return absl::InvalidArgumentError(
        "Trace does not cover the full allocation");
  }

  std::cout << "allocation=" << allocation->index()
            << " recorded_bytes=" << allocation->size()
            << " buffers=" << values.size() << " alignment=" << alignment
            << '\n';
  std::cout
      << "| Order            | Placement     | Arena (bytes) | Time (ms) |\n"
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
      std::cout << "| " << std::left << std::setw(16) << order << " | "
                << std::setw(13) << policy << " | " << std::right
                << std::setw(13) << result.heap_size << " | " << std::setw(9)
                << std::fixed << std::setprecision(2) << milliseconds << " |\n";
    }
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  int64_t alignment = 256;
  if (argc < 2 || argc > 3 ||
      (argc == 3 && !absl::SimpleAtoi(argv[2], &alignment)) || alignment <= 0) {
    std::cerr
        << "Usage: heap_simulator_replay module.hlo.pb [alignment_bytes]\n";
    return 1;
  }
  const absl::Status status = xla::Replay(argv[1], alignment);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
