/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HOST_OFFLOAD_UTILS_H_
#define XLA_SERVICE_HOST_OFFLOAD_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"

namespace xla {

class CallGraph;

namespace host_offload_utils {

struct InstructionAndShapeIndex {
  explicit InstructionAndShapeIndex(HloInstruction* instruction)
      : instruction(instruction) {}
  InstructionAndShapeIndex(HloInstruction* instruction, ShapeIndex shape_index)
      : instruction(instruction), shape_index(shape_index) {}
  HloInstruction* instruction;
  ShapeIndex shape_index;
  std::string ToString() const;

  template <typename H>
  static H Hash(H h, const InstructionAndShapeIndex& i) {
    h = H::combine(std::move(h), i.instruction);
    h = H::combine(std::move(h), i.shape_index);
    return std::move(h);
  }

  template <typename H>
  friend H AbslHashValue(H h, const InstructionAndShapeIndex& i) {
    return InstructionAndShapeIndex::Hash(std::move(h), i);
  }
};

bool operator==(const InstructionAndShapeIndex& lhs,
                const InstructionAndShapeIndex& rhs);

// If an instruction's user is a call, we descend into the call first.
// Eventually, a later invocation of this function while walking the graph will
// return the call itself as a successor of the ROOT instruction of the
// computation. The call sites of the root's computation are read from
// call_graph, the graph of the instruction's module: a graph built before a
// walk stays valid as long as no computation and no instruction with called
// computations is added, removed or reordered.
absl::StatusOr<std::vector<InstructionAndShapeIndex>> GetSuccessors(
    const InstructionAndShapeIndex& instruction_and_shape_index,
    const CallGraph& call_graph);

// If an instruction's operand is a call, return the call now. A follow up call
// of this function on that call returns the ROOT. Eventually, once the given
// instruction is a parameter, the returned predecessor will be the appropriate
// operand of the call (not the call itself, since we already returned it). The
// call sites of the parameter's computation are read from call_graph, under
// the same condition as for GetSuccessors.
std::vector<InstructionAndShapeIndex> GetPredecessors(
    const InstructionAndShapeIndex& instruction_and_shape_index,
    const CallGraph& call_graph,
    std::optional<int64_t> operand_index = std::nullopt);

// Returns true if the instruction is allowed to be in the
// middle of a pure memory offload path.
bool IsValidDuringPureMemoryOffload(const HloInstruction* instruction);

// Returns true if the instruction is an async-start with host thread.
bool IsHostAsyncStart(const HloInstruction* instruction);

// Returns true if the copy is from or to host memory space.
bool IsSynchronousCopyFromOrToHost(const HloInstruction* instruction);

bool ComputeTypeIsHost(const HloInstruction* hlo_instruction);

// Sets the frontend attribute of the instruction to indicate that the
// instruction should be lowered as host compute.
void SetHostComputeFrontendAttribute(HloInstruction& host_instruction);

bool IsMoveToHostWithDynamicUpdateSlice(const HloInstruction* instr);

bool IsMoveToDeviceWithDynamicSlice(const HloInstruction* instr);

// Returns the set of `while_loop` tuple-element indices that flow into the
// index operands of DS/DUS instructions in the loop body, restricted to bodies
// that contain host-offload move-to-host/move-to-device custom-call patterns.
// Returns an empty set otherwise. Pure analysis; does not mutate the module.
absl::flat_hash_set<int64_t> CollectDynamicVariableTupleIndices(
    const HloInstruction* while_loop);

// Operand edges `(user, operand_index)` where a MoveToHost custom call stood
// before HostOffloader removed it.
using MovedToHostOperands =
    absl::flat_hash_set<std::pair<const HloInstruction*, int64_t>>;

// Returns true if every element of the array at `value` is either
// uninitialized storage or data annotated with MoveToHost, so writing the
// whole array to host memory offloads nothing the user kept on device.
bool HoldsOnlyDataMovedToHost(
    const InstructionAndShapeIndex& value, const CallGraph& call_graph,
    const MovedToHostOperands& moved_to_host_operands = {});

// Follows the value at `start` through instructions that forward a buffer
// unchanged (bitcasts, copies, tuples and loop or call boundaries) and returns
// the dynamic-update-slices which receive it as their update operand. Returns
// nullopt unless every path ends at such an update. With
// `through_in_place_updates`, a dynamic-update-slice which receives the value
// as its buffer operand updates it in place, so the walk continues from its
// result.
absl::StatusOr<std::optional<std::vector<InstructionAndShapeIndex>>>
FindDynamicUpdateSlicesReceivingUpdate(const InstructionAndShapeIndex& start,
                                       const CallGraph& call_graph,
                                       bool through_in_place_updates);

// Returns the dynamic-update-slices at which an offload reaching
// `dynamic_update_slice` should start. A nested scan stacks its residual
// twice: the inner loop fills a block buffer, and the outer loop body writes
// that whole block as the update of a dynamic-update-slice into the stacked
// buffer. Offloading at the outer dynamic-update-slice keeps the block buffer
// on device for one outer iteration and moves every element to host once.
absl::StatusOr<std::vector<InstructionAndShapeIndex>>
GetOutermostStackingDynamicUpdateSlices(
    const InstructionAndShapeIndex& dynamic_update_slice,
    const CallGraph& call_graph,
    const MovedToHostOperands& moved_to_host_operands = {});

// Returns true if every path from `slice` (`slice` or `dynamic-slice` on host
// memory) reaches a MoveToDevice custom call through buffer forwarding ops or
// nested slices of the resulting array.
absl::StatusOr<bool> SliceLeadsToMoveToDeviceCustomCall(
    HloInstruction* slice, const CallGraph& call_graph);

}  // namespace host_offload_utils
}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOAD_UTILS_H_
