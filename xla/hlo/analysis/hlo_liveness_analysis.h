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

#ifndef XLA_HLO_ANALYSIS_HLO_LIVENESS_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_HLO_LIVENESS_ANALYSIS_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"
#include "xla/shape_util.h"
#include "xla/tuple_tree.h"

namespace xla {

// Analysis which identifies all live {HloInstruction, ShapeIndex} pairs in
// an HLO module.
//
// HloLivenessAnalysis marks the shape index of each live output of each
// instruction in the module, by propagating live shape index information
// from an instruction to its called computations and operands.
class HloLivenessAnalysis {
 public:
  // Runs liveness analysis on 'module'. Returns HloLivenessAnalysis object
  // which exports liveness for each {HloInstruction, ShapeIndex} in 'module'.
  static absl::StatusOr<std::unique_ptr<HloLivenessAnalysis>> Run(
      const HloModule& module);

  // Returns true if output of 'instruction' at 'shape_index' is live.
  // Returns false otherwise, also for an instruction that was not in the
  // module when the analysis ran or that was removed since. The answers hold
  // until instructions are deleted, as a new instruction may take the address
  // of a deleted one and read the deleted one's answer.
  bool IsLive(const HloInstruction* instruction,
              const ShapeIndex& shape_index) const;

 private:
  // Live output indices of one instruction. An array shape has the single bit
  // `live`; a tuple shape has one bit per subshape in `live_subshapes`,
  // allocated when its first index becomes live.
  struct InstructionLiveness {
    const HloInstruction* instruction = nullptr;
    bool live = false;
    bool on_worklist = false;
    std::unique_ptr<TupleTree<bool>> live_subshapes;
  };
  using EntriesByInstruction =
      absl::flat_hash_map<const HloInstruction*, const InstructionLiveness*>;

  // Liveness of the instructions of one computation, indexed by local id.
  struct ComputationLiveness {
    std::vector<InstructionLiveness> entries;
    // The marks PropagateLivenessThroughControlFlow makes depend on the
    // computation alone, so they run once.
    bool control_flow_propagated = false;
  };

  explicit HloLivenessAnalysis(const HloModule& module);

  void RunAnalysis();

  // Returns the liveness of 'instruction', or nullptr if 'instruction' was not
  // in the module when the analysis ran or has been removed since.
  const InstructionLiveness* FindLiveness(
      const HloInstruction* instruction) const;
  // Like FindLiveness for an instruction of the module, which is all
  // RunAnalysis reaches, so it indexes the table without the checks.
  InstructionLiveness& GetLiveness(const HloInstruction* instruction);

  void AddToWorklist(InstructionLiveness& liveness);
  void MarkLiveAtIndex(const HloInstruction* instruction,
                       const ShapeIndex& shape_index);
  void MarkLiveAtAllIndices(const HloInstruction* instruction);
  static void ForEachLiveIndex(const InstructionLiveness& liveness,
                               absl::FunctionRef<void(const ShapeIndex&)> func);

  void PropagateLivenessThroughTuple(const InstructionLiveness& liveness);
  void PropagateLivenessThroughGTE(const InstructionLiveness& liveness);
  void PropagateLivenessThroughWhile(const InstructionLiveness& liveness);
  void PropagateLivenessThroughCall(const InstructionLiveness& liveness);
  void PropagateLivenessToParameterCallers(const InstructionLiveness& liveness);
  void PropagateLivenessThroughControlFlow(const InstructionLiveness& liveness);

  const HloModule& module_;
  std::unique_ptr<CallGraph> call_graph_;
  // Indexed by computation unique id, so sized by the largest id the module
  // has assigned. Sized once by RunAnalysis before the first mark, so the
  // entry pointers the worklist holds stay valid.
  std::vector<ComputationLiveness> table_;
  std::deque<InstructionLiveness*> worklist_;
  // The entries of each computation by instruction, keyed by unique id and
  // built on the computation's first slot miss. The mutex keeps IsLive, a
  // const method, safe to call concurrently.
  mutable absl::Mutex entries_by_instruction_mutex_;
  mutable absl::flat_hash_map<int64_t, EntriesByInstruction>
      entries_by_instruction_ ABSL_GUARDED_BY(entries_by_instruction_mutex_);
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_LIVENESS_ANALYSIS_H_
