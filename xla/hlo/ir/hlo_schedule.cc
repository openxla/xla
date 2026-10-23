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

#include "xla/hlo/ir/hlo_schedule.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {

/* static */ absl::StatusOr<HloSchedule> HloSchedule::CreateFromProto(
    const HloModule* module, const HloScheduleProto& proto,
    const absl::flat_hash_map<int64_t, absl::flat_hash_map<int64_t, int64_t>>*
        computation_id_to_instruction_id_remap) {
  absl::flat_hash_map<int64_t, const HloComputation*> id_to_computation;
  for (const HloComputation* computation : module->computations()) {
    id_to_computation[computation->unique_id()] = computation;
  }

  HloSchedule schedule(module);
  for (const auto& id_sequence : proto.sequences()) {
    int64_t computation_id = id_sequence.first;

    auto comp_it = id_to_computation.find(computation_id);
    // Computation could have been removed if unused, so
    // skip if not found.
    if (comp_it == id_to_computation.end()) {
      continue;
    }
    const HloComputation* computation = comp_it->second;

    absl::flat_hash_map<int64_t, HloInstruction*> id_to_instruction;
    for (HloInstruction* instruction : computation->instructions()) {
      id_to_instruction[instruction->unique_id()] = instruction;
    }

    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);
    if (computation_id_to_instruction_id_remap != nullptr) {
      TF_RET_CHECK(
          computation_id_to_instruction_id_remap->contains(computation_id))
          << "Computation id " << computation_id
          << " not found in computation_id_to_instruction_id_remap";
    }

    for (const int64_t instruction_id : id_sequence.second.instruction_ids()) {
      int64_t corrected_instruction_id = instruction_id;
      if (computation_id_to_instruction_id_remap != nullptr) {
        TF_RET_CHECK(computation_id_to_instruction_id_remap->at(computation_id)
                         .contains(instruction_id))
            << "Instruction id " << instruction_id
            << " not found in its computation's proto_id_to_instruction_id_map";
        corrected_instruction_id =
            computation_id_to_instruction_id_remap->at(computation_id)
                .at(instruction_id);
      }
      int64_t complete_unique_id = HloInstruction::CalculateUniqueId(
          computation->unique_id(), corrected_instruction_id);
      auto instr_it = id_to_instruction.find(complete_unique_id);
      TF_RET_CHECK(instr_it != id_to_instruction.end())
          << "No instruction exists in HLO computation " << computation->name()
          << " with unique id " << corrected_instruction_id
          << " (complete unique id " << complete_unique_id << ")";
      sequence.push_back(instr_it->second);
    }
  }
  ABSL_RETURN_IF_ERROR(schedule.Verify());
  return schedule;
}

absl::StatusOr<HloScheduleProto> HloSchedule::ToProto() const {
  ABSL_RETURN_IF_ERROR(Verify());
  HloScheduleProto proto;
  for (const auto& id_sequence : sequences_) {
    int64_t computation_id = id_sequence.first;
    const HloInstructionSequence& sequence = id_sequence.second;
    HloScheduleProto::InstructionSequence& proto_sequence =
        (*proto.mutable_sequences())[computation_id];
    proto_sequence.mutable_instruction_ids()->Reserve(sequence.size());
    for (const int64_t id : sequence.ids()) {
      proto_sequence.add_instruction_ids(id);
    }
  }
  return proto;
}

void HloSchedule::set_sequence(const HloComputation* computation,
                               absl::Span<HloInstruction* const> sequence) {
  set_sequence(computation, HloInstructionSequence(sequence));
}

void HloSchedule::set_sequence(const HloComputation* computation,
                               HloInstructionSequence sequence) {
  CHECK(computation->parent() == module_);
  sequences_[computation->unique_id()] = std::move(sequence);
  execution_threads_[computation->unique_id()] =
      std::string(computation->execution_thread());
}

HloInstructionSequence& HloSchedule::GetOrCreateSequence(
    const HloComputation* computation) {
  auto it = sequences_.find(computation->unique_id());
  if (it == sequences_.end()) {
    // No sequence found for computation. Create and return an empty one.
    CHECK(computation->parent() == module_);
    execution_threads_[computation->unique_id()] =
        std::string(computation->execution_thread());
    return sequences_[computation->unique_id()];
  }
  return it->second;
}

const HloInstructionSequence& HloSchedule::sequence(
    const HloComputation* computation) const {
  return sequences_.at(computation->unique_id());
}

namespace {

enum InstructionState : uint8_t {
  kInSchedule = 1 << 0,
  kScheduled = 1 << 1,
  kDeferred = 1 << 2,
};

// Tracks in-degrees and successors for dependency-driven schedule
// reconstruction.
struct DependencyTracker {
  // Map from local_id to HloInstruction pointer for instructions in the
  // computation.
  std::vector<HloInstruction*> local_id_to_instruction;

  // Bitflags for each local_id tracking schedule and worklist state.
  std::vector<uint8_t> state;

  // Number of unscheduled operands and control predecessors for each
  // instruction in the computation.
  std::vector<int32_t> in_degree;

  // CSR offsets for successors.
  std::vector<int32_t> successor_offsets;

  // Flat array of successor local_ids in CSR layout.
  std::vector<int32_t> successors;

  // Instructions currently in the schedule that are still present in the
  // computation, in their existing relative order.
  std::vector<HloInstruction*> existing_live_instructions;

  // Worklist of instructions whose dependencies are fully satisfied and are
  // ready to be placed in the schedule.
  std::vector<HloInstruction*> ready_worklist;

  absl::Span<const int32_t> GetSuccessors(int32_t local_id) const {
    return absl::MakeConstSpan(successors)
        .subspan(successor_offsets[local_id],
                 successor_offsets[local_id + 1] - successor_offsets[local_id]);
  }
};

// Builds in-degree counts and successor lists for all instructions in the
// computation, and seeds the ready worklist with newly added 0-predecessor
// instructions.
absl::StatusOr<DependencyTracker> BuildDependencyTracker(
    const HloComputation* computation,
    const HloInstructionSequence& existing_sequence) {
  DependencyTracker tracker;

  int32_t num_local_ids =
      std::max<int32_t>(0, computation->next_unique_instruction_internal_id());
  for (HloInstruction* instruction : computation->instructions()) {
    num_local_ids = std::max(num_local_ids, instruction->local_id() + 1);
  }

  tracker.local_id_to_instruction.assign(num_local_ids, nullptr);
  for (HloInstruction* instruction : computation->instructions()) {
    tracker.local_id_to_instruction[instruction->local_id()] = instruction;
  }

  tracker.state.assign(num_local_ids, 0);
  tracker.existing_live_instructions.reserve(existing_sequence.size());
  for (int64_t id : existing_sequence.ids()) {
    const int32_t local_id = HloInstruction::CalculateLocalId(id);
    if (local_id >= 0 && local_id < num_local_ids) {
      HloInstruction* inst = tracker.local_id_to_instruction[local_id];
      if (inst != nullptr && inst->unique_id() == id) {
        tracker.state[local_id] |= kInSchedule;
        tracker.existing_live_instructions.push_back(inst);
      }
    }
  }

  tracker.in_degree.assign(num_local_ids, 0);
  std::vector<int32_t> out_degree(num_local_ids, 0);
  std::vector<int32_t> last_seen_by(num_local_ids, -1);

  struct Edge {
    int32_t pred_id;
    int32_t succ_id;
  };
  std::vector<Edge> edges;
  edges.reserve(computation->instruction_count() * 2);

  for (HloInstruction* instruction : computation->instructions()) {
    const int32_t inst_id = instruction->local_id();
    int32_t deg = 0;

    for (const HloInstruction* operand : instruction->operands()) {
      const int32_t pred_id = operand->local_id();
      if (last_seen_by[pred_id] != inst_id) {
        last_seen_by[pred_id] = inst_id;
        ++out_degree[pred_id];
        edges.push_back({pred_id, inst_id});
        ++deg;
      }
    }

    while (absl::c_linear_search(instruction->control_predecessors(),
                                 instruction)) {
      ABSL_RETURN_IF_ERROR(instruction->RemoveControlDependencyTo(instruction));
    }

    for (HloInstruction* control_pred : instruction->control_predecessors()) {
      const int32_t pred_id = control_pred->local_id();
      if (last_seen_by[pred_id] != inst_id) {
        last_seen_by[pred_id] = inst_id;
        ++out_degree[pred_id];
        edges.push_back({pred_id, inst_id});
        ++deg;
      }
    }

    tracker.in_degree[inst_id] = deg;

    // Newly added instructions with zero predecessors are ready immediately.
    if (deg == 0 && (tracker.state[inst_id] & kInSchedule) == 0) {
      tracker.ready_worklist.push_back(instruction);
    }
  }

  // Build the CSR graph from out_degree and edges.
  tracker.successor_offsets.assign(num_local_ids + 1, 0);
  for (int32_t i = 0; i < num_local_ids; ++i) {
    tracker.successor_offsets[i + 1] =
        tracker.successor_offsets[i] + out_degree[i];
  }
  tracker.successors.resize(edges.size());
  std::vector<int32_t> write_offsets = tracker.successor_offsets;
  for (const Edge& edge : edges) {
    tracker.successors[write_offsets[edge.pred_id]++] = edge.succ_id;
  }

  return tracker;
}

// Reconstructs the schedule by replaying existing_sequence and interleaving
// ready instructions as their dependencies are satisfied.
HloInstructionSequence BuildUpdatedSequence(const HloComputation* computation,
                                            DependencyTracker& tracker) {
  HloInstructionSequence new_sequence;
  new_sequence.reserve(computation->instruction_count());

  size_t ready_head = 0;

  auto schedule_instruction = [&](HloInstruction* instruction) {
    const int32_t inst_id = instruction->local_id();
    if ((tracker.state[inst_id] & kScheduled) != 0) {
      return;
    }
    tracker.state[inst_id] |= kScheduled;
    new_sequence.push_back(instruction);

    for (int32_t succ_id : tracker.GetSuccessors(inst_id)) {
      int32_t& in_degree = tracker.in_degree[succ_id];
      --in_degree;
      CHECK_GE(in_degree, 0);
      if (in_degree == 0) {
        const uint8_t s = tracker.state[succ_id];
        // Push newly added instructions and previously deferred instructions
        // to the ready worklist. Instructions still awaiting their turn in
        // existing_sequence are not pushed early.
        if ((s & kInSchedule) == 0 || (s & kDeferred) != 0) {
          tracker.ready_worklist.push_back(
              tracker.local_id_to_instruction[succ_id]);
        }
      }
    }
  };

  auto drain_ready_worklist = [&]() {
    while (ready_head < tracker.ready_worklist.size()) {
      HloInstruction* ready_instruction = tracker.ready_worklist[ready_head++];
      schedule_instruction(ready_instruction);
    }
    tracker.ready_worklist.clear();
    ready_head = 0;
  };

  // 1. Schedule any 0-predecessor newly added instructions first.
  drain_ready_worklist();

  // 2. Iterate through the existing sequence in order.
  for (HloInstruction* instruction : tracker.existing_live_instructions) {
    const int32_t inst_id = instruction->local_id();
    if ((tracker.state[inst_id] & kScheduled) != 0) {
      // Already scheduled via the ready worklist.
      continue;
    }
    if (tracker.in_degree[inst_id] == 0) {
      schedule_instruction(instruction);
      drain_ready_worklist();
    } else {
      // Instruction has unsatisfied dependencies and cannot be scheduled yet;
      // defer it until its in-degree reaches 0.
      tracker.state[inst_id] |= kDeferred;
    }
  }

  // 3. Drain any remaining ready instructions.
  drain_ready_worklist();

  return new_sequence;
}

}  // namespace

absl::Status HloSchedule::UpdateComputationSchedule(
    const HloComputation* computation) {
  HloInstructionSequence& current_sequence =
      sequences_.at(computation->unique_id());

  // Build in-degree dependency tracking for all instructions.
  ABSL_ASSIGN_OR_RETURN(DependencyTracker tracker,
                        BuildDependencyTracker(computation, current_sequence));

  // Reconstruct schedule in topological order.
  set_sequence(computation, BuildUpdatedSequence(computation, tracker));
  return absl::OkStatus();
}

absl::Status HloSchedule::Update(
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // The schedule must contain a sequence for every non-fusion computation in
  // the module for the specified threads, but can have sequences for
  // computations which no longer exist (these are removed).
  std::vector<HloComputation*> nonfusion_computations =
      module_->MakeNonfusionComputations(execution_threads);
  for (const HloComputation* computation : nonfusion_computations) {
    if (!is_computation_scheduled(computation)) {
      GetOrCreateSequence(computation);
      ABSL_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
    }
  }
  auto sum_of_sequences_for_threads = [&]() -> int64_t {
    if (execution_threads.empty()) {
      return sequences_.size();
    }
    int64_t sequences_num_for_threads = 0;
    for (const auto& [thread_name, sequence_num] :
         num_sequences_by_execution_thread()) {
      sequences_num_for_threads +=
          execution_threads.contains(thread_name) ? sequence_num : 0;
    }
    return sequences_num_for_threads;
  };
  int64_t sequence_sum = sum_of_sequences_for_threads();
  if (sequence_sum > nonfusion_computations.size()) {
    // Schedule contains some computations which have been removed from the
    // HloModule. Remove them from the schedule as well.
    absl::flat_hash_set<int64_t> nonfusion_computations_ids;
    for (const HloComputation* computation : nonfusion_computations) {
      nonfusion_computations_ids.insert(computation->unique_id());
    }
    for (auto it = sequences_.begin(); it != sequences_.end();) {
      std::string sequence_thread_name = tsl::gtl::FindWithDefault(
          execution_threads_, it->first, HloInstruction::kMainExecutionThread);
      bool is_thread_included =
          execution_threads.empty() ||
          execution_threads.contains(sequence_thread_name);
      if (!nonfusion_computations_ids.contains(it->first) &&
          is_thread_included) {
        execution_threads_.erase(it->first);
        sequences_.erase(it++);
      } else {
        ++it;
      }
    }
  }
  sequence_sum = sum_of_sequences_for_threads();
  CHECK_EQ(sequence_sum, nonfusion_computations.size());

  for (const HloComputation* computation : nonfusion_computations) {
    ABSL_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
  }

  ABSL_RETURN_IF_ERROR(Verify());
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, int64_t>
HloSchedule::num_sequences_by_execution_thread() const {
  absl::flat_hash_map<std::string, int64_t> sequence_num_by_execution_threads;
  for (const auto& id_sequence_item : sequences_) {
    ++sequence_num_by_execution_threads[tsl::gtl::FindWithDefault(
        execution_threads_, id_sequence_item.first,
        HloInstruction::kMainExecutionThread)];
  }
  return sequence_num_by_execution_threads;
}

absl::Status HloSchedule::Verify() const {
  VLOG(2) << "VerifySchedule()";
  XLA_VLOG_LINES(2, ToString());

  // Verify schedule contains exactly the same set of non-fusion computations as
  // module currently does for each thread that has schedule.
  absl::flat_hash_map<std::string, int64_t> sequence_num_by_execution_threads =
      num_sequences_by_execution_thread();
  for (const auto& [thread_name, sequence_size] :
       sequence_num_by_execution_threads) {
    std::vector<HloComputation*> nonfusion_computations =
        module_->MakeNonfusionComputations({thread_name});

    // TODO(dasenov): Replace with std::erase_if after XLA uses C++20.
    auto remove_it = std::remove_if(nonfusion_computations.begin(),
                                    nonfusion_computations.end(),
                                    [](const HloComputation* computation) {
                                      return computation->IsDeadComputation();
                                    });
    nonfusion_computations.erase(remove_it, nonfusion_computations.end());

    // It's possible to have more sequences than non_fusion_computations.
    // This is because in some cases computations that have schedules are
    // actually dead. The important thing to check is that each live non-fusion
    // computation has a sequence.
    //
    // TODO(b/418034360): Consider strenghtening this check to equality. That
    // would require cleaning up dead computations and/or recomputing the
    // schedule in a number of tests. In its present state (using less or equal)
    // this check is subsumed by the next one.
    TF_RET_CHECK(nonfusion_computations.size() <= sequence_size)
        << "For thread " << thread_name << ", schedule has " << sequence_size
        << " sequences, but module has " << nonfusion_computations.size()
        << " non-fusion computations for thread " << thread_name;
    for (const HloComputation* computation : nonfusion_computations) {
      TF_RET_CHECK(sequences_.contains(computation->unique_id()))
          << "Computation " << computation->name()
          << " missing from HLO schedule.";
    }

    // For each computation verify the set of instructions is the same and
    // that each dependency and control edge is honored.
    for (const HloComputation* computation : nonfusion_computations) {
      ABSL_RETURN_IF_ERROR(Verify(computation));
    }
  }

  return absl::OkStatus();
}

absl::Status HloSchedule::Verify(const HloComputation* computation) const {
  const HloInstructionSequence& seq = sequence(computation);
  const std::vector<HloInstruction*>& seq_instructions = seq.instructions();
  const std::vector<int64_t>& seq_ids = seq.ids();

  int32_t num_local_ids =
      std::max<int32_t>(0, computation->next_unique_instruction_internal_id());
  for (const HloInstruction* instruction : computation->instructions()) {
    num_local_ids = std::max(num_local_ids, instruction->local_id() + 1);
  }

  std::vector<const HloInstruction*> local_id_to_inst(num_local_ids, nullptr);
  for (const HloInstruction* instruction : computation->instructions()) {
    local_id_to_inst[instruction->local_id()] = instruction;
  }

  std::vector<int32_t> inst_pos(num_local_ids, -1);
  bool fast_path_matched = (seq_instructions.size() == seq_ids.size());

  if (fast_path_matched) {
    for (size_t pos = 0; pos < seq_instructions.size(); ++pos) {
      const int64_t id = seq_ids[pos];
      const int32_t lid = HloInstruction::CalculateLocalId(id);
      if (lid >= 0 && lid < num_local_ids &&
          local_id_to_inst[lid] == seq_instructions[pos] &&
          inst_pos[lid] == -1) {
        inst_pos[lid] = static_cast<int32_t>(pos);
      } else {
        fast_path_matched = false;
        break;
      }
    }
  }

  if (fast_path_matched) {
    TF_RET_CHECK(seq_instructions.size() == computation->instruction_count())
        << "Schedule for computation " << computation->name() << " has "
        << seq_instructions.size() << " instructions, expected "
        << computation->instruction_count();
  } else {
    // Fall back to exact original map-based check for invalid or out-of-sync
    // schedules to preserve all error messages and check ordering verbatim.
    absl::flat_hash_map<const HloInstruction*, int> instruction_position;
    instruction_position.reserve(seq_instructions.size());
    int pos = 0;
    for (const HloInstruction* instruction : seq_instructions) {
      TF_RET_CHECK(instruction_position.insert({instruction, pos}).second)
          << "Instruction " << instruction->name()
          << " appears more than once in the schedule";
      pos++;
    }
    TF_RET_CHECK(instruction_position.size() ==
                 computation->instruction_count())
        << "Schedule for computation " << computation->name() << " has "
        << instruction_position.size() << " instructions, expected "
        << computation->instruction_count();
    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(instruction_position.contains(instruction))
          << "Instruction " << instruction->name() << " is not in schedule";
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      inst_pos[instruction->local_id()] = instruction_position.at(instruction);
    }
  }

  for (const HloInstruction* instruction : computation->instructions()) {
    const int32_t current_pos = inst_pos[instruction->local_id()];
    for (const HloInstruction* operand : instruction->operands()) {
      TF_RET_CHECK(inst_pos[operand->local_id()] < current_pos)
          << "Instruction " << instruction->name()
          << " is not scheduled after its operand " << operand->name();
    }

    for (const HloInstruction* pred : instruction->control_predecessors()) {
      TF_RET_CHECK(inst_pos[pred->local_id()] < current_pos)
          << "Instruction " << instruction->name()
          << " is not scheduled after its control predecessor " << pred->name();
    }
  }
  return absl::OkStatus();
}

namespace {

// Returns the computation in the given module with the given unique ID. Returns
// nullptr if no such computation exists.
const HloComputation* IdToComputation(const HloModule* module, int64_t id) {
  for (const HloComputation* computation : module->computations()) {
    if (computation->unique_id() == id) {
      return computation;
    }
  }
  return nullptr;
}

}  // namespace

std::string HloSchedule::ToString() const {
  std::vector<std::string> pieces;

  pieces.push_back("HloSchedule");
  std::vector<int64_t> sorted_ids;
  for (const auto& id_sequence : sequences_) {
    sorted_ids.push_back(id_sequence.first);
  }
  absl::c_sort(sorted_ids);

  for (const int64_t id : sorted_ids) {
    const HloComputation* computation = IdToComputation(module_, id);
    const HloInstructionSequence& sequence = sequences_.at(id);
    if (computation == nullptr) {
      // The computation is not in the module and may have been deleted so it is
      // not safe to dereference any HLO pointers. Just use the HLO unique ids
      // stored in this object.
      pieces.push_back(absl::StrFormat(
          "computation with id %d (no longer in HLO module):", id));
      for (int64_t id : sequence.ids()) {
        pieces.push_back(absl::StrCat("  ", id));
      }
    } else {
      pieces.push_back(
          absl::StrFormat("computation %s (thread %s):", computation->name(),
                          computation->execution_thread()));
      for (const HloInstruction* instruction : sequence.instructions()) {
        pieces.push_back(absl::StrCat("  ", instruction->name()));
      }
    }
  }
  return absl::StrJoin(pieces, "\n");
}

std::ostream& operator<<(std::ostream& out, const HloSchedule& schedule) {
  return out << schedule.ToString();
}

}  // namespace xla
