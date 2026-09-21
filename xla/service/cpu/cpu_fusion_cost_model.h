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

#ifndef XLA_SERVICE_CPU_CPU_FUSION_COST_MODEL_H_
#define XLA_SERVICE_CPU_CPU_FUSION_COST_MODEL_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_module_config.h"

namespace xla::cpu {

// A roofline-style profitability model for XLA:CPU fusion decisions.
//
// XLA:CPU currently decides "may I duplicate this producer into its consumers?"
// with a boolean opcode classification (`InstructionFusion::is_expensive_`,
// `CpuInstructionFusion::IsExpensive`). That judgement has no term for array
// size: a `sqrt` over a 4 MB buffer is classified exactly like a `sqrt` over
// four elements. Refusing the duplication does not avoid the work -- it forces
// the producer's entire output through memory, which for a buffer larger than
// cache costs far more than the duplicated arithmetic.
//
// This model replaces that boolean with the comparison that actually decides
// the question on a CPU:
//
//     recompute  iff  extra_flops <= machine_balance * saved_bytes
//
// where `machine_balance` is the machine's flops-per-byte ratio (how many
// arithmetic operations the core retires in the time it takes to move one byte
// to or from DRAM). Below `cache_bytes` the intermediate stays resident, the
// round trip is not paid, and the model declines -- which is why very small
// problems correctly keep materializing.
//
// Two tunables, both exposed as `xla_cpu_fusion_*` debug options.
//
// This class is CPU-only by construction. The generic fusion passes reach it
// only through virtuals that default to the pre-existing behaviour, so GPU
// never constructs it and its decisions cannot affect any other backend.
class CpuFusionCostModel {
 public:
  struct Params {
    // Arithmetic operations retired in the time one byte moves to/from DRAM.
    // A modern multi-core x86 socket is in the range 5-20; 8 is a middle
    // value that separates the cases in the pairwise-reduction testbed by a
    // comfortable margin in both directions.
    int64_t machine_balance_flops_per_byte = 8;
    // Below this many bytes an intermediate stays resident in cache, so
    // materializing it costs nothing and the model should not trade flops for
    // it. Roughly a per-core L2.
    int64_t cache_bytes = 256 * 1024;
  };

  static Params ParamsFromConfig(const HloModuleConfig& config);

  explicit CpuFusionCostModel(Params params) : params_(params) {}

  // Estimated arithmetic cost of producing one element of `instr`, in
  // flop-equivalents. Counts only `instr` itself (its operands become fusion
  // parameters and are read, not recomputed), except that reductions include
  // their reduced extent times their body, and fusions include everything
  // they contain.
  //
  // A fusion's inner instructions are summed without normalizing for their own
  // output sizes, so an elementwise op on a larger shape feeding a reduction
  // to a smaller one is charged once rather than once per element it
  // contributes. That biases slightly toward recomputing. It is deliberate:
  // the comparison downstream is against a memory round trip worth tens to
  // hundreds of flops, so this never decides a verdict on its own.
  int64_t PerElementFlops(const HloInstruction* instr);

  // Is recomputing `producer` in each of its users cheaper than writing it to
  // memory and reading it back?
  //
  // The verdict is computed once per producer and cached, so every consumer
  // edge gets the same answer. Answering "fuse" on some edges and "do not" on
  // others is how a shared intermediate turns into one buffer per consumer.
  bool RecomputeBeatsMaterialize(const HloInstruction* producer);

  // Same question, but for a producer that will be re-emitted a known number
  // of times inside a single fusion (the cache-invalidating-op case in
  // `FusionNodeIndexingEvaluation::CodeDuplicationTooHigh`). `emitted_copies`
  // is how many distinct index vectors the producer would be emitted under.
  bool RematerializationBeatsMaterialization(const HloInstruction* producer,
                                             int64_t emitted_copies);

  // Drops all caches. Must be called whenever the graph is rebuilt; the
  // caches are keyed on instruction pointers, which fusion invalidates.
  void Clear();

  const Params& params() const { return params_; }

 private:
  // Core comparison, shared by both public predicates. `total_reads` is how
  // many elements of `producer` are read in total across all its consumers
  // (>= producer_elements).
  bool RecomputeBeatsMaterializeForReads(const HloInstruction* producer,
                                         int64_t total_reads);

  // How many elements of `producer` its users read in total.
  int64_t TotalElementReads(const HloInstruction* producer);

  int64_t ComputationPerElementFlops(const HloComputation* computation);

  Params params_;
  // Keyed on unique_id(), not on the pointer: fusion deletes instructions as
  // it goes, and a later allocation can reuse a freed address, which would
  // silently return another instruction's cached answer.
  absl::flat_hash_map<int64_t, int64_t> per_element_flops_;
  absl::flat_hash_map<int64_t, int64_t> computation_flops_;
  absl::flat_hash_map<int64_t, bool> verdict_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_FUSION_COST_MODEL_H_
