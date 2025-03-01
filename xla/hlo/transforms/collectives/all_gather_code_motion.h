/* Copyright 2025 The OpenXLA Authors.

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

#ifndef ALL_GATHER_CODE_MOTION_H
#define ALL_GATHER_CODE_MOTION_H

#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// AllGatherCodeMotion optimizes HLO modules by hoisting all-gather operations
// on while loop parameters out of the loop body into the parent computation.
// This transformation modifies buffer shapes in-place and adjusts the while
// loop condition and body computations accordingly. The pass also accepts a
// replica group that forces the pass to only code motion that replica group.
//
// The pass handles code motion and shape adjustments for:
// - All-gather operations and their corresponding convert ops
// - Optimization barriers referencing the modified shapes
// - Tuples and get-tuple-element operations that contain or extract the
// modified values
//
// Example transformation:
// clang-format off
// while_body {
//   body_param = (f32[128,256], f32[128,256]) parameter(0)
//   p0 = f32[128,256] get-tuple-element(body_param), index=0
//   p1 = f32[128,256] get-tuple-element(body_param), index=1
//   ag0 = f32[512,256] all-gather(p0), dimensions={0}, replica_groups={{0,1,2,3}}
//   ag1 = f32[512,256] all-gather(p1), dimensions={0}, replica_groups={{0,1,2,3}}
//   compute = f32[128,256] some-computation(ag0, ag1)
//   ROOT tuple = (f32[128,256], f32[128,256]) tuple(compute, p1)
// }
//
// while_cond {
//   cond_param = (f32[128,256], f32[128,256]) parameter(0)
//   ROOT result = pred[] custom-call(cond_param)
// }
//
// ENTRY main {
//   param0 = f32[128,256]{1,0} parameter(0)
//   param1 = f32[128,256]{1,0} parameter(1)
//   while = (f32[128,256], f32[128,256]) while(param0, param1), condition=while_cond, body=while_body
// }
//
// ==> Transforms to:
//
// while_body {
//   body_param = (f32[512,256], f32[512,256]) parameter(0)
//   p0 = f32[512,256] get-tuple-element(body_param), index=0
//   p1 = f32[512,256] get-tuple-element(body_param), index=1
//   compute = f32[128,256] some-computation(p0, p1)
//   ROOT tuple = (f32[128,256], f32[512,256]) tuple(compute, p1)
// }
//
// while_cond {
//   cond_param = (f32[512,256], f32[512,256]) parameter(0)
//   ROOT result = pred[] custom-call(cond_param)
// }
//
// ENTRY main {
//   param0 = f32[128,256]{1,0} parameter(0)
//   param1 = f32[128,256]{1,0} parameter(1)
//   ag0 = f32[512,256] all-gather(param0), dimensions={0}, replica_groups={{0,1,2,3}}
//   ag1 = f32[512,256] all-gather(param1), dimensions={0}, replica_groups={{0,1,2,3}}
//   while = (f32[128,256], f32[512,256]) while(ag0, ag1), condition=while_cond, body=while_body
// }
// clang-format on

class AllGatherCodeMotion : public HloModulePass {
 public:
  absl::string_view name() const override { return "all-gather-code-motion"; }

  AllGatherCodeMotion(std::vector<ReplicaGroup>* moveable_group = nullptr);

  using HloPassInterface::Run;

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::Status RewriteGetTupleElement(HloInstruction* while_op);
  int64_t CountWhileLoops(HloModule* module);
  absl::Status RewriteGetTupleElementUsers(HloInstruction* inst,
                                           HloInstruction* new_tuple);
  absl::Status RewriteTuple(HloInstruction* original_input_tuple,
                            HloComputation* computation,
                            HloInstruction* while_loop);
  absl::Status RewriteConvert(HloInstruction* convert_element,
                              HloInstruction* tuple_element_in_body);
  bool MaybeSkipCodeMotion(int64_t num_while_loops,
                           HloInstruction* convert_element);
  absl::StatusOr<HloInstruction*> CreateAndReplaceAllGather(
      HloComputation* computation, HloComputation* while_body,
      HloInstruction* original_input_tuple, HloInstruction* while_loop,
      HloInstruction* operand, HloInstruction* all_gather, int64_t tuple_index);
  absl::StatusOr<bool> TransformWhileLoop(HloInstruction* instruction,
                                          int64_t num_while_loops,
                                          bool rewrite_tuple,
                                          bool rewrite_convert);

 protected:
  // Explicit replica group to code motion.
  std::vector<ReplicaGroup>* moveable_group_ = nullptr;
};

}  // namespace xla

#endif  // ALL_GATHER_CODE_MOTION_H
