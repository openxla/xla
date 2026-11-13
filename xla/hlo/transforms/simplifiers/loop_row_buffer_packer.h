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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_LOOP_ROW_BUFFER_PACKER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_LOOP_ROW_BUFFER_PACKER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Packs the small buffers that a while loop only writes, one row per iteration
// at a shared row index, into one buffer per element type.
//
// A scan that stacks many small per iteration values carries one [N] or [N, m]
// buffer per value, each written by its own dynamic-update-slice. The
// collective pipeliner's forward sink mode makes such buffers too: one per sunk
// collective operand. For buffers b_1..b_K that the loop body only writes,
//
//   b_k' = dynamic-update-slice(b_k, u_k, i, 0)    u_k of shape [1] or [1, m_k]
//
// become one buffer p of shape [N, m_1 + ... + m_K] with
//
//   p' = dynamic-update-slice(p, concatenate(u_1, ..., u_K), i, 0)
//
// and every loop output b_k becomes a slice of p. Values do not change. The old
// buffers stay in the loop state unused, for WhileLoopSimplifier to remove.
//
// Run it after SPMD partitioning and before HostOffloader and layout
// assignment. Before partitioning a packed loop would lose its shardings, the
// result sharding of a root loop included, and with manual shardings the
// partitioner can then abort. What the pass adds carries no layout or memory
// space, and it leaves the buffers that a MoveToHost annotation places in host
// memory.
class LoopRowBufferPacker : public HloModulePass {
 public:
  absl::string_view name() const override { return "loop-row-buffer-packer"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_LOOP_ROW_BUFFER_PACKER_H_
