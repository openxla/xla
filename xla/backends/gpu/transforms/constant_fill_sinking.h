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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CONSTANT_FILL_SINKING_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CONSTANT_FILL_SINKING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Shortens the lifetimes of independent, large scalar-constant fills in the
// entry schedule. When a fill precedes a while but all its users follow that
// while, moves the fill immediately before its first user. The graph and the
// relative ordering of every other instruction are unchanged.
//
// Run after latency-hiding scheduling and before buffer assignment. Fills with
// explicit ordering or stream annotations are left alone. Only cross-while
// moves are considered, to avoid changing useful overlap within a loop phase.
class ConstantFillSinking : public HloModulePass {
 public:
  absl::string_view name() const override { return "constant-fill-sinking"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CONSTANT_FILL_SINKING_H_
