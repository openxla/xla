/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_RENAME_INSTRUCTIONS_H_
#define XLA_SERVICE_GPU_RENAME_INSTRUCTIONS_H_

#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// Appends ".0" suffix to instruction names.
//
// Every time an instruction is duplicated, the last integer suffix is
// incremented.
// Before: broadcast.123 -> broadcast.124
// After: broadcast.123.0 -> broadcast.123.1
//
// With this modification it will be easier to match intructions before and
// after fusion passes, because they will have the same unique prefix. Names
// are not used in the pipeline, but it makes debugging much easier.
class RenameInstructions : public HloModulePass {
 public:
  absl::string_view name() const override { return "rename-instructions"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RENAME_INSTRUCTIONS_H_
