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
#ifndef CANCEL_ALL_GATHER_DYNAMIC_SLICE
#define CANCEL_ALL_GATHER_DYNAMIC_SLICE

#include "xla/service/op_expander_pass.h"

namespace xla {

class CancelAllGatherDynamicSlice : public OpExpanderPass {
 public:
  absl::string_view name() const override {
    return "cancel-all-gather-dynamic-slice";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // CANCEL_ALL_GATHER_DYNAMIC_SLICE