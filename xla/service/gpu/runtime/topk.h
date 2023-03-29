/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_TOPK_H_
#define XLA_SERVICE_GPU_RUNTIME_TOPK_H_

#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla::gpu {

// This pass transforms eligible TopK CustomCall into a GpuTopK call that will
// then be executed by the FFI module in the .cc
class SpecializeTopk : public HloModulePass {
 public:
  absl::string_view name() const override { return "specialize-topk"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_TOPK_H_
