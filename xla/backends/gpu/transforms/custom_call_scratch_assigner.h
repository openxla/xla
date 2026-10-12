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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUSTOM_CALL_SCRATCH_ASSIGNER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUSTOM_CALL_SCRATCH_ASSIGNER_H_

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {

// Asks the scratch handler of every native custom call (see
// `NativeCustomCallHandlerBundle::request_scratch_buffers`) which scratch
// buffers it needs and appends them to the custom call's result.
//
// A custom call with result shape `R` and scratch shapes `S0, ..., Sn-1` is
// replaced by one with result shape `(R, S0, ..., Sn-1)`. The original result
// `R` is always preserved as element 0 (unflattened, whether it is an array or
// a tuple). The original users of the custom call are rewired through
// get-tuple-element instructions, so the rest of the module doesn't observe the
// shape change. Scratch shapes keep the layout (in particular the memory space)
// the handler asked for; shapes without a layout get the default layout.
//
// Rewritten custom calls get the frontend attribute
// `kNativeCustomCallNumScratchBuffersAttr` (declared in
// native_custom_call_handler_registry.h) that records how many scratch
// buffers were appended. Custom calls without a scratch handler, custom calls
// whose handler returns no scratch buffers, and custom calls that already carry
// the attribute are left alone, so the pass is idempotent.
class CustomCallScratchAssigner : public HloModulePass {
 public:
  // `gpu_topology` and `mlir_context` must outlive the pass. `mlir_context` is
  // used to decode backend configs into FFI attributes for the handlers.
  CustomCallScratchAssigner(const GpuTopology* absl_nonnull gpu_topology,
                            mlir::MLIRContext* absl_nonnull mlir_context)
      : gpu_topology_(gpu_topology), mlir_context_(mlir_context) {}

  absl::string_view name() const override {
    return "custom-call-scratch-assigner";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);
  absl::StatusOr<bool> RunOnCustomCall(HloCustomCallInstruction* custom_call);

  const GpuTopology* absl_nonnull gpu_topology_;
  mlir::MLIRContext* absl_nonnull mlir_context_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUSTOM_CALL_SCRATCH_ASSIGNER_H_
