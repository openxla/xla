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
#ifndef XLA_SERVICE_GPU_MULTI_STREAMING_SCHEDULING_H_
#define XLA_SERVICE_GPU_MULTI_STREAMING_SCHEDULING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/status.h"

namespace xla::gpu {

// This pass modifies the schedule of the module so that instructions may run
// on multiple streams in parallel. It will find instructions that are
// parallelizable, assign streams to the instructions, and reschedule them to
// run together. The parallel instruction will also need to be wrapped by
// an async instruction to extend the liveness of its assigned buffers. For
// example, the following code contains parallelizable instr A and C.
//
// %A = f32[] fusion(f32[] %arg0, f32[] %arg1)
// %B = f32[] fusion(f32[] %A)
// %C = f32[] fusion(f32[] %arg2, f32[] %arg3)
//
// After the transformation, we will get:
//
// %A-start = f32[] fusion(f32[] %arg0, f32[] %arg1), operation_queue_id=1
// %C = f32[] fusion(f32[] %arg2, f32[] %arg3)
// &A-done = fusion-done(%A-start)
// %B = f32[] fusion(f32[] %A)
//
class MultiStreamingScheduling : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "multi-streaming-scheduling";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MULTI_STREAMING_SCHEDULING_H_
