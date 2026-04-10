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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CONV_FP8_FALLBACK_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CONV_FP8_FALLBACK_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

class HloCustomCallInstruction;

namespace gpu {

// Rewrites FP8 cuDNN convolution custom calls to BF16 when cuDNN does not
// support the specific FP8 configuration (dimensions, strides, groups, etc.).
//
// This pass dynamically probes cuDNN for available execution plans. If no
// plans are found for an FP8 convolution, it inserts FP8→BF16 convert ops
// on the operands, changes the custom call to use BF16 types and target,
// and converts the result back to FP8.
//
// The pass should run after ConvRewriter and CudnnFusedConvRewriter (which
// create the FP8 custom calls) but before the autotuner (which selects an
// algorithm from the available plans).
//
// When StreamExecutor is null (AOT compilation), the pass is a no-op.

// Rewrites one FP8 cuDNN convolution custom call to the BF16 equivalent (see
// ConvFp8Fallback). Used by the pass after cuDNN probing; exposed for tests.
absl::Status RewriteFp8ConvCustomCallToBf16(HloCustomCallInstruction* instr);

class ConvFp8Fallback : public HloModulePass {
 public:
  explicit ConvFp8Fallback(stream_executor::StreamExecutor* stream_exec)
      : stream_exec_(stream_exec) {}

  absl::string_view name() const override { return "conv-fp8-fallback"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  stream_executor::StreamExecutor* stream_exec_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CONV_FP8_FALLBACK_H_
