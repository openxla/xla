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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_CUSTOM_REDUCTION_ANALYSIS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_CUSTOM_REDUCTION_ANALYSIS_H_

#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// Checks that XLA can map the stream types, the constant identities and the
// reducer ops of the HLO reduction to the cuDNN frontend. cuDNN decides whether
// it supports the translated reduction. A refusal lets the autotuner keep the
// native emitter.
absl::Status AnalyzeCudnnCustomReduction(const HloInstruction& reduce);

// Returns a key that ignores HLO names and keeps the exact bits of reducer
// constants. LowerCudnnCustomReduction passes this key to the cuDNN frontend
// as the semantic_key argument of Custom_reduction_attributes::set_reducer.
// The frontend makes it part of Graph::key().
std::string CudnnCustomReductionSemanticKey(const HloComputation& body);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_CUSTOM_REDUCTION_ANALYSIS_H_
