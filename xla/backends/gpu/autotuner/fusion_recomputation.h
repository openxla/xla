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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_FUSION_RECOMPUTATION_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_FUSION_RECOMPUTATION_H_

#include "xla/backends/gpu/transforms/recompute_fusion_side_outputs.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
// The evaluator and its cache are scoped to ONE compilation on ONE device.
// Compiler/target/executor must outlive the returned function. No cross-process
// decisions are reused: codegen, flags and device changes require
// remeasurement.
RecomputeFusionSideOutputs::Evaluator MakeFusionRecomputationEvaluator(
    Compiler* compiler, const Compiler::GpuTargetConfig& target,
    se::StreamExecutor* executor,
    se::DeviceAddressAllocator* allocator = nullptr);
}  // namespace xla::gpu
#endif  // XLA_BACKENDS_GPU_AUTOTUNER_FUSION_RECOMPUTATION_H_
