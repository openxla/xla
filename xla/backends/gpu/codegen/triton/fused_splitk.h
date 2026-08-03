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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSED_SPLITK_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSED_SPLITK_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Fused split-K: for narrow dots (tiny non-contracting dims, long
// contraction), `split_k` programs cooperate on each output tile, each
// reducing K/split_k and atomically accumulating its partial into the
// pre-zeroed output — a single kernel, in contrast to the 2-kernel
// dot+reduce of the HLO split-k rewriter.
//
// split_k is not a config parameter: the kernel is HBM-bandwidth-bound and
// its latency-vs-split_k curve is a wide plateau, so any value works that
// (a) yields enough programs to fully occupy the SMs, which is what hiding
// DRAM latency with low-ILP loads requires, and (b) divides the contraction
// tile count. ChooseFusedSplitK picks such a value from the device
// description alone.

// Non-contracting size products (m, n) and total contraction size (k) of a
// dot. All strictly positive.
struct NarrowDotSizes {
  int64_t m;
  int64_t n;
  int64_t k;
};

// Returns the dot's sizes iff its *shape* qualifies for the fused split-K
// emission: output type is f32 and equals the GEMM accumulator type (the
// atomic add is the cross-split reduction, so there is no place to round to
// a different output type), a single contracting dimension, non-contracting
// products <= 16 each, and a contraction of at least 4096. This is the
// pre-fusion test used by the split-k rewriter; codegen additionally
// requires the dot to be the root of the fused computation.
std::optional<NarrowDotSizes> FusedSplitKQualifyingSizes(
    const HloDotInstruction* dot);

// Returns true if fused split-K emission is allowed for this module: the
// xla_gpu_enable_fused_split_k flag is on, determinism is not required (the
// atomic float additions are nondeterministic), and the experimental tiling
// propagation emitter (which does not support split-K) is not enabled.
bool FusedSplitKEnabled(const HloModuleConfig& config);

// Returns the split_k to use for `dot` given the contraction tile size
// `block_k`; 1 if the dot does not qualify or no saturating split divides
// the contraction tile count.
int64_t ChooseFusedSplitK(const HloDotInstruction* dot, int64_t block_k,
                          const se::DeviceDescription& device_info);

// Returns the split_k for a fusion computation root: 1 unless `root` is a
// qualifying dot annotated with a contraction tile (its Tile backend config;
// the last entry is the contraction tile size, see ConvertTritonGemmConfig).
int64_t ChooseFusedSplitKForFusionRoot(
    const HloInstruction& root, const se::DeviceDescription& device_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSED_SPLITK_H_
