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

#ifndef XLA_SERVICE_GPU_MODEL_TRITON_SHARED_MEMORY_ESTIMATOR_H_
#define XLA_SERVICE_GPU_MODEL_TRITON_SHARED_MEMORY_ESTIMATOR_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Architecture-parameterized model for estimating the amount of per-block
// shared memory that a candidate Triton tiling requires.
//
// The estimate is intentionally a *conservative predictor used to prune the
// tile search early* (so that the search can fall back to a smaller tile
// instead of failing late with a shared-memory RESOURCE_EXHAUSTED error during
// Triton compilation). It is NOT an exact model: whether a given tile lives in
// registers or shared memory is ultimately decided by Triton's MLIR lowering
// and its GPU layout/pipeline passes, which run after this estimate. The
// authoritative value is the `ttg.shared` attribute checked in
// `CompileTritonToLLVM` after lowering; this estimate only trades late failures
// for early tile fallbacks.
//
// Design notes (see the .cc for the full rationale):
//   * A tile is charged to shared memory only at "staging events":
//       - layout conversions (transpose / convert-layout), and
//       - dot/scaled-dot operand K-tile staging (global -> shared prefetch),
//         subject to architecture- and pipeline-depth-dependent rules.
//     Everything else (elementwise, broadcast, iota, reduction inputs and
//     accumulators, MMA accumulators) is register-resident and is handled by
//     the register-pressure model, not here.
//   * Cross-wave reduction combine buffers are *intentionally overlooked* to
//     avoid over-rejecting valid reduction/softmax tilings. See
//     `ClassifyStaging` for details.
//   * AMD GPUs have no dedicated matrix memory: MFMA (CDNA) and WMMA (RDNA)
//     read operands from registers, using LDS only as a staging/prefetch and
//     pipeline buffer. NVIDIA Blackwell (tcgen05) has dedicated tensor memory
//     for (part of) the dot datapath, which is accounted separately by the
//     existing `ttg.tensor_memory_size` check and therefore excluded here.

// Device- and launch-parameter-derived constants used by the estimator.
struct SmemModelParams {
  // Maximum shared memory available for a single block, including the
  // dynamically allocated / opt-in portion.
  int64_t smem_budget_bytes = 0;

  // Software-pipeline depth (number of in-flight loop iterations / staging
  // buffers). Taken from BlockLevelParameters::num_stages.
  int num_stages = 1;

  // Number of warps/wavefronts per block. Taken from
  // BlockLevelParameters::num_warps.
  int64_t num_warps = 1;

  // Whether dot/scaled-dot operands are staged through shared memory on this
  // architecture. True for architectures whose MMA datapath sources operands
  // from shared memory or uses shared memory as the K-loop staging buffer
  // (NVIDIA Ampere/Ada/Hopper, AMD MFMA/WMMA). The portion of the dot datapath
  // that lives in dedicated tensor memory (NVIDIA Blackwell/tcgen05) is
  // excluded and accounted by the separate tensor-memory check.
  bool dot_operands_use_smem = true;

  // Whether MMA operands can be sourced directly from registers when there is
  // no software pipelining (num_stages <= 1). For such architectures a
  // single-stage dot does not require a persistent shared-memory operand
  // buffer. True for NVIDIA Ampere/Ada and AMD MFMA/WMMA; false for
  // architectures whose MMA requires shared-memory operands even without
  // pipelining (NVIDIA Hopper wgmma).
  bool mma_operands_can_source_from_registers = true;
};

// Builds the model parameters from a device description and the launch
// parameters of the fusion.
SmemModelParams MakeSmemModelParams(const se::DeviceDescription& device_info,
                                    int num_stages, int64_t num_warps);

// The way in which an instruction's tile is staged in shared memory.
enum class SmemStagingKind {
  // The tile is not staged in shared memory (register-resident, or handled by a
  // separate resource budget such as tensor memory).
  kNone,
  // The tile is staged in shared memory because of a layout conversion
  // (transpose / convert-layout / layout-changing reshape).
  kLayout,
  // The instruction is a dot/scaled-dot whose operand K-tiles are staged in
  // shared memory.
  kDot,
};

// Classifies how (if at all) `hlo`'s tile is staged in shared memory.
//
// This is a deliberately conservative, heuristic allowlist: the final residency
// decision is Triton's, made after this classification. In particular:
//   * reductions/scans are classified as kNone -- their inputs stream into
//     register accumulators and any cross-wave combine buffer is small and
//     intentionally overlooked to avoid over-rejecting valid tilings;
//   * elementwise/broadcast/iota are kNone (register-resident);
//   * dots are kDot only when `params.dot_operands_use_smem` is true.
SmemStagingKind ClassifyStaging(const HloInstruction& hlo,
                                const SmemModelParams& params);

// Returns the estimated shared-memory bytes contributed by a layout-staging op
// whose padded tile contains `padded_tile_elements` elements of a type
// occupying `element_byte_size` bytes.
//
// The staging buffer is multiplied by the pipeline depth (num_stages) to
// account for double/multi-buffering when the loop is pipelined.
int64_t EstimateLayoutStagingBytes(int64_t padded_tile_elements,
                                   int64_t element_byte_size,
                                   const SmemModelParams& params);

// Returns the estimated shared-memory bytes contributed by a dot/scaled-dot
// whose operand K-tile staging buffers total `padded_operand_tile_elements`
// elements (i.e. block_m*block_k + block_k*block_n, each dimension padded to a
// power of 2) of a type occupying `element_byte_size` bytes.
//
// Returns 0 when the architecture can source MMA operands from registers and
// there is no software pipelining (num_stages <= 1), reflecting that a
// single-stage dot streams operands through registers rather than staging a
// persistent shared-memory buffer. Otherwise the staging buffers are multiplied
// by the pipeline depth (num_stages).
int64_t EstimateDotStagingBytes(int64_t padded_operand_tile_elements,
                                int64_t element_byte_size,
                                const SmemModelParams& params);

// A tiling-representation-agnostic description of a single shared-memory
// staging op, used by `EstimateBlockSmemBytes`. Callers (which may hold either
// the symbolic or the experimental tiling representation) are responsible for
// computing the power-of-2-padded element counts from their own tile
// representation and populating this struct.
struct SmemStagingOp {
  // Name of the instruction, used only for diagnostics.
  absl::string_view name;
  // How the tile is staged in shared memory. Must not be kNone.
  SmemStagingKind kind;
  // Byte size of the element type of the staging instruction.
  int64_t element_byte_size = 0;
  // For kLayout: the number of power-of-2-padded elements in the op's own tile.
  int64_t padded_tile_elements = 0;
  // For kDot: the total number of power-of-2-padded elements across the operand
  // K-tile staging buffers (block_m*block_k + block_k*block_n).
  int64_t padded_operand_tile_elements = 0;
};

// Result of the block-level shared-memory estimate.
struct BlockSmemEstimate {
  // Estimated per-block shared-memory usage in bytes.
  int64_t bytes = 0;
  // Name of the staging op that dominated the estimate (for diagnostics), or
  // empty if there were no staging ops.
  absl::string_view dominant_op_name;
};

// Estimates the per-block shared-memory usage of a tiling from its staging ops.
//
// The staging buffers are assumed to be reused across ops within the kernel, so
// the block-level estimate is the *maximum* over staging ops rather than their
// sum. This is shared by both the symbolic (legacy) and experimental tiling
// constraint paths so that they apply an identical shared-memory model.
BlockSmemEstimate EstimateBlockSmemBytes(
    absl::Span<const SmemStagingOp> staging_ops, const SmemModelParams& params);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_TRITON_SHARED_MEMORY_ESTIMATOR_H_
