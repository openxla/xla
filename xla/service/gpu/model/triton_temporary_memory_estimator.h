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

#ifndef XLA_SERVICE_GPU_MODEL_TRITON_TEMPORARY_MEMORY_ESTIMATOR_H_
#define XLA_SERVICE_GPU_MODEL_TRITON_TEMPORARY_MEMORY_ESTIMATOR_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Architecture-parameterized models for estimating the amount of on-chip
// "temporary" memory that a candidate Triton tiling requires. This covers two
// distinct, separately-budgeted resources:
//
//   * Shared memory (a.k.a. LDS on AMD).
//   * Tensor memory (NVIDIA Blackwell / tcgen05 only)
//
// Both estimates are intentionally *conservative predictors used to prune the
// tile search early* (so that the search can fall back to a smaller tile
// instead of failing late with a RESOURCE_EXHAUSTED error during Triton
// compilation). They are NOT exact models: the actual residency of a tile is
// decided by Triton's MLIR lowering and its GPU layout/pipeline passes, which
// run after these estimates. The authoritative values are the `ttg.shared` and
// `ttg.tensor_memory_size` attributes checked in `CompileTritonToLLVM` after
// lowering; these estimates only trade late failures for early tile fallbacks.

// ===========================================================================
// Shared memory.
// ===========================================================================
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
//     tensor-memory model below and therefore excluded here.

// Device- and launch-parameter-derived constants used by the shared-memory
// estimator.
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

// Builds the shared-memory model parameters from a device description and the
// launch parameters of the fusion.
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

// ===========================================================================
// Tensor memory (NVIDIA Blackwell / tcgen05 only).
// ===========================================================================
//
// On NVIDIA Blackwell (tcgen05) the MMA accumulator (and, depending on the
// instruction, one operand) lives in a dedicated on-chip resource called
// *tensor memory* (TMEM), which is separate from shared memory.
//
// TMEM is organized as `tensor_memory_lanes()` lanes (rows) x
// `tensor_memory_columns()` columns, each cell holding a 32-bit value. A dot
// accumulator tile of shape [block_m, block_n] therefore occupies
// approximately
//     ceil(block_m / lanes) * block_n * ceil(acc_bytes / 4)
// columns, where `acc_bytes` is the accumulator element byte size (typically 4
// for an fp32 accumulator).
//
// On any non-tcgen05 architecture (all other NVIDIA GPUs and all AMD GPUs)
// there is no tensor memory; `tmem_columns_budget`/`tmem_lanes` are 0 and no op
// is charged to tensor memory.

// Device-derived constants used by the tensor-memory estimator. All values are
// read from `DeviceDescription` (see `MakeTmemModelParams`), not hardcoded.
struct TmemModelParams {
  // Maximum number of tensor-memory columns available. 0 on architectures
  // without dedicated tensor memory (all non-tcgen05 NVIDIA GPUs and all AMD
  // GPUs), in which case no op is charged to tensor memory.
  int64_t tmem_columns_budget = 0;

  // Number of tensor-memory lanes (rows). 0 on architectures without dedicated
  // tensor memory.
  int64_t tmem_lanes = 0;
};

// Builds the tensor-memory model parameters from a device description.
TmemModelParams MakeTmemModelParams(const se::DeviceDescription& device_info);

// Returns true if `hlo`'s accumulator is placed in tensor memory on this
// architecture. This is true for dot / scaled-dot when the device has dedicated
// tensor memory (tcgen05) and false everywhere else (including all AMD GPUs).
bool UsesTensorMemory(const HloInstruction& hlo, const TmemModelParams& params);

// Returns the estimated number of tensor-memory columns occupied by a dot whose
// accumulator tile is [block_m, block_n] (each dimension padded to a power of
// 2) with an accumulator element size of `accumulator_byte_size` bytes.
//
// Returns 0 when the device has no tensor memory.
int64_t EstimateDotTensorMemoryColumns(int64_t padded_block_m,
                                       int64_t padded_block_n,
                                       int64_t accumulator_byte_size,
                                       const TmemModelParams& params);

// A tiling-representation-agnostic description of a single tensor-memory-using
// op, used by `EstimateBlockTmemColumns`.
struct TmemUsingOp {
  // Name of the instruction, used only for diagnostics.
  absl::string_view name;
  // Power-of-2-padded accumulator tile dimensions (block_m x block_n).
  int64_t padded_block_m = 0;
  int64_t padded_block_n = 0;
  // Byte size of the accumulator element type.
  int64_t accumulator_byte_size = 0;
};

// Result of the block-level tensor-memory estimate.
struct BlockTmemEstimate {
  // Estimated per-block tensor-memory usage in columns.
  int64_t columns = 0;
  // Name of the op that dominated the estimate (for diagnostics), or empty if
  // there were no tensor-memory-using ops.
  absl::string_view dominant_op_name;
};

// Estimates the per-block tensor-memory usage (in columns) of a tiling from its
// tensor-memory-using ops.
//
// Accumulators are assumed to be reused across ops within the kernel, so the
// block-level estimate is the *maximum* over ops rather than their sum
// (matching the shared-memory aggregation).
BlockTmemEstimate EstimateBlockTmemColumns(
    absl::Span<const TmemUsingOp> tmem_ops, const TmemModelParams& params);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_TRITON_TEMPORARY_MEMORY_ESTIMATOR_H_
