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

#include "xla/service/gpu/model/triton_temporary_memory_estimator.h"

#include <algorithm>
#include <cstdint>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla::gpu {

// ===========================================================================
// Shared memory.
// ===========================================================================

SmemModelParams MakeSmemModelParams(const se::DeviceDescription& device_info,
                                    int num_stages, int64_t num_warps) {
  SmemModelParams params;
  params.smem_budget_bytes = device_info.shared_memory_per_block_optin();
  params.num_stages = std::max(1, num_stages);
  params.num_warps = std::max<int64_t>(1, num_warps);

  const se::GpuComputeCapability& gpu_cc = device_info.gpu_compute_capability();

  if (const auto* cuda_cc = gpu_cc.cuda_compute_capability();
      cuda_cc != nullptr) {
    // NVIDIA.
    //
    // Blackwell (tcgen05) moves the dot accumulator and (at least) one operand
    // to dedicated tensor memory, which is accounted separately by the
    // tensor-memory model below. We therefore do not charge that portion to
    // shared memory here.
    params.dot_operands_use_smem = !cuda_cc->HasTcgen05();

    // Hopper `wgmma` sources operands from shared memory (descriptor-based), so
    // even a single-stage dot needs a shared-memory operand buffer. Ampere/Ada
    // `mma.sync` can source operands from registers, so a single-stage dot does
    // not require a persistent shared-memory buffer.
    params.mma_operands_can_source_from_registers = !cuda_cc->IsAtLeastHopper();
  } else {
    // AMD (CDNA MFMA / RDNA WMMA) and any other non-CUDA backend.
    //
    // AMD has no dedicated matrix memory: MFMA/WMMA read operands from
    // registers (VGPRs), using LDS only as a staging/prefetch and pipeline
    // buffer. So dot operands use shared memory for K-loop staging, and a
    // single-stage dot can source operands directly from registers.
    params.dot_operands_use_smem = true;
    params.mma_operands_can_source_from_registers = true;
  }

  return params;
}

SmemStagingKind ClassifyStaging(const HloInstruction& hlo,
                                const SmemModelParams& params) {
  switch (hlo.opcode()) {
    case HloOpcode::kTranspose:
      // Transposes shuffle data across lanes and use a shared-memory scratch
      // buffer for the layout conversion.
      return SmemStagingKind::kLayout;
    case HloOpcode::kDot:
    case HloOpcode::kScaledDot:
      // Dot operands are staged in shared memory only on architectures whose
      // MMA datapath uses shared memory for operand staging. On architectures
      // with dedicated tensor memory (NVIDIA tcgen05) the relevant portion is
      // accounted by the separate tensor-memory model.
      return params.dot_operands_use_smem ? SmemStagingKind::kDot
                                          : SmemStagingKind::kNone;
    default:
      // All other ops are treated as register-resident for the purpose of the
      // shared-memory estimate:
      //   * elementwise / broadcast / iota / bitcast / reshape / slice: their
      //     tiles live in registers;
      //   * reduce / scan: the streamed input and the accumulator are
      //     register-resident. Any cross-wave combine buffer is small (bounded
      //     by the output tile times the number of warps) and is
      //     *intentionally overlooked* here to avoid over-rejecting otherwise
      //     valid reduction/softmax tilings. The authoritative `ttg.shared`
      //     check performed after Triton lowering remains the backstop for the
      //     rare case where such a buffer would push a kernel over budget.
      return SmemStagingKind::kNone;
  }
}

int64_t EstimateLayoutStagingBytes(int64_t padded_tile_elements,
                                   int64_t element_byte_size,
                                   const SmemModelParams& params) {
  // A layout-conversion scratch buffer is double/multi-buffered when the loop
  // is pipelined, so it scales with the pipeline depth.
  return padded_tile_elements * element_byte_size * params.num_stages;
}

int64_t EstimateDotStagingBytes(int64_t padded_operand_tile_elements,
                                int64_t element_byte_size,
                                const SmemModelParams& params) {
  if (!params.dot_operands_use_smem) {
    return 0;
  }
  // On architectures that can source MMA operands from registers, a
  // non-pipelined (single-stage) dot streams operands through registers and
  // does not allocate a persistent shared-memory operand buffer.
  if (params.num_stages <= 1 && params.mma_operands_can_source_from_registers) {
    return 0;
  }
  // Otherwise the operand K-tiles are staged in shared memory, double/multi-
  // buffered across the pipeline depth.
  return padded_operand_tile_elements * element_byte_size * params.num_stages;
}

BlockSmemEstimate EstimateBlockSmemBytes(
    absl::Span<const SmemStagingOp> staging_ops,
    const SmemModelParams& params) {
  BlockSmemEstimate estimate;
  // Staging buffers are assumed to be reused across ops within the kernel, so
  // the block-level estimate is the maximum over staging ops rather than their
  // sum.
  for (const SmemStagingOp& op : staging_ops) {
    int64_t op_bytes = 0;
    switch (op.kind) {
      case SmemStagingKind::kLayout:
        op_bytes = EstimateLayoutStagingBytes(op.padded_tile_elements,
                                              op.element_byte_size, params);
        break;
      case SmemStagingKind::kDot:
        op_bytes = EstimateDotStagingBytes(op.padded_operand_tile_elements,
                                           op.element_byte_size, params);
        break;
      case SmemStagingKind::kNone:
        // Callers should not include kNone ops, but tolerate them.
        op_bytes = 0;
        break;
    }
    if (op_bytes > estimate.bytes) {
      estimate.bytes = op_bytes;
      estimate.dominant_op_name = op.name;
    }
  }
  return estimate;
}

// ===========================================================================
// Tensor memory (NVIDIA Blackwell / tcgen05 only).
// ===========================================================================

TmemModelParams MakeTmemModelParams(const se::DeviceDescription& device_info) {
  TmemModelParams params;
  // These are 0 on architectures without dedicated tensor memory (all AMD GPUs
  // and all non-tcgen05 NVIDIA GPUs), which makes the tensor-memory estimate a
  // no-op there.
  params.tmem_columns_budget = device_info.tensor_memory_columns();
  params.tmem_lanes = device_info.tensor_memory_lanes();

  // Fall back to the tcgen05 architectural defaults (128 lanes x 512 columns)
  // when the device description does not carry the tensor-memory geometry (e.g.
  // devices built from a target-config spec that predates these fields). This
  // keeps the pre-lowering estimate consistent with the authoritative
  // post-lowering check in `CompileTritonToLLVM`.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory
  const auto* cuda_cc =
      device_info.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc != nullptr && cuda_cc->HasTcgen05()) {
    constexpr int64_t kDefaultTensorMemoryColumns = 512;
    constexpr int64_t kDefaultTensorMemoryLanes = 128;
    if (params.tmem_columns_budget <= 0) {
      params.tmem_columns_budget = kDefaultTensorMemoryColumns;
    }
    if (params.tmem_lanes <= 0) {
      params.tmem_lanes = kDefaultTensorMemoryLanes;
    }
  }
  return params;
}

bool UsesTensorMemory(const HloInstruction& hlo,
                      const TmemModelParams& params) {
  if (params.tmem_columns_budget <= 0 || params.tmem_lanes <= 0) {
    // No dedicated tensor memory on this architecture.
    return false;
  }
  switch (hlo.opcode()) {
    case HloOpcode::kDot:
    case HloOpcode::kScaledDot:
      return true;
    default:
      return false;
  }
}

int64_t EstimateDotTensorMemoryColumns(int64_t padded_block_m,
                                       int64_t padded_block_n,
                                       int64_t accumulator_byte_size,
                                       const TmemModelParams& params) {
  if (params.tmem_columns_budget <= 0 || params.tmem_lanes <= 0) {
    return 0;
  }
  // TMEM cells hold 32-bit (4-byte) values. Accumulators wider than 4 bytes
  // occupy proportionally more columns.
  constexpr int64_t kTmemCellBytes = 4;
  const int64_t cells_per_element =
      CeilOfRatio<int64_t>(accumulator_byte_size, kTmemCellBytes);
  // A [block_m, block_n] accumulator maps block_m rows onto TMEM lanes; if
  // block_m exceeds the number of lanes, multiple lane-groups are stacked into
  // additional columns.
  const int64_t lane_groups =
      CeilOfRatio<int64_t>(padded_block_m, params.tmem_lanes);
  return lane_groups * padded_block_n * cells_per_element;
}

BlockTmemEstimate EstimateBlockTmemColumns(
    absl::Span<const TmemUsingOp> tmem_ops, const TmemModelParams& params) {
  BlockTmemEstimate estimate;
  // Accumulators are assumed to be reused across ops within the kernel, so the
  // block-level estimate is the maximum over ops rather than their sum.
  for (const TmemUsingOp& op : tmem_ops) {
    const int64_t op_columns = EstimateDotTensorMemoryColumns(
        op.padded_block_m, op.padded_block_n, op.accumulator_byte_size, params);
    if (op_columns > estimate.columns) {
      estimate.columns = op_columns;
      estimate.dominant_op_name = op.name;
    }
  }
  return estimate;
}

}  // namespace xla::gpu
