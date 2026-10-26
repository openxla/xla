/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_OCCUPANCY_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_OCCUPANCY_H_

// Theoretical-occupancy model for AMDGPU kernels.
//
// This header deliberately has NO ROCm includes. It is a pure arithmetic
// library so that the model can be unit-tested on any CPU host, without a
// ROCm toolchain and without a GPU. rocm_collector.h pulls in hip_runtime.h
// and rocprofiler-sdk/agent.h, so keeping the formula there would mean it
// could never be covered by CPU CI.
//
// The model is a port of, and in two places a deliberate divergence from,
// llvm/lib/Target/AMDGPU: AMDGPUAsmPrinter -> AMDGPUMCExpr::evaluateOccupancy
// -> GCNSubtarget::computeOccupancy / getOccupancyWith{WorkGroupSizes,
// NumSGPRs, NumVGPRs}. The two divergences are documented at GetOccupancy()
// in the .cc; do not "fix" them back.
//
// UNITS -- these three quantities are all called "occupancy" in different
// parts of the stack, and they are not the same number:
//   * waves_per_simd (a.k.a. waves/EU) is what LLVM's evaluateOccupancy, the
//     "; Occupancy:" asm comment and -Rpass-analysis=kernel-resource-usage
//     report. Range [1, 8] on CDNA2/3/4.
//   * active_waves_per_cu is per CU = waves_per_simd * simd_per_cu.
//   * occupancy_pct is percent of THREADS per CU, matching CUPTI's
//     activeBlocksPerMultiprocessor * block_size * 100 /
//     maxThreadsPerMultiprocessor. It is the number users compare across
//     vendors on one dashboard, so it takes CUDA's definition.

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace xla {
namespace profiler {

// Which hardware resource bounded occupancy. Ordered by how actionable it is.
//
// This is the highest-value thing the model produces. A bare percentage
// prompts the wrong reflex -- chase 100% -- when a well-tuned MFMA GEMM
// legitimately runs at low occupancy by design. Each limiter value maps to a
// distinct concrete action, and kNone tells the engineer to stop looking at
// occupancy altogether.
enum class OccupancyLimiter : uint8_t {
  kNone = 0,   // at the wave-slot ceiling; occupancy is not the problem
  kVGPR,       // unified arch+accum VGPR file
  kLDS,        // group segment
  kSGPR,       // scalar registers (gfx9 only)
  kWorkgroup,  // wave-slot quantization by workgroup size
  kBarrier,    // max concurrent workgroups per CU (never binds on CDNA)
};

const char* OccupancyLimiterName(OccupancyLimiter l);

// Per-target constants that rocprofiler-sdk does NOT expose. Mirrors
// llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp; every field names the LLVM
// accessor it was read from.
struct AmdGpuTargetConstants {
  const char* name;             // "gfx942"
  uint32_t max_waves_per_simd;  // getMaxWavesPerEU
  uint32_t simd_per_cu;         // getEUsPerCU
  uint32_t wave_front_size;
  uint32_t total_vgprs;        // getTotalNumVGPRs, per SIMD lane (unified)
  uint32_t vgpr_granule;       // getVGPRAllocGranule
  uint32_t arch_vgpr_granule;  // getArchVGPRAllocGranule
  uint32_t lds_per_cu;         // getAddressableLocalMemorySize
  uint32_t lds_granule;        // getLdsDwGranularity * 4
  uint32_t total_sgprs;        // getTotalNumSGPRs
  uint32_t sgpr_granule;       // getSGPRAllocGranule
  uint32_t sgpr_trap_reserve;  // TRAP_NUM_SGPRS if +trap-handler, else 0
  uint32_t max_barriers;       // getMaxWorkGroupsPerCU cap
  bool sgpr_limited;           // isSGPROccupancyLimited (IsaVersion.Major < 10)
  bool unified_vgpr_file;      // FeatureGFX90AInsts: arch and accum share 512
  bool exact;                  // false => generic fallback, values are a guess
};

// gfx_target_version decodes as major=(v/10000)%100, minor=(v/100)%100,
// step=v%100 -- all decimal (rocprofiler-sdk agent.h). The *name* renders the
// step in hex, which is why gfx90a is 90010 and not 9001a.
//
// Returns nullopt for targets we cannot model (anything with major != 9).
// The caller MUST then emit no occupancy stats rather than guess: the
// coincidence that makes a naive formula work on CDNA does not hold on
// gfx10/11/12, which have wave32, a much larger VGPR file and different
// allocation granules.
std::optional<AmdGpuTargetConstants> LookupTargetConstants(
    uint32_t gfx_target_version);

// The number of VGPRs the hardware actually charges a wave.
//
// Exposed rather than kept private because the max() branch is load-bearing
// rather than defensive -- see the gfx908 trap documented in the .cc -- so any
// caller that needs a register count must route through here instead of adding
// arch and accum itself.
uint32_t UnifiedVgprCount(const AmdGpuTargetConstants& tc, uint32_t arch,
                          uint32_t accum);

// Everything that determines theoretical occupancy for one kernel launch.
// Doubles as the key of the collector's memoization cache.
struct RocmDeviceOccupancyParams {
  // Kernel symbol data (DEVICE_KERNEL_SYMBOL_REGISTER callback).
  uint32_t arch_vgpr_count = 0;   // callback_tracing.h arch_vgpr_count
  uint32_t accum_vgpr_count = 0;  // callback_tracing.h accum_vgpr_count (AGPRs)
  uint32_t sgpr_count = 0;        // callback_tracing.h, already 16-granulated

  // Kernel dispatch record (rocprofiler_kernel_dispatch_info_t).
  uint32_t block_size = 0;  // workgroup_size.x*y*z, zero dims treated as 1
  // group_segment_size from the *dispatch* record: the TOTAL LDS per workgroup
  // (static + runtime). Do NOT add the symbol's group_segment_size on top,
  // that would double-count.
  uint32_t smem_bytes = 0;

  // Device identity. Everything else comes from LookupTargetConstants; the
  // agent's own capability scalars carry no discriminating power here because
  // they are invariant per device, while this field is the one that actually
  // distinguishes a gfx90a from a gfx950 from a gfx1100.
  uint32_t gfx_target_version = 0;

  friend bool operator==(const RocmDeviceOccupancyParams& a,
                         const RocmDeviceOccupancyParams& b) noexcept {
    return std::tie(a.arch_vgpr_count, a.accum_vgpr_count, a.sgpr_count,
                    a.block_size, a.smem_bytes, a.gfx_target_version) ==
           std::tie(b.arch_vgpr_count, b.accum_vgpr_count, b.sgpr_count,
                    b.block_size, b.smem_bytes, b.gfx_target_version);
  }

  friend bool operator!=(const RocmDeviceOccupancyParams& a,
                         const RocmDeviceOccupancyParams& b) noexcept {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const RocmDeviceOccupancyParams& p) {
    return H::combine(std::move(h), p.arch_vgpr_count, p.accum_vgpr_count,
                      p.sgpr_count, p.block_size, p.smem_bytes,
                      p.gfx_target_version);
  }
};

// CONSUMERS. There are none in production yet: this library has exactly one
// caller in the tree, rocm_occupancy_test.cc, and the collector wiring lands
// in the follow-up change. `limiter` in particular costs ~30 lines of the
// subtlest reasoning in the .cc and is here because it is the number an
// engineer can act on. If the wiring is abandoned, delete the attribution
// block rather than leaving it to rot untested against a real consumer.
struct OccupancyStats {
  // Percent of THREADS per CU: active_blocks * block_size * 100 /
  // (max_waves_per_cu * wave_front_size). Matches CUPTI exactly.
  double occupancy_pct = 0.0;
  // Waves resident per SIMD (== per EU). The unit LLVM's evaluateOccupancy and
  // -Rpass-analysis=kernel-resource-usage report.
  double waves_per_simd = 0.0;
  uint32_t active_waves_per_cu = 0;
  uint32_t active_blocks_per_cu = 0;
  OccupancyLimiter limiter = OccupancyLimiter::kNone;
  uint32_t total_vgprs = 0;  // what LLVM calls NumVGPRs: the unified charge
  // Blocks needed to fill the whole DEVICE at this occupancy:
  // active_blocks_per_cu * cu_count. 0 if cu_count was unknown.
  //
  // Same units as CUDA's min_grid_size but NOT the same quantity.
  // cudaOccupancyMaxPotentialBlockSize reports the grid needed at the block
  // size it *suggests*, which is a tuning hint independent of the launch; this
  // is the grid needed at the block size actually launched. For a
  // low-occupancy MFMA kernel the CUDA number is large and this one is small.
  // A caller wiring this into the shared kOccupancyMinGridSize stat is
  // changing what that stat means on ROCm, and should say so.
  uint32_t min_grid_size = 0;
  // Propagated from AmdGpuTargetConstants::exact. False means the target was
  // not in the table and the generic gfx9 fallback supplied the constants, so
  // every number above is an estimate that can be wrong in EITHER direction --
  // the fallback raises max_waves_per_simd (8 -> 10) while lowering
  // total_vgprs (512 -> 256), so which way it moves depends on the kernel.
  //
  // A consumer that renders an estimate identically to a measurement is
  // lying by omission. Either mark it in the UI or drop the row; do not
  // silently average the two together.
  bool exact = true;
};

// Pure function, no device access.
//
// nullopt means "cannot model" -- an unrecognised target, a missing block
// size, no register data at all, input the version-skew guard rejects, or a
// geometry the hardware could not have made resident (a workgroup that needs
// more wave slots or more registers than a CU has). The caller must then emit
// no occupancy stats, NOT zero ones.
std::optional<OccupancyStats> GetOccupancy(
    const RocmDeviceOccupancyParams& params, uint32_t cu_count);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_OCCUPANCY_H_
