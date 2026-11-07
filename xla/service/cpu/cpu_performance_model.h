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

#ifndef XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_

#include <cstdint>
#include <string>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla::cpu {

struct EstimateRunTimeData {
  int64_t flops;
  int64_t bytes_read;
  int64_t bytes_written;
  absl::Duration read_time;
  absl::Duration write_time;
  absl::Duration compute_time;
  absl::Duration exec_time;

  // Returns an estimate that is guaranteed to be larger than any real runtime.
  static EstimateRunTimeData Infinite();

  // Returns true if the estimate is guaranteed to be larger than any real
  // runtime.
  bool IsInfinite() const { return exec_time == absl::InfiniteDuration(); }

  std::string ToString() const;
};

// Analytical performance model of loop fusion on CPU, following
// GpuPerformanceModel: a fusion's run time combines the time to compute its
// flops and the time to read its operands and write its outputs.
class CpuPerformanceModel {
 public:
  struct RunTimes {
    absl::Duration time_unfused;
    absl::Duration time_fused;
  };

  // Speedup of reading from the cache over reading from memory.
  static constexpr float kCacheSpeedup = 4;

  // See GpuPerformanceModelBase::kMemoryComputeParallelism.
  static constexpr double kMemoryComputeParallelism = 0.95;

  // Parameters of a typical multi-core x86 host.
  static se::DeviceDescription DefaultDeviceInfo();

  explicit CpuPerformanceModel(const se::DeviceDescription& device_info)
      : device_info_(device_info) {}

  const se::DeviceDescription& device_info() const { return device_info_; }

  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr,
      const CpuHloCostAnalysis* cost_analysis) const;

  // Estimates the run time of `consumer` with `producer` fused into it.
  absl::Duration EstimateRunTimeForFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const CpuHloCostAnalysis* cost_analysis) const;

  // Estimates the run time of `producer` and `fused_consumers` with and without
  // fusing `producer` into each of `fused_consumers`.
  RunTimes EstimateRunTimes(
      const HloInstruction* producer, const CpuHloCostAnalysis* cost_analysis,
      absl::Span<const HloInstruction* const> fused_consumers) const;

  // Returns bytes accessed of operand output by instruction. Returns 0, if the
  // operand is not used by the instruction.
  static int64_t GetOperandBytesAccessed(
      const CpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
      const HloInstruction* operand);

  // Returns utilization of operand by instruction. Returns 0, if the operand is
  // not used by the instruction.
  static float GetOperandUtilization(const CpuHloCostAnalysis* cost_analysis,
                                     const HloInstruction* instr,
                                     const HloInstruction* operand);

  // Returns bytes accessed of operand after producer and consumer are fused
  // together.
  static int64_t GetSharedOperandBytesAccessed(
      const CpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
      const HloInstruction* consumer, const HloInstruction* operand);

  // Estimates the time to read `n_bytes_total` bytes of an operand of size
  // `n_bytes_net`. The first `n_bytes_net` bytes are read from memory, or from
  // the cache if they fit into it; the remaining bytes are read from the cache.
  static absl::Duration ReadTimeWithDRAMHeuristic(
      const se::DeviceDescription& device_info, int64_t n_bytes_net,
      int64_t n_bytes_total);

  static absl::Duration WriteTime(const se::DeviceDescription& device_info,
                                  int64_t bytes_written);

  static absl::Duration ComputeTime(const se::DeviceDescription& device_info,
                                    int64_t flops);

  static absl::Duration CombineComputeAndMemoryAccessTime(
      absl::Duration compute_time, absl::Duration memory_access_time);

 private:
  se::DeviceDescription device_info_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_
