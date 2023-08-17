/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_

#include <optional>
#include <vector>

#include "absl/time/time.h"
#include "xla/service/gpu/gpu_hlo_cost_analysis.h"

namespace xla {
namespace gpu {

struct EstimateRunTimeData {
  int64_t flops;
  float bytes_written;
  float elements_out;
  absl::Duration write_time;
  absl::Duration exec_time;
};

class GpuPerformanceModel {
 public:
  struct RunTimes {
    absl::Duration time_unfused;
    absl::Duration time_fused;
  };

  static EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis);

  static RunTimes EstimateRunTimes(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      std::vector<HloInstruction*> fused_consumers = {},
      bool multi_output = false);

  // Writes estimated execution time to FusionBackendConfig.reification_cost.
  static void RecordEstimatedRunTime(HloInstruction* instruction,
                                     const GpuHloCostAnalysis* cost_analysis);
  static absl::Duration ComputeTime(const GpuDeviceInfo& gpu_device_info,
                                    int64_t n_flops, int64_t n_threads);

  static absl::Duration ProducerInputAccessTime(
      const GpuHloCostAnalysis* cost_analysis,
      const GpuDeviceInfo& gpu_device_info, const HloInstruction* producer,
      const HloInstruction* fused_consumer = nullptr);
};

class GpuPerformanceWithCollectiveModel : public GpuPerformanceModel {
 public:
  // Different algorithms that can be used to perform the collective.
  enum CollectiveAlgo {
    RING = 0,
    TREE,
  };

  // Table for max system bandwidths.
  // Index 1 is for well-known GPU link BWs, used for intra-node estimate.
  static constexpr double low_latency_max_bandwidths[3] = {
      39.0 /* Volta-N1 */, 87.7 /* Ampere-N1 */, 87.7 /* Hopper-N1 */
  };

  static constexpr double per_channel_max_ring_LL128_bandwidths[3] = {
      20.0 /* Volta (N1/N2/N4) */,
      20.0 /* Ampere (N1/N2/N4) */,
      36.7 /* Hopper (N1/N2/N4) */,
  };

  // Nvlink bandwidth for different compute cap.
  // Note this is per lane bandwidth.
  static constexpr double sm60_nvlink_bw = 18.0;
  static constexpr double sm70_nvlink_bw = 20.0;
  static constexpr double sm80_nvlink_bw = 20.0;
  static constexpr double sm90_nvlink_bw = 20.0;

  // PCIE bandwidth
  static constexpr double pci_bw = 12.0;

  // Discount factor for ring algorithm
  static constexpr double ring_algo_factor = 0.92;

  // Different tiers for intra-node bandwidth.
  static constexpr double intra_node_speeds[13] = {
      40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0};
  // SM90 has different bandwidths.
  static constexpr double intra_node_speeds_sm90[9] = {
      60.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 6.0, 3.0};

  // Maximum number of channels allowed by NCCL
  static constexpr int64_t max_nchannels_ring = 16;

  // ll128 is by default enabled for Volta, Ampere and Hopper, ll128 by default
  // launches 640 threads.
  static constexpr int64_t ll128_nthreads = 640;

  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const GpuDeviceInfo& gpu_device_info);
  // Returns NVLink bw in GB/s
  static float GetNvlinkBw(se::CudaComputeCapability compute_capability);

 private:
  static absl::Duration ComputeAllreduceTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const GpuDeviceInfo& gpu_device_info);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_
