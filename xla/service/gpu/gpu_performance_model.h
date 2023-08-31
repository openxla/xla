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

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

class GpuPerformanceModel {
 public:
  struct RunTimes {
    absl::Duration time_unfused;
    absl::Duration time_fused;
  };

  GpuPerformanceModel(const GpuHloCostAnalysis::Options& cost_analysis_options,
                      const GpuDeviceInfo* device_info)
      : cost_analysis_(cost_analysis_options, device_info) {}

  RunTimes EstimateRunTimes(const HloInstruction* producer,
                            std::vector<HloInstruction*> fused_users = {},
                            bool multi_output = false);

  // Writes estimated execution time to FusionBackendConfig.reification_cost.
  void RecordEstimatedRunTime(HloInstruction* instruction);

  Status Accept(HloComputation* computation) {
    return computation->Accept(&cost_analysis_);
  }

  Status RemoveInstruction(HloInstruction* instruction) {
    return cost_analysis_.RemoveInstruction(instruction);
  }

  Status RevisitInstruction(HloInstruction* instruction) {
    return cost_analysis_.RevisitInstruction(instruction);
  }

  float ProducerConsumerMergedTooLarge(const HloInstruction& producer,
                                       const HloInstruction& consumer) {
    return cost_analysis_.ProducerConsumerMergedTooLarge(producer, consumer);
  }

  const GpuDeviceInfo& device_info() { return *cost_analysis_.device_info_; }

 private:
  struct EstimateRunTimeData {
    int64_t flops;
    float bytes_written;
    float elements_out;
    absl::Duration write_time;
    absl::Duration exec_time;
  };

  // Returns whether a fusion uses the parameter at the given index elementwise
  // from its root.
  bool FusionUsesParameterElementwiseFromRoot(const HloInstruction* fusion,
                                              int parameter_index);

  // Estimate read time of n_bytes_total bytes from global memory on a
  // given GPU. Account for L1 / L2 cache speedup if the input's nominal size
  // n_bytes_net is small.
  absl::Duration ReadTime(int64_t n_bytes_net, int64_t n_bytes_total);

  // Tells input access time of the producer alone if fused_consumer
  // is not specified. Otherwise estimates the access time to producer's
  // inputs as if it is fused into the consumer.
  absl::Duration ProducerInputAccessTime(
      const HloInstruction* producer,
      const HloInstruction* fused_consumer = nullptr);

  // Use HloFusionAnalysis for computing the actual number of threads that the
  // IR emitter will use. Return std::nullopt if this data is not available.
  std::optional<int64_t> EstimateThreadCount(const HloInstruction* instr);

  absl::Duration ComputeTime(int64_t n_flops, int64_t n_threads);

  EstimateRunTimeData EstimateRunTimeImpl(const HloInstruction* instr);

  GpuHloCostAnalysis cost_analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_
