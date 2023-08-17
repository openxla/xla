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

#include "xla/service/gpu/gpu_performance_model.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace {

// Estimated values in the absence of easy ways to query them.
static constexpr absl::Duration kKernelLaunchOverhead = absl::Microseconds(5);
static constexpr float kL2CacheSpeedup = 2.5;
static constexpr float kL1CacheSpeedup = 8;
// A very conservative estimate. L1 size varies because it can be dynamically
// configured as shared memory; there is no easy way to query its actual size;
// also we do not count what occupies cache, but rather claim that what is
// much smaller than the cache size will likely stay in it.
// For reference, it can be up to 256 kB per SM on RTX A6000.
static constexpr float kL1CacheSizePerSM = 2 * 1024;

// Returns whether a fusion uses the parameter at the given index elementwise
// from its root.
bool FusionUsesParameterElementwiseFromRoot(
    const HloInstruction* fusion, int parameter_index,
    const GpuHloCostAnalysis* cost_analysis) {
  return cost_analysis->CommonElementwiseUtilization(
             fusion->fused_parameter(parameter_index),
             fusion->fused_expression_root()) == 1.f;
}

// Estimate read time of n_bytes_total bytes from global memory on a
// given GPU. Account for L1 / L2 cache speedup if the input's nominal size
// n_bytes_net is small.
absl::Duration ReadTime(const se::DeviceDescription& gpu_device_info,
                        int64_t n_bytes_net, int64_t n_bytes_total) {
  float bandwidth = gpu_device_info.memory_bandwidth();
  if (n_bytes_net < gpu_device_info.l2_cache_size()) {
    bandwidth *= kL2CacheSpeedup;
    if (n_bytes_net < kL1CacheSizePerSM * gpu_device_info.core_count()) {
      bandwidth *= kL1CacheSpeedup;
    }
  }
  return absl::Seconds(n_bytes_total / bandwidth);
}

int64_t GetMaxNumberOfChannels(
    GpuPerformanceWithCollectiveModel::CollectiveAlgo algorithm) {
  int64_t max_nchannels = 0;
  switch (algorithm) {
    // Tree and Ring algos share the same max channel number.
    case GpuPerformanceWithCollectiveModel::RING:
    case GpuPerformanceWithCollectiveModel::TREE:
      max_nchannels = GpuPerformanceWithCollectiveModel::max_nchannels_ring;
      break;
  }
  const char* env = std::getenv("NCCL_MAX_NCHANNELS");
  if (env != nullptr) {
    int64_t max_nchannels_from_env;
    if (absl::SimpleAtoi(env, &max_nchannels_from_env)) {
      max_nchannels = std::min(max_nchannels_from_env, max_nchannels);
    }
  }
  return max_nchannels;
}

int64_t GetMinNumberOfChannels(
    GpuPerformanceWithCollectiveModel::CollectiveAlgo algorithm) {
  int64_t min_nchannels = 0;
  switch (algorithm) {
    // Tree and Ring algos share the same max channel number.
    case GpuPerformanceWithCollectiveModel::RING:
    case GpuPerformanceWithCollectiveModel::TREE:
      min_nchannels = 1;
      break;
  }
  const char* env = std::getenv("NCCL_MIN_NCHANNELS");
  if (env != nullptr) {
    int64_t min_nchannels_from_env;
    if (absl::SimpleAtoi(env, &min_nchannels_from_env)) {
      min_nchannels = std::min(min_nchannels_from_env, min_nchannels);
    }
  }
  return min_nchannels;
}

int GetNumThreads(int warp_size, int min_num_threads, int max_num_threads,
                  int default_num_threads) {
  int threads_from_env = default_num_threads;
  const char* env = std::getenv("NCCL_NTHREADS");
  if (env != nullptr) {
    CHECK(absl::SimpleAtoi(env, &threads_from_env));
  }
  int num_threads = threads_from_env;
  if (num_threads > 0) {
    if (num_threads % warp_size != 0) {
      num_threads = max_num_threads;
    } else if (num_threads > max_num_threads) {
      num_threads = max_num_threads;
    } else if (num_threads < min_num_threads) {
      num_threads = min_num_threads;
    }
  } else {
    num_threads = default_num_threads;
  }
  return num_threads;
}

float GetMaxSysBwFromGpu(const GpuDeviceInfo& gpu_device_info,
                         const double* bandwidths_table) {
  switch (gpu_device_info.cuda_comp_capability_major) {
    // Volta
    case se::CudaComputeCapability::VOLTA:
      return bandwidths_table[0];
    // Ampere
    case se::CudaComputeCapability::AMPERE:
      return bandwidths_table[1];
    // Hopper
    case se::CudaComputeCapability::HOPPER:
      return bandwidths_table[2];
  }
  return -1;
}

// Use HloFusionAnalysis for computing the actual number of threads that the
// IR emitter will use. Return std::nullopt if this data is not available.
std::optional<int64_t> EstimateThreadCountForFusionOp(
    const HloInstruction* instr, const GpuDeviceInfo& gpu_device_info,
    std::optional<se::CudaComputeCapability> cc) {
  auto fusion = DynCast<const HloFusionInstruction>(instr);
  if (fusion != nullptr && cc.has_value()) {
    auto analysis =
        HloFusionAnalysis::Create(fusion, &gpu_device_info, cc.value());
    if (analysis.ok()) {
      auto launch_dimensions = analysis->GetLaunchDimensions();
      if (launch_dimensions.ok()) {
        return launch_dimensions->launch_bound();
      }
    }
  }
  return std::nullopt;
}
}  // namespace

/*static*/ EstimateRunTimeData
GpuPerformanceModel::EstimateRunTimeForInstruction(
  const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  int64_t flops = cost_analysis->flop_count(*instr);
  int64_t bytes_written = cost_analysis->output_bytes_accessed(*instr);
  int64_t bytes_read = cost_analysis->bytes_accessed(*instr) - bytes_written;
  int64_t num_threads = ShapeUtil::ElementsInRecursive(instr->shape());

  auto fusion_analysis = HloFusionAnalysis::Create(
      FusionBackendConfig::default_instance(), {instr}, DefaultFusionBoundaryFn,
      cost_analysis->device_info_);
  if (fusion_analysis.ok() && fusion_analysis->GetLaunchDimensions().ok()) {
    VLOG(10) << "Launch dimensions for unfused producer: "
             << fusion_analysis->GetLaunchDimensions()->ToString() << ".";
    num_threads = fusion_analysis->GetLaunchDimensions()->launch_bound();
  }

  absl::Duration compute_time = ComputeTime(*device_info, flops, num_threads);
  absl::Duration read_time =
      ProducerInputAccessTime(cost_analysis, *device_info, /*producer=*/instr);
  absl::Duration write_time =
      absl::Seconds(1.0f * bytes_written / device_info->memory_bandwidth());
  absl::Duration exec_time = std::max(compute_time, read_time + write_time);

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "FLOPs: " << flops;
    LOG(INFO) << "Bytes read: " << bytes_read;
    LOG(INFO) << "Bytes written: " << bytes_written;
    LOG(INFO) << "Num threads:" << num_threads;
    LOG(INFO) << "Compute time: " << compute_time;
    LOG(INFO) << "Input read time: " << read_time;
    LOG(INFO) << "Output write time: " << write_time;
  }

  return {flops, bytes_written, num_threads, write_time, exec_time};
}

// Tells input access time of the producer alone if fused_consumer
// is not specified. Otherwise estimates the access time to producer's
// inputs as if it is fused into the consumer.
/*static*/ absl::Duration GpuPerformanceModel::ProducerInputAccessTime(
    const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info,
    const HloInstruction* producer,
    const HloInstruction* fused_consumer = nullptr) {
  absl::Duration ret = absl::ZeroDuration();
  float producer_output_utilization = 1.f;
  ConstHloInstructionSet consumer_operands;
  if (fused_consumer) {
    producer_output_utilization = cost_analysis->operand_utilization(
        *fused_consumer, fused_consumer->operand_index(producer));
    for (const HloInstruction* op : fused_consumer->operands()) {
      consumer_operands.insert(op);
    }
  }
  for (int i = 0; i < producer->operand_count(); ++i) {
    // Information about data read taking into account utilization.
    // If `operand_utilization` is 0, `operand_bytes_accessed` should be also 0.
    int64_t operand_bytes_accessed =
        cost_analysis->operand_bytes_accessed(*producer, i);
    float operand_utilization =
        cost_analysis->operand_utilization(*producer, i);

    // An estimate how much data would need to fit into L1/L2 cache to speed up
    // the operand access.
    // If `operand_utilization` < 1, only a part of the full operand size should
    // be read. Otherwise, `operand_bytes_accessed / operand_utilization` is the
    // size of the operand without reuse.
    int64_t n_bytes_net = std::llround(operand_bytes_accessed /
                                       std::max(operand_utilization, 1.0f));

    // Look for common operands of producer and consumer that are accessed
    // more efficiently on merge:
    // 1) Producer has to use the common operand elementwise from its root if
    //    it is a fusion or just be an elementwise instruction.
    // 2) Consumer has to have common elementwise roots for the producer
    //    and the common operand if it is a fusion or just be an elementwise
    //    instruction.
    float common_utilization = 0;
    if (consumer_operands.count(producer->operand(i)) &&
        (producer->IsElementwise() ||
         (producer->opcode() == HloOpcode::kFusion &&
          FusionUsesParameterElementwiseFromRoot(producer, i,
                                                 cost_analysis)))) {
      if (fused_consumer->opcode() == HloOpcode::kFusion) {
        int64_t consumer_idx_of_common_operand =
            fused_consumer->operand_index(producer->operand(i));
        int64_t consumer_idx_of_producer =
            fused_consumer->operand_index(producer);
        common_utilization = cost_analysis->CommonElementwiseUtilization(
            fused_consumer->fused_parameter(consumer_idx_of_common_operand),
            fused_consumer->fused_parameter(consumer_idx_of_producer));
      } else {
        if (fused_consumer->IsElementwise()) {
          common_utilization = 1.f;
        }
      }
    }
    CHECK_LE(common_utilization, producer_output_utilization);
    float n_bytes_total = operand_bytes_accessed *
                          (producer_output_utilization - common_utilization);
    ret += ReadTime(gpu_device_info, n_bytes_net, n_bytes_total);
  }
  return ret;
}

// Uses HloFusionAnalysis for computing the actual number of threads that the
// IR emitter for the fusion of `producer` and `consumer` will use. Returns
// std::nullopt if this data is not available.
std::optional<int64_t> EstimateFusionThreadCount(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info) {
  auto roots = consumer.opcode() == HloOpcode::kFusion
                   ? GetFusionRoots(*consumer.fused_instructions_computation())
                   : std::vector<const HloInstruction*>{&consumer};
  auto fusion_analysis = HloFusionAnalysis::Create(
      FusionBackendConfig::default_instance(), std::move(roots),
      MakeProducerConsumerFusion(producer, consumer), &device_info);
  if (fusion_analysis.ok()) {
    VLOG(10) << "Fusion analysis for " << producer.ToString() << " and "
             << consumer.ToString() << " successful.";
    auto launch_dimensions = fusion_analysis->GetLaunchDimensions();
    if (launch_dimensions.ok()) {
      VLOG(10) << "Launch dimensions for " << producer.ToString() << " and "
               << consumer.ToString() << ": " << launch_dimensions->ToString()
               << ".";
      return launch_dimensions->launch_bound();
    }
  } else {
    VLOG(10) << "Fusion analysis for " << producer.ToString() << " and "
             << consumer.ToString() << " unsuccessful.";
  }

  return std::nullopt;
}

absl::Duration GpuPerformanceModel::ComputeTime(const se::DeviceDescription& gpu_device_info,
                           int64_t flops, int64_t num_threads) {
  int64_t fpu_count =
      gpu_device_info.core_count() * gpu_device_info.fpus_per_core();
  int64_t n_threads_active = std::min(num_threads, fpu_count);
  int64_t flop_per_ns_per_fpu = gpu_device_info.clock_rate_ghz() * /*fma:*/ 2;
  int64_t flop_per_ns_effective = flop_per_ns_per_fpu * n_threads_active;
  return absl::Nanoseconds(1.0f * flops / flop_per_ns_effective);
}

GpuPerformanceModel::RunTimes GpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    std::vector<HloInstruction*> fused_consumers, bool multi_output) {
  VLOG(8) << "Producer: " << producer->name();
  if (producer->opcode() == HloOpcode::kFusion) {
    VLOG(10) << producer->fused_instructions_computation()->ToString();
  }

  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  EstimateRunTimeData producer_data =
      EstimateRunTimeForInstruction(producer, cost_analysis);

  int64_t fused_consumer_count = fused_consumers.size();
  float total_producer_utilization = 0;

  absl::Duration exec_time_fused = absl::ZeroDuration();
  absl::Duration producer_output_read_time_unfused = absl::ZeroDuration();
  for (const HloInstruction* fused_consumer : fused_consumers) {
    float utilization_by_this_consumer = cost_analysis->operand_utilization(
        *fused_consumer, fused_consumer->operand_index(producer));
    total_producer_utilization += utilization_by_this_consumer;

    // The model ignores consumer computation and output writes. The main goal
    // of the model is to compare estimates of fused and unfused cases. Since
    // epilog of the consumers remains unchanged in both bases, we only consider
    // duplication of the producer computation and repeated access to producer
    // inputs.
    //
    // TODO(shyshkov): Add calculations for consumer epilogue in the formula to
    // make it complete.
    int64_t num_threads =
        std::llround(producer_data.num_threads * utilization_by_this_consumer);
    if (auto thread_count = EstimateFusionThreadCount(
            *producer, *fused_consumer, *cost_analysis->device_info_)) {
      num_threads = std::min(*thread_count, num_threads);
    }
    absl::Duration compute_time_by_this_consumer = ComputeTime(
        *device_info, producer_data.flops * utilization_by_this_consumer,
        num_threads);

    absl::Duration input_access_time_by_this_consumer =
        ProducerInputAccessTime(cost_analysis, *device_info, producer,
                                /*fused_consumer=*/fused_consumer);

    exec_time_fused += std::max(compute_time_by_this_consumer,
                                input_access_time_by_this_consumer);

    int64_t n_bytes_total = std::llround(producer_data.bytes_written *
                                         utilization_by_this_consumer);
    int64_t n_bytes_net = std::min(producer_data.bytes_written, n_bytes_total);
    producer_output_read_time_unfused +=
        ReadTime(*device_info, n_bytes_net, n_bytes_total);
  }

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumer_count + 1) +
      producer_data.exec_time + producer_output_read_time_unfused;

  absl::Duration time_fused =
      kKernelLaunchOverhead * fused_consumer_count + exec_time_fused;
  // Multi-output fusion still writes the initial output of the producer.
  // For now assume that the producer's output does not need to be recomputed.
  if (multi_output) {
    time_fused += producer_data.write_time;
  }

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Consumer count: " << fused_consumer_count;
    LOG(INFO) << "Utilization of producer output: "
              << total_producer_utilization;
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

void GpuPerformanceModel::RecordEstimatedRunTime(
    HloInstruction* instruction, const GpuHloCostAnalysis* cost_analysis) {
  DCHECK(Cast<const HloFusionInstruction>(instruction)) << "expected fusion";
  DCHECK(cost_analysis != nullptr) << "expected cost analysis";

  EstimateRunTimeData data =
      EstimateRunTimeForInstruction(instruction, cost_analysis);
  double cycles = absl::ToDoubleNanoseconds(data.exec_time) *
                  cost_analysis->device_info_->clock_rate_ghz();

  auto backend_config = instruction->backend_config<FusionBackendConfig>();
  TF_CHECK_OK(backend_config.status()) << instruction->ToString();
  backend_config->mutable_reification_cost()->set_end_to_end_cycles(cycles);
  TF_CHECK_OK(instruction->set_backend_config(*backend_config));

  VLOG(8) << "RecordEstimatedRunTime: " << instruction->ToString();
}

// Returns NVLink bw in GB/s
/*static*/
float GpuPerformanceWithCollectiveModel::GetNvlinkBw(
    se::CudaComputeCapability compute_capability) {
  return compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER)
             ? sm90_nvlink_bw
         : compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE)
             ? sm80_nvlink_bw
         : compute_capability.IsAtLeast(se::CudaComputeCapability::VOLTA)
             ? sm70_nvlink_bw
         : compute_capability.IsAtLeast(se::CudaComputeCapability::PASCAL_)
             ? sm60_nvlink_bw
             : sm80_nvlink_bw;
}

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeAllreduceTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const GpuDeviceInfo& gpu_device_info) {
  float elements_out = ShapeUtil::ElementsInRecursive(instr.shape());
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  absl::Duration total_time = kKernelLaunchOverhead;

  int64_t size_of_speed_array = (sizeof(intra_node_speeds) / sizeof(double));
  int64_t size_of_sm90_speed_array =
      (sizeof(intra_node_speeds_sm90) / sizeof(double));

  int num_speeds = gpu_device_info.cuda_comp_capability_major >=
                               se::CudaComputeCapability::HOPPER &&
                           gpu_device_info.cuda_comp_capability_minor >= 0
                       ? size_of_sm90_speed_array
                       : size_of_speed_array;
  const double* speeds = gpu_device_info.cuda_comp_capability_major >=
                                     se::CudaComputeCapability::HOPPER &&
                                 gpu_device_info.cuda_comp_capability_minor >= 0
                             ? intra_node_speeds_sm90
                             : intra_node_speeds;

  int speed_index = 0;
  float max_sys_bw =
      GetMaxSysBwFromGpu(gpu_device_info, low_latency_max_bandwidths);

  CHECK_GT(max_sys_bw, 0);

  while ((speed_index < num_speeds - 1) && speeds[speed_index] > max_sys_bw) {
    speed_index++;
  }
  float bw_intra_node = speeds[speed_index];
  int64_t num_devices = cost_analysis->NumOfDevices(instr);

  int64_t min_nchannels =
      std::max(num_devices, GetMinNumberOfChannels(CollectiveAlgo::RING));
  int64_t num_channels =
      std::max(min_nchannels, GetMaxNumberOfChannels(CollectiveAlgo::RING));
  int default_threads =
      (bw_intra_node * num_channels <= pci_bw) ? 256 : ll128_nthreads;

  int warp_size = gpu_device_info.threads_per_warp;
  int num_threads = GetNumThreads(warp_size, ll128_nthreads / 4, ll128_nthreads,
                                  default_threads);

  // Since channels are pipelined together, compute time will only occur as in a
  // single channel.
  absl::Duration compute_time_per_channel =
      ComputeTime(gpu_device_info,
                  cost_analysis->flop_count(instr) / num_channels, num_threads);
  total_time += compute_time_per_channel;
  double bus_bandwidth = bw_intra_node * num_channels;

  // Get per channel LL128 ring bandwidth
  double per_channel_ring_ll128_Bw = GetMaxSysBwFromGpu(
      gpu_device_info, per_channel_max_ring_LL128_bandwidths);
  ;
  bus_bandwidth = std::min(
      bus_bandwidth * ring_algo_factor /*discount factor for ring algo*/,
      num_channels * per_channel_ring_ll128_Bw);
  double actual_bandwidth = bus_bandwidth * cost_analysis->ScalingRatio(instr);

  absl::Duration communication_time = absl::Microseconds(
      cost_analysis->bytes_accessed(instr) / (1e6 * actual_bandwidth));
  total_time += communication_time;
  return total_time;
}

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeCollectiveTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const GpuDeviceInfo& gpu_device_info) {
  if (cost_analysis->NumOfDevices(instr) == 1) {
    VLOG(8) << "Returning only kernel launch overhead for a single partition.";
    return kKernelLaunchOverhead;
  }

  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::Microseconds(0);
  }
  switch (instr.opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
      return ComputeAllreduceTime(instr, cost_analysis, gpu_device_info);
    default: {
      LOG(WARNING)
          << "Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      return kKernelLaunchOverhead;
    }
  }
}

}  // namespace gpu
}  // namespace xla
