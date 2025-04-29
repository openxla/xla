/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_collective_performance_model.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/time/time.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/nvml/include/nvml.h"
#endif  // GOOGLE_CUDA
namespace xla {
namespace gpu {

namespace {

const std::vector<double>& GetSpeeds(
    const stream_executor::CudaComputeCapability& compute_cap) {
  // Different tiers for intra-node bandwidth.
  static const std::vector<double> kIntraNodeSpeeds = {
      40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0};
  // SM90 has different bandwidths.
  static std::vector<double> kIntraNodeSpeedsSm90 = {
      60.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 6.0, 3.0};
  return compute_cap.major >= se::CudaComputeCapability::HOPPER
             ? kIntraNodeSpeedsSm90
             : kIntraNodeSpeeds;
}

const std::vector<double>& GetSpeeds(
    const stream_executor::RocmComputeCapability& compute_cap) {
  static const std::vector<double> intraNodeSpeeds = {
      1225.0, 1000.0, 900.0, 800.0, 700.0, 600.0, 500.0,
      400.0,  300.0,  200.0, 100.0, 80.0,  60.0};
  return intraNodeSpeeds;
}

// Different algorithms that can be used to perform the collective.
enum class CollectiveAlgo {
  RING = 0,
  TREE,
};

struct CudaBandwidthSettings {
  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  static constexpr std::array<double, 5> kLowLatencyMaxBandwidths = {
      39.0 /* Volta */,      87.7 /* Ampere */,    141.0 /* Hopper */,
      141.0 /* Blackwell */, 141.0 /* next-gen */,
  };

  // Max bandwidth in GB/s for ring low latency 128 algorithm per channel on a
  // single-node
  static constexpr std::array<double, 5> kPerChannelMaxRingLL128Bandwidths = {
      20.0 /* Volta */,     20.0 /* Ampere */,   36.7 /* Hopper */,
      36.7 /* Blackwell */, 36.7 /* next-gen */,
  };

  // Nvlink unidirectional bandwidth for different compute cap. Note this is per
  // lane bandwidth.
  static constexpr double kSm60NvlinkBandwidth = 18.0;
  static constexpr double kSm70NvlinkBandwidth = 20.0;
  static constexpr double kSm80NvlinkBandwidth = 20.0;
  static constexpr double kSm90NvlinkBandwidth = 20.0;

  // PCIE bandwidth for PCI Gen3 x16
  static constexpr double kPciBandwidth = 12.0;

  // Discount factor for ring algorithm
  static constexpr double kRingAlgorithmDiscountFactor = 0.92;

  // Maximum number of channels allowed by NCCL
  static constexpr int64_t kMaxNumChannelsRing = 16;

  // ll128 is by default enabled for Volta, Ampere and Hopper, ll128 by default
  // launches 640 threads.
  static constexpr int64_t kLL128NumThreads = 640;
};

struct RocmBandwidthSettings {
  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  static constexpr std::array<double, 4> kLowLatencyMaxBandwidths = {
      122.0,  // MI100: ~122 GB/s peak (via Infinity Fabric)
      220.0,  // MI200: dual-die, up to ~220 GB/s combined
      340.0,  // MI300X: up to ~340 GB/s (HBM + IF bandwidth)
      340.0   // next-gen: placeholder same as MI300
  };

  // Max bandwidth in GB/s for ring low latency 128 algorithm per channel on a
  // single-node
  static constexpr std::array<double, 5> kPerChannelMaxRingLL128Bandwidths = {
      20.0,  // legacy (placeholder, e.g., Vega/Volta)
      25.0,  // MI100
      35.0,  // MI200
      45.0,  // MI300
      45.0   // next-gen: same as MI300
  };

  // Nvlink unidirectional bandwidth for different compute cap. Note this is per
  // lane bandwidth.
  static constexpr double kMi300NvlinkBandwidth = 100.0;

  // PCIE bandwidth for PCI Gen3 x16
  static constexpr double kPciBandwidth = 12.0;

  // Discount factor for ring algorithm
  static constexpr double kRingAlgorithmDiscountFactor = 0.7;

  // Maximum number of channels allowed by NCCL
  static constexpr int64_t kMaxNumChannelsRing = 16;

  // ll128 is by default enabled for Volta, Ampere and Hopper, ll128 by default
  // launches 640 threads.
  static constexpr int64_t kLL128NumThreads = 256;
};

static constexpr absl::Duration kNcclKernelLaunchOverhead =
    absl::Microseconds(5);

int64_t GetNcclMaxNumChannels(CollectiveAlgo algorithm) {
  int64_t max_nchannels = 0;
  switch (algorithm) {
      // Tree and Ring algos share the same max channel number.
    case CollectiveAlgo::RING:
    case CollectiveAlgo::TREE:
      max_nchannels = CudaBandwidthSettings::kMaxNumChannelsRing;
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

int64_t GetMinNumberOfChannels(CollectiveAlgo algorithm) {
  int64_t min_nchannels = 0;
  switch (algorithm) {
      // Tree and Ring algos share the same min channel number.
    case CollectiveAlgo::RING:
    case CollectiveAlgo::TREE:
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

float GetMaxSysBwFromGpu(const se::CudaComputeCapability cc,
                         const double* bandwidths_table) {
  switch (cc.major) {
    case cc.VOLTA:
      return bandwidths_table[0];
    case cc.AMPERE:
      return bandwidths_table[1];
    case cc.HOPPER:
      return bandwidths_table[2];
    case cc.BLACKWELL:
      return bandwidths_table[3];
    default:
      return bandwidths_table[4];
  }
}

float GetMaxSysBwFromGpu(const se::RocmComputeCapability& cc,
                         const double* bandwidths_table) {
  if (cc.gfx9_mi100()) {
    return bandwidths_table[0];
  } else if (cc.gfx9_mi200()) {
    return bandwidths_table[1];
  } else if (cc.gfx9_mi300()) {
    return bandwidths_table[2];
  } else {
    return bandwidths_table[3];
  }
}

float GetNvlinkBw(const se::RocmComputeCapability& compute_capability) {
  return RocmBandwidthSettings::kMi300NvlinkBandwidth;
}

// Returns NVLink bw in GB/s
float GetNvlinkBw(const se::CudaComputeCapability& compute_capability) {
  return compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER)
             ? CudaBandwidthSettings::kSm90NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE)
             ? CudaBandwidthSettings::kSm80NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::VOLTA)
             ? CudaBandwidthSettings::kSm70NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::PASCAL_)
             ? CudaBandwidthSettings::kSm60NvlinkBandwidth
             : CudaBandwidthSettings::kSm80NvlinkBandwidth;
}

template <typename ComputeCapability, typename GpuBandwidthSettings>
absl::Duration ComputeAllreduceTimeImpl(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info,
    const ComputeCapability& compute_cap,
    const GpuBandwidthSettings& bandwidth_settings) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  absl::Duration total_time = kNcclKernelLaunchOverhead;

  const auto& speeds = GetSpeeds(compute_cap);

  int speed_index = 0;
  float max_sys_bw = GetMaxSysBwFromGpu(
      compute_cap, bandwidth_settings.kLowLatencyMaxBandwidths.data());

  CHECK_GT(max_sys_bw, 0);

  while ((speed_index < speeds.size() - 1) &&
         speeds[speed_index] > max_sys_bw) {
    speed_index++;
  }

  float bw_intra_node = speeds[speed_index];
  int64_t num_devices = cost_analysis->NumOfDevices(instr);

  int64_t min_nchannels =
      std::max(num_devices, GetMinNumberOfChannels(CollectiveAlgo::RING));
  int64_t num_channels =
      std::max(min_nchannels, GetNcclMaxNumChannels(CollectiveAlgo::RING));
  int default_threads =
      (bw_intra_node * num_channels <= bandwidth_settings.kPciBandwidth)
          ? 256
          : bandwidth_settings.kLL128NumThreads;

  int warp_size = gpu_device_info.threads_per_warp();
  int num_threads =
      GetNumThreads(warp_size, bandwidth_settings.kLL128NumThreads / 4,
                    bandwidth_settings.kLL128NumThreads, default_threads);

  // Since channels are pipelined together, compute time will only occur as in a
  // single channel.
  absl::Duration compute_time_per_channel =
      GpuPerformanceWithCollectiveModel::ComputeTime(
          gpu_device_info, cost_analysis->flop_count(instr) / num_channels,
          /*num_blocks=*/num_channels, /*num_threads_per_block=*/num_threads);
  total_time += compute_time_per_channel;

  uint32_t supported_p2p =
      GpuPerformanceWithCollectiveModel::CheckIfNvlinkSupportsP2P();

  if (supported_p2p == 0) {
    VLOG(8) << "Nvlink doesn't support p2p communication. Model will "
               "continue using default system bandwidth.";
  } else {
    VLOG(8) << "Nvlink supports p2p communication, setting intra node "
               "bandwidth to nvlink bw.";
    bw_intra_node = GetNvlinkBw(compute_cap);
  }

  double bus_bandwidth = bw_intra_node * num_channels;

  // Get per channel LL128 ring bandwidth
  double per_channel_ring_ll128_Bw = GetMaxSysBwFromGpu(
      compute_cap, bandwidth_settings.kPerChannelMaxRingLL128Bandwidths.data());

  bus_bandwidth =
      std::min(bus_bandwidth * bandwidth_settings.kRingAlgorithmDiscountFactor,
               num_channels * per_channel_ring_ll128_Bw);
  double actual_bandwidth = bus_bandwidth * cost_analysis->ScalingRatio(instr);

  absl::Duration communication_time = absl::Milliseconds(
      cost_analysis->bytes_accessed(instr) / (1e6 * actual_bandwidth));
  total_time += communication_time;
  return total_time;
}

}  // namespace

/*static*/ bool GpuPerformanceWithCollectiveModel::InitNvml() {
#if GOOGLE_CUDA && (defined(PLATFORM_POSIX) || defined(PLATFORM_GOOGLE))
  void* libhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  CHECK(libhandle != nullptr) << "Failed to open libnvidia-ml.so.1";

  struct SymbolEntry {
    void** functor;
    char const* name;
  };

  std::vector<SymbolEntry> symbols = {
      {(void**)&xla_nvmlInit, "nvmlInit_v2"},
      {(void**)&xla_nvmlShutdown, "nvmlShutdown"},
      {(void**)&xla_nvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex"},
      {(void**)&xla_nvmlDeviceGetNvLinkCapability,
       "nvmlDeviceGetNvLinkCapability"},
  };
  for (SymbolEntry se : symbols) {
    *se.functor = dlsym(libhandle, se.name);
  }
  nvmlReturn_t init_result = xla_nvmlInit();
  return init_result == NVML_SUCCESS;
#elif TENSORFLOW_USE_ROCM
  return true;
#else
  return false
#endif  // GOOGLE_CUDA
}

/*static*/ bool GpuPerformanceWithCollectiveModel::ShutdownNvml() {
#if GOOGLE_CUDA
  nvmlReturn_t shutdown_result = xla_nvmlShutdown();
  return shutdown_result == NVML_SUCCESS;
#elif TENSORFLOW_USE_ROCM
  return true;
#else
  return false
#endif  // GOOGLE_CUDA
}

/*static*/ uint32_t
GpuPerformanceWithCollectiveModel::CheckIfNvlinkSupportsP2P() {
#if GOOGLE_CUDA
  // We will use nvml library to detect nvlink capability
  // to see if it supports p2p communication.
  // We first load libnvidia-ml.so and assign symbols to function pointers
  // to avoid linking errors.
  // Then gpu 0 will be used to query for nvlink capability, note that
  // we only look at link 0 of gpu 0 since all other links are assumed
  // to have the same capability.
  CHECK(InitNvml()) << "NVML init failed.";
  nvmlDevice_t nvml_device;
  nvmlReturn_t get_device_result =
      xla_nvmlDeviceGetHandleByIndex(0, &nvml_device);
  CHECK(get_device_result == NVML_SUCCESS);

  uint32_t supported_p2p = 0;

  nvmlReturn_t nvlink_cap_result = xla_nvmlDeviceGetNvLinkCapability(
      nvml_device, /*nvlink link number*/ 0, NVML_NVLINK_CAP_P2P_SUPPORTED,
      &supported_p2p);
  CHECK(nvlink_cap_result == NVML_SUCCESS ||
        nvlink_cap_result == NVML_ERROR_NOT_SUPPORTED);
  CHECK(ShutdownNvml()) << "NVML shutdown failed.";
  return supported_p2p;
#else
  return 0;
#endif  // GOOGLE_CUDA
}

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeAllreduceTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  absl::Duration total_time = kNcclKernelLaunchOverhead;
  absl::Duration result;
  const auto visitor = [&](const auto& cc) {
    using compute_capability =
        std::remove_const_t<std::remove_reference_t<decltype(cc)>>;
    if constexpr (std::is_same_v<compute_capability,
                                 stream_executor::CudaComputeCapability>) {
      result = ComputeAllreduceTimeImpl(instr, cost_analysis, gpu_device_info,
                                        cc, CudaBandwidthSettings{});
    } else if (std::is_same_v<compute_capability,
                              stream_executor::RocmComputeCapability>) {
      result = ComputeAllreduceTimeImpl(instr, cost_analysis, gpu_device_info,
                                        cc, RocmBandwidthSettings{});
    }
  };

  std::visit(visitor, gpu_device_info.gpu_compute_capability());
  return result;
}

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeCollectiveTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info) {
  if (cost_analysis->NumOfDevices(instr) == 1) {
    VLOG(8) << "Returning only kernel launch overhead for a single partition.";
    return kNcclKernelLaunchOverhead;
  }

  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::ZeroDuration();
  }
  switch (instr.opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
      return ComputeAllreduceTime(instr, cost_analysis, gpu_device_info);
    default: {
      LOG(WARNING)
          << "Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      return kNcclKernelLaunchOverhead;
    }
  }
}

}  // namespace gpu
}  // namespace xla
