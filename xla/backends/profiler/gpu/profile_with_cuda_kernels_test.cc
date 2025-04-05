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

#include "xla/backends/profiler/gpu/profile_with_cuda_kernels.h"
#include "cupti_collector.h"
#include "cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"


#include <atomic>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace xla {
namespace profiler {
namespace test {

using xla::profiler::CuptiInterface;
using xla::profiler::CuptiTracer;
using xla::profiler::CuptiTracerCollectorOptions;
using xla::profiler::CuptiTracerOptions;
using xla::profiler::CuptiWrapper;

// Needed to create different cupti tracer for each test cases.
class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer()
      : CuptiTracer(new CuptiErrorManager(std::make_unique<CuptiWrapper>())) {}
};


namespace {

std::atomic_uint64_t atomic_total_fp64 = 0;
std::atomic_uint64_t atomic_total_read = 0;
std::atomic_uint64_t atomic_total_write = 0;
bool skip_first = true;

void HandleRecords(struct PmSamplingDecodeInfo* info) {
  // Validate some samples were recorded
  EXPECT_GT(info->num_completed, 0);

  auto back = info->sampler_ranges.back();
  auto front = info->sampler_ranges.front();
  double ranges_duration = back.end_timestamp_ns -
      front.start_timestamp_ns;
  double ns_per_sample = ranges_duration / info->num_completed;
  // First pass may have large initial sample
  if (skip_first) {
    skip_first = false;
  } else {
    EXPECT_GT(ns_per_sample, 500000. * .95);
    EXPECT_LT(ns_per_sample, 500000. * 1.05);
  }

  for (int i = 0; i < info->metrics.size(); i++) {
    double sum = 0;
    for (int j = 0; j < info->sampler_ranges.size(); j++) {
      sum += info->sampler_ranges[j].metric_values[i];
    }

    if (strcmp("sm__inst_executed_pipe_fp64.sum", info->metrics[i]) == 0) {
      atomic_total_fp64 += sum;
    }
    else if (strcmp("pcie__read_bytes.sum", info->metrics[i]) == 0) {
      atomic_total_read += sum;
    }
    else if (strcmp("pcie__write_bytes.sum", info->metrics[i]) == 0) {
      atomic_total_write += sum;
    }
  }
  return; };

TEST(ProfilerCudaKernelSanityTest, SimpleAddSub) {
    // Ensure this is only run on CUPTI > 26 (paired w/ CUDA 12.6)
    uint32_t cupti_version = 0;
    cuptiGetVersion(&cupti_version);
    EXPECT_GE(cupti_version, 24);

    if (! CUPTI_PM_SAMPLING || (cupti_version < 24)) {
        GTEST_SKIP() << "PM Sampling not supported on this version of CUPTI";
    }

    constexpr int kNumElements = 256*1024;

    CuptiTracerCollectorOptions collector_options;
    collector_options.num_gpus = CuptiTracer::NumGpus();
    auto start_walltime_ns = absl::GetCurrentTimeNanos();
    auto start_gputime_ns = CuptiTracer::GetTimestamp();
    auto collector = CreateCuptiCollector(collector_options, start_walltime_ns,
        start_gputime_ns);

    PmSamplingConfig sampling_config;
    sampling_config.enable_pm_sampling = true;
    // Metrics can be queried with Nsight Compute
    // ncu --query-metrics
    sampling_config.metrics = {
      "sm__cycles_active.sum",
      "sm__inst_executed_pipe_fp64.sum",
      "pcie__read_bytes.sum",
      "pcie__write_bytes.sum"
    };
    sampling_config.process_samples = HandleRecords;

    CuptiTracerOptions tracer_options;
    tracer_options.enable_nvtx_tracking = false;
    tracer_options.pm_sampling_config = &sampling_config;

    TestableCuptiTracer tracer;
    tracer.Enable(tracer_options, collector.get());

    // SimpleAddSub does num_elements * 4 integer add / subs
    std::vector<double> vec = SimpleAddSubWithProfiler(kNumElements);

    tracer.Disable();

    EXPECT_EQ(vec.size(), kNumElements);
    for (int i = 0; i < kNumElements; ++i) {
        EXPECT_GT(vec[i], (0. - 0.001)) << "index: " << i;
        EXPECT_LT(vec[i], (0. + 0.001)) << "index: " << i;
    }

    // Expect 4 * elems / (32 elemn / warp) +- 5% double instructions
    LOG(INFO) << "Sampled " << atomic_total_fp64 << " fp64 instructions";
    EXPECT_GT(atomic_total_fp64, kNumElements * 4 * 95 / 32 / 100);
    EXPECT_LT(atomic_total_fp64, kNumElements * 4 * 105 / 32 / 100);

    // Expect > 4 * elems * sizeof(double) bytes written to pcie
    // 3 copies to device, 1 copy back
    // This is just a basic algorithmic minimum, there are more loads and
    // stores due to copying kernel itself, etc
    LOG(INFO) << "Sampled " << atomic_total_read << "B pcie reads";
    LOG(INFO) << "Sampled " << atomic_total_write << "B pcie writes";
    EXPECT_GE(atomic_total_write, kNumElements * 4 * sizeof(double));
}

}  // namespace

}  // namespace test
}  // namespace profiler
}  // namespace xla
