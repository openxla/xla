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

#include <atomic>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "absl/time/time.h"

#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

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

std::atomic_uint64_t sampled_fp64 = 0;
std::atomic_uint64_t atomic_total_fp64 = 0;
std::atomic_uint64_t sampled_read = 0;
std::atomic_uint64_t atomic_total_read = 0;
std::atomic_uint64_t sampled_write = 0;
std::atomic_uint64_t atomic_total_write = 0;
std::atomic_bool skip_first = true;

void HandleRecords(PmSamples* samples) {
  // Validate some samples were recorded
  EXPECT_GT(samples->GetNumSamples(), 0);

  // Validate we have the expected metrics
  const std::vector<std::string>& metrics = samples->GetMetrics();
  const std::vector<SamplerRange>& sampler_ranges = samples->GetSamplerRanges();
  auto back = sampler_ranges.back();
  auto front = sampler_ranges.front();
  double ranges_duration = back.end_timestamp_ns - front.start_timestamp_ns;
  double ns_per_sample = ranges_duration / samples->GetNumSamples();

  // First pass may have large initial sample duration
  if (skip_first) {
    skip_first = false;
  } else {
    EXPECT_GT(ns_per_sample, 500000.0 * 0.95);
    EXPECT_LT(ns_per_sample, 500000.0 * 1.05);
  }

  for (int i = 0; i < metrics.size(); i++) {
    double sum = 0;

    for (int j = 0; j < sampler_ranges.size(); j++) {
      sum += sampler_ranges[j].metric_values[i];
    }

    if (strcmp("sm__inst_executed_pipe_fp64.sum", metrics[i].c_str()) == 0) {
      sampled_fp64 = 1;
      atomic_total_fp64 += sum;
    } else if (strcmp("pcie__read_bytes.sum", metrics[i].c_str()) == 0) {
      sampled_read = 1;
      atomic_total_read += sum;
    } else if (strcmp("pcie__write_bytes.sum", metrics[i].c_str()) == 0) {
      sampled_write = 1;
      atomic_total_write += sum;
    }
  }

  return;
}

TEST(ProfilerCudaKernelSanityTest, SimpleAddSub) {
  // Ensure this is only run on CUPTI > 26 (paired w/ CUDA 12.6)
  if (CUPTI_API_VERSION < 24) {
    GTEST_SKIP() << "PM Sampling not supported on this version of CUPTI";
  }

  uint32_t cupti_version = 0;
  cuptiGetVersion(&cupti_version);
  EXPECT_EQ(CUPTI_API_VERSION, cupti_version);

  if (cupti_version < 24) {
    GTEST_SKIP() << "PM Sampling not supported on this version of CUPTI";
  }

  constexpr int kNumElements = 256 * 1024;

  CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = CuptiTracer::NumGpus();
  uint64_t start_walltime_ns = absl::GetCurrentTimeNanos();
  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  auto collector = CreateCuptiCollector(collector_options, start_walltime_ns,
                                        start_gputime_ns);

  CuptiPmSamplerOptions sampler_options;
  sampler_options.enable = true;
  // Metrics can be queried with Nsight Compute
  // ncu --query-metrics-collection pmsampling --chip <CHIP>
  // Any metrics marked with a particular Triage group naming should be 
  // configurable in a single pass on this chip.
  sampler_options.metrics = {"sm__cycles_active.sum",
                             "sm__inst_executed_pipe_fp64.sum",
                             "pcie__read_bytes.sum", "pcie__write_bytes.sum", "lts__average_t_sector_aperture_device_lookup_hit.pct", "lts__average_t_sector_aperture_device_lookup_miss.pct"};
  sampler_options.process_samples = HandleRecords;

  CuptiTracerOptions tracer_options;
  tracer_options.enable_nvtx_tracking = false;

  tracer_options.pm_sampler_options = sampler_options;

  TestableCuptiTracer tracer;
  auto err = tracer.Enable(tracer_options, collector.get());

  if (absl::IsPermissionDenied(err)) {
    GTEST_SKIP() << "PM Sampling requires root access";
  }

  // SimpleAddSub does num_elements * 4 integer add / subs
  std::vector<double> vec = SimpleAddSubWithProfiler(kNumElements);

  tracer.Disable();

  // Validate functional correctness - ie, the kernel ran
  EXPECT_EQ(vec.size(), kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_GT(vec[i], (0.0 - 0.001)) << "index: " << i;
    EXPECT_LT(vec[i], (0.0 + 0.001)) << "index: " << i;
  }

  // Expect 4 * elems / (32 elemn / warp) +- 5% double instructions
  // (if they were sampled)
  if (sampled_fp64) {
    LOG(INFO) << "Sampled " << atomic_total_fp64 << " fp64 instructions";
    EXPECT_GT(atomic_total_fp64, kNumElements * 4 * 95 / 32 / 100);
    EXPECT_LT(atomic_total_fp64, kNumElements * 4 * 105 / 32 / 100);
  }

  // Expect > 4 * elems * sizeof(double) bytes written to pcie
  // 3 copies to device, 1 copy back
  // This is just a basic algorithmic minimum, there are more loads and
  // stores due to copying kernel itself, etc
  if (sampled_read) {
    LOG(INFO) << "Sampled " << atomic_total_read << "B pcie reads";
  }
  if (sampled_write) {
    LOG(INFO) << "Sampled " << atomic_total_write << "B pcie writes";
    EXPECT_GE(atomic_total_write, kNumElements * 4 * sizeof(double));
  }
}

}  // namespace
}  // namespace test
}  // namespace profiler
}  // namespace xla
