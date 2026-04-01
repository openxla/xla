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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"
#include "xla/backends/profiler/gpu/cupti_range_profiler.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

using ::testing::DistanceFrom;
using ::testing::Each;
using ::testing::Lt;
using xla::profiler::CuptiTracer;
using xla::profiler::CuptiTracerCollectorOptions;
using xla::profiler::CuptiTracerOptions;
using xla::profiler::CuptiWrapper;

// Needed to create different cupti tracer for each test cases.
class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer(CuptiErrorManager* error_manager)
      : CuptiTracer(error_manager) {}
};

std::atomic_uint64_t records_fp64 = 0;
std::atomic_uint64_t total_fp64 = 0;
std::atomic_uint64_t records_cycles = 0;
std::atomic_uint64_t total_cycles = 0;
std::atomic_bool skip_first = true;

void HandleRecords(PmSamples* samples) {
  // Validate some samples were recorded
  EXPECT_GT(samples->GetNumSamples(), 0);

  LOG(INFO) << "PM Sampling buffer flushed with " << samples->GetNumSamples()
            << " samples";

  // Validate we have the expected metrics
  const std::vector<std::string>& metrics = samples->GetMetrics();
  const std::vector<SamplerRange>& sampler_ranges = samples->GetSamplerRanges();
  const auto& back = sampler_ranges.back();
  const auto& front = sampler_ranges.front();
  double ranges_duration = back.end_timestamp_ns - front.start_timestamp_ns;
  double ns_per_sample = ranges_duration / samples->GetNumSamples();

  // First pass may have large initial sample duration
  if (skip_first) {
    skip_first = false;
  } else {
    EXPECT_GT(ns_per_sample, 500000.0 * 0.9);
    EXPECT_LT(ns_per_sample, 500000.0 * 1.1);
  }

  for (size_t i = 0; i < metrics.size(); i++) {
    double sum = 0;

    for (size_t j = 0; j < sampler_ranges.size(); j++) {
      sum += sampler_ranges[j].metric_values[i];
    }

    if ("sm__inst_executed_pipe_fp64.sum" == metrics[i]) {
      records_fp64 += 1;
      total_fp64 += sum;
    } else if ("sm__inst_executed_pipe_fmaheavy.sum" == metrics[i]) {
      records_fp64 += 1;
      total_fp64 += sum;
    } else if ("sm__cycles_active.sum" == metrics[i]) {
      records_cycles += 1;
      total_cycles += sum;
    } else {
      GTEST_FAIL() << "Unknown metric: " << metrics[i];
    }
  }
}

void SimpleAddSubWithProfilerTest(bool enable_activity_hardware_tracing,
                                  bool enable_pm_sampling) {
  uint32_t cupti_version = 0;
  cuptiGetVersion(&cupti_version);
  LOG(INFO) << "RUNTIME CUPTI version " << cupti_version
            << " and CUPTI_API_VERSION " << CUPTI_API_VERSION;

  enable_pm_sampling =
      enable_pm_sampling && (cupti_version >= 24 && CUPTI_API_VERSION >= 24);
  LOG(INFO) << "PM Sampling enabled: " << enable_pm_sampling;

  constexpr int kNumElements = 256 * 1024;

  CuptiTracerCollectorOptions collector_options{};
  collector_options.num_gpus = CuptiTracer::NumGpus();
  LOG(INFO) << "Cupti found #gpus: " << collector_options.num_gpus;
  uint64_t start_walltime_ns = absl::GetCurrentTimeNanos();
  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  auto collector = CreateCuptiCollector(collector_options, start_walltime_ns,
                                        start_gputime_ns);

  CuptiPmSamplerOptions sampler_options;
  sampler_options.enable = enable_pm_sampling;
  // Metrics can be queried with Nsight Compute
  // ncu --query-metrics-collection pmsampling --chip <CHIP>
  // Any metrics marked with a particular Triage group naming should be
  // configurable in a single pass on this chip.  Other combinations may not be
  // possible in a single pass and are not valid for pm sampling.
  sampler_options.metrics = {"sm__cycles_active.sum",
                             "sm__inst_executed_pipe_fp64.sum"};
  sampler_options.process_samples = HandleRecords;

  CuptiTracerOptions tracer_options{};
  tracer_options.enable_nvtx_tracking = false;
  tracer_options.activities_selected.push_back(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  tracer_options.enable_activity_hardware_tracing =
      enable_activity_hardware_tracing;
  tracer_options.cupti_finalize = true;

  tracer_options.pm_sampler_options = sampler_options;

  CuptiErrorManager error_manager(std::make_unique<CuptiWrapper>());
  TestableCuptiTracer tracer(&error_manager);

  std::vector<std::unique_ptr<tensorflow::profiler::XPlane>> xplanes;
  xplanes.reserve(collector_options.num_gpus);
  for (uint32_t i = 0; i < collector_options.num_gpus; ++i) {
    xplanes.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  auto err = tracer.Enable(tracer_options, collector.get(), xplanes);

  if (absl::IsPermissionDenied(err)) {
    GTEST_SKIP() << "PM Sampling requires root access";
  }

  // SimpleAddSub does num_elements * 4 integer add / subs
  std::vector<double> vec = SimpleAddSubWithProfiler(kNumElements);

  tracer.Disable();

  // Go for a second pass to ensure we can sample again, with an un-sampled func
  absl::SleepFor(absl::Milliseconds(1000));
  skip_first = true;
  vec = SimpleAddSubWithProfiler(kNumElements);

  xplanes.clear();
  xplanes.reserve(collector_options.num_gpus);
  for (uint32_t i = 0; i < collector_options.num_gpus; ++i) {
    xplanes.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  err = tracer.Enable(tracer_options, collector.get(), xplanes);

  vec = SimpleAddSubWithProfiler(kNumElements);

  tracer.Disable();

  // Validate functional correctness - ie, the kernel ran
  EXPECT_EQ(vec.size(), kNumElements);
  EXPECT_THAT(vec, Each(DistanceFrom(0, Lt(0.001))));

  auto space = std::make_unique<tensorflow::profiler::XSpace>();
  collector->Export(space.get(), CuptiTracer::GetTimestamp());
  EXPECT_GE(space->planes_size(), 1);

  if (enable_pm_sampling) {
    // Expect 4 * elems / (32 elemn / warp) +- 5% double instructions
    // (if they were sampled)
    // Double this as two kernels are sampled (middle kernel is not)
    if (records_fp64 > 0) {
      LOG(INFO) << "Sampled " << total_fp64 << " fp64 instructions";
      double target = kNumElements * 4 * 2 / 32;
      EXPECT_THAT(total_fp64, DistanceFrom(target, Lt(target * 5 / 100)));
    }
    if (records_cycles > 0) {
      LOG(INFO) << "Sampled " << total_cycles << " cycles";
    }
  }
}

TEST(ProfilerCudaKernelSanityTest, SimpleAddSubWithPMSampling) {
  SimpleAddSubWithProfilerTest(/*enable_activity_hardware_tracing=*/false,
                               /*enable_pm_sampling=*/true);
}

TEST(ProfilerCudaKernelSanityTest, SimpleAddSubWithHESDisabled) {
  SimpleAddSubWithProfilerTest(/*enable_activity_hardware_tracing=*/false,
                               /*enable_pm_sampling=*/false);
}

TEST(ProfilerCudaKernelSanityTest, SimpleAddSubWithHESEnabled) {
  SimpleAddSubWithProfilerTest(/*enable_activity_hardware_tracing=*/true,
                               /*enable_pm_sampling=*/false);
}

// --- Range Profiling Tests ---

std::atomic_bool range_results_received = false;
std::atomic_int range_results_num_ranges = 0;
std::atomic_int range_results_num_metrics = 0;
std::atomic<double> range_results_fp64_sum = 0;

void HandleRangeResults(RangeProfilerResults* results) {
  LOG(INFO) << "Range profiling results received for device "
            << results->GetDeviceId();
  LOG(INFO) << "  Metrics: " << results->GetMetrics().size()
            << ", Ranges: " << results->GetRanges().size();

  const auto& metrics = results->GetMetrics();
  for (const auto& range : results->GetRanges()) {
    LOG(INFO) << "  Range: " << range.range_name;
    for (size_t i = 0; i < range.metric_values.size(); ++i) {
      LOG(INFO) << "    " << metrics[i] << " = " << range.metric_values[i];
      if (metrics[i] == "sm__inst_executed_pipe_fp64.sum") {
        range_results_fp64_sum = range_results_fp64_sum + range.metric_values[i];
      }
    }
  }

  range_results_received = true;
  range_results_num_ranges += results->GetRanges().size();
  range_results_num_metrics += results->GetMetrics().size();
}

TEST(ProfilerCudaKernelSanityTest, SimpleAddSubWithRangeProfiling) {
  uint32_t cupti_version = 0;
  cuptiGetVersion(&cupti_version);
  LOG(INFO) << "RUNTIME CUPTI version " << cupti_version
            << " and CUPTI_API_VERSION " << CUPTI_API_VERSION;

  if (cupti_version < 24 || CUPTI_API_VERSION < 24) {
    GTEST_SKIP() << "Range profiling requires CUPTI API version >= 24";
  }

  constexpr int kNumElements = 256 * 1024;

  // Ensure a CUDA context exists before range profiler initialization.
  SimpleAddSubWithProfiler(kNumElements);

  CuptiTracerCollectorOptions collector_options{};
  collector_options.num_gpus = CuptiTracer::NumGpus();
  LOG(INFO) << "Cupti found #gpus: " << collector_options.num_gpus;
  uint64_t start_walltime_ns = absl::GetCurrentTimeNanos();
  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  auto collector = CreateCuptiCollector(collector_options, start_walltime_ns,
                                        start_gputime_ns);

  CuptiRangeProfilerOptions range_options;
  range_options.enable = true;
  range_options.metrics = {"sm__cycles_active.sum"};
  range_options.range_mode = CuptiRangeMode::kUser;
  range_options.range_name = "test_range";
  range_options.process_results = HandleRangeResults;

  CuptiTracerOptions tracer_options{};
  tracer_options.enable_nvtx_tracking = false;
  tracer_options.activities_selected.push_back(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  tracer_options.cupti_finalize = true;
  tracer_options.range_profiler_options = range_options;

  CuptiErrorManager error_manager(std::make_unique<CuptiWrapper>());
  TestableCuptiTracer tracer(&error_manager);

  std::vector<std::unique_ptr<tensorflow::profiler::XPlane>> xplanes;
  xplanes.reserve(collector_options.num_gpus);
  for (uint32_t i = 0; i < collector_options.num_gpus; ++i) {
    xplanes.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  auto err = tracer.Enable(tracer_options, collector.get(), xplanes);

  if (absl::IsPermissionDenied(err)) {
    GTEST_SKIP() << "Range profiling requires root/admin access";
  }
  if (!err.ok()) {
    GTEST_SKIP() << "Range profiling Enable failed: " << err;
  }

  ASSERT_TRUE(tracer.IsRangeProfilingEnabled());

  int num_passes = tracer.NumRangeProfilingPasses();
  LOG(INFO) << "Range profiling requires " << num_passes << " passes";
  ASSERT_GT(num_passes, 0);

  // Reset result tracking.
  range_results_received = false;
  range_results_num_ranges = 0;
  range_results_num_metrics = 0;

  // Enable() started the first pass (BeginPass + PushRange).
  // Run the workload once per pass, with inter-pass transitions in between.
  for (int pass = 0; pass < num_passes; ++pass) {
    LOG(INFO) << "Starting pass " << pass;

    std::vector<double> vec = SimpleAddSubWithProfiler(kNumElements);

    // Validate kernel correctness.
    EXPECT_EQ(vec.size(), kNumElements);
    EXPECT_THAT(vec, Each(DistanceFrom(0, Lt(0.001))));

    // Transition between passes (not after the last — Disable handles that).
    if (pass < num_passes - 1) {
      ASSERT_TRUE(tracer.PopProfilingRange().ok());
      ASSERT_TRUE(tracer.EndRangePass().ok());
      ASSERT_TRUE(tracer.BeginRangePass().ok());
      ASSERT_TRUE(tracer.PushProfilingRange("test_range").ok());
    }
  }

  // Disable() ends the last pass (PopRange + EndPass) then FlushAndDecode.
  tracer.Disable();

  // Validate that results were delivered.
  EXPECT_TRUE(range_results_received);
  EXPECT_GT(range_results_num_ranges, 0);
  EXPECT_GT(range_results_num_metrics, 0);
}

// Multi-pass variant: request metrics from different counter groups to force
// CUPTI to require more than one replay pass.
TEST(ProfilerCudaKernelSanityTest, SimpleAddSubWithMultiPassRangeProfiling) {
  uint32_t cupti_version = 0;
  cuptiGetVersion(&cupti_version);
  if (cupti_version < 24 || CUPTI_API_VERSION < 24) {
    GTEST_SKIP() << "Range profiling requires CUPTI API version >= 24";
  }

  constexpr int kNumElements = 256 * 1024;

  // Run a trivial kernel to ensure a CUDA context exists before range
  // profiler initialization (cuptiProfilerGetCounterAvailability requires it).
  SimpleAddSubWithProfiler(kNumElements);

  CuptiTracerCollectorOptions collector_options{};
  collector_options.num_gpus = CuptiTracer::NumGpus();
  uint64_t start_walltime_ns = absl::GetCurrentTimeNanos();
  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  auto collector = CreateCuptiCollector(collector_options, start_walltime_ns,
                                        start_gputime_ns);

  // Metrics spanning multiple HW units to force multi-pass collection.
  // ~25 metrics across smsp, sm, l1tex, dram, and fbpa units.
  CuptiRangeProfilerOptions range_options;
  range_options.enable = true;
  range_options.metrics = {
      // FP64 pipe instruction count — used to validate kernel correctness.
      "sm__inst_executed_pipe_fp64.sum",
      "smsp__inst_executed.sum",
      "smsp__cycles_active.sum",
      "smsp__inst_executed_op_global_ld.sum",
      "smsp__thread_inst_executed_pred_on.sum",
      // DRAM counters.
      "dram__bytes.sum",
      "dram__throughput.avg_pct_of_peak_sustained_elapsed",
      "dram__cycles_active.sum",
      "dram__sectors.sum",
      // FBPA counters.
      "fbpa__dram_read_bytes.sum",
      "fbpa__dram_write_bytes.sum",
      "fbpa__dram_sectors.sum",
  };
  range_options.range_mode = CuptiRangeMode::kUser;
  range_options.range_name = "test_multipass";
  range_options.allow_multipass = true;
  range_options.process_results = HandleRangeResults;

  CuptiTracerOptions tracer_options{};
  tracer_options.enable_nvtx_tracking = false;
  tracer_options.activities_selected.push_back(
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  tracer_options.cupti_finalize = true;
  tracer_options.range_profiler_options = range_options;

  CuptiErrorManager error_manager(std::make_unique<CuptiWrapper>());
  TestableCuptiTracer tracer(&error_manager);

  std::vector<std::unique_ptr<tensorflow::profiler::XPlane>> xplanes;
  xplanes.reserve(collector_options.num_gpus);
  for (uint32_t i = 0; i < collector_options.num_gpus; ++i) {
    xplanes.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  auto err = tracer.Enable(tracer_options, collector.get(), xplanes);

  if (absl::IsPermissionDenied(err)) {
    GTEST_SKIP() << "Range profiling requires root/admin access";
  }
  if (!err.ok()) {
    GTEST_SKIP() << "Range profiling Enable failed: " << err;
  }

  ASSERT_TRUE(tracer.IsRangeProfilingEnabled());

  int num_passes = tracer.NumRangeProfilingPasses();
  LOG(INFO) << "Multi-pass range profiling requires " << num_passes
            << " passes";
  ASSERT_GT(num_passes, 0);

  // Reset result tracking.
  range_results_received = false;
  range_results_num_ranges = 0;
  range_results_num_metrics = 0;
  range_results_fp64_sum = 0;

  // Enable() started the first pass (BeginPass + PushRange).
  // Run the workload once per pass, with inter-pass transitions in between.
  for (int pass = 0; pass < num_passes; ++pass) {
    LOG(INFO) << "Starting pass " << pass << " of " << num_passes;

    std::vector<double> vec = SimpleAddSubWithProfiler(kNumElements);
    EXPECT_EQ(vec.size(), kNumElements);
    EXPECT_THAT(vec, Each(DistanceFrom(0, Lt(0.001))));

    // Transition between passes (not after the last — Disable handles that).
    if (pass < num_passes - 1) {
      ASSERT_TRUE(tracer.PopProfilingRange().ok());
      ASSERT_TRUE(tracer.EndRangePass().ok());
      ASSERT_TRUE(tracer.BeginRangePass().ok());
      ASSERT_TRUE(tracer.PushProfilingRange("test_multipass").ok());
    }
  }

  // Disable() ends the last pass (PopRange + EndPass) then FlushAndDecode.
  tracer.Disable();

  EXPECT_TRUE(range_results_received);
  EXPECT_GT(range_results_num_ranges, 0);
  EXPECT_GT(range_results_num_metrics, 0);

  // Validate FP64 instruction count.
  // The kernel launches 4 VecAdd/VecSub kernels, each executing
  // num_elements / 32 warp-level FP64 instructions.
  double fp64_target = kNumElements * 4.0 / 32.0;
  double fp64_actual = range_results_fp64_sum.load();
  LOG(INFO) << "FP64 instructions: expected " << fp64_target
            << ", actual " << fp64_actual;
  EXPECT_THAT(fp64_actual, DistanceFrom(fp64_target, Lt(fp64_target * 0.05)));

  if (num_passes == 1) {
    LOG(WARNING) << "All metrics fit in a single pass on this GPU; "
                 << "multi-pass replay path was not exercised. "
                 << "This is expected on some architectures.";
  } else {
    LOG(INFO) << "Multi-pass collection confirmed: " << num_passes
              << " passes required.";
  }
}

}  // namespace
}  // namespace test
}  // namespace profiler
}  // namespace xla
