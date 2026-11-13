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

// End-to-end tests for ProfileOptions.advanced_configuration in the ROCm
// GpuTracer. They need a ROCm GPU.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Defined in device_tracer_rocm.cc.
std::unique_ptr<tsl::profiler::ProfilerInterface> CreateGpuTracer(
    const tensorflow::ProfileOptions& options);

namespace {

using ::tensorflow::ProfileOptions;
using ::tensorflow::profiler::XSpace;

constexpr char kMisspelledKey[] = "gpu_max_callbac_api_events";

// These tests call CreateGpuTracer() directly. Version 1 mirrors real callers:
// ProfilerSession drops advanced_configuration when version() is 0.
ProfileOptions MakeGpuOptions() {
  ProfileOptions options;
  options.set_version(1);
  options.set_device_type(ProfileOptions::GPU);
  return options;
}

ProfileOptions::AdvancedConfigValue& Config(ProfileOptions& options,
                                            const std::string& key) {
  return (*options.mutable_advanced_configuration())[key];
}

// Appends the deprecated --xla_gpu_rocm_max_trace_events to XLA_FLAGS while in
// scope.
class ScopedMaxTraceEventsFlag {
 public:
  explicit ScopedMaxTraceEventsFlag(int64_t value) {
    const char* old_flags = std::getenv("XLA_FLAGS");
    if (old_flags != nullptr) old_flags_ = old_flags;
    tsl::setenv("XLA_FLAGS",
                absl::StrCat(old_flags_.value_or(""),
                             " --xla_gpu_rocm_max_trace_events=", value)
                    .c_str(),
                /*overwrite=*/1);
    ParseDebugOptionFlagsFromEnv(/*reset_envvar=*/true);
  }
  ~ScopedMaxTraceEventsFlag() {
    if (old_flags_.has_value()) {
      tsl::setenv("XLA_FLAGS", old_flags_->c_str(), /*overwrite=*/1);
    } else {
      tsl::unsetenv("XLA_FLAGS");
    }
    ResetFlagValues();
    ParseDebugOptionFlagsFromEnv(/*reset_envvar=*/true);
  }

 private:
  std::optional<std::string> old_flags_;
};

// Issues 2 * `iterations` hipMemcpy calls.
void RunSomeHipWork(int iterations) {
  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;
  ASSERT_EQ(hipMalloc(&device_data, kSize), hipSuccess);
  for (int i = 0; i < iterations; ++i) {
    ASSERT_EQ(
        hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(host_data.data(), device_data, kSize, hipMemcpyDeviceToHost),
        hipSuccess);
  }
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipFree(device_data), hipSuccess);
}

bool HasGpu() {
  int device_count = 0;
  return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
}

// Runs a traced workload into `space`. Callers wrap it in
// ASSERT_NO_FATAL_FAILURE. The tracer is stopped even if the HIP work fails, so
// the RocmTracer singleton stays usable by later tests.
void TraceHipWorkload(const ProfileOptions& options, int iterations,
                      XSpace* space) {
  std::unique_ptr<tsl::profiler::ProfilerInterface> tracer =
      CreateGpuTracer(options);
  ASSERT_NE(tracer, nullptr);
  ASSERT_OK(tracer->Start());
  RunSomeHipWork(iterations);  // A failed ASSERT only leaves this function.
  absl::SleepFor(absl::Milliseconds(100));
  ASSERT_OK(tracer->Stop());
  if (testing::Test::HasFatalFailure()) return;
  ASSERT_OK(tracer->CollectData(space));
}

size_t CountGpuPlaneEvents(const XSpace& space) {
  size_t total = 0;
  for (const auto& plane : space.planes()) {
    if (!absl::StartsWith(plane.name(), "/device:GPU:")) {
      continue;
    }
    for (const auto& line : plane.lines()) {
      total += line.events_size();
    }
  }
  return total;
}

template <typename Messages>
bool AnyContains(const Messages& messages, absl::string_view needle) {
  for (const std::string& message : messages) {
    if (absl::StrContains(message, needle)) return true;
  }
  return false;
}

TEST(DeviceTracerRocmTest, BadKeyStillProducesATraceAndReportsIt) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  ProfileOptions options = MakeGpuOptions();
  Config(options, kMisspelledKey).set_int64_value(5000);
  XSpace space;
  ASSERT_NO_FATAL_FAILURE(TraceHipWorkload(options, /*iterations=*/4, &space));

  EXPECT_GT(CountGpuPlaneEvents(space), 0u);
  EXPECT_TRUE(AnyContains(space.errors(), kMisspelledKey))
      << "errors: " << space.errors_size();
}

TEST(DeviceTracerRocmTest, WarningReachesXSpace) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  ProfileOptions options = MakeGpuOptions();
  Config(options, "gpu_enable_nvtx_tracking").set_bool_value(false);
  XSpace space;
  ASSERT_NO_FATAL_FAILURE(TraceHipWorkload(options, /*iterations=*/1, &space));

  EXPECT_TRUE(space.errors().empty());
  EXPECT_TRUE(AnyContains(space.warnings(), "gpu_enable_nvtx_tracking"))
      << "warnings: " << space.warnings_size();
}

TEST(DeviceTracerRocmTest, RaiseErrorOnStartFailureMakesABadKeyFatal) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  ProfileOptions options = MakeGpuOptions();
  options.set_raise_error_on_start_failure(true);
  Config(options, kMisspelledKey).set_int64_value(5000);

  std::unique_ptr<tsl::profiler::ProfilerInterface> tracer =
      CreateGpuTracer(options);
  ASSERT_NE(tracer, nullptr);

  const absl::Status status = tracer->Start();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), kMisspelledKey)) << status;

  // Start() failed before enabling the tracer, so Stop() is a no-op and the
  // singleton stays usable by the next test.
  EXPECT_OK(tracer->Stop());
}

// Compares the same workload with and without the key, rather than predicting
// how many device-plane events a callback limit leaves.
TEST(DeviceTracerRocmTest, CallbackLimitKeyIsApplied) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  constexpr int64_t kCallbackLimit = 3;
  constexpr int kIterations = 20;

  ProfileOptions limited_options = MakeGpuOptions();
  Config(limited_options, "gpu_max_callback_api_events")
      .set_int64_value(kCallbackLimit);
  XSpace limited_space;
  ASSERT_NO_FATAL_FAILURE(
      TraceHipWorkload(limited_options, kIterations, &limited_space));
  XSpace unlimited_space;
  ASSERT_NO_FATAL_FAILURE(
      TraceHipWorkload(MakeGpuOptions(), kIterations, &unlimited_space));

  EXPECT_TRUE(limited_space.errors().empty());
  const size_t limited = CountGpuPlaneEvents(limited_space);
  const size_t unlimited = CountGpuPlaneEvents(unlimited_space);
  // Guards against a pass where tracing produced nothing at all.
  EXPECT_GT(unlimited, static_cast<size_t>(kCallbackLimit));
  EXPECT_LT(limited, unlimited)
      << "limited=" << limited << " unlimited=" << unlimited;
  // The drop report reaches the caller's XSpace through
  // GpuTracer::CollectData(), and only when events were dropped.
  EXPECT_TRUE(AnyContains(limited_space.warnings(), "GPU events dropped"))
      << "warnings: " << limited_space.warnings_size();
  EXPECT_FALSE(AnyContains(unlimited_space.warnings(), "GPU events dropped"));
}

// The deprecated flag still sets the callback limit when no key does.
TEST(DeviceTracerRocmTest, DeprecatedFlagLimitsCallbacks) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  constexpr int64_t kCallbackLimit = 3;
  constexpr int kIterations = 20;

  XSpace unlimited_space;
  ASSERT_NO_FATAL_FAILURE(
      TraceHipWorkload(MakeGpuOptions(), kIterations, &unlimited_space));
  XSpace limited_space;
  {
    ScopedMaxTraceEventsFlag flag(kCallbackLimit);
    ASSERT_NO_FATAL_FAILURE(
        TraceHipWorkload(MakeGpuOptions(), kIterations, &limited_space));
  }

  const size_t limited = CountGpuPlaneEvents(limited_space);
  const size_t unlimited = CountGpuPlaneEvents(unlimited_space);
  EXPECT_GT(unlimited, static_cast<size_t>(kCallbackLimit));
  EXPECT_LT(limited, unlimited)
      << "limited=" << limited << " unlimited=" << unlimited;
  // Remote capture can only raise the limit with the flag, so the drop report
  // must name it.
  EXPECT_TRUE(
      AnyContains(limited_space.warnings(), "--xla_gpu_rocm_max_trace_events"))
      << "warnings: " << limited_space.warnings_size();
}

TEST(DeviceTracerRocmTest, CallbackLimitKeyOverridesDeprecatedFlag) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  constexpr int64_t kFlagLimit = 3;
  constexpr int64_t kKeyLimit = 4 * 1024 * 1024;
  constexpr int kIterations = 20;
  ScopedMaxTraceEventsFlag flag(kFlagLimit);

  ProfileOptions key_options = MakeGpuOptions();
  Config(key_options, "gpu_max_callback_api_events").set_int64_value(kKeyLimit);
  XSpace flag_space;
  ASSERT_NO_FATAL_FAILURE(
      TraceHipWorkload(MakeGpuOptions(), kIterations, &flag_space));
  XSpace key_space;
  ASSERT_NO_FATAL_FAILURE(
      TraceHipWorkload(key_options, kIterations, &key_space));

  EXPECT_TRUE(key_space.errors().empty());
  const size_t flag_only = CountGpuPlaneEvents(flag_space);
  const size_t with_key = CountGpuPlaneEvents(key_space);
  EXPECT_LT(flag_only, with_key)
      << "flag_only=" << flag_only << " with_key=" << with_key;
}

}  // namespace
}  // namespace profiler
}  // namespace xla
