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

// Regression test for the ROCm tracer library's constructor running from
// inside rocprofiler-sdk initialization. rocm_tracer_early_client starts that
// initialization from its own constructor. If the tracer's constructor then
// forced configuration, rocprofiler-sdk 1.4 and later would deadlock before
// main(), and this test would time out.
//
// Deliberately not covered:
// - The tracer's constructor running after initialization has finished. This
//   change leaves that path as it was, and rocprofiler-sdk 1.3.5 rejects the
//   late call there, so a test for it would fail on TheRock 10.0.0.
// - Discovery missing rocprofiler_configure. The tracer then stays
//   unconfigured, and RocmTracer::Enable() reports that.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/backends/profiler/gpu/rocm_tracer_early_client.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

class EventCountingCollector : public RocmTraceCollector {
 public:
  EventCountingCollector() : RocmTraceCollector(MakeCollectorOptions()) {}

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override {
    ++event_count_;
  }
  void OnEventsDropped(const std::string& reason,
                       uint64_t num_events) override {}
  void Flush() override {}
  void Export(tsl::profiler::XSpace* space) override {}

  int event_count() const { return event_count_; }

 private:
  static RocmTraceCollectorOptions MakeCollectorOptions() {
    RocmTraceCollectorOptions options;
    options.max_callback_api_events = 2 * 1024 * 1024;
    options.max_activity_api_events = 2 * 1024 * 1024;
    options.max_annotation_strings = 1024 * 1024;
    options.num_gpus = RocmTracer::GetRocmTracerSingleton().NumGpus();
    return options;
  }

  int event_count_ = 0;
};

TEST(RocmTracerEarlyInitTest, TracesWhenRocprofilerInitializedFirst) {
  const EarlyClientResult& early_client = GetEarlyClientResult();
  ASSERT_TRUE(early_client.constructor_ran);
  ASSERT_EQ(early_client.is_initialized_status, ROCPROFILER_STATUS_SUCCESS);
  // If the tracer's constructor had run first, it would have initialized
  // rocprofiler-sdk itself and this test would not cover the case it is for.
  ASSERT_EQ(early_client.init_status_before, 0)
      << "rocprofiler-sdk was already initialized when the early client's "
         "constructor ran. :rocm_tracer_early_client must be a separate shared "
         "library linked after :rocm_tracer; see the BUILD target.";
  ASSERT_EQ(early_client.force_configure_status, ROCPROFILER_STATUS_SUCCESS);

  // rocprofiler-sdk's client discovery must still have configured the tracer.
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  EventCountingCollector collector;
  ASSERT_OK(tracer.Enable(RocmTracerOptions{/*max_annotation_strings=*/1024},
                          &collector));
  // Disable even if an assertion below fails, so the tracer does not keep
  // pointing at `collector` after it is destroyed.
  absl::Cleanup disable_tracer = [&tracer] { tracer.Disable(); };

  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;
  ASSERT_EQ(hipMalloc(&device_data, kSize), hipSuccess);
  absl::Cleanup free_device_data = [device_data] {
    EXPECT_EQ(hipFree(device_data), hipSuccess);
  };
  ASSERT_EQ(
      hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
      hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  std::move(disable_tracer).Invoke();
  EXPECT_GT(collector.event_count(), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
