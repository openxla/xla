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

#include "xla/backends/profiler/gpu/rocm_tracer_options_utils.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::tensorflow::ProfileOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

void SetInt(ProfileOptions& options, absl::string_view key, int64_t value) {
  (*options.mutable_advanced_configuration())[std::string(key)].set_int64_value(
      value);
}

void SetBool(ProfileOptions& options, absl::string_view key, bool value) {
  (*options.mutable_advanced_configuration())[std::string(key)].set_bool_value(
      value);
}

void SetString(ProfileOptions& options, absl::string_view key,
               absl::string_view value) {
  (*options.mutable_advanced_configuration())[std::string(key)]
      .set_string_value(std::string(value));
}

void SetNoValue(ProfileOptions& options, absl::string_view key) {
  (*options.mutable_advanced_configuration())[std::string(key)] =
      ProfileOptions::AdvancedConfigValue();
}

MATCHER_P(MentionsKey, key, "mentions the key '" + std::string(key) + "'") {
  return absl::StrContains(arg, "'" + std::string(key) + "'");
}

// Distinct defaults, so that an untouched field can be told apart from one set
// to a plausible value.
constexpr uint64_t kAnnotationDefault = 111;
constexpr uint64_t kCallbackDefault = 222;
constexpr uint64_t kActivityDefault = 333;
constexpr uint32_t kDeviceCount = 8;
constexpr uint64_t kMaxCount = kMaxRocmTraceEvents;

constexpr absl::string_view kCountKeys[] = {
    "gpu_max_callback_api_events",
    "gpu_max_activity_api_events",
    "gpu_max_annotation_strings",
};

struct Fixture {
  ProfileOptions options;
  RocmTraceCollectorOptions collector = {
      .max_callback_api_events = kCallbackDefault,
      .max_activity_api_events = kActivityDefault,
      .max_annotation_strings = kAnnotationDefault,
      .num_gpus = kDeviceCount,
  };
  uint32_t device_count = kDeviceCount;
  RocmTracerOptionDiagnostics diagnostics;

  void Run() {
    diagnostics = UpdateRocmTracerOptionsFromProfilerOptions(
        options, device_count, collector);
  }

  bool AllFieldsUntouched() const {
    return collector.max_annotation_strings == kAnnotationDefault &&
           collector.max_callback_api_events == kCallbackDefault &&
           collector.max_activity_api_events == kActivityDefault &&
           collector.num_gpus == kDeviceCount;
  }
};

TEST(RocmTracerOptionsUtilsTest, EmptyMapIsANoOp) {
  Fixture f;
  f.Run();

  EXPECT_TRUE(f.AllFieldsUntouched());
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, SetsCallbackEventLimit) {
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", 4242);
  f.Run();

  EXPECT_EQ(f.collector.max_callback_api_events, 4242);
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, SetsActivityEventLimitAndWarnsNotEnforced) {
  Fixture f;
  SetInt(f.options, "gpu_max_activity_api_events", 4242);
  f.Run();

  EXPECT_EQ(f.collector.max_activity_api_events, 4242);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings,
              ElementsAre(MentionsKey("gpu_max_activity_api_events")));
}

TEST(RocmTracerOptionsUtilsTest, SetsAnnotationLimit) {
  Fixture f;
  SetInt(f.options, "gpu_max_annotation_strings", 777);
  f.Run();

  EXPECT_EQ(f.collector.max_annotation_strings, 777);
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, NumGpusBelowDeviceCountIsAppliedWithCaveat) {
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 2);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, 2);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  ASSERT_THAT(f.diagnostics.warnings,
              ElementsAre(MentionsKey("gpu_num_chips_to_profile_per_task")));
  // The collector keeps a run of ids from the lowest one seen, not from GPU 0.
  EXPECT_THAT(f.diagnostics.warnings[0],
              HasSubstr("2 consecutive GPU ids are kept, starting at the "
                        "lowest id that recorded an event"));
}

TEST(RocmTracerOptionsUtilsTest, NumGpusEqualToDeviceCountIsQuiet) {
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", kDeviceCount);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, kDeviceCount);
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, NumGpusAboveDeviceCountProfilesAll) {
  for (int64_t value : {int64_t{kDeviceCount} + 1, int64_t{1} << 40}) {
    Fixture f;
    f.collector.num_gpus = 1;  // Must be reset to the device count.
    SetInt(f.options, "gpu_num_chips_to_profile_per_task", value);
    f.Run();

    EXPECT_EQ(f.collector.num_gpus, kDeviceCount) << value;
    EXPECT_THAT(f.diagnostics.errors, IsEmpty()) << value;
    EXPECT_THAT(f.diagnostics.warnings,
                ElementsAre(MentionsKey("gpu_num_chips_to_profile_per_task")))
        << value;
  }
}

TEST(RocmTracerOptionsUtilsTest, ZeroNumGpusProfilesAllQuietly) {
  // 0 means every GPU, as on CUDA. Passing it through would empty the trace.
  Fixture f;
  f.collector.num_gpus = 1;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 0);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, kDeviceCount);
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, NumGpusHasNoEffectWithoutGpus) {
  Fixture f;
  f.device_count = 0;
  f.collector.num_gpus = 0;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 2);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, 0);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  ASSERT_THAT(f.diagnostics.warnings,
              ElementsAre(MentionsKey("gpu_num_chips_to_profile_per_task")));
  EXPECT_THAT(f.diagnostics.warnings[0], HasSubstr("no GPUs are present"));
}

TEST(RocmTracerOptionsUtilsTest, ZeroNumGpusWithoutGpusIsQuiet) {
  // 0 asks for every GPU, which with no GPUs present is already the case.
  Fixture f;
  f.device_count = 0;
  f.collector.num_gpus = 0;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 0);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, 0);
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, NegativeNumGpusIsRejected) {
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", -1);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, kDeviceCount);
  EXPECT_THAT(f.diagnostics.errors,
              ElementsAre(MentionsKey("gpu_num_chips_to_profile_per_task")));
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, NegativeCountsAreRejected) {
  // -1 would wrap to 2^64-1 in the uint64_t field and disable the limit.
  for (absl::string_view key : kCountKeys) {
    SCOPED_TRACE(key);
    Fixture f;
    SetInt(f.options, key, -1);
    f.Run();

    EXPECT_TRUE(f.AllFieldsUntouched());
    ASSERT_THAT(f.diagnostics.errors, ElementsAre(MentionsKey(key)));
    // Names the value to use instead.
    EXPECT_THAT(f.diagnostics.errors[0],
                HasSubstr(std::to_string(kMaxRocmTraceEvents)));
    EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
  }
}

TEST(RocmTracerOptionsUtilsTest, OverLargeCountsAreClamped) {
  Fixture f;
  for (absl::string_view key : kCountKeys) {
    SetInt(f.options, key, kMaxRocmTraceEvents + 1);
  }
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_EQ(f.collector.max_callback_api_events, kMaxCount);
  EXPECT_EQ(f.collector.max_activity_api_events, kMaxCount);
  EXPECT_EQ(f.collector.max_annotation_strings, kMaxCount);
  // One clamp warning per key, plus the activity limit's "not enforced".
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(4));
  for (absl::string_view key : kCountKeys) {
    EXPECT_THAT(f.diagnostics.warnings, Contains(MentionsKey(key)));
  }
}

TEST(RocmTracerOptionsUtilsTest, LargestSupportedCountIsQuiet) {
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", kMaxRocmTraceEvents);
  SetInt(f.options, "gpu_max_annotation_strings", kMaxRocmTraceEvents);
  f.Run();

  EXPECT_EQ(f.collector.max_callback_api_events, kMaxCount);
  EXPECT_EQ(f.collector.max_annotation_strings, kMaxCount);
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, ZeroEventLimitsMeanNoLimitAsOnCuda) {
  // CUDA skips both event limits when they are 0. Passed through, 0 would make
  // the ROCm collector drop every event.
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", 0);
  SetInt(f.options, "gpu_max_activity_api_events", 0);
  f.Run();

  EXPECT_EQ(f.collector.max_callback_api_events, kMaxCount);
  EXPECT_EQ(f.collector.max_activity_api_events, kMaxCount);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  // One each: the value used for the callback limit, and the activity limit's
  // "not enforced", which already covers 0.
  EXPECT_THAT(f.diagnostics.warnings,
              UnorderedElementsAre(
                  AllOf(MentionsKey("gpu_max_callback_api_events"),
                        HasSubstr(std::to_string(kMaxRocmTraceEvents))),
                  MentionsKey("gpu_max_activity_api_events")));
}

TEST(RocmTracerOptionsUtilsTest, ZeroAnnotationStringsIsAppliedWithAWarning) {
  // Unlike the event limits, 0 retains no annotations on CUDA too.
  Fixture f;
  SetInt(f.options, "gpu_max_annotation_strings", 0);
  f.Run();

  EXPECT_EQ(f.collector.max_annotation_strings, 0);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings,
              ElementsAre(MentionsKey("gpu_max_annotation_strings")));
}

TEST(RocmTracerOptionsUtilsTest, KeyWithNoValueIsReportedOnce) {
  // Reported as having no value, not also as unrecognized.
  for (absl::string_view key :
       {"gpu_max_callback_api_events", "gpu_dump_graph_node_mapping"}) {
    SCOPED_TRACE(key);
    Fixture f;
    SetNoValue(f.options, key);
    f.Run();

    ASSERT_THAT(f.diagnostics.errors, ElementsAre(MentionsKey(key)));
    EXPECT_THAT(f.diagnostics.errors[0], Not(HasSubstr("not recognized")));
    EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
    EXPECT_TRUE(f.AllFieldsUntouched());
  }
}

TEST(RocmTracerOptionsUtilsTest, WrongTypeIsReportedOnce) {
  Fixture f;
  SetBool(f.options, "gpu_max_callback_api_events", true);
  f.Run();

  ASSERT_THAT(f.diagnostics.errors,
              ElementsAre(MentionsKey("gpu_max_callback_api_events")));
  EXPECT_THAT(f.diagnostics.errors[0], HasSubstr("int64"));
  EXPECT_THAT(f.diagnostics.errors[0], HasSubstr("bool"));
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, AllUnknownGpuKeysAreReportedInOrder) {
  Fixture f;
  SetInt(f.options, "gpu_zzz", 1);
  SetBool(f.options, "gpu_max_callbac_api_events", true);  // Typo.
  SetString(f.options, "gpu_aaa", "x");
  f.Run();

  EXPECT_THAT(f.diagnostics.errors,
              ElementsAre(MentionsKey("gpu_aaa"),
                          MentionsKey("gpu_max_callbac_api_events"),
                          MentionsKey("gpu_zzz")));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, KeysOwnedByOtherComponentsAreIgnored) {
  Fixture f;
  SetBool(f.options, "enable_continuous_profiling", true);
  SetBool(f.options, "use_system_hostname", true);
  SetString(f.options, "override_hostnames", "a,b");
  SetInt(f.options, "tracemark_lower", 1);
  SetBool(f.options, "tpu_some_key", true);
  SetInt(f.options, "gpu_max_callback_api_events", 4242);
  f.Run();

  EXPECT_TRUE(f.diagnostics.empty());
  EXPECT_EQ(f.collector.max_callback_api_events, 4242);
}

TEST(RocmTracerOptionsUtilsTest, ABadKeyDoesNotStopTheOthers) {
  Fixture f;
  SetInt(f.options, "gpu_not_a_key", 1);
  SetBool(f.options, "gpu_max_activity_api_events", true);  // Wrong type.
  SetInt(f.options, "gpu_max_callback_api_events", 4242);
  SetInt(f.options, "gpu_max_annotation_strings", 777);
  f.Run();

  EXPECT_EQ(f.collector.max_callback_api_events, 4242);
  EXPECT_EQ(f.collector.max_annotation_strings, 777);
  EXPECT_EQ(f.collector.max_activity_api_events, kActivityDefault);
  EXPECT_THAT(f.diagnostics.errors, SizeIs(2));
  EXPECT_THAT(f.diagnostics.errors, Contains(MentionsKey("gpu_not_a_key")));
  EXPECT_THAT(f.diagnostics.errors,
              Contains(MentionsKey("gpu_max_activity_api_events")));
}

TEST(RocmTracerOptionsUtilsTest, UnimplementedKeysWarn) {
  Fixture f;
  SetBool(f.options, "gpu_enable_nvtx_tracking", false);
  SetBool(f.options, "gpu_aggregated_tracing", true);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", true);
  SetString(f.options, "gpu_pm_sample_counters", "SQ_WAVES");
  SetInt(f.options, "gpu_pm_sample_interval_us", 500);
  SetInt(f.options, "gpu_pm_sample_buffer_size_per_gpu_mb", 64);
  SetBool(f.options, "gpu_dump_graph_node_mapping", true);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(7));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, BenignValuesOnUnimplementedKeysAreSilent) {
  // Each asks for what the ROCm backend already does.
  Fixture f;
  SetBool(f.options, "gpu_enable_nvtx_tracking", true);
  SetBool(f.options, "gpu_aggregated_tracing", false);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", false);
  SetString(f.options, "gpu_pm_sample_counters", "");
  SetBool(f.options, "gpu_dump_graph_node_mapping", false);
  f.Run();

  EXPECT_TRUE(f.diagnostics.empty());
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, PmSampleSettingsWithoutCountersAreSilent) {
  // As on CUDA, the interval and buffer size only matter once
  // gpu_pm_sample_counters names a counter.
  for (bool empty_counters : {false, true}) {
    SCOPED_TRACE(empty_counters ? "empty counters" : "no counters");
    Fixture f;
    if (empty_counters) {
      SetString(f.options, "gpu_pm_sample_counters", "");
    }
    SetInt(f.options, "gpu_pm_sample_interval_us", 500);
    SetInt(f.options, "gpu_pm_sample_buffer_size_per_gpu_mb", 64);
    f.Run();

    EXPECT_TRUE(f.diagnostics.empty());
  }
}

TEST(RocmTracerOptionsUtilsTest, UntypedKeyAcceptsAnyValueType) {
  const std::vector<std::function<void(ProfileOptions&)>> setters = {
      [](ProfileOptions& o) {
        SetString(o, "gpu_dump_graph_node_mapping", "/tmp/nodes.json");
      },
      [](ProfileOptions& o) { SetInt(o, "gpu_dump_graph_node_mapping", 1); },
      [](ProfileOptions& o) {
        SetBool(o, "gpu_dump_graph_node_mapping", true);
      },
  };
  for (const auto& set : setters) {
    Fixture f;
    set(f.options);
    f.Run();

    EXPECT_THAT(f.diagnostics.errors, IsEmpty());
    EXPECT_THAT(f.diagnostics.warnings,
                ElementsAre(MentionsKey("gpu_dump_graph_node_mapping")));
  }
}

TEST(RocmTracerOptionsUtilsTest, WrongTypeOnUnimplementedKeyIsOnlyAnError) {
  Fixture f;
  SetString(f.options, "gpu_pm_sample_interval_us", "500");
  f.Run();

  EXPECT_THAT(f.diagnostics.errors,
              ElementsAre(MentionsKey("gpu_pm_sample_interval_us")));
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, AllDocumentedKeysAreRecognized) {
  // The gpu_* keys documented at openxla.org/xprof/jax_profiling, with their
  // documented types. Copying the documentation must not produce an error.
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", 2 * 1024 * 1024);
  SetInt(f.options, "gpu_max_activity_api_events", 2 * 1024 * 1024);
  SetInt(f.options, "gpu_max_annotation_strings", 1024 * 1024);
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 4);
  SetBool(f.options, "gpu_enable_nvtx_tracking", true);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", true);
  SetBool(f.options, "gpu_dump_graph_node_mapping", true);
  SetString(f.options, "gpu_pm_sample_counters", "SQ_WAVES,GRBM_GUI_ACTIVE");
  SetInt(f.options, "gpu_pm_sample_interval_us", 500);
  SetInt(f.options, "gpu_pm_sample_buffer_size_per_gpu_mb", 64);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings,
              Not(Contains(MentionsKey("gpu_enable_nvtx_tracking"))));
  EXPECT_EQ(f.collector.max_callback_api_events, 2 * 1024 * 1024);
  EXPECT_EQ(f.collector.max_activity_api_events, 2 * 1024 * 1024);
  EXPECT_EQ(f.collector.max_annotation_strings, 1024 * 1024);
  EXPECT_EQ(f.collector.num_gpus, 4);
}

// The CUPTI parser spells each key it reads as a "gpu_..." literal. A key it
// gains must be applied or warned about here as well, or a configuration that
// works on CUDA fails on ROCm.
TEST(RocmTracerOptionsUtilsTest, EveryCuptiKeyIsRecognized) {
  std::string cupti_source;
  ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "backends", "profiler",
                        "gpu", "cupti_tracer_options_utils.cc"),
      &cupti_source));
  // Only a literal that is a whole key counts: "gpu_", then [a-z0-9_], then
  // the closing quote. A message such as "gpu_foo must be positive" is not one.
  auto is_key_char = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
  };
  std::vector<std::string> cupti_keys;
  for (size_t start = cupti_source.find("\"gpu_"); start != std::string::npos;
       start = cupti_source.find("\"gpu_", start + 1)) {
    size_t end = start + 1;
    while (end < cupti_source.size() && is_key_char(cupti_source[end])) {
      ++end;
    }
    if (end < cupti_source.size() && cupti_source[end] == '"') {
      cupti_keys.push_back(cupti_source.substr(start + 1, end - start - 1));
    }
  }
  // Fails rather than passing on nothing if the CUPTI parser stops spelling
  // its keys as literals.
  ASSERT_THAT(cupti_keys, SizeIs(Ge(10)));

  for (const std::string& key : cupti_keys) {
    Fixture f;
    // Only "not recognized" matters here; a type error is fine.
    SetInt(f.options, key, 1);
    f.Run();
    EXPECT_THAT(
        f.diagnostics.errors,
        Not(Contains(AllOf(MentionsKey(key), HasSubstr("is not recognized")))))
        << key
        << " is parsed by CUPTI but unknown to the ROCm tracer. Apply it in "
           "UpdateRocmTracerOptionsFromProfilerOptions() or add it to "
           "kUnimplementedOnRocm.";
  }
}

TEST(RocmTracerOptionsUtilsTest, AppendOptionDiagnosticsWritesToXSpace) {
  RocmTracerOptionDiagnostics diagnostics;
  diagnostics.errors = {"first error", "second error"};
  diagnostics.warnings = {"a warning"};

  XSpace space;
  AppendOptionDiagnostics(diagnostics, &space);

  EXPECT_THAT(space.errors(), ElementsAre("first error", "second error"));
  EXPECT_THAT(space.warnings(), ElementsAre("a warning"));
}

TEST(RocmTracerOptionsUtilsTest, AppendOptionDiagnosticsAcceptsNullSpace) {
  RocmTracerOptionDiagnostics diagnostics;
  diagnostics.errors = {"logged only"};
  AppendOptionDiagnostics(diagnostics, nullptr);  // Must not crash.
}

}  // namespace
}  // namespace profiler
}  // namespace xla
