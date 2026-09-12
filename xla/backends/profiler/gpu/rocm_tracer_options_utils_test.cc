/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::tensorflow::ProfileOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::Contains;
using ::testing::IsEmpty;
using ::testing::SizeIs;

void SetInt(ProfileOptions& options, absl::string_view key, int64_t value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_int64_value(value);
  (*options.mutable_advanced_configuration())[std::string(key)] = config_value;
}

void SetBool(ProfileOptions& options, absl::string_view key, bool value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_bool_value(value);
  (*options.mutable_advanced_configuration())[std::string(key)] = config_value;
}

void SetString(ProfileOptions& options, absl::string_view key,
               absl::string_view value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_string_value(std::string(value));
  (*options.mutable_advanced_configuration())[std::string(key)] = config_value;
}

MATCHER_P(MentionsKey, key, absl::StrCat("mentions the key '", key, "'")) {
  return absl::StrContains(arg, absl::StrCat("'", key, "'"));
}

// Sentinel defaults, chosen so that "the field was never touched" is
// distinguishable from "the field was set to a plausible value".
constexpr uint64_t kTracerAnnotationSentinel = 111;
constexpr uint64_t kCallbackSentinel = 222;
constexpr uint64_t kActivitySentinel = 333;
constexpr uint64_t kCollectorAnnotationSentinel = 444;
constexpr uint32_t kNumGpusSentinel = 8;

struct Fixture {
  ProfileOptions options;
  RocmTracerOptions tracer = {kTracerAnnotationSentinel};
  RocmTraceCollectorOptions collector = {kCallbackSentinel, kActivitySentinel,
                                         kCollectorAnnotationSentinel,
                                         kNumGpusSentinel};
  RocmTracerOptionDiagnostics diagnostics;

  void Run() {
    UpdateRocmTracerOptionsFromProfilerOptions(options, tracer, collector,
                                               diagnostics);
  }

  // True when no option field was modified.
  bool AllFieldsUntouched() const {
    return tracer.max_annotation_strings == kTracerAnnotationSentinel &&
           collector.max_callback_api_events == kCallbackSentinel &&
           collector.max_activity_api_events == kActivitySentinel &&
           collector.max_annotation_strings == kCollectorAnnotationSentinel &&
           collector.num_gpus == kNumGpusSentinel;
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
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsActivityEventLimit) {
  Fixture f;
  SetInt(f.options, "gpu_max_activity_api_events", 4242);
  f.Run();

  EXPECT_EQ(f.collector.max_activity_api_events, 4242);
  // Wired silently: the cap that reads this field is not yet enforced, but
  // fixing that requires no further plumbing, so there is no warning to
  // retract later.
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsBothAnnotationLimitsFromOneKey) {
  Fixture f;
  SetInt(f.options, "gpu_max_annotation_strings", 777);
  f.Run();

  EXPECT_EQ(f.tracer.max_annotation_strings, 777);
  EXPECT_EQ(f.collector.max_annotation_strings, 777);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  // Wired silently, like the two event caps above. Asserted so that attaching
  // a caveat to this key later has to be a deliberate change.
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsNumGpusAndWarnsAboutPostHocDrop) {
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 2);
  f.Run();

  // The reset-to-all for out-of-range values lives in the caller, so the raw
  // value is visible here.
  EXPECT_EQ(f.collector.num_gpus, 2);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_TRUE(absl::StrContains(f.diagnostics.warnings[0], "post-hoc"));
}

TEST(RocmTracerOptionsUtilsTest, ZeroNumGpusIsAppliedAndWarnsAboutMeaningOnly) {
  // 0 has its own branch and its own warning: it is CUDA's "all GPUs", not
  // "no GPUs", and the post-hoc caveat must not also fire because nothing was
  // excluded. Without this, a refactor that dropped the early return would
  // emit two contradictory warnings for the same key with every test green.
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 0);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, 0);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_TRUE(
      absl::StrContains(f.diagnostics.warnings[0], "all available GPUs"));
  EXPECT_FALSE(absl::StrContains(f.diagnostics.warnings[0], "post-hoc"));
}

TEST(RocmTracerOptionsUtilsTest,
     RecognisedKeyWithUnsetValueIsNotCalledUnknown) {
  // An AdvancedConfigValue with no oneof case set. GetConfigValue returns
  // nullopt for it and SetValue reports success without erasing the key, so
  // without explicit handling it reaches the unknown-key loop and the user is
  // told a correctly-spelled key is not recognised -- sending them to fix the
  // one thing that is right.
  Fixture f;
  (*f.options.mutable_advanced_configuration())["gpu_max_callback_api_events"] =
      ProfileOptions::AdvancedConfigValue();
  f.Run();

  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0],
              MentionsKey("gpu_max_callback_api_events"));
  EXPECT_TRUE(absl::StrContains(f.diagnostics.errors[0], "unset"));
  EXPECT_FALSE(absl::StrContains(f.diagnostics.errors[0], "not recognised"));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest,
     OutOfRangeNumGpusIsReportedNotSilentlyDropped) {
  // int64 values that cannot be represented in the uint32_t field. Neither is
  // reachable through the caller's reset-to-all fixup, which only sees values
  // that survived the narrowing -- so if this function stays quiet, nothing
  // else will speak up and the user is told nothing at all.
  for (int64_t value : {int64_t{-1}, int64_t{1} << 40}) {
    Fixture f;
    SetInt(f.options, "gpu_num_chips_to_profile_per_task", value);
    f.Run();

    EXPECT_EQ(f.collector.num_gpus, kNumGpusSentinel)
        << "value " << value << " must not be applied";
    ASSERT_THAT(f.diagnostics.errors, SizeIs(1)) << "value " << value;
    EXPECT_THAT(f.diagnostics.errors[0],
                MentionsKey("gpu_num_chips_to_profile_per_task"));
    // The post-hoc caveat describes a key that took effect. Attaching it to a
    // value that was thrown away tells the user the opposite of the truth.
    EXPECT_THAT(f.diagnostics.warnings, IsEmpty()) << "value " << value;
  }
}

TEST(RocmTracerOptionsUtilsTest, UnknownKeyIsReportedByName) {
  Fixture f;
  SetInt(f.options, "gpu_max_callbac_api_events", 4242);  // realistic typo
  f.Run();

  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0],
              MentionsKey("gpu_max_callbac_api_events"));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, AllUnknownKeysAreReported) {
  Fixture f;
  SetInt(f.options, "gpu_not_a_key", 1);
  SetBool(f.options, "gpu_also_not_a_key", true);
  SetString(f.options, "gpu_still_not_a_key", "x");
  f.Run();

  // Every unknown key is named. A parser that returns on the first failure --
  // as the CUPTI one does -- reports only one of these.
  EXPECT_THAT(f.diagnostics.errors, SizeIs(3));
  EXPECT_THAT(f.diagnostics.errors, Contains(MentionsKey("gpu_not_a_key")));
  EXPECT_THAT(f.diagnostics.errors,
              Contains(MentionsKey("gpu_also_not_a_key")));
  EXPECT_THAT(f.diagnostics.errors,
              Contains(MentionsKey("gpu_still_not_a_key")));
}

TEST(RocmTracerOptionsUtilsTest, KeysOwnedByOtherComponentsAreLeftAlone) {
  // advanced_configuration is shared. ProfilerSession reads
  // enable_continuous_profiling out of it and passes the proto through to
  // CreateProfilers without erasing the key; session_manager injects the
  // hostname and tracemark keys; TPU keys ride the same map. Claiming them
  // would put an error in the XSpace of every correctly configured session,
  // and would cost the GPU plane outright under
  // raise_error_on_start_failure.
  Fixture f;
  SetBool(f.options, "enable_continuous_profiling", true);
  SetBool(f.options, "use_system_hostname", true);
  SetString(f.options, "override_hostnames", "a,b");
  SetInt(f.options, "tracemark_lower", 1);
  SetInt(f.options, "tracemark_upper", 2);
  SetBool(f.options, "tpu_some_future_key", true);
  SetInt(f.options, "gpu_max_callback_api_events", 4242);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
  EXPECT_EQ(f.collector.max_callback_api_events, 4242)
      << "our own key must still be applied alongside foreign ones";
}

TEST(RocmTracerOptionsUtilsTest, ABadKeyDoesNotStopTheGoodOnesApplying) {
  // The whole premise of this file. CUPTI returns on the first failure, so
  // every key after the bad one is dropped; here they must all still land.
  // Without this, a regression that reintroduced early return would leave
  // every other test in this file green.
  Fixture f;
  SetInt(f.options, "gpu_not_a_key", 1);                    // unknown
  SetBool(f.options, "gpu_max_activity_api_events", true);  // wrong type
  SetInt(f.options, "gpu_max_callback_api_events", 4242);   // good
  SetInt(f.options, "gpu_max_annotation_strings", 777);     // good
  f.Run();

  EXPECT_EQ(f.collector.max_callback_api_events, 4242);
  EXPECT_EQ(f.tracer.max_annotation_strings, 777);
  EXPECT_EQ(f.collector.max_annotation_strings, 777);
  EXPECT_EQ(f.collector.max_activity_api_events, kActivitySentinel)
      << "the wrong-typed key must not be applied";
  EXPECT_THAT(f.diagnostics.errors, SizeIs(2));
  EXPECT_THAT(f.diagnostics.errors, Contains(MentionsKey("gpu_not_a_key")));
  EXPECT_THAT(f.diagnostics.errors,
              Contains(MentionsKey("gpu_max_activity_api_events")));
}

TEST(RocmTracerOptionsUtilsTest, WrongTypeIsReportedOnce) {
  Fixture f;
  SetBool(f.options, "gpu_max_callback_api_events", true);  // documented int64
  f.Run();

  // Exactly one message. tsl::profiler::SetValue returns its type error before
  // erasing the key from the working set, so a parser that forgets the
  // explicit erase reports this key twice: once as a type error and once as
  // unrecognised.
  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0],
              MentionsKey("gpu_max_callback_api_events"));
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
  EXPECT_EQ(f.collector.max_callback_api_events, kCallbackSentinel);
}

TEST(RocmTracerOptionsUtilsTest, UnimplementedKeysAreAcceptedWithWarnings) {
  Fixture f;
  SetBool(f.options, "gpu_enable_nvtx_tracking", true);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", true);
  SetBool(f.options, "gpu_dump_graph_node_mapping", true);
  SetString(f.options, "gpu_pm_sample_counters", "SQ_WAVES");
  SetInt(f.options, "gpu_pm_sample_interval_us", 500);
  SetInt(f.options, "gpu_pm_sample_buffer_size_per_gpu_mb", 64);
  SetBool(f.options, "gpu_aggregated_tracing", true);
  f.Run();

  // A script written against CUDA must not fail on ROCm just because a key is
  // not implemented here yet.
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(7));
  EXPECT_THAT(f.diagnostics.warnings,
              Contains(MentionsKey("gpu_enable_nvtx_tracking")));
  EXPECT_THAT(f.diagnostics.warnings,
              Contains(MentionsKey("gpu_pm_sample_counters")));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, WrongTypeOnAnUnimplementedKeyStillErrors) {
  Fixture f;
  SetString(f.options, "gpu_pm_sample_interval_us", "500");  // documented int64
  f.Run();

  // Type-checking the unimplemented keys is not decorative: a user should hear
  // about this now, not in the release that implements the key.
  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0],
              MentionsKey("gpu_pm_sample_interval_us"));
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(1));
}

TEST(RocmTracerOptionsUtilsTest, AllDocumentedKeysAreRecognised) {
  // The ten gpu_* keys published at openxla.org/xprof/jax_profiling (the
  // advanced_profiler_options page is linked as "the complete list" but is
  // TPU-only and lists none of these),
  // with their documented types. A user copying the page verbatim must not see
  // a single error on ROCm.
  //
  // Note what this does and does not pin. It fails if a key listed here stops
  // being recognised, or if its documented type stops being accepted. It
  // cannot notice a key the page gains later: the list is hardcoded and
  // nothing here reads the page. Keeping the two in step is still manual.
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
  // Six unimplemented keys, plus the post-hoc caveat on num_chips.
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(7));
  EXPECT_EQ(f.collector.max_callback_api_events, 2 * 1024 * 1024);
  EXPECT_EQ(f.tracer.max_annotation_strings, 1024 * 1024);
  EXPECT_EQ(f.collector.num_gpus, 4);
}

TEST(RocmTracerOptionsUtilsTest, AppendOptionDiagnosticsWritesToXSpace) {
  RocmTracerOptionDiagnostics diagnostics;
  diagnostics.errors = {"first error", "second error"};
  diagnostics.warnings = {"a warning"};

  XSpace space;
  AppendOptionDiagnostics(diagnostics, &space);

  ASSERT_EQ(space.errors_size(), 2);
  EXPECT_EQ(space.errors(0), "first error");
  EXPECT_EQ(space.errors(1), "second error");
  ASSERT_EQ(space.warnings_size(), 1);
  EXPECT_EQ(space.warnings(0), "a warning");
}

TEST(RocmTracerOptionsUtilsTest, AppendOptionDiagnosticsToleratesEmptyAndNull) {
  XSpace space;
  AppendOptionDiagnostics(RocmTracerOptionDiagnostics{}, &space);
  EXPECT_EQ(space.errors_size(), 0);
  EXPECT_EQ(space.warnings_size(), 0);

  // A null space must not crash, and must not swallow the diagnostic either:
  // the LOG mirror runs regardless, so the message still reaches somebody.
  // Only the XSpace writes are skipped. Asserting on the log itself would need
  // a scoped log sink and a dependency this target does not otherwise carry;
  // what is pinned here is that the call is safe and unconditional.
  RocmTracerOptionDiagnostics diagnostics;
  diagnostics.errors = {"still reaches the log"};
  AppendOptionDiagnostics(diagnostics, nullptr);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
