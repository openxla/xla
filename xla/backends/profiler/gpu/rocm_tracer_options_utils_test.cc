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
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
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
// Not a sentinel: the contract says num_gpus arrives holding the number of
// GPUs present, and the parser reads it back to resolve out-of-range values.
constexpr uint32_t kDeviceCount = 8;

// The ceiling rocm_tracer_options_utils.cc clamps the three count keys to,
// which is also what device_tracer_rocm.cc clamps
// --xla_gpu_rocm_max_trace_events to. Duplicated rather than exported: the
// constant is an internal detail of the parser, and a test that reached in for
// it could no longer notice the two drifting apart.
constexpr int64_t kMaxCount = 1'000'000'000;
constexpr uint64_t kMaxCountU = static_cast<uint64_t>(kMaxCount);

// Every key that takes one of those counts, so that the range rules can be
// asserted once for all three rather than once per key.
constexpr absl::string_view kCountKeys[] = {
    "gpu_max_callback_api_events",
    "gpu_max_activity_api_events",
    "gpu_max_annotation_strings",
};

struct Fixture {
  ProfileOptions options;
  // Designated initialisers, not positional ones: a field inserted into
  // either struct would otherwise re-bind these sentinels to the wrong members
  // and surface as an assertion failure in some unrelated test.
  RocmTracerOptions tracer = {.max_annotation_strings =
                                  kTracerAnnotationSentinel};
  RocmTraceCollectorOptions collector = {
      .max_callback_api_events = kCallbackSentinel,
      .max_activity_api_events = kActivitySentinel,
      .max_annotation_strings = kCollectorAnnotationSentinel,
      .num_gpus = kDeviceCount,
  };
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
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsActivityEventLimitAndSaysItIsNotEnforced) {
  Fixture f;
  SetInt(f.options, "gpu_max_activity_api_events", 4242);
  f.Run();

  EXPECT_EQ(f.collector.max_activity_api_events, 4242);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  // The value lands but nothing enforces it: the cap counts only HIP_API
  // events and activity events are all HIP_OPS. Silence here would let someone
  // capping activity events to bound a long capture watch the trace grow
  // unbounded with no indication why.
  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_THAT(f.diagnostics.warnings[0],
              MentionsKey("gpu_max_activity_api_events"));
  EXPECT_THAT(f.diagnostics.warnings[0], HasSubstr("no effect yet"));
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

  EXPECT_EQ(f.collector.num_gpus, 2);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_TRUE(absl::StrContains(f.diagnostics.warnings[0], "post-hoc"));
  EXPECT_TRUE(
      absl::StrContains(f.diagnostics.warnings[0], "outside the selected set"))
      << f.diagnostics.warnings[0];
}

TEST(RocmTracerOptionsUtilsTest, NumGpusEqualToTheDeviceCountIsQuiet) {
  // Nothing is discarded, so the post-hoc caveat would be describing a drop
  // that does not happen. The boundary that separates it from the test above.
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", kDeviceCount);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, kDeviceCount);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, NumGpusAboveTheDeviceCountIsResetToAll) {
  // Not passed through: RocmTraceCollectorImpl::Export loops from 0 to
  // num_gpus_ allocating an XPlane per id, so an implausible value is not a
  // harmless over-request, it is an out-of-memory hang inside the profiled
  // process. The reset mirrors device_tracer_cuda.cc, which does the same for
  // CUPTI.
  for (int64_t value : {int64_t{kDeviceCount} + 1, int64_t{4096},
                        int64_t{3'000'000'000}, int64_t{1} << 40}) {
    Fixture f;
    SetInt(f.options, "gpu_num_chips_to_profile_per_task", value);
    f.Run();

    EXPECT_EQ(f.collector.num_gpus, kDeviceCount) << "value " << value;
    EXPECT_THAT(f.diagnostics.errors, IsEmpty()) << "value " << value;
    ASSERT_THAT(f.diagnostics.warnings, SizeIs(1)) << "value " << value;
    EXPECT_THAT(f.diagnostics.warnings[0],
                MentionsKey("gpu_num_chips_to_profile_per_task"));
    // The post-hoc caveat describes a selection that excludes devices. Nothing
    // was excluded here, so attaching it would be false.
    EXPECT_FALSE(absl::StrContains(f.diagnostics.warnings[0], "post-hoc"))
        << f.diagnostics.warnings[0];
  }
}

TEST(RocmTracerOptionsUtilsTest, ZeroNumGpusMeansEveryDevice) {
  // 0 is CUDA's spelling of "all GPUs" and must be resolved to the device
  // count here rather than written through: num_gpus_==0 makes
  // RocmTraceCollectorImpl drop every event and export no device plane, so
  // passing it on turns "profile everything" into an empty trace. Quiet,
  // because the request is honoured exactly as asked -- the same reason
  // device_tracer_cuda.cc suppresses its warning for 0 alone.
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 0);
  f.Run();

  EXPECT_EQ(f.collector.num_gpus, kDeviceCount);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
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

TEST(RocmTracerOptionsUtilsTest, NegativeNumGpusIsReportedNotWrappedAround) {
  // The field is uint32_t, so -1 would arrive as 4294967295 and take Export's
  // per-device loop with it. Rejected rather than reset to "all", because
  // unlike 0 a negative count is not a documented way to ask for anything.
  for (int64_t value : {int64_t{-1}, -(int64_t{1} << 40)}) {
    Fixture f;
    SetInt(f.options, "gpu_num_chips_to_profile_per_task", value);
    f.Run();

    EXPECT_EQ(f.collector.num_gpus, kDeviceCount)
        << "value " << value << " must not be applied";
    ASSERT_THAT(f.diagnostics.errors, SizeIs(1)) << "value " << value;
    EXPECT_THAT(f.diagnostics.errors[0],
                MentionsKey("gpu_num_chips_to_profile_per_task"));
    // The message has to say what to write instead.
    EXPECT_THAT(f.diagnostics.errors[0], HasSubstr("Pass 0"));
    // The post-hoc caveat describes a key that took effect. Attaching it to a
    // value that was thrown away tells the user the opposite of the truth.
    EXPECT_THAT(f.diagnostics.warnings, IsEmpty()) << "value " << value;
  }
}

TEST(RocmTracerOptionsUtilsTest, NegativeCountsAreRejectedNotWrappedAround) {
  // The bug this guards is invisible at the call site: the keys are documented
  // as int64 and the fields are uint64_t, so -1 does not become a small cap,
  // it becomes 2^64-1. Every `count >= max` guard in RocmTraceCollectorImpl
  // then becomes unreachable, the cap is silently switched off, and the
  // std::atomic<int> counters run past INT32_MAX into signed overflow -- while
  // device_tracer_rocm.cc clamps the flag that writes these same fields to
  // 1e9 precisely to keep that from happening.
  for (absl::string_view key : kCountKeys) {
    Fixture f;
    SetInt(f.options, key, -1);
    f.Run();

    EXPECT_TRUE(f.AllFieldsUntouched()) << key << " = -1 must not be applied";
    ASSERT_THAT(f.diagnostics.errors, SizeIs(1)) << key;
    EXPECT_THAT(f.diagnostics.errors[0], MentionsKey(key));
    // The message has to say what to write instead; "invalid" alone leaves
    // someone who meant "no limit" with nowhere to go.
    EXPECT_TRUE(absl::StrContains(f.diagnostics.errors[0],
                                  absl::StrCat(kMaxCount)))
        << f.diagnostics.errors[0];
    EXPECT_THAT(f.diagnostics.warnings, IsEmpty()) << key;
  }
}

TEST(RocmTracerOptionsUtilsTest, OverLargeCountsAreClampedWithAWarning) {
  // Clamped rather than rejected: "as high as possible" is an honourable ask,
  // and rejecting it would leave the field at the flag-derived default, which
  // is *lower* than what was requested -- less than the user wanted, plus an
  // error telling them so.
  Fixture f;
  for (absl::string_view key : kCountKeys) SetInt(f.options, key, kMaxCount + 1);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_EQ(f.collector.max_callback_api_events, kMaxCountU);
  EXPECT_EQ(f.collector.max_activity_api_events, kMaxCountU);
  EXPECT_EQ(f.collector.max_annotation_strings, kMaxCountU);
  EXPECT_EQ(f.tracer.max_annotation_strings, kMaxCountU);
  // One clamp warning per key, plus the standing caveat that the activity cap
  // is not enforced yet.
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(4));
  for (absl::string_view key : kCountKeys) {
    EXPECT_THAT(f.diagnostics.warnings, Contains(MentionsKey(key)));
  }
}

TEST(RocmTracerOptionsUtilsTest, TheLargestSupportedCountIsAcceptedQuietly) {
  // The boundary the two tests above straddle. Without it, an off-by-one in
  // either direction leaves both of them passing.
  Fixture f;
  for (absl::string_view key : kCountKeys) SetInt(f.options, key, kMaxCount);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  // Quiet about the range. The one warning is the activity cap's standing
  // "not enforced yet" caveat, which is about the field, not the value.
  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_THAT(f.diagnostics.warnings[0],
              MentionsKey("gpu_max_activity_api_events"));
  EXPECT_EQ(f.collector.max_callback_api_events, kMaxCountU);
  EXPECT_EQ(f.tracer.max_annotation_strings, kMaxCountU);
}

TEST(RocmTracerOptionsUtilsTest, ZeroCountsAreAppliedButExplained) {
  // 0 is a legal value that produces a trace with nothing in it. num_gpus==0
  // already gets its own explanation; these three are just as destructive and
  // used to get none, which is the silent empty result this file exists to
  // prevent.
  for (absl::string_view key : kCountKeys) {
    Fixture f;
    SetInt(f.options, key, 0);
    f.Run();

    EXPECT_THAT(f.diagnostics.errors, IsEmpty()) << key;
    // The activity cap carries a second, value-independent caveat.
    const int expected = key == "gpu_max_activity_api_events" ? 2 : 1;
    ASSERT_THAT(f.diagnostics.warnings, SizeIs(expected)) << key;
    EXPECT_THAT(f.diagnostics.warnings[0], MentionsKey(key));
    EXPECT_FALSE(f.AllFieldsUntouched()) << key << " = 0 is still applied";
  }
}

TEST(RocmTracerOptionsUtilsTest, ZeroCallbackEventsIsExplainedAsAnEmptyTrace) {
  // The tempting reading of 0 -- "drop the host rows, keep the GPU timeline" --
  // is wrong on ROCm: activity events are matched to their API counterpart
  // before export, so capping callbacks at 0 discards the kernels too. Pinned
  // because the wording is the only thing standing between a user and a
  // silently empty profile.
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", 0);
  f.Run();

  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_THAT(f.diagnostics.warnings[0], HasSubstr("trace is empty"));
  EXPECT_THAT(f.diagnostics.warnings[0], HasSubstr("activity event"));
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
  // Every value here is a request the ROCm backend genuinely cannot honour:
  // turning ROCTX capture off, turning aggregation and graph tracing on,
  // naming PM counters. Values the backend already satisfies are the subject
  // of BenignValuesOnUnimplementedKeysAreSilent below.
  SetBool(f.options, "gpu_enable_nvtx_tracking", false);
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

TEST(RocmTracerOptionsUtilsTest, BenignValuesOnUnimplementedKeysAreSilent) {
  // Each of these asks for the disposition the ROCm backend already has:
  // ROCTX capture on, aggregation off, graph tracing off, no PM counters, no
  // graph-node dump. Telling the user the key "has no effect" would be false
  // -- they got exactly what they asked for -- and would put a warning into
  // XSpace.warnings on every trace of a correctly configured session.
  Fixture f;
  SetBool(f.options, "gpu_enable_nvtx_tracking", true);
  SetBool(f.options, "gpu_aggregated_tracing", false);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", false);
  SetString(f.options, "gpu_pm_sample_counters", "");
  SetBool(f.options, "gpu_dump_graph_node_mapping", false);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest,
     UntypedKeyWithInt64ZeroIsTreatedAsBenignFalse) {
  // gpu_dump_graph_node_mapping has kAny type, so int64(0) is as valid as
  // bool(false). Both express "disabled" and ValueIsBenign must accept both.
  Fixture f;
  SetInt(f.options, "gpu_dump_graph_node_mapping", 0);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, TheOppositeOfABenignValueStillWarns) {
  // The mirror of the test above, and the reason the suppression is keyed on
  // the value rather than on the key: gpu_enable_nvtx_tracking=false is a real
  // request that ROCm cannot satisfy, and it must not go quiet along with its
  // benign twin.
  Fixture f;
  SetBool(f.options, "gpu_enable_nvtx_tracking", false);
  SetBool(f.options, "gpu_aggregated_tracing", true);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(2));
  EXPECT_THAT(f.diagnostics.warnings,
              Contains(MentionsKey("gpu_enable_nvtx_tracking")));
  EXPECT_THAT(f.diagnostics.warnings,
              Contains(MentionsKey("gpu_aggregated_tracing")));
}

TEST(RocmTracerOptionsUtilsTest, UntypedKeyAcceptsAnyValueType) {
  // gpu_dump_graph_node_mapping is parsed by no backend, so it has no
  // documented type. Guessing bool would turn '/tmp/nodes.json' -- a plausible
  // reading of a key named "dump ... mapping" -- into an error, and under
  // raise_error_on_start_failure that costs the session over a feature that
  // does not exist anywhere.
  for (int variant = 0; variant < 3; ++variant) {
    Fixture f;
    switch (variant) {
      case 0:
        SetString(f.options, "gpu_dump_graph_node_mapping", "/tmp/nodes.json");
        break;
      case 1:
        SetInt(f.options, "gpu_dump_graph_node_mapping", 1);
        break;
      case 2:
        SetBool(f.options, "gpu_dump_graph_node_mapping", true);
        break;
    }
    f.Run();

    EXPECT_THAT(f.diagnostics.errors, IsEmpty()) << "variant " << variant;
    ASSERT_THAT(f.diagnostics.warnings, SizeIs(1)) << "variant " << variant;
    EXPECT_THAT(f.diagnostics.warnings[0],
                MentionsKey("gpu_dump_graph_node_mapping"));
  }
}

TEST(RocmTracerOptionsUtilsTest, UntypedKeyWithNoValueIsStillReported) {
  // No type to check does not mean nothing to check: an entry carrying no
  // oneof case at all is malformed, and without this it would fall through to
  // the unknown-key loop and be reported as a misspelling.
  Fixture f;
  (*f.options.mutable_advanced_configuration())["gpu_dump_graph_node_mapping"] =
      ProfileOptions::AdvancedConfigValue();
  f.Run();

  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0],
              MentionsKey("gpu_dump_graph_node_mapping"));
  EXPECT_FALSE(absl::StrContains(f.diagnostics.errors[0], "not recognised"));
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
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
  // A type error means the key was rejected; the "not an error" warning must
  // not also fire -- that would tell the user contradictory things about the
  // same key in the same trace.
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
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
  // Five of the six unimplemented keys, plus the post-hoc caveat on num_chips
  // (4 of 8 devices) and the activity cap's "not enforced yet" caveat.
  // gpu_enable_nvtx_tracking=true is missing on purpose: ROCTX capture is
  // unconditional, so that value is already in effect and warning about it
  // would be wrong.
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(7));
  EXPECT_THAT(f.diagnostics.warnings,
              Not(Contains(MentionsKey("gpu_enable_nvtx_tracking"))));
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
