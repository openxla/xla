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
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tensorflow::ProfileOptions;
using tsl::profiler::SetValue;

// Keys the CUPTI backend recognises that have no ROCm implementation yet.
// Each carries the type it is documented with, so that a wrong-typed value is
// still reported now rather than in the release that implements the key.
//
// Listed in the order cupti_tracer_options_utils.cc parses them, so that the
// two parsers can be read side by side. The one exception is
// gpu_dump_graph_node_mapping, which CUPTI does not parse at all; it is last.
struct UnimplementedKey {
  enum Type { kInt64, kBool, kString };

  absl::string_view name;
  Type type;
  absl::string_view reason;
};

constexpr UnimplementedKey kUnimplementedOnRocm[] = {
    // The listener registers MARKER_CORE_API unconditionally on the session
    // context, so ROCTX capture is always on and this key cannot turn it off.
    // Making it switchable needs a dedicated rocprofiler context that can be
    // started and stopped independently of the tracing services.
    {"gpu_enable_nvtx_tracking", UnimplementedKey::kBool,
     "ROCTX marker capture is unconditionally enabled on the ROCm backend and "
     "this key cannot switch it off; a dedicated rocprofiler marker context is "
     "required first"},
    {"gpu_aggregated_tracing", UnimplementedKey::kBool,
     "the ROCm collector has no aggregated-tracing mode"},
    {"gpu_enable_cupti_activity_graph_trace", UnimplementedKey::kBool,
     "HIP graph tracing is not yet wired into the ROCm tracer"},
    {"gpu_pm_sample_counters", UnimplementedKey::kString,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_pm_sample_interval_us", UnimplementedKey::kInt64,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_pm_sample_buffer_size_per_gpu_mb", UnimplementedKey::kInt64,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_dump_graph_node_mapping", UnimplementedKey::kBool,
     "not implemented on any backend, including CUDA"},
};

// The name of the type a key expects. SetValue's own failure message says only
// "Expected a different type", which omits the one fact the user needs.
template <typename T>
constexpr absl::string_view ExpectedTypeName();
template <>
constexpr absl::string_view ExpectedTypeName<int64_t>() {
  return "int64";
}
template <>
constexpr absl::string_view ExpectedTypeName<bool>() {
  return "bool";
}
template <>
constexpr absl::string_view ExpectedTypeName<std::string>() {
  return "string";
}

// The name of the type actually supplied, so that a type error can report both
// halves of the mismatch rather than only what was expected.
absl::string_view SuppliedTypeName(const ProfileOptions& options,
                                   absl::string_view key) {
  const auto value = tsl::profiler::GetConfigValue(options, std::string(key));
  if (!value.has_value()) return "unset";
  return std::visit(
      [](const auto& alternative) -> absl::string_view {
        using Alternative = std::decay_t<decltype(alternative)>;
        if constexpr (std::is_same_v<Alternative, std::string>) {
          return "string";
        } else if constexpr (std::is_same_v<Alternative, bool>) {
          return "bool";
        } else {
          return "int64";
        }
      },
      *value);
}

// The accumulating analogue of the ABSL_RETURN_IF_ERROR that wraps every
// SetValue call in cupti_tracer_options_utils.cc: same helper, same type
// semantics, but one bad key does not hide the others and does not abort the
// session. That difference is the point of this file -- see the class comment
// on RocmTracerOptionDiagnostics.
//
// SetValue returns its type error *before* erasing the key from the working
// set, so the erase below is load-bearing: without it a wrong-typed key would
// be reported twice, once as a type error and once as unrecognised. The CUPTI
// parser never hits this because it stops at the first failure.
//
// The setter is std::function rather than absl::FunctionRef because that is
// what SetValue itself takes; a FunctionRef here would only force a conversion.
template <typename T>
void Apply(const ProfileOptions& options, absl::string_view key,
           absl::flat_hash_set<absl::string_view>& keys,
           RocmTracerOptionDiagnostics& diagnostics,
           std::function<void(T)> setter) {
  const bool key_was_present = keys.contains(key);
  const absl::Status status =
      SetValue<T>(options, std::string(key), keys, std::move(setter));
  if (!status.ok()) {
    keys.erase(key);
    // SetValue's message names the key and says only that the type differs, so
    // build our own rather than wrapping it: naming the key once and the
    // expected type explicitly is the whole value of this diagnostic.
    diagnostics.errors.push_back(absl::StrCat(
        "advanced_configuration key '", key, "' expects a value of type ",
        ExpectedTypeName<T>(), ", but a ", SuppliedTypeName(options, key),
        " was supplied. The key was ignored."));
    return;
  }
  // SetValue erases the key only when it found a usable value. A key that was
  // in the map and is still here therefore carried an AdvancedConfigValue with
  // no oneof case set -- GetConfigValue returns nullopt for that, and SetValue
  // reports it as success. Left alone it would fall through to the unknown-key
  // loop and be reported as a misspelling, which is the one thing it is not.
  if (key_was_present && keys.contains(key)) {
    keys.erase(key);
    diagnostics.errors.push_back(absl::StrCat(
        "advanced_configuration key '", key,
        "' is recognised but its value is unset; expected a value of type ",
        ExpectedTypeName<T>(), ". The key was ignored."));
  }
}

void WarnUnimplemented(const ProfileOptions& options,
                       absl::flat_hash_set<absl::string_view>& keys,
                       RocmTracerOptionDiagnostics& diagnostics) {
  for (const UnimplementedKey& key : kUnimplementedOnRocm) {
    if (!keys.contains(key.name)) continue;
    // Type-check even though the value is discarded.
    switch (key.type) {
      case UnimplementedKey::kInt64:
        Apply<int64_t>(options, key.name, keys, diagnostics, [](int64_t) {});
        break;
      case UnimplementedKey::kBool:
        Apply<bool>(options, key.name, keys, diagnostics, [](bool) {});
        break;
      case UnimplementedKey::kString:
        Apply<std::string>(options, key.name, keys, diagnostics,
                           [](const std::string&) {});
        break;
    }
    keys.erase(key.name);
    diagnostics.warnings.push_back(absl::StrCat(
        "advanced_configuration key '", key.name,
        "' names a feature the ROCm backend does not implement: ", key.reason,
        ". The key is recognised, so it is not an error, but it has no "
        "effect."));
  }
}

}  // namespace

void UpdateRocmTracerOptionsFromProfilerOptions(
    const ProfileOptions& profile_options, RocmTracerOptions& tracer_options,
    RocmTraceCollectorOptions& collector_options,
    RocmTracerOptionDiagnostics& diagnostics) {
  absl::flat_hash_set<absl::string_view> input_keys;
  for (const auto& [key, unused_value] :
       profile_options.advanced_configuration()) {
    input_keys.insert(key);
  }

  // Note this budget is no longer callbacks alone: the ROCTX listener charges
  // every Generic marker event against the same counter, so lowering this key
  // also caps how many application roctx ranges survive a session.
  Apply<int64_t>(profile_options, "gpu_max_callback_api_events", input_keys,
                 diagnostics, [&](int64_t value) {
                   collector_options.max_callback_api_events = value;
                 });

  // The cap that reads this field is currently gated on a predicate that no
  // activity event satisfies, so the value lands but is not yet enforced. The
  // fix is a separate change; no further plumbing is needed here.
  Apply<int64_t>(profile_options, "gpu_max_activity_api_events", input_keys,
                 diagnostics, [&](int64_t value) {
                   collector_options.max_activity_api_events = value;
                 });

  // One key, two fields. The tracer-side field sizes the AnnotationMap; the
  // collector-side field is currently unread and is set for consistency.
  Apply<int64_t>(profile_options, "gpu_max_annotation_strings", input_keys,
                 diagnostics, [&](int64_t value) {
                   tracer_options.max_annotation_strings = value;
                   collector_options.max_annotation_strings = value;
                 });

  // Matches the CUPTI backend, including its reset-to-all disposition for
  // values above the device count. That reset is performed by the caller,
  // which is the only place that knows how many GPUs are present -- but only
  // for values that fit in the uint32_t field. A value that does not fit never
  // reaches the caller's fixup, so it has to be reported here or not at all.
  Apply<int64_t>(
      profile_options, "gpu_num_chips_to_profile_per_task", input_keys,
      diagnostics, [&](int64_t value) {
        // The bound is int32, not uint32. The field here is uint32_t, but
        // RocmTraceCollectorImpl stores it in a signed int, so anything
        // above INT32_MAX arrives there negative: Export loops zero times
        // and writes no device planes, while Flush admits every event.
        // Checking the wider bound would let that through silently.
        if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
          diagnostics.errors.push_back(absl::StrCat(
              "advanced_configuration key "
              "'gpu_num_chips_to_profile_per_task': ",
              value, " is outside the representable range [0, ",
              std::numeric_limits<int32_t>::max(), "]. The key was ignored."));
          return;
        }
        collector_options.num_gpus = static_cast<uint32_t>(value);
        if (value == 0) {
          diagnostics.warnings.push_back(
              "advanced_configuration key 'gpu_num_chips_to_profile_per_task' "
              "is 0. The value is passed through unchanged here; the tracer "
              "then profiles all available GPUs, matching the CUDA backend. "
              "Note that 0 is not meaningful to the collector on its own.");
          return;
        }
        diagnostics.warnings.push_back(
            "advanced_configuration key 'gpu_num_chips_to_profile_per_task' "
            "is applied post-hoc on ROCm: events from excluded devices are "
            "discarded after collection, so tracing overhead is unchanged. "
            "Devices are selected by ascending device id, not by topology.");
      });

  WarnUnimplemented(profile_options, input_keys, diagnostics);

  // advanced_configuration is a namespace shared with the rest of the profiler,
  // not private to this backend: ProfilerSession reads
  // "enable_continuous_profiling" out of it and hands the same proto, key
  // included, to CreateProfilers; session_manager injects
  // "use_system_hostname", "override_hostnames", "tracemark_lower" and
  // "tracemark_upper"; TPU keys pass through the same map. None of them are
  // erased before a tracer factory runs.
  //
  // So claim only the gpu_ prefix. Treating the whole map as ours would put
  // "advanced_configuration key 'enable_continuous_profiling' is not
  // recognised" into XSpace.errors on every trace of a correctly configured
  // session -- and, under raise_error_on_start_failure, would cost the user the
  // GPU plane over an option that has nothing to do with this backend. That is
  // precisely the failure this file exists to prevent. An unrecognised gpu_ key
  // is still an error: that is a typo in a key we do own.
  std::vector<absl::string_view> unknown;
  for (absl::string_view key : input_keys) {
    if (absl::StartsWith(key, "gpu_")) unknown.push_back(key);
  }
  // Sorted so that the output is stable across runs.
  absl::c_sort(unknown);
  for (absl::string_view key : unknown) {
    diagnostics.errors.push_back(absl::StrCat(
        "advanced_configuration key '", key,
        "' is not recognised by the ROCm GPU tracer and was ignored."));
  }
}

void AppendOptionDiagnostics(const RocmTracerOptionDiagnostics& diagnostics,
                             tensorflow::profiler::XSpace* space) {
  // The log mirror happens even without an XSpace. A null space is the one
  // case where the diagnostic has nowhere else to go, so returning early here
  // would drop it entirely -- reproducing, for that path, exactly the silent
  // discard this file exists to prevent.
  for (const std::string& message : diagnostics.errors) {
    LOG(ERROR) << message;
    if (space != nullptr) space->add_errors(message);
  }
  for (const std::string& message : diagnostics.warnings) {
    LOG(WARNING) << message;
    if (space != nullptr) space->add_warnings(message);
  }
}

}  // namespace profiler
}  // namespace xla
