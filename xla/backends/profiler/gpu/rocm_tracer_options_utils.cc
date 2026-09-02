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
#include <string>
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
//
// TODO(rocm-profiler): this table, and the four keys applied below, restate by
// hand the key list in cupti_tracer_options_utils.cc. Nothing keeps the two in
// step, and the failure is asymmetric: a key CUPTI gains later is not merely
// ignored here, it falls through to the unknown-key loop and is reported as a
// typo -- which costs the GPU plane outright under
// raise_error_on_start_failure. The fix is a single key table in
// xla/tsl/profiler/utils that both backends read, carrying a per-backend
// implemented/unimplemented mark. That is a cross-backend change and is
// deliberately not attempted here.
struct UnimplementedKey {
  enum Type {
    kInt64,
    kBool,
    kString,
    // No backend parses the key, so it has no documented type and there is
    // nothing to check a value against. Guessing one would turn a plausible
    // reading of the key's name into a hard error over a feature that does
    // not exist -- see gpu_dump_graph_node_mapping below.
    kAny,
  };

  // The value, if any, that asks for behaviour the ROCm backend already has.
  // Supplying it is not a request the backend is failing to honour, so it
  // draws no warning: telling someone who asked for ROCTX capture that ROCTX
  // capture "has no effect" is false, and it lands in XSpace.warnings on every
  // trace of a session that is configured correctly.
  //
  // Expressed as a value rather than a type because it is compared against the
  // supplied AdvancedConfigValue directly, which is what lets kAny keys carry
  // one too.
  enum Benign {
    kNothingIsBenign,
    kFalseIsBenign,
    kTrueIsBenign,
    kEmptyStringIsBenign,
  };

  absl::string_view name;
  Type type;
  absl::string_view reason;
  Benign benign = kNothingIsBenign;
};

constexpr UnimplementedKey kUnimplementedOnRocm[] = {
    // The listener registers MARKER_CORE_API unconditionally on the session
    // context, so ROCTX capture is always on and this key cannot turn it off.
    // Making it switchable needs a dedicated rocprofiler context that can be
    // started and stopped independently of the tracing services.
    //
    // Which is also why `true` is benign: asking for marker capture is a
    // request ROCm already satisfies. Only `false` is unhonourable.
    {"gpu_enable_nvtx_tracking", UnimplementedKey::kBool,
     "ROCTX marker capture is unconditionally enabled on the ROCm backend and "
     "this key cannot switch it off; a dedicated rocprofiler marker context is "
     "required first",
     UnimplementedKey::kTrueIsBenign},
    {"gpu_aggregated_tracing", UnimplementedKey::kBool,
     "the ROCm collector has no aggregated-tracing mode",
     UnimplementedKey::kFalseIsBenign},
    {"gpu_enable_cupti_activity_graph_trace", UnimplementedKey::kBool,
     "HIP graph tracing is not yet wired into the ROCm tracer",
     UnimplementedKey::kFalseIsBenign},
    // An empty counter list is how CUPTI spells "no PM sampling"
    // (pm_sampler_options.enable is set from !metrics.empty()), so it asks for
    // exactly what ROCm does.
    {"gpu_pm_sample_counters", UnimplementedKey::kString,
     "performance-monitor counter sampling is not yet available on ROCm",
     UnimplementedKey::kEmptyStringIsBenign},
    // No benign value: these two size a sampler that does not run, so every
    // value is a request ROCm cannot honour. They are moot without
    // gpu_pm_sample_counters, but "moot" is not "already satisfied", and the
    // warning is what tells the user the whole feature is missing.
    {"gpu_pm_sample_interval_us", UnimplementedKey::kInt64,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_pm_sample_buffer_size_per_gpu_mb", UnimplementedKey::kInt64,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_dump_graph_node_mapping", UnimplementedKey::kAny,
     "not implemented on any backend, including CUDA",
     UnimplementedKey::kFalseIsBenign},
};

// Whether the value supplied for `key` is the one this backend already
// delivers. False when the key is absent, carries no value, or carries
// anything other than `benign`.
bool ValueIsBenign(const ProfileOptions& options, absl::string_view key,
                   UnimplementedKey::Benign benign) {
  if (benign == UnimplementedKey::kNothingIsBenign) return false;
  const auto value = tsl::profiler::GetConfigValue(options, std::string(key));
  if (!value.has_value()) return false;
  switch (benign) {
    case UnimplementedKey::kFalseIsBenign:
      // bool(false) is the canonical form, but kAny keys have no documented
      // type, so int64(0) and string("false"/"0") are equally valid encodings
      // of "disabled". Accept any value that is unambiguously falsy.
      if (std::holds_alternative<bool>(*value)) {
        return !std::get<bool>(*value);
      }
      if (std::holds_alternative<int64_t>(*value)) {
        return std::get<int64_t>(*value) == 0;
      }
      return false;
    case UnimplementedKey::kTrueIsBenign:
      return std::holds_alternative<bool>(*value) && std::get<bool>(*value);
    case UnimplementedKey::kEmptyStringIsBenign:
      return std::holds_alternative<std::string>(*value) &&
             std::get<std::string>(*value).empty();
    case UnimplementedKey::kNothingIsBenign:
      return false;
  }
  return false;
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
//
// Returns whether the key was accepted (present and type-matched), NOT whether
// `setter` ran. For count keys, ApplyCount's outer lambda may push an error
// and return without calling the inner setter even on a type-match (e.g., a
// negative value). Apply still returns true for those: the type was right.
// Callers that need to know "did the field get written" must track that inside
// their own setter; callers that only need "was the key valid enough to
// consume" can use this return directly. WarnUnimplemented falls in the second
// category: its accepted bool gates the unimplemented warning, not a
// subsequent field read. Count keys do not go through WarnUnimplemented at all,
// so the distinction never bites in practice, but it is still worth naming.
template <typename T>
bool Apply(const ProfileOptions& options, absl::string_view key,
           absl::flat_hash_set<absl::string_view>& keys,
           RocmTracerOptionDiagnostics& diagnostics,
           std::function<void(T)> setter) {
  const bool key_was_present = keys.contains(key);
  const absl::Status status =
      SetValue<T>(options, std::string(key), keys, std::move(setter));
  if (!status.ok()) {
    keys.erase(key);
    // SetValue's message now names both halves of the mismatch, but it says
    // "key: X" rather than naming advanced_configuration, and it has no reason
    // to know that this backend continues afterwards. Both matter in an XSpace
    // error, which arrives with no surrounding context, so compose the
    // sentence here and take only the type names from the shared helper.
    diagnostics.errors.push_back(absl::StrCat(
        "advanced_configuration key '", key, "' expects a value of type ",
        tsl::profiler::ConfigValueTypeName<T>(), ", but a ",
        tsl::profiler::SuppliedConfigValueTypeName(options, std::string(key)),
        " was supplied. The key was ignored."));
    return false;
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
        tsl::profiler::ConfigValueTypeName<T>(), ". The key was ignored."));
    return false;
  }
  return true;
}

// The largest value the three "maximum number of X" keys may take.
//
// device_tracer_rocm.cc clamps --xla_gpu_rocm_max_trace_events to this same
// number before writing the same three fields, and for the same reason:
// RocmTraceCollectorImpl counts callback and activity events in
// std::atomic<int> (rocm_collector.h), so a cap above INT32_MAX is not merely
// generous, it is unreachable -- the counter signed-overflows before it can
// compare equal. A key and a flag that write the same field must not disagree
// about what that field may hold, or lowering the flag to bound trace memory
// stops meaning anything as soon as the key is set.
constexpr int64_t kMaxCount = 1'000'000'000;

// Applies one of the three "maximum number of X" keys, which are documented as
// int64 but land in uint64_t fields.
//
// The narrowing is why this cannot be a bare Apply<int64_t>. A negative value
// does not become a small cap, it becomes a colossal one: -1 arrives as
// 2^64-1, every `count >= max` guard in the collector becomes unreachable, and
// the limit is silently switched off rather than removed -- with the event
// counters then running past INT32_MAX into signed overflow. Someone typing a
// negative number is asking for "no limit", which this backend cannot express,
// so say so instead of guessing.
//
// An over-large value is clamped rather than rejected, because the request
// ("as high as possible") is honourable and rejecting it would leave the field
// at the flag-derived default -- lower than what was asked for, plus an error.
//
// `zero_means` completes "... is 0, so <zero_means>."
void ApplyCount(const ProfileOptions& options, absl::string_view key,
                absl::flat_hash_set<absl::string_view>& keys,
                RocmTracerOptionDiagnostics& diagnostics,
                absl::string_view zero_means,
                std::function<void(uint64_t)> setter) {
  Apply<int64_t>(
      options, key, keys, diagnostics, [&](int64_t value) {
        if (value < 0) {
          diagnostics.errors.push_back(absl::StrCat(
              "advanced_configuration key '", key, "': ", value,
              " is negative, but the field it sets is unsigned. The value "
              "would wrap to a cap no counter can ever reach, disabling the "
              "limit rather than removing it; the ROCm backend has no "
              "encoding for \"no limit\". Pass ", kMaxCount,
              " for the largest supported value. The key was ignored."));
          return;
        }
        if (value > kMaxCount) {
          diagnostics.warnings.push_back(absl::StrCat(
              "advanced_configuration key '", key, "': ", value,
              " exceeds the largest value the ROCm backend supports and was "
              "reduced to ", kMaxCount,
              ", the same ceiling --xla_gpu_rocm_max_trace_events is clamped "
              "to."));
          value = kMaxCount;
        } else if (value == 0) {
          diagnostics.warnings.push_back(absl::StrCat(
              "advanced_configuration key '", key, "' is 0, so ", zero_means,
              ". Set a positive value if that is not what was intended."));
        }
        setter(static_cast<uint64_t>(value));
      });
}

void WarnUnimplemented(const ProfileOptions& options,
                       absl::flat_hash_set<absl::string_view>& keys,
                       RocmTracerOptionDiagnostics& diagnostics) {
  for (const UnimplementedKey& key : kUnimplementedOnRocm) {
    if (!keys.contains(key.name)) continue;
    // Whether the value survived checking. Type-check even though the value is
    // discarded: a user should hear about a wrong type now, not in the release
    // that implements the key.
    bool accepted = true;
    switch (key.type) {
      case UnimplementedKey::kInt64:
        accepted =
            Apply<int64_t>(options, key.name, keys, diagnostics, [](int64_t) {});
        break;
      case UnimplementedKey::kBool:
        accepted = Apply<bool>(options, key.name, keys, diagnostics, [](bool) {});
        break;
      case UnimplementedKey::kString:
        accepted = Apply<std::string>(options, key.name, keys, diagnostics,
                                      [](const std::string&) {});
        break;
      case UnimplementedKey::kAny:
        // No documented type, so no type to check against -- but an entry that
        // carries no value at all is still malformed rather than untyped, and
        // reporting it here keeps it out of the unknown-key loop below.
        if (!tsl::profiler::GetConfigValue(options, std::string(key.name))
                 .has_value()) {
          accepted = false;
          diagnostics.errors.push_back(absl::StrCat(
              "advanced_configuration key '", key.name,
              "' is recognised but carries no value. The key was ignored."));
        }
        break;
    }
    // Load-bearing for kAny: that branch does not call Apply(), which is the
    // only other place that erases from keys. For typed keys Apply already
    // erased, so this is a no-op -- but removing it would leave every kAny
    // success in the unknown-key loop below.
    keys.erase(key.name);
    // Apply() has already said why the key was ignored. Adding "it is not an
    // error" would tell the user contradictory things about the same key.
    if (!accepted) continue;
    // The user asked for what this backend already does. Saying the key "has
    // no effect" would be false, and would put a warning in every trace of a
    // correctly configured session.
    if (ValueIsBenign(options, key.name, key.benign)) continue;
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
  // 0 is not "drop the host-side rows and keep the GPU timeline", which is the
  // natural reading and the one CUPTI's comment invites. RocmTraceCollectorImpl
  // matches every activity event to its API counterpart in
  // ApiActivityInfoExchange and drops the ones it cannot match, so an empty
  // api_events_map_ takes the kernels down with it. The is_auxiliary escape
  // hatch that spares CUPTI's correlation map is dead code here: both ROCm call
  // sites pass false.
  ApplyCount(profile_options, "gpu_max_callback_api_events", input_keys,
             diagnostics,
             "the resulting trace is empty. Not only are HIP API callback and "
             "ROCTX marker events dropped, but every activity event is "
             "discarded with them: export matches each one to the API event "
             "that shares its correlation id, and the cap prevented that event "
             "from ever being recorded",
             [&](uint64_t value) {
               collector_options.max_callback_api_events = value;
             });

  // The cap that reads this field is currently gated on a predicate that no
  // activity event satisfies, so the value lands but is not yet enforced. The
  // fix is a separate change; no further plumbing is needed here.
  //
  // Said out loud for every value that is applied, not just 0. A user who caps
  // activity events to bound a long capture's memory otherwise gets silence
  // and an unbounded trace, which reads as "the cap is working".
  bool activity_cap_applied = false;
  ApplyCount(profile_options, "gpu_max_activity_api_events", input_keys,
             diagnostics,
             "no activity events would be recorded once the cap that reads "
             "this field is enforced",
             [&](uint64_t value) {
               collector_options.max_activity_api_events = value;
               activity_cap_applied = true;
             });
  if (activity_cap_applied) {
    diagnostics.warnings.push_back(
        "advanced_configuration key 'gpu_max_activity_api_events' was stored "
        "but has no effect yet: the cap that reads it only counts events in "
        "the HIP_API domain, and every activity event is recorded as HIP_OPS, "
        "so no activity event is ever charged against the limit. The trace is "
        "currently bounded by gpu_max_callback_api_events alone.");
  }

  // One key, two fields. The tracer-side field sizes the AnnotationMap; the
  // collector-side field is currently unread and is set for consistency.
  ApplyCount(profile_options, "gpu_max_annotation_strings", input_keys,
             diagnostics,
             "no annotations are retained and kernel events lose their op "
             "names",
             [&](uint64_t value) {
               tracer_options.max_annotation_strings = value;
               collector_options.max_annotation_strings = value;
             });

  // The number of GPUs present: by this function's contract the field already
  // holds the caller's default, and device_tracer_rocm.cc derives that from
  // RocmTracer::NumGpus(). Read before the setter can overwrite it, because it
  // is what lets the two out-of-range dispositions below be carried out here
  // rather than deferred to a caller fixup that does not exist.
  const uint32_t device_count = collector_options.num_gpus;

  // Matches the CUPTI backend, whose reset-to-all for 0 and for values above
  // the device count lives in device_tracer_cuda.cc. Doing it here instead is
  // not just convenience: 0 is CUDA's spelling of "all devices", but writing it
  // through means num_gpus_==0 in RocmTraceCollectorImpl, which drops every
  // event and exports no device plane -- an empty profile for a value that on
  // CUDA profiles everything. An over-large value is just as unusable: Export
  // loops from 0 to num_gpus_ allocating an XPlane per id.
  Apply<int64_t>(
      profile_options, "gpu_num_chips_to_profile_per_task", input_keys,
      diagnostics, [&](int64_t value) {
        if (value < 0) {
          diagnostics.errors.push_back(absl::StrCat(
              "advanced_configuration key "
              "'gpu_num_chips_to_profile_per_task': ",
              value,
              " is negative; the field it sets is unsigned. Pass 0 to profile "
              "every device. The key was ignored."));
          return;
        }
        // Both dispositions leave the field at the device count. Silent for 0,
        // like the CUDA backend: it is the documented way to ask for every
        // device, and it is honoured exactly: warning about it would put a
        // message in the XSpace of every session that spelled it correctly.
        if (value == 0 || value > device_count) {
          collector_options.num_gpus = device_count;
          if (value != 0) {
            diagnostics.warnings.push_back(
                absl::StrCat("advanced_configuration key "
                             "'gpu_num_chips_to_profile_per_task': ",
                             value, " exceeds the ", device_count,
                             " GPU(s) visible to this task, so all ",
                             device_count, " are profiled."));
          }
          return;
        }
        collector_options.num_gpus = static_cast<uint32_t>(value);
        // Only when devices are actually left out. At value == device_count
        // nothing is discarded and the caveat would be noise.
        if (value < device_count) {
          diagnostics.warnings.push_back(absl::StrCat(
              "advanced_configuration key 'gpu_num_chips_to_profile_per_task' "
              "is applied post-hoc on ROCm: all ",
              device_count, " devices are traced and the events of the ",
              device_count - value,
              " outside the selected set are discarded after collection, so "
              "tracing overhead is unchanged. The set is the lowest ",
              value, " device ids, not a topology-aware choice."));
        }
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
