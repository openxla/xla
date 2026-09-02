/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
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
using ConfigValue = std::variant<std::string, bool, int64_t>;

enum class ValueType { kInt64, kBool, kString };

// A value that asks for what the ROCm backend already does, so it draws no
// warning. std::monostate means no value does.
using BenignValue = std::variant<std::monostate, bool, absl::string_view>;

struct UnimplementedKey {
  absl::string_view name;
  // std::nullopt accepts a value of any type.
  std::optional<ValueType> type;
  absl::string_view reason;
  BenignValue benign;
};

// Keys the CUPTI backend implements and the ROCm backend does not, in the
// order cupti_tracer_options_utils.cc parses them, followed by one that no
// backend parses. They are type-checked and warned about rather than rejected,
// so that a configuration written for CUDA still works here.
//
// TODO(rocm-profiler): this table and the keys applied below restate the CUPTI
// key list by hand, so a key CUPTI gains later is reported here as
// unrecognised. A key table in xla/tsl/profiler/utils read by both backends
// would keep them in step. Until then, cupti_tracer_options_utils.cc points
// here.
constexpr UnimplementedKey kUnimplementedOnRocm[] = {
    {"gpu_enable_nvtx_tracking", ValueType::kBool,
     "ROCTX marker capture is always requested and cannot be switched off",
     true},
    {"gpu_aggregated_tracing", ValueType::kBool,
     "the ROCm collector has no aggregated-tracing mode", false},
    {"gpu_enable_cupti_activity_graph_trace", ValueType::kBool,
     "graph-level activity records are not supported; kernels launched from "
     "HIP graphs are still traced individually",
     false},
    // An empty list is how CUDA spells "no PM sampling".
    {"gpu_pm_sample_counters", ValueType::kString,
     "performance-monitor counter sampling is not supported",
     absl::string_view()},
    {"gpu_pm_sample_interval_us", ValueType::kInt64,
     "performance-monitor counter sampling is not supported", std::monostate()},
    {"gpu_pm_sample_buffer_size_per_gpu_mb", ValueType::kInt64,
     "performance-monitor counter sampling is not supported", std::monostate()},
    // Documented for JAX but parsed by no backend, so it has no type to check.
    {"gpu_dump_graph_node_mapping", std::nullopt, "no backend implements it",
     false},
};

absl::string_view TypeName(ValueType type) {
  switch (type) {
    case ValueType::kInt64:
      return "int64";
    case ValueType::kBool:
      return "bool";
    case ValueType::kString:
      return "string";
  }
  return "unknown";
}

absl::string_view TypeName(const ConfigValue& value) {
  if (std::holds_alternative<int64_t>(value)) return "int64";
  if (std::holds_alternative<bool>(value)) return "bool";
  return "string";
}

bool HasType(const ConfigValue& value, ValueType type) {
  switch (type) {
    case ValueType::kInt64:
      return std::holds_alternative<int64_t>(value);
    case ValueType::kBool:
      return std::holds_alternative<bool>(value);
    case ValueType::kString:
      return std::holds_alternative<std::string>(value);
  }
  return false;
}

bool IsBenign(const ConfigValue& value, const BenignValue& benign) {
  if (const bool* expected = std::get_if<bool>(&benign)) {
    const bool* actual = std::get_if<bool>(&value);
    return actual != nullptr && *actual == *expected;
  }
  if (const absl::string_view* expected =
          std::get_if<absl::string_view>(&benign)) {
    const std::string* actual = std::get_if<std::string>(&value);
    return actual != nullptr && *actual == *expected;
  }
  return false;
}

// Returns "advanced_configuration key '<key>' <predicate>".
std::string KeyMessage(absl::string_view key, absl::string_view predicate) {
  return absl::StrCat("advanced_configuration key '", key, "' ", predicate);
}

// Removes `key` from `keys` and returns its value if it has type `type`, or any
// type if `type` is nullopt. Returns nullopt if the key is absent, and also,
// after recording an error, if it has no value or a value of another type.
std::optional<ConfigValue> TakeValue(
    const ProfileOptions& options, absl::string_view key,
    std::optional<ValueType> type, absl::flat_hash_set<absl::string_view>& keys,
    RocmTracerOptionDiagnostics& diagnostics) {
  if (keys.erase(key) == 0) return std::nullopt;
  std::optional<ConfigValue> value =
      tsl::profiler::GetConfigValue(options, std::string(key));
  if (!value.has_value()) {
    diagnostics.errors.push_back(
        KeyMessage(key, "has no value. The key was ignored."));
    return std::nullopt;
  }
  if (type.has_value() && !HasType(*value, *type)) {
    diagnostics.errors.push_back(KeyMessage(
        key,
        absl::StrCat("expects a value of type ", TypeName(*type), " but got ",
                     TypeName(*value), ". The key was ignored.")));
    return std::nullopt;
  }
  return value;
}

std::optional<int64_t> TakeInt64(const ProfileOptions& options,
                                 absl::string_view key,
                                 absl::flat_hash_set<absl::string_view>& keys,
                                 RocmTracerOptionDiagnostics& diagnostics) {
  std::optional<ConfigValue> value =
      TakeValue(options, key, ValueType::kInt64, keys, diagnostics);
  if (!value.has_value()) return std::nullopt;
  return std::get<int64_t>(*value);
}

// Reads one of the gpu_max_* keys. They are documented as int64 but stored in
// uint64_t fields, where a negative value would wrap to a cap that is never
// reached. If `zero_effect` is set, a value of 0 draws a warning that completes
// the sentence "... is 0, which <zero_effect>".
std::optional<uint64_t> TakeCount(const ProfileOptions& options,
                                  absl::string_view key,
                                  std::optional<absl::string_view> zero_effect,
                                  absl::flat_hash_set<absl::string_view>& keys,
                                  RocmTracerOptionDiagnostics& diagnostics) {
  std::optional<int64_t> value = TakeInt64(options, key, keys, diagnostics);
  if (!value.has_value()) return std::nullopt;
  if (*value < 0) {
    diagnostics.errors.push_back(KeyMessage(
        key, absl::StrCat("is ", *value,
                          ", but negative values are not supported and there "
                          "is no \"no limit\" value. The largest supported "
                          "value is ",
                          kMaxRocmTraceEvents, ". The key was ignored.")));
    return std::nullopt;
  }
  if (*value > kMaxRocmTraceEvents) {
    diagnostics.warnings.push_back(KeyMessage(
        key,
        absl::StrCat("is ", *value,
                     ", which was reduced to the largest supported value, ",
                     kMaxRocmTraceEvents, ".")));
    return static_cast<uint64_t>(kMaxRocmTraceEvents);
  }
  if (*value == 0 && zero_effect.has_value()) {
    diagnostics.warnings.push_back(
        KeyMessage(key, absl::StrCat("is 0, which ", *zero_effect, ".")));
  }
  return static_cast<uint64_t>(*value);
}

void WarnUnimplemented(const ProfileOptions& options,
                       absl::flat_hash_set<absl::string_view>& keys,
                       RocmTracerOptionDiagnostics& diagnostics) {
  for (const UnimplementedKey& key : kUnimplementedOnRocm) {
    // Type-checked even though the value is unused, so that a wrong type is
    // reported now rather than when the key is implemented.
    std::optional<ConfigValue> value =
        TakeValue(options, key.name, key.type, keys, diagnostics);
    if (!value.has_value() || IsBenign(*value, key.benign)) continue;
    diagnostics.warnings.push_back(KeyMessage(
        key.name, absl::StrCat("has no effect on ROCm: ", key.reason, ".")));
  }
}

}  // namespace

RocmTracerOptionDiagnostics UpdateRocmTracerOptionsFromProfilerOptions(
    const ProfileOptions& profile_options, uint32_t device_count,
    RocmTracerOptions& tracer_options,
    RocmTraceCollectorOptions& collector_options) {
  RocmTracerOptionDiagnostics diagnostics;
  absl::flat_hash_set<absl::string_view> keys;
  for (const auto& [key, unused_value] :
       profile_options.advanced_configuration()) {
    keys.insert(key);
  }

  // This budget also covers ROCTX marker events.
  if (std::optional<uint64_t> value = TakeCount(
          profile_options, "gpu_max_callback_api_events",
          "empties the whole trace, because activity events are dropped "
          "unless they match a recorded API event",
          keys, diagnostics)) {
    collector_options.max_callback_api_events = *value;
  }

  // No separate warning for 0: the one below already says the value has no
  // effect.
  if (std::optional<uint64_t> value =
          TakeCount(profile_options, "gpu_max_activity_api_events",
                    /*zero_effect=*/std::nullopt, keys, diagnostics)) {
    collector_options.max_activity_api_events = *value;
    // The collector only counts HIP_API-domain activity events, and every
    // activity event is HIP_OPS, so this limit is never reached yet.
    diagnostics.warnings.push_back(KeyMessage(
        "gpu_max_activity_api_events",
        "is stored but not yet enforced on ROCm; the trace is bounded by "
        "gpu_max_callback_api_events only."));
  }

  if (std::optional<uint64_t> value =
          TakeCount(profile_options, "gpu_max_annotation_strings",
                    "retains no annotations, so kernel events have no op names",
                    keys, diagnostics)) {
    tracer_options.max_annotation_strings = *value;
  }

  if (std::optional<int64_t> value =
          TakeInt64(profile_options, "gpu_num_chips_to_profile_per_task", keys,
                    diagnostics)) {
    if (*value < 0) {
      diagnostics.errors.push_back(KeyMessage(
          "gpu_num_chips_to_profile_per_task",
          absl::StrCat("is ", *value,
                       ", but negative values are not supported. Pass 0 to "
                       "profile every GPU. The key was ignored.")));
    } else if (device_count == 0) {
      // 0 asks for every GPU, which is already the case.
      if (*value != 0) {
        diagnostics.warnings.push_back(
            KeyMessage("gpu_num_chips_to_profile_per_task",
                       "has no effect because no GPUs are present."));
      }
    } else if (*value == 0 || *value > device_count) {
      // As on CUDA (device_tracer_cuda.cc), 0 means every GPU. It must not be
      // passed through: the collector drops every event when num_gpus is 0.
      collector_options.num_gpus = device_count;
      if (*value != 0) {
        diagnostics.warnings.push_back(KeyMessage(
            "gpu_num_chips_to_profile_per_task",
            absl::StrCat("is ", *value, ", which exceeds the ", device_count,
                         " GPUs present, so all of them are profiled.")));
      }
    } else {
      collector_options.num_gpus = static_cast<uint32_t>(*value);
      if (*value < device_count) {
        // RocmTraceCollectorImpl::Flush() numbers devices from the lowest
        // device id among the collected events, not from GPU 0.
        diagnostics.warnings.push_back(KeyMessage(
            "gpu_num_chips_to_profile_per_task",
            absl::StrCat(
                "is ", *value, ". All ", device_count,
                " GPUs are still traced, so tracing overhead is unchanged. "
                "After collection, only events from ",
                *value,
                " consecutive GPU ids are kept, starting at the lowest id "
                "that recorded an event; the rest are dropped.")));
      }
    }
  }

  WarnUnimplemented(profile_options, keys, diagnostics);

  // advanced_configuration is shared with other profiler components, for
  // example session_manager's "use_system_hostname", so only unrecognised
  // gpu_* keys are errors.
  std::vector<absl::string_view> unknown;
  for (absl::string_view key : keys) {
    if (absl::StartsWith(key, "gpu_")) unknown.push_back(key);
  }
  absl::c_sort(unknown);
  for (absl::string_view key : unknown) {
    diagnostics.errors.push_back(KeyMessage(
        key, "is not recognised by the ROCm GPU tracer. The key was ignored."));
  }
  return diagnostics;
}

void AppendOptionDiagnostics(const RocmTracerOptionDiagnostics& diagnostics,
                             tensorflow::profiler::XSpace* space) {
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
