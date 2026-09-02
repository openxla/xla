/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TSL_PROFILER_UTILS_PROFILER_OPTIONS_UTIL_H_
#define XLA_TSL_PROFILER_UTILS_PROFILER_OPTIONS_UTIL_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
// Get config value from the profiler options, if the key is not found, return
// std::nullopt.
std::optional<std::variant<std::string, bool, int64_t>> GetConfigValue(
    const tensorflow::ProfileOptions& options, const std::string& key);

// The name of the AdvancedConfigValue oneof case that T corresponds to.
//
// Lives here rather than in any one backend because a type mismatch is
// reported the same way everywhere: the caller knows T, GetConfigValue knows
// what was actually supplied, and the user needs both halves to fix the
// config. A backend that composes its own message can reuse these instead of
// re-deriving the mapping.
template <typename T>
constexpr absl::string_view ConfigValueTypeName() {
  static_assert(std::is_same_v<T, std::string> || std::is_same_v<T, bool> ||
                    std::is_same_v<T, int64_t>,
                "advanced_configuration values are string, bool or int64");
  if constexpr (std::is_same_v<T, std::string>) {
    return "string";
  } else if constexpr (std::is_same_v<T, bool>) {
    return "bool";
  } else {
    return "int64";
  }
}

// The name of the type a value actually holds.
inline absl::string_view ConfigValueTypeName(
    const std::variant<std::string, bool, int64_t>& value) {
  return std::visit(
      [](const auto& alternative) -> absl::string_view {
        return ConfigValueTypeName<std::decay_t<decltype(alternative)>>();
      },
      value);
}

// The name of the type `key` holds in `options`, or "unset" when the key is
// absent or carries an AdvancedConfigValue with no oneof case set. The two are
// deliberately not distinguished: GetConfigValue collapses them, and a caller
// that needs to tell them apart already has to look at the map itself.
absl::string_view SuppliedConfigValueTypeName(
    const tensorflow::ProfileOptions& options, const std::string& key);

template <typename T>
absl::Status SetValue(const tensorflow::ProfileOptions& options,
                      const std::string& key,
                      absl::flat_hash_set<absl::string_view>& input_keys,
                      std::function<void(T)> setter) {
  auto value = tsl::profiler::GetConfigValue(options, key);
  if (!value.has_value()) {
    // Assumed value is not set.
    return absl::OkStatus();
  }
  if (!std::holds_alternative<T>(*value)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid value type for key: ", key, ". Expected ",
        ConfigValueTypeName<T>(), ", but a ", ConfigValueTypeName(*value),
        " was supplied."));
  }
  input_keys.erase(key);
  setter(std::get<T>(*value));
  return absl::OkStatus();
}

template <typename T>
absl::Status SetValueWithStatus(
    const tensorflow::ProfileOptions& options, const std::string& key,
    absl::flat_hash_set<absl::string_view>& input_keys,
    std::function<absl::Status(T)> setter) {
  auto value = tsl::profiler::GetConfigValue(options, key);
  if (!value.has_value()) {
    // Assumed value is not set.
    return absl::OkStatus();
  }
  if (!std::holds_alternative<T>(*value)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid value type for key: ", key, ". Expected ",
        ConfigValueTypeName<T>(), ", but a ", ConfigValueTypeName(*value),
        " was supplied."));
  }
  input_keys.erase(key);
  absl::Status status = setter(std::get<T>(*value));
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid value for key: ", key,
        ". Setter function failed with error: ", status.message()));
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_PROFILER_OPTIONS_UTIL_H_
