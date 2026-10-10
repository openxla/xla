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

#include "xla/pjrt/pjrt_build_stamp.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/fingerprint.h"
#include "xla/tsl/builddata/builddata.h"

namespace xla {

namespace {

std::string FingerprintComponents(
    absl::Span<const PjRtComponentBuildStamp> components) {
  if (components.empty()) {
    return "";
  }
  std::string serialized;
  for (const auto& comp : components) {
    absl::StrAppend(&serialized, comp.kind, ":", comp.name, ":", comp.owner,
                    ":", comp.fingerprint, ";");
  }
  const tsl::Fprint128 fprint = tsl::Fingerprint128(serialized);
  return absl::StrCat(absl::Hex(fprint.high64, absl::kZeroPad16),
                      absl::Hex(fprint.low64, absl::kZeroPad16));
}

}  // namespace

PjRtBuildStamp::Options PjRtBuildStamp::Options::FromBuildData() {
  Options options;
  options.build_timestamp = absl::FromTimeT(tsl::builddata::TimestampAsInt());
  options.vcs_revision = std::string(tsl::builddata::SourceRevision());
  options.build_label = std::string(tsl::builddata::BuildLabel());
  options.build_cl = tsl::builddata::SourceRevisionAsInt();
  return options;
}

PjRtBuildStamp::PjRtBuildStamp(Options options, std::string base_identity,
                               std::string cache_key_fingerprint)
    : build_timestamp_(options.build_timestamp),
      vcs_revision_(std::move(options.vcs_revision)),
      build_label_(std::move(options.build_label)),
      build_cl_(options.build_cl),
      base_source_hash_(std::move(options.base_source_hash)),
      components_(std::move(options.components)),
      attributes_(std::move(options.attributes)),
      base_identity_(std::move(base_identity)),
      cache_key_fingerprint_(std::move(cache_key_fingerprint)) {}

absl::StatusOr<PjRtBuildStamp> PjRtBuildStamp::Create(Options options) {
  if (options.base_source_hash.empty() && options.build_cl <= 0 &&
      options.vcs_revision.empty() &&
      options.build_timestamp == absl::InfinitePast()) {
    return absl::InvalidArgumentError(
        "PjRtBuildStamp requires at least one build identity field "
        "(base_source_hash, build_cl, vcs_revision, or build_timestamp)");
  }
  for (const auto& comp : options.components) {
    if (comp.name.empty() || comp.fingerprint.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Component build stamp name and fingerprint must not "
                       "be empty. name='",
                       comp.name, "', fingerprint='", comp.fingerprint, "'"));
    }
  }

  absl::c_sort(options.components);

  std::string base_id = options.base_source_hash;
  if (base_id.empty()) {
    if (options.build_cl > 0) {
      base_id = absl::StrCat("cl:", options.build_cl);
    } else if (!options.vcs_revision.empty()) {
      base_id = absl::StrCat("vcs:", options.vcs_revision);
    } else if (options.build_timestamp != absl::InfinitePast()) {
      base_id =
          absl::StrCat("ts:", absl::ToUnixSeconds(options.build_timestamp));
    }
    if (!options.attributes.empty()) {
      std::vector<absl::string_view> keys;
      keys.reserve(options.attributes.size());
      // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
      for (const auto& [k, _] : options.attributes) {
        keys.push_back(k);
      }
      absl::c_sort(keys);
      for (absl::string_view k : keys) {
        if (const auto* s = std::get_if<std::string>(&options.attributes.at(k));
            s != nullptr && !s->empty()) {
          absl::StrAppend(&base_id, ";", k, "=", *s);
        } else if (const auto* i =
                       std::get_if<int64_t>(&options.attributes.at(k));
                   i != nullptr) {
          absl::StrAppend(&base_id, ";", k, "=", *i);
        }
      }
    }
  }

  std::string cache_key_fp = base_id;
  if (!options.components.empty()) {
    const std::string components_fp = FingerprintComponents(options.components);
    cache_key_fp = absl::StrCat(base_id, "+", components_fp);
  }

  return PjRtBuildStamp(std::move(options), std::move(base_id),
                        std::move(cache_key_fp));
}

PjRtBuildStamp::Options PjRtBuildStamp::ToOptions() const {
  Options options;
  options.build_timestamp = build_timestamp_;
  options.vcs_revision = vcs_revision_;
  options.build_label = build_label_;
  options.build_cl = build_cl_;
  options.base_source_hash = base_source_hash_;
  options.components = components_;
  options.attributes = attributes_;
  return options;
}

std::optional<absl::string_view> PjRtBuildStamp::GetStringAttribute(
    absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND {
  auto it = attributes_.find(key);
  if (it == attributes_.end()) {
    return std::nullopt;
  }
  if (const auto* val = std::get_if<std::string>(&it->second)) {
    return *val;
  }
  return std::nullopt;
}

std::string PjRtBuildStamp::CacheKeyFingerprintForComponents(
    const absl::flat_hash_set<absl::string_view>& active_component_names)
    const {
  std::vector<PjRtComponentBuildStamp> filtered;
  filtered.reserve(components_.size());
  for (const auto& comp : components_) {
    if (active_component_names.contains(comp.name)) {
      filtered.push_back(comp);
    }
  }

  if (filtered.empty()) {
    return base_identity_;
  }

  const std::string components_fp = FingerprintComponents(filtered);
  return absl::StrCat(base_identity_, "+", components_fp);
}

}  // namespace xla
