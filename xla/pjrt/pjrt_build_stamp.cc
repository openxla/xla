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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tsl/platform/fingerprint.h"
#include "xla/pjrt/pjrt_compiler_variant.h"
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

PjRtBuildStampRegistry& PjRtBuildStampRegistry::Global() {
  static absl::NoDestructor<PjRtBuildStampRegistry> registry;
  return *registry;
}

absl::Status PjRtBuildStampRegistry::RegisterPlatformBuildStampFactory(
    absl::string_view platform_name, Factory factory) {
  return RegisterCompilerBuildStampFactory(platform_name, kLinkedVariant,
                                           std::move(factory));
}

absl::Status PjRtBuildStampRegistry::RegisterCompilerBuildStampFactory(
    absl::string_view platform_name, absl::string_view variant_name,
    Factory factory) {
  const PjRtBuildStampKey key(platform_name, variant_name);
  absl::MutexLock lock(mu_);
  auto [it, inserted] = entries_.try_emplace(key, nullptr);
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Build stamp factory already registered for platform '",
                     platform_name, "' and variant '", variant_name, "'"));
  }
  it->second = std::make_unique<Entry>(std::move(factory));
  return absl::OkStatus();
}

absl::Status PjRtBuildStampRegistry::RegisterComponentFingerprintProvider(
    absl::string_view platform_name, ComponentProvider provider) {
  absl::MutexLock lock(mu_);
  component_providers_[platform_name].push_back(std::move(provider));
  return absl::OkStatus();
}

void PjRtBuildStampRegistry::RegisterVariantPicker(
    absl::string_view platform_name, VariantPicker picker) {
  absl::MutexLock lock(mu_);
  variant_pickers_[platform_name] = std::move(picker);
}

absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>>
PjRtBuildStampRegistry::GetBuildStampForCompiler(
    absl::string_view platform_name, absl::string_view variant_name) {
  const PjRtBuildStampKey key(platform_name, variant_name);
  Entry* entry = nullptr;
  {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No registered build stamp factory for platform '", platform_name,
          "' and compiler variant '", variant_name, "'"));
    }
    entry = it->second.get();
  }

  absl::call_once(entry->once, [&]() {
    absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> stamp_or =
        entry->factory();
    if (!stamp_or.ok()) {
      entry->memoized = stamp_or.status();
      return;
    }

    std::shared_ptr<const PjRtBuildStamp> stamp = *std::move(stamp_or);

    // For linked compilers, aggregate any dynamically registered component
    // fingerprint providers (e.g. GPU native custom call handlers or FFI).
    if (variant_name == kLinkedVariant) {
      std::vector<ComponentProvider> providers;
      {
        absl::MutexLock lock(mu_);
        auto it = component_providers_.find(platform_name);
        if (it != component_providers_.end()) {
          providers = it->second;
        }
      }

      std::vector<PjRtComponentBuildStamp> extra_components;
      for (const auto& provider : providers) {
        auto components_or = provider();
        if (components_or.ok()) {
          extra_components.insert(extra_components.end(),
                                  components_or->begin(), components_or->end());
        }
      }

      if (!extra_components.empty()) {
        PjRtBuildStamp::Options options = stamp->ToOptions();
        options.components.insert(options.components.end(),
                                  extra_components.begin(),
                                  extra_components.end());
        auto updated_or = PjRtBuildStamp::Create(std::move(options));
        if (updated_or.ok()) {
          stamp =
              std::make_shared<const PjRtBuildStamp>(*std::move(updated_or));
        }
      }
    }

    entry->memoized = std::move(stamp);
  });

  return entry->memoized;
}

absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>>
PjRtBuildStampRegistry::GetBuildStampForPlatform(
    absl::string_view platform_name) {
  std::string picked_variant;
  absl::string_view variant = kLinkedVariant;
  {
    absl::MutexLock lock(mu_);
    auto it = variant_pickers_.find(platform_name);
    if (it != variant_pickers_.end()) {
      auto picked = it->second();
      if (picked.ok() && !picked->empty()) {
        picked_variant = *std::move(picked);
        variant = picked_variant;
      }
    }
  }

  return GetBuildStampForCompiler(platform_name, variant);
}

void PjRtBuildStampRegistry::ClearCacheForTesting() {
  absl::MutexLock lock(mu_);
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (auto& [key, entry] : entries_) {
    entry = std::make_unique<Entry>(entry->factory);
  }
}

void PjRtRegisterPlatformBuildStampFactory(
    absl::string_view platform_name, PjRtBuildStampRegistry::Factory factory) {
  CHECK_OK(PjRtBuildStampRegistry::Global().RegisterPlatformBuildStampFactory(
      platform_name, std::move(factory)));
}

void PjRtRegisterCompilerBuildStampFactory(
    absl::string_view platform_name, absl::string_view variant_name,
    PjRtBuildStampRegistry::Factory factory) {
  CHECK_OK(PjRtBuildStampRegistry::Global().RegisterCompilerBuildStampFactory(
      platform_name, variant_name, std::move(factory)));
}

void PjRtRegisterComponentFingerprintProvider(
    absl::string_view platform_name,
    PjRtBuildStampRegistry::ComponentProvider provider) {
  CHECK_OK(
      PjRtBuildStampRegistry::Global().RegisterComponentFingerprintProvider(
          platform_name, std::move(provider)));
}

void PjRtRegisterBuildStampVariantPicker(
    absl::string_view platform_name,
    PjRtBuildStampRegistry::VariantPicker picker) {
  PjRtBuildStampRegistry::Global().RegisterVariantPicker(platform_name,
                                                         std::move(picker));
}

absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> GetBuildStampForPlatform(
    absl::string_view platform_name) {
  return PjRtBuildStampRegistry::Global().GetBuildStampForPlatform(
      platform_name);
}

absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> GetBuildStampForCompiler(
    absl::string_view platform_name, absl::string_view variant_name) {
  return PjRtBuildStampRegistry::Global().GetBuildStampForCompiler(
      platform_name, variant_name);
}

absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> GetBuildStampForCompiler(
    absl::string_view platform_name, PjRtCompilerVariant variant) {
  return GetBuildStampForCompiler(platform_name,
                                  CompilerVariantToString(variant));
}

}  // namespace xla
