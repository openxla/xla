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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler_variant.h"

namespace xla {

// Well-known attribute key for the Mosaic compiler dialect version.
inline constexpr char kBuildStampAttrMosaicVersion[] = "mosaic_version";

// Represents a self-registering compile-time component (such as an XLA:GPU
// native custom call handler, an FFI compile-time Instantiate handler, or a
// dialect lowering engine) whose code or version affects compiled executables.
struct PjRtComponentBuildStamp {
  std::string kind;
  std::string name;
  std::string owner;
  std::string fingerprint;

  PjRtComponentBuildStamp() = default;
  PjRtComponentBuildStamp(std::string kind_in, std::string name_in,
                          std::string owner_in, std::string fingerprint_in)
      : kind(std::move(kind_in)),
        name(std::move(name_in)),
        owner(std::move(owner_in)),
        fingerprint(std::move(fingerprint_in)) {}

  bool operator==(const PjRtComponentBuildStamp& other) const {
    return kind == other.kind && name == other.name && owner == other.owner &&
           fingerprint == other.fingerprint;
  }
  bool operator!=(const PjRtComponentBuildStamp& other) const {
    return !(*this == other);
  }
  bool operator<(const PjRtComponentBuildStamp& other) const {
    if (kind != other.kind) {
      return kind < other.kind;
    }
    if (name != other.name) {
      return name < other.name;
    }
    if (owner != other.owner) {
      return owner < other.owner;
    }
    return fingerprint < other.fingerprint;
  }
};

// Immutable value type representing the build provenance, binary hash,
// compile-time components, and extensible attributes for a given PJRT plugin or
// compiler.
//
// Constructed via `PjRtBuildStamp::Create(Options)` to ensure required
// identity invariants are validated and cache key fingerprints are canonical.
class PjRtBuildStamp final {
 public:
  // Options struct used to configure and construct a PjRtBuildStamp.
  struct Options {
    // The timestamp at which the binary was built.
    absl::Time build_timestamp = absl::InfinitePast();

    // The VCS revision (Git commit SHA or Piper source URI) from which the
    // binary was built.
    std::string vcs_revision;

    // The build label with which the binary was built.
    std::string build_label;

    // The changelist number if built from a numeric version control system like
    // Piper. 0 if unavailable.
    int64_t build_cl = 0;

    // Hash of the base C++ binary that went into this PJRT entity (produced by
    // the source_hash rule). Should be independent of host CPU architecture.
    std::string base_source_hash;

    // Self-registering compile-time components linked into the compiler
    // process.
    std::vector<PjRtComponentBuildStamp> components;

    // Strongly-typed extensible attributes for dialect format versions,
    // compiler capabilities, and vendor- or environment-specific build
    // provenance.
    absl::flat_hash_map<std::string, PjRtValueType> attributes;

    // Populates build provenance from tsl::builddata (timestamp, VCS revision,
    // build label, and changelist).
    static Options FromBuildData();

    // Convenience helper to set the Mosaic IR compatibility version.
    void SetMosaicVersion(int64_t mosaic_version) {
      attributes[kBuildStampAttrMosaicVersion] = mosaic_version;
    }
  };

  // Validates invariants, canonicalizes components, precomputes cache key
  // fingerprints, and creates an immutable PjRtBuildStamp.
  static absl::StatusOr<PjRtBuildStamp> Create(Options options);

  // Converts this build stamp back into an Options struct, enabling
  // modification and re-creation (e.g. adding dynamic components).
  Options ToOptions() const;

  ~PjRtBuildStamp() = default;

  PjRtBuildStamp(const PjRtBuildStamp&) = default;
  PjRtBuildStamp& operator=(const PjRtBuildStamp&) = default;
  PjRtBuildStamp(PjRtBuildStamp&&) = default;
  PjRtBuildStamp& operator=(PjRtBuildStamp&&) = default;

  // The timestamp at which the binary was built.
  absl::Time build_timestamp() const { return build_timestamp_; }

  // The VCS revision (Git commit SHA or Piper source URI) from which the binary
  // was built.
  absl::string_view build_vcs_revision() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return vcs_revision_;
  }

  // The build label with which the binary was built.
  absl::string_view build_label() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return build_label_;
  }

  // The changelist number if built from a numeric version control system like
  // Piper. 0 if unavailable.
  int64_t build_cl() const { return build_cl_; }

  // The hash of the base C++ binary that went into this PJRT entity (produced
  // by the source_hash rule).
  absl::string_view base_source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return base_source_hash_;
  }

  // Backward compatibility alias returning base_source_hash().
  absl::string_view source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return base_source_hash_;
  }

  // Self-registering compile-time components linked into the compiler process.
  absl::Span<const PjRtComponentBuildStamp> components() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return components_;
  }

  // Mosaic IR compatibility version supported by this compiler, if set.
  std::optional<int64_t> mosaic_version() const {
    return GetAttribute<int64_t>(kBuildStampAttrMosaicVersion);
  }

  // Extensible strongly-typed attribute map.
  const absl::flat_hash_map<std::string, PjRtValueType>& attributes() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return attributes_;
  }

  template <typename T>
  std::optional<T> GetAttribute(absl::string_view key) const {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) {
      return std::nullopt;
    }
    if (const T* val = std::get_if<T>(&it->second)) {
      return *val;
    }
    return std::nullopt;
  }

  // Zero-copy lookup for a string attribute in `attributes()`.
  // Returns `std::nullopt` if `key` is not present or does not hold a string.
  std::optional<absl::string_view> GetStringAttribute(
      absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Precomputed canonical composite cache key fingerprint. Combines
  // base_source_hash, sorted components, and fallback environment identity into
  // a deterministic string for compilation cache keys.
  absl::string_view CacheKeyFingerprint() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return cache_key_fingerprint_;
  }

  // Computes a composite cache key fingerprint restricted to the components
  // whose name is present in active_component_names (e.g. custom call targets
  // actually referenced by a specific HloModule).
  std::string CacheKeyFingerprintForComponents(
      const absl::flat_hash_set<absl::string_view>& active_component_names)
      const;

 private:
  explicit PjRtBuildStamp(Options options, std::string base_identity,
                          std::string cache_key_fingerprint);

  absl::Time build_timestamp_;
  std::string vcs_revision_;
  std::string build_label_;
  int64_t build_cl_ = 0;
  std::string base_source_hash_;
  std::vector<PjRtComponentBuildStamp> components_;
  absl::flat_hash_map<std::string, PjRtValueType> attributes_;
  std::string base_identity_;
  std::string cache_key_fingerprint_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_H_
