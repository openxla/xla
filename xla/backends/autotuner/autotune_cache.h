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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

enum class CacheMode {
  kReadOnly,    // Lookup only (e.g., immutable cache formats)
  kWriteOnly,   // Insert/Update only (e.g., logging sinks)
  kReadWrite,   // Lookup and Insert (allows adding new entries)
  kReadUpdate,  // Lookup, Insert, and explicit Overwrite/Updates.
};

struct AutotuneContext {
  std::string device;
  std::string explicit_version;
  std::string codegen_version;
  absl::flat_hash_map<autotuner::Backend, std::string> per_backend_versions;
  // Loose key matching mode allows matching cache entries with different
  // codegen version and codegen options, given that the version of the optimal
  // config's backend is the same.
  // It guarantees that the config works but it might not be the best config if
  // the instruction is supported by different backends, and the other backend
  // has improved enough to overtake current optimal backend.
  bool use_loose_matching = false;
  CacheMode mode = CacheMode::kReadUpdate;
};

// AutotuneCache is an interface for managing autotuning cache.
// It provides methods for looking up and inserting configs, serializing and
// deserializing the cache, and retrieving cache statistics and mode.
// Cross implementation cache lookup/insert and serialize/deserialize are not
// compatible.
class AutotuneCache {
 public:
  struct Config {
    autotuner::Backend codegen_backend;
    autotuner::BackendConfig backend_config;
  };

  struct CacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
  };

  virtual ~AutotuneCache() = default;

  // TODO(b/444398084): Remove these methods once all callers are migrated to
  // the new interface.
  virtual std::optional<Config> Lookup(const HloInstruction* instr) {
    return Lookup(instr, "");
  }
  virtual absl::Status Insert(const HloInstruction* instr,
                              const Config& config) {
    return Insert(instr, "", config);
  }

  virtual std::optional<Config> Lookup(
      const HloInstruction* instr,
      absl::string_view codegen_options_fingerprint) = 0;
  virtual absl::Status Insert(const HloInstruction* instr,
                              absl::string_view codegen_options_fingerprint,
                              const Config& config) = 0;
  virtual absl::StatusOr<std::string> Serialize(
      absl::Span<const HloInstruction* const> instructions_to_serialize) {
    return absl::UnimplementedError("Serialize is not implemented.");
  };

  // Deserializes the string and updates the cache, overwriting the keys if they
  // already exist.
  virtual absl::Status Deserialize(absl::string_view serialized_cache) {
    return absl::UnimplementedError("Deserialize is not implemented.");
  };
  virtual CacheStats GetCacheStats() const = 0;
  virtual CacheMode GetMode() const = 0;
};

class NoOpAutotuneCache : public AutotuneCache {
 public:
  NoOpAutotuneCache() = default;
  explicit NoOpAutotuneCache(AutotuneContext context)
      : context_(std::move(context)) {}
  ~NoOpAutotuneCache() override = default;

  using AutotuneCache::Insert;
  using AutotuneCache::Lookup;

  std::optional<Config> Lookup(
      const HloInstruction* instr,
      absl::string_view codegen_options_fingerprint) override {
    return std::nullopt;
  }
  absl::Status Insert(const HloInstruction* instr,
                      absl::string_view codegen_options_fingerprint,
                      const Config& config) override {
    return absl::OkStatus();
  }
  CacheStats GetCacheStats() const override { return {}; }
  CacheMode GetMode() const override { return CacheMode::kReadUpdate; }

 private:
  AutotuneContext context_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_H_
