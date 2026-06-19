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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_LEGACY_CACHE_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_LEGACY_CACHE_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotune_cache.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuning/autotune_cache_key.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {

namespace gpu {

// Wrapper around the legacy autotune cache from the AutotunerUtil which uses
// AutotuneResult proto.
class LegacyCache : public AutotuneCache {
 public:
  LegacyCache(std::string cache_dir, DebugOptions::AutotuneCacheMode cache_mode,
              se::DeviceDescription device_desc)
      : cache_dir_(std::move(cache_dir)),
        cache_mode_(cache_mode),
        device_desc_(std::move(device_desc)) {}

  using AutotuneCache::Insert;
  using AutotuneCache::Lookup;

  std::optional<Config> Lookup(
      const HloInstruction* instr,
      absl::string_view codegen_options_fingerprint) override;
  absl::Status Insert(const HloInstruction* instr,
                      absl::string_view codegen_options_fingerprint,
                      const Config& best_config) override;

  absl::StatusOr<std::string> Serialize(absl::Span<const HloInstruction* const>
                                            instructions_to_serialize) override;
  absl::Status Deserialize(absl::string_view serialized_cache) override;

  CacheStats GetCacheStats() const override { return stats_; }
  CacheMode GetMode() const override {
    switch (cache_mode_) {
      case DebugOptions::AUTOTUNE_CACHE_MODE_UNSPECIFIED:
        return CacheMode::kReadUpdate;
      case DebugOptions::AUTOTUNE_CACHE_MODE_READ:
        return CacheMode::kReadOnly;
      case DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE:
        return CacheMode::kReadUpdate;
      default:
        return CacheMode::kReadUpdate;
    }
  }

  void ClearCache();

 private:
  AutotuneCacheKey GetAutotuneCacheKey(const HloInstruction& instr);

  // Translates between the AutotunerCacheInterface::Config and the
  // AutotuneResult.
  std::optional<Config> GetConfig(const AutotuneResult& result,
                                  bool is_fusion_instruction);
  AutotuneResult GetAutotuneResult(const Config& config);

  const std::string cache_dir_;
  const DebugOptions::AutotuneCacheMode cache_mode_;
  const se::DeviceDescription device_desc_;
  CacheStats stats_;
};

}  // namespace gpu

}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_LEGACY_CACHE_H_
