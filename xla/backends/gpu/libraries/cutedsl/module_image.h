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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_IMAGE_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_IMAGE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla::gpu::cutedsl {

inline constexpr size_t kModuleDigestSize = 32;

// An immutable module image with a validated SHA-256 digest.
class ModuleImage {
 public:
  static absl::StatusOr<ModuleImage> Create(absl::string_view bytes);
  static absl::StatusOr<ModuleImage> Create(absl::string_view bytes,
                                            absl::string_view sha256);

  ModuleImage(ModuleImage&&) = default;
  ModuleImage& operator=(ModuleImage&&) = default;
  ModuleImage(const ModuleImage&) = delete;
  ModuleImage& operator=(const ModuleImage&) = delete;

  absl::string_view bytes() const { return bytes_; }
  absl::string_view sha256() const;

 private:
  ModuleImage(std::string bytes, std::array<uint8_t, kModuleDigestSize> sha256)
      : bytes_(std::move(bytes)), sha256_(sha256) {}

  std::string bytes_;
  std::array<uint8_t, kModuleDigestSize> sha256_;
};

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_IMAGE_H_
