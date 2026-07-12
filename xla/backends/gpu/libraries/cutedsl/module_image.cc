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

#include "xla/backends/gpu/libraries/cutedsl/module_image.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

namespace xla::gpu::cutedsl {

absl::StatusOr<ModuleImage> ModuleImage::Create(absl::string_view bytes) {
  if (bytes.empty()) {
    return absl::InvalidArgumentError("`module` must not be empty");
  }

  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(bytes.data(), bytes.size()));
  std::array<uint8_t, kModuleDigestSize> digest = hasher.final();
  return ModuleImage(std::string(bytes), digest);
}

absl::StatusOr<ModuleImage> ModuleImage::Create(absl::string_view bytes,
                                                absl::string_view sha256) {
  absl::StatusOr<ModuleImage> image = Create(bytes);
  if (!image.ok()) return image.status();
  if (sha256.size() != kModuleDigestSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`key` must contain one %d-byte SHA-256 digest", kModuleDigestSize));
  }
  if (!std::equal(image->sha256().begin(), image->sha256().end(),
                  sha256.begin())) {
    return absl::InvalidArgumentError(
        "SHA-256 `key` does not match the module image");
  }
  return std::move(*image);
}

absl::string_view ModuleImage::sha256() const {
  return absl::string_view(reinterpret_cast<const char*>(sha256_.data()),
                           sha256_.size());
}

}  // namespace xla::gpu::cutedsl
